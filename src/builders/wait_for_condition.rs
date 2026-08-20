//! The polling operation: [`WaitForConditionBuilder`], returned by
//! [`DurableContext::wait_for_condition`](crate::DurableContext::wait_for_condition),
//! plus its configuration ([`WaitStrategy`]) and the per-check decision
//! type ([`WaitDecision`]).

use std::future::IntoFuture;
use std::sync::Arc;
use std::time::Duration;

use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::OperationId;
use crate::error::OperationError;
use crate::future::DurableFuture;

pub use crate::wait_for_condition::WaitDecision;

// ============================================================
// WaitForConditionBuilder
// ============================================================

/// Builder for a wait-for-condition operation.
///
/// Created by [`DurableContext::wait_for_condition`]. Configure the polling
/// strategy with `.wait_strategy_fn()`.
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
pub struct WaitForConditionBuilder<S> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    initial_state: S,
    wait_strategy: Option<crate::wait_for_condition::WaitStrategyFn<S>>,
    serdes: Option<Arc<dyn Serdes>>,
    check: crate::wait_for_condition::BoxedCheckFn<S>,
}

impl<S> std::fmt::Debug for WaitForConditionBuilder<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaitForConditionBuilder")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<S: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + Sync + 'static>
    WaitForConditionBuilder<S>
{
    /// Creates a new builder (internal). Taking the check closure here
    /// keeps the field non-optional: a builder without a check is
    /// unrepresentable.
    pub(crate) fn new(
        ctx: DurableContext,
        op_id: OperationId,
        initial_state: S,
        check: crate::wait_for_condition::BoxedCheckFn<S>,
    ) -> Self {
        Self {
            ctx,
            op_id,
            name: None,
            initial_state,
            wait_strategy: None,
            serdes: None,
            check,
        }
    }

    /// Sets a human-readable name for this operation.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Sets the wait strategy (polling interval and backoff config).
    ///
    /// This converts the [`WaitStrategy`] config struct into a functional
    /// strategy internally.
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
    ///             WaitStrategy::builder()
    ///                 .initial_delay(Duration::from_secs(2))
    ///                 .max_delay(Duration::from_secs(30))
    ///                 .build(),
    ///         )
    ///         .await?;
    ///     Ok(done)
    /// }
    /// ```
    #[allow(clippy::needless_pass_by_value)] // reason: API consistency with other builder chain methods
    pub fn wait_strategy(mut self, strategy: WaitStrategy) -> Self {
        // Convert the config struct into a functional strategy with
        // exponential backoff.
        let initial = strategy.initial_delay();
        let max = strategy.max_delay();
        let factor = strategy.backoff_factor();
        self.wait_strategy = Some(Box::new(move |_state: S, attempt: u32| {
            // Default behavior: always continue with backoff.
            #[allow(clippy::cast_possible_truncation)] // reason: attempt is small
            let exponent = attempt.saturating_sub(1);
            let base_secs =
                initial.as_secs_f64() * factor.powi(i32::try_from(exponent).unwrap_or(0));
            let capped = base_secs.min(max.as_secs_f64());
            #[allow(clippy::cast_possible_truncation)]
            #[allow(clippy::cast_sign_loss)]
            let delay_secs = capped.ceil().max(1.0) as u64;
            WaitDecision::Continue {
                delay: Duration::from_secs(delay_secs),
            }
        }));
        self
    }

    /// Sets a custom wait strategy closure.
    ///
    /// The strategy receives the current (deserialized) state and the
    /// 1-based attempt number, and returns a [`WaitDecision`].
    /// The SDK boxes the closure internally — no `Box::new` at the call site.
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
    pub fn wait_strategy_fn<F>(mut self, strategy: F) -> Self
    where
        F: Fn(S, u32) -> WaitDecision + Send + Sync + 'static,
    {
        self.wait_strategy = Some(Box::new(strategy));
        self
    }

    /// Sets a custom serializer/deserializer for the state.
    pub fn serdes(mut self, serdes: impl Serdes + 'static) -> Self {
        self.serdes = Some(Arc::new(serdes));
        self
    }

    /// Converts this builder into a [`DurableFuture`] explicitly.
    pub fn future(self) -> DurableFuture<S> {
        <Self as IntoFuture>::into_future(self)
    }

    /// Eagerly spawns the operation on a tokio task.
    pub fn spawn(self) -> DurableFuture<S> {
        spawn_terminal!(self)
    }
}

impl<S: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + Sync + 'static> IntoFuture
    for WaitForConditionBuilder<S>
{
    type Output = Result<S, OperationError>;
    type IntoFuture = DurableFuture<S>;

    fn into_future(self) -> Self::IntoFuture {
        use crate::wait_for_condition::WaitForConditionExecution;

        preflight_identity!(self, "Step", crate::wait_for_condition::WFC_SUB_TYPE);

        let execution = WaitForConditionExecution {
            ctx: self.ctx,
            op_id: self.op_id,
            name: self.name,
            initial_state: self.initial_state,
            wait_strategy: self.wait_strategy,
            serdes: self.serdes,
            check: self.check,
        };

        DurableFuture::from_async(async move { execution.execute().await })
    }
}

/// Configuration for the wait strategy used by
/// [`DurableContext::wait_for_condition`].
///
/// Controls the polling interval and backoff behavior for condition checks.
/// Construct with [`WaitStrategy::builder`] — unset values keep their
/// [`Default`] value. Fields are private per C-STRUCT-PRIVATE; read values
/// back through the accessors.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitStrategy;
/// use std::time::Duration;
///
/// let strategy = WaitStrategy::builder()
///     .initial_delay(Duration::from_secs(2))
///     .max_delay(Duration::from_secs(30))
///     .build();
/// assert_eq!(strategy.initial_delay(), Duration::from_secs(2));
/// // Unset values keep the default.
/// let default_factor = WaitStrategy::default().backoff_factor();
/// assert!((strategy.backoff_factor() - default_factor).abs() < f64::EPSILON);
/// ```
#[derive(Debug, Clone)]
pub struct WaitStrategy {
    /// Initial delay between condition checks.
    initial_delay: Duration,
    /// Maximum delay between condition checks.
    max_delay: Duration,
    /// Backoff multiplier applied after each check.
    backoff_factor: f64,
}

impl Default for WaitStrategy {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_mins(1),
            backoff_factor: 2.0,
        }
    }
}

impl WaitStrategy {
    /// Creates a new [`WaitStrategyBuilder`] seeded with the default values.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitStrategy;
    /// use std::time::Duration;
    ///
    /// let strategy = WaitStrategy::builder()
    ///     .backoff_factor(1.5)
    ///     .build();
    /// assert!((strategy.backoff_factor() - 1.5).abs() < f64::EPSILON);
    /// ```
    pub fn builder() -> WaitStrategyBuilder {
        WaitStrategyBuilder::default()
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
    pub fn backoff_factor(&self) -> f64 {
        self.backoff_factor
    }
}

/// Builder for [`WaitStrategy`].
///
/// Follows the Rust API Guidelines C-BUILDER pattern. All methods consume
/// and return `self` for chaining. Values left unset keep the
/// [`WaitStrategy::default`] value.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::builders::wait_for_condition::WaitStrategy;
/// use std::time::Duration;
///
/// let strategy = WaitStrategy::builder()
///     .initial_delay(Duration::from_millis(500))
///     .max_delay(Duration::from_secs(10))
///     .backoff_factor(3.0)
///     .build();
/// assert_eq!(strategy.max_delay(), Duration::from_secs(10));
/// ```
#[derive(Debug, Clone, Default)]
#[must_use = "builders do nothing unless .build() is called"]
#[non_exhaustive]
pub struct WaitStrategyBuilder {
    initial_delay: Option<Duration>,
    max_delay: Option<Duration>,
    backoff_factor: Option<f64>,
}

impl WaitStrategyBuilder {
    /// Sets the initial delay between condition checks.
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = Some(delay);
        self
    }

    /// Sets the maximum delay between condition checks.
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = Some(delay);
        self
    }

    /// Sets the backoff multiplier applied after each check.
    pub fn backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = Some(factor);
        self
    }

    /// Builds the [`WaitStrategy`], filling unset values from
    /// [`WaitStrategy::default`].
    #[must_use]
    pub fn build(self) -> WaitStrategy {
        let defaults = WaitStrategy::default();
        WaitStrategy {
            initial_delay: self.initial_delay.unwrap_or(defaults.initial_delay),
            max_delay: self.max_delay.unwrap_or(defaults.max_delay),
            backoff_factor: self.backoff_factor.unwrap_or(defaults.backoff_factor),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
#[allow(clippy::panic)] // reason: test assertions on unexpected variants
mod tests {
    use super::*;

    /// The `WaitStrategy` builder sets each knob independently and leaves
    /// unset knobs at their `Default` value.
    #[test]
    fn wait_strategy_builder_overrides_only_what_is_set() {
        let defaults = WaitStrategy::default();

        let strategy = WaitStrategy::builder()
            .initial_delay(Duration::from_secs(5))
            .build();
        assert_eq!(strategy.initial_delay(), Duration::from_secs(5));
        assert_eq!(strategy.max_delay(), defaults.max_delay());
        assert!((strategy.backoff_factor() - defaults.backoff_factor()).abs() < f64::EPSILON);

        let full = WaitStrategy::builder()
            .initial_delay(Duration::from_millis(500))
            .max_delay(Duration::from_secs(10))
            .backoff_factor(3.0)
            .build();
        assert_eq!(full.initial_delay(), Duration::from_millis(500));
        assert_eq!(full.max_delay(), Duration::from_secs(10));
        assert!((full.backoff_factor() - 3.0).abs() < f64::EPSILON);

        // Default behavior is unchanged by the builder rework.
        assert_eq!(defaults.initial_delay(), Duration::from_secs(1));
        assert_eq!(defaults.max_delay(), Duration::from_mins(1));
        assert!((defaults.backoff_factor() - 2.0).abs() < f64::EPSILON);
    }

    /// A [`WaitStrategy`] built through its builder drives the derived polling
    /// schedule: the first delay is `initial_delay`, each subsequent delay
    /// grows by `backoff_factor`, and the sequence is capped at `max_delay`.
    #[test]
    fn wait_strategy_builder_drives_polling_schedule() {
        let ctx = DurableContext::__test_context();

        let builder = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .wait_strategy(
                WaitStrategy::builder()
                    .initial_delay(Duration::from_secs(2))
                    .max_delay(Duration::from_secs(10))
                    .backoff_factor(3.0)
                    .build(),
            );

        let strategy = builder
            .wait_strategy
            .as_ref()
            .expect("wait_strategy must install a strategy");

        let delay_of = |attempt: u32| match strategy(0_i32, attempt) {
            WaitDecision::Continue { delay } => delay,
            other => panic!("config-derived strategy must always continue, got {other:?}"),
        };

        // attempt 1 → initial (2s); attempt 2 → 2s * 3 = 6s;
        // attempt 3 → 18s, capped at max_delay (10s).
        assert_eq!(delay_of(1), Duration::from_secs(2));
        assert_eq!(delay_of(2), Duration::from_secs(6));
        assert_eq!(delay_of(3), Duration::from_secs(10));
        assert_eq!(delay_of(9), Duration::from_secs(10));
    }

    /// The default [`WaitStrategy`] keeps its pre-builder behavior: a 1 second
    /// first delay doubling up to the 1 minute cap.
    #[test]
    fn default_wait_strategy_schedule_unchanged() {
        let ctx = DurableContext::__test_context();

        let builder = ctx
            .wait_for_condition(|_sc, state: i32| async move { Ok(state) }, 0_i32)
            .wait_strategy(WaitStrategy::default());

        let strategy = builder
            .wait_strategy
            .as_ref()
            .expect("wait_strategy must install a strategy");

        let delay_of = |attempt: u32| match strategy(0_i32, attempt) {
            WaitDecision::Continue { delay } => delay,
            other => panic!("config-derived strategy must always continue, got {other:?}"),
        };

        assert_eq!(delay_of(1), Duration::from_secs(1));
        assert_eq!(delay_of(2), Duration::from_secs(2));
        assert_eq!(delay_of(3), Duration::from_secs(4));
        assert_eq!(delay_of(20), Duration::from_mins(1));
    }
}
