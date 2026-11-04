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

/// Eagerly validates the builder's claimed replay identity right after the
/// builder claims its operation ID.
///
/// Every builder except a combinator claims its ID when it is created, and
/// its `into_future` runs this FIRST. `.future()`, `.spawn()`, and `.await`
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
/// Builds the operation future with the builder's lazy terminal
/// (`into_future`, which runs the identity preflight and the admission
/// check). For a combinator it then runs the pending claim
/// ([`DurableFuture::claim_now`](crate::future::DurableFuture)), so the
/// combinator claims its ID and runs the identity preflight at the
/// `.spawn()` call. Then it starts the future on its own task through
/// [`DurableFuture::spawn_blessed`](crate::future::DurableFuture), which
/// blesses the task with the builder context's ownership handle and
/// registers it with the context's scopes. Dropping the returned handle
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
/// `wait_for_callback` and `with_retry`. They check for a runtime first,
/// then run the operation's start step (see [`crate::start`]) on the
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
        let mut future = ::std::future::IntoFuture::into_future(builder);
        future.claim_now();
        $crate::future::DurableFuture::spawn_blessed(future, &ctx, None)
    }};
    ($builder:expr, batch_future) => {{
        let builder = $builder;
        let ctx = builder.ctx.clone();
        let future = builder.batch_future();
        $crate::future::DurableFuture::spawn_blessed(future, &ctx, None)
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
        let future = ::std::future::IntoFuture::into_future(builder);
        $crate::future::DurableFuture::spawn_blessed(future, &ctx, Some(flag))
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
/// then jittered per [`JitterStrategy`] and rounded to the nearest whole
/// second with a one-second minimum, for every jitter mode, as in the JS
/// and Go SDKs. A delay that a custom retry closure returns is rounded up
/// to whole seconds instead, with the same one-second minimum, so it never
/// ends early.
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
    /// use aws_durable_execution_sdk::builders::RetryStrategyConfig;
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

    /// Returns the multiplier applied to the delay after each failed
    /// attempt.
    pub fn backoff_rate(&self) -> f64 {
        self.backoff_rate
    }

    /// Returns the jitter applied to each computed delay.
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
                    ctx.wait_for_callback::<String, _, _>(|_, _id| async { Ok(()) })
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
                    ctx.with_retry(|_child| async { Ok(1_u32) })
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

    /// The admission check runs before the task-ownership check on the
    /// caller, as in the start step: `.spawn()` from a task the SDK did not
    /// start, inside an abandoned owner, leaves the handle refused rather
    /// than resolving it to the task-ownership error.
    #[tokio::test]
    #[expect(clippy::disallowed_methods)] // reason: the test calls `.spawn()` from a task the SDK did not start
    async fn spawn_from_unblessed_caller_in_abandoned_owner_stays_refused() {
        let (root, client) = recording_context(Some(Duration::ZERO));
        let child = root.new_child("1", None);
        root.admission()
            .abandon([crate::engine::compute_wire_id_public("1").as_str()]);
        for index in 0..STARTED_SPAWNS {
            let spawn_ctx = child.clone();
            let (name, resolved) = tokio::spawn(async move {
                let (name, handle) = started_spawn(&spawn_ctx, index);
                let resolved = tokio::time::timeout(Duration::from_millis(20), handle)
                    .await
                    .is_ok();
                (name, resolved)
            })
            .await
            .expect("the bare task completes");
            assert!(!resolved, "{name}: the handle never resolves");
            assert_eq!(child.queued_update_ids(), Vec::<String>::new(), "{name}");
        }
        assert_eq!(client.recorded_updates(), Vec::new());
    }

    /// `.spawn()` of an invoke and of a `wait_for_condition` runs no
    /// caller-side start step: before the spawned tasks run, neither has a
    /// `START` in the coalescer. Both queue it once their tasks run.
    #[tokio::test(flavor = "current_thread")]
    async fn spawn_of_invoke_and_condition_queues_no_start_before_returning() {
        let (ctx, _client) = recording_context(Some(Duration::from_hours(1)));
        let invoke = ctx
            .invoke::<serde_json::Value, _>("target", serde_json::json!({}))
            .name("i")
            .spawn();
        let condition = ctx
            .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0)
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

        let ctx = DurableContext::test_context();

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
