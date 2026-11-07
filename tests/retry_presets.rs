//! Integration tests for the linear backoff and the retry presets: the
//! `linear()` and `no_retry()` presets drive a step through `LocalRunner`,
//! a `Backoff` variant builds from outside the crate, and a partial builder
//! fills the JS `createRetryStrategy` and `createLinearRetryStrategy`
//! values.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::{self, JitterStrategy};
use durable::test_util::{ExecutionStatus, LocalRunner};
use durable::{BoxError, DurableContext};

/// Runs a step with `config` whose body fails on every attempt before
/// `succeed_on` (never, for `None`), and returns the result and the number
/// of times the body ran.
async fn run_step(
    config: builders::RetryStrategyConfig,
    succeed_on: Option<u32>,
) -> (durable::test_util::TestResult<u32>, u32) {
    let runs = Arc::new(AtomicU32::new(0));
    let runs_h = Arc::clone(&runs);
    let result = LocalRunner::new()
        .run(
            move |_: serde_json::Value, ctx: DurableContext| {
                let runs = Arc::clone(&runs_h);
                let config = config.clone();
                async move {
                    let value = ctx
                        .step(move |step_ctx| {
                            let runs = Arc::clone(&runs);
                            async move {
                                runs.fetch_add(1, Ordering::SeqCst);
                                let attempt = step_ctx.attempt();
                                if succeed_on.is_some_and(|n| attempt >= n) {
                                    Ok(attempt * 10)
                                } else {
                                    Err::<u32, BoxError>(format!("fail {attempt}").into())
                                }
                            }
                        })
                        .name("flaky")
                        .retry_strategy_config(config)
                        .await?;
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner completes");
    (result, runs.load(Ordering::SeqCst))
}

/// The linear preset retries a body that fails twice, and the third
/// attempt's value is the output.
#[tokio::test]
async fn linear_preset_retries_until_success() {
    let (result, runs) = run_step(builders::RetryStrategyConfig::linear(), Some(3)).await;
    assert_eq!(result.status(), ExecutionStatus::Succeeded, "{result}");
    assert_eq!(result.output(), Some(&30));
    assert_eq!(runs, 3);
}

/// The no-retry preset fails the step on the first error: the body runs
/// once.
#[tokio::test]
async fn no_retry_preset_fails_on_the_first_error() {
    let (result, runs) = run_step(builders::RetryStrategyConfig::no_retry(), None).await;
    assert_eq!(result.status(), ExecutionStatus::Failed, "{result}");
    assert_eq!(runs, 1);
}

/// A caller outside the crate builds each variant of the `#[non_exhaustive]`
/// `Backoff` enum with its constructor, and matches it with `..`.
#[test]
fn backoff_constructors_build_each_variant() {
    let config = builders::RetryStrategyConfig::builder()
        .backoff(builders::Backoff::linear(Duration::from_secs(2)))
        .build();
    assert!(matches!(
        config.backoff(),
        builders::Backoff::Linear { increment, .. } if increment == Duration::from_secs(2)
    ));
    let config = builders::RetryStrategyConfig::builder()
        .backoff(builders::Backoff::exponential(3.0))
        .build();
    assert!(matches!(
        config.backoff(),
        builders::Backoff::Exponential { rate, .. } if (rate - 3.0).abs() < f64::EPSILON
    ));
}

/// A partial exponential builder fills the JS `createRetryStrategy` values,
/// and `default_preset()` keeps the step default.
#[test]
fn partial_builder_fills_js_values() {
    let config = builders::RetryStrategyConfig::builder()
        .initial_delay(Duration::from_secs(1))
        .build();
    assert_eq!(config.max_attempts(), 3);
    assert_eq!(config.max_delay(), Duration::from_secs(300));
    assert_eq!(config.backoff(), builders::Backoff::exponential(2.0));
    assert_eq!(config.jitter(), JitterStrategy::Full);

    let default = builders::RetryStrategyConfig::default_preset();
    assert_eq!(default.max_attempts(), 6);
    assert_eq!(default.max_delay(), Duration::from_secs(60));
}

/// A partial linear builder fills the JS `createLinearRetryStrategy`
/// values, the fill follows the final backoff, and a set field keeps its
/// value.
#[test]
fn partial_linear_builder_fills_js_linear_values() {
    let linear = builders::Backoff::linear(Duration::from_secs(2));
    let config = builders::RetryStrategyConfig::builder()
        .backoff(linear)
        .build();
    assert_eq!(config.max_attempts(), 6);
    assert_eq!(config.initial_delay(), Duration::from_secs(1));
    assert_eq!(config.max_delay(), Duration::from_secs(300));
    assert_eq!(config.jitter(), JitterStrategy::Full);

    let replaced = builders::RetryStrategyConfig::builder()
        .backoff(linear)
        .backoff_rate(3.0)
        .build();
    assert_eq!(replaced.max_attempts(), 3);
    assert_eq!(replaced.initial_delay(), Duration::from_secs(5));

    let capped = builders::RetryStrategyConfig::builder()
        .backoff(linear)
        .max_attempts(2)
        .build();
    assert_eq!(capped.max_attempts(), 2);
    assert_eq!(capped.initial_delay(), Duration::from_secs(1));
}
