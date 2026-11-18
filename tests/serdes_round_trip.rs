//! On a live run an operation returns `deserialize(serialize(value))`, not
//! the value its body produced, and on replay it returns
//! `deserialize(stored)`. So a serdes that changes the value gives the
//! handler the same changed value live and on replay.
//!
//! Each test runs the operation, records what it returned, and then waits
//! 1 s. So the operation runs live in the first invocation and replays from
//! its record in the second.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::serdes::SerdesContext;
use durable::test_util::{LocalRunner, TestResult};
use durable::{BoxError, DurableContext, Serdes};

/// Writes the value as a JSON string unchanged, and lowercases it when it
/// reads the JSON string back.
struct LowercaseOnDeserialize;

impl Serdes<String> for LowercaseOnDeserialize {
    fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(serde_json::to_string(&value).map_err(Into::into))
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(
            serde_json::from_str::<String>(&wire)
                .map(|value| value.to_lowercase())
                .map_err(Into::into),
        )
    }
}

/// Checks the four results both tests expect, for the operation `name`.
fn assert_round_tripped(result: &TestResult<String>, seen: &[String], name: &str) {
    assert_eq!(
        seen,
        ["hello", "hello"],
        "live and replay return the decoded value"
    );
    assert_eq!(result.output(), Some(&"hello".to_owned()));
    assert_eq!(result.invocation_count(), 2);
    let recorded = result
        .operation(name)
        .expect("the operation is recorded")
        .result();
    assert_eq!(
        recorded,
        Some("\"HELLO\""),
        "serialize stored the value the body produced"
    );
}

#[tokio::test]
async fn step_returns_round_tripped_value_live_and_on_replay() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let seen_in_handler = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |(): (), ctx: DurableContext| {
                let seen = Arc::clone(&seen_in_handler);
                async move {
                    let value = ctx
                        .step(|_| async { Ok("HELLO".to_owned()) })
                        .name("s")
                        .serdes(LowercaseOnDeserialize)
                        .await?;
                    seen.lock().expect("lock").push(value.clone());
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(value)
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    let seen = seen.lock().expect("lock").clone();
    assert_round_tripped(&result, &seen, "s");
}

#[tokio::test]
async fn child_returns_round_tripped_value_live_and_on_replay() {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let seen_in_handler = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |(): (), ctx: DurableContext| {
                let seen = Arc::clone(&seen_in_handler);
                async move {
                    let value = ctx
                        .run_in_child_context(|_child| async { Ok("HELLO".to_owned()) })
                        .name("c")
                        .serdes(LowercaseOnDeserialize)
                        .await?;
                    seen.lock().expect("lock").push(value.clone());
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(value)
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    let seen = seen.lock().expect("lock").clone();
    assert_round_tripped(&result, &seen, "c");
}

/// How a map ended: `"Ok"`, or the child-context error kind and whether
/// the error carries a recorded wire record.
fn outcome_class<T>(outcome: &Result<T, durable::OperationError>) -> String {
    match outcome {
        Ok(_) => "Ok".to_owned(),
        Err(err) => match err.kind() {
            durable::OperationErrorKind::ChildContext(child) => {
                format!("{:?}, wire {}", child.kind(), err.wire().is_some())
            }
            other => format!("{other:?}"),
        },
    }
}

/// Runs a map over one item whose value is `len` NaNs, which `JsonSerdes`
/// writes as `null` and cannot read back, with `max_concurrency`, then
/// waits 1 s. Returns the outcome class of the map in each invocation,
/// awaited with `.await_batch()` when `batch` is true and with `.await`
/// otherwise. A `max_concurrency` of 1 rebuilds the item at the sequential
/// dispatch cursor, and 2 rebuilds it in the concurrent replay pass.
async fn map_of_nan_then_wait(
    runner: LocalRunner,
    len: usize,
    max_concurrency: usize,
    batch: bool,
) -> Vec<String> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let result = runner
        .run(
            move |(), ctx: DurableContext| {
                let seen = Arc::clone(&handler_seen);
                async move {
                    let map = ctx
                        .map(vec![0_u32], move |_child, _item, _idx| async move {
                            Ok(vec![f64::NAN; len])
                        })
                        .name("m")
                        .max_concurrency(max_concurrency);
                    let class = if batch {
                        outcome_class(&map.await_batch().await)
                    } else {
                        outcome_class(&map.await)
                    };
                    seen.lock().expect("lock").push(class);
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(())
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    assert_eq!(result.invocation_count(), 2, "{result}");
    seen.lock().expect("lock").clone()
}

/// What both invocations must see: the live decode error.
fn live_error_twice() -> [String; 2] {
    let internal = "Internal, wire false".to_owned();
    [internal.clone(), internal]
}

#[tokio::test]
async fn map_item_decode_failure_after_oversized_succeed_fails_the_resumed_map() {
    // 60 000 `null,` are 300 000 bytes, so the item records `ReplayChildren`.
    let seen = map_of_nan_then_wait(LocalRunner::new(), 60_000, 1, true).await;
    assert_eq!(seen, live_error_twice());
}

#[tokio::test]
async fn map_item_decode_failure_with_preserved_children_fails_the_resumed_map() {
    let options = durable::Options::builder()
        .plugin_child_operations_depth(2)
        .build()
        .expect("valid options");
    let seen = map_of_nan_then_wait(LocalRunner::new().options(options), 1, 1, true).await;
    assert_eq!(seen, live_error_twice());
}

#[tokio::test]
async fn map_item_decode_failure_after_oversized_succeed_returns_the_live_error_on_resume() {
    let seen = map_of_nan_then_wait(LocalRunner::new(), 60_000, 1, false).await;
    assert_eq!(seen, live_error_twice());
}

#[tokio::test]
async fn concurrent_map_item_decode_failure_after_oversized_succeed_fails_the_resumed_map() {
    let seen = map_of_nan_then_wait(LocalRunner::new(), 60_000, 2, true).await;
    assert_eq!(seen, live_error_twice());
}
