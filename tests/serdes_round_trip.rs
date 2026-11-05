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
