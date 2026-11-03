//! Running a handler installs no global subscriber.
//!
//! This file holds one test, so no other test in its binary installs a
//! subscriber before it checks.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use aws_durable_execution_sdk as durable;
use durable::test_util::LocalRunner;

#[tokio::test]
async fn running_a_handler_installs_no_subscriber() {
    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                let value = ctx.step(|_| async { Ok(1_u32) }).name("one").await?;
                Ok::<_, durable::BoxError>(value)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

    assert_eq!(result.output(), Some(&1));
    assert!(
        !tracing::dispatcher::has_been_set(),
        "running a handler must not install a global subscriber"
    );
}
