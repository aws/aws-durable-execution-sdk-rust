//! `CloudRunner` live-control test against a deployed function.
//!
//! The cloud example suite (`examples/cloud/run_cloud_tests.sh`) runs this
//! test with `--ignored` after it deployed the example stacks, and sets
//! `CLOUD_CALLBACK_FUNCTION` to the deployed `create_callback_simple`
//! function. A plain `cargo test` skips it, because it needs AWS.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{
    CallbackReply, CloudRunner, ExecutionStatus, InvocationType, WaitingStatus,
};

/// The `create_callback_simple` example names its callback after the event.
const CALLBACK_NAME: &str = "cloud-runner-approval";

#[tokio::test]
#[ignore = "needs the deployed example stacks; the cloud example suite runs it"]
async fn cloud_answers_a_callback_while_running() {
    let function = std::env::var("CLOUD_CALLBACK_FUNCTION")
        .expect("CLOUD_CALLBACK_FUNCTION names the deployed create_callback_simple function");
    let run = CloudRunner::new(function)
        .invocation_type(InvocationType::Event)
        .poll_interval(Duration::from_secs(2))
        .run::<String>(CALLBACK_NAME);
    let callback = run.operation(CALLBACK_NAME);
    callback
        .wait_for_data(WaitingStatus::Started)
        .await
        .expect("the callback started");
    callback
        .send_callback(CallbackReply::success(&"approved by CloudRunner").expect("encodes"))
        .await
        .expect("the reply was sent");
    let result = run.await.expect("runner");
    assert_eq!(result.status(), ExecutionStatus::Succeeded);
    assert_eq!(
        result.output().map(String::as_str),
        Some("approved by CloudRunner")
    );
}
