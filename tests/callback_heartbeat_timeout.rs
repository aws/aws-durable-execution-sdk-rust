//! The public `heartbeat_timeout` setters of `create_callback` and
//! `wait_for_callback`: a callback that receives no heartbeat in time
//! fails with `CallbackErrorKind::HeartbeatTimedOut`.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{CallbackReply, LocalRunner};
use durable::{CallbackErrorKind, OperationError, OperationErrorKind};

/// Whether `err` is a callback error of kind `HeartbeatTimedOut`.
fn heartbeat_timed_out(err: &OperationError) -> bool {
    matches!(
        err.kind(),
        OperationErrorKind::Callback(e)
            if matches!(e.kind(), CallbackErrorKind::HeartbeatTimedOut { .. })
    )
}

#[tokio::test]
async fn heartbeat_timeout_reply_gives_heartbeat_timed_out() {
    let runner = LocalRunner::new()
        .callback_for("cb", CallbackReply::heartbeat_timeout())
        .callback_for("wcb", CallbackReply::heartbeat_timeout());
    let result = runner
        .run(
            |(): (), ctx: durable::DurableContext| async move {
                let cb = ctx
                    .create_callback::<String>()
                    .name("cb")
                    .heartbeat_timeout(Duration::from_secs(90))
                    .await?;
                let created = cb
                    .result()
                    .await
                    .expect_err("no heartbeat arrived for the created callback");
                let waited = ctx
                    .wait_for_callback::<String>(|_step, _callback_id| async { Ok(()) })
                    .name("wcb")
                    .heartbeat_timeout(Duration::from_secs(90))
                    .await
                    .expect_err("no heartbeat arrived for the waited callback");
                assert!(heartbeat_timed_out(&created), "{created:?}");
                assert!(heartbeat_timed_out(&waited), "{waited:?}");
                Ok::<_, durable::BoxError>(true)
            },
            (),
        )
        .await
        .expect("the run completes");
    assert_eq!(result.output(), Some(&true), "{result}");
}
