//! Wait for a callback with a timeout, and handle the timeout.
//!
//! [`WaitForCallbackBuilder::timeout`](aws_durable_execution_sdk_rust::WaitForCallbackBuilder::timeout)
//! bounds the wait. No external completion is arranged here, so the callback
//! always times out; the handler catches the error and returns a fallback
//! rather than failing the execution.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let outcome: Result<String, _> = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .timeout(Duration::from_secs(3))
            .await;
        match outcome {
            Ok(result) => Ok(result),
            Err(_) => Ok("no external response before timeout; using fallback".to_owned()),
        }
    })
    .await
}
