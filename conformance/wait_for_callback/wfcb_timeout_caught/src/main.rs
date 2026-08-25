//! Conformance handler for requirement 7-14: wait_for_callback timeout caught.
//! Handler catches timeout and returns "timed-out-handled".

use aws_durable_execution_sdk as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: Result<String, _> = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .timeout(Duration::from_secs(3))
            .await;
        match result {
            Ok(val) => Ok(val),
            Err(_) => Ok("timed-out-handled".to_owned()),
        }
    })
    .await
}
