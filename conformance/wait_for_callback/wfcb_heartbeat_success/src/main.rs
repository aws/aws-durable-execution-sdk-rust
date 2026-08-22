//! Conformance handler for requirement 7-13: wait_for_callback with 10s heartbeat, success.
//! External system sends heartbeat then success.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .heartbeat(Duration::from_secs(10))
            .await?;
        Ok(result)
    })
    .await
}
