//! Conformance handler for requirement 7-12: wait_for_callback with 5s heartbeat timeout.
//! No heartbeat sent, so it times out.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .heartbeat(Duration::from_secs(5))
            .await?;
        Ok(result)
    })
    .await
}
