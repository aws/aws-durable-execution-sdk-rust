//! Conformance handler for requirement 7-5: wait_for_callback with 3s timeout.
//! No external completion, so it times out.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .timeout(Duration::from_secs(3))
            .await?;
        Ok(result)
    })
    .await
}
