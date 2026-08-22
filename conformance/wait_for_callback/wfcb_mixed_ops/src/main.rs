//! Conformance handler for requirement 7-10: wait + step + wait_for_callback.
//! Handler first waits 1s, then runs a step, then wait_for_callback.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        ctx.wait(Duration::from_secs(1)).await?;
        let _: String = ctx.step(|_| async { Ok("fixed-data".to_owned()) }).await?;
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .await?;
        Ok(result)
    })
    .await
}
