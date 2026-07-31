//! Conformance handler for requirement 4-11: callback (3s timeout) + wait 6s + await.
//! Callback times out during the wait period.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx
            .create_callback::<String>()
            .name(&name)
            .timeout(Duration::from_secs(3))
            .await?;
        ctx.wait(Duration::from_secs(6)).name("delay").await?;
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
