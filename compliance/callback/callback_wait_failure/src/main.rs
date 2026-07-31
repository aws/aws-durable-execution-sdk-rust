//! Conformance handler for requirement 4-10: callback + wait + await failure.
//! Creates a callback, waits 5s, then awaits callback result (failure).

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx.create_callback::<String>().name(&name).await?;
        ctx.wait(Duration::from_secs(5)).name("delay").await?;
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
