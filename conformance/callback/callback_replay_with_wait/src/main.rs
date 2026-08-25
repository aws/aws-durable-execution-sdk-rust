//! Conformance handler for requirement 4-12: callback success → wait → return.
//! Verifies replay of both callback and wait across 3 invocations.

use aws_durable_execution_sdk as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx.create_callback::<String>().name(&name).await?;
        let result = cb.result().await?;
        ctx.wait(Duration::from_secs(2)).name("after-cb").await?;
        Ok(result)
    })
    .await
}
