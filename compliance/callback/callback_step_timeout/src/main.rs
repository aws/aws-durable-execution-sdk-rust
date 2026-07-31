//! Conformance handler for requirement 4-8: callback (5s timeout) + step + await timeout.
//! Creates a callback with 5s timeout, runs a step, then awaits (timeout fires).

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx
            .create_callback::<String>()
            .name(&name)
            .timeout(Duration::from_secs(5))
            .await?;
        let _: String = ctx
            .step(|_| async { Ok("notified".to_owned()) })
            .name("notify-external")
            .await?;
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
