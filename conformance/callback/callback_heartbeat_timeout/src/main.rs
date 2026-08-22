//! Conformance handler for requirement 4-4: create callback with heartbeat timeout.
//! No heartbeat is sent, so it times out on the heartbeat interval.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx
            .create_callback::<String>()
            .name(&name)
            .heartbeat(Duration::from_secs(5))
            .await?;
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
