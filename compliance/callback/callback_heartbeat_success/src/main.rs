//! Conformance handler for requirement 4-5: create callback with heartbeat then success.
//! Heartbeat keeps callback alive past initial heartbeat interval.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx
            .create_callback::<String>()
            .name(&name)
            .heartbeat(Duration::from_secs(10))
            .await?;
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
