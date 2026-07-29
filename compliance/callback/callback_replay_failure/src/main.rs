//! Conformance handler for requirement 4-13: callback failure caught → wait → return.
//! Handler catches callback failure and continues with a wait.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx.create_callback::<String>().name(&name).await?;
        let outcome = match cb.result().await {
            Ok(val) => val,
            Err(e) => format!("caught_failure:{e}"),
        };
        ctx.wait(Duration::from_secs(2)).name("after-cb").await?;
        Ok(outcome)
    })
    .await
}
