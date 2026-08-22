//! Conformance handler for requirement 4-7: callback + step + await failure.
//! Creates a callback, runs a step, then awaits callback result (failure).

use aws_durable_execution_sdk_rust as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx.create_callback::<String>().name(&name).await?;
        let _: String = ctx
            .step(|_| async { Ok("notified".to_owned()) })
            .name("notify-external")
            .await?;
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
