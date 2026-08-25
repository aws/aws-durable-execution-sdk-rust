//! Conformance handler for requirement 4-1: create callback basic.
//! Creates a callback using the input as the name, blocks on result, returns it.

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx.create_callback::<String>().name(&name).await?;
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
