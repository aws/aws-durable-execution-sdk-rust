//! Conformance handler for requirement 4-2: create callback with explicit name.
//! Creates a callback named "approval" and blocks on the result.

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let cb = ctx.create_callback::<String>().name("approval").await?;
            let result = cb.result().await?;
            Ok(result)
        },
    )
    .await
}
