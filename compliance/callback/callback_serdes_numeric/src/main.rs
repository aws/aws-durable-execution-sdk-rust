//! Conformance handler for requirement 4-16: callback with numeric deserialization.
//! Callback payload is a number, handler returns count and doubled value.

use aws_durable_execution_sdk_rust as durable;
use serde::Serialize;

/// Output returned by the handler.
#[derive(Serialize)]
struct NumericResult {
    count: i32,
    doubled: i32,
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx.create_callback::<i32>().name(&name).await?;
        let value = cb.result().await?;
        Ok(NumericResult {
            count: value,
            doubled: value * 2,
        })
    })
    .await
}
