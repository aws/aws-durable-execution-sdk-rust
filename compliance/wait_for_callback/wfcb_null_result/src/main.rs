//! Conformance handler for requirement 7-15: wait_for_callback with null/empty payload.
//! External system completes with success but no payload.

use aws_durable_execution_sdk_rust as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: serde_json::Value = ctx
            .wait_for_callback::<serde_json::Value, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .await?;
        Ok(result)
    })
    .await
}
