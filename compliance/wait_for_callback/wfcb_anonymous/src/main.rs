//! Conformance handler for requirement 7-3: wait_for_callback with no operation name.

use aws_durable_execution_sdk_rust as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let result: String = ctx
                .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
                .await?;
            Ok(result)
        },
    )
    .await
}
