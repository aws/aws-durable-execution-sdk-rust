//! Conformance handler for requirement 7-1: wait_for_callback basic.
//! Submitter is a no-op; external system completes the callback with success.

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .await?;
        Ok(result)
    })
    .await
}
