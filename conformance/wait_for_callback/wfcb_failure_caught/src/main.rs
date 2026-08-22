//! Conformance handler for requirement 7-6: wait_for_callback failure caught.
//! Handler catches failure and returns "recovered".

use aws_durable_execution_sdk_rust as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: Result<String, _> = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .await;
        match result {
            Ok(val) => Ok(val),
            Err(_) => Ok("recovered".to_owned()),
        }
    })
    .await
}
