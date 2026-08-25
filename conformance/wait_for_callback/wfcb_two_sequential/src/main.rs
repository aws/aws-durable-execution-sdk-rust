//! Conformance handler for requirement 7-9: two sequential wait_for_callback operations.
//! First named "first", second named "second"; returns the second result.

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let _first: String = ctx
                .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
                .name("first")
                .await?;
            let second: String = ctx
                .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
                .name("second")
                .await?;
            Ok(second)
        },
    )
    .await
}
