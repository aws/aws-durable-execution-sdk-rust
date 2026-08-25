//! Conformance handler for requirement 7-8: wait_for_callback inside a child context.
//! Nested inside a child context named "wrapper".

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .run_in_child_context(|child_ctx| async move {
                let val: String = child_ctx
                    .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
                    .name(&name)
                    .await?;
                Ok(val)
            })
            .name("wrapper")
            .await?;
        Ok(result)
    })
    .await
}
