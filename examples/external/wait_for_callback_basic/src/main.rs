//! Wait for a callback whose token is delivered by a submitter closure.
//!
//! [`DurableContext::wait_for_callback`](aws_durable_execution_sdk::DurableContext::wait_for_callback)
//! combines minting a callback and running a submitter into one operation: the
//! submitter receives the callback id and is responsible for delivering it to
//! the external system, then the execution suspends until the callback is
//! completed. Here the submitter is a no-op (the smoke test delivers the id out
//! of band); a real submitter would enqueue a job or call an API.

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .await?;
        Ok(result)
    })
    .await
}
