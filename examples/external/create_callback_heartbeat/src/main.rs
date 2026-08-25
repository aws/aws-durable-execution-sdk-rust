//! Create a callback that expects periodic heartbeats.
//!
//! When an external worker may take a long time, a heartbeat interval keeps the
//! callback alive: the worker calls `SendDurableExecutionCallbackHeartbeat`
//! before each interval elapses, and the callback only times out if a heartbeat
//! is missed. Configure it with
//! [`CreateCallbackBuilder::heartbeat`](aws_durable_execution_sdk::builders::CreateCallbackBuilder::heartbeat).
//! Otherwise this behaves exactly like the basic create-callback example.

use std::time::Duration;

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx
            .create_callback::<String>()
            .name(&name)
            .heartbeat(Duration::from_secs(10))
            .await?;
        tracing::info!(callback_id = %cb.id(), "callback created with heartbeat; awaiting completion");
        let result = cb.result().await?;
        Ok(result)
    })
    .await
}
