//! Create a callback token, hand it to an external system, and block until that
//! system reports the result.
//!
//! [`DurableContext::create_callback`](aws_durable_execution_sdk_rust::DurableContext::create_callback)
//! mints a durable callback and returns a
//! [`Callback`](aws_durable_execution_sdk_rust::Callback) immediately. The
//! callback id is what you deliver to the outside world (a webhook target, a
//! queue message, an approval email). The execution then suspends on
//! `cb.result()` until an external caller completes the callback with
//! `SendDurableExecutionCallbackSuccess`, at which point the recorded result is
//! returned. The event is used as the callback's name.

use aws_durable_execution_sdk_rust as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx.create_callback::<String>().name(&name).await?;
        // Deliver cb.id() to the external system here. It is logged so an
        // operator can trace (and, in this example's smoke test, complete) it.
        tracing::info!(callback_id = %cb.id(), "callback created; awaiting external completion");
        let approval = cb.result().await?;
        Ok(approval)
    })
    .await
}
