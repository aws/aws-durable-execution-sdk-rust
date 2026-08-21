//! Create a callback with a timeout and handle the timeout gracefully.
//!
//! [`CreateCallbackBuilder::timeout`](aws_durable_execution_sdk_rust::builders::CreateCallbackBuilder::timeout)
//! bounds how long the execution waits for external completion. If no result
//! arrives in time, awaiting the callback yields a
//! [`CallbackError`](aws_durable_execution_sdk_rust::CallbackError). This
//! example does not arrange any external completion, so it always times out;
//! it catches the error and returns a fallback rather than failing the
//! execution: the same pattern you would use for a real optional approval.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::OperationErrorKind;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let cb = ctx
            .create_callback::<String>()
            .name(&name)
            .timeout(Duration::from_secs(5))
            .await?;
        match cb.result().await {
            Ok(approved) => Ok(approved),
            Err(err) if matches!(err.kind(), OperationErrorKind::Callback(_)) => {
                Ok("no external response before timeout; using fallback".to_owned())
            }
            Err(err) => Err(err.to_string().into()),
        }
    })
    .await
}
