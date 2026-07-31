//! Wait for a callback with a real submitter and a submitter retry policy.
//!
//! The submitter is where you hand the callback id to the outside world. It
//! runs as a durable step, so a submitter retry policy
//! ([`WaitForCallbackBuilder::submitter_retry`](aws_durable_execution_sdk_rust::WaitForCallbackBuilder::submitter_retry))
//! governs transient delivery failures independently of how long the external
//! system then takes to respond. This submitter logs the id (standing in for a
//! queue put or API call) and succeeds; the execution suspends until the
//! callback is completed externally.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::RetryDecision;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, cb_id| {
                let cb_id = cb_id.to_owned();
                async move {
                    // Deliver the token to the external system. Logged here.
                    tracing::info!(callback_id = %cb_id, "submitted callback token");
                    Ok(())
                }
            })
            .name(&name)
            .submitter_retry(|_err, attempt| {
                if attempt >= 3 {
                    RetryDecision::Stop
                } else {
                    RetryDecision::Retry {
                        delay: Duration::from_secs(1),
                    }
                }
            })
            .await?;
        Ok(result)
    })
    .await
}
