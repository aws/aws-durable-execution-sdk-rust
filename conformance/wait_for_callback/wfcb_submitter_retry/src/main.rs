//! Conformance handler for requirement 7-7: wait_for_callback submitter retry exhaustion.
//! Submitter always throws; retry policy allows 2 attempts (one retry) with 1s delay.

use std::time::Duration;

use aws_durable_execution_sdk as durable;

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: String = ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _cb_id| async {
                Err("submitter failure".into())
            })
            .name(&name)
            .submitter_retry(|_err, attempt| {
                if attempt >= 2 {
                    durable::RetryDecision::Stop
                } else {
                    durable::RetryDecision::Retry {
                        delay: Duration::from_secs(1),
                    }
                }
            })
            .await?;
        Ok(result)
    })
    .await
}
