//! Conformance requirement 3-12: child step interrupted on first attempt.

use aws_durable_execution_sdk_rust as durable;

/// Handler: child step exits process on first attempt, succeeds on second.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let execution_id = ctx.execution_arn().to_owned();

    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let eid = execution_id.clone();
            let v: serde_json::Value = child_ctx
                .step(move |_step_ctx| {
                    let e = event.clone();
                    let id = eid.clone();
                    async move {
                        let count = conformance::increment_attempt(&id).await?;
                        if count < 2 {
                            // Allow checkpoint to flush before crashing.
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                            std::process::exit(1);
                        }
                        Ok(e)
                    }
                })
                .retry_strategy(|_, _| durable::RetryDecision::Stop)
                .await?;
            Ok(v)
        })
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
