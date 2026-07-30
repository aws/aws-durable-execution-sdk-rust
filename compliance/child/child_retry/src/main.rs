//! Conformance requirement 3-7: child with retry (fails first, succeeds second).

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: child step fails first attempt, succeeds on retry.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let v: serde_json::Value = child_ctx
                .step(move |step_ctx| {
                    let e = event.clone();
                    async move {
                        if step_ctx.attempt() < 2 {
                            Err(format!("Attempt {} failed", step_ctx.attempt()).into())
                        } else {
                            Ok(e)
                        }
                    }
                })
                .retry_strategy(|_, attempt| {
                    if attempt >= 3 {
                        durable::RetryDecision::Stop
                    } else {
                        durable::RetryDecision::Retry {
                            delay: Duration::from_secs(1),
                        }
                    }
                })
                .await?;
            Ok(v)
        })
        .name("retry-child")
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
