//! Conformance requirement 3-8: child with retry exhaustion.

use aws_durable_execution_sdk_rust as durable;
use std::time::Duration;

/// Handler: child step always fails, retries exhaust.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result: String = ctx
        .run_in_child_context(|child_ctx| async move {
            let v: String = child_ctx
                .step(|_| async move { Err("Always fails".into()) })
                .retry_strategy(|_, attempt| {
                    if attempt >= 2 {
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
        .name("exhaust-child")
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
