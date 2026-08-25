//! Conformance requirement 3-15: child throws error without any durable op.

use aws_durable_execution_sdk as durable;

/// Handler: child function returns error directly (no ops).
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result: String = ctx
        .run_in_child_context(|_child_ctx| async move { Err("direct error".into()) })
        .name("direct-error")
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
