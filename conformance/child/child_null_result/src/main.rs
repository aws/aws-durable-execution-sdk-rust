//! Conformance requirement 3-16: child returns null without any durable op.

use aws_durable_execution_sdk as durable;

/// Handler: child returns null (None) without calling any operations.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result: Option<String> = ctx
        .run_in_child_context(|_child_ctx| async move { Ok(None) })
        .name("null-child")
        .await?;
    match result {
        Some(v) => Ok(serde_json::Value::String(v)),
        None => Ok(serde_json::Value::Null),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
