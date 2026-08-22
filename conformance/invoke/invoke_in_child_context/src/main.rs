//! Conformance requirement 5-13: invoke inside a child context.

use aws_durable_execution_sdk_rust as durable;

/// Handler: invoke inside a child context.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;

    let result: serde_json::Value = ctx
        .run_in_child_context(move |child_ctx| async move {
            let r: serde_json::Value = child_ctx
                .invoke::<serde_json::Value, _>(&target, serde_json::Value::Null)
                .await?;
            Ok(r)
        })
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
