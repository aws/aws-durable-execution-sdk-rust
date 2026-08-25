//! Conformance requirement 3-2: named child context.

use aws_durable_execution_sdk as durable;
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    name: String,
    value: String,
}

/// Handler: named child context with a single step returning the value.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let input: Input = serde_json::from_value(event)?;
    let result = ctx
        .run_in_child_context(move |child_ctx| async move {
            let v: String = child_ctx
                .step(move |_| {
                    let val = input.value.clone();
                    async move { Ok(val) }
                })
                .await?;
            Ok(v)
        })
        .name(input.name)
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
