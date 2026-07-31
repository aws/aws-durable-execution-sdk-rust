//! Conformance requirement 3-6: nested child contexts.

use aws_durable_execution_sdk_rust as durable;

/// Handler: outer child with a step, then inner child with its own step.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let ev = event.clone();
    let result = ctx
        .run_in_child_context(move |outer| async move {
            let ev2 = ev.clone();
            let _: serde_json::Value = outer.step(move |_| async move { Ok(ev2) }).await?;

            let ev3 = ev.clone();
            let inner_result: serde_json::Value = outer
                .run_in_child_context(move |inner| async move {
                    let v: serde_json::Value = inner.step(move |_| async move { Ok(ev3) }).await?;
                    Ok(v)
                })
                .name("inner")
                .await?;
            Ok(inner_result)
        })
        .name("outer")
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
