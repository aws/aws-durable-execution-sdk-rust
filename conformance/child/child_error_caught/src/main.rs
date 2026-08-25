//! Conformance requirement 3-5: child error caught, execution continues.

use aws_durable_execution_sdk as durable;

/// Handler: child fails, error caught, recovery step follows.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let child_result: Result<String, durable::OperationError> = ctx
        .run_in_child_context(|child_ctx| async move {
            let v: String = child_ctx
                .step(|_| async move { Err("Child step failed".into()) })
                .retry_strategy(|_, _| durable::RetryDecision::Stop)
                .await?;
            Ok(v)
        })
        .await;

    // Catch the ChildContextError and continue.
    if let Err(ref e) = child_result {
        if !matches!(e.kind(), durable::OperationErrorKind::ChildContext(_)) {
            return Err(e.to_string().into());
        }
    }

    let ev = event.clone();
    let result: serde_json::Value = ctx.step(move |_| async move { Ok(ev) }).await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
