//! Conformance requirement 13-3: `select_ok` task ownership (issue #7).
//!
//! Without task-ownership blessing, every branch is rejected as a foreign
//! task, so no branch can win. With the blessing in place, the failing
//! branch loses and the succeeding branch's value is returned.

use aws_durable_execution_sdk_rust as durable;

/// Handler: selects the first successful of a failing and a succeeding step.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let no_retry = |_err: &durable::StepError, _attempt: u32| durable::RetryDecision::Stop;

    let bad = ctx
        .step(|_| async { Err("intentional failure".into()) })
        .name("bad")
        .retry_strategy(no_retry)
        .future();
    let ok = ctx
        .step(|_| async { Ok("winner".to_owned()) })
        .name("ok")
        .future();

    let winner = ctx.select_ok([bad, ok]).name("select").await?;
    Ok(serde_json::Value::String(winner))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
