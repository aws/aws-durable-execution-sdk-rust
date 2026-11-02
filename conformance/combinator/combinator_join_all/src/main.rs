//! Conformance requirement combinator-2: `join_all` task ownership (issue #7).
//!
//! The silent-corruption case: without task-ownership blessing, every
//! branch settles as `Err` with the ownership error while `join_all`
//! still reports overall success. Counting `Ok` and `Err` outcomes
//! distinguishes genuine branch execution (exactly one of each)
//! from the all-rejected failure mode.

use aws_durable_execution_sdk as durable;

/// Handler: joins one succeeding and one failing step, returns the counts.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let no_retry = |_err: &durable::StepError, _attempt: u32| durable::RetryDecision::Stop;

    let ok = ctx
        .step(|_| async { Ok("fine".to_owned()) })
        .name("ok")
        .future();
    let bad = ctx
        .step(|_| async { Err("intentional failure".into()) })
        .name("bad")
        .retry_strategy(no_retry)
        .future();

    let outcomes: Vec<Result<String, durable::OperationError>> =
        ctx.join_all([ok, bad]).name("collect").await?;

    let fulfilled = outcomes.iter().filter(|s| s.is_ok()).count();
    let rejected = outcomes.iter().filter(|s| s.is_err()).count();

    Ok(serde_json::Value::String(format!(
        "fulfilled={fulfilled},rejected={rejected}"
    )))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
