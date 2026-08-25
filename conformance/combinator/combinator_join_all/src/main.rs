//! Conformance requirement 13-2: `join_all` task ownership (issue #7).
//!
//! The silent-corruption case: without task-ownership blessing, every
//! branch settles `Rejected` with the ownership error while `join_all`
//! still reports overall success. Counting fulfilled and rejected
//! outcomes distinguishes genuine branch execution (exactly one of each)
//! from the all-rejected failure mode.

use aws_durable_execution_sdk as durable;
use durable::Settled;

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

    let settled: Vec<Settled<String>> = ctx.join_all([ok, bad]).name("collect").await?;

    let fulfilled = settled
        .iter()
        .filter(|s| matches!(s, Settled::Fulfilled(_)))
        .count();
    let rejected = settled
        .iter()
        .filter(|s| matches!(s, Settled::Rejected(_)))
        .count();

    Ok(serde_json::Value::String(format!(
        "fulfilled={fulfilled},rejected={rejected}"
    )))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
