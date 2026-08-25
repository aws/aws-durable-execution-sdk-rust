//! Conformance requirement 13-1: `try_join_all` task ownership (issue #7).
//!
//! The combinator drives its branch futures on internal `JoinSet` tasks,
//! which must be blessed with the task-ownership guard: an unblessed
//! branch is rejected as a foreign task and the combinator fails.

use aws_durable_execution_sdk as durable;

/// Handler: joins two step futures and returns the sum of their results.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let a = ctx.step(|_| async { Ok(1_i64) }).name("one").future();
    let b = ctx.step(|_| async { Ok(2_i64) }).name("two").future();

    let results: Vec<i64> = ctx.try_join_all([a, b]).name("join").await?;
    let sum: i64 = results.iter().sum();
    Ok(serde_json::Value::from(sum))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
