//! Basic step: checkpoint one unit of work.
//!
//! [`ctx.step`](aws_durable_execution_sdk_rust::DurableContext::step) runs a
//! closure and records its result. The first time the function runs, the
//! closure executes and the return value is checkpointed. If the execution is
//! later interrupted and replayed, the closure is NOT run again — the recorded
//! value is returned in its place. This is the core durable-execution
//! guarantee: work inside a completed step happens exactly once across all
//! replays.
//!
//! A step body may be nondeterministic (call a service, read the clock); only
//! the recorded result participates in replay. Note that the step closure
//! receives a
//! [`StepContext`](aws_durable_execution_sdk_rust::StepContext), which
//! deliberately exposes no durable operations — nesting durable operations
//! inside a step is a compile error, not a runtime surprise.

use aws_durable_execution_sdk_rust as durable;

/// Runs one checkpointed step and returns its result.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let result = ctx
        .step(|_ctx| async { Ok("step completed".to_owned()) })
        .await?;
    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
