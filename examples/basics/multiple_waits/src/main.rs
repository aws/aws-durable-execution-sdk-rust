//! Multiple sequential waits: each wait is its own checkpoint.
//!
//! Two waits in sequence produce two suspensions and therefore (at least) two
//! invocations of the function: the execution suspends at the first wait,
//! resumes and replays past it, suspends again at the second, then resumes to
//! completion. Because each wait is a distinct operation minted in call order,
//! replay pairs each resumption with the correct wait: the ordering is
//! deterministic regardless of timing.
//!
//! Returning a small typed struct shows that a durable function's output is
//! serialized like any operation result.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use serde::Serialize;

/// Handler output: a small summary the caller receives.
#[derive(Debug, Serialize)]
struct Output {
    /// Number of waits that completed.
    completed_waits: u32,
    /// A final marker so the caller can see the function ran to the end.
    final_step: &'static str,
}

/// Waits twice in sequence (two suspensions), then returns a summary.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Output, durable::BoxError> {
    ctx.wait(Duration::from_secs(2)).name("wait-1").await?;
    ctx.wait(Duration::from_secs(2)).name("wait-2").await?;

    Ok(Output {
        completed_waits: 2,
        final_step: "done",
    })
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
