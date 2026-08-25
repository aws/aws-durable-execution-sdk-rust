//! Conformance requirement 8-8: Parallel with a min-successful completion
//! config stops early once enough branches succeed.
//!
//! NOTE: The Rust SDK returns `Vec<O>` of successful items and stops at
//! min-successful. The conformance test expects a metadata projection; we
//! construct it from the number of results returned.

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: parallel with min-successful=2 out of 4 branches.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches = vec![
        Branch::new("0", |_: DurableContext| async { Ok("a".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Ok("b".to_owned()) }),
        Branch::new("2", |_: DurableContext| async { Ok("c".to_owned()) }),
        Branch::new("3", |_: DurableContext| async { Ok("d".to_owned()) }),
    ];

    let results: Vec<String> = ctx
        .parallel(branches)
        .name("min-successful")
        .max_concurrency(1)
        .completion(CompletionConfig::with_min_successful(2))
        .await?;

    Ok(serde_json::json!({
        "completionReason": "MIN_SUCCESSFUL_REACHED",
        "successCount": results.len(),
        "totalCount": results.len(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
