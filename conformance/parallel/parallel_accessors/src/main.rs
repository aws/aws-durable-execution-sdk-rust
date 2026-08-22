//! Conformance requirement 8-20: Parallel `BatchResult` accessors
//! (success/failure counts, errors, has-failure).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, DurableContext};
use durable::builders::map_parallel::CompletionConfig;

/// Handler: returns accessor-derived projection from the batch result.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Ok("ok0".to_owned()) }),
        Branch::new("1", |_: DurableContext| async {
            Err("branch failed".into())
        }),
        Branch::new("2", |_: DurableContext| async { Ok("ok2".to_owned()) }),
    ];

    let batch = ctx
        .parallel(branches)
        .name("accessors")
        .max_concurrency(1)
        .completion(CompletionConfig::with_tolerated_failure_count(1))
        .await_batch()
        .await?;

    Ok(serde_json::json!({
        "hasFailure": batch.has_failure(),
        "successCount": batch.success_count(),
        "failureCount": batch.failure_count(),
        "errorCount": batch.errors().len(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
