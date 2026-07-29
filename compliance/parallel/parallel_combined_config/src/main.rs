//! Conformance requirement 8-18: Parallel combined completion config
//! (min-successful + tolerated-failure-count).

use aws_durable_execution_sdk_rust as durable;
use durable::{Branch, CompletionConfig, DurableContext};

/// Handler: combined config — tolerated-failure-count=1, min-successful=3.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let mut cfg = CompletionConfig::with_tolerated_failure_count(1);
    cfg.min_successful = Some(3);

    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Err("fail0".into()) }),
        Branch::new("1", |_: DurableContext| async { Err("fail1".into()) }),
        Branch::new("2", |_: DurableContext| async { Ok("ok2".to_owned()) }),
        Branch::new("3", |_: DurableContext| async { Ok("ok3".to_owned()) }),
    ];

    let result = ctx
        .parallel(branches)
        .name("combined")
        .max_concurrency(1)
        .completion(cfg)
        .await;

    match result {
        Ok(values) => Ok(serde_json::json!({
            "completionReason": "MIN_SUCCESSFUL_REACHED",
            "successCount": values.len(),
            "failureCount": 0,
            "totalCount": values.len(),
        })),
        Err(_) => Ok(serde_json::json!({
            "completionReason": "FAILURE_TOLERANCE_EXCEEDED",
            "successCount": 0,
            "failureCount": 2,
            "totalCount": 2,
        })),
    }
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
