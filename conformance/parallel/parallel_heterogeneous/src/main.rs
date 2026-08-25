//! Conformance requirement 8-4: Parallel whose branches return different types
//! (string, number, object).

use aws_durable_execution_sdk as durable;
use durable::{Branch, DurableContext};

/// Handler: heterogeneous parallel branches using `serde_json::Value`.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let branches: Vec<Branch<serde_json::Value>> = vec![
        Branch::new("0", |_: DurableContext| async {
            Ok(serde_json::Value::String("hello".to_owned()))
        }),
        Branch::new("1", |_: DurableContext| async { Ok(serde_json::json!(42)) }),
        Branch::new("2", |_: DurableContext| async {
            Ok(serde_json::json!({"k": "v"}))
        }),
    ];

    let results: Vec<serde_json::Value> = ctx
        .parallel(branches)
        .name("hetero")
        .max_concurrency(1)
        .await?;

    Ok(serde_json::to_value(results)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
