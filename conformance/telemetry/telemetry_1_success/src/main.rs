//! Telemetry conformance requirement telemetry-1: the plugin exports one
//! complete `SUCCEEDED` record for a single-step workflow.

use aws_durable_execution_sdk as durable;
use aws_durable_execution_sdk_telemetry_exporter::{
    CloudWatchLogsExporter, DurableExecutionTelemetry, EmitMode, OperationDetail,
};
use serde_json::{Value, json};

/// Handler: the step `increment` adds one to `event.value`.
async fn handler(event: Value, ctx: durable::DurableContext) -> Result<Value, durable::BoxError> {
    let value = event
        .get("value")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let next = ctx
        .step(move |_| async move { Ok(value + 1) })
        .name("increment")
        .await?;
    Ok(json!({ "value": next }))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    // The template sets the variable to the log group that the suite reads.
    let log_group = std::env::var("TELEMETRY_LOG_GROUP_NAME")?;
    let telemetry = DurableExecutionTelemetry::builder()
        .exporter(CloudWatchLogsExporter::builder(log_group).build())
        .emit_mode(EmitMode::OnComplete)
        .operation_detail(OperationDetail::TopLevel)
        .sampling_rate(1.0)
        .build();
    let options = durable::Options::builder().plugin(telemetry).build()?;
    durable::run_with_options(handler, options).await
}
