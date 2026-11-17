//! Telemetry conformance requirement telemetry-2: the plugin exports one
//! complete `FAILED` record when a single step fails.

use aws_durable_execution_sdk as durable;
use aws_durable_execution_sdk_telemetry_exporter::{
    CloudWatchLogsExporter, DurableExecutionTelemetry, EmitMode, OperationDetail,
};
use durable::RetryDecision;
use serde_json::Value;

/// Handler: the step `charge` fails with `card declined` and no retry.
/// The handler then fails with `payment failed`.
///
/// The requirement asks for an error whose cause is the step error. The SDK
/// records an error's message as the messages of its whole `source()` chain,
/// joined with `": "`. So an error with the step error as its source records
/// `payment failed: ...: card declined`. The requirement expects the message
/// `payment failed`. So the handler's error has no source.
async fn handler(_event: Value, ctx: durable::DurableContext) -> Result<Value, durable::BoxError> {
    ctx.step(|_| async { Err::<(), durable::BoxError>("card declined".into()) })
        .name("charge")
        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
        .await
        .map_err(|_step_error| "payment failed")?;
    Ok(Value::Null)
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
