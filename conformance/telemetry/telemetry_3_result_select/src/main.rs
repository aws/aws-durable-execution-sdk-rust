//! Telemetry conformance requirement telemetry-3: a per-operation result
//! transform exports exactly one selected key of a step's JSON result.

use aws_durable_execution_sdk as durable;
use aws_durable_execution_sdk_telemetry_exporter::{
    CloudWatchLogsExporter, ContentConfig, ContentSetting, DurableExecutionTelemetry, EmitMode,
    OperationDetail, OperationOverride,
};
use serde_json::{Value, json};

/// Handler: the step `fetch-user` returns a three-key user object, and the
/// handler returns it unchanged.
async fn handler(event: Value, ctx: durable::DurableContext) -> Result<Value, durable::BoxError> {
    let user_id = event.get("userId").cloned().unwrap_or(Value::Null);
    ctx.step(move |_| async move {
        Ok(json!({
            "userId": user_id,
            "email": "test@example.com",
            "token": "tok-456",
        }))
    })
    .name("fetch-user")
    .await
    .map_err(Into::into)
}

/// Keeps only `userId` of the step result.
fn select_user_id(result: &Value) -> Value {
    json!({ "userId": result.get("userId").cloned() })
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    // The template sets the variable to the log group that the suite reads.
    let log_group = std::env::var("TELEMETRY_LOG_GROUP_NAME")?;
    let content = ContentConfig::new()
        .input(ContentSetting::Omit)
        .output(ContentSetting::Omit)
        .operation_override(
            OperationOverride::new("fetch-user").result(|result| Ok(select_user_id(&result))),
        );
    let telemetry = DurableExecutionTelemetry::builder()
        .exporter(CloudWatchLogsExporter::builder(log_group).build())
        .emit_mode(EmitMode::OnComplete)
        .operation_detail(OperationDetail::TopLevel)
        .sampling_rate(1.0)
        .content(content)
        .build();
    let options = durable::Options::builder().plugin(telemetry).build()?;
    durable::run_with_options(handler, options).await
}
