//! Telemetry conformance requirement telemetry-4: in full-tree mode the
//! record lists a map's iterations and the steps inside them.

use aws_durable_execution_sdk as durable;
use aws_durable_execution_sdk_telemetry_exporter::{
    CloudWatchLogsExporter, DurableExecutionTelemetry, EmitMode, OperationDetail,
};

/// Handler: the map `process-items` runs one item at a time. The item
/// namer names each iteration `item-<item>`. Each iteration runs the step
/// `greet-<item>`, which returns `Hello, <item>!`.
async fn handler(
    items: Vec<i64>,
    ctx: durable::DurableContext,
) -> Result<Vec<String>, durable::BoxError> {
    let results = ctx
        .map(items, |child, item, _index| async move {
            child
                .step(move |_| async move { Ok(format!("Hello, {item}!")) })
                .name(format!("greet-{item}"))
                .await
                .map_err(Into::into)
        })
        .name("process-items")
        .max_concurrency(1)
        .item_namer(|item, _index| format!("item-{item}"))
        .await?;
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    // The template sets the variable to the log group that the suite reads.
    let log_group = std::env::var("TELEMETRY_LOG_GROUP_NAME")?;
    let telemetry = DurableExecutionTelemetry::builder()
        .exporter(CloudWatchLogsExporter::builder(log_group).build())
        .emit_mode(EmitMode::OnComplete)
        .operation_detail(OperationDetail::FullTree)
        .sampling_rate(1.0)
        .build();
    let options = durable::Options::builder().plugin(telemetry).build()?;
    durable::run_with_options(handler, options).await
}
