//! Durable Execution Telemetry for the AWS Lambda Durable Functions Rust
//! SDK.
//!
//! The plugin builds one telemetry record per durable execution and hands
//! it to one or more exporters. A record is a complete snapshot: the
//! execution's identity, status, times, input, output, error and
//! operations. The schema matches the JavaScript, Python and Java
//! telemetry exporters, `recordType` `DurableExecutionTelemetry`,
//! `schemaVersion` `1.0`.
//!
//! Register [`DurableExecutionTelemetry`] as a plugin factory:
//!
//! ```no_run
//! use aws_durable_execution_sdk as durable;
//! use aws_durable_execution_sdk_telemetry_exporter::DurableExecutionTelemetry;
//!
//! async fn handler(event: u32, ctx: durable::DurableContext) -> Result<u32, durable::BoxError> {
//!     let next = ctx
//!         .step(move |_| async move { Ok(event + 1) })
//!         .name("increment")
//!         .await?;
//!     Ok(next)
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), durable::BoxError> {
//!     let options = durable::Options::builder()
//!         .plugin(DurableExecutionTelemetry::builder().build())
//!         .build()?;
//!     durable::run_with_options(handler, options).await
//! }
//! ```
//!
//! Without an exporter, the plugin writes each record to standard output
//! with the [`LambdaLogExporter`]. [`CloudWatchLogsExporter`] writes to a
//! log group of your choice. Implement [`TelemetryExporter`] for any other
//! destination.

// Compiles the README's Rust examples as doctests, so a README example that
// stops compiling fails `cargo test`.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

mod arn;
mod config;
mod exporter;
mod exporters;
mod plugin;
mod record;
mod render;
mod scheduler;
mod timestamp;
mod truncation;

pub use config::{
    ContentConfig, ContentSetting, EmitMode, OperationDetail, OperationOverride, Transform,
};
pub use exporter::{ExportFuture, TelemetryExporter};
pub use exporters::{
    CloudWatchLogsExporter, CloudWatchLogsExporterBuilder, DEFAULT_MAX_RECORD_SIZE_BYTES,
    LambdaLogExporter,
};
pub use plugin::{DurableExecutionTelemetry, DurableExecutionTelemetryBuilder};
pub use record::{
    ErrorRecord, ExecutionStatus, OperationRecord, RECORD_TYPE, SCHEMA_VERSION, TelemetryRecord,
};
pub use render::{
    OperationSummary, OperationsByName, OperationsFormat, build_operations_by_name, render_json,
    render_value,
};
pub use truncation::truncate_record;

/// The `tracing` target of every event this crate logs.
pub(crate) const TARGET: &str = "aws_durable_execution_sdk_telemetry_exporter";
