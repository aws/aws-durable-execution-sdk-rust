//! The exporters this crate ships.

mod cloudwatch_logs;
mod lambda_log;

pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
pub use lambda_log::LambdaLogExporter;

/// The `CloudWatch` Logs limit on one log event, which both exporters use as
/// their default record size limit.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
