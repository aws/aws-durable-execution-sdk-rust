//! The exporters this crate ships.

#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
#[cfg(feature = "firehose")]
mod firehose;
mod lambda_log;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
#[cfg(feature = "firehose")]
pub use firehose::{FirehoseExporter, FirehoseExporterBuilder};
pub use lambda_log::LambdaLogExporter;

/// The `CloudWatch` Logs limit on one log event. [`LambdaLogExporter`] and
/// `CloudWatchLogsExporter` use it as their default record size limit.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
