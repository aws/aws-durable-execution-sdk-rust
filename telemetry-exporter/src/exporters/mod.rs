//! The exporters this crate ships.

#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
mod lambda_log;
#[cfg(feature = "redshift")]
mod redshift;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
pub use lambda_log::LambdaLogExporter;
#[cfg(feature = "redshift")]
pub use redshift::{RedshiftExporter, RedshiftExporterBuilder};

/// The `CloudWatch` Logs limit on one log event. `LambdaLogExporter` and
/// `CloudWatchLogsExporter` use it as their default record size limit.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
