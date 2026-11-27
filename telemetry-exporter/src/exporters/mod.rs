//! The exporters this crate ships.

#[cfg(feature = "aurora")]
mod aurora;
#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
mod lambda_log;

#[cfg(feature = "aurora")]
pub use aurora::{AuroraEngine, AuroraExporter, AuroraExporterBuilder};
#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
pub use lambda_log::LambdaLogExporter;

/// The `CloudWatch` Logs limit on one log event, which both exporters use as
/// their default record size limit.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
