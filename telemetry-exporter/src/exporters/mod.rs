//! The exporters this crate ships.

#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
mod lambda_log;
#[cfg(feature = "sqs")]
mod sqs;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
pub use lambda_log::LambdaLogExporter;
#[cfg(feature = "sqs")]
pub use sqs::{SqsExporter, SqsExporterBuilder};

/// The default record size limit of both exporters. It is below the
/// `CloudWatch` Logs limit of 1 MB on one log event.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
