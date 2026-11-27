//! The exporters this crate ships.

#[cfg(any(feature = "cloudwatch-logs", feature = "dynamodb"))]
mod call_error;
#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
#[cfg(feature = "dynamodb")]
mod dynamodb;
mod lambda_log;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
#[cfg(feature = "dynamodb")]
pub use dynamodb::{DynamoDbExporter, DynamoDbExporterBuilder};
pub use lambda_log::LambdaLogExporter;

/// The `CloudWatch` Logs limit on one log event, which `LambdaLogExporter` and
/// `CloudWatchLogsExporter` use as their default record size limit.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
