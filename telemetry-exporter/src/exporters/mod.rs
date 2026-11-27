//! The exporters this crate ships.

#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
#[cfg(feature = "eventbridge")]
mod eventbridge;
mod lambda_log;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
#[cfg(feature = "eventbridge")]
pub use eventbridge::{EventBridgeExporter, EventBridgeExporterBuilder};
pub use lambda_log::LambdaLogExporter;

/// The `CloudWatch` Logs limit on one log event. The `CloudWatch` Logs and
/// `EventBridge` exporters use it as their default record size limit.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
