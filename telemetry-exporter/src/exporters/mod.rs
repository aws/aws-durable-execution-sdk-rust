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

/// The default record size limit of `LambdaLogExporter`,
/// `CloudWatchLogsExporter` and `EventBridgeExporter`.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
