//! The exporters this crate ships.

#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
#[cfg(feature = "http")]
mod http;
mod lambda_log;
#[cfg(feature = "opensearch")]
mod opensearch;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
#[cfg(feature = "http")]
pub use http::{HttpExporter, HttpExporterBuilder, HttpMethod};
pub use lambda_log::LambdaLogExporter;
#[cfg(feature = "opensearch")]
pub use opensearch::{OpenSearchExporter, OpenSearchExporterBuilder};

/// The default record size limit of `LambdaLogExporter` and
/// `CloudWatchLogsExporter`.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
