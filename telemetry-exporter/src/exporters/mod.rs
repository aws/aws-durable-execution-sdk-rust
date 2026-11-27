//! The exporters this crate ships.

#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
#[cfg(feature = "http")]
mod http;
mod lambda_log;
#[cfg(feature = "otel")]
mod otel;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
#[cfg(feature = "http")]
pub use http::{HttpExporter, HttpExporterBuilder, HttpMethod};
pub use lambda_log::LambdaLogExporter;
#[cfg(feature = "otel")]
pub use otel::{OtelExporter, OtelExporterBuilder, OtlpProtocol};

/// The default record size limit of both exporters.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;
