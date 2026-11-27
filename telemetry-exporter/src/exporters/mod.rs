//! The exporters this crate ships.

#[cfg(feature = "cloudwatch-logs")]
mod cloudwatch_logs;
mod lambda_log;
#[cfg(feature = "s3")]
mod s3;

#[cfg(feature = "cloudwatch-logs")]
pub use cloudwatch_logs::{CloudWatchLogsExporter, CloudWatchLogsExporterBuilder};
pub use lambda_log::LambdaLogExporter;
#[cfg(feature = "s3")]
pub use s3::{S3Exporter, S3ExporterBuilder, S3Partitioning};

/// The `CloudWatch` Logs limit on one log event, which
/// [`LambdaLogExporter`] and `CloudWatchLogsExporter`
/// use as their default record size limit.
pub const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 256_000;

/// The export error for a failed call of `operation`.
///
/// The `Display` text of an AWS SDK `SdkError` is a fixed string for each
/// variant, such as "service error". The service's error code and message
/// are in the error's sources. The scheduler logs only the `Display` text
/// of an export error. So the caller passes the error wrapped in the SDK's
/// `DisplayErrorContext`, and the text here includes the sources.
#[cfg(any(feature = "cloudwatch-logs", feature = "s3"))]
fn call_error(
    operation: &str,
    error: impl std::fmt::Display,
) -> aws_durable_execution_sdk::BoxError {
    format!("{operation} failed: {error}").into()
}
