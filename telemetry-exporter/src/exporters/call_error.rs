//! The export error of a failed AWS call.

use aws_durable_execution_sdk::BoxError;
use aws_smithy_types::error::display::DisplayErrorContext;

/// The export error for a failed call of `operation`.
///
/// The `Display` text of an `SdkError` is a fixed string for each variant,
/// such as "service error". The service's error code and message are in
/// the error's sources. The scheduler logs only the `Display` text of an
/// export error. So the text here includes the sources.
pub(super) fn call_error(operation: &str, error: &impl std::error::Error) -> BoxError {
    format!("{operation} failed: {}", DisplayErrorContext(error)).into()
}
