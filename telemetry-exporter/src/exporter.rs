//! The exporter interface.

use std::future::Future;
use std::pin::Pin;

use aws_durable_execution_sdk::BoxError;

use crate::record::TelemetryRecord;
use crate::render::OperationsFormat;

/// The future an exporter method returns.
pub type ExportFuture<'a> = Pin<Box<dyn Future<Output = Result<(), BoxError>> + Send + 'a>>;

/// A destination for telemetry records.
///
/// The plugin calls one exporter at a time per [`DurableExecutionTelemetry`]:
/// no `export` call overlaps another `export` or a `flush` of the same
/// plugin factory. Two factories that share one exporter can call it at
/// the same time.
///
/// An `Err` or a panic from `export` or `flush` is logged at WARN and
/// never changes the execution. A failing exporter does not stop the
/// other exporters.
///
/// [`DurableExecutionTelemetry`]: crate::DurableExecutionTelemetry
pub trait TelemetryExporter: Send + Sync + 'static {
    /// Writes one record.
    ///
    /// The record is already truncated to
    /// [`max_record_size_bytes`](Self::max_record_size_bytes), measured on
    /// the [`operations_format`](Self::operations_format) rendering. Render
    /// it with [`render_json`](crate::render_json) in the same format.
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a>;

    /// Writes any buffered records.
    ///
    /// The plugin calls it before each invocation returns, after the
    /// invocation's last record was exported. The default does nothing.
    fn flush(&self) -> ExportFuture<'_> {
        Box::pin(std::future::ready(Ok(())))
    }

    /// The largest rendered record, in bytes, the exporter accepts. `None`
    /// disables truncation. The default is `None`.
    fn max_record_size_bytes(&self) -> Option<usize> {
        None
    }

    /// The form in which the exporter writes a record's operations. The
    /// plugin measures the size limit on this form. The default is
    /// [`OperationsFormat::Array`].
    fn operations_format(&self) -> OperationsFormat {
        OperationsFormat::Array
    }
}
