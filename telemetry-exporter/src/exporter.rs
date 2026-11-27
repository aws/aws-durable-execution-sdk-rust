//! The exporter interface.

use std::future::Future;
use std::pin::Pin;

use aws_durable_execution_sdk::BoxError;

use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};

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
    /// the text of [`measured_text`](Self::measured_text).
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
    /// default [`measured_text`](Self::measured_text) renders this form.
    /// The default is [`OperationsFormat::Array`].
    fn operations_format(&self) -> OperationsFormat {
        OperationsFormat::Array
    }

    /// The text on which the plugin measures the size limit.
    ///
    /// The plugin calls it for the whole record and again for each
    /// shrunk candidate. The byte length of the text is compared with
    /// [`max_record_size_bytes`](Self::max_record_size_bytes). An exporter
    /// whose request is larger than the record's JSON overrides it to
    /// return the text of the whole request. The default is the record's
    /// JSON in the [`operations_format`](Self::operations_format), from
    /// [`render_json`](crate::render_json).
    ///
    /// An `Err` means the record cannot be measured. The plugin then
    /// exports the record unchanged. A panic skips the exporter for that
    /// record, as a panic in
    /// [`max_record_size_bytes`](Self::max_record_size_bytes) does.
    ///
    /// # Errors
    ///
    /// The default returns the serializer's error.
    fn measured_text(&self, record: &TelemetryRecord) -> Result<String, BoxError> {
        Ok(render_json(record, self.operations_format())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tests::{op, record};

    struct Minimal(OperationsFormat);

    impl TelemetryExporter for Minimal {
        fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn operations_format(&self) -> OperationsFormat {
            self.0
        }
    }

    #[test]
    fn the_default_measured_text_is_the_json_in_the_operations_format() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        for format in [
            OperationsFormat::Array,
            OperationsFormat::ByName,
            OperationsFormat::Both,
        ] {
            assert_eq!(
                Minimal(format).measured_text(&r).ok(),
                render_json(&r, format).ok(),
                "{format:?}"
            );
        }
    }
}
