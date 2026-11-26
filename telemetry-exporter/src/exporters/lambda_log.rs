//! Writes records to standard output, which Lambda sends to the function's
//! log group.

use std::io::Write;
#[cfg(test)]
use std::sync::{Arc, Mutex};

use super::DEFAULT_MAX_RECORD_SIZE_BYTES;
use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};
#[cfg(test)]
use crate::scheduler::lock;

/// Where the exporter writes its lines.
#[derive(Debug, Clone, Default)]
enum Sink {
    #[default]
    Stdout,
    #[cfg(test)]
    Memory(Arc<Mutex<Vec<String>>>),
}

/// Writes each record as one JSON line to standard output.
///
/// Lambda sends standard output to the function's `CloudWatch` Logs group.
/// So this exporter needs no IAM permission and no configuration. It is
/// the plugin's default exporter. The line is the `operationsByName` form
/// of the record.
#[derive(Debug, Clone)]
pub struct LambdaLogExporter {
    max_record_size_bytes: usize,
    sink: Sink,
}

impl Default for LambdaLogExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl LambdaLogExporter {
    /// An exporter with the default size limit, 256,000 bytes.
    pub fn new() -> Self {
        Self {
            max_record_size_bytes: DEFAULT_MAX_RECORD_SIZE_BYTES,
            sink: Sink::Stdout,
        }
    }

    /// Sets the largest record, in bytes, before truncation.
    #[must_use]
    pub fn with_max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    #[cfg(test)]
    fn in_memory(lines: Arc<Mutex<Vec<String>>>) -> Self {
        Self {
            sink: Sink::Memory(lines),
            ..Self::new()
        }
    }

    fn write_line(&self, line: &str) -> std::io::Result<()> {
        match &self.sink {
            Sink::Stdout => {
                // The standard output lock is process-wide. It is held for
                // the write and the flush. So two lines never interleave.
                let mut out = std::io::stdout().lock();
                writeln!(out, "{line}")?;
                out.flush()
            }
            #[cfg(test)]
            Sink::Memory(lines) => {
                lock(lines).push(line.to_owned());
                Ok(())
            }
        }
    }
}

impl TelemetryExporter for LambdaLogExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        let written = render_json(record, OperationsFormat::ByName)
            .map_err(Into::into)
            .and_then(|line| self.write_line(&line).map_err(Into::into));
        Box::pin(std::future::ready(written))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }

    fn operations_format(&self) -> OperationsFormat {
        OperationsFormat::ByName
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "test assertions on known JSON keys"
)]
mod tests {
    use super::*;
    use crate::render::tests::{op, record};

    #[tokio::test]
    async fn writes_the_by_name_form_as_one_line() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let exporter = LambdaLogExporter::in_memory(Arc::clone(&lines));
        let result = exporter.export(&record(vec![op("a", "SUCCEEDED")])).await;
        assert!(result.is_ok());
        let lines = lock(&lines);
        assert_eq!(lines.len(), 1);
        let line = lines.first().cloned().unwrap_or_default();
        assert!(!line.contains('\n'));
        let value: serde_json::Value = serde_json::from_str(&line).unwrap_or_default();
        assert!(value.get("operations").is_none());
        assert_eq!(value["operationsByName"]["a"]["count"], 1);
    }

    #[tokio::test]
    async fn writes_to_stdout() {
        let exporter = LambdaLogExporter::default();
        assert!(exporter.export(&record(Vec::new())).await.is_ok());
        assert!(exporter.flush().await.is_ok());
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = LambdaLogExporter::new();
        assert_eq!(exporter.max_record_size_bytes(), Some(256_000));
        assert_eq!(exporter.operations_format(), OperationsFormat::ByName);
        let exporter = exporter.with_max_record_size_bytes(10);
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
    }
}
