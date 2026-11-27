//! Writes records to files in a directory.

use std::path::{Path, PathBuf};

use aws_durable_execution_sdk::BoxError;
use tokio::io::AsyncWriteExt;

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::{OperationRecord, TelemetryRecord};
use crate::render::{OperationsFormat, RecordView, render_json};

/// How a [`FileExporter`] lays out its files.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum FileMode {
    /// Appends each record as one line to the file of its UTC day,
    /// `{directory}/{YYYY-MM-DD}.ndjson`. This is the default.
    #[default]
    Ndjson,
    /// Writes each execution to its own file, `{directory}/{name}.json`,
    /// as indented JSON. A later record of the same execution replaces the
    /// file.
    Json,
}

/// Writes records to files in a directory.
///
/// The directory can be any writable path, such as an EFS mount, a mounted
/// bucket or `/tmp`. The exporter creates the directory, with its parents,
/// before the first write. It needs no IAM permission of its own.
///
/// In [`FileMode::Ndjson`] mode, the file name is the first 10 characters
/// of the record's `emittedAt`, which is the UTC date. In
/// [`FileMode::Json`] mode, the file name is the record's execution name,
/// or its execution ARN when it has no name. Each character outside
/// `a-z`, `A-Z`, `0-9`, `.`, `_` and `-` becomes `_`.
///
/// The exporter writes the `operations` array by default. It sets no size
/// limit by default.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_telemetry_exporter::{
///     DurableExecutionTelemetry, FileExporter, FileMode,
/// };
///
/// let telemetry = DurableExecutionTelemetry::builder()
///     .exporter(
///         FileExporter::builder("/mnt/efs/telemetry")
///             .mode(FileMode::Json)
///             .build(),
///     )
///     .build();
/// # drop(telemetry);
/// ```
#[derive(Debug)]
pub struct FileExporter {
    directory: PathBuf,
    mode: FileMode,
    operations_format: OperationsFormat,
    max_record_size_bytes: Option<usize>,
    directory_created: tokio::sync::OnceCell<()>,
}

/// Builds a [`FileExporter`].
#[derive(Debug)]
#[must_use]
pub struct FileExporterBuilder {
    directory: PathBuf,
    mode: FileMode,
    operations_format: OperationsFormat,
    max_record_size_bytes: Option<usize>,
}

impl FileExporter {
    /// A builder for an exporter that writes into `directory`.
    pub fn builder(directory: impl Into<PathBuf>) -> FileExporterBuilder {
        FileExporterBuilder {
            directory: directory.into(),
            mode: FileMode::default(),
            operations_format: OperationsFormat::Array,
            max_record_size_bytes: None,
        }
    }

    /// Creates the directory unless this exporter already created it.
    ///
    /// A failure leaves the directory uncreated. So the next record tries
    /// again.
    async fn ensure_directory(&self) -> Result<(), BoxError> {
        self.directory_created
            .get_or_try_init(|| async {
                tokio::fs::create_dir_all(&self.directory)
                    .await
                    .map_err(|error| {
                        format!(
                            "creating the directory {} failed: {error}",
                            self.directory.display()
                        )
                    })
            })
            .await?;
        Ok(())
    }

    /// Appends `text`, which ends with a newline, to the file at `path`.
    async fn append(path: &Path, text: &str) -> Result<(), BoxError> {
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(|error| io_error("opening", path, &error))?;
        // A tokio file splits a write at its buffer cap, 2 MiB by default.
        // Two writes of one line could interleave with another writer's
        // line. So the cap is raised to the length of the line, and the
        // line and its newline reach the file as one write.
        file.set_max_buf_size(text.len());
        // A tokio file hands its writes to a blocking thread, and reports a
        // failed write on a later call. Dropping the file does not wait for
        // the thread. So the flush is required, and its error is reported.
        async {
            file.write_all(text.as_bytes()).await?;
            file.flush().await
        }
        .await
        .map_err(|error| io_error("appending to", path, &error))?;
        Ok(())
    }

    /// Writes the record to its file.
    async fn write(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        self.ensure_directory().await?;
        match self.mode {
            FileMode::Ndjson => {
                let date = record.emitted_at.get(..10).ok_or_else(|| {
                    format!(
                        "emittedAt {:?} is shorter than a YYYY-MM-DD date",
                        record.emitted_at
                    )
                })?;
                let path = self.directory.join(format!("{date}.ndjson"));
                let mut line = render_json(record, self.operations_format)?;
                line.push('\n');
                Self::append(&path, &line).await
            }
            FileMode::Json => {
                let name = record
                    .execution_name
                    .as_deref()
                    .unwrap_or(&record.execution_arn);
                let path = self.directory.join(format!("{}.json", sanitize(name)));
                let operations: Vec<&OperationRecord> = record.operations.iter().collect();
                let text = serde_json::to_string_pretty(&RecordView {
                    record,
                    operations: &operations,
                    format: self.operations_format,
                })?;
                tokio::fs::write(&path, text)
                    .await
                    .map_err(|error| io_error("writing", &path, &error))?;
                Ok(())
            }
        }
    }
}

/// The export error for a failed file operation.
fn io_error(action: &str, path: &Path, error: &std::io::Error) -> BoxError {
    format!("{action} {} failed: {error}", path.display()).into()
}

/// Replaces each character outside `a-z`, `A-Z`, `0-9`, `.`, `_` and `-`
/// with `_`.
fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

impl FileExporterBuilder {
    /// Sets the file layout. The default is [`FileMode::Ndjson`].
    pub fn mode(mut self, mode: FileMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets how the written record lists its operations. The default is
    /// [`OperationsFormat::Array`].
    pub fn operations_format(mut self, format: OperationsFormat) -> Self {
        self.operations_format = format;
        self
    }

    /// Sets the largest record, in bytes, before truncation. Without it,
    /// the exporter does not truncate. In [`FileMode::Json`] mode, the
    /// limit applies to the compact form of the record, not to the longer
    /// indented text in the file.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = Some(max);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> FileExporter {
        FileExporter {
            directory: self.directory,
            mode: self.mode,
            operations_format: self.operations_format,
            max_record_size_bytes: self.max_record_size_bytes,
            directory_created: tokio::sync::OnceCell::new(),
        }
    }
}

impl TelemetryExporter for FileExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.write(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        self.max_record_size_bytes
    }

    fn operations_format(&self) -> OperationsFormat {
        self.operations_format
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "test assertions on known JSON keys"
)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use serde_json::{Value, json};

    use super::*;
    use crate::render::tests::{op, record};

    /// A directory under the system temporary directory, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            Self(std::env::temp_dir().join(format!(
                "durable-telemetry-file-test-{}-{nanos}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            )))
        }

        fn read(&self, name: &str) -> String {
            std::fs::read_to_string(self.0.join(name)).unwrap_or_default()
        }

        fn file_names(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|entry| entry.file_name().into_string().ok())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            drop(std::fs::remove_dir_all(&self.0));
        }
    }

    fn lines(text: &str) -> Vec<Value> {
        text.lines()
            .map(|line| serde_json::from_str(line).unwrap_or_default())
            .collect()
    }

    #[tokio::test]
    async fn ndjson_appends_one_line_per_record_to_the_file_of_the_day() {
        let dir = TempDir::new();
        let exporter = FileExporter::builder(&dir.0).build();
        let mut second = record(vec![op("b", "FAILED")]);
        second.execution_arn = "arn2".to_owned();
        assert!(
            exporter
                .export(&record(vec![op("a", "SUCCEEDED")]))
                .await
                .is_ok()
        );
        assert!(exporter.export(&second).await.is_ok());
        assert_eq!(dir.file_names(), ["2026-01-01.ndjson"]);
        let text = dir.read("2026-01-01.ndjson");
        assert!(text.ends_with('\n'));
        let records = lines(&text);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["executionArn"], "arn");
        assert_eq!(records[0]["operations"][0]["name"], "a");
        assert_eq!(records[1]["executionArn"], "arn2");
    }

    #[tokio::test]
    async fn ndjson_starts_a_file_for_each_date() {
        let dir = TempDir::new();
        let exporter = FileExporter::builder(&dir.0).build();
        let mut next_day = record(Vec::new());
        next_day.emitted_at = "2026-01-02T00:00:00.000Z".to_owned();
        assert!(exporter.export(&record(Vec::new())).await.is_ok());
        assert!(exporter.export(&next_day).await.is_ok());
        assert_eq!(dir.file_names(), ["2026-01-01.ndjson", "2026-01-02.ndjson"]);
    }

    #[tokio::test]
    async fn the_directory_is_created_with_its_parents() {
        let dir = TempDir::new();
        let nested = dir.0.join("a").join("b");
        let exporter = FileExporter::builder(&nested).build();
        assert!(exporter.export(&record(Vec::new())).await.is_ok());
        assert!(nested.join("2026-01-01.ndjson").is_file());
    }

    #[tokio::test]
    async fn an_existing_directory_keeps_its_files() {
        let dir = TempDir::new();
        assert!(std::fs::create_dir_all(&dir.0).is_ok());
        assert!(std::fs::write(dir.0.join("other.txt"), "x").is_ok());
        let exporter = FileExporter::builder(&dir.0).build();
        assert!(exporter.export(&record(Vec::new())).await.is_ok());
        assert_eq!(dir.read("other.txt"), "x");
    }

    #[tokio::test]
    async fn json_mode_writes_indented_json_and_replaces_the_file() {
        let dir = TempDir::new();
        let exporter = FileExporter::builder(&dir.0).mode(FileMode::Json).build();
        let mut running = record(Vec::new());
        running.execution_name = Some("my exec/1".to_owned());
        let mut done = running.clone();
        done.duration_ms = Some(7);
        assert!(exporter.export(&running).await.is_ok());
        assert!(exporter.export(&done).await.is_ok());
        assert_eq!(dir.file_names(), ["my_exec_1.json"]);
        let text = dir.read("my_exec_1.json");
        assert!(text.starts_with("{\n  \"recordType\": \"DurableExecutionTelemetry\",\n"));
        assert!(!text.ends_with('\n'));
        let value: Value = serde_json::from_str(&text).unwrap_or_default();
        assert_eq!(value["durationMs"], 7);
        assert_eq!(value["operations"], json!([]));
    }

    #[tokio::test]
    async fn json_mode_names_an_unnamed_execution_by_its_arn() {
        let dir = TempDir::new();
        let exporter = FileExporter::builder(&dir.0).mode(FileMode::Json).build();
        let mut unnamed = record(Vec::new());
        unnamed.execution_name = None;
        unnamed.execution_arn =
            "arn:aws:lambda:us-west-2:123:function:f:1/durable-execution/e/i".to_owned();
        assert!(exporter.export(&unnamed).await.is_ok());
        assert_eq!(
            dir.file_names(),
            ["arn_aws_lambda_us-west-2_123_function_f_1_durable-execution_e_i.json"]
        );
    }

    #[test]
    fn sanitize_replaces_each_character_outside_the_allowed_set() {
        assert_eq!(sanitize("aZ09._-"), "aZ09._-");
        assert_eq!(sanitize("a b/c:d"), "a_b_c_d");
        assert_eq!(sanitize("é😀"), "__");
        assert_eq!(sanitize("../x"), ".._x");
    }

    #[tokio::test]
    async fn the_operations_format_selects_the_written_form() {
        let dir = TempDir::new();
        let by_name = FileExporter::builder(&dir.0)
            .operations_format(OperationsFormat::ByName)
            .build();
        assert!(
            by_name
                .export(&record(vec![op("a", "SUCCEEDED")]))
                .await
                .is_ok()
        );
        let value = &lines(&dir.read("2026-01-01.ndjson"))[0];
        assert!(value.get("operations").is_none());
        assert_eq!(value["operationsByName"]["a"]["count"], 1);

        let dir = TempDir::new();
        let both = FileExporter::builder(&dir.0)
            .mode(FileMode::Json)
            .operations_format(OperationsFormat::Both)
            .build();
        assert!(
            both.export(&record(vec![op("a", "SUCCEEDED")]))
                .await
                .is_ok()
        );
        let value: Value = serde_json::from_str(&dir.read("exec.json")).unwrap_or_default();
        assert!(value.get("operations").is_some());
        assert_eq!(value["operationsByName"]["a"]["count"], 1);
    }

    #[tokio::test]
    async fn a_directory_that_cannot_be_created_fails_the_export_and_is_retried() {
        let dir = TempDir::new();
        assert!(std::fs::create_dir_all(&dir.0).is_ok());
        let blocker = dir.0.join("blocker");
        assert!(std::fs::write(&blocker, "x").is_ok());
        let exporter = FileExporter::builder(blocker.join("sub")).build();
        let error = exporter
            .export(&record(Vec::new()))
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.starts_with("creating the directory "), "{error}");
        assert!(error.contains("blocker"), "{error}");
        assert!(std::fs::remove_file(&blocker).is_ok());
        assert!(exporter.export(&record(Vec::new())).await.is_ok());
    }

    #[tokio::test]
    async fn a_file_that_cannot_be_opened_names_the_path() {
        let dir = TempDir::new();
        assert!(std::fs::create_dir_all(dir.0.join("2026-01-01.ndjson")).is_ok());
        let exporter = FileExporter::builder(&dir.0).build();
        let error = exporter
            .export(&record(Vec::new()))
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.starts_with("opening "), "{error}");
        assert!(error.contains("2026-01-01.ndjson"), "{error}");
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_file_that_cannot_be_appended_to_names_the_path() {
        let dir = TempDir::new();
        assert!(std::fs::create_dir_all(&dir.0).is_ok());
        // Every write to /dev/full fails with "no space left on device".
        let link = dir.0.join("2026-01-01.ndjson");
        assert!(std::os::unix::fs::symlink("/dev/full", link).is_ok());
        let exporter = FileExporter::builder(&dir.0).build();
        let error = exporter
            .export(&record(Vec::new()))
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.starts_with("appending to "), "{error}");
        assert!(error.contains("2026-01-01.ndjson"), "{error}");
    }

    #[tokio::test]
    async fn a_line_over_the_default_buffer_cap_is_appended_whole() {
        let dir = TempDir::new();
        let exporter = FileExporter::builder(&dir.0).build();
        let mut big = record(Vec::new());
        big.output = Some(json!("x".repeat(3 * 1024 * 1024)));
        assert!(exporter.export(&big).await.is_ok());
        let text = dir.read("2026-01-01.ndjson");
        assert!(text.len() > 3 * 1024 * 1024);
        assert_eq!(lines(&text).len(), 1);
    }

    #[tokio::test]
    async fn a_file_that_cannot_be_written_names_the_path() {
        let dir = TempDir::new();
        assert!(std::fs::create_dir_all(dir.0.join("exec.json")).is_ok());
        let exporter = FileExporter::builder(&dir.0).mode(FileMode::Json).build();
        let error = exporter
            .export(&record(Vec::new()))
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.starts_with("writing "), "{error}");
        assert!(error.contains("exec.json"), "{error}");
    }

    #[tokio::test]
    async fn an_emitted_at_without_a_date_fails_the_export() {
        let dir = TempDir::new();
        let exporter = FileExporter::builder(&dir.0).build();
        let mut short = record(Vec::new());
        short.emitted_at = "2026".to_owned();
        let error = exporter
            .export(&short)
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("shorter than a YYYY-MM-DD date"), "{error}");
        assert_eq!(dir.file_names(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn flush_succeeds() {
        let exporter = FileExporter::builder(std::env::temp_dir()).build();
        assert!(exporter.flush().await.is_ok());
    }

    #[test]
    fn the_defaults_are_no_limit_and_the_array_form() {
        let exporter = FileExporter::builder("dir").build();
        assert_eq!(exporter.max_record_size_bytes(), None);
        assert_eq!(exporter.operations_format(), OperationsFormat::Array);
        assert_eq!(exporter.mode, FileMode::Ndjson);
    }

    #[test]
    fn the_builder_sets_the_limit_and_the_format() {
        let exporter = FileExporter::builder("dir")
            .max_record_size_bytes(10)
            .operations_format(OperationsFormat::ByName)
            .build();
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::ByName);
    }
}
