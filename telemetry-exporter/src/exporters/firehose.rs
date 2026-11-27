//! Sends records to an Amazon Data Firehose delivery stream with
//! `PutRecord`.

use aws_durable_execution_sdk::BoxError;
use aws_sdk_firehose::Client;
use aws_sdk_firehose::error::{DisplayErrorContext, SdkError};
use aws_sdk_firehose::primitives::Blob;
use aws_sdk_firehose::types::Record;

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};

/// The default record size limit: the Firehose limit of 1,000 KiB for one
/// record, rounded down to 1,000,000 bytes.
const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 1_000_000;

/// Sends each record to a Firehose delivery stream.
///
/// Each record is one `PutRecord` call. The data is the record's JSON
/// followed by one newline. Firehose concatenates records into the
/// destination's objects, so the newline keeps the output parseable as
/// newline-delimited JSON. Firehose does the buffering, batching and
/// delivery. So [`flush`](TelemetryExporter::flush) does nothing.
///
/// The operations render as the `operations` array unless
/// [`operations_format`](FirehoseExporterBuilder::operations_format) says
/// otherwise.
///
/// The function's role needs `firehose:PutRecord` on the delivery stream.
#[derive(Debug)]
pub struct FirehoseExporter {
    delivery_stream_name: String,
    region: Option<String>,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
}

/// Builds a [`FirehoseExporter`].
#[derive(Debug)]
#[must_use]
pub struct FirehoseExporterBuilder {
    delivery_stream_name: String,
    region: Option<String>,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

impl FirehoseExporter {
    /// A builder for an exporter that sends records to the delivery stream
    /// `delivery_stream_name`.
    pub fn builder(delivery_stream_name: impl Into<String>) -> FirehoseExporterBuilder {
        FirehoseExporterBuilder {
            delivery_stream_name: delivery_stream_name.into(),
            region: None,
            operations_format: OperationsFormat::Array,
            max_record_size_bytes: DEFAULT_MAX_RECORD_SIZE_BYTES,
            client: None,
        }
    }

    /// The client: the configured one, or one built from the default
    /// credential and region chain on first use.
    async fn client(&self) -> &Client {
        self.client
            .get_or_init(|| async {
                let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
                if let Some(region) = &self.region {
                    loader = loader.region(aws_config::Region::new(region.clone()));
                }
                Client::new(&loader.load().await)
            })
            .await
    }

    /// Sends `record` with one `PutRecord` call.
    async fn put(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        // The scheduler truncated the record to the size of the
        // `operations_format` rendering. So the same format is rendered here.
        let mut data = render_json(record, self.operations_format)?;
        data.push('\n');
        let client = self.client().await;
        client
            .put_record()
            .delivery_stream_name(&self.delivery_stream_name)
            .record(Record::builder().data(Blob::new(data)).build()?)
            .send()
            .await
            .map_err(|error| call_error("PutRecord", &error))?;
        Ok(())
    }
}

/// The export error for a failed call of `operation`.
///
/// The `Display` text of an `SdkError` is a fixed string for each variant,
/// such as "service error". The service's error code and message are in
/// the error's sources. The scheduler logs only the `Display` text of an
/// export error. So the text here includes the sources.
fn call_error<E>(operation: &str, error: &SdkError<E>) -> BoxError
where
    E: std::error::Error + 'static,
{
    format!("{operation} failed: {}", DisplayErrorContext(error)).into()
}

impl FirehoseExporterBuilder {
    /// Sets the region of the client the exporter builds. Without it, the
    /// default region chain applies. A [`client`](Self::client) ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets how a record lists its operations. The default is
    /// [`OperationsFormat::Array`], the `operations` array. Choose the form
    /// that the delivery stream's consumer reads.
    pub fn operations_format(mut self, format: OperationsFormat) -> Self {
        self.operations_format = format;
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default is
    /// 1,000,000, below the Firehose limit of 1,000 KiB for one record. The
    /// limit applies to the record's JSON. The newline that the exporter
    /// appends adds one byte.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Uses `client` instead of building one from the default chain.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`FirehoseClient`](crate::FirehoseClient), so callers can name it
    /// without depending on `aws-sdk-firehose` directly.
    ///
    /// # Examples
    ///
    /// The SDK crate re-exports the configuration type. So this compiles
    /// with no direct `aws-config` or `aws-sdk-firehose` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::SdkConfig;
    /// use aws_durable_execution_sdk_telemetry_exporter::{FirehoseClient, FirehoseExporter};
    ///
    /// let config = SdkConfig::builder().build();
    /// let exporter = FirehoseExporter::builder("my-telemetry-stream")
    ///     .client(FirehoseClient::new(&config))
    ///     .build();
    /// # drop(exporter);
    /// ```
    pub fn client(mut self, client: crate::FirehoseClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> FirehoseExporter {
        FirehoseExporter {
            delivery_stream_name: self.delivery_stream_name,
            region: self.region,
            operations_format: self.operations_format,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
        }
    }
}

impl TelemetryExporter for FirehoseExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.put(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }

    fn operations_format(&self) -> OperationsFormat {
        self.operations_format
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tests::{op, record};
    use aws_sdk_firehose::operation::put_record::{PutRecordError, PutRecordOutput};
    use aws_sdk_firehose::types::error::ResourceNotFoundException;
    use aws_smithy_mocks::{RuleMode, mock, mock_client};

    fn output() -> PutRecordOutput {
        PutRecordOutput::builder()
            .record_id("id")
            .build()
            .unwrap_or_else(|_| unreachable!("record_id is set"))
    }

    fn data_of(req: &aws_sdk_firehose::operation::put_record::PutRecordInput) -> String {
        req.record()
            .map(|r| String::from_utf8_lossy(r.data().as_ref()).into_owned())
            .unwrap_or_default()
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = FirehoseExporter::builder("s")
            .region("us-west-2")
            .max_record_size_bytes(10)
            .operations_format(OperationsFormat::Both)
            .build();
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::Both);
        assert_eq!(exporter.region.as_deref(), Some("us-west-2"));
    }

    #[test]
    fn the_defaults_are_the_array_form_and_one_million_bytes() {
        let exporter = FirehoseExporter::builder("s").build();
        assert_eq!(exporter.max_record_size_bytes(), Some(1_000_000));
        assert_eq!(exporter.operations_format(), OperationsFormat::Array);
        assert_eq!(exporter.region, None);
    }

    #[tokio::test]
    async fn sends_the_record_json_and_one_newline_to_the_stream() {
        let put = mock!(Client::put_record)
            .match_requests(|req| {
                let data = data_of(req);
                req.delivery_stream_name() == Some("stream")
                    && data.starts_with('{')
                    && data.contains("\"operations\":[{")
                    && data.ends_with("}\n")
                    && !data.ends_with("\n\n")
                    && !data.contains("operationsByName")
            })
            .then_output(output);
        let client = mock_client!(aws_sdk_firehose, RuleMode::MatchAny, [&put]);
        let exporter = FirehoseExporter::builder("stream").client(client).build();

        assert!(
            exporter
                .export(&record(vec![op("a", "SUCCEEDED")]))
                .await
                .is_ok()
        );
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn each_record_is_one_call() {
        let put = mock!(Client::put_record).then_output(output);
        let client = mock_client!(aws_sdk_firehose, RuleMode::MatchAny, [&put]);
        let exporter = FirehoseExporter::builder("stream").client(client).build();

        let r = record(Vec::new());
        assert!(exporter.export(&r).await.is_ok());
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(put.num_calls(), 2);
    }

    #[tokio::test]
    async fn renders_the_configured_operations_format() {
        let put = mock!(Client::put_record)
            .match_requests(|req| {
                let data = data_of(req);
                data.contains("\"operationsByName\":{\"a\"") && !data.contains("\"operations\":[")
            })
            .then_output(output);
        let client = mock_client!(aws_sdk_firehose, RuleMode::MatchAny, [&put]);
        let exporter = FirehoseExporter::builder("stream")
            .operations_format(OperationsFormat::ByName)
            .client(client)
            .build();

        assert!(
            exporter
                .export(&record(vec![op("a", "SUCCEEDED")]))
                .await
                .is_ok()
        );
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_service_error_fails_the_export_and_names_the_call() {
        let put = mock!(Client::put_record).then_error(|| {
            PutRecordError::ResourceNotFoundException(ResourceNotFoundException::builder().build())
        });
        let client = mock_client!(aws_sdk_firehose, RuleMode::MatchAny, [&put]);
        let exporter = FirehoseExporter::builder("stream").client(client).build();

        let error = exporter
            .export(&record(Vec::new()))
            .await
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.starts_with("PutRecord failed: "), "{error}");
        assert!(error.contains("ResourceNotFoundException"), "{error}");
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn flush_sends_nothing() {
        let put = mock!(Client::put_record).then_output(output);
        let client = mock_client!(aws_sdk_firehose, RuleMode::MatchAny, [&put]);
        let exporter = FirehoseExporter::builder("stream").client(client).build();

        assert!(exporter.flush().await.is_ok());
        assert_eq!(put.num_calls(), 0);
    }

    #[tokio::test]
    async fn builds_a_client_from_the_default_chain() {
        let exporter = FirehoseExporter::builder("stream")
            .region("eu-west-1")
            .build();
        let client = exporter.client().await;
        assert_eq!(
            client.config().region().map(ToString::to_string).as_deref(),
            Some("eu-west-1")
        );
    }
}
