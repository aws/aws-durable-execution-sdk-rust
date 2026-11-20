//! Writes records to a `CloudWatch` Logs group with `PutLogEvents`.

use std::collections::HashSet;
use std::sync::Mutex;

use aws_durable_execution_sdk::BoxError;
use aws_sdk_cloudwatchlogs::Client;
use aws_sdk_cloudwatchlogs::error::SdkError;
use aws_sdk_cloudwatchlogs::operation::create_log_stream::CreateLogStreamError;
use aws_sdk_cloudwatchlogs::operation::put_log_events::PutLogEventsError;
use aws_sdk_cloudwatchlogs::types::InputLogEvent;

use super::DEFAULT_MAX_RECORD_SIZE_BYTES;
use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};
use crate::scheduler::lock;
use crate::timestamp::{Civil, now_ms};

/// The default log stream prefix.
const DEFAULT_LOG_STREAM_PREFIX: &str = "durable-execution-telemetry/";

/// Writes each record as one log event to a `CloudWatch` Logs group.
///
/// Unlike [`LambdaLogExporter`](crate::LambdaLogExporter), this exporter
/// writes to any log group. Each UTC day gets its own log stream, named
/// `{prefix}YYYY/MM/DD`, which bounds a stream's size. The event message
/// is the `operationsByName` form of the record.
///
/// The function's role needs `logs:CreateLogStream` and
/// `logs:PutLogEvents` on the log group. The exporter creates each day's
/// stream on its first record and treats an existing stream as created.
/// If a write fails because the stream no longer exists, the exporter
/// creates the stream again and resends the record once.
#[derive(Debug)]
pub struct CloudWatchLogsExporter {
    log_group_name: String,
    log_stream_prefix: String,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
    created_streams: Mutex<HashSet<String>>,
}

/// Builds a [`CloudWatchLogsExporter`].
#[derive(Debug)]
#[must_use]
pub struct CloudWatchLogsExporterBuilder {
    log_group_name: String,
    log_stream_prefix: String,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

impl CloudWatchLogsExporter {
    /// A builder for an exporter that writes to `log_group_name`.
    pub fn builder(log_group_name: impl Into<String>) -> CloudWatchLogsExporterBuilder {
        CloudWatchLogsExporterBuilder {
            log_group_name: log_group_name.into(),
            log_stream_prefix: DEFAULT_LOG_STREAM_PREFIX.to_owned(),
            region: None,
            max_record_size_bytes: DEFAULT_MAX_RECORD_SIZE_BYTES,
            client: None,
        }
    }

    /// The log stream for a time in epoch milliseconds.
    fn stream_name(&self, ms: i64) -> String {
        let day = Civil::from_epoch_ms(ms);
        format!(
            "{}{:04}/{:02}/{:02}",
            self.log_stream_prefix, day.year, day.month, day.day
        )
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

    /// Creates the stream unless this exporter already created it.
    async fn ensure_stream(&self, client: &Client, stream: &str) -> Result<(), BoxError> {
        if lock(&self.created_streams).contains(stream) {
            return Ok(());
        }
        let created = client
            .create_log_stream()
            .log_group_name(&self.log_group_name)
            .log_stream_name(stream)
            .send()
            .await;
        if let Err(error) = created {
            let exists = error
                .as_service_error()
                .is_some_and(CreateLogStreamError::is_resource_already_exists_exception);
            if !exists {
                return Err(error.into());
            }
        }
        lock(&self.created_streams).insert(stream.to_owned());
        Ok(())
    }

    async fn put(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        let message = render_json(record, OperationsFormat::ByName)?;
        let now = now_ms();
        let stream = self.stream_name(now);
        let client = self.client().await;
        self.ensure_stream(client, &stream).await?;
        let event = InputLogEvent::builder()
            .timestamp(now)
            .message(message)
            .build()?;
        match self.put_event(client, &stream, event.clone()).await {
            Ok(()) => Ok(()),
            Err(error) if is_stream_missing(&error) => {
                // `created_streams` records that a stream existed at its first
                // write. The stream can be deleted later. Without this removal,
                // `ensure_stream` would skip CreateLogStream, and every later
                // write to the stream would fail. So the name is removed, the
                // stream is created again, and the event is resent once.
                lock(&self.created_streams).remove(&stream);
                self.ensure_stream(client, &stream).await?;
                self.put_event(client, &stream, event).await?;
                Ok(())
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Sends `event` to `stream` with one `PutLogEvents` request.
    ///
    /// The error is boxed because `SdkError<PutLogEventsError>` is several
    /// hundred bytes, which clippy's `result_large_err` rejects.
    async fn put_event(
        &self,
        client: &Client,
        stream: &str,
        event: InputLogEvent,
    ) -> Result<(), Box<SdkError<PutLogEventsError>>> {
        client
            .put_log_events()
            .log_group_name(&self.log_group_name)
            .log_stream_name(stream)
            .log_events(event)
            .send()
            .await
            .map_err(Box::new)?;
        Ok(())
    }
}

/// Whether `PutLogEvents` failed because the log stream does not exist.
fn is_stream_missing(error: &SdkError<PutLogEventsError>) -> bool {
    error
        .as_service_error()
        .is_some_and(PutLogEventsError::is_resource_not_found_exception)
}

impl CloudWatchLogsExporterBuilder {
    /// Sets the log stream prefix. The default is
    /// `durable-execution-telemetry/`.
    pub fn log_stream_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.log_stream_prefix = prefix.into();
        self
    }

    /// Sets the region of the client the exporter builds. Without it, the
    /// default region chain applies. A [`client`](Self::client) ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default
    /// is 256,000, the `CloudWatch` Logs event limit. The service also counts
    /// 26 bytes per event, so a limit close to the maximum can still be
    /// rejected.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Uses `client` instead of building one from the default chain.
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> CloudWatchLogsExporter {
        CloudWatchLogsExporter {
            log_group_name: self.log_group_name,
            log_stream_prefix: self.log_stream_prefix,
            region: self.region,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
            created_streams: Mutex::new(HashSet::new()),
        }
    }
}

impl TelemetryExporter for CloudWatchLogsExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.put(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }

    fn operations_format(&self) -> OperationsFormat {
        OperationsFormat::ByName
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tests::{op, record};
    use aws_sdk_cloudwatchlogs::operation::create_log_stream::{
        CreateLogStreamError, CreateLogStreamOutput,
    };
    use aws_sdk_cloudwatchlogs::operation::put_log_events::{
        PutLogEventsError, PutLogEventsOutput,
    };
    use aws_sdk_cloudwatchlogs::types::error::{
        InvalidParameterException, ResourceAlreadyExistsException, ResourceNotFoundException,
    };
    use aws_smithy_mocks::{RuleMode, mock, mock_client};

    fn today() -> String {
        let day = Civil::from_epoch_ms(now_ms());
        format!(
            "durable-execution-telemetry/{:04}/{:02}/{:02}",
            day.year, day.month, day.day
        )
    }

    #[test]
    fn stream_names_are_per_utc_day() {
        let exporter = CloudWatchLogsExporter::builder("g")
            .log_stream_prefix("p/")
            .build();
        // 2024-02-29T23:59:59.999Z
        assert_eq!(exporter.stream_name(1_709_251_199_999), "p/2024/02/29");
        assert_eq!(exporter.stream_name(1_709_251_200_000), "p/2024/03/01");
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = CloudWatchLogsExporter::builder("g")
            .region("us-west-2")
            .max_record_size_bytes(10)
            .build();
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::ByName);
        assert_eq!(exporter.region.as_deref(), Some("us-west-2"));
        assert_eq!(
            CloudWatchLogsExporter::builder("g")
                .build()
                .max_record_size_bytes(),
            Some(256_000)
        );
    }

    #[tokio::test]
    async fn creates_the_stream_once_and_puts_each_record() {
        let stream = today();
        let expected = stream.clone();
        let create = mock!(Client::create_log_stream)
            .match_requests(move |req| {
                req.log_group_name() == Some("group") && req.log_stream_name() == Some(&expected)
            })
            .then_output(|| CreateLogStreamOutput::builder().build());
        let expected = stream.clone();
        let put = mock!(Client::put_log_events)
            .match_requests(move |req| {
                let message = req.log_events().first().map(InputLogEvent::message);
                req.log_group_name() == Some("group")
                    && req.log_stream_name() == Some(&expected)
                    && req.log_events().len() == 1
                    && message.is_some_and(|m| m.contains("\"operationsByName\":{\"a\""))
            })
            .then_output(|| PutLogEventsOutput::builder().build());
        let client = mock_client!(aws_sdk_cloudwatchlogs, RuleMode::MatchAny, [&create, &put]);
        let exporter = CloudWatchLogsExporter::builder("group")
            .client(client)
            .build();

        let r = record(vec![op("a", "SUCCEEDED")]);
        assert!(exporter.export(&r).await.is_ok());
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(create.num_calls(), 1);
        assert_eq!(put.num_calls(), 2);
    }

    #[tokio::test]
    async fn an_existing_stream_counts_as_created() {
        let create = mock!(Client::create_log_stream).then_error(|| {
            CreateLogStreamError::ResourceAlreadyExistsException(
                ResourceAlreadyExistsException::builder().build(),
            )
        });
        let put =
            mock!(Client::put_log_events).then_output(|| PutLogEventsOutput::builder().build());
        let client = mock_client!(aws_sdk_cloudwatchlogs, RuleMode::MatchAny, [&create, &put]);
        let exporter = CloudWatchLogsExporter::builder("group")
            .client(client)
            .build();

        let r = record(Vec::new());
        assert!(exporter.export(&r).await.is_ok());
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(create.num_calls(), 1);
        assert_eq!(put.num_calls(), 2);
    }

    #[tokio::test]
    async fn another_create_error_fails_the_export_and_is_retried_next_time() {
        let create = mock!(Client::create_log_stream).then_error(|| {
            CreateLogStreamError::InvalidParameterException(
                InvalidParameterException::builder().build(),
            )
        });
        let put =
            mock!(Client::put_log_events).then_output(|| PutLogEventsOutput::builder().build());
        let client = mock_client!(aws_sdk_cloudwatchlogs, RuleMode::MatchAny, [&create, &put]);
        let exporter = CloudWatchLogsExporter::builder("group")
            .client(client)
            .build();

        let r = record(Vec::new());
        assert!(exporter.export(&r).await.is_err());
        assert!(exporter.export(&r).await.is_err());
        assert_eq!(create.num_calls(), 2);
        assert_eq!(put.num_calls(), 0);
    }

    fn stream_not_found() -> PutLogEventsError {
        PutLogEventsError::ResourceNotFoundException(ResourceNotFoundException::builder().build())
    }

    #[tokio::test]
    async fn a_deleted_stream_is_created_again_and_the_record_is_resent() {
        let create = mock!(Client::create_log_stream)
            .then_output(|| CreateLogStreamOutput::builder().build());
        let put = mock!(Client::put_log_events)
            .sequence()
            .error(stream_not_found)
            .output(|| PutLogEventsOutput::builder().build())
            .times(2)
            .build();
        let client = mock_client!(aws_sdk_cloudwatchlogs, RuleMode::MatchAny, [&create, &put]);
        let exporter = CloudWatchLogsExporter::builder("group")
            .client(client)
            .build();

        let r = record(Vec::new());
        assert!(exporter.export(&r).await.is_ok());
        // The stream is cached again, so the next record makes no create call.
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(create.num_calls(), 2);
        assert_eq!(put.num_calls(), 3);
    }

    #[tokio::test]
    async fn a_deleted_log_group_fails_the_export_and_is_retried_next_time() {
        let create = mock!(Client::create_log_stream)
            .sequence()
            .output(|| CreateLogStreamOutput::builder().build())
            .error(|| {
                CreateLogStreamError::ResourceNotFoundException(
                    ResourceNotFoundException::builder().build(),
                )
            })
            .output(|| CreateLogStreamOutput::builder().build())
            .build();
        let put = mock!(Client::put_log_events)
            .sequence()
            .error(stream_not_found)
            .output(|| PutLogEventsOutput::builder().build())
            .build();
        let client = mock_client!(aws_sdk_cloudwatchlogs, RuleMode::MatchAny, [&create, &put]);
        let exporter = CloudWatchLogsExporter::builder("group")
            .client(client)
            .build();

        let r = record(Vec::new());
        assert!(exporter.export(&r).await.is_err());
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(create.num_calls(), 3);
        assert_eq!(put.num_calls(), 2);
    }

    #[tokio::test]
    async fn a_stream_that_stays_missing_is_resent_only_once() {
        let create = mock!(Client::create_log_stream)
            .then_output(|| CreateLogStreamOutput::builder().build());
        let put = mock!(Client::put_log_events).then_error(stream_not_found);
        let client = mock_client!(aws_sdk_cloudwatchlogs, RuleMode::MatchAny, [&create, &put]);
        let exporter = CloudWatchLogsExporter::builder("group")
            .client(client)
            .build();

        assert!(exporter.export(&record(Vec::new())).await.is_err());
        assert_eq!(create.num_calls(), 2);
        assert_eq!(put.num_calls(), 2);
    }

    #[tokio::test]
    async fn another_put_error_fails_the_export_without_a_resend() {
        let create = mock!(Client::create_log_stream)
            .then_output(|| CreateLogStreamOutput::builder().build());
        let put = mock!(Client::put_log_events).then_error(|| {
            PutLogEventsError::InvalidParameterException(
                InvalidParameterException::builder().build(),
            )
        });
        let client = mock_client!(aws_sdk_cloudwatchlogs, RuleMode::MatchAny, [&create, &put]);
        let exporter = CloudWatchLogsExporter::builder("group")
            .client(client)
            .build();

        assert!(exporter.export(&record(Vec::new())).await.is_err());
        assert_eq!(create.num_calls(), 1);
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn builds_a_client_from_the_default_chain() {
        let exporter = CloudWatchLogsExporter::builder("group")
            .region("eu-west-1")
            .build();
        let client = exporter.client().await;
        assert_eq!(
            client.config().region().map(ToString::to_string).as_deref(),
            Some("eu-west-1")
        );
    }
}
