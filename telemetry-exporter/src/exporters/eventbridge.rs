//! Publishes records to an `EventBridge` event bus with `PutEvents`.

use aws_durable_execution_sdk::BoxError;
use aws_sdk_eventbridge::Client;
use aws_sdk_eventbridge::error::{DisplayErrorContext, SdkError};
use aws_sdk_eventbridge::primitives::{DateTime, DateTimeFormat};
use aws_sdk_eventbridge::types::PutEventsRequestEntry;

use super::DEFAULT_MAX_RECORD_SIZE_BYTES;
use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};

/// The default event bus: the account's default bus.
const DEFAULT_EVENT_BUS_NAME: &str = "default";

/// The default event source.
const DEFAULT_SOURCE: &str = "software.amazon.lambda.durable.telemetry";

/// Publishes each record as one event to an `EventBridge` event bus.
///
/// Each event has these fields:
///
/// - `Source`: the configured source.
/// - `DetailType`: the record status, `RUNNING`, `SUCCEEDED` or `FAILED`.
/// - `Detail`: the record as JSON.
/// - `Time`: the record's `emittedAt`.
///
/// An `EventBridge` rule can match these fields. A rule can notify a
/// topic, start a workflow or fan out to other targets.
///
/// The function's role needs `events:PutEvents` on the event bus. A
/// `PutEvents` response can report success for the request and still
/// report that it rejected the entry. The exporter treats a rejected entry
/// as a failed export.
#[derive(Debug)]
pub struct EventBridgeExporter {
    event_bus_name: String,
    source: String,
    region: Option<String>,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
}

/// Builds an [`EventBridgeExporter`].
#[derive(Debug)]
#[must_use]
pub struct EventBridgeExporterBuilder {
    event_bus_name: String,
    source: String,
    region: Option<String>,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

impl EventBridgeExporter {
    /// A builder for an exporter that publishes to the default event bus.
    pub fn builder() -> EventBridgeExporterBuilder {
        EventBridgeExporterBuilder {
            event_bus_name: DEFAULT_EVENT_BUS_NAME.to_owned(),
            source: DEFAULT_SOURCE.to_owned(),
            region: None,
            operations_format: OperationsFormat::default(),
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

    /// The `PutEvents` entry for `record`.
    fn entry(&self, record: &TelemetryRecord) -> Result<PutEventsRequestEntry, BoxError> {
        // The scheduler truncated the record to the size of the
        // `operations_format` rendering. So the same format is rendered here.
        let detail = render_json(record, self.operations_format)?;
        let time = DateTime::from_str(&record.emitted_at, DateTimeFormat::DateTimeWithOffset)
            .map_err(|error| format!("emittedAt is not a valid time: {error}"))?;
        Ok(PutEventsRequestEntry::builder()
            .event_bus_name(&self.event_bus_name)
            .source(&self.source)
            .detail_type(record.status.as_str())
            .detail(detail)
            .time(time)
            .build())
    }

    /// Publishes `record` with one `PutEvents` request.
    async fn put(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        let entry = self.entry(record)?;
        let client = self.client().await;
        let response = client
            .put_events()
            .entries(entry)
            .send()
            .await
            .map_err(|error| call_error("PutEvents", &error))?;
        if response.failed_entry_count() > 0 {
            let entry = response.entries().first();
            let code = entry
                .and_then(|entry| entry.error_code())
                .unwrap_or("unknown");
            let message = entry
                .and_then(|entry| entry.error_message())
                .unwrap_or("no message");
            return Err(format!("PutEvents rejected the entry: {code}: {message}").into());
        }
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

impl EventBridgeExporterBuilder {
    /// Sets the event bus, by name or ARN. The default is `default`, the
    /// account's default event bus.
    pub fn event_bus_name(mut self, name: impl Into<String>) -> Self {
        self.event_bus_name = name.into();
        self
    }

    /// Sets the event source. The default is
    /// `software.amazon.lambda.durable.telemetry`.
    pub fn source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Sets the region of the client the exporter builds. Without it, the
    /// default region chain applies. A [`client`](Self::client) ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets how the event detail lists operations. The default is
    /// [`OperationsFormat::Array`]. Choose the form that your rule targets
    /// consume.
    pub fn operations_format(mut self, format: OperationsFormat) -> Self {
        self.operations_format = format;
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default
    /// is 256,000, which fits the 256 KB limit of one `PutEvents` entry.
    /// The service also counts the entry's other fields toward that limit,
    /// so a limit close to the maximum can still be rejected.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Uses `client` instead of building one from the default chain.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`EventBridgeClient`](crate::EventBridgeClient), so callers can
    /// name it without depending on `aws-sdk-eventbridge` directly.
    ///
    /// # Examples
    ///
    /// The SDK crate re-exports the configuration type. So this compiles
    /// with no direct `aws-config` or `aws-sdk-eventbridge` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::SdkConfig;
    /// use aws_durable_execution_sdk_telemetry_exporter::{
    ///     EventBridgeClient, EventBridgeExporter,
    /// };
    ///
    /// let config = SdkConfig::builder().build();
    /// let exporter = EventBridgeExporter::builder()
    ///     .client(EventBridgeClient::new(&config))
    ///     .build();
    /// # drop(exporter);
    /// ```
    pub fn client(mut self, client: crate::EventBridgeClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> EventBridgeExporter {
        EventBridgeExporter {
            event_bus_name: self.event_bus_name,
            source: self.source,
            region: self.region,
            operations_format: self.operations_format,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
        }
    }
}

impl TelemetryExporter for EventBridgeExporter {
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
#[expect(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    reason = "test assertions on known JSON keys and serializable values"
)]
mod tests {
    use super::*;
    use crate::record::ExecutionStatus;
    use crate::render::tests::{op, record};
    use aws_sdk_eventbridge::operation::put_events::{PutEventsError, PutEventsOutput};
    use aws_sdk_eventbridge::types::PutEventsResultEntry;
    use aws_sdk_eventbridge::types::error::InternalException;
    use aws_smithy_mocks::{RuleMode, mock, mock_client};
    use serde_json::Value;

    fn success() -> PutEventsOutput {
        PutEventsOutput::builder()
            .failed_entry_count(0)
            .entries(PutEventsResultEntry::builder().event_id("e1").build())
            .build()
    }

    fn error_text(result: Result<(), BoxError>) -> String {
        result
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = EventBridgeExporter::builder()
            .region("us-west-2")
            .max_record_size_bytes(10)
            .operations_format(OperationsFormat::Both)
            .build();
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::Both);
        assert_eq!(exporter.region.as_deref(), Some("us-west-2"));
        let default = EventBridgeExporter::builder().build();
        assert_eq!(default.max_record_size_bytes(), Some(256_000));
        assert_eq!(default.operations_format(), OperationsFormat::Array);
    }

    #[test]
    fn status_text_matches_the_serialized_status() {
        for status in [
            ExecutionStatus::Running,
            ExecutionStatus::Succeeded,
            ExecutionStatus::Failed,
        ] {
            assert_eq!(
                serde_json::to_value(status).unwrap(),
                Value::String(status.as_str().to_owned())
            );
        }
    }

    #[tokio::test]
    async fn publishes_one_event_with_the_default_fields() {
        let put = mock!(Client::put_events)
            .match_requests(|req| {
                let entry = req.entries().first();
                req.entries().len() == 1
                    && entry.is_some_and(|e| {
                        let detail: Value =
                            serde_json::from_str(e.detail().unwrap_or_default()).unwrap();
                        e.event_bus_name() == Some("default")
                            && e.source() == Some("software.amazon.lambda.durable.telemetry")
                            && e.detail_type() == Some("SUCCEEDED")
                            && e.time().map(DateTime::secs) == Some(1_767_225_601)
                            && detail["operations"][0]["name"] == "a"
                            && detail.get("operationsByName").is_none()
                    })
            })
            .then_output(success);
        let client = mock_client!(aws_sdk_eventbridge, RuleMode::MatchAny, [&put]);
        let exporter = EventBridgeExporter::builder().client(client).build();

        let r = record(vec![op("a", "SUCCEEDED")]);
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn uses_the_configured_bus_source_format_and_status() {
        let put = mock!(Client::put_events)
            .match_requests(|req| {
                req.entries().first().is_some_and(|e| {
                    let detail: Value =
                        serde_json::from_str(e.detail().unwrap_or_default()).unwrap();
                    e.event_bus_name() == Some("arn:aws:events:us-west-2:123:event-bus/b")
                        && e.source() == Some("my.app")
                        && e.detail_type() == Some("FAILED")
                        && detail.get("operations").is_none()
                        && detail["operationsByName"]["a"]["count"] == 1
                })
            })
            .then_output(success);
        let client = mock_client!(aws_sdk_eventbridge, RuleMode::MatchAny, [&put]);
        let exporter = EventBridgeExporter::builder()
            .event_bus_name("arn:aws:events:us-west-2:123:event-bus/b")
            .source("my.app")
            .operations_format(OperationsFormat::ByName)
            .client(client)
            .build();

        let mut r = record(vec![op("a", "SUCCEEDED")]);
        r.status = ExecutionStatus::Failed;
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_rejected_entry_fails_the_export_with_its_code_and_message() {
        let put = mock!(Client::put_events).then_output(|| {
            PutEventsOutput::builder()
                .failed_entry_count(1)
                .entries(
                    PutEventsResultEntry::builder()
                        .error_code("InternalFailure")
                        .error_message("try later")
                        .build(),
                )
                .build()
        });
        let client = mock_client!(aws_sdk_eventbridge, RuleMode::MatchAny, [&put]);
        let exporter = EventBridgeExporter::builder().client(client).build();

        let error = error_text(exporter.export(&record(Vec::new())).await);
        assert_eq!(
            error,
            "PutEvents rejected the entry: InternalFailure: try later"
        );
    }

    #[tokio::test]
    async fn a_rejection_without_an_entry_still_fails_the_export() {
        let put = mock!(Client::put_events)
            .then_output(|| PutEventsOutput::builder().failed_entry_count(1).build());
        let client = mock_client!(aws_sdk_eventbridge, RuleMode::MatchAny, [&put]);
        let exporter = EventBridgeExporter::builder().client(client).build();

        let error = error_text(exporter.export(&record(Vec::new())).await);
        assert_eq!(error, "PutEvents rejected the entry: unknown: no message");
    }

    #[tokio::test]
    async fn a_failed_call_fails_the_export_with_the_service_error() {
        let put = mock!(Client::put_events)
            .then_error(|| PutEventsError::InternalException(InternalException::builder().build()));
        let client = mock_client!(aws_sdk_eventbridge, RuleMode::MatchAny, [&put]);
        let exporter = EventBridgeExporter::builder().client(client).build();

        let error = error_text(exporter.export(&record(Vec::new())).await);
        assert!(error.starts_with("PutEvents failed: "), "{error}");
        assert!(error.contains("InternalException"), "{error}");
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn an_invalid_emitted_time_fails_the_export_without_a_call() {
        let put = mock!(Client::put_events).then_output(success);
        let client = mock_client!(aws_sdk_eventbridge, RuleMode::MatchAny, [&put]);
        let exporter = EventBridgeExporter::builder().client(client).build();

        let mut r = record(Vec::new());
        r.emitted_at = "yesterday".to_owned();
        let error = error_text(exporter.export(&r).await);
        assert!(
            error.starts_with("emittedAt is not a valid time: "),
            "{error}"
        );
        assert_eq!(put.num_calls(), 0);
    }

    #[tokio::test]
    async fn builds_a_client_from_the_default_chain() {
        let exporter = EventBridgeExporter::builder().region("eu-west-1").build();
        let client = exporter.client().await;
        assert_eq!(
            client.config().region().map(ToString::to_string).as_deref(),
            Some("eu-west-1")
        );
    }
}
