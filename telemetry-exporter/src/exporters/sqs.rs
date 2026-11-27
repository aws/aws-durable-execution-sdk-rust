//! Sends records to an Amazon SQS queue with `SendMessage`.

use aws_durable_execution_sdk::BoxError;
use aws_sdk_sqs::Client;
use aws_sdk_sqs::error::{DisplayErrorContext, SdkError};
use aws_sdk_sqs::types::MessageAttributeValue;
use sha2::{Digest, Sha256};

use super::DEFAULT_MAX_RECORD_SIZE_BYTES;
use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::{ExecutionStatus, TelemetryRecord};
use crate::render::{OperationsFormat, render_json};

/// The longest `MessageGroupId` or `MessageDeduplicationId` that a FIFO
/// queue accepts, in characters.
const MAX_FIFO_ID_LENGTH: usize = 128;

/// Sends each record as one message to an SQS queue.
///
/// The message body is the record's JSON. The message has two `String`
/// attributes: `status`, the record's execution status, and
/// `functionName`.
///
/// A queue URL that ends in `.fifo` names a FIFO queue. For a FIFO queue,
/// the message group ID is the configured group ID, or the execution ARN
/// when none is configured. The deduplication ID is
/// `{executionArn}:{emittedAt}`. SQS limits both IDs to 128 characters.
/// The exporter replaces an ID that is longer than 128 UTF-16 code units
/// with the lowercase hex SHA-256 digest of the ID's UTF-8 bytes. A
/// standard queue receives neither ID.
///
/// The function's role needs `sqs:SendMessage` on the queue.
#[derive(Debug)]
pub struct SqsExporter {
    queue_url: String,
    message_group_id: Option<String>,
    is_fifo: bool,
    region: Option<String>,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
}

/// Builds a [`SqsExporter`].
#[derive(Debug)]
#[must_use]
pub struct SqsExporterBuilder {
    queue_url: String,
    message_group_id: Option<String>,
    region: Option<String>,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

impl SqsExporter {
    /// A builder for an exporter that sends to the queue at `queue_url`.
    pub fn builder(queue_url: impl Into<String>) -> SqsExporterBuilder {
        SqsExporterBuilder {
            queue_url: queue_url.into(),
            message_group_id: None,
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

    /// Sends `record` as one message.
    async fn send(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        // The scheduler truncated the record to the size of the
        // `operations_format` rendering. So the same format is rendered here.
        let body = render_json(record, self.operations_format)?;
        let (group_id, deduplication_id) = if self.is_fifo {
            (
                Some(fifo_id(
                    self.message_group_id
                        .as_deref()
                        .unwrap_or(&record.execution_arn),
                )),
                Some(fifo_id(&format!(
                    "{}:{}",
                    record.execution_arn, record.emitted_at
                ))),
            )
        } else {
            (None, None)
        };
        self.client()
            .await
            .send_message()
            .queue_url(&self.queue_url)
            .message_body(body)
            .set_message_group_id(group_id)
            .set_message_deduplication_id(deduplication_id)
            .message_attributes("status", string_attribute(status_text(record.status)))
            .message_attributes("functionName", string_attribute(&record.function_name))
            .send()
            .await
            .map_err(|error| call_error("SendMessage", &error))?;
        Ok(())
    }
}

/// A message attribute of the data type `String`.
fn string_attribute(value: &str) -> MessageAttributeValue {
    #[expect(
        clippy::expect_used,
        reason = "build fails only when the data type is missing, and it is set"
    )]
    MessageAttributeValue::builder()
        .data_type("String")
        .string_value(value)
        .build()
        .expect("the data type is set")
}

/// The wire text of an execution status, the same text the record's JSON
/// carries.
fn status_text(status: ExecutionStatus) -> &'static str {
    match status {
        ExecutionStatus::Running => "RUNNING",
        ExecutionStatus::Succeeded => "SUCCEEDED",
        ExecutionStatus::Failed => "FAILED",
    }
}

/// `value` when it fits the FIFO ID limit, otherwise the lowercase hex
/// SHA-256 digest of its UTF-8 bytes, which is 64 characters.
///
/// The length is the number of UTF-16 code units. For the ASCII text
/// that SQS accepts in an ID, this equals the number of characters.
fn fifo_id(value: &str) -> String {
    if value.encode_utf16().count() <= MAX_FIFO_ID_LENGTH {
        return value.to_owned();
    }
    format!("{:x}", Sha256::digest(value.as_bytes()))
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

impl SqsExporterBuilder {
    /// Sets the message group ID of a FIFO queue. The default is the
    /// execution ARN, which groups the messages of one execution. An empty
    /// string counts as not set. A standard queue ignores the group ID.
    pub fn message_group_id(mut self, group_id: impl Into<String>) -> Self {
        self.message_group_id = Some(group_id.into()).filter(|id| !id.is_empty());
        self
    }

    /// Sets how the message body lists operations. The default is
    /// [`OperationsFormat::Array`].
    pub fn operations_format(mut self, format: OperationsFormat) -> Self {
        self.operations_format = format;
        self
    }

    /// Sets the region of the client the exporter builds. Without it, the
    /// default region chain applies. A [`client`](Self::client) ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default
    /// is 256,000, which fits the SQS message limit of 256 KiB.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Uses `client` instead of building one from the default chain.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`SqsClient`](crate::SqsClient), so callers can name it without
    /// depending on `aws-sdk-sqs` directly.
    ///
    /// # Examples
    ///
    /// The SDK crate re-exports the configuration type. So this compiles
    /// with no direct `aws-config` or `aws-sdk-sqs` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::SdkConfig;
    /// use aws_durable_execution_sdk_telemetry_exporter::{SqsClient, SqsExporter};
    ///
    /// let config = SdkConfig::builder().build();
    /// let exporter = SqsExporter::builder("https://sqs.us-west-2.amazonaws.com/123456789012/telemetry")
    ///     .client(SqsClient::new(&config))
    ///     .build();
    /// # drop(exporter);
    /// ```
    pub fn client(mut self, client: crate::SqsClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> SqsExporter {
        #[expect(
            clippy::case_sensitive_file_extension_comparisons,
            reason = "the queue URL is not a file path, and SQS requires the exact lowercase suffix `.fifo` on a FIFO queue name"
        )]
        let is_fifo = self.queue_url.ends_with(".fifo");
        SqsExporter {
            is_fifo,
            queue_url: self.queue_url,
            message_group_id: self.message_group_id,
            region: self.region,
            operations_format: self.operations_format,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
        }
    }
}

impl TelemetryExporter for SqsExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.send(record))
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
    use aws_sdk_sqs::operation::send_message::{SendMessageError, SendMessageOutput};
    use aws_sdk_sqs::types::error::InvalidMessageContents;
    use aws_smithy_mocks::{RuleMode, mock, mock_client};

    const STANDARD: &str = "https://sqs.us-west-2.amazonaws.com/123456789012/telemetry";
    const FIFO: &str = "https://sqs.us-west-2.amazonaws.com/123456789012/telemetry.fifo";

    fn sha256_hex(value: &str) -> String {
        format!("{:x}", Sha256::digest(value.as_bytes()))
    }

    /// The text of a failed export's error. A successful export gives an
    /// empty text, which fails the caller's assertion on the text.
    fn error_text(result: Result<(), BoxError>) -> String {
        result
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default()
    }

    fn attribute(
        req: &aws_sdk_sqs::operation::send_message::SendMessageInput,
        name: &str,
    ) -> Option<String> {
        req.message_attributes()?
            .get(name)
            .filter(|value| value.data_type() == "String")
            .and_then(|value| value.string_value())
            .map(ToOwned::to_owned)
    }

    #[test]
    fn short_ids_are_kept_and_long_ids_are_hashed() {
        let exact = "a".repeat(128);
        assert_eq!(fifo_id(&exact), exact);
        let long = "a".repeat(129);
        assert_eq!(fifo_id(&long), sha256_hex(&long));
        assert_eq!(fifo_id(&long).len(), 64);
        // The SHA-256 digest of "abc".
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn the_id_length_counts_utf16_code_units() {
        // 64 U+1F600 characters are 64 chars, 256 UTF-8 bytes and 128
        // UTF-16 code units: kept.
        let kept = "\u{1F600}".repeat(64);
        assert_eq!(fifo_id(&kept), kept);
        // 65 are 65 chars but 130 UTF-16 code units: hashed.
        let hashed = "\u{1F600}".repeat(65);
        assert_eq!(fifo_id(&hashed), sha256_hex(&hashed));
    }

    #[test]
    fn status_text_matches_the_serialized_status() {
        for status in [
            ExecutionStatus::Running,
            ExecutionStatus::Succeeded,
            ExecutionStatus::Failed,
        ] {
            assert_eq!(
                serde_json::to_value(status).ok(),
                Some(serde_json::Value::from(status_text(status)))
            );
        }
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = SqsExporter::builder(STANDARD)
            .region("us-west-2")
            .max_record_size_bytes(10)
            .operations_format(OperationsFormat::ByName)
            .build();
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::ByName);
        assert_eq!(exporter.region.as_deref(), Some("us-west-2"));
        let default = SqsExporter::builder(STANDARD).build();
        assert_eq!(default.max_record_size_bytes(), Some(256_000));
        assert_eq!(default.operations_format(), OperationsFormat::Array);
    }

    #[test]
    fn only_a_fifo_url_makes_a_fifo_exporter() {
        assert!(SqsExporter::builder(FIFO).build().is_fifo);
        assert!(!SqsExporter::builder(STANDARD).build().is_fifo);
    }

    #[test]
    fn an_empty_group_id_counts_as_not_set() {
        assert_eq!(
            SqsExporter::builder(FIFO)
                .message_group_id("")
                .build()
                .message_group_id,
            None
        );
        assert_eq!(
            SqsExporter::builder(FIFO)
                .message_group_id("g")
                .build()
                .message_group_id,
            Some("g".to_owned())
        );
    }

    #[tokio::test]
    async fn a_standard_queue_gets_the_body_and_attributes_and_no_ids() {
        let send = mock!(Client::send_message)
            .match_requests(|req| {
                req.queue_url() == Some(STANDARD)
                    && req.message_group_id().is_none()
                    && req.message_deduplication_id().is_none()
                    && attribute(req, "status").as_deref() == Some("SUCCEEDED")
                    && attribute(req, "functionName").as_deref() == Some("fn")
                    && req
                        .message_body()
                        .is_some_and(|body| body.contains("\"operations\":[{"))
            })
            .then_output(|| SendMessageOutput::builder().build());
        let client = mock_client!(aws_sdk_sqs, RuleMode::MatchAny, [&send]);
        let exporter = SqsExporter::builder(STANDARD)
            .message_group_id("ignored")
            .client(client)
            .build();

        assert!(
            exporter
                .export(&record(vec![op("a", "SUCCEEDED")]))
                .await
                .is_ok()
        );
        assert_eq!(send.num_calls(), 1);
    }

    #[tokio::test]
    async fn the_body_uses_the_configured_operations_format() {
        let send = mock!(Client::send_message)
            .match_requests(|req| {
                req.message_body()
                    .is_some_and(|body| body.contains("\"operationsByName\":{\"a\""))
            })
            .then_output(|| SendMessageOutput::builder().build());
        let client = mock_client!(aws_sdk_sqs, RuleMode::MatchAny, [&send]);
        let exporter = SqsExporter::builder(STANDARD)
            .operations_format(OperationsFormat::ByName)
            .client(client)
            .build();

        assert!(
            exporter
                .export(&record(vec![op("a", "SUCCEEDED")]))
                .await
                .is_ok()
        );
        assert_eq!(send.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_fifo_queue_groups_by_execution_and_deduplicates_by_emission() {
        let r = record(Vec::new());
        let group = r.execution_arn.clone();
        let dedup = format!("{}:{}", r.execution_arn, r.emitted_at);
        let send = mock!(Client::send_message)
            .match_requests(move |req| {
                req.message_group_id() == Some(group.as_str())
                    && req.message_deduplication_id() == Some(dedup.as_str())
            })
            .then_output(|| SendMessageOutput::builder().build());
        let client = mock_client!(aws_sdk_sqs, RuleMode::MatchAny, [&send]);
        let exporter = SqsExporter::builder(FIFO).client(client).build();

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(send.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_fifo_queue_uses_the_configured_group_id() {
        let send = mock!(Client::send_message)
            .match_requests(|req| req.message_group_id() == Some("my-group"))
            .then_output(|| SendMessageOutput::builder().build());
        let client = mock_client!(aws_sdk_sqs, RuleMode::MatchAny, [&send]);
        let exporter = SqsExporter::builder(FIFO)
            .message_group_id("my-group")
            .client(client)
            .build();

        assert!(exporter.export(&record(Vec::new())).await.is_ok());
        assert_eq!(send.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_fifo_queue_with_an_empty_group_id_groups_by_execution() {
        let r = record(Vec::new());
        let group = r.execution_arn.clone();
        let send = mock!(Client::send_message)
            .match_requests(move |req| req.message_group_id() == Some(group.as_str()))
            .then_output(|| SendMessageOutput::builder().build());
        let client = mock_client!(aws_sdk_sqs, RuleMode::MatchAny, [&send]);
        let exporter = SqsExporter::builder(FIFO)
            .message_group_id("")
            .client(client)
            .build();

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(send.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_fifo_id_over_the_limit_is_hashed() {
        let mut r = record(Vec::new());
        r.execution_arn = format!("arn:{}", "x".repeat(150));
        let group = sha256_hex(&r.execution_arn);
        let dedup = sha256_hex(&format!("{}:{}", r.execution_arn, r.emitted_at));
        let send = mock!(Client::send_message)
            .match_requests(move |req| {
                req.message_group_id() == Some(group.as_str())
                    && req.message_deduplication_id() == Some(dedup.as_str())
            })
            .then_output(|| SendMessageOutput::builder().build());
        let client = mock_client!(aws_sdk_sqs, RuleMode::MatchAny, [&send]);
        let exporter = SqsExporter::builder(FIFO).client(client).build();

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(send.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_send_error_names_the_call_and_the_service_error() {
        let send = mock!(Client::send_message).then_error(|| {
            SendMessageError::InvalidMessageContents(InvalidMessageContents::builder().build())
        });
        let client = mock_client!(aws_sdk_sqs, RuleMode::MatchAny, [&send]);
        let exporter = SqsExporter::builder(STANDARD).client(client).build();

        let error = error_text(exporter.export(&record(Vec::new())).await);
        assert!(error.starts_with("SendMessage failed: "), "{error}");
        assert!(error.contains("InvalidMessageContents"), "{error}");
        assert_eq!(send.num_calls(), 1);
    }

    #[tokio::test]
    async fn builds_a_client_from_the_default_chain() {
        let exporter = SqsExporter::builder(STANDARD).region("eu-west-1").build();
        let client = exporter.client().await;
        assert_eq!(
            client.config().region().map(ToString::to_string).as_deref(),
            Some("eu-west-1")
        );
    }
}
