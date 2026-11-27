//! Writes records to an S3 bucket with `PutObject`.

use aws_durable_execution_sdk::BoxError;
use aws_sdk_s3::Client;
use aws_sdk_s3::error::DisplayErrorContext;
use aws_sdk_s3::primitives::ByteStream;

use super::call_error;
use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};
use crate::timestamp::{Civil, parse_iso_ms};

/// The default key prefix.
const DEFAULT_PREFIX: &str = "durable-execution-telemetry/";

/// The default record size limit, in bytes. S3 has no small per-object
/// limit, so this is a guard against an unbounded record.
const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 5_000_000;

/// The partition a record's object key has, between the prefix and the file
/// name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum S3Partitioning {
    /// `year=YYYY/month=MM/day=DD/`, the UTC date of the record's
    /// `start_time`. A `start_time` that is not a valid timestamp gives
    /// `year=unknown/month=unknown/day=unknown/`.
    #[default]
    Date,
    /// `function=<function name>/`, with the function name sanitized.
    FunctionName,
    /// No partition. The object is directly under the prefix.
    None,
}

/// Writes each record as one JSON object to an S3 bucket.
///
/// The key is `{prefix}{partition}{name}.json`. The name is the record's
/// `execution_name`, or its `execution_arn` when it has no name. Every
/// character of the name outside `a-z`, `A-Z`, `0-9`, `.`, `_` and `-` is
/// replaced with `_`. An execution keeps its name across invocations. So
/// a later record of the execution overwrites the same object, as long as
/// the record's `start_time` is on the same UTC day. The plugin takes
/// `start_time` from the execution start that the SDK reports. When the SDK
/// reports none, `start_time` can differ between invocations. Under
/// [`S3Partitioning::Date`], the execution's records then go to more than one
/// date partition. The `operations` field of the object is an array.
///
/// The function's role needs `s3:PutObject` on the objects under the prefix.
#[derive(Debug)]
pub struct S3Exporter {
    bucket: String,
    prefix: String,
    partitioning: S3Partitioning,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
}

/// Builds an [`S3Exporter`].
#[derive(Debug)]
#[must_use]
pub struct S3ExporterBuilder {
    bucket: String,
    prefix: String,
    partitioning: S3Partitioning,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

impl S3Exporter {
    /// A builder for an exporter that writes to `bucket`.
    pub fn builder(bucket: impl Into<String>) -> S3ExporterBuilder {
        S3ExporterBuilder {
            bucket: bucket.into(),
            prefix: DEFAULT_PREFIX.to_owned(),
            partitioning: S3Partitioning::Date,
            region: None,
            max_record_size_bytes: DEFAULT_MAX_RECORD_SIZE_BYTES,
            client: None,
        }
    }

    /// The object key of `record`.
    fn key(&self, record: &TelemetryRecord) -> String {
        let partition = match self.partitioning {
            S3Partitioning::Date => date_partition(&record.start_time),
            S3Partitioning::FunctionName => {
                format!("function={}/", sanitize(&record.function_name))
            }
            S3Partitioning::None => String::new(),
        };
        let name = record
            .execution_name
            .as_deref()
            .unwrap_or(&record.execution_arn);
        format!("{}{partition}{}.json", self.prefix, sanitize(name))
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

    /// Writes `record` with one `PutObject` request.
    async fn put(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        // The scheduler truncated the record to the size of the
        // `operations_format` rendering. So the same format is rendered here.
        let body = render_json(record, self.operations_format())?;
        let client = self.client().await;
        client
            .put_object()
            .bucket(&self.bucket)
            .key(self.key(record))
            .content_type("application/json")
            .body(ByteStream::from(body.into_bytes()))
            .send()
            .await
            .map_err(|error| call_error("PutObject", DisplayErrorContext(&error)))?;
        Ok(())
    }
}

/// The `year=YYYY/month=MM/day=DD/` partition of a `start_time` in the form
/// `YYYY-MM-DDTHH:MM:SS.sssZ`.
fn date_partition(start_time: &str) -> String {
    match parse_iso_ms(start_time) {
        Some(ms) => {
            let day = Civil::from_epoch_ms(ms);
            format!(
                "year={:04}/month={:02}/day={:02}/",
                day.year, day.month, day.day
            )
        }
        None => "year=unknown/month=unknown/day=unknown/".to_owned(),
    }
}

/// Replaces every character outside `a-z`, `A-Z`, `0-9`, `.`, `_` and `-`
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

impl S3ExporterBuilder {
    /// Sets the key prefix. The default is `durable-execution-telemetry/`.
    /// The exporter does not add a separator, so a prefix that names a
    /// folder ends with `/`.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Sets how objects are partitioned under the prefix. The default is
    /// [`S3Partitioning::Date`].
    pub fn partitioning(mut self, partitioning: S3Partitioning) -> Self {
        self.partitioning = partitioning;
        self
    }

    /// Sets the region of the client the exporter builds. Without it, the
    /// default region chain applies. A [`client`](Self::client) ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default
    /// is 5,000,000.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Uses `client` instead of building one from the default chain.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`S3Client`](crate::S3Client), so callers can name it without
    /// depending on `aws-sdk-s3` directly.
    ///
    /// # Examples
    ///
    /// The SDK crate re-exports the configuration type. So this compiles
    /// with no direct `aws-config` or `aws-sdk-s3` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::SdkConfig;
    /// use aws_durable_execution_sdk_telemetry_exporter::{S3Client, S3Exporter};
    ///
    /// let config = SdkConfig::builder().build();
    /// let exporter = S3Exporter::builder("my-telemetry-bucket")
    ///     .client(S3Client::new(&config))
    ///     .build();
    /// # drop(exporter);
    /// ```
    pub fn client(mut self, client: crate::S3Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> S3Exporter {
        S3Exporter {
            bucket: self.bucket,
            prefix: self.prefix,
            partitioning: self.partitioning,
            region: self.region,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
        }
    }
}

impl TelemetryExporter for S3Exporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.put(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }

    fn operations_format(&self) -> OperationsFormat {
        OperationsFormat::Array
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tests::{op, record};
    use aws_sdk_s3::operation::put_object::{PutObjectError, PutObjectOutput};
    use aws_smithy_mocks::{RuleMode, mock, mock_client};

    #[test]
    fn sanitize_replaces_every_character_outside_the_allowed_set() {
        assert_eq!(sanitize("aZ09._-"), "aZ09._-");
        assert_eq!(
            sanitize("arn:aws:lambda:us-west-2:1:f/my exec"),
            "arn_aws_lambda_us-west-2_1_f_my_exec"
        );
        // One replacement for each character, not for each byte.
        assert_eq!(sanitize("é€"), "__");
    }

    #[test]
    fn the_date_partition_is_the_utc_date_of_the_start_time() {
        assert_eq!(
            date_partition("2024-02-29T23:59:59.999Z"),
            "year=2024/month=02/day=29/"
        );
        assert_eq!(
            date_partition("2024-03-01T00:00:00.000Z"),
            "year=2024/month=03/day=01/"
        );
    }

    #[test]
    fn an_unparsable_start_time_gives_the_unknown_partition() {
        for text in [
            "",
            "yesterday",
            "2024-02-30T00:00:00.000Z",
            "2024-02-29T12:00:00+02:00",
        ] {
            assert_eq!(
                date_partition(text),
                "year=unknown/month=unknown/day=unknown/",
                "{text}"
            );
        }
    }

    #[test]
    fn the_key_has_the_prefix_the_date_partition_and_the_execution_name() {
        let exporter = S3Exporter::builder("b").build();
        assert_eq!(
            exporter.key(&record(Vec::new())),
            "durable-execution-telemetry/year=2026/month=01/day=01/exec.json"
        );
    }

    #[test]
    fn records_of_one_execution_on_different_start_days_have_different_keys() {
        let exporter = S3Exporter::builder("b").build();
        let first = record(Vec::new());
        let mut second = record(Vec::new());
        second.start_time = "2026-01-02T00:00:00.000Z".to_owned();
        assert_ne!(exporter.key(&first), exporter.key(&second));
    }

    #[test]
    fn the_key_uses_the_sanitized_arn_when_the_record_has_no_name() {
        let exporter = S3Exporter::builder("b").prefix("p/").build();
        let mut r = record(Vec::new());
        r.execution_name = None;
        r.execution_arn =
            "arn:aws:lambda:us-west-2:1:function:f:1/durable-execution/n/id".to_owned();
        assert_eq!(
            exporter.key(&r),
            "p/year=2026/month=01/day=01/arn_aws_lambda_us-west-2_1_function_f_1_durable-execution_n_id.json"
        );
    }

    #[test]
    fn an_empty_name_is_kept_and_does_not_fall_back_to_the_arn() {
        let exporter = S3Exporter::builder("b")
            .partitioning(S3Partitioning::None)
            .build();
        let mut r = record(Vec::new());
        r.execution_name = Some(String::new());
        assert_eq!(exporter.key(&r), "durable-execution-telemetry/.json");
    }

    #[test]
    fn function_name_partitioning_sanitizes_the_function_name() {
        let exporter = S3Exporter::builder("b")
            .partitioning(S3Partitioning::FunctionName)
            .build();
        let mut r = record(Vec::new());
        r.function_name = "my fn/1".to_owned();
        assert_eq!(
            exporter.key(&r),
            "durable-execution-telemetry/function=my_fn_1/exec.json"
        );
    }

    #[test]
    fn no_partitioning_puts_the_object_directly_under_the_prefix() {
        let exporter = S3Exporter::builder("b")
            .prefix("")
            .partitioning(S3Partitioning::None)
            .build();
        assert_eq!(exporter.key(&record(Vec::new())), "exec.json");
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = S3Exporter::builder("b")
            .region("us-west-2")
            .max_record_size_bytes(10)
            .build();
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::Array);
        assert_eq!(exporter.region.as_deref(), Some("us-west-2"));
        assert_eq!(
            S3Exporter::builder("b").build().max_record_size_bytes(),
            Some(5_000_000)
        );
    }

    #[tokio::test]
    async fn puts_the_record_as_json_with_the_array_form_of_the_operations() {
        let put = mock!(Client::put_object)
            .match_requests(|req| {
                req.bucket() == Some("bucket")
                    && req.key()
                        == Some("durable-execution-telemetry/year=2026/month=01/day=01/exec.json")
                    && req.content_type() == Some("application/json")
            })
            .then_output(|| PutObjectOutput::builder().build());
        let client = mock_client!(aws_sdk_s3, RuleMode::MatchAny, [&put]);
        let exporter = S3Exporter::builder("bucket").client(client).build();

        let r = record(vec![op("a", "SUCCEEDED")]);
        assert!(exporter.export(&r).await.is_ok());
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(put.num_calls(), 2);
    }

    #[tokio::test]
    async fn the_body_is_the_rendered_record() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        let expected = render_json(&r, OperationsFormat::Array).unwrap_or_default();
        let put = mock!(Client::put_object)
            .match_requests(move |req| {
                let body = req.body().bytes().unwrap_or_default();
                !expected.is_empty()
                    && body == expected.as_bytes()
                    && expected.contains("\"operations\":[{")
            })
            .then_output(|| PutObjectOutput::builder().build());
        let client = mock_client!(aws_sdk_s3, RuleMode::MatchAny, [&put]);
        let exporter = S3Exporter::builder("bucket").client(client).build();

        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_put_error_fails_the_export_and_names_the_call_and_the_error() {
        let put = mock!(Client::put_object).then_error(|| {
            PutObjectError::generic(
                aws_sdk_s3::error::ErrorMetadata::builder()
                    .code("NoSuchBucket")
                    .message("The specified bucket does not exist")
                    .build(),
            )
        });
        let client = mock_client!(aws_sdk_s3, RuleMode::MatchAny, [&put]);
        let exporter = S3Exporter::builder("bucket").client(client).build();

        let error = exporter
            .export(&record(Vec::new()))
            .await
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.starts_with("PutObject failed: "), "{error}");
        assert!(error.contains("NoSuchBucket"), "{error}");
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn builds_a_client_from_the_default_chain() {
        let exporter = S3Exporter::builder("bucket").region("eu-west-1").build();
        let client = exporter.client().await;
        assert_eq!(
            client.config().region().map(ToString::to_string).as_deref(),
            Some("eu-west-1")
        );
    }
}
