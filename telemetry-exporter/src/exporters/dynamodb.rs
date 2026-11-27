//! Writes records to a `DynamoDB` table with `PutItem`.

use std::collections::HashMap;

use aws_durable_execution_sdk::BoxError;
use aws_sdk_dynamodb::Client;
use aws_sdk_dynamodb::types::AttributeValue;
use serde_json::Value;

use super::call_error::call_error;
use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_value};

/// The default partition key attribute name.
const DEFAULT_PARTITION_KEY: &str = "pk";

/// The default sort key attribute name.
const DEFAULT_SORT_KEY: &str = "sk";

/// The default record size limit, in bytes.
const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 400_000;

/// The largest integer, 2^53 - 1, that a JSON number holds without losing
/// precision in an IEEE 754 double.
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// [`MAX_SAFE_INTEGER`] as a double. The value is exact in a double.
const MAX_SAFE_FLOAT: f64 = 9_007_199_254_740_991.0;

/// Writes each record as one item to a `DynamoDB` table.
///
/// The item holds the `operationsByName` form of the record, one attribute
/// for each top-level field. It also holds the partition key attribute,
/// which is the execution ARN, and the sort key attribute, which is the
/// record's `emittedAt` time. If the record has a field with the name of a
/// key attribute, the key replaces it.
///
/// With a sort key, each record is a new item. So the table keeps the
/// history of an execution. Without a sort key, the table must have only
/// the partition key. Then each record replaces the previous record of the
/// same execution.
///
/// The record size limit applies to the JSON rendering of the record. It
/// does not apply to the size of the item that the exporter sends.
///
/// The function's role needs `dynamodb:PutItem` on the table.
#[derive(Debug)]
pub struct DynamoDbExporter {
    table_name: String,
    partition_key: String,
    sort_key: Option<String>,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: tokio::sync::OnceCell<Client>,
}

/// Builds a [`DynamoDbExporter`].
#[derive(Debug)]
#[must_use]
pub struct DynamoDbExporterBuilder {
    table_name: String,
    partition_key: String,
    sort_key: Option<String>,
    region: Option<String>,
    max_record_size_bytes: usize,
    client: Option<Client>,
}

impl DynamoDbExporter {
    /// A builder for an exporter that writes to the table `table_name`.
    pub fn builder(table_name: impl Into<String>) -> DynamoDbExporterBuilder {
        DynamoDbExporterBuilder {
            table_name: table_name.into(),
            partition_key: DEFAULT_PARTITION_KEY.to_owned(),
            sort_key: Some(DEFAULT_SORT_KEY.to_owned()),
            region: None,
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

    /// The item for `record`.
    ///
    /// # Errors
    ///
    /// Returns an error if the record has a number that [`attribute_value`]
    /// rejects.
    fn item(&self, record: &TelemetryRecord) -> Result<HashMap<String, AttributeValue>, BoxError> {
        // The scheduler truncated the record to the size of the
        // `operations_format` rendering. So the same format is rendered here.
        let rendered = render_value(record, self.operations_format())?;
        #[expect(
            clippy::expect_used,
            reason = "a record serializes as a struct, which is always a JSON object"
        )]
        let fields = rendered
            .as_object()
            .expect("a rendered record is a JSON object");
        let mut item = HashMap::with_capacity(fields.len() + 2);
        for (name, value) in fields {
            item.insert(name.clone(), attribute_value(value)?);
        }
        item.insert(
            self.partition_key.clone(),
            AttributeValue::S(record.execution_arn.clone()),
        );
        if let Some(sort_key) = &self.sort_key {
            item.insert(
                sort_key.clone(),
                AttributeValue::S(record.emitted_at.clone()),
            );
        }
        Ok(item)
    }

    /// Writes `record` with one `PutItem` request.
    async fn put(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        let item = self.item(record)?;
        self.client()
            .await
            .put_item()
            .table_name(&self.table_name)
            .set_item(Some(item))
            .send()
            .await
            .map_err(|error| call_error("PutItem", &error))?;
        Ok(())
    }
}

/// The `DynamoDB` attribute value of a JSON value.
///
/// An object is a map (`M`), an array is a list (`L`), a string is a string
/// (`S`), a number is a number (`N`), a boolean is a boolean (`BOOL`) and
/// `null` is `NULL`. An empty string, array or object keeps its type.
///
/// # Errors
///
/// Returns an error for a number whose magnitude is above 2^53 - 1. A
/// number of that size can lose precision before it reaches this function.
/// So the record is not written.
fn attribute_value(value: &Value) -> Result<AttributeValue, BoxError> {
    Ok(match value {
        Value::Null => AttributeValue::Null(true),
        Value::Bool(flag) => AttributeValue::Bool(*flag),
        Value::String(text) => AttributeValue::S(text.clone()),
        Value::Number(number) => {
            let safe = if let Some(int) = number.as_i64() {
                int.unsigned_abs() <= MAX_SAFE_INTEGER
            } else if let Some(int) = number.as_u64() {
                int <= MAX_SAFE_INTEGER
            } else {
                // A number that is neither integer type is a finite `f64`.
                number
                    .as_f64()
                    .is_some_and(|float| float.abs() <= MAX_SAFE_FLOAT)
            };
            if !safe {
                return Err(format!(
                    "the number {number} is outside the range -(2^53 - 1) to 2^53 - 1"
                )
                .into());
            }
            AttributeValue::N(number.to_string())
        }
        Value::Array(items) => AttributeValue::L(
            items
                .iter()
                .map(attribute_value)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Value::Object(fields) => AttributeValue::M(
            fields
                .iter()
                .map(|(name, field)| Ok((name.clone(), attribute_value(field)?)))
                .collect::<Result<HashMap<_, _>, BoxError>>()?,
        ),
    })
}

impl DynamoDbExporterBuilder {
    /// Sets the partition key attribute name. Its value is the execution
    /// ARN. The default is `pk`.
    pub fn partition_key(mut self, name: impl Into<String>) -> Self {
        self.partition_key = name.into();
        self
    }

    /// Sets the sort key attribute name. Its value is the record's
    /// `emittedAt` time. The default is `sk`.
    ///
    /// The exporter sends the name as given, including an empty name. To
    /// write no sort key, call
    /// [`without_sort_key`](Self::without_sort_key).
    pub fn sort_key(mut self, name: impl Into<String>) -> Self {
        self.sort_key = Some(name.into());
        self
    }

    /// Writes no sort key attribute. The table must then have only the
    /// partition key, and each record replaces the previous record of the
    /// same execution.
    pub fn without_sort_key(mut self) -> Self {
        self.sort_key = None;
        self
    }

    /// Sets the region of the client the exporter builds. Without it, the
    /// default region chain applies. A [`client`](Self::client) ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default
    /// is 400,000.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Uses `client` instead of building one from the default chain.
    ///
    /// The parameter type is re-exported at the crate root as
    /// [`DynamoDbClient`](crate::DynamoDbClient), so callers can name it
    /// without depending on `aws-sdk-dynamodb` directly.
    ///
    /// # Examples
    ///
    /// The SDK crate re-exports the configuration type. So this compiles
    /// with no direct `aws-config` or `aws-sdk-dynamodb` dependency:
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk::SdkConfig;
    /// use aws_durable_execution_sdk_telemetry_exporter::{DynamoDbClient, DynamoDbExporter};
    ///
    /// let config = SdkConfig::builder().build();
    /// let exporter = DynamoDbExporter::builder("my-telemetry-table")
    ///     .client(DynamoDbClient::new(&config))
    ///     .build();
    /// # drop(exporter);
    /// ```
    pub fn client(mut self, client: crate::DynamoDbClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> DynamoDbExporter {
        DynamoDbExporter {
            table_name: self.table_name,
            partition_key: self.partition_key,
            sort_key: self.sort_key,
            region: self.region,
            max_record_size_bytes: self.max_record_size_bytes,
            client: tokio::sync::OnceCell::new_with(self.client),
        }
    }
}

impl TelemetryExporter for DynamoDbExporter {
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
    use aws_sdk_dynamodb::operation::put_item::{PutItemError, PutItemOutput};
    use aws_sdk_dynamodb::types::error::ResourceNotFoundException;
    use aws_smithy_mocks::{RuleMode, mock, mock_client};
    use serde_json::json;

    fn default_exporter() -> DynamoDbExporter {
        DynamoDbExporter::builder("table").build()
    }

    fn text(value: &str) -> AttributeValue {
        AttributeValue::S(value.to_owned())
    }

    #[test]
    fn reports_its_limit_and_format() {
        let exporter = DynamoDbExporter::builder("t")
            .region("us-west-2")
            .max_record_size_bytes(10)
            .build();
        assert_eq!(exporter.max_record_size_bytes(), Some(10));
        assert_eq!(exporter.operations_format(), OperationsFormat::ByName);
        assert_eq!(exporter.region.as_deref(), Some("us-west-2"));
        assert_eq!(default_exporter().max_record_size_bytes(), Some(400_000));
    }

    #[test]
    fn the_item_has_the_by_name_record_and_both_keys() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        let item = default_exporter().item(&r).ok().unwrap_or_default();
        assert_eq!(item.get("pk"), Some(&text("arn")));
        assert_eq!(item.get("sk"), Some(&text("2026-01-01T00:00:01.000Z")));
        assert_eq!(
            item.get("recordType"),
            Some(&text("DurableExecutionTelemetry"))
        );
        assert_eq!(item.get("executionArn"), Some(&text("arn")));
        assert_eq!(
            item.get("durationMs"),
            Some(&AttributeValue::N("1000".to_owned()))
        );
        assert!(matches!(
            item.get("operationsByName"),
            Some(AttributeValue::M(_))
        ));
        assert!(!item.contains_key("operations"));
    }

    #[test]
    fn the_key_attribute_names_are_configurable() {
        let r = record(Vec::new());
        let item = DynamoDbExporter::builder("t")
            .partition_key("arn")
            .sort_key("at")
            .build()
            .item(&r)
            .ok()
            .unwrap_or_default();
        assert_eq!(item.get("arn"), Some(&text("arn")));
        assert_eq!(item.get("at"), Some(&text("2026-01-01T00:00:01.000Z")));
        assert!(!item.contains_key("pk"));
        assert!(!item.contains_key("sk"));
    }

    #[test]
    fn without_a_sort_key_the_item_has_no_sort_key_attribute() {
        let r = record(Vec::new());
        let item = DynamoDbExporter::builder("t")
            .without_sort_key()
            .build()
            .item(&r)
            .ok()
            .unwrap_or_default();
        assert_eq!(item.get("pk"), Some(&text("arn")));
        assert!(!item.contains_key("sk"));
    }

    #[test]
    fn a_key_replaces_a_record_field_of_the_same_name() {
        let r = record(Vec::new());
        let item = DynamoDbExporter::builder("t")
            .partition_key("status")
            .sort_key("functionName")
            .build()
            .item(&r)
            .ok()
            .unwrap_or_default();
        assert_eq!(item.get("status"), Some(&text("arn")));
        assert_eq!(
            item.get("functionName"),
            Some(&text("2026-01-01T00:00:01.000Z"))
        );
    }

    #[test]
    fn json_values_convert_to_attribute_values() {
        let value = json!({
            "s": "x",
            "empty": "",
            "i": -5,
            "f": 1.5,
            "t": true,
            "n": null,
            "l": [1, "a", [], {}],
        });
        let converted = attribute_value(&value).ok();
        let expected = AttributeValue::M(HashMap::from([
            ("s".to_owned(), text("x")),
            ("empty".to_owned(), text("")),
            ("i".to_owned(), AttributeValue::N("-5".to_owned())),
            ("f".to_owned(), AttributeValue::N("1.5".to_owned())),
            ("t".to_owned(), AttributeValue::Bool(true)),
            ("n".to_owned(), AttributeValue::Null(true)),
            (
                "l".to_owned(),
                AttributeValue::L(vec![
                    AttributeValue::N("1".to_owned()),
                    text("a"),
                    AttributeValue::L(Vec::new()),
                    AttributeValue::M(HashMap::new()),
                ]),
            ),
        ]));
        assert_eq!(converted, Some(expected));
    }

    #[test]
    fn numbers_up_to_the_safe_integer_limit_convert() {
        for (number, text) in [
            (json!(9_007_199_254_740_991_i64), "9007199254740991"),
            (json!(-9_007_199_254_740_991_i64), "-9007199254740991"),
            (json!(9_007_199_254_740_991_u64), "9007199254740991"),
            (json!(9_007_199_254_740_991.0_f64), "9007199254740991.0"),
        ] {
            assert_eq!(
                attribute_value(&number).ok(),
                Some(AttributeValue::N(text.to_owned())),
                "{number}"
            );
        }
    }

    #[test]
    fn numbers_beyond_the_safe_integer_limit_are_rejected() {
        for number in [
            json!(9_007_199_254_740_992_i64),
            json!(-9_007_199_254_740_992_i64),
            json!(i64::MIN),
            json!(u64::MAX),
            json!(1e300),
            json!(-1e300),
        ] {
            let error = attribute_value(&number).err().map(|e| e.to_string());
            assert!(
                error.is_some_and(|e| e.contains("outside the range")),
                "{number}"
            );
        }
    }

    #[test]
    fn a_number_nested_in_the_record_fails_the_item() {
        let mut r = record(Vec::new());
        r.output = Some(json!({"list": [{"big": u64::MAX}]}));
        assert!(default_exporter().item(&r).is_err());
    }

    #[tokio::test]
    async fn puts_the_item_in_the_table() {
        let put = mock!(Client::put_item)
            .match_requests(|req| {
                let item = req.item();
                req.table_name() == Some("table")
                    && item.is_some_and(|item| {
                        item.get("pk") == Some(&text("arn"))
                            && item.get("sk") == Some(&text("2026-01-01T00:00:01.000Z"))
                            && matches!(item.get("operationsByName"), Some(AttributeValue::M(m)) if m.contains_key("a"))
                    })
            })
            .then_output(|| PutItemOutput::builder().build());
        let client = mock_client!(aws_sdk_dynamodb, RuleMode::MatchAny, [&put]);
        let exporter = DynamoDbExporter::builder("table").client(client).build();

        let r = record(vec![op("a", "SUCCEEDED")]);
        assert!(exporter.export(&r).await.is_ok());
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_failed_put_names_the_call_and_the_service_error() {
        let put = mock!(Client::put_item).then_error(|| {
            PutItemError::ResourceNotFoundException(ResourceNotFoundException::builder().build())
        });
        let client = mock_client!(aws_sdk_dynamodb, RuleMode::MatchAny, [&put]);
        let exporter = DynamoDbExporter::builder("table").client(client).build();

        let error = exporter
            .export(&record(Vec::new()))
            .await
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(error.starts_with("PutItem failed: "), "{error}");
        assert!(error.contains("ResourceNotFoundException"), "{error}");
        assert_eq!(put.num_calls(), 1);
    }

    #[tokio::test]
    async fn a_record_with_an_unsafe_number_makes_no_call() {
        let put = mock!(Client::put_item).then_output(|| PutItemOutput::builder().build());
        let client = mock_client!(aws_sdk_dynamodb, RuleMode::MatchAny, [&put]);
        let exporter = DynamoDbExporter::builder("table").client(client).build();

        let mut r = record(Vec::new());
        r.input = Some(json!(u64::MAX));
        assert!(exporter.export(&r).await.is_err());
        assert_eq!(put.num_calls(), 0);
    }

    #[tokio::test]
    async fn builds_a_client_from_the_default_chain() {
        let exporter = DynamoDbExporter::builder("table")
            .region("eu-west-1")
            .build();
        let client = exporter.client().await;
        assert_eq!(
            client.config().region().map(ToString::to_string).as_deref(),
            Some("eu-west-1")
        );
    }
}
