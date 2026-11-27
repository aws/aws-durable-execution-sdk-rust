//! Sends records to an OpenTelemetry endpoint as OTLP logs.

use std::time::Duration;

use aws_durable_execution_sdk::BoxError;
use serde_json::{Value, json};

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::http::HttpSender;
use crate::record::{ExecutionStatus, TelemetryRecord};
use crate::render::{OperationsFormat, render_json};
use crate::timestamp::parse_iso_ms;

/// The default request timeout, in milliseconds.
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// The default size limit of the OTLP request body, in bytes.
const DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 1_000_000;

/// The most characters of a response body that an error message includes.
const MAX_ERROR_BODY_CHARS: usize = 200;

/// The OTLP wire encoding of a request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OtlpProtocol {
    /// JSON over HTTP, `http/json`. This is the default.
    #[default]
    HttpJson,
    /// Protobuf over HTTP, `http/protobuf`. The exporter does not support
    /// it: [`OtelExporterBuilder::build`] returns an error for it.
    HttpProtobuf,
}

/// Sends each record to an OTLP endpoint as one OTLP log record.
///
/// Each record is one `POST` of an OTLP `ExportLogsServiceRequest` in JSON.
/// The request has the header `Content-Type: application/json`. A header
/// that you add with [`header`](OtelExporterBuilder::header) replaces a
/// default header with the same name, ignoring case.
///
/// The log record's body is the record's JSON, as a string. The resource
/// carries the function's name, region, account and qualifier. The log
/// record's attributes carry the execution ARN, name, status and duration.
/// The severity is 17 (`ERROR`) for a failed execution, 9 (`INFO`) for a
/// running or succeeded one.
///
/// The size limit measures the whole request body, not only the record,
/// because the record is embedded in the body as an escaped string.
///
/// The endpoint must answer with a status from 200 to 299. Any other
/// status is an export error that names the status and includes the first
/// 200 characters of the response body. An `https` URL uses `rustls`. A
/// request that does not finish within the timeout is an export error. The
/// exporter does not follow redirects and does not retry.
///
/// The exporter checks the URL and the headers when it sends, not when it
/// is built. An invalid value is an export error.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_telemetry_exporter::OtelExporter;
///
/// let exporter = OtelExporter::builder("https://otlp.example.com/v1/logs")
///     .header("Authorization", "Bearer token")
///     .timeout_ms(5_000)
///     .build()?;
/// # drop(exporter);
/// # Ok::<(), aws_durable_execution_sdk::BoxError>(())
/// ```
#[derive(Debug)]
pub struct OtelExporter {
    endpoint: String,
    headers: Vec<(String, String)>,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
    sender: HttpSender,
}

/// Builds an [`OtelExporter`].
#[derive(Debug)]
#[must_use]
pub struct OtelExporterBuilder {
    endpoint: String,
    headers: Vec<(String, String)>,
    protocol: OtlpProtocol,
    timeout_ms: u64,
    operations_format: OperationsFormat,
    max_record_size_bytes: usize,
}

impl OtelExporter {
    /// A builder for an exporter that sends OTLP logs to `endpoint`, for
    /// example `http://localhost:4318/v1/logs`.
    pub fn builder(endpoint: impl Into<String>) -> OtelExporterBuilder {
        OtelExporterBuilder {
            endpoint: endpoint.into(),
            headers: Vec::new(),
            protocol: OtlpProtocol::HttpJson,
            timeout_ms: DEFAULT_TIMEOUT_MS,
            operations_format: OperationsFormat::Array,
            max_record_size_bytes: DEFAULT_MAX_RECORD_SIZE_BYTES,
        }
    }

    /// The OTLP request body for `record`.
    fn payload(&self, record: &TelemetryRecord) -> Result<String, BoxError> {
        let time_ms = parse_iso_ms(&record.emitted_at).ok_or_else(|| {
            format!(
                "OtelExporter: emittedAt is not an ISO 8601 UTC time: {}",
                record.emitted_at
            )
        })?;
        // OTLP defines `timeUnixNano` as an unsigned 64-bit count. So a
        // time before the epoch or after the year 2554 has no valid
        // value. The product uses `i128` because it can exceed `i64`.
        let time_nanos = u64::try_from(i128::from(time_ms) * 1_000_000).map_err(|_| {
            format!(
                "OtelExporter: emittedAt is outside the OTLP time range: {}",
                record.emitted_at
            )
        })?;
        let status = match record.status {
            ExecutionStatus::Failed => "FAILED",
            ExecutionStatus::Succeeded => "SUCCEEDED",
            _ => "RUNNING",
        };
        let severity = if record.status == ExecutionStatus::Failed {
            17
        } else {
            9
        };
        let body = render_json(record, self.operations_format)?;
        let payload = json!({
            "resourceLogs": [{
                "resource": {
                    "attributes": [
                        string_attribute("service.name", &record.function_name),
                        string_attribute("cloud.region", &record.region),
                        string_attribute("cloud.account.id", &record.account_id),
                        string_attribute("faas.name", &record.function_name),
                        string_attribute("faas.version", &record.function_qualifier),
                    ],
                },
                "scopeLogs": [{
                    "scope": {
                        "name": env!("CARGO_PKG_NAME"),
                        "version": record.schema_version,
                    },
                    "logRecords": [{
                        "timeUnixNano": time_nanos.to_string(),
                        "severityNumber": severity,
                        "severityText": status,
                        "body": { "stringValue": body },
                        "attributes": [
                            string_attribute("workflow.execution_arn", &record.execution_arn),
                            string_attribute(
                                "workflow.execution_name",
                                record.execution_name.as_deref().unwrap_or(""),
                            ),
                            string_attribute("workflow.status", status),
                            {
                                "key": "workflow.duration_ms",
                                "value": { "intValue": record.duration_ms.unwrap_or(0).to_string() },
                            },
                        ],
                    }],
                }],
            }],
        });
        Ok(serde_json::to_string(&payload)?)
    }

    /// Sends `record` and checks the status.
    async fn post(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        let body = self.payload(record)?;
        let mut headers = Vec::with_capacity(self.headers.len() + 1);
        headers.push(("Content-Type".to_owned(), "application/json".to_owned()));
        headers.extend(self.headers.iter().cloned());
        let response = self
            .sender
            .send("POST", &self.endpoint, &headers, body)
            .await
            .map_err(|error| format!("OtelExporter: {error}"))?;
        if response.is_success() {
            Ok(())
        } else {
            let detail: String = response.body.chars().take(MAX_ERROR_BODY_CHARS).collect();
            let separator = if detail.is_empty() { "" } else { ": " };
            Err(format!(
                "OtelExporter: endpoint returned status {}{separator}{detail}",
                response.status
            )
            .into())
        }
    }
}

/// An OTLP attribute with a string value.
fn string_attribute(key: &str, value: &str) -> Value {
    json!({ "key": key, "value": { "stringValue": value } })
}

impl OtelExporterBuilder {
    /// Adds a request header, for example an API key. A header with the
    /// same name, ignoring case, replaces an earlier one and the default
    /// `Content-Type`.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Sets the OTLP protocol. The default is [`OtlpProtocol::HttpJson`].
    /// [`build`](Self::build) returns an error for any other value.
    pub fn protocol(mut self, protocol: OtlpProtocol) -> Self {
        self.protocol = protocol;
        self
    }

    /// Sets the request timeout in milliseconds. The default is 10,000.
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Sets how the log body lists operations. The default is
    /// [`OperationsFormat::Array`].
    pub fn operations_format(mut self, format: OperationsFormat) -> Self {
        self.operations_format = format;
        self
    }

    /// Sets the largest request body, in bytes, before truncation. The
    /// default is 1,000,000. Zero disables truncation.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Builds the exporter.
    ///
    /// # Errors
    ///
    /// Returns an error when the protocol is [`OtlpProtocol::HttpProtobuf`].
    pub fn build(self) -> Result<OtelExporter, BoxError> {
        if self.protocol != OtlpProtocol::HttpJson {
            return Err("OtelExporter: http/protobuf is not supported. Use http/json.".into());
        }
        let timeout = Duration::from_millis(self.timeout_ms);
        Ok(OtelExporter {
            endpoint: self.endpoint,
            headers: self.headers,
            operations_format: self.operations_format,
            max_record_size_bytes: self.max_record_size_bytes,
            sender: HttpSender::new(timeout),
        })
    }
}

impl TelemetryExporter for OtelExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.post(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }

    fn operations_format(&self) -> OperationsFormat {
        self.operations_format
    }

    fn measured_text(&self, record: &TelemetryRecord) -> Result<String, BoxError> {
        self.payload(record)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "a test fails by panicking"
)]
mod tests {
    use super::*;
    use crate::http::tests::{Reply, ok, reply, serve_once};
    use crate::render::tests::{op, record};
    use crate::truncation::{Measure, truncate_measured};
    use serde_json::Value;

    fn exporter() -> OtelExporter {
        OtelExporter::builder("http://x/").build().expect("built")
    }

    fn payload_of(exporter: &OtelExporter, record: &TelemetryRecord) -> Value {
        serde_json::from_str(&exporter.payload(record).expect("payload")).expect("json")
    }

    fn body_of(request: &str) -> Value {
        let (_, body) = request.split_once("\r\n\r\n").expect("head and body");
        serde_json::from_str(body).expect("json body")
    }

    fn log_record(payload: &Value) -> &Value {
        &payload["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    }

    fn attribute<'a>(attributes: &'a Value, key: &str) -> &'a Value {
        attributes
            .as_array()
            .and_then(|all| all.iter().find(|item| item["key"] == key))
            .map_or(&Value::Null, |item| &item["value"])
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let exporter = exporter();
        assert_eq!(exporter.sender.timeout(), Duration::from_secs(10));
        assert_eq!(exporter.max_record_size_bytes(), Some(1_000_000));
        assert_eq!(exporter.operations_format(), OperationsFormat::Array);
        assert_eq!(exporter.headers, []);
    }

    #[test]
    fn builder_settings_reach_the_exporter() {
        let exporter = OtelExporter::builder("http://x/")
            .header("a", "1")
            .protocol(OtlpProtocol::HttpJson)
            .timeout_ms(250)
            .operations_format(OperationsFormat::Both)
            .max_record_size_bytes(99)
            .build()
            .expect("built");
        assert_eq!(exporter.headers, [("a".to_owned(), "1".to_owned())]);
        assert_eq!(exporter.sender.timeout(), Duration::from_millis(250));
        assert_eq!(exporter.max_record_size_bytes(), Some(99));
        assert_eq!(exporter.operations_format(), OperationsFormat::Both);
    }

    #[test]
    fn the_protobuf_protocol_is_a_build_error() {
        let error = OtelExporter::builder("http://x/")
            .protocol(OtlpProtocol::HttpProtobuf)
            .build()
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "OtelExporter: http/protobuf is not supported. Use http/json."
        );
    }

    #[test]
    fn the_payload_maps_the_record_to_an_otlp_log_record() {
        let mut r = record(vec![op("fetch", "SUCCEEDED")]);
        r.emitted_at = "2026-11-27T10:20:30.123Z".to_owned();
        r.execution_name = Some("order-1".to_owned());
        r.duration_ms = Some(42);
        let payload = payload_of(&exporter(), &r);
        let resource = &payload["resourceLogs"][0]["resource"]["attributes"];
        for (key, value) in [
            ("service.name", &r.function_name),
            ("faas.name", &r.function_name),
            ("cloud.region", &r.region),
            ("cloud.account.id", &r.account_id),
            ("faas.version", &r.function_qualifier),
        ] {
            assert_eq!(attribute(resource, key)["stringValue"], *value, "{key}");
        }
        let scope = &payload["resourceLogs"][0]["scopeLogs"][0]["scope"];
        assert_eq!(scope["name"], env!("CARGO_PKG_NAME"));
        assert_eq!(scope["version"], r.schema_version);
        let log = log_record(&payload);
        assert_eq!(log["timeUnixNano"], "1795774830123000000");
        assert_eq!(log["severityNumber"], 9);
        assert_eq!(log["severityText"], "SUCCEEDED");
        let attributes = &log["attributes"];
        assert_eq!(
            attribute(attributes, "workflow.execution_arn")["stringValue"],
            r.execution_arn
        );
        assert_eq!(
            attribute(attributes, "workflow.execution_name")["stringValue"],
            "order-1"
        );
        assert_eq!(
            attribute(attributes, "workflow.status")["stringValue"],
            "SUCCEEDED"
        );
        assert_eq!(
            attribute(attributes, "workflow.duration_ms")["intValue"],
            "42"
        );
        let body: Value =
            serde_json::from_str(log["body"]["stringValue"].as_str().expect("string body"))
                .expect("json body");
        assert_eq!(body["operations"][0]["name"], "fetch");
        assert!(body.get("operationsByName").is_none(), "{body}");
    }

    #[test]
    fn a_missing_name_and_duration_become_an_empty_string_and_zero() {
        let mut r = record(vec![]);
        r.execution_name = None;
        r.duration_ms = None;
        let payload = payload_of(&exporter(), &r);
        let attributes = &log_record(&payload)["attributes"];
        assert_eq!(
            attribute(attributes, "workflow.execution_name")["stringValue"],
            ""
        );
        assert_eq!(
            attribute(attributes, "workflow.duration_ms")["intValue"],
            "0"
        );
    }

    #[test]
    fn severity_follows_the_status() {
        for (status, number, text) in [
            (ExecutionStatus::Failed, 17, "FAILED"),
            (ExecutionStatus::Running, 9, "RUNNING"),
            (ExecutionStatus::Succeeded, 9, "SUCCEEDED"),
        ] {
            let mut r = record(vec![]);
            r.status = status;
            let payload = payload_of(&exporter(), &r);
            let log = log_record(&payload);
            assert_eq!(log["severityNumber"], number, "{text}");
            assert_eq!(log["severityText"], text);
            assert_eq!(
                attribute(&log["attributes"], "workflow.status")["stringValue"],
                text
            );
        }
    }

    #[test]
    fn the_body_lists_operations_in_the_configured_format() {
        let r = record(vec![op("fetch", "SUCCEEDED")]);
        for (format, array, by_name) in [
            (OperationsFormat::Array, true, false),
            (OperationsFormat::ByName, false, true),
            (OperationsFormat::Both, true, true),
        ] {
            let exporter = OtelExporter::builder("http://x/")
                .operations_format(format)
                .build()
                .expect("built");
            let payload = payload_of(&exporter, &r);
            let body: Value = serde_json::from_str(
                log_record(&payload)["body"]["stringValue"]
                    .as_str()
                    .expect("string body"),
            )
            .expect("json body");
            assert_eq!(body.get("operations").is_some(), array, "{format:?}");
            assert_eq!(
                body.get("operationsByName").is_some(),
                by_name,
                "{format:?}"
            );
        }
    }

    #[test]
    fn an_emitted_at_that_is_not_a_time_is_an_error() {
        let mut r = record(vec![]);
        r.emitted_at = "yesterday".to_owned();
        let error = exporter().payload(&r).unwrap_err();
        assert_eq!(
            error.to_string(),
            "OtelExporter: emittedAt is not an ISO 8601 UTC time: yesterday"
        );
    }

    #[test]
    fn an_emitted_at_outside_the_otlp_time_range_is_an_error() {
        for emitted_at in ["1969-12-31T23:59:59.999Z", "2554-07-21T23:34:33.710Z"] {
            let mut r = record(vec![]);
            r.emitted_at = emitted_at.to_owned();
            let error = exporter().payload(&r).unwrap_err();
            assert_eq!(
                error.to_string(),
                format!("OtelExporter: emittedAt is outside the OTLP time range: {emitted_at}")
            );
        }
    }

    #[test]
    fn the_first_and_last_representable_times_are_accepted() {
        for (emitted_at, nanos) in [
            ("1970-01-01T00:00:00.000Z", "0"),
            ("2554-07-21T23:34:33.709Z", "18446744073709000000"),
        ] {
            let mut r = record(vec![]);
            r.emitted_at = emitted_at.to_owned();
            let payload = payload_of(&exporter(), &r);
            assert_eq!(log_record(&payload)["timeUnixNano"], nanos);
        }
    }

    #[test]
    fn the_measured_text_is_the_whole_request_body() {
        let exporter = exporter();
        let r = record(vec![op("fetch", "SUCCEEDED")]);
        let text = exporter.measured_text(&r).expect("text");
        assert_eq!(text, exporter.payload(&r).expect("payload"));
        let bare = render_json(&r, OperationsFormat::Array).expect("json");
        assert!(text.len() > bare.len());
    }

    #[test]
    fn truncation_keeps_the_whole_request_under_the_limit() {
        let big = "x".repeat(300);
        let operations = (0..4)
            .map(|n| {
                let mut o = op(&format!("op{n}"), "SUCCEEDED");
                o.start_ms = Some(n);
                o.result = Some(Value::String(big.clone()));
                o
            })
            .collect();
        let r = record(operations);
        let exporter = exporter();
        let bare = render_json(&r, OperationsFormat::Array)
            .expect("json")
            .len();
        let full = exporter.payload(&r).expect("payload").len();
        // The bare record's JSON fits here. The request does not.
        let limit = bare + 100;
        assert!(full > limit);
        let measure =
            |record: &TelemetryRecord| exporter.measured_text(record).ok().map(|t| t.len());
        let out = truncate_measured(&r, Some(limit), Measure::Text(&measure));
        assert!(out.truncated);
        assert!(exporter.payload(&out).expect("payload").len() <= limit);
        assert!(out.operations.iter().any(|o| o.truncated));
    }

    #[tokio::test]
    async fn posts_the_payload_as_json() {
        let (url, server) = serve_once(ok("")).await;
        let exporter = OtelExporter::builder(url).build().expect("built");
        let r = record(vec![op("fetch", "SUCCEEDED")]);
        exporter.export(&r).await.expect("exported");
        let request = server.await.expect("server");
        assert!(
            request.starts_with("POST /path?q=1 HTTP/1.1\r\n"),
            "{request}"
        );
        assert!(
            request
                .to_lowercase()
                .contains("content-type: application/json\r\n"),
            "{request}"
        );
        assert_eq!(body_of(&request), payload_of(&exporter, &r));
    }

    #[tokio::test]
    async fn sends_caller_headers_and_lets_them_override_the_content_type() {
        let (url, server) = serve_once(ok("")).await;
        let exporter = OtelExporter::builder(url)
            .header("x-api-key", "k")
            .header("content-type", "application/vnd.test+json")
            .build()
            .expect("built");
        exporter.export(&record(vec![])).await.expect("exported");
        let request = server.await.expect("server").to_lowercase();
        assert!(request.contains("x-api-key: k\r\n"), "{request}");
        assert!(
            request.contains("content-type: application/vnd.test+json\r\n"),
            "{request}"
        );
        assert!(
            !request.contains("content-type: application/json"),
            "{request}"
        );
    }

    #[tokio::test]
    async fn accepts_any_2xx_status() {
        let (url, _server) = serve_once(reply(202, "Accepted", "")).await;
        let exporter = OtelExporter::builder(url).build().expect("built");
        exporter
            .export(&record(vec![]))
            .await
            .expect("202 is success");
    }

    #[tokio::test]
    async fn a_non_2xx_status_is_an_error_that_names_the_status() {
        let (url, _server) = serve_once(reply(500, "Internal Server Error", "oops")).await;
        let exporter = OtelExporter::builder(url).build().expect("built");
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "OtelExporter: endpoint returned status 500: oops"
        );
    }

    #[tokio::test]
    async fn an_error_without_a_body_names_only_the_status() {
        let (url, _server) = serve_once(reply(302, "Found", "")).await;
        let exporter = OtelExporter::builder(url).build().expect("built");
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "OtelExporter: endpoint returned status 302"
        );
    }

    #[tokio::test]
    async fn an_error_includes_only_the_start_of_a_long_body() {
        let body = "x".repeat(300);
        let (url, _server) = serve_once(reply(400, "Bad Request", &body)).await;
        let exporter = OtelExporter::builder(url).build().expect("built");
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "OtelExporter: endpoint returned status 400: {}",
                &body[..200]
            )
        );
    }

    #[tokio::test]
    async fn a_slow_endpoint_is_a_timeout_error() {
        let (url, _server) = serve_once(Reply::Silent).await;
        let exporter = OtelExporter::builder(url)
            .timeout_ms(100)
            .build()
            .expect("built");
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "OtelExporter: request timed out after 100 ms"
        );
    }

    #[tokio::test]
    async fn an_invalid_url_is_an_export_error() {
        let exporter = OtelExporter::builder("not a url").build().expect("built");
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert!(
            error.to_string().starts_with("OtelExporter: invalid URL"),
            "{error}"
        );
    }
}
