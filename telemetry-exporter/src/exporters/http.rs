//! Sends records to an HTTP endpoint.

use std::time::Duration;

use aws_durable_execution_sdk::BoxError;

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::http::HttpSender;
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};

/// The default request timeout, in milliseconds.
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// The most characters of a response body that an error message includes.
const MAX_ERROR_BODY_CHARS: usize = 200;

/// The HTTP method of the request that carries a record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HttpMethod {
    /// `POST`. This is the default.
    #[default]
    Post,
    /// `PUT`, for an endpoint that stores the record at the URL.
    Put,
}

impl HttpMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::Post => "POST",
            Self::Put => "PUT",
        }
    }
}

/// Sends each record to an HTTP endpoint as a JSON body.
///
/// Each record is one request, with `POST` unless
/// [`method`](HttpExporterBuilder::method) says otherwise. The request has
/// the header `Content-Type: application/json`. A header that you add with
/// [`header`](HttpExporterBuilder::header) replaces a default header with
/// the same name, ignoring case.
///
/// The endpoint must answer with a status from 200 to 299. Any other
/// status is an export error that names the status and includes the first
/// 200 characters of the response body. An `https` URL uses
/// `rustls`. A request that does not finish within the timeout, from
/// connecting to reading the response, is an export error. The exporter
/// does not follow redirects and does not retry.
///
/// The exporter checks the URL, the method and the headers when it sends,
/// not when it is built. An invalid value is an export error.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_telemetry_exporter::{HttpExporter, HttpMethod};
///
/// let exporter = HttpExporter::builder("https://example.com/telemetry")
///     .method(HttpMethod::Put)
///     .header("Authorization", "Bearer token")
///     .timeout_ms(5_000)
///     .build();
/// # drop(exporter);
/// ```
#[derive(Debug)]
pub struct HttpExporter {
    url: String,
    method: HttpMethod,
    headers: Vec<(String, String)>,
    operations_format: OperationsFormat,
    max_record_size_bytes: Option<usize>,
    sender: HttpSender,
}

/// Builds an [`HttpExporter`].
#[derive(Debug)]
#[must_use]
pub struct HttpExporterBuilder {
    url: String,
    method: HttpMethod,
    headers: Vec<(String, String)>,
    timeout_ms: u64,
    operations_format: OperationsFormat,
    max_record_size_bytes: Option<usize>,
}

impl HttpExporter {
    /// A builder for an exporter that sends records to `url`.
    pub fn builder(url: impl Into<String>) -> HttpExporterBuilder {
        HttpExporterBuilder {
            url: url.into(),
            method: HttpMethod::Post,
            headers: Vec::new(),
            timeout_ms: DEFAULT_TIMEOUT_MS,
            operations_format: OperationsFormat::Array,
            max_record_size_bytes: None,
        }
    }

    /// Sends `record` and checks the status.
    async fn post(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        // The scheduler truncated the record to the size of the
        // `operations_format` rendering. So the same format is rendered here.
        let body = render_json(record, self.operations_format)?;
        let mut headers = Vec::with_capacity(self.headers.len() + 1);
        headers.push(("Content-Type".to_owned(), "application/json".to_owned()));
        headers.extend(self.headers.iter().cloned());
        let response = self
            .sender
            .send(self.method.as_str(), &self.url, &headers, body)
            .await
            .map_err(|error| format!("HttpExporter: {error}"))?;
        if response.is_success() {
            Ok(())
        } else {
            let detail: String = response.body.chars().take(MAX_ERROR_BODY_CHARS).collect();
            let separator = if detail.is_empty() { "" } else { ": " };
            Err(format!(
                "HttpExporter: endpoint returned status {}{separator}{detail}",
                response.status
            )
            .into())
        }
    }
}

impl HttpExporterBuilder {
    /// Sets the HTTP method. The default is [`HttpMethod::Post`].
    pub fn method(mut self, method: HttpMethod) -> Self {
        self.method = method;
        self
    }

    /// Adds a request header, for example an authorization token. A header
    /// with the same name, ignoring case, replaces an earlier one and the
    /// default `Content-Type`.
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Sets the request timeout in milliseconds. The default is 10,000.
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Sets how the body lists operations. The default is
    /// [`OperationsFormat::Array`].
    pub fn operations_format(mut self, format: OperationsFormat) -> Self {
        self.operations_format = format;
        self
    }

    /// Sets the largest record, in bytes, before truncation. By default
    /// there is no limit and the exporter never truncates. Set a limit if
    /// the endpoint caps the size of a request.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = Some(max);
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> HttpExporter {
        let timeout = Duration::from_millis(self.timeout_ms);
        HttpExporter {
            url: self.url,
            method: self.method,
            headers: self.headers,
            operations_format: self.operations_format,
            max_record_size_bytes: self.max_record_size_bytes,
            sender: HttpSender::new(timeout),
        }
    }
}

impl TelemetryExporter for HttpExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.post(record))
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
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "a test fails by panicking"
)]
mod tests {
    use super::*;
    use crate::http::tests::{Reply, ok, reply, serve_once};
    use crate::render::tests::{op, record};
    use serde_json::Value;

    fn body_of(request: &str) -> Value {
        let (_, body) = request.split_once("\r\n\r\n").expect("head and body");
        serde_json::from_str(body).expect("json body")
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let exporter = HttpExporter::builder("http://x/").build();
        assert_eq!(exporter.method, HttpMethod::Post);
        assert_eq!(exporter.sender.timeout(), Duration::from_secs(10));
        assert_eq!(exporter.max_record_size_bytes(), None);
        assert_eq!(exporter.operations_format(), OperationsFormat::Array);
        assert_eq!(exporter.headers, []);
    }

    #[test]
    fn builder_settings_reach_the_exporter() {
        let exporter = HttpExporter::builder("http://x/")
            .method(HttpMethod::Put)
            .header("a", "1")
            .timeout_ms(250)
            .operations_format(OperationsFormat::ByName)
            .max_record_size_bytes(99)
            .build();
        assert_eq!(exporter.method, HttpMethod::Put);
        assert_eq!(exporter.headers, [("a".to_owned(), "1".to_owned())]);
        assert_eq!(exporter.sender.timeout(), Duration::from_millis(250));
        assert_eq!(exporter.max_record_size_bytes(), Some(99));
        assert_eq!(exporter.operations_format(), OperationsFormat::ByName);
    }

    #[tokio::test]
    async fn posts_the_array_form_as_json_by_default() {
        let (url, server) = serve_once(ok("")).await;
        let exporter = HttpExporter::builder(url).build();
        exporter
            .export(&record(vec![op("fetch", "SUCCEEDED")]))
            .await
            .expect("exported");
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
        let body = body_of(&request);
        assert!(body["operations"].is_array(), "{body}");
        assert!(body.get("operationsByName").is_none(), "{body}");
        assert_eq!(body["operations"][0]["name"], "fetch");
    }

    #[tokio::test]
    async fn puts_the_by_name_form_when_configured() {
        let (url, server) = serve_once(ok("")).await;
        let exporter = HttpExporter::builder(url)
            .method(HttpMethod::Put)
            .operations_format(OperationsFormat::ByName)
            .build();
        exporter
            .export(&record(vec![op("fetch", "SUCCEEDED")]))
            .await
            .expect("exported");
        let request = server.await.expect("server");
        assert!(
            request.starts_with("PUT /path?q=1 HTTP/1.1\r\n"),
            "{request}"
        );
        let body = body_of(&request);
        assert!(body["operationsByName"]["fetch"].is_object(), "{body}");
        assert!(body.get("operations").is_none(), "{body}");
    }

    #[tokio::test]
    async fn sends_caller_headers_and_lets_them_override_the_content_type() {
        let (url, server) = serve_once(ok("")).await;
        let exporter = HttpExporter::builder(url)
            .header("Authorization", "Bearer t")
            .header("content-type", "application/vnd.test+json")
            .build();
        exporter.export(&record(vec![])).await.expect("exported");
        let request = server.await.expect("server").to_lowercase();
        assert!(request.contains("authorization: bearer t\r\n"), "{request}");
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
        let (url, _server) = serve_once(reply(204, "No Content", "")).await;
        let exporter = HttpExporter::builder(url).build();
        exporter
            .export(&record(vec![]))
            .await
            .expect("204 is success");
    }

    #[tokio::test]
    async fn a_non_2xx_status_is_an_error_that_names_the_status() {
        let (url, _server) = serve_once(reply(500, "Internal Server Error", "oops")).await;
        let exporter = HttpExporter::builder(url).build();
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "HttpExporter: endpoint returned status 500: oops"
        );
    }

    #[tokio::test]
    async fn an_error_includes_only_the_start_of_a_long_body() {
        let body = "x".repeat(300);
        let (url, _server) = serve_once(reply(400, "Bad Request", &body)).await;
        let exporter = HttpExporter::builder(url).build();
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "HttpExporter: endpoint returned status 400: {}",
                &body[..200]
            )
        );
    }

    #[tokio::test]
    async fn a_redirect_is_an_error_and_is_not_followed() {
        let (url, _server) = serve_once(reply(302, "Found", "")).await;
        let exporter = HttpExporter::builder(url).build();
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "HttpExporter: endpoint returned status 302"
        );
    }

    #[tokio::test]
    async fn a_slow_endpoint_is_a_timeout_error() {
        let (url, _server) = serve_once(Reply::Silent).await;
        let exporter = HttpExporter::builder(url).timeout_ms(100).build();
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "HttpExporter: request timed out after 100 ms"
        );
    }

    #[tokio::test]
    async fn an_invalid_url_is_an_export_error() {
        let exporter = HttpExporter::builder("not a url").build();
        let error = exporter.export(&record(vec![])).await.unwrap_err();
        assert!(
            error.to_string().starts_with("HttpExporter: invalid URL"),
            "{error}"
        );
    }
}
