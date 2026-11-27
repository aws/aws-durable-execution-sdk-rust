//! Writes records to an `OpenSearch` index with the Index API.

use std::fmt;
use std::time::{Duration, SystemTime};

use aws_config::SdkConfig;
use aws_credential_types::provider::ProvideCredentials;
use aws_durable_execution_sdk::BoxError;
use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use aws_smithy_types::error::display::DisplayErrorContext;

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::http::HttpSender;
use crate::record::TelemetryRecord;
use crate::render::{OperationsFormat, render_json};

/// The default index name.
const DEFAULT_INDEX_NAME: &str = "durable-execution-telemetry";

/// The default record size limit. The exporter truncates a larger record
/// before it sends the record.
const OPENSEARCH_DEFAULT_MAX_RECORD_SIZE_BYTES: usize = 10_000_000;

/// The default request timeout, in milliseconds.
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// The most characters of a response body that an error message includes.
const MAX_ERROR_BODY_CHARS: usize = 500;

/// The service name in the `SigV4` credential scope.
const SIGNING_NAME: &str = "es";

/// How the exporter authenticates.
#[derive(Clone, PartialEq, Eq)]
enum Auth {
    /// `SigV4` with the credentials of the default chain.
    SigV4,
    /// HTTP basic authentication.
    Basic { username: String, password: String },
}

impl fmt::Debug for Auth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SigV4 => formatter.write_str("SigV4"),
            Self::Basic { username, .. } => formatter
                .debug_struct("Basic")
                .field("username", username)
                .field("password", &"<redacted>")
                .finish(),
        }
    }
}

/// Writes each record as one document in an Amazon `OpenSearch` Service
/// index.
///
/// The request is `PUT {endpoint}/{index}/_doc/{id}`. The id is the
/// execution ARN, percent-encoded. A later record of the same execution
/// has the same id, so it replaces the earlier document. The body is the
/// `operations` array form of the record.
///
/// The default authentication is `SigV4` with the service name `es` and the
/// credentials of the default chain. The role needs `es:ESHttpPut` on the
/// index. [`basic_auth`](OpenSearchExporterBuilder::basic_auth) sends HTTP
/// basic authentication instead. The exporter does not create the index or
/// set its mapping.
///
/// The endpoint must answer with a status from 200 to 299. Any other
/// status is an export error that names the status and includes the first
/// 500 characters of the response body. The exporter does not follow
/// redirects and does not retry.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_telemetry_exporter::OpenSearchExporter;
///
/// let exporter = OpenSearchExporter::builder("https://my-domain.us-east-1.es.amazonaws.com")
///     .index_name("telemetry")
///     .region("us-east-1")
///     .build();
/// # drop(exporter);
/// ```
#[derive(Debug)]
pub struct OpenSearchExporter {
    endpoint: String,
    index_name: String,
    region: Option<String>,
    auth: Auth,
    max_record_size_bytes: usize,
    sender: HttpSender,
    config: tokio::sync::OnceCell<SdkConfig>,
}

/// Builds an [`OpenSearchExporter`].
#[derive(Debug)]
#[must_use]
pub struct OpenSearchExporterBuilder {
    endpoint: String,
    index_name: String,
    region: Option<String>,
    auth: Auth,
    timeout_ms: u64,
    max_record_size_bytes: usize,
}

impl OpenSearchExporter {
    /// A builder for an exporter that writes to the domain at `endpoint`,
    /// for example `https://my-domain.us-east-1.es.amazonaws.com`. One
    /// trailing slash is removed.
    pub fn builder(endpoint: impl Into<String>) -> OpenSearchExporterBuilder {
        OpenSearchExporterBuilder {
            endpoint: endpoint.into(),
            index_name: DEFAULT_INDEX_NAME.to_owned(),
            region: None,
            auth: Auth::SigV4,
            timeout_ms: DEFAULT_TIMEOUT_MS,
            max_record_size_bytes: OPENSEARCH_DEFAULT_MAX_RECORD_SIZE_BYTES,
        }
    }

    /// The URL of the document for `execution_arn`.
    fn document_url(&self, execution_arn: &str) -> String {
        format!(
            "{}/{}/_doc/{}",
            self.endpoint,
            self.index_name,
            encode_uri_component(execution_arn)
        )
    }

    /// The default credential and region configuration, loaded on first
    /// use.
    async fn config(&self) -> &SdkConfig {
        self.config
            .get_or_init(|| default_loader(self.region.as_deref()).load())
            .await
    }

    /// The authentication headers of a `PUT` of `body` to `url` at `time`.
    async fn auth_headers(
        &self,
        url: &str,
        body: &str,
        time: SystemTime,
    ) -> Result<Vec<(String, String)>, BoxError> {
        match &self.auth {
            Auth::Basic { username, password } => {
                let token = base64_encode(format!("{username}:{password}").as_bytes());
                Ok(vec![("Authorization".to_owned(), format!("Basic {token}"))])
            }
            Auth::SigV4 => {
                let config = self.config().await;
                let region = config
                    .region()
                    .ok_or("OpenSearchExporter: no AWS region is configured")?;
                let provider = config
                    .credentials_provider()
                    .ok_or("OpenSearchExporter: no AWS credentials provider is configured")?;
                let credentials = provider.provide_credentials().await.map_err(|error| {
                    format!(
                        "OpenSearchExporter: loading credentials failed: {}",
                        DisplayErrorContext(&error)
                    )
                })?;
                sign_put(url, body, &Identity::from(credentials), region.as_ref(), time)
            }
        }
    }

    /// Indexes `record` and checks the status.
    async fn index(&self, record: &TelemetryRecord) -> Result<(), BoxError> {
        // The scheduler truncated the record to the size of the Array
        // rendering. So the same format is rendered here.
        let body = render_json(record, OperationsFormat::Array)?;
        let url = self.document_url(&record.execution_arn);
        let mut headers = vec![("Content-Type".to_owned(), "application/json".to_owned())];
        headers.extend(self.auth_headers(&url, &body, SystemTime::now()).await?);
        let response = self
            .sender
            .send("PUT", &url, &headers, body)
            .await
            .map_err(|error| format!("OpenSearchExporter: {error}"))?;
        if response.is_success() {
            Ok(())
        } else {
            let detail: String = response.body.chars().take(MAX_ERROR_BODY_CHARS).collect();
            let separator = if detail.is_empty() { "" } else { ": " };
            Err(format!(
                "OpenSearchExporter: index request returned status {}{separator}{detail}",
                response.status
            )
            .into())
        }
    }
}

/// The loader of the default credential and region chain. `region`, when
/// set, replaces the region of the chain.
fn default_loader(region: Option<&str>) -> aws_config::ConfigLoader {
    let loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
    match region {
        Some(region) => loader.region(aws_config::Region::new(region.to_owned())),
        None => loader,
    }
}

/// The `SigV4` headers for a `PUT` of the JSON `body` to `url`.
///
/// The signature covers `host`, which the signer derives from the URL, and
/// `content-type`. The returned headers are the ones the signer adds, such
/// as `authorization` and `x-amz-date`. The caller sends them with the
/// `content-type` header.
fn sign_put(
    url: &str,
    body: &str,
    identity: &Identity,
    region: &str,
    time: SystemTime,
) -> Result<Vec<(String, String)>, BoxError> {
    #[expect(
        clippy::expect_used,
        reason = "build fails only when a field is missing, and every field is set"
    )]
    let params = v4::SigningParams::builder()
        .identity(identity)
        .region(region)
        .name(SIGNING_NAME)
        .time(time)
        .settings(SigningSettings::default())
        .build()
        .expect("every signing field is set")
        .into();
    let request = SignableRequest::new(
        "PUT",
        url,
        std::iter::once(("content-type", "application/json")),
        SignableBody::Bytes(body.as_bytes()),
    )
    .map_err(|error| format!("OpenSearchExporter: cannot sign the request: {error}"))?;
    let (instructions, _signature) = sign(request, &params)
        .map_err(|error| format!("OpenSearchExporter: signing failed: {error}"))?
        .into_parts();
    Ok(instructions
        .headers()
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect())
}

/// Percent-encodes `text` as a URL path segment. It keeps `A-Z a-z 0-9` and
/// `- _ . ! ~ * ' ( )`, and encodes every other byte of the UTF-8 form.
fn encode_uri_component(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for &byte in text.as_bytes() {
        if byte.is_ascii_alphanumeric() || b"-_.!~*'()".contains(&byte) {
            encoded.push(char::from(byte));
        } else {
            encoded.push('%');
            encoded.push(hex_digit(byte >> 4));
            encoded.push(hex_digit(byte & 0x0f));
        }
    }
    encoded
}

/// The upper-case hexadecimal digit of `value`, which is less than 16.
fn hex_digit(value: u8) -> char {
    #[expect(
        clippy::expect_used,
        reason = "both callers pass a value below 16, which is a hexadecimal digit"
    )]
    char::from_digit(u32::from(value), 16)
        .expect("value is below 16")
        .to_ascii_uppercase()
}

/// The base64 character of the 6-bit `value`.
fn base64_digit(value: u32) -> char {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    #[expect(
        clippy::expect_used,
        reason = "the caller masks the value to 6 bits, and the alphabet has 64 characters"
    )]
    let byte = usize::try_from(value)
        .ok()
        .and_then(|index| ALPHABET.get(index))
        .expect("value is below 64");
    char::from(*byte)
}

/// The standard base64 form of `bytes`, with padding.
fn base64_encode(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let group = chunk
            .iter()
            .enumerate()
            .fold(0_u32, |group, (index, &byte)| {
                group | (u32::from(byte) << (16 - 8 * index))
            });
        for index in 0..4 {
            if index <= chunk.len() {
                encoded.push(base64_digit((group >> (18 - 6 * index)) & 0x3f));
            } else {
                encoded.push('=');
            }
        }
    }
    encoded
}

impl OpenSearchExporterBuilder {
    /// Sets the index name. The default is `durable-execution-telemetry`.
    pub fn index_name(mut self, index_name: impl Into<String>) -> Self {
        self.index_name = index_name.into();
        self
    }

    /// Sets the region of the `SigV4` signature. Without it, the default
    /// region chain applies. Basic authentication ignores it.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Authenticates with HTTP basic authentication instead of `SigV4`.
    pub fn basic_auth(mut self, username: impl Into<String>, password: impl Into<String>) -> Self {
        self.auth = Auth::Basic {
            username: username.into(),
            password: password.into(),
        };
        self
    }

    /// Sets the request timeout in milliseconds. The default is 10,000.
    pub fn timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Sets the largest record, in bytes, before truncation. The default
    /// is 10,000,000.
    pub fn max_record_size_bytes(mut self, max: usize) -> Self {
        self.max_record_size_bytes = max;
        self
    }

    /// Builds the exporter.
    pub fn build(self) -> OpenSearchExporter {
        let endpoint = match self.endpoint.strip_suffix('/') {
            Some(trimmed) => trimmed.to_owned(),
            None => self.endpoint,
        };
        OpenSearchExporter {
            endpoint,
            index_name: self.index_name,
            region: self.region,
            auth: self.auth,
            max_record_size_bytes: self.max_record_size_bytes,
            sender: HttpSender::new(Duration::from_millis(self.timeout_ms)),
            config: tokio::sync::OnceCell::new(),
        }
    }
}

impl TelemetryExporter for OpenSearchExporter {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(self.index(record))
    }

    fn max_record_size_bytes(&self) -> Option<usize> {
        Some(self.max_record_size_bytes)
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "a test fails by panicking"
)]
mod tests {
    use super::*;
    use crate::http::tests::{Reply, ok, reply, serve_once};
    use crate::render::tests::{op, record};
    use aws_config::{Region, SdkConfig};
    use aws_credential_types::Credentials;
    use aws_credential_types::provider::error::CredentialsError;
    use aws_credential_types::provider::{SharedCredentialsProvider, future};
    use serde_json::Value;
    use std::time::UNIX_EPOCH;

    const ARN: &str = "arn:aws:lambda:us-east-1:123456789012:function:f:1/durable-execution/e/i";
    const ENCODED_ARN: &str =
        "arn%3Aaws%3Alambda%3Aus-east-1%3A123456789012%3Afunction%3Af%3A1%2Fdurable-execution%2Fe%2Fi";

    fn credentials() -> Credentials {
        Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("session-token".to_owned()),
            None,
            "test",
        )
    }

    /// The endpoint of a [`serve_once`] URL: scheme and authority only.
    fn endpoint_of(url: &str) -> String {
        url.strip_suffix("/path?q=1").expect("test URL").to_owned()
    }

    /// The header lines of a raw request whose name is `name`.
    fn header_values(request: &str, name: &str) -> Vec<String> {
        let (head, _) = request.split_once("\r\n\r\n").expect("head and body");
        head.lines()
            .skip(1)
            .filter_map(|line| {
                let (key, value) = line.split_once(':')?;
                key.eq_ignore_ascii_case(name)
                    .then(|| value.trim().to_owned())
            })
            .collect()
    }

    fn body_of(request: &str) -> Value {
        let (_, body) = request.split_once("\r\n\r\n").expect("head and body");
        serde_json::from_str(body).expect("json body")
    }

    /// An exporter whose default configuration is `config`.
    fn with_config(exporter: OpenSearchExporter, config: SdkConfig) -> OpenSearchExporter {
        OpenSearchExporter {
            config: tokio::sync::OnceCell::new_with(Some(config)),
            ..exporter
        }
    }

    fn signing_config() -> SdkConfig {
        SdkConfig::builder()
            .region(Region::new("us-east-1"))
            .credentials_provider(SharedCredentialsProvider::new(credentials()))
            .build()
    }

    #[test]
    fn defaults_match_the_documented_values() {
        let exporter = OpenSearchExporter::builder("https://x.example").build();
        assert_eq!(exporter.endpoint, "https://x.example");
        assert_eq!(exporter.index_name, "durable-execution-telemetry");
        assert_eq!(exporter.region, None);
        assert_eq!(exporter.auth, Auth::SigV4);
        assert_eq!(exporter.sender.timeout(), Duration::from_secs(10));
        assert_eq!(exporter.max_record_size_bytes(), Some(10_000_000));
        assert_eq!(exporter.operations_format(), OperationsFormat::Array);
    }

    #[test]
    fn builder_settings_reach_the_exporter() {
        let exporter = OpenSearchExporter::builder("https://x.example/")
            .index_name("idx")
            .region("eu-west-1")
            .basic_auth("user", "pass")
            .timeout_ms(250)
            .max_record_size_bytes(99)
            .build();
        assert_eq!(exporter.endpoint, "https://x.example");
        assert_eq!(exporter.index_name, "idx");
        assert_eq!(exporter.region.as_deref(), Some("eu-west-1"));
        assert_eq!(
            exporter.auth,
            Auth::Basic {
                username: "user".to_owned(),
                password: "pass".to_owned()
            }
        );
        assert_eq!(exporter.sender.timeout(), Duration::from_millis(250));
        assert_eq!(exporter.max_record_size_bytes(), Some(99));
    }

    #[tokio::test]
    async fn the_builder_region_replaces_the_region_of_the_default_chain() {
        let config = default_loader(Some("eu-west-1"))
            .credentials_provider(SharedCredentialsProvider::new(credentials()))
            .load()
            .await;
        assert_eq!(config.region().map(AsRef::as_ref), Some("eu-west-1"));
    }

    #[test]
    fn only_one_trailing_slash_is_removed() {
        let exporter = OpenSearchExporter::builder("https://x.example//").build();
        assert_eq!(exporter.endpoint, "https://x.example/");
    }

    #[test]
    fn debug_output_hides_the_password() {
        let exporter = OpenSearchExporter::builder("https://x.example")
            .basic_auth("user", "hunter2")
            .build();
        let text = format!("{exporter:?}");
        assert!(text.contains("user"), "{text}");
        assert!(!text.contains("hunter2"), "{text}");
        assert!(format!("{:?}", Auth::SigV4).contains("SigV4"));
    }

    #[test]
    fn the_document_url_encodes_the_execution_arn() {
        let exporter = OpenSearchExporter::builder("https://x.example/")
            .index_name("idx")
            .build();
        assert_eq!(
            exporter.document_url(ARN),
            format!("https://x.example/idx/_doc/{ENCODED_ARN}")
        );
    }

    #[test]
    fn uri_component_encoding_keeps_unreserved_characters_and_encodes_the_rest() {
        assert_eq!(
            encode_uri_component("aZ09-_.!~*'() :/?#é"),
            "aZ09-_.!~*'()%20%3A%2F%3F%23%C3%A9"
        );
    }

    #[test]
    fn base64_matches_the_standard_test_vectors() {
        let cases = [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ];
        for (text, expected) in cases {
            assert_eq!(base64_encode(text.as_bytes()), expected, "{text:?}");
        }
        assert_eq!(base64_encode(&[0xfb, 0xff, 0xfe]), "+//+");
    }

    #[test]
    fn sigv4_signature_matches_an_independent_computation() {
        // The expected signature comes from a separate SigV4 implementation
        // in a script, using the service `es` and a path that is encoded
        // twice in the canonical request.
        let time = UNIX_EPOCH + Duration::from_secs(1_767_323_045);
        let url = format!(
            "https://search-demo.us-east-1.es.amazonaws.com/durable-execution-telemetry/_doc/{ENCODED_ARN}"
        );
        let headers = sign_put(
            &url,
            "{\"a\":1}",
            &Identity::from(credentials()),
            "us-east-1",
            time,
        )
        .expect("signed");
        let find = |name: &str| {
            headers
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value.as_str())
        };
        assert_eq!(
            find("authorization"),
            Some(
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260102/us-east-1/es/aws4_request, \
                 SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, \
                 Signature=9ebe6ddcbbedbe50668808ff7ef1dc7eeb1da38f50ae86c67d54775dbace07f5"
            )
        );
        assert_eq!(find("x-amz-date"), Some("20260102T030405Z"));
        assert_eq!(find("x-amz-security-token"), Some("session-token"));
    }

    #[test]
    fn signing_an_invalid_url_is_an_error() {
        let error = sign_put(
            "not a url",
            "{}",
            &Identity::from(credentials()),
            "us-east-1",
            UNIX_EPOCH,
        )
        .expect_err("invalid URL");
        assert!(
            error
                .to_string()
                .starts_with("OpenSearchExporter: cannot sign the request: "),
            "{error}"
        );
    }

    #[tokio::test]
    async fn puts_the_array_form_with_basic_auth() {
        let (url, server) = serve_once(ok("{\"result\":\"created\"}")).await;
        let exporter = OpenSearchExporter::builder(endpoint_of(&url))
            .index_name("idx")
            .basic_auth("user", "pass")
            .build();
        let mut execution = record(vec![op("fetch", "SUCCEEDED")]);
        execution.execution_arn = ARN.to_owned();
        exporter.export(&execution).await.expect("exported");
        let request = server.await.expect("server");
        assert!(
            request.starts_with(&format!("PUT /idx/_doc/{ENCODED_ARN} HTTP/1.1\r\n")),
            "{request}"
        );
        assert_eq!(
            header_values(&request, "authorization"),
            ["Basic dXNlcjpwYXNz"]
        );
        assert_eq!(
            header_values(&request, "content-type"),
            ["application/json"]
        );
        let body = body_of(&request);
        assert_eq!(body["executionArn"], ARN);
        assert!(body["operations"].is_array(), "{body}");
        assert!(body.get("operationsByName").is_none(), "{body}");
    }

    #[tokio::test]
    async fn signs_the_request_with_sigv4_by_default() {
        let (url, server) = serve_once(reply(201, "Created", "")).await;
        let exporter = with_config(
            OpenSearchExporter::builder(endpoint_of(&url)).build(),
            signing_config(),
        );
        exporter
            .export(&record(vec![op("fetch", "SUCCEEDED")]))
            .await
            .expect("a 201 is a success");
        let request = server.await.expect("server");
        let authorization = header_values(&request, "authorization");
        assert_eq!(authorization.len(), 1, "{request}");
        let authorization = authorization.first().expect("one header");
        assert!(
            authorization.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
            "{authorization}"
        );
        assert!(
            authorization.contains("/us-east-1/es/aws4_request, SignedHeaders="),
            "{authorization}"
        );
        assert_eq!(header_values(&request, "x-amz-security-token").len(), 1);
        assert_eq!(header_values(&request, "x-amz-date").len(), 1);
        assert_eq!(header_values(&request, "host").len(), 1, "{request}");
        assert_eq!(header_values(&request, "content-type").len(), 1);
    }

    #[tokio::test]
    async fn a_failed_status_names_the_status_and_the_start_of_the_body() {
        let body = "x".repeat(600);
        let (url, _server) = serve_once(reply(403, "Forbidden", &body)).await;
        let exporter = OpenSearchExporter::builder(endpoint_of(&url))
            .basic_auth("u", "p")
            .build();
        let error = exporter
            .export(&record(vec![]))
            .await
            .expect_err("a 403 fails");
        assert_eq!(
            error.to_string(),
            format!(
                "OpenSearchExporter: index request returned status 403: {}",
                &body[..500]
            )
        );
    }

    #[tokio::test]
    async fn a_failed_status_with_no_body_names_only_the_status() {
        let (url, _server) = serve_once(reply(500, "Internal Server Error", "")).await;
        let exporter = OpenSearchExporter::builder(endpoint_of(&url))
            .basic_auth("u", "p")
            .build();
        let error = exporter
            .export(&record(vec![]))
            .await
            .expect_err("a 500 fails");
        assert_eq!(
            error.to_string(),
            "OpenSearchExporter: index request returned status 500"
        );
    }

    #[tokio::test]
    async fn a_request_that_does_not_finish_is_an_export_error() {
        let (url, _server) = serve_once(Reply::Silent).await;
        let exporter = OpenSearchExporter::builder(endpoint_of(&url))
            .basic_auth("u", "p")
            .timeout_ms(50)
            .build();
        let error = exporter
            .export(&record(vec![]))
            .await
            .expect_err("times out");
        assert_eq!(
            error.to_string(),
            "OpenSearchExporter: request timed out after 50 ms"
        );
    }

    #[tokio::test]
    async fn sigv4_without_a_region_is_an_export_error() {
        let config = SdkConfig::builder()
            .credentials_provider(SharedCredentialsProvider::new(credentials()))
            .build();
        let exporter = with_config(OpenSearchExporter::builder("http://x").build(), config);
        let error = exporter
            .export(&record(vec![]))
            .await
            .expect_err("no region");
        assert_eq!(
            error.to_string(),
            "OpenSearchExporter: no AWS region is configured"
        );
    }

    #[tokio::test]
    async fn sigv4_without_a_credentials_provider_is_an_export_error() {
        let config = SdkConfig::builder()
            .region(Region::new("us-east-1"))
            .build();
        let exporter = with_config(OpenSearchExporter::builder("http://x").build(), config);
        let error = exporter
            .export(&record(vec![]))
            .await
            .expect_err("no provider");
        assert_eq!(
            error.to_string(),
            "OpenSearchExporter: no AWS credentials provider is configured"
        );
    }

    #[derive(Debug)]
    struct FailingCredentials;

    impl ProvideCredentials for FailingCredentials {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            future::ProvideCredentials::ready(Err(CredentialsError::not_loaded("no source")))
        }
    }

    #[tokio::test]
    async fn a_credentials_failure_is_an_export_error() {
        let config = SdkConfig::builder()
            .region(Region::new("us-east-1"))
            .credentials_provider(SharedCredentialsProvider::new(FailingCredentials))
            .build();
        let exporter = with_config(OpenSearchExporter::builder("http://x").build(), config);
        let error = exporter
            .export(&record(vec![]))
            .await
            .expect_err("no credentials");
        let text = error.to_string();
        assert!(
            text.starts_with("OpenSearchExporter: loading credentials failed: "),
            "{text}"
        );
        assert!(text.contains("no source"), "{text}");
    }
}
