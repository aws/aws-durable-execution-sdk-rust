//! Sends one HTTP request and returns the status and the response body.
//!
//! An exporter that sends HTTP requests uses [`HttpSender`]. The module is
//! compiled when the `http-client` feature is on. Every feature that
//! enables an HTTP exporter lists `http-client`.

use std::error::Error;
use std::sync::OnceLock;
use std::time::Duration;

use aws_durable_execution_sdk::BoxError;
use aws_smithy_http_client::tls::rustls_provider::CryptoMode;
use aws_smithy_http_client::{Connector, tls};
use aws_smithy_runtime_api::client::http::HttpConnector;
use aws_smithy_runtime_api::client::orchestrator::HttpRequest;
use aws_smithy_types::body::SdkBody;
use aws_smithy_types::byte_stream::ByteStream;

/// The status and the body text of an HTTP response.
#[derive(Debug)]
pub(crate) struct HttpResponse {
    /// The status code, for example 200.
    pub(crate) status: u16,
    /// The body. Bytes that are not UTF-8 are replaced with U+FFFD.
    pub(crate) body: String,
}

impl HttpResponse {
    /// Whether the status is in the range 200 to 299.
    pub(crate) fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// Sends HTTP requests over one connection pool.
///
/// The pool is built on the first request. So building a sender needs no
/// Tokio runtime. Requests to `https` URLs use `rustls`.
#[derive(Debug)]
pub(crate) struct HttpSender {
    connector: OnceLock<Connector>,
    timeout: Duration,
}

impl HttpSender {
    /// A sender whose requests, from connecting to reading the whole
    /// response body, fail after `timeout`.
    pub(crate) fn new(timeout: Duration) -> Self {
        Self {
            connector: OnceLock::new(),
            timeout,
        }
    }

    /// The timeout of each request.
    #[cfg(test)]
    pub(crate) fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The connector, built on first use.
    fn connector(&self) -> &Connector {
        self.connector.get_or_init(|| {
            Connector::builder()
                .tls_provider(tls::Provider::Rustls(CryptoMode::AwsLc))
                .build()
        })
    }

    /// Sends `body` to `url` with `method` and `headers`.
    ///
    /// A later entry of `headers` replaces an earlier entry with the same
    /// name, ignoring case. A response with any status is a success of this
    /// function. The caller checks [`HttpResponse::is_success`].
    ///
    /// # Errors
    ///
    /// Returns an error when `method`, `url` or a header is invalid, when
    /// the request fails, or when it takes longer than the timeout.
    pub(crate) async fn send(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: String,
    ) -> Result<HttpResponse, BoxError> {
        let mut request = HttpRequest::new(SdkBody::from(body));
        request
            .set_method(method)
            .map_err(|error| format!("invalid HTTP method {method:?}: {error}"))?;
        request
            .set_uri(url)
            .map_err(|error| format!("invalid URL {url:?}: {error}"))?;
        for (name, value) in headers {
            request
                .headers_mut()
                .try_insert(name.clone(), value.clone())
                .map_err(|error| format!("invalid header {name:?}: {error}"))?;
        }
        let timeout = self.timeout;
        tokio::time::timeout(timeout, self.round_trip(request))
            .await
            .map_err(|_| format!("request timed out after {} ms", timeout.as_millis()))?
    }

    /// Sends `request` and reads the whole response body.
    async fn round_trip(&self, request: HttpRequest) -> Result<HttpResponse, BoxError> {
        let response = self
            .connector()
            .call(request)
            .await
            .map_err(|error| error_chain("request failed", &error))?;
        let status = response.status().as_u16();
        let bytes = ByteStream::new(response.into_body())
            .collect()
            .await
            .map_err(|error| error_chain("reading the response failed", &error))?
            .into_bytes();
        Ok(HttpResponse {
            status,
            body: String::from_utf8_lossy(&bytes).into_owned(),
        })
    }
}

/// `context` followed by `error` and each of its sources.
///
/// The `Display` text of a connector error is a fixed phrase such as "other
/// error". The cause is in the sources. The scheduler logs only the
/// `Display` text of an export error. So the text here includes the sources.
fn error_chain(context: &str, error: &(dyn Error + 'static)) -> BoxError {
    let mut text = format!("{context}: {error}");
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text.into()
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "a test fails by panicking, and the test server runs on a spawned task"
)]
pub(crate) mod tests {
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    /// How a [`serve_once`] server answers.
    pub(crate) enum Reply {
        /// Writes this raw HTTP response, then closes the connection.
        Raw(String),
        /// Reads the request and never answers.
        Silent,
    }

    /// A `200 OK` response with `body`.
    pub(crate) fn ok(body: &str) -> Reply {
        reply(200, "OK", body)
    }

    /// A response with the given status line and `body`.
    pub(crate) fn reply(status: u16, reason: &str, body: &str) -> Reply {
        Reply::Raw(format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ))
    }

    /// Serves one connection on a local port. Returns the URL of the server
    /// and the raw text of the request it received, once the exchange ends.
    pub(crate) async fn serve_once(reply: Reply) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("http://{}/path?q=1", listener.local_addr().expect("addr"));
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut received = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).await.expect("read");
                received.extend_from_slice(buffer.get(..read).expect("in range"));
                if read == 0 || request_complete(&received) {
                    break;
                }
            }
            match reply {
                Reply::Raw(text) => {
                    stream.write_all(text.as_bytes()).await.expect("write");
                    stream.shutdown().await.expect("shutdown");
                }
                Reply::Silent => std::future::pending::<()>().await,
            }
            String::from_utf8(received).expect("utf8 request")
        });
        (url, task)
    }

    /// Whether `received` holds the whole head and the whole body.
    fn request_complete(received: &[u8]) -> bool {
        let text = String::from_utf8_lossy(received);
        let Some((head, body)) = text.split_once("\r\n\r\n") else {
            return false;
        };
        let length = head
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0);
        body.len() >= length
    }

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[tokio::test]
    async fn sends_method_url_headers_and_body_and_returns_status_and_text() {
        let (url, server) = serve_once(ok("accepted")).await;
        let sender = HttpSender::new(Duration::from_secs(5));
        let response = sender
            .send(
                "PUT",
                &url,
                &headers(&[("X-Key", "abc")]),
                "{\"a\":1}".to_owned(),
            )
            .await
            .expect("sent");
        assert_eq!(response.status, 200);
        assert_eq!(response.body, "accepted");
        assert!(response.is_success());
        let request = server.await.expect("server");
        assert!(
            request.starts_with("PUT /path?q=1 HTTP/1.1\r\n"),
            "{request}"
        );
        assert!(
            request.to_lowercase().contains("\r\nx-key: abc\r\n"),
            "{request}"
        );
        assert!(request.ends_with("{\"a\":1}"), "{request}");
    }

    #[tokio::test]
    async fn a_later_header_replaces_an_earlier_one_ignoring_case() {
        let (url, server) = serve_once(ok("")).await;
        let sender = HttpSender::new(Duration::from_secs(5));
        let pairs = headers(&[
            ("Content-Type", "application/json"),
            ("content-type", "text/x"),
        ]);
        sender
            .send("POST", &url, &pairs, String::new())
            .await
            .expect("sent");
        let request = server.await.expect("server").to_lowercase();
        assert!(request.contains("content-type: text/x\r\n"), "{request}");
        assert!(!request.contains("application/json"), "{request}");
    }

    #[tokio::test]
    async fn reports_a_non_success_status_without_failing() {
        let (url, _server) = serve_once(reply(503, "Service Unavailable", "busy")).await;
        let sender = HttpSender::new(Duration::from_secs(5));
        let response = sender
            .send("POST", &url, &[], String::new())
            .await
            .expect("sent");
        assert_eq!(response.status, 503);
        assert_eq!(response.body, "busy");
        assert!(!response.is_success());
    }

    #[tokio::test]
    async fn times_out_when_the_server_does_not_answer() {
        let (url, _server) = serve_once(Reply::Silent).await;
        let sender = HttpSender::new(Duration::from_millis(100));
        let error = sender
            .send("POST", &url, &[], String::new())
            .await
            .unwrap_err();
        assert_eq!(error.to_string(), "request timed out after 100 ms");
    }

    #[tokio::test]
    async fn names_the_cause_of_a_connection_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("http://{}/", listener.local_addr().expect("addr"));
        drop(listener);
        let sender = HttpSender::new(Duration::from_secs(5));
        let error = sender
            .send("POST", &url, &[], String::new())
            .await
            .unwrap_err();
        let text = error.to_string();
        assert!(text.starts_with("request failed: "), "{text}");
        assert!(text.matches(": ").count() >= 2, "no cause in {text}");
    }

    #[tokio::test]
    async fn rejects_an_invalid_method_url_or_header_name() {
        let sender = HttpSender::new(Duration::from_secs(5));
        let url = "http://127.0.0.1:1/";
        let method = sender.send("NOT A METHOD", url, &[], String::new()).await;
        assert!(
            method
                .unwrap_err()
                .to_string()
                .starts_with("invalid HTTP method")
        );
        let uri = sender
            .send("POST", "http://exa mple/", &[], String::new())
            .await;
        assert!(uri.unwrap_err().to_string().starts_with("invalid URL"));
        let header = sender
            .send("POST", url, &headers(&[("bad name", "v")]), String::new())
            .await;
        assert!(
            header
                .unwrap_err()
                .to_string()
                .starts_with("invalid header \"bad name\"")
        );
    }

    #[tokio::test]
    async fn an_https_request_to_a_plain_http_server_fails_with_a_tls_error() {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("https://{}/", listener.local_addr().expect("addr"));
        // Answers the TLS client hello with plain text, then closes.
        let _server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut buffer = [0_u8; 4096];
            let _bytes = stream.read(&mut buffer).await.expect("read");
            stream
                .write_all(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                .await
                .expect("write");
            stream.shutdown().await.expect("shutdown");
        });
        let sender = HttpSender::new(Duration::from_secs(5));
        let error = sender
            .send("POST", &url, &[], String::new())
            .await
            .unwrap_err();
        assert!(error.to_string().starts_with("request failed"), "{error}");
    }

    #[test]
    fn building_a_sender_needs_no_runtime() {
        let sender = Arc::new(HttpSender::new(Duration::from_secs(1)));
        assert!(sender.connector.get().is_none());
    }
}
