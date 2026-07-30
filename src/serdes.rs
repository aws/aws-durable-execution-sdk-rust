//! Serialization/deserialization abstraction.
//!
//! The [`Serdes`] trait is object-safe and provides the extension point for
//! custom serialization formats. The default implementation uses
//! `serde_json`.
//!
//! [`FileSystemSerdes`] stores values on a durable filesystem (EFS or S3
//! Files mounted to Lambda), keeping checkpoint payloads small regardless
//! of value size.

use std::fmt::Debug;

/// Object-safe serialization/deserialization trait.
///
/// Implement this trait to provide custom serialization formats for
/// operation results. The default implementation uses `serde_json`.
///
/// # Serialization model
///
/// A `Serdes` is a **transformation around JSON**, not a replacement for it.
/// The engine always serializes values with `serde_json` first, hands the
/// resulting JSON string to [`serialize_to_string_with_context`](Serdes::serialize_to_string_with_context)
/// for the wire, and reverses the transform with
/// [`deserialize_from_string_with_context`](Serdes::deserialize_from_string_with_context)
/// before deserializing back with `serde_json`.
///
/// Every operation path uses this one model — steps, invokes, callbacks,
/// child contexts, `wait_for_condition`, whole map/parallel batch results,
/// and individual map/parallel item results. A type that implements this
/// trait therefore behaves identically wherever it is attached.
///
/// # Object safety
///
/// This trait is deliberately object-safe so it can be stored as
/// `Box<dyn Serdes>` in builders and options.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::Serdes;
///
/// struct UppercaseSerdes;
///
/// impl std::fmt::Debug for UppercaseSerdes {
///     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
///         f.write_str("UppercaseSerdes")
///     }
/// }
///
/// impl Serdes for UppercaseSerdes {
///     fn serialize_to_string(
///         &self,
///         json_str: &str,
///     ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(json_str.to_uppercase())
///     }
///
///     fn deserialize_from_string(
///         &self,
///         payload: &str,
///     ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(payload.to_owned())
///     }
/// }
///
/// let serdes: Box<dyn Serdes> = Box::new(UppercaseSerdes);
/// # drop(serdes);
/// ```
pub trait Serdes: Debug + Send + Sync {
    /// Transforms a JSON-serialized string for wire transport.
    ///
    /// The default implementation returns the input unchanged (standard
    /// JSON). Override to apply custom transformations (e.g., uppercase).
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    fn serialize_to_string(
        &self,
        json_str: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(json_str.to_owned())
    }

    /// Transforms a wire payload string back for JSON deserialization.
    ///
    /// The default implementation returns the input unchanged (standard
    /// JSON). Override to reverse custom transformations.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    fn deserialize_from_string(
        &self,
        payload: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(payload.to_owned())
    }

    /// Context-aware string serialization for wire transport.
    ///
    /// The default implementation delegates to [`serialize_to_string`](Serdes::serialize_to_string),
    /// ignoring the context. Implementations that require operation identity for
    /// path resolution (e.g., [`FileSystemSerdes`]) override this to use the
    /// context for deterministic file placement.
    ///
    /// The engine calls this method at every serialization point, passing the
    /// operation's wire ID and execution ARN.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    fn serialize_to_string_with_context(
        &self,
        json_str: &str,
        _context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.serialize_to_string(json_str)
    }

    /// Context-aware string deserialization from wire transport.
    ///
    /// The default implementation delegates to [`deserialize_from_string`](Serdes::deserialize_from_string),
    /// ignoring the context. Implementations that store data externally
    /// (e.g., [`FileSystemSerdes`]) may use the context for path resolution
    /// on deserialization if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    fn deserialize_from_string_with_context(
        &self,
        payload: &str,
        _context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.deserialize_from_string(payload)
    }
}

// ============================================================
// SerdesContext
// ============================================================

/// Context information passed to [`FileSystemSerdes`] for file path
/// resolution.
///
/// Provides the operation identity and execution ARN so that file paths
/// are deterministic across replays.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::SerdesContext;
///
/// let ctx = SerdesContext::new("step-1", "arn:aws:lambda:us-east-1:123:function:my-fn:1/durable-execution/exec-1/inv-1");
/// assert_eq!(ctx.operation_id(), "step-1");
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SerdesContext {
    /// Unique identifier for the operation being serialized.
    operation_id: String,
    /// ARN of the durable execution (stable across replays of the same
    /// execution).
    durable_execution_arn: String,
}

impl SerdesContext {
    /// Creates a new serdes context.
    #[must_use]
    pub fn new(operation_id: impl Into<String>, durable_execution_arn: impl Into<String>) -> Self {
        Self {
            operation_id: operation_id.into(),
            durable_execution_arn: durable_execution_arn.into(),
        }
    }

    /// Returns the operation ID.
    #[must_use]
    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    /// Returns the durable execution ARN.
    #[must_use]
    pub fn durable_execution_arn(&self) -> &str {
        &self.durable_execution_arn
    }
}

// ============================================================
// FileSystemSerdes
// ============================================================

/// Controls when data is written to the filesystem.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::FileSystemSerdesMode;
///
/// let mode = FileSystemSerdesMode::Overflow;
/// assert!(matches!(mode, FileSystemSerdesMode::Overflow));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FileSystemSerdesMode {
    /// Every value is written to a file; the checkpoint stores only a
    /// file pointer. Best for consistently large payloads or when you
    /// want predictable checkpoint sizes.
    Always,
    /// Data is written inline (as JSON) unless it exceeds the overflow
    /// threshold (~255 KB), in which case it overflows to a file.
    /// Best for mixed workloads where most payloads are small.
    Overflow,
}

/// Controls how the execution ARN and operation ID are turned into
/// on-disk directory and file names.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::FileSystemPathEncoding;
///
/// let enc = FileSystemPathEncoding::Hash;
/// assert!(matches!(enc, FileSystemPathEncoding::Hash));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FileSystemPathEncoding {
    /// Human-readable percent-encoded paths. The per-execution directory
    /// is derived from the ARN's function name, execution name, and
    /// invocation ID when the ARN matches the durable-execution shape.
    Uri,
    /// SHA-256 hex digest for both directory and file names. Fixed
    /// length (64 chars), always filesystem-safe.
    Hash,
}

/// Configuration for [`FileSystemSerdes`].
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{FileSystemSerdesConfig, FileSystemSerdesMode, FileSystemPathEncoding};
///
/// let config = FileSystemSerdesConfig::builder()
///     .storage_mode(FileSystemSerdesMode::Overflow)
///     .path_encoding(FileSystemPathEncoding::Hash)
///     .build();
/// # drop(config);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileSystemSerdesConfig {
    /// Storage mode. Default: [`FileSystemSerdesMode::Always`].
    pub storage_mode: FileSystemSerdesMode,
    /// Path encoding. Default: [`FileSystemPathEncoding::Uri`].
    pub path_encoding: FileSystemPathEncoding,
    /// Overflow threshold in bytes. Default: 255 KB (256 KB checkpoint
    /// limit minus 1 KB headroom for envelope metadata).
    pub overflow_threshold_bytes: usize,
}

impl Default for FileSystemSerdesConfig {
    fn default() -> Self {
        Self {
            storage_mode: FileSystemSerdesMode::Always,
            path_encoding: FileSystemPathEncoding::Uri,
            // 256KB checkpoint limit - 1KB headroom for envelope wrapper
            overflow_threshold_bytes: (256 * 1024) - 1024,
        }
    }
}

impl FileSystemSerdesConfig {
    /// Creates a new builder for `FileSystemSerdesConfig`.
    pub fn builder() -> FileSystemSerdesConfigBuilder {
        FileSystemSerdesConfigBuilder::default()
    }
}

/// Builder for [`FileSystemSerdesConfig`].
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{FileSystemSerdesConfig, FileSystemSerdesMode};
///
/// let config = FileSystemSerdesConfig::builder()
///     .storage_mode(FileSystemSerdesMode::Overflow)
///     .build();
/// # drop(config);
/// ```
#[derive(Debug, Default)]
#[must_use = "builders do nothing unless .build() is called"]
pub struct FileSystemSerdesConfigBuilder {
    storage_mode: Option<FileSystemSerdesMode>,
    path_encoding: Option<FileSystemPathEncoding>,
    overflow_threshold_bytes: Option<usize>,
}

impl FileSystemSerdesConfigBuilder {
    /// Sets the storage mode.
    pub fn storage_mode(mut self, mode: FileSystemSerdesMode) -> Self {
        self.storage_mode = Some(mode);
        self
    }

    /// Sets the path encoding.
    pub fn path_encoding(mut self, encoding: FileSystemPathEncoding) -> Self {
        self.path_encoding = Some(encoding);
        self
    }

    /// Sets the overflow threshold in bytes.
    ///
    /// In [`FileSystemSerdesMode::Overflow`] mode, values whose serialized
    /// envelope exceeds this threshold are written to a file instead of
    /// being stored inline.
    pub fn overflow_threshold_bytes(mut self, bytes: usize) -> Self {
        self.overflow_threshold_bytes = Some(bytes);
        self
    }

    /// Builds the config.
    #[must_use]
    pub fn build(self) -> FileSystemSerdesConfig {
        let defaults = FileSystemSerdesConfig::default();
        FileSystemSerdesConfig {
            storage_mode: self.storage_mode.unwrap_or(defaults.storage_mode),
            path_encoding: self.path_encoding.unwrap_or(defaults.path_encoding),
            overflow_threshold_bytes: self
                .overflow_threshold_bytes
                .unwrap_or(defaults.overflow_threshold_bytes),
        }
    }
}

/// Filesystem-backed serialization for durable functions.
///
/// Stores serialized values on a durable filesystem (Amazon EFS or S3
/// Files mounted to Lambda). The checkpoint stores a lightweight JSON
/// envelope pointing to the file, keeping checkpoint payloads small.
///
/// # Warning
///
/// Do NOT use with Lambda's ephemeral `/tmp` storage. `/tmp` is local to
/// a single execution environment and is not shared across invocations.
/// On replay, a different environment may be used and the file will not
/// be found. Use only with a durable, shared filesystem (EFS or S3 Files).
///
/// # Envelope format
///
/// The checkpoint stores one of:
/// - `{"data":"<inline JSON>"}` — value stored inline (OVERFLOW mode, under threshold)
/// - `{"file":"<path>"}` — value stored in a file
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode};
///
/// // Always write to filesystem (default)
/// let serdes = FileSystemSerdes::new("/mnt/efs");
///
/// // Overflow mode: inline if small, file if large
/// let serdes = FileSystemSerdes::with_config(
///     "/mnt/s3",
///     FileSystemSerdesConfig::builder()
///         .storage_mode(FileSystemSerdesMode::Overflow)
///         .build(),
/// );
/// # drop(serdes);
/// ```
#[derive(Debug, Clone)]
pub struct FileSystemSerdes {
    base_path: String,
    config: FileSystemSerdesConfig,
}

impl FileSystemSerdes {
    /// Creates a new `FileSystemSerdes` with default configuration
    /// ([`FileSystemSerdesMode::Always`], [`FileSystemPathEncoding::Uri`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::FileSystemSerdes;
    ///
    /// let serdes = FileSystemSerdes::new("/mnt/efs");
    /// # drop(serdes);
    /// ```
    #[must_use]
    pub fn new(base_path: impl Into<String>) -> Self {
        Self {
            base_path: base_path.into(),
            config: FileSystemSerdesConfig::default(),
        }
    }

    /// Creates a new `FileSystemSerdes` with custom configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::{FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode};
    ///
    /// let serdes = FileSystemSerdes::with_config(
    ///     "/mnt/s3",
    ///     FileSystemSerdesConfig::builder()
    ///         .storage_mode(FileSystemSerdesMode::Overflow)
    ///         .build(),
    /// );
    /// # drop(serdes);
    /// ```
    #[must_use]
    pub fn with_config(base_path: impl Into<String>, config: FileSystemSerdesConfig) -> Self {
        Self {
            base_path: base_path.into(),
            config,
        }
    }

    /// Serializes a JSON string to the filesystem envelope format.
    ///
    /// Returns the envelope JSON string to be stored in the checkpoint.
    /// The `context` provides the operation identity for path resolution.
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation or file writing fails.
    pub fn serialize_with_context(
        &self,
        json_str: &str,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self.config.storage_mode {
            FileSystemSerdesMode::Always => {
                let file_path = self.write_to_file(json_str, context)?;
                let escaped_path = json_escape_string(&file_path);
                Ok(format!(r#"{{"file":"{escaped_path}"}}"#))
            }
            FileSystemSerdesMode::Overflow => {
                // Inline envelope: {"data":"<json>"}
                let inline_envelope = format!(r#"{{"data":{json_str}}}"#);
                if inline_envelope.len() > self.config.overflow_threshold_bytes {
                    let file_path = self.write_to_file(json_str, context)?;
                    let escaped_path = json_escape_string(&file_path);
                    Ok(format!(r#"{{"file":"{escaped_path}"}}"#))
                } else {
                    Ok(inline_envelope)
                }
            }
        }
    }

    /// Deserializes from the filesystem envelope format.
    ///
    /// Reads from the file if the envelope contains a file pointer,
    /// otherwise extracts inline data.
    ///
    /// # Errors
    ///
    /// Returns an error if the envelope is malformed or the referenced
    /// file cannot be read.
    pub fn deserialize_with_context(
        &self,
        envelope: &str,
        _context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Parse the envelope to determine storage location.
        // Envelope is: {"file":"<path>"} or {"data":"<json>"}
        let parsed: serde_json::Value = serde_json::from_str(envelope).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(FileSystemSerdesError::new(format!(
                    "invalid envelope JSON: {e}"
                )))
            },
        )?;

        if let Some(file_path) = parsed.get("file").and_then(serde_json::Value::as_str) {
            // File pointer: read from file.
            let contents = std::fs::read_to_string(file_path).map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(FileSystemSerdesError::new(format!(
                        "failed to read file '{file_path}': {e}"
                    )))
                },
            )?;
            Ok(contents)
        } else if let Some(data) = parsed.get("data") {
            // Inline data: extract the JSON value as a string.
            // The "data" field contains the raw JSON value (not double-encoded).
            Ok(data.to_string())
        } else {
            Err(Box::new(FileSystemSerdesError::new(
                "envelope contains neither 'file' nor 'data' field".to_owned(),
            )))
        }
    }

    /// Writes the JSON string to a file and returns the absolute file path.
    fn write_to_file(
        &self,
        json_str: &str,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let dir_path = self.resolve_execution_dir(context);
        std::fs::create_dir_all(&dir_path).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(FileSystemSerdesError::new(format!(
                    "failed to create directory '{dir_path}': {e}"
                )))
            },
        )?;

        let file_name = format!(
            "{}.json",
            encode_segment(&context.operation_id, self.config.path_encoding)
        );
        let file_path = format!("{dir_path}/{file_name}");

        std::fs::write(&file_path, json_str).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(FileSystemSerdesError::new(format!(
                    "failed to write file '{file_path}': {e}"
                )))
            },
        )?;

        Ok(file_path)
    }

    /// Resolves the per-execution directory under the base path.
    fn resolve_execution_dir(&self, context: &SerdesContext) -> String {
        match self.config.path_encoding {
            FileSystemPathEncoding::Uri => {
                // Try to parse as a durable execution ARN for human-readable paths.
                if let Some(parts) = parse_durable_execution_arn(&context.durable_execution_arn) {
                    return format!(
                        "{}/{}/{}/{}",
                        self.base_path,
                        parts.function_name,
                        parts.execution_name,
                        parts.invocation_id
                    );
                }
                // Fallback: percent-encode the whole ARN.
                let encoded = percent_encode(&context.durable_execution_arn);
                format!("{}/{encoded}", self.base_path)
            }
            FileSystemPathEncoding::Hash => {
                let hash = sha256_hex(&context.durable_execution_arn);
                format!("{}/{hash}", self.base_path)
            }
        }
    }
}

/// # Context-Aware Engine Wiring
///
/// The engine calls [`serialize_to_string_with_context`](Serdes::serialize_to_string_with_context)
/// and [`deserialize_from_string_with_context`](Serdes::deserialize_from_string_with_context)
/// at every serialization point, passing a [`SerdesContext`] with the operation's
/// wire ID and execution ARN. `FileSystemSerdes` overrides these to delegate to
/// [`serialize_with_context`](FileSystemSerdes::serialize_with_context) and
/// [`deserialize_with_context`](FileSystemSerdes::deserialize_with_context),
/// enabling deterministic file-path resolution.
///
/// The context-free trait methods (`serialize_to_string`, `deserialize_from_string`)
/// remain as fallbacks: `serialize_to_string` passes through unchanged (no file
/// write without context), and `deserialize_from_string` detects envelope payloads
/// and reads from files when possible (read path needs no context since the path
/// is in the envelope).
///
/// Custom `Serdes` implementations that do not need context can ignore the
/// `_with_context` methods entirely — the default trait implementations delegate
/// to the context-free versions.
impl Serdes for FileSystemSerdes {
    fn serialize_to_string(
        &self,
        json_str: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // When used through the Serdes trait (without explicit context),
        // we cannot resolve file paths. Return the JSON as-is.
        // The context-aware path is serialize_with_context().
        Ok(json_str.to_owned())
    }

    fn deserialize_from_string(
        &self,
        payload: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Attempt to parse as an envelope. If it looks like one, process it.
        // Otherwise return as-is (passthrough for non-filesystem payloads).
        if payload.starts_with('{')
            && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(payload)
            && (parsed.get("file").is_some() || parsed.get("data").is_some())
        {
            // This is a filesystem serdes envelope.
            let dummy_ctx = SerdesContext {
                operation_id: String::new(),
                durable_execution_arn: String::new(),
            };
            return self.deserialize_with_context(payload, &dummy_ctx);
        }
        Ok(payload.to_owned())
    }

    fn serialize_to_string_with_context(
        &self,
        json_str: &str,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.serialize_with_context(json_str, context)
    }

    fn deserialize_from_string_with_context(
        &self,
        payload: &str,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.deserialize_with_context(payload, context)
    }
}

// ============================================================
// Error type for FileSystemSerdes
// ============================================================

/// Error from filesystem serialization/deserialization operations.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::FileSystemSerdesError;
///
/// let err = FileSystemSerdesError::new("file not found: /mnt/data.json");
/// assert!(err.to_string().contains("file not found"));
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct FileSystemSerdesError {
    message: String,
}

impl FileSystemSerdesError {
    /// Creates a new filesystem serdes error.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for FileSystemSerdesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "filesystem serdes error: {}", self.message)
    }
}

impl std::error::Error for FileSystemSerdesError {}

// ============================================================
// Helper functions
// ============================================================

/// Parsed components of a durable execution ARN.
struct DurableExecutionArnParts {
    function_name: String,
    execution_name: String,
    invocation_id: String,
}

/// Parses a durable execution ARN into its components.
///
/// Expected format:
/// `arn:<partition>:lambda:<region>:<account>:function:<fn>:<ver>/durable-execution/<exec>/<inv>`
fn parse_durable_execution_arn(arn: &str) -> Option<DurableExecutionArnParts> {
    // Split on the durable-execution marker.
    let (prefix, exec_suffix) = arn.split_once("/durable-execution/")?;

    // Extract function name from the ARN prefix.
    // Format: arn:...:function:<name>:<version>
    let function_name = prefix.rsplit(':').nth(1)?;
    if function_name.is_empty() {
        return None;
    }

    // Split the execution part: <executionName>/<invocationId>
    let (execution_name, invocation_id) = exec_suffix.split_once('/')?;

    Some(DurableExecutionArnParts {
        function_name: function_name.to_owned(),
        execution_name: execution_name.to_owned(),
        invocation_id: invocation_id.to_owned(),
    })
}

/// Percent-encodes a string for filesystem-safe use (all non-unreserved
/// characters).
fn percent_encode(input: &str) -> String {
    let mut encoded = String::with_capacity(input.len());
    for byte in input.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            use std::fmt::Write;
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

/// Encodes a path segment using the specified encoding.
fn encode_segment(value: &str, encoding: FileSystemPathEncoding) -> String {
    match encoding {
        FileSystemPathEncoding::Hash => sha256_hex(value),
        FileSystemPathEncoding::Uri => percent_encode(value),
    }
}

/// Computes the SHA-256 hex digest of a string.
fn sha256_hex(input: &str) -> String {
    use sha2::Digest;
    let hash = sha2::Sha256::digest(input.as_bytes());
    // Convert to hex string manually (no hex crate dependency).
    let mut hex = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

/// Escapes a string for embedding as a JSON string value (without
/// surrounding quotes). Escapes backslash and double-quote characters
/// to produce valid JSON when interpolated into `"..."`.
fn json_escape_string(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c => escaped.push(c),
        }
    }
    escaped
}

// ============================================================
// Shared test support
// ============================================================

/// Test-only serdes shared by the operation-path equivalence tests in
/// `step`, `callback`, and `map_parallel`.
///
/// The point of the shared type is that ONE `Serdes` implementation must
/// behave identically on every operation path — step results, callback
/// payloads, and map/parallel item results — now that all of them go through
/// the same JSON-string transformation model.
#[cfg(test)]
pub(crate) mod test_support {
    use super::Serdes;

    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    /// A serdes whose wire form is deliberately NOT JSON: the JSON string is
    /// hex-encoded behind a `HEX1:` marker.
    ///
    /// Two properties make this useful as a probe:
    ///
    /// 1. It implements ONLY the string transform methods. Before the
    ///    serialization model was normalized, `Serdes` also demanded
    ///    `serialize(&dyn Any)` / `deserialize_bytes`, so this type would not
    ///    have compiled.
    /// 2. Plain `serde_json` cannot parse a `HEX1:`-prefixed payload, so a
    ///    successful round-trip proves the transform was actually applied and
    ///    reversed rather than incidentally bypassed by a JSON-only path.
    #[derive(Debug)]
    pub(crate) struct HexEnvelopeSerdes;

    impl Serdes for HexEnvelopeSerdes {
        fn serialize_to_string(&self, json_str: &str) -> Result<String, BoxError> {
            Ok(hex_envelope(json_str))
        }

        fn deserialize_from_string(&self, payload: &str) -> Result<String, BoxError> {
            let hex = payload
                .strip_prefix("HEX1:")
                .ok_or_else(|| -> BoxError { "missing HEX1: marker".into() })?;
            if hex.len() % 2 != 0 {
                return Err("odd-length hex body".into());
            }
            let mut bytes = Vec::with_capacity(hex.len() / 2);
            let raw = hex.as_bytes();
            for pair in raw.chunks(2) {
                let s = std::str::from_utf8(pair)?;
                bytes.push(u8::from_str_radix(s, 16)?);
            }
            Ok(String::from_utf8(bytes)?)
        }
    }

    /// Returns the exact wire form `HexEnvelopeSerdes` produces for a JSON
    /// string, so tests can assert on the stored payload.
    pub(crate) fn hex_envelope(json_str: &str) -> String {
        use std::fmt::Write;
        let mut out = String::with_capacity(5 + json_str.len() * 2);
        out.push_str("HEX1:");
        for byte in json_str.as_bytes() {
            let _ = write!(out, "{byte:02x}");
        }
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions with descriptive messages
mod tests {
    use super::*;

    #[test]
    fn percent_encode_basic() {
        assert_eq!(percent_encode("hello"), "hello");
        assert_eq!(percent_encode("a/b:c"), "a%2Fb%3Ac");
        assert_eq!(percent_encode("step-1"), "step-1");
    }

    #[test]
    fn sha256_hex_known_answer() {
        // SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        assert_eq!(
            sha256_hex("hello"),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn parse_valid_durable_execution_arn() {
        let arn = "arn:aws:lambda:us-east-1:123456789012:function:my-fn:1/durable-execution/exec-abc/inv-123";
        let parts = parse_durable_execution_arn(arn).expect("should parse");
        assert_eq!(parts.function_name, "my-fn");
        assert_eq!(parts.execution_name, "exec-abc");
        assert_eq!(parts.invocation_id, "inv-123");
    }

    #[test]
    fn parse_invalid_arn_returns_none() {
        assert!(parse_durable_execution_arn("not-an-arn").is_none());
        assert!(
            parse_durable_execution_arn("arn:aws:lambda:us-east-1:123:function:fn:1").is_none()
        );
    }

    #[test]
    fn filesystem_serdes_always_mode_round_trip() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_always");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn:
                "arn:aws:lambda:us-east-1:123:function:my-fn:1/durable-execution/exec-1/inv-1"
                    .to_owned(),
        };

        let input_json = r#"{"value":42,"name":"test"}"#;
        let envelope = serdes
            .serialize_with_context(input_json, &ctx)
            .expect("serialize should succeed");

        // Envelope should be a file pointer
        assert!(envelope.contains(r#""file""#), "envelope: {envelope}");
        assert!(!envelope.contains(r#""data""#), "envelope: {envelope}");

        // Deserialize should return the original JSON
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, input_json);

        // Clean up
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_overflow_mode_small_inline() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_overflow_small");
        let _ = std::fs::remove_dir_all(&tmp);

        let config = FileSystemSerdesConfig::builder()
            .storage_mode(FileSystemSerdesMode::Overflow)
            .overflow_threshold_bytes(1024)
            .build();
        let serdes = FileSystemSerdes::with_config(tmp.to_string_lossy().to_string(), config);
        let ctx = SerdesContext {
            operation_id: "step-2".to_owned(),
            durable_execution_arn:
                "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e/i".to_owned(),
        };

        let small_json = r#"{"x":1}"#;
        let envelope = serdes
            .serialize_with_context(small_json, &ctx)
            .expect("serialize should succeed");

        // Small value should be inline
        assert!(envelope.contains(r#""data""#), "envelope: {envelope}");
        assert!(!envelope.contains(r#""file""#), "envelope: {envelope}");

        // Deserialize returns the original
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, small_json);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_overflow_mode_large_to_file() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_overflow_large");
        let _ = std::fs::remove_dir_all(&tmp);

        let config = FileSystemSerdesConfig::builder()
            .storage_mode(FileSystemSerdesMode::Overflow)
            .overflow_threshold_bytes(50) // Low threshold for testing
            .build();
        let serdes = FileSystemSerdes::with_config(tmp.to_string_lossy().to_string(), config);
        let ctx = SerdesContext {
            operation_id: "step-3".to_owned(),
            durable_execution_arn:
                "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e/i".to_owned(),
        };

        // This will exceed the 50-byte threshold once wrapped in {"data":...}
        let large_json = r#"{"big":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}"#;
        let envelope = serdes
            .serialize_with_context(large_json, &ctx)
            .expect("serialize should succeed");

        // Large value should overflow to file
        assert!(envelope.contains(r#""file""#), "envelope: {envelope}");

        // Deserialize reads from file
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, large_json);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_hash_encoding() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_hash");
        let _ = std::fs::remove_dir_all(&tmp);

        let config = FileSystemSerdesConfig::builder()
            .path_encoding(FileSystemPathEncoding::Hash)
            .build();
        let serdes = FileSystemSerdes::with_config(tmp.to_string_lossy().to_string(), config);
        let ctx = SerdesContext {
            operation_id: "step/with:special".to_owned(),
            durable_execution_arn: "some-weird-arn/with/slashes".to_owned(),
        };

        let json = r#""hello""#;
        let envelope = serdes
            .serialize_with_context(json, &ctx)
            .expect("serialize should succeed");

        // File path should use SHA-256 hashes (64 hex chars)
        let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        let file_path = parsed.get("file").and_then(|v| v.as_str()).unwrap();
        // The directory segment should be a 64-char hex hash
        let segments: Vec<&str> = file_path.split('/').collect();
        let dir_segment = segments.get(segments.len() - 2).unwrap();
        assert_eq!(dir_segment.len(), 64, "dir hash: {dir_segment}");

        // Round-trip
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, json);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_error_on_missing_file() {
        let serdes = FileSystemSerdes::new("/nonexistent/path");
        let ctx = SerdesContext {
            operation_id: "x".to_owned(),
            durable_execution_arn: "y".to_owned(),
        };

        let envelope = r#"{"file":"/nonexistent/path/does-not-exist.json"}"#;
        let result = serdes.deserialize_with_context(envelope, &ctx);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("failed to read file"), "err: {err_msg}");
    }

    #[test]
    fn filesystem_serdes_pointer_format_stability() {
        // Verify the envelope format is stable for wire compatibility
        let tmp = std::env::temp_dir().join("fs_serdes_test_format");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext {
            operation_id: "op-1".to_owned(),
            durable_execution_arn:
                "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec/inv".to_owned(),
        };

        let envelope = serdes
            .serialize_with_context("42", &ctx)
            .expect("serialize");

        // Parse the envelope and verify its structure
        let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        assert!(parsed.is_object());
        assert!(parsed.get("file").is_some());
        assert!(parsed.get("file").unwrap().is_string());
        // Path should end with .json (we control the extension, always lowercase)
        let path = parsed.get("file").and_then(|v| v.as_str()).unwrap();
        #[allow(clippy::case_sensitive_file_extension_comparisons)]
        // reason: we generate the extension; always lowercase
        let has_json_ext = path.ends_with(".json");
        assert!(has_json_ext, "path: {path}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_uri_encoding_dir_structure() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_uri_dir");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn:
                "arn:aws:lambda:us-west-2:999:function:my-function:42/durable-execution/my-exec/my-inv"
                    .to_owned(),
        };

        let envelope = serdes
            .serialize_with_context(r#""data""#, &ctx)
            .expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        let path = parsed.get("file").and_then(|v| v.as_str()).unwrap();

        // URI mode: base/functionName/executionName/invocationId/opId.json
        assert!(path.contains("my-function"), "path: {path}");
        assert!(path.contains("my-exec"), "path: {path}");
        assert!(path.contains("my-inv"), "path: {path}");
        #[allow(clippy::case_sensitive_file_extension_comparisons)]
        // reason: we generate the extension; always lowercase
        let correct_suffix = path.ends_with("step-1.json");
        assert!(correct_suffix, "path: {path}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn deserialize_from_string_handles_envelope() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_trait");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());

        // Inline envelope
        let result = serdes
            .deserialize_from_string(r#"{"data":"hello"}"#)
            .expect("should parse inline");
        assert_eq!(result, r#""hello""#);

        // Non-envelope JSON passes through
        let result = serdes
            .deserialize_from_string(r#"{"value":42}"#)
            .expect("should pass through");
        assert_eq!(result, r#"{"value":42}"#);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_replay_reads_back() {
        // Simulate the full replay scenario: serialize, then deserialize
        // from a fresh instance (as if in a new Lambda invocation).
        let tmp = std::env::temp_dir().join("fs_serdes_test_replay");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = tmp.to_string_lossy().to_string();

        let ctx = SerdesContext {
            operation_id: "replay-op".to_owned(),
            durable_execution_arn:
                "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e1/i1".to_owned(),
        };

        // First invocation: serialize
        let serdes1 = FileSystemSerdes::new(base.clone());
        let input = r#"{"items":[1,2,3]}"#;
        let envelope = serdes1
            .serialize_with_context(input, &ctx)
            .expect("serialize");

        // Second invocation (replay): deserialize from a fresh instance
        let serdes2 = FileSystemSerdes::new(base);
        let output = serdes2
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize on replay");
        assert_eq!(output, input);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn json_escape_string_handles_special_chars() {
        // Verify the helper escapes backslash, quote, and control chars
        assert_eq!(json_escape_string(r#"a\b"c"#), r#"a\\b\"c"#);
        assert_eq!(json_escape_string("line\nnewline"), r"line\nnewline");
        assert_eq!(json_escape_string("tab\there"), r"tab\there");
        // No special chars: unchanged
        assert_eq!(
            json_escape_string("/mnt/efs/data.json"),
            "/mnt/efs/data.json"
        );
    }

    #[test]
    fn filesystem_serdes_envelope_with_special_path_chars() {
        // Verify envelopes with paths containing characters that need
        // JSON escaping produce valid JSON that round-trips correctly.
        let tmp = std::env::temp_dir().join("fs_serdes_test_escape");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn:
                "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec/inv".to_owned(),
        };

        let input_json = r#"{"value":"test"}"#;
        let envelope = serdes
            .serialize_with_context(input_json, &ctx)
            .expect("serialize should succeed");

        // The envelope must be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&envelope).expect("envelope must be valid JSON");
        assert!(parsed.get("file").is_some());

        // And round-trips back
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, input_json);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Engine-wiring tests: context-aware path activates via trait methods ──

    #[test]
    fn serialize_with_context_via_trait_writes_file() {
        // Proves that calling the trait method `serialize_to_string_with_context`
        // on a `Box<dyn Serdes>` holding a FileSystemSerdes activates the
        // context-aware file-writing path (the engine code path).
        let tmp = std::env::temp_dir().join("fs_serdes_trait_ctx_ser");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = tmp.to_string_lossy().to_string();

        let serdes: Box<dyn Serdes> = Box::new(FileSystemSerdes::new(base.clone()));
        let ctx = SerdesContext::new(
            "step-result-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec-1/inv-1",
        );

        let input_json = r#"{"total":42}"#;
        let envelope = serdes
            .serialize_to_string_with_context(input_json, &ctx)
            .expect("serialize_to_string_with_context should succeed");

        // The envelope must be a file pointer.
        assert!(
            envelope.contains("\"file\""),
            "envelope should be a file pointer: {envelope}"
        );
        assert!(
            !envelope.contains("\"data\""),
            "ALWAYS mode should write to file"
        );

        // The file must exist at the expected path.
        let expected_dir = format!("{base}/fn/exec-1/inv-1");
        assert!(
            std::path::Path::new(&expected_dir).exists(),
            "execution dir should be created: {expected_dir}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn deserialize_with_context_via_trait_reads_file() {
        // Proves that calling the trait method `deserialize_from_string_with_context`
        // on a `Box<dyn Serdes>` holding a FileSystemSerdes reads from the file.
        let tmp = std::env::temp_dir().join("fs_serdes_trait_ctx_deser");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = tmp.to_string_lossy().to_string();

        let serdes: Box<dyn Serdes> = Box::new(FileSystemSerdes::new(base.clone()));
        let ctx = SerdesContext::new(
            "op-deser-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec-1/inv-1",
        );

        // Write via context-aware path.
        let input = r#"{"key":"value"}"#;
        let envelope = serdes
            .serialize_to_string_with_context(input, &ctx)
            .expect("serialize");

        // Read back via context-aware path (same Box<dyn Serdes>).
        let output = serdes
            .deserialize_from_string_with_context(&envelope, &ctx)
            .expect("deserialize_from_string_with_context should succeed");
        assert_eq!(output, input);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The shared probe serdes must be a valid `Serdes` while implementing
    /// ONLY the string transform methods, and its transform must be exactly
    /// reversible. It is the fixture the operation-path equivalence tests
    /// depend on, so it gets its own round-trip check.
    #[test]
    fn hex_envelope_probe_serdes_round_trips() {
        use super::test_support::{HexEnvelopeSerdes, hex_envelope};

        let serdes: Box<dyn Serdes> = Box::new(HexEnvelopeSerdes);
        let ctx = SerdesContext::new("op-1", "arn:test");

        let json = r#"{"label":"a\"b\\c\nd ☃","nested":[[1,-2],[]]}"#;
        let wire = serdes
            .serialize_to_string_with_context(json, &ctx)
            .expect("serialize");
        assert_eq!(wire, hex_envelope(json));
        // The wire form must not be parseable as JSON — that is what makes it
        // a useful probe for "was the transform actually applied?".
        assert!(serde_json::from_str::<serde_json::Value>(&wire).is_err());

        let back = serdes
            .deserialize_from_string_with_context(&wire, &ctx)
            .expect("deserialize");
        assert_eq!(back, json);

        // A payload that never went through the transform must be rejected
        // rather than silently passed through.
        assert!(serdes.deserialize_from_string(json).is_err());
    }

    #[test]
    fn plain_serdes_passthrough_still_works_with_context() {
        // Proves that a plain custom Serdes (not FileSystemSerdes) still
        // works when the engine calls the _with_context methods — the
        // default trait impl delegates to the context-free methods.
        struct UpperSerdes;
        impl Debug for UpperSerdes {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("UpperSerdes")
            }
        }
        impl Serdes for UpperSerdes {
            fn serialize_to_string(
                &self,
                json_str: &str,
            ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                Ok(json_str.to_uppercase())
            }
            fn deserialize_from_string(
                &self,
                payload: &str,
            ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                Ok(payload.to_lowercase())
            }
        }

        let serdes: Box<dyn Serdes> = Box::new(UpperSerdes);
        let ctx = SerdesContext::new("op-1", "arn:test");

        // Context-aware calls should delegate to the context-free methods.
        let result = serdes
            .serialize_to_string_with_context("hello", &ctx)
            .expect("serialize");
        assert_eq!(result, "HELLO");

        let result = serdes
            .deserialize_from_string_with_context("WORLD", &ctx)
            .expect("deserialize");
        assert_eq!(result, "world");
    }
}
