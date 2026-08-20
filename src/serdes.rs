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
/// A `Serdes` sits between the SDK's typed values and the checkpoint wire
/// format. On the way out it receives the operation's value **erased to
/// [`serde_json::Value`]** and returns the string to store on the wire; on
/// the way back it receives that string and returns a [`serde_json::Value`],
/// which the SDK deserializes into the target type.
///
/// This mirrors the Go, JavaScript, and Java SDKs, whose custom serdes also
/// receive the value rather than pre-rendered JSON text.
///
/// # One rule on every path
///
/// Steps, invoke payloads, invoke results, callback payloads, child-context
/// results, `wait_for_condition` state, individual map/parallel item results,
/// and whole map/parallel batch results all hand the serdes the *same* shape:
/// the [`serde_json::Value`] of the value being serialized. A `String` result
/// of `X` arrives as `Value::String("X")` on every path — there is no
/// per-path quoting rule to compensate for, so a type that implements this
/// trait behaves identically wherever it is attached.
///
/// ```
/// use aws_durable_execution_sdk_rust::serdes::{Serdes, SerdesContext};
/// use serde_json::Value;
///
/// #[derive(Debug)]
/// struct WrapSerdes;
///
/// impl Serdes for WrapSerdes {
///     fn serialize(
///         &self,
///         value: &Value,
///         _context: &SerdesContext,
///     ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
///         // The value arrives typed — a string result is a `Value::String`,
///         // so there is no JSON quoting to strip first.
///         let raw = value.as_str().unwrap_or_default();
///         Ok(format!("wrapped:{raw}"))
///     }
///
///     fn deserialize(
///         &self,
///         data: &str,
///         _context: &SerdesContext,
///     ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
///         let raw = data.strip_prefix("wrapped:").unwrap_or(data);
///         Ok(Value::String(raw.to_owned()))
///     }
/// }
///
/// let context = SerdesContext::new("step-1", "arn:test");
/// let wire = WrapSerdes.serialize(&Value::String("X".to_owned()), &context)?;
/// assert_eq!(wire, "wrapped:X");
/// assert_eq!(
///     WrapSerdes.deserialize(&wire, &context)?,
///     Value::String("X".to_owned()),
/// );
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
///
/// # Object safety
///
/// This trait is deliberately object-safe so it can be stored as
/// `Box<dyn Serdes>` in builders and options.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::serdes::{Serdes, SerdesContext};
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
///     fn serialize(
///         &self,
///         value: &serde_json::Value,
///         _context: &SerdesContext,
///     ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(value.to_string().to_uppercase())
///     }
///
///     fn deserialize(
///         &self,
///         data: &str,
///         _context: &SerdesContext,
///     ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(serde_json::from_str(&data.to_lowercase())?)
///     }
/// }
///
/// let serdes: Box<dyn Serdes> = Box::new(UppercaseSerdes);
/// # drop(serdes);
/// ```
pub trait Serdes: Debug + Send + Sync {
    /// Serializes a structured value to the string stored on the wire.
    ///
    /// `value` is the operation's value erased to [`serde_json::Value`]. The
    /// default implementation renders compact JSON via `value.to_string()`.
    /// Note that this may differ from what `serde_json::to_string` would
    /// produce on the original typed value — struct field order may change
    /// (without `preserve_order`), 128-bit integers outside i64/u64 range
    /// cannot be represented in `Value`, and duplicate keys collapse.
    ///
    /// `context` carries the operation's wire ID and the execution ARN.
    /// Implementations that store data externally (e.g.
    /// [`FileSystemSerdes`]) use it for deterministic path resolution;
    /// implementations that do not need it can ignore it.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        Ok(value.to_string())
    }

    /// Deserializes a wire string back into a structured value.
    ///
    /// `data` is the string a previous [`serialize`](Serdes::serialize) call
    /// returned (or, for callbacks, the payload an external caller delivered).
    /// The returned [`serde_json::Value`] is deserialized into the target type
    /// by the SDK, so no runtime downcast is involved.
    ///
    /// The default implementation parses `data` as JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if the transformation fails.
    fn deserialize(
        &self,
        data: &str,
        _context: &SerdesContext,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        Ok(serde_json::from_str(data)?)
    }
}

// ============================================================
// Off-runtime invocation helpers
// ============================================================

/// Runs a custom serdes `serialize` on the blocking thread pool.
///
/// The `Serdes` trait is sync, but implementations like [`FileSystemSerdes`]
/// perform filesystem I/O that can stall for arbitrarily long on network
/// mounts (EFS, S3 Files). Every async call site in the SDK routes custom
/// serdes invocations through `tokio::task::spawn_blocking` so that a slow
/// serialize never blocks an executor thread.
///
/// A `JoinError` (the blocking task panicked or was cancelled at runtime
/// shutdown) is mapped to an ordinary error — it never panics the caller.
pub(crate) async fn serialize_off_runtime(
    serdes: &std::sync::Arc<dyn Serdes>,
    value: serde_json::Value,
    serdes_ctx: &SerdesContext,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let serdes = std::sync::Arc::clone(serdes);
    let serdes_ctx = serdes_ctx.clone();
    tokio::task::spawn_blocking(move || serdes.serialize(&value, &serdes_ctx))
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("serdes serialize task did not complete: {e}").into()
        })?
}

/// Runs a custom serdes `deserialize` on the blocking thread pool.
///
/// See [`serialize_off_runtime`] for why: sync serdes implementations may
/// block on filesystem I/O, which must not run on the async executor.
///
/// A `JoinError` (the blocking task panicked or was cancelled at runtime
/// shutdown) is mapped to an ordinary error — it never panics the caller.
pub(crate) async fn deserialize_off_runtime(
    serdes: &std::sync::Arc<dyn Serdes>,
    data: String,
    serdes_ctx: &SerdesContext,
) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
    let serdes = std::sync::Arc::clone(serdes);
    let serdes_ctx = serdes_ctx.clone();
    tokio::task::spawn_blocking(move || serdes.deserialize(&data, &serdes_ctx))
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("serdes deserialize task did not complete: {e}").into()
        })?
}

/// A value prepared for the wire: either already-rendered JSON (no custom
/// serdes) or the erased [`serde_json::Value`] awaiting a custom serdes.
///
/// The two-phase split exists for `Send` reasons: preparation borrows the
/// typed `&O` value and runs synchronously, so the borrow ends before any
/// `.await`. The async completion phase ([`PreparedValue::into_wire`]) then
/// owns everything it touches — otherwise every operation future would
/// require `O: Sync` just to hold `&O` across the `spawn_blocking` await.
pub(crate) enum PreparedValue<'a> {
    /// Default path: the wire string is already rendered by `serde_json`.
    Wire(String),
    /// Custom path: the erased value plus the serdes that will render it.
    Erased(serde_json::Value, &'a std::sync::Arc<dyn Serdes>),
}

/// Prepares a typed value for the wire, consuming the `&O` borrow now.
///
/// With no serdes this renders compact JSON directly from the typed value
/// (`serde_json::to_string`), preserving the historical default-path bytes.
/// With a custom serdes it erases to [`serde_json::Value`], to be completed
/// off-runtime by [`PreparedValue::into_wire`].
pub(crate) fn prepare_value<'a, O: serde::Serialize>(
    serdes: Option<&'a std::sync::Arc<dyn Serdes>>,
    value: &O,
) -> Result<PreparedValue<'a>, serde_json::Error> {
    match serdes {
        None => serde_json::to_string(value).map(PreparedValue::Wire),
        Some(s) => serde_json::to_value(value).map(|v| PreparedValue::Erased(v, s)),
    }
}

impl PreparedValue<'_> {
    /// Completes preparation: returns the wire string, invoking a custom
    /// serdes on the blocking thread pool when one is attached.
    pub(crate) async fn into_wire(
        self,
        serdes_ctx: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Wire(wire) => Ok(wire),
            Self::Erased(value, serdes) => serialize_off_runtime(serdes, value, serdes_ctx).await,
        }
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
/// use aws_durable_execution_sdk_rust::serdes::SerdesContext;
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
/// use aws_durable_execution_sdk_rust::serdes::FileSystemSerdesMode;
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
/// use aws_durable_execution_sdk_rust::serdes::FileSystemPathEncoding;
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
/// use aws_durable_execution_sdk_rust::serdes::{FileSystemSerdesConfig, FileSystemSerdesMode, FileSystemPathEncoding};
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
/// use aws_durable_execution_sdk_rust::serdes::{FileSystemSerdesConfig, FileSystemSerdesMode};
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
/// # Blocking I/O
///
/// [`serialize`](Serdes::serialize) and [`deserialize`](Serdes::deserialize)
/// perform synchronous `std::fs` I/O, which can stall for arbitrarily long
/// on a network mount. When the SDK invokes a serdes from its async
/// operation paths it routes the call through
/// `tokio::task::spawn_blocking`, so the executor is never blocked. If you
/// call these methods directly from your own async code, apply the same
/// treatment rather than calling them inline on the runtime.
///
/// # Envelope format
///
/// The checkpoint stores one of:
/// - `{"data":<inline JSON value>}` — value stored inline (OVERFLOW mode, under threshold)
/// - `{"file":"<path>"}` — value stored in a file
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::serdes::{FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode};
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
    /// use aws_durable_execution_sdk_rust::serdes::FileSystemSerdes;
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
    /// use aws_durable_execution_sdk_rust::serdes::{FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode};
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

    /// Serializes a value to the filesystem envelope format.
    ///
    /// Returns the envelope JSON string to be stored in the checkpoint.
    /// The `context` provides the operation identity for path resolution.
    ///
    /// # Errors
    ///
    /// Returns an error if directory creation or file writing fails.
    pub fn serialize_with_context(
        &self,
        value: &serde_json::Value,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let json_str = serde_json::to_string(value).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(FileSystemSerdesError::new(format!(
                    "failed to render value as JSON: {e}"
                )))
            },
        )?;
        match self.config.storage_mode {
            FileSystemSerdesMode::Always => {
                let file_path = self.write_to_file(&json_str, context)?;
                let escaped_path = json_escape_string(&file_path);
                Ok(format!(r#"{{"file":"{escaped_path}"}}"#))
            }
            FileSystemSerdesMode::Overflow => {
                // The value is always valid JSON (it came from a `Value`), so
                // it can be embedded directly in the `{"data":...}` envelope.
                // No sniffing, and no `{"raw":...}` variant to fall back to.
                let inline_envelope = format!(r#"{{"data":{json_str}}}"#);
                if inline_envelope.len() > self.config.overflow_threshold_bytes {
                    let file_path = self.write_to_file(&json_str, context)?;
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
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        // Parse the envelope to determine storage location.
        // Envelope is: {"file":"<path>"} or {"data":<json>}
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
            serde_json::from_str(&contents).map_err(
                |e| -> Box<dyn std::error::Error + Send + Sync> {
                    Box::new(FileSystemSerdesError::new(format!(
                        "invalid JSON in file '{file_path}': {e}"
                    )))
                },
            )
        } else if let Some(raw) = parsed.get("raw").and_then(serde_json::Value::as_str) {
            // Legacy inline envelope. The writer no longer produces `{"raw":..}`
            // (a `Value` is always embeddable under `"data"`), but executions
            // checkpointed by an earlier version may still contain one, so the
            // read path keeps handling it.
            Ok(serde_json::Value::String(raw.to_owned()))
        } else if let Some(data) = parsed.get("data") {
            // Inline data: the "data" field holds the value verbatim.
            Ok(data.clone())
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
        let dir_path = self.resolve_execution_dir(context)?;
        std::fs::create_dir_all(&dir_path).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(FileSystemSerdesError::new(format!(
                    "failed to create directory '{dir_path}': {e}"
                )))
            },
        )?;

        // Defense-in-depth (physical): `resolve_execution_dir` checks only the
        // lexically-cleaned path, so a pre-existing symlink beneath `base_path`
        // can still redirect `create_dir_all` outside the base directory.
        // Canonicalize both sides (resolving symlinks) and verify real
        // containment before writing anything.
        self.assert_canonical_containment(&dir_path)?;

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
    ///
    /// # Errors
    ///
    /// Returns an error if the resolved path escapes `base_path` (defense-in-depth).
    fn resolve_execution_dir(
        &self,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let dir = match self.config.path_encoding {
            FileSystemPathEncoding::Uri => {
                // Try to parse as a durable execution ARN for human-readable paths.
                if let Some(parts) = parse_durable_execution_arn(&context.durable_execution_arn) {
                    let enc_fn = encode_segment(&parts.function_name, self.config.path_encoding);
                    let enc_exec = encode_segment(&parts.execution_name, self.config.path_encoding);
                    let enc_inv = encode_segment(&parts.invocation_id, self.config.path_encoding);
                    format!("{}/{enc_fn}/{enc_exec}/{enc_inv}", self.base_path)
                } else {
                    // Fallback: percent-encode the whole ARN.
                    let encoded = percent_encode(&context.durable_execution_arn);
                    format!("{}/{encoded}", self.base_path)
                }
            }
            FileSystemPathEncoding::Hash => {
                let hash = sha256_hex(&context.durable_execution_arn);
                format!("{}/{hash}", self.base_path)
            }
        };

        // Defense-in-depth: verify the resolved path is within base_path.
        // Compare lexically-cleaned paths using Path::starts_with (which checks
        // component-by-component, not as a string prefix).
        let cleaned_base = path_clean(&self.base_path);
        let cleaned_dir = path_clean(&dir);
        if !std::path::Path::new(&cleaned_dir).starts_with(std::path::Path::new(&cleaned_base)) {
            return Err(Box::new(FileSystemSerdesError::new(format!(
                "resolved path '{}' escapes base_path '{}'",
                cleaned_dir, self.base_path
            ))));
        }

        Ok(dir)
    }

    /// Verifies that `dir_path` physically resolves to a location inside
    /// `base_path`, following symlinks on both sides.
    ///
    /// The lexical check in [`resolve_execution_dir`](Self::resolve_execution_dir)
    /// cannot see symlinks, so a pre-existing symlink under the base directory
    /// could redirect the resolved directory elsewhere. Both paths must exist
    /// when this is called (the caller creates `dir_path` first).
    ///
    /// # Errors
    ///
    /// Returns an error if either path cannot be canonicalized or if the
    /// canonical directory is not contained within the canonical base path.
    fn assert_canonical_containment(
        &self,
        dir_path: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let canonical_dir = std::fs::canonicalize(dir_path).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(FileSystemSerdesError::new(format!(
                    "failed to canonicalize directory '{dir_path}': {e}"
                )))
            },
        )?;
        let canonical_base = std::fs::canonicalize(&self.base_path).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                Box::new(FileSystemSerdesError::new(format!(
                    "failed to canonicalize base_path '{}': {e}",
                    self.base_path
                )))
            },
        )?;
        if !canonical_dir.starts_with(&canonical_base) {
            return Err(Box::new(FileSystemSerdesError::new(format!(
                "canonical path '{}' escapes base_path '{}' (symlink redirection)",
                canonical_dir.display(),
                canonical_base.display()
            ))));
        }
        Ok(())
    }
}

/// # Context-Aware Engine Wiring
///
/// The engine calls [`Serdes::serialize`] and [`Serdes::deserialize`] at every
/// serialization point, passing a [`SerdesContext`] with the operation's wire
/// ID and execution ARN. `FileSystemSerdes` implements them by delegating to
/// [`serialize_with_context`](FileSystemSerdes::serialize_with_context) and
/// [`deserialize_with_context`](FileSystemSerdes::deserialize_with_context),
/// enabling deterministic file-path resolution.
///
/// There is no context-free fallback: both trait methods carry the context, so
/// there is no path on which `FileSystemSerdes` silently declines to persist.
impl Serdes for FileSystemSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.serialize_with_context(value, context)
    }

    fn deserialize(
        &self,
        data: &str,
        context: &SerdesContext,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
        self.deserialize_with_context(data, context)
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
/// use aws_durable_execution_sdk_rust::serdes::FileSystemSerdesError;
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
///
/// For [`FileSystemPathEncoding::Uri`], this additionally ensures that the
/// result is never `.` or `..` — those survive standard percent-encoding
/// because `.` is in the unreserved set, but they form traversal components
/// when used as path segments.
fn encode_segment(value: &str, encoding: FileSystemPathEncoding) -> String {
    match encoding {
        FileSystemPathEncoding::Hash => sha256_hex(value),
        FileSystemPathEncoding::Uri => {
            let encoded = percent_encode(value);
            // A segment that resolves to "." or ".." is unsafe as a directory
            // component — encode the leading dot to neutralize it.
            if encoded == "." || encoded == ".." {
                format!("%2E{}", &encoded[1..])
            } else {
                encoded
            }
        }
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

/// Lexically resolves `.` and `..` in a path, collapsing traversals without
/// touching the filesystem. This is used for the defense-in-depth check before
/// the directory is actually created.
fn path_clean(path: &str) -> String {
    use std::path::{Component, Path};
    let mut components: Vec<&std::ffi::OsStr> = Vec::new();
    let p = Path::new(path);
    let mut has_root = false;
    for component in p.components() {
        match component {
            Component::RootDir => {
                has_root = true;
                components.clear();
            }
            Component::CurDir => {}
            Component::ParentDir => {
                components.pop();
            }
            Component::Normal(c) => {
                components.push(c);
            }
            Component::Prefix(prefix) => {
                // Windows prefix handling — unlikely but safe.
                components.clear();
                components.push(prefix.as_os_str());
            }
        }
    }
    let joined: std::path::PathBuf = components.iter().collect();
    if has_root {
        format!("/{}", joined.to_string_lossy())
    } else {
        joined.to_string_lossy().to_string()
    }
}

// ============================================================
// Shared test support
// ============================================================

/// Test-only serdes shared by the operation-path equivalence tests in
/// `step`, `callback`, and `map_parallel`.
///
/// The point of the shared types is that ONE `Serdes` implementation must
/// behave identically on every operation path — step results, invoke payloads,
/// callback payloads, child results, and map/parallel item and batch results —
/// now that all of them hand the serdes the same [`serde_json::Value`].
#[cfg(test)]
pub(crate) mod test_support {
    use super::{Serdes, SerdesContext};
    use std::sync::{Arc, Mutex};

    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    /// A serdes whose wire form is deliberately NOT JSON: the compact JSON
    /// rendering of the value is hex-encoded behind a `HEX1:` marker.
    ///
    /// Two properties make this useful as a probe:
    ///
    /// 1. It receives the typed value as a [`serde_json::Value`], so it needs
    ///    no per-path decoding to find the payload.
    /// 2. Plain `serde_json` cannot parse a `HEX1:`-prefixed payload, so a
    ///    successful round-trip proves the transform was actually applied and
    ///    reversed rather than incidentally bypassed by a JSON-only path.
    #[derive(Debug)]
    pub(crate) struct HexEnvelopeSerdes;

    impl Serdes for HexEnvelopeSerdes {
        fn serialize(
            &self,
            value: &serde_json::Value,
            _context: &SerdesContext,
        ) -> Result<String, BoxError> {
            Ok(hex_envelope(&value.to_string()))
        }

        fn deserialize(
            &self,
            data: &str,
            _context: &SerdesContext,
        ) -> Result<serde_json::Value, BoxError> {
            Ok(serde_json::from_str(&hex_decode(data)?)?)
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

    /// Returns the wire form `HexEnvelopeSerdes` produces for a value, i.e.
    /// the hex envelope of the value's compact JSON rendering.
    pub(crate) fn hex_envelope_of(value: &serde_json::Value) -> String {
        hex_envelope(&value.to_string())
    }

    /// Reverses [`hex_envelope`], erroring on anything that did not go
    /// through the transform.
    fn hex_decode(payload: &str) -> Result<String, BoxError> {
        let hex = payload
            .strip_prefix("HEX1:")
            .ok_or_else(|| -> BoxError { "missing HEX1: marker".into() })?;
        if hex.len() % 2 != 0 {
            return Err("odd-length hex body".into());
        }
        let mut bytes = Vec::with_capacity(hex.len() / 2);
        for pair in hex.as_bytes().chunks(2) {
            let s = std::str::from_utf8(pair)?;
            bytes.push(u8::from_str_radix(s, 16)?);
        }
        Ok(String::from_utf8(bytes)?)
    }

    /// A `HexEnvelopeSerdes` that also records every value the engine hands it.
    ///
    /// Where `HexEnvelopeSerdes` pins the wire form a path *produces*, this
    /// pins the shape a path *provides*: the exact [`serde_json::Value`] passed
    /// to [`Serdes::serialize`] and the exact `Value` returned from
    /// [`Serdes::deserialize`]. It keeps the non-JSON hex wire form, so a path
    /// that bypassed the transform would fail rather than quietly pass.
    ///
    /// Cloning shares the recording buffers, so a clone can be attached to an
    /// operation while the original is inspected.
    #[derive(Debug, Clone, Default)]
    pub(crate) struct RecordingSerdes {
        serialize_inputs: Arc<Mutex<Vec<serde_json::Value>>>,
        deserialize_outputs: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    impl RecordingSerdes {
        pub(crate) fn new() -> Self {
            Self::default()
        }

        /// Every value handed to [`Serdes::serialize`], in call order.
        pub(crate) fn serialize_inputs(&self) -> Vec<serde_json::Value> {
            self.serialize_inputs
                .lock()
                .map(|seen| seen.clone())
                .unwrap_or_default()
        }

        /// The distinct values handed to [`Serdes::serialize`], first-seen
        /// order preserved (`serde_json::Value` is not `Ord`, so this cannot
        /// sort-then-dedup).
        pub(crate) fn distinct_serialize_inputs(&self) -> Vec<serde_json::Value> {
            let mut distinct: Vec<serde_json::Value> = Vec::new();
            for value in self.serialize_inputs() {
                if !distinct.contains(&value) {
                    distinct.push(value);
                }
            }
            distinct
        }

        /// Every value returned from [`Serdes::deserialize`], in call order.
        pub(crate) fn deserialize_outputs(&self) -> Vec<serde_json::Value> {
            self.deserialize_outputs
                .lock()
                .map(|seen| seen.clone())
                .unwrap_or_default()
        }
    }

    impl Serdes for RecordingSerdes {
        fn serialize(
            &self,
            value: &serde_json::Value,
            _context: &SerdesContext,
        ) -> Result<String, BoxError> {
            if let Ok(mut seen) = self.serialize_inputs.lock() {
                seen.push(value.clone());
            }
            Ok(hex_envelope(&value.to_string()))
        }

        fn deserialize(
            &self,
            data: &str,
            _context: &SerdesContext,
        ) -> Result<serde_json::Value, BoxError> {
            let value: serde_json::Value = serde_json::from_str(&hex_decode(data)?)?;
            if let Ok(mut seen) = self.deserialize_outputs.lock() {
                seen.push(value.clone());
            }
            Ok(value)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions with descriptive messages
#[allow(clippy::indexing_slicing)] // reason: test code with known-good structures
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

        let input = serde_json::json!({"value": 42, "name": "test"});
        let envelope = serdes
            .serialize_with_context(&input, &ctx)
            .expect("serialize should succeed");

        // Envelope should be a file pointer
        assert!(envelope.contains(r#""file""#), "envelope: {envelope}");
        assert!(!envelope.contains(r#""data""#), "envelope: {envelope}");

        // Deserialize should return the original value
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, input);

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

        let small = serde_json::json!({"x": 1});
        let envelope = serdes
            .serialize_with_context(&small, &ctx)
            .expect("serialize should succeed");

        // Small value should be inline
        assert!(envelope.contains(r#""data""#), "envelope: {envelope}");
        assert!(!envelope.contains(r#""file""#), "envelope: {envelope}");

        // Deserialize returns the original
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, small);

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
        let large =
            serde_json::json!({"big": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"});
        let envelope = serdes
            .serialize_with_context(&large, &ctx)
            .expect("serialize should succeed");

        // Large value should overflow to file
        assert!(envelope.contains(r#""file""#), "envelope: {envelope}");

        // Deserialize reads from file
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, large);

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

        let value = serde_json::json!("hello");
        let envelope = serdes
            .serialize_with_context(&value, &ctx)
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
        assert_eq!(output, value);

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
            .serialize_with_context(&serde_json::json!(42), &ctx)
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
            .serialize_with_context(&serde_json::json!("data"), &ctx)
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
    fn deserialize_reads_inline_data_envelope() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_trait");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new("op", "arn:test");

        // Inline envelope: the value is returned verbatim, not re-rendered.
        let result = serdes
            .deserialize_with_context(r#"{"data":"hello"}"#, &ctx)
            .expect("should parse inline");
        assert_eq!(result, serde_json::json!("hello"));

        let result = serdes
            .deserialize_with_context(r#"{"data":{"value":42}}"#, &ctx)
            .expect("should parse inline object");
        assert_eq!(result, serde_json::json!({"value": 42}));

        // Neither 'file' nor 'data': a loud error, never a silent passthrough.
        assert!(
            serdes
                .deserialize_with_context(r#"{"value":42}"#, &ctx)
                .is_err()
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The writer no longer emits `{"raw":...}` — a `Value` is always
    /// embeddable under `"data"` — but executions checkpointed before the
    /// value-based boundary may still contain one, so the READ path must keep
    /// handling it. Deleting the reader alongside the writer would strand
    /// in-flight executions.
    #[test]
    fn deserialize_still_reads_legacy_raw_envelope() {
        let serdes = FileSystemSerdes::new("/unused");
        let ctx = SerdesContext::new("op", "arn:test");

        let result = serdes
            .deserialize_with_context(r#"{"raw":"not json at all"}"#, &ctx)
            .expect("legacy raw envelope must still be readable");
        assert_eq!(result, serde_json::Value::String("not json at all".into()));
    }

    /// Overflow mode must never write a `{"raw":...}` envelope now that the
    /// input is always a `Value`.
    #[test]
    fn overflow_mode_never_writes_raw_envelope() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_no_raw");
        let _ = std::fs::remove_dir_all(&tmp);

        let config = FileSystemSerdesConfig::builder()
            .storage_mode(FileSystemSerdesMode::Overflow)
            .overflow_threshold_bytes(4096)
            .build();
        let serdes = FileSystemSerdes::with_config(tmp.to_string_lossy().to_string(), config);
        let ctx = SerdesContext::new("op", "arn:test");

        // A bare string is the case that previously took the {"raw":...} arm.
        let value = serde_json::json!("plain text");
        let envelope = serdes
            .serialize_with_context(&value, &ctx)
            .expect("serialize");
        assert_eq!(envelope, r#"{"data":"plain text"}"#);
        assert_eq!(
            serdes
                .deserialize_with_context(&envelope, &ctx)
                .expect("deserialize"),
            value
        );

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
        let input = serde_json::json!({"items": [1, 2, 3]});
        let envelope = serdes1
            .serialize_with_context(&input, &ctx)
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

        let input = serde_json::json!({"value": "test"});
        let envelope = serdes
            .serialize_with_context(&input, &ctx)
            .expect("serialize should succeed");

        // The envelope must be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&envelope).expect("envelope must be valid JSON");
        assert!(parsed.get("file").is_some());

        // And round-trips back
        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, input);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Engine-wiring tests: context-aware path activates via trait methods ──

    #[test]
    fn serialize_via_trait_writes_file() {
        // Proves that calling the trait method `Serdes::serialize` on a
        // `Box<dyn Serdes>` holding a FileSystemSerdes activates the
        // context-aware file-writing path (the engine code path).
        let tmp = std::env::temp_dir().join("fs_serdes_trait_ctx_ser");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = tmp.to_string_lossy().to_string();

        let serdes: Box<dyn Serdes> = Box::new(FileSystemSerdes::new(base.clone()));
        let ctx = SerdesContext::new(
            "step-result-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec-1/inv-1",
        );

        let input = serde_json::json!({"total": 42});
        let envelope = serdes
            .serialize(&input, &ctx)
            .expect("Serdes::serialize should succeed");

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
    fn deserialize_via_trait_reads_file() {
        // Proves that calling the trait method `Serdes::deserialize` on a
        // `Box<dyn Serdes>` holding a FileSystemSerdes reads from the file.
        let tmp = std::env::temp_dir().join("fs_serdes_trait_ctx_deser");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = tmp.to_string_lossy().to_string();

        let serdes: Box<dyn Serdes> = Box::new(FileSystemSerdes::new(base.clone()));
        let ctx = SerdesContext::new(
            "op-deser-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec-1/inv-1",
        );

        // Write via the trait.
        let input = serde_json::json!({"key": "value"});
        let envelope = serdes.serialize(&input, &ctx).expect("serialize");

        // Read back via the trait (same Box<dyn Serdes>).
        let output = serdes
            .deserialize(&envelope, &ctx)
            .expect("Serdes::deserialize should succeed");
        assert_eq!(output, input);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The shared probe serdes must be a valid `Serdes` while implementing
    /// only the two value methods, and its transform must be exactly
    /// reversible. It is the fixture the operation-path equivalence tests
    /// depend on, so it gets its own round-trip check.
    #[test]
    fn hex_envelope_probe_serdes_round_trips() {
        use super::test_support::{HexEnvelopeSerdes, hex_envelope_of};

        let serdes: Box<dyn Serdes> = Box::new(HexEnvelopeSerdes);
        let ctx = SerdesContext::new("op-1", "arn:test");

        let value = serde_json::json!({"label": "a\"b\\c\nd ☃", "nested": [[1, -2], []]});
        let wire = serdes.serialize(&value, &ctx).expect("serialize");
        assert_eq!(wire, hex_envelope_of(&value));
        // The wire form must not be parseable as JSON — that is what makes it
        // a useful probe for "was the transform actually applied?".
        assert!(serde_json::from_str::<serde_json::Value>(&wire).is_err());

        let back = serdes.deserialize(&wire, &ctx).expect("deserialize");
        assert_eq!(back, value);

        // A payload that never went through the transform must be rejected
        // rather than silently passed through.
        assert!(serdes.deserialize(&value.to_string(), &ctx).is_err());
    }

    #[test]
    fn default_trait_methods_are_plain_json() {
        // A serdes that overrides nothing must behave exactly like plain
        // `serde_json`: compact rendering out, JSON parse in.
        #[derive(Debug)]
        struct Passthrough;
        impl Serdes for Passthrough {}

        let serdes: Box<dyn Serdes> = Box::new(Passthrough);
        let ctx = SerdesContext::new("op-1", "arn:test");

        let value = serde_json::json!({"a": [1, "two"], "b": null});
        let wire = serdes.serialize(&value, &ctx).expect("serialize");
        assert_eq!(wire, value.to_string());
        assert_eq!(serdes.deserialize(&wire, &ctx).expect("deserialize"), value);
    }

    #[test]
    fn plain_custom_serdes_receives_the_value() {
        // Proves that a plain custom Serdes (not FileSystemSerdes) is handed
        // the typed value, so a string payload needs no quote-stripping.
        struct UpperSerdes;
        impl Debug for UpperSerdes {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("UpperSerdes")
            }
        }
        impl Serdes for UpperSerdes {
            fn serialize(
                &self,
                value: &serde_json::Value,
                _context: &SerdesContext,
            ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
                let raw = value.as_str().unwrap_or_default();
                Ok(raw.to_uppercase())
            }
            fn deserialize(
                &self,
                data: &str,
                _context: &SerdesContext,
            ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>> {
                Ok(serde_json::Value::String(data.to_lowercase()))
            }
        }

        let serdes: Box<dyn Serdes> = Box::new(UpperSerdes);
        let ctx = SerdesContext::new("op-1", "arn:test");

        let result = serdes
            .serialize(&serde_json::json!("hello"), &ctx)
            .expect("serialize");
        assert_eq!(result, "HELLO");

        let result = serdes.deserialize("WORLD", &ctx).expect("deserialize");
        assert_eq!(result, serde_json::json!("world"));
    }

    // ================================================================
    // Path traversal / containment tests (issue #9)
    // ================================================================

    /// Helper to build an ARN with custom components for traversal tests.
    fn traversal_arn(function_name: &str, execution_name: &str, invocation_id: &str) -> String {
        format!(
            "arn:aws:lambda:us-east-1:123456789012:function:{function_name}:1/durable-execution/{execution_name}/{invocation_id}"
        )
    }

    #[test]
    fn traversal_in_function_name_is_contained() {
        let tmp = std::env::temp_dir().join("fs_serdes_traversal_fn");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let arn = traversal_arn("../../etc", "exec-1", "inv-1");
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: arn,
        };

        let input = serde_json::json!({"data": "test"});
        let result = serdes.serialize_with_context(&input, &ctx);

        // Either serialize succeeds and writes inside base_path, or it errors.
        match result {
            Ok(envelope) => {
                // If it succeeded, the file MUST be inside base_path.
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file = std::fs::canonicalize(file_path).expect("file should exist");
                let canonical_base = std::fs::canonicalize(&tmp).unwrap();
                assert!(
                    canonical_file.starts_with(&canonical_base),
                    "file {canonical_file:?} is outside base {canonical_base:?}"
                );
            }
            Err(e) => {
                // Error is acceptable — the traversal was rejected.
                assert!(
                    e.to_string().contains("escapes base_path"),
                    "unexpected error: {e}"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn traversal_in_execution_name_is_contained() {
        let tmp = std::env::temp_dir().join("fs_serdes_traversal_exec");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let arn = traversal_arn("my-fn", "../../etc", "inv-1");
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: arn,
        };

        let input = serde_json::json!({"data": "test"});
        let result = serdes.serialize_with_context(&input, &ctx);

        match result {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file = std::fs::canonicalize(file_path).expect("file should exist");
                let canonical_base = std::fs::canonicalize(&tmp).unwrap();
                assert!(
                    canonical_file.starts_with(&canonical_base),
                    "file {canonical_file:?} is outside base {canonical_base:?}"
                );
            }
            Err(e) => {
                assert!(
                    e.to_string().contains("escapes base_path"),
                    "unexpected error: {e}"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn traversal_in_invocation_id_is_contained() {
        let tmp = std::env::temp_dir().join("fs_serdes_traversal_inv");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let arn = traversal_arn("my-fn", "exec-1", "../../etc");
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: arn,
        };

        let input = serde_json::json!({"data": "test"});
        let result = serdes.serialize_with_context(&input, &ctx);

        match result {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file = std::fs::canonicalize(file_path).expect("file should exist");
                let canonical_base = std::fs::canonicalize(&tmp).unwrap();
                assert!(
                    canonical_file.starts_with(&canonical_base),
                    "file {canonical_file:?} is outside base {canonical_base:?}"
                );
            }
            Err(e) => {
                assert!(
                    e.to_string().contains("escapes base_path"),
                    "unexpected error: {e}"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn encoded_components_round_trip() {
        // Prove that after encoding, deserialize can still find what serialize
        // wrote — the envelope carries the resolved path, not the raw ARN.
        let tmp = std::env::temp_dir().join("fs_serdes_encoded_rt");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        // Use components with special chars that WILL be percent-encoded.
        let arn = traversal_arn("fn/special:chars", "exec name+here", "inv/id");
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: arn,
        };

        let input = serde_json::json!({"round": "trip", "value": 99});
        let envelope = serdes
            .serialize_with_context(&input, &ctx)
            .expect("serialize should succeed with encoded components");

        let output = serdes
            .deserialize_with_context(&envelope, &ctx)
            .expect("deserialize should find the file written by serialize");

        assert_eq!(output, input);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn bare_dotdot_in_each_component_is_contained() {
        // A bare ".." (without "/") survives standard percent-encoding because
        // "." is unreserved. This test verifies that encode_segment neutralizes
        // it, and the defense-in-depth check catches any that slip through.
        let tmp = std::env::temp_dir().join("fs_serdes_bare_dotdot");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());

        // Test bare ".." in function_name
        let arn_fn = traversal_arn("..", "exec-1", "inv-1");
        let ctx_fn = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: arn_fn,
        };
        let input = serde_json::json!({"test": "bare_dotdot"});
        match serdes.serialize_with_context(&input, &ctx_fn) {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file = std::fs::canonicalize(file_path).expect("file should exist");
                let canonical_base = std::fs::canonicalize(&tmp).unwrap();
                assert!(
                    canonical_file.starts_with(&canonical_base),
                    "bare '..' in function_name: file {canonical_file:?} escapes base {canonical_base:?}"
                );
            }
            Err(e) => {
                assert!(
                    e.to_string().contains("escapes base_path"),
                    "unexpected error for bare '..' in function_name: {e}"
                );
            }
        }

        // Test bare ".." in execution_name
        let arn_exec = traversal_arn("my-fn", "..", "inv-1");
        let ctx_exec = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: arn_exec,
        };
        match serdes.serialize_with_context(&input, &ctx_exec) {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file = std::fs::canonicalize(file_path).expect("file should exist");
                let canonical_base = std::fs::canonicalize(&tmp).unwrap();
                assert!(
                    canonical_file.starts_with(&canonical_base),
                    "bare '..' in execution_name: file {canonical_file:?} escapes base {canonical_base:?}"
                );
            }
            Err(e) => {
                assert!(
                    e.to_string().contains("escapes base_path"),
                    "unexpected error for bare '..' in execution_name: {e}"
                );
            }
        }

        // Test bare ".." in invocation_id
        let arn_inv = traversal_arn("my-fn", "exec-1", "..");
        let ctx_inv = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: arn_inv,
        };
        match serdes.serialize_with_context(&input, &ctx_inv) {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file = std::fs::canonicalize(file_path).expect("file should exist");
                let canonical_base = std::fs::canonicalize(&tmp).unwrap();
                assert!(
                    canonical_file.starts_with(&canonical_base),
                    "bare '..' in invocation_id: file {canonical_file:?} escapes base {canonical_base:?}"
                );
            }
            Err(e) => {
                assert!(
                    e.to_string().contains("escapes base_path"),
                    "unexpected error for bare '..' in invocation_id: {e}"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A pre-existing symlink beneath `base_path` must not let a write escape
    /// the base directory: lexical cleaning cannot see symlinks, so the
    /// canonical containment assertion has to reject the redirection.
    #[cfg(unix)]
    #[test]
    fn symlink_under_base_path_rejected() {
        let root = std::env::temp_dir().join("fs_serdes_symlink_escape");
        let _ = std::fs::remove_dir_all(&root);
        let base = root.join("base");
        let outside = root.join("outside");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&outside).unwrap();

        // Plant a symlink where the encoded function-name segment will land:
        // base/<fn> -> outside. "my-fn" percent-encodes to itself.
        std::os::unix::fs::symlink(&outside, base.join("my-fn")).unwrap();

        let serdes = FileSystemSerdes::new(base.to_string_lossy().to_string());
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: traversal_arn("my-fn", "exec-1", "inv-1"),
        };

        let result = serdes.serialize_with_context(&serde_json::json!({"data": "test"}), &ctx);
        let err = result.expect_err("symlink escape must be rejected");
        assert!(
            err.to_string().contains("escapes base_path"),
            "unexpected error: {err}"
        );

        // The directory skeleton may exist (create_dir_all runs before the
        // canonical check), but no payload file may be written through the
        // symlink.
        let redirected = outside.join("exec-1").join("inv-1");
        if redirected.exists() {
            let files: Vec<_> = std::fs::read_dir(&redirected)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| e.path().is_file())
                .collect();
            assert!(
                files.is_empty(),
                "no file may be written outside base_path: {files:?}"
            );
        }

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A benign symlink as the base path itself (e.g. macOS `/tmp` ->
    /// `/private/tmp`) must still work: canonicalization applies to both
    /// sides, so containment holds.
    #[cfg(unix)]
    #[test]
    fn symlinked_base_path_itself_still_works() {
        let root = std::env::temp_dir().join("fs_serdes_symlink_base");
        let _ = std::fs::remove_dir_all(&root);
        let real_base = root.join("real-base");
        std::fs::create_dir_all(&real_base).unwrap();
        let link_base = root.join("link-base");
        std::os::unix::fs::symlink(&real_base, &link_base).unwrap();

        let serdes = FileSystemSerdes::new(link_base.to_string_lossy().to_string());
        let ctx = SerdesContext {
            operation_id: "step-1".to_owned(),
            durable_execution_arn: traversal_arn("my-fn", "exec-1", "inv-1"),
        };

        let envelope = serdes
            .serialize_with_context(&serde_json::json!({"data": "test"}), &ctx)
            .expect("write through a symlinked base path must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        let file_path = parsed["file"].as_str().unwrap();
        let canonical_file = std::fs::canonicalize(file_path).expect("file should exist");
        let canonical_base = std::fs::canonicalize(&real_base).unwrap();
        assert!(
            canonical_file.starts_with(&canonical_base),
            "file {canonical_file:?} must be inside base {canonical_base:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
