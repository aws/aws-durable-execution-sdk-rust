//! Serialization/deserialization abstraction.
//!
//! The [`Serdes`] trait is generic over the operation's Rust type and
//! provides the extension point for custom serialization formats. The
//! default implementation, [`JsonSerdes`], renders compact JSON with
//! `serde_json`.
//!
//! # Configuration model
//!
//! Serdes are configured **per operation**: every serdes-bearing builder
//! carries its serdes implementation as a generic type parameter defaulting
//! to [`JsonSerdes`], and its `.serdes(...)` method swaps that parameter.
//! There is no execution-wide serdes slot — a single trait-object slot
//! cannot represent `Serdes<T>` for every operation output type without
//! erasing the value again. To share one instance across a handler, create
//! an `Arc<S>` and clone it into each operation; `Arc<S>` forwards to `S`
//! through a blanket implementation.
//!
//! # Scheduling contract
//!
//! The SDK awaits the future a serdes returns **directly on the executor
//! thread** — it never wraps the call in another blocking task. Each
//! implementation therefore decides where its work runs:
//!
//! - [`JsonSerdes`] (and other cheap in-memory codecs) perform their work
//!   inline in the returned future, paying no scheduling hop.
//! - [`FileSystemSerdes`] moves its complete serialization or
//!   deserialization path into one [`tokio::task::spawn_blocking`] call per
//!   invocation.
//!
//! Custom implementations must follow the same rule: do not perform
//! blocking filesystem calls or long-running synchronous work on the
//! executor thread — move that work into a blocking task instead.
//!
//! [`FileSystemSerdes`] stores values on a durable filesystem (EFS or S3
//! Files mounted to Lambda), keeping checkpoint payloads small regardless
//! of value size.

use std::future::Future;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::BoxError;

/// Typed, asynchronous serialization/deserialization extension point.
///
/// Implement this trait to provide custom serialization formats for
/// operation values. The trait is generic over `T`, the operation's actual
/// Rust type: `serialize` receives the owned value and `deserialize`
/// returns it, so a custom format sees the real type — struct field
/// declaration order, `i128` values outside the `i64`/`u64` ranges, and
/// anything else a lossy intermediate representation would drop.
///
/// # Ownership
///
/// Both methods take owned inputs. Taking `value: T` (rather than `&T`)
/// lets an implementation move the complete operation into a `'static`
/// [`tokio::task::spawn_blocking`] closure, which is what preserves the
/// SDK's `T: Send` output bound: a `Send` future that retained `&T` would
/// require `T: Sync` instead. Operations round-trip successful values
/// through the configured wire format before returning them, so
/// transferring ownership to `serialize` loses nothing.
///
/// # Scheduling
///
/// The SDK awaits the returned future without wrapping it in another
/// blocking task. Implementations must not perform blocking filesystem
/// calls or long-running synchronous work on the executor thread; move
/// such work into [`tokio::task::spawn_blocking`] (see
/// [`FileSystemSerdes`], which does exactly this). Cheap transformations
/// should run inline in the returned future (see [`JsonSerdes`]).
///
/// # Type pairing
///
/// A type-specific format implements `Serdes<ConcreteType>` directly and
/// can only be attached to an operation producing that type — attaching it
/// elsewhere fails at compile time. A type-agnostic format (a wire-format
/// swap such as CBOR, or storage indirection such as
/// [`FileSystemSerdes`]) implements `Serdes<T>` for all supported `T`
/// through a blanket `impl`.
///
/// ```compile_fail,E0277
/// use aws_durable_execution_sdk_rust as durable;
/// use aws_durable_execution_sdk_rust::serdes::{Serdes, SerdesContext};
///
/// #[derive(serde::Serialize, serde::Deserialize)]
/// struct Order { id: u64 }
///
/// struct OrderSerdes;
///
/// impl Serdes<Order> for OrderSerdes {
///     async fn serialize(
///         &self,
///         value: Order,
///         _context: SerdesContext,
///     ) -> Result<String, durable::BoxError> {
///         Ok(value.id.to_string())
///     }
///
///     async fn deserialize(
///         &self,
///         wire: String,
///         _context: SerdesContext,
///     ) -> Result<Order, durable::BoxError> {
///         Ok(Order { id: wire.parse()? })
///     }
/// }
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     // ERROR: OrderSerdes implements `Serdes<Order>`, not `Serdes<String>`.
///     let result: String = ctx
///         .step(|_| async { Ok("not an order".to_owned()) })
///         .serdes(OrderSerdes)
///         .await?;
///     Ok(result)
/// }
/// ```
///
/// # Examples
///
/// A custom wire format over one concrete type. `async fn` in the trait
/// implementation satisfies the `impl Future` return type:
///
/// ```
/// use aws_durable_execution_sdk_rust::BoxError;
/// use aws_durable_execution_sdk_rust::serdes::{Serdes, SerdesContext};
///
/// struct UppercaseSerdes;
///
/// impl Serdes<String> for UppercaseSerdes {
///     async fn serialize(
///         &self,
///         value: String,
///         _context: SerdesContext,
///     ) -> Result<String, BoxError> {
///         Ok(value.to_uppercase())
///     }
///
///     async fn deserialize(
///         &self,
///         wire: String,
///         _context: SerdesContext,
///     ) -> Result<String, BoxError> {
///         Ok(wire.to_lowercase())
///     }
/// }
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let context = SerdesContext::new("step-1", "arn:test");
/// let wire = UppercaseSerdes
///     .serialize("hello".to_owned(), context.clone())
///     .await?;
/// assert_eq!(wire, "HELLO");
/// assert_eq!(UppercaseSerdes.deserialize(wire, context).await?, "hello");
/// # Ok::<(), BoxError>(())
/// # }).unwrap();
/// ```
pub trait Serdes<T>: Send + Sync + 'static {
    /// Serializes an owned value to the string stored on the wire.
    ///
    /// `context` carries the operation's wire ID and the execution ARN.
    /// Implementations that store data externally (e.g.
    /// [`FileSystemSerdes`]) use it for deterministic path resolution;
    /// implementations that do not need it can ignore it.
    ///
    /// The returned future resolves to an error if the transformation
    /// fails. The SDK awaits the future on the executor thread, so the
    /// implementation must not block it (see the trait-level scheduling
    /// contract).
    fn serialize(
        &self,
        value: T,
        context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send;

    /// Deserializes a wire string back into the operation's value.
    ///
    /// `wire` is the string a previous [`serialize`](Serdes::serialize)
    /// call returned (or, for callbacks, the payload an external caller
    /// delivered).
    ///
    /// The returned future resolves to an error if the transformation
    /// fails. The SDK awaits the future on the executor thread, so the
    /// implementation must not block it (see the trait-level scheduling
    /// contract).
    fn deserialize(
        &self,
        wire: String,
        context: SerdesContext,
    ) -> impl Future<Output = Result<T, BoxError>> + Send;
}

/// Forwards to the inner implementation, so one instance can be shared
/// across operations (and output types) without erasing `T`.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use aws_durable_execution_sdk_rust as durable;
/// use aws_durable_execution_sdk_rust::serdes::FileSystemSerdes;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<u32, durable::BoxError> {
///     let shared = Arc::new(FileSystemSerdes::new("/mnt/efs"));
///     let a: String = ctx
///         .step(|_| async { Ok("large".to_owned()) })
///         .serdes(Arc::clone(&shared))
///         .await?;
///     let b: u32 = ctx
///         .step(|_| async { Ok(42_u32) })
///         .serdes(shared)
///         .await?;
///     let _ = a;
///     Ok(b)
/// }
/// ```
impl<T, S> Serdes<T> for std::sync::Arc<S>
where
    S: Serdes<T> + ?Sized,
{
    fn serialize(
        &self,
        value: T,
        context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        self.as_ref().serialize(value, context)
    }

    fn deserialize(
        &self,
        wire: String,
        context: SerdesContext,
    ) -> impl Future<Output = Result<T, BoxError>> + Send {
        self.as_ref().deserialize(wire, context)
    }
}

// ============================================================
// JsonSerdes
// ============================================================

/// The default serdes: compact JSON via `serde_json`.
///
/// Implements [`Serdes<T>`] for every `T` that is `Serialize +
/// DeserializeOwned + Send + 'static`, so a builder that never calls
/// `.serdes(...)` needs no configuration and no type annotation.
///
/// Serialization and deserialization run **inline** in the returned
/// future — an in-memory JSON transform is cheap, so it pays no
/// `spawn_blocking` scheduling hop.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::BoxError;
/// use aws_durable_execution_sdk_rust::serdes::{JsonSerdes, Serdes, SerdesContext};
///
/// # tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(async {
/// let context = SerdesContext::new("step-1", "arn:test");
/// let wire = JsonSerdes.serialize(vec![1_u32, 2, 3], context.clone()).await?;
/// assert_eq!(wire, "[1,2,3]");
/// let back: Vec<u32> = JsonSerdes.deserialize(wire, context).await?;
/// assert_eq!(back, vec![1, 2, 3]);
/// # Ok::<(), BoxError>(())
/// # }).unwrap();
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JsonSerdes;

impl<T> Serdes<T> for JsonSerdes
where
    T: Serialize + DeserializeOwned + Send + 'static,
{
    fn serialize(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(serde_json::to_string(&value).map_err(Into::into))
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<T, BoxError>> + Send {
        std::future::ready(serde_json::from_str(&wire).map_err(Into::into))
    }
}

// ============================================================
// SerdesContext
// ============================================================

/// Where the wire string handed to [`Serdes::deserialize`] came from.
///
/// An SDK-written record is a payload a previous [`Serdes::serialize`]
/// call produced and the SDK checkpointed. An external payload was
/// delivered by a caller outside the execution (a callback completion)
/// and never went through the SDK's serialize path, so storage
/// indirection recorded in it — such as a [`FileSystemSerdes`] file
/// reference — must not be honored.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::serdes::{PayloadOrigin, SerdesContext};
///
/// let ctx = SerdesContext::new("step-1", "arn:test");
/// assert_eq!(ctx.origin(), PayloadOrigin::SdkRecord);
///
/// let external = ctx.with_origin(PayloadOrigin::External);
/// assert_eq!(external.origin(), PayloadOrigin::External);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PayloadOrigin {
    /// The wire string was written by this SDK's serialize path and read
    /// back from a checkpoint record. Envelope metadata in it (including
    /// file references) is trusted.
    SdkRecord,
    /// The wire string was delivered by an external caller (a callback
    /// completion). It is opaque application data: [`FileSystemSerdes`]
    /// deserializes it directly as the value and never resolves file
    /// references from it.
    External,
}

/// Context information passed to [`FileSystemSerdes`] for file path
/// resolution.
///
/// Provides the operation identity and execution ARN so that on-disk
/// directories are deterministic across replays, plus the
/// [`PayloadOrigin`] of the wire string being deserialized.
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
    /// Where the payload being deserialized came from.
    origin: PayloadOrigin,
}

impl SerdesContext {
    /// Creates a new serdes context with [`PayloadOrigin::SdkRecord`].
    #[must_use]
    pub fn new(operation_id: impl Into<String>, durable_execution_arn: impl Into<String>) -> Self {
        Self {
            operation_id: operation_id.into(),
            durable_execution_arn: durable_execution_arn.into(),
            origin: PayloadOrigin::SdkRecord,
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

    /// Returns where the payload being deserialized came from.
    #[must_use]
    pub fn origin(&self) -> PayloadOrigin {
        self.origin
    }

    /// Returns this context with the payload origin replaced.
    ///
    /// The SDK marks callback payloads [`PayloadOrigin::External`] before
    /// handing them to [`Serdes::deserialize`]; custom serdes tests can
    /// use this to exercise the same path.
    #[must_use]
    pub fn with_origin(mut self, origin: PayloadOrigin) -> Self {
        self.origin = origin;
        self
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
/// # Blocking I/O and Tokio runtime requirement
///
/// Each [`serialize`](Serdes::serialize) or [`deserialize`](Serdes::deserialize)
/// call moves its **entire** implementation — JSON rendering or parsing,
/// hashing, path validation and canonicalization, directory creation, and
/// the file read or write — into one [`tokio::task::spawn_blocking`] task,
/// so no blocking filesystem operation ever runs on the executor thread
/// and each call pays exactly one blocking-pool hop. Related filesystem
/// operations are deliberately batched into that single task rather than
/// issued through individual `tokio::fs` helpers (each of which would
/// spawn its own blocking task).
///
/// Because of this, the returned futures must be awaited **inside a Tokio
/// runtime**; awaiting them elsewhere panics in `spawn_blocking`. A
/// blocking-task join failure (the task panicked or the runtime shut
/// down) is mapped to an ordinary [`BoxError`] rather than propagating a
/// panic.
///
/// # Envelope format
///
/// The checkpoint stores a versioned envelope carrying a
/// `"__aws_durable_serdes"` marker, one of:
///
/// - `{"__aws_durable_serdes":{"version":1,"kind":"inline"},"value":<JSON>}`
///   — value stored inline (OVERFLOW mode, under threshold)
/// - `{"__aws_durable_serdes":{"version":1,"kind":"file"},"file":"<relative path>"}`
///   — value stored in a file under `base_path`
///
/// The marker key is reserved, so an inline value that itself contains a
/// `"file"` key can never be misread as a file reference. Envelopes
/// written by earlier versions (`{"file":...}`, `{"data":...}`,
/// `{"raw":...}`) remain readable.
///
/// # Write and read guarantees
///
/// Every `serialize` call writes a **new, unique** file (staged to a
/// temporary name and atomically renamed into place), so a retried
/// attempt never mutates a file that an already-accepted checkpoint
/// references — committed history stays immutable, and a failed
/// checkpoint leaves prior files untouched. The checkpoint repoints to
/// the new file only when the checkpoint itself succeeds.
///
/// File references are honored only for payloads the SDK itself wrote
/// ([`PayloadOrigin::SdkRecord`]) and are resolved strictly under
/// `base_path` (symlinks followed and containment re-verified). An
/// externally delivered payload ([`PayloadOrigin::External`], e.g. a
/// callback completion) is deserialized directly as the value: it can
/// never name a local file to read.
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

    /// Serializes a value to the filesystem envelope format (synchronous).
    ///
    /// This is the complete serialization path — JSON rendering, path
    /// resolution, directory creation, and the file write. The
    /// [`Serdes`] implementation runs it inside one
    /// [`tokio::task::spawn_blocking`] task.
    fn serialize_sync<T: Serialize>(
        &self,
        value: &T,
        context: &SerdesContext,
    ) -> Result<String, BoxError> {
        let json_str = serde_json::to_string(value).map_err(|e| -> BoxError {
            Box::new(FileSystemSerdesError::new(format!(
                "failed to render value as JSON: {e}"
            )))
        })?;
        match self.config.storage_mode {
            FileSystemSerdesMode::Always => {
                let relative_path = self.write_to_file(&json_str, context)?;
                Ok(file_envelope(&relative_path))
            }
            FileSystemSerdesMode::Overflow => {
                // `json_str` is valid JSON (it came straight from
                // `serde_json::to_string`), so it can be embedded directly
                // under the envelope's `"value"` key. The reserved
                // `"__aws_durable_serdes"` marker is what distinguishes the
                // envelope from the value, so inline data containing a
                // `"file"` key cannot collide with a file reference.
                let inline = inline_envelope(&json_str);
                if inline.len() > self.config.overflow_threshold_bytes {
                    let relative_path = self.write_to_file(&json_str, context)?;
                    Ok(file_envelope(&relative_path))
                } else {
                    Ok(inline)
                }
            }
        }
    }

    /// Deserializes from the filesystem envelope format (synchronous).
    ///
    /// This is the complete deserialization path — envelope parsing, the
    /// file read, and JSON parsing into `T`. The [`Serdes`] implementation
    /// runs it inside one [`tokio::task::spawn_blocking`] task.
    ///
    /// An externally delivered payload ([`PayloadOrigin::External`]) is
    /// parsed directly as `T` — it never went through
    /// [`serialize_sync`](Self::serialize_sync), so it carries no envelope
    /// and can never name a file to read. SDK-written records are parsed
    /// as envelopes; their file references are resolved strictly under
    /// `base_path`.
    fn deserialize_sync<T: DeserializeOwned>(
        &self,
        envelope: &str,
        context: &SerdesContext,
    ) -> Result<T, BoxError> {
        /// The versioned envelope marker: `{"version":1,"kind":"inline"|"file"}`.
        #[derive(serde::Deserialize)]
        struct EnvelopeMarker {
            version: u64,
            kind: String,
        }

        /// Envelope shape. Versioned envelopes carry the
        /// `"__aws_durable_serdes"` marker with `value` (inline) or `file`
        /// (relative path). Records written by earlier SDK versions carry
        /// `{"file":"<path>"}`, `{"data":<json>}`, or `{"raw":"<string>"}`
        /// with no marker. `RawValue` keeps inline data as unparsed JSON
        /// text so `T` is parsed straight from the wire bytes with no
        /// intermediate DOM.
        #[derive(serde::Deserialize)]
        struct Envelope<'a> {
            #[serde(rename = "__aws_durable_serdes")]
            marker: Option<EnvelopeMarker>,
            #[serde(borrow)]
            value: Option<&'a serde_json::value::RawValue>,
            #[serde(borrow)]
            file: Option<std::borrow::Cow<'a, str>>,
            #[serde(borrow)]
            data: Option<&'a serde_json::value::RawValue>,
            raw: Option<String>,
        }

        // External payloads (callback completions) are opaque application
        // data: parse them directly as the value. Honoring envelope
        // metadata from them would let an external caller name arbitrary
        // local files, and a realistic inline payload containing a "file"
        // key would be misparsed as a file reference.
        if context.origin() == PayloadOrigin::External {
            return serde_json::from_str(envelope).map_err(|e| -> BoxError {
                Box::new(FileSystemSerdesError::new(format!(
                    "invalid external payload JSON: {e}"
                )))
            });
        }

        let parsed: Envelope<'_> = serde_json::from_str(envelope).map_err(|e| -> BoxError {
            Box::new(FileSystemSerdesError::new(format!(
                "invalid envelope JSON: {e}"
            )))
        })?;

        if let Some(marker) = parsed.marker {
            // Versioned envelope.
            if marker.version != ENVELOPE_VERSION {
                return Err(Box::new(FileSystemSerdesError::new(format!(
                    "unsupported envelope version {} (this SDK reads version {ENVELOPE_VERSION})",
                    marker.version
                ))));
            }
            return match marker.kind.as_str() {
                "inline" => {
                    let value = parsed.value.ok_or_else(|| -> BoxError {
                        Box::new(FileSystemSerdesError::new(
                            "inline envelope is missing its 'value' field".to_owned(),
                        ))
                    })?;
                    serde_json::from_str(value.get()).map_err(|e| -> BoxError {
                        Box::new(FileSystemSerdesError::new(format!(
                            "invalid inline data in envelope: {e}"
                        )))
                    })
                }
                "file" => {
                    let relative = parsed.file.ok_or_else(|| -> BoxError {
                        Box::new(FileSystemSerdesError::new(
                            "file envelope is missing its 'file' field".to_owned(),
                        ))
                    })?;
                    self.read_relative_file(&relative)
                }
                other => Err(Box::new(FileSystemSerdesError::new(format!(
                    "unsupported envelope kind '{other}'"
                )))),
            };
        }

        // No marker: a record written by an earlier SDK version.
        if let Some(file_path) = parsed.file.as_deref() {
            // Legacy file pointer (stored as the resolved path). It came
            // from an SDK-written record, but resolution is still strict:
            // the file must physically live under `base_path`.
            self.read_contained_file(std::path::Path::new(file_path), file_path)
        } else if let Some(data) = parsed.data {
            // Legacy inline data: the "data" field holds the value's JSON
            // verbatim.
            serde_json::from_str(data.get()).map_err(|e| -> BoxError {
                Box::new(FileSystemSerdesError::new(format!(
                    "invalid inline data in envelope: {e}"
                )))
            })
        } else if let Some(raw) = parsed.raw {
            // Legacy inline envelope. The writer no longer produces
            // `{"raw":..}`, but executions checkpointed by an earlier version
            // may still contain one, so the read path keeps handling it. The
            // raw form stored the value as a plain string; re-encode it as a
            // JSON string literal and parse `T` from that.
            let as_json = serde_json::to_string(&raw).map_err(|e| -> BoxError {
                Box::new(FileSystemSerdesError::new(format!(
                    "failed to re-encode legacy raw envelope: {e}"
                )))
            })?;
            serde_json::from_str(&as_json).map_err(|e| -> BoxError {
                Box::new(FileSystemSerdesError::new(format!(
                    "invalid legacy raw data in envelope: {e}"
                )))
            })
        } else {
            Err(Box::new(FileSystemSerdesError::new(
                "envelope carries no '__aws_durable_serdes' marker and no legacy 'file' or 'data' field"
                    .to_owned(),
            )))
        }
    }

    /// Reads a payload file named by a **relative** path from a versioned
    /// envelope, resolving it strictly under `base_path`.
    ///
    /// Rejects absolute paths and any path that lexically escapes the base
    /// directory, then follows symlinks (canonicalization of the full file
    /// path) and re-verifies physical containment before reading.
    fn read_relative_file<T: DeserializeOwned>(&self, relative: &str) -> Result<T, BoxError> {
        let rel = std::path::Path::new(relative);
        if rel.is_absolute() {
            return Err(Box::new(FileSystemSerdesError::new(format!(
                "file reference '{relative}' must be relative to base_path"
            ))));
        }
        let joined = format!("{}/{relative}", self.base_path);
        // Lexical containment first: a ".." component must not climb out of
        // the base directory even before the filesystem is consulted.
        let cleaned_base = path_clean(&self.base_path);
        let cleaned_joined = path_clean(&joined);
        if !std::path::Path::new(&cleaned_joined).starts_with(std::path::Path::new(&cleaned_base)) {
            return Err(Box::new(FileSystemSerdesError::new(format!(
                "file reference '{relative}' escapes base_path '{}'",
                self.base_path
            ))));
        }
        self.read_contained_file(std::path::Path::new(&joined), relative)
    }

    /// Reads a payload file after verifying it physically resolves under
    /// `base_path` (symlinks followed on both sides), then parses `T` from
    /// its contents. `display_path` is the path to name in errors.
    fn read_contained_file<T: DeserializeOwned>(
        &self,
        path: &std::path::Path,
        display_path: &str,
    ) -> Result<T, BoxError> {
        let canonical_base = std::fs::canonicalize(&self.base_path).map_err(|e| -> BoxError {
            Box::new(FileSystemSerdesError::new(format!(
                "failed to canonicalize base_path '{}': {e}",
                self.base_path
            )))
        })?;
        // Canonicalizing the full file path resolves every symlink,
        // including one at the final component, so the containment check
        // below judges the file's real location.
        let canonical_file = std::fs::canonicalize(path).map_err(|e| -> BoxError {
            Box::new(FileSystemSerdesError::new(format!(
                "failed to resolve file '{display_path}': {e}"
            )))
        })?;
        if !canonical_file.starts_with(&canonical_base) {
            return Err(Box::new(FileSystemSerdesError::new(format!(
                "file reference '{display_path}' escapes base_path '{}'",
                self.base_path
            ))));
        }
        let contents = std::fs::read_to_string(&canonical_file).map_err(|e| -> BoxError {
            Box::new(FileSystemSerdesError::new(format!(
                "failed to read file '{display_path}': {e}"
            )))
        })?;
        serde_json::from_str(&contents).map_err(|e| -> BoxError {
            Box::new(FileSystemSerdesError::new(format!(
                "invalid JSON in file '{display_path}': {e}"
            )))
        })
    }

    /// Writes the JSON string to a new, unique payload file and returns
    /// its path **relative to `base_path`**.
    ///
    /// The payload is staged to a temporary name in the destination
    /// directory and atomically published into place with no-clobber
    /// semantics, so no reader ever observes a partially written file and
    /// no writer ever replaces an existing one. Every call produces a
    /// distinct file name — even across Lambda execution environments
    /// sharing the same EFS or S3 Files mount, where PIDs, counters, and
    /// clock readings can coincide — so a retried attempt never overwrites
    /// a file that an already-accepted checkpoint references. The
    /// checkpoint repoints to the new file only if the checkpoint itself
    /// succeeds, and a failed checkpoint leaves prior files untouched.
    fn write_to_file(
        &self,
        json_str: &str,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let relative_dir = self.resolve_execution_dir(context)?;
        let dir_path = format!("{}/{relative_dir}", self.base_path);
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

        let encoded_op = encode_segment(&context.operation_id, self.config.path_encoding);
        let file_name =
            publish_unique_payload_file(&dir_path, &encoded_op, json_str, unique_file_suffix)?;

        Ok(format!("{relative_dir}/{file_name}"))
    }

    /// Resolves the per-execution directory as a path **relative to
    /// `base_path`**.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory joined onto `base_path` lexically
    /// escapes it (defense-in-depth).
    fn resolve_execution_dir(
        &self,
        context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let relative_dir = match self.config.path_encoding {
            FileSystemPathEncoding::Uri => {
                // Try to parse as a durable execution ARN for human-readable paths.
                if let Some(parts) = parse_durable_execution_arn(&context.durable_execution_arn) {
                    let enc_fn = encode_segment(&parts.function_name, self.config.path_encoding);
                    let enc_exec = encode_segment(&parts.execution_name, self.config.path_encoding);
                    let enc_inv = encode_segment(&parts.invocation_id, self.config.path_encoding);
                    format!("{enc_fn}/{enc_exec}/{enc_inv}")
                } else {
                    // Fallback: percent-encode the whole ARN.
                    percent_encode(&context.durable_execution_arn)
                }
            }
            FileSystemPathEncoding::Hash => sha256_hex(&context.durable_execution_arn),
        };

        // Defense-in-depth: verify the joined path is within base_path.
        // Compare lexically-cleaned paths using Path::starts_with (which checks
        // component-by-component, not as a string prefix).
        let cleaned_base = path_clean(&self.base_path);
        let cleaned_dir = path_clean(&format!("{}/{relative_dir}", self.base_path));
        if !std::path::Path::new(&cleaned_dir).starts_with(std::path::Path::new(&cleaned_base)) {
            return Err(Box::new(FileSystemSerdesError::new(format!(
                "resolved path '{}' escapes base_path '{}'",
                cleaned_dir, self.base_path
            ))));
        }

        Ok(relative_dir)
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

/// One blocking-pool hop per call, containing the entire implementation.
///
/// The engine calls [`Serdes::serialize`] and [`Serdes::deserialize`] at
/// every serialization point, passing a [`SerdesContext`] with the
/// operation's wire ID and execution ARN for deterministic file-path
/// resolution. Each call moves the complete synchronous path — including
/// JSON rendering or parsing — into a single
/// [`tokio::task::spawn_blocking`] task, so the executor thread never
/// touches the filesystem. Blocking-task join failures are mapped to
/// [`BoxError`].
///
/// There is no context-free fallback: both trait methods carry the
/// context, so there is no path on which `FileSystemSerdes` silently
/// declines to persist.
impl<T> Serdes<T> for FileSystemSerdes
where
    T: Serialize + DeserializeOwned + Send + 'static,
{
    fn serialize(
        &self,
        value: T,
        context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        let this = self.clone();
        async move {
            tokio::task::spawn_blocking(move || this.serialize_sync(&value, &context))
                .await
                .map_err(|e| -> BoxError {
                    format!("filesystem serdes serialize task did not complete: {e}").into()
                })?
        }
    }

    fn deserialize(
        &self,
        wire: String,
        context: SerdesContext,
    ) -> impl Future<Output = Result<T, BoxError>> + Send {
        let this = self.clone();
        async move {
            tokio::task::spawn_blocking(move || this.deserialize_sync(&wire, &context))
                .await
                .map_err(|e| -> BoxError {
                    format!("filesystem serdes deserialize task did not complete: {e}").into()
                })?
        }
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

/// The envelope version this SDK writes and reads.
const ENVELOPE_VERSION: u64 = 1;

/// Builds the versioned inline envelope. `json_str` must be valid JSON
/// (it comes straight from `serde_json::to_string`).
fn inline_envelope(json_str: &str) -> String {
    format!(
        r#"{{"__aws_durable_serdes":{{"version":{ENVELOPE_VERSION},"kind":"inline"}},"value":{json_str}}}"#
    )
}

/// Builds the versioned file envelope around a path relative to
/// `base_path`.
fn file_envelope(relative_path: &str) -> String {
    let escaped = json_escape_string(relative_path);
    format!(
        r#"{{"__aws_durable_serdes":{{"version":{ENVELOPE_VERSION},"kind":"file"}},"file":"{escaped}"}}"#
    )
}

/// Returns a filename suffix unique across concurrent writers: wall-clock
/// nanoseconds, the process ID, and a per-process counter.
///
/// Uniqueness of the payload file name is what makes committed files
/// immutable — no later `serialize` call can target the same path.
/// Upper bound on candidate-name collisions tolerated when publishing a
/// payload file. Each retry draws a fresh candidate name, so in practice
/// one retry resolves a collision; the bound only guards against a broken
/// suffix source producing the same name forever.
const MAX_PUBLISH_ATTEMPTS: u32 = 16;

/// Publishes `json_str` under `dir_path` as a new, unique, immutable file
/// and returns its file name.
///
/// [`unique_file_suffix`] alone cannot guarantee uniqueness across Lambda
/// execution environments sharing one EFS or S3 Files mount: distinct
/// processes can share a PID and counter value and can observe the same
/// clock reading. Uniqueness is therefore *enforced* by the filesystem,
/// not assumed from the name:
///
/// - the staging file is opened with `create_new`, so a colliding staging
///   path fails instead of truncating another writer's bytes in flight;
/// - the finished staging file is published via [`std::fs::hard_link`],
///   which — unlike `rename` — fails with `AlreadyExists` rather than
///   replacing an existing destination.
///
/// On either collision the attempt is discarded and retried with a fresh
/// candidate name, so a file an accepted checkpoint references is never
/// corrupted or overwritten.
fn publish_unique_payload_file(
    dir_path: &str,
    encoded_op: &str,
    json_str: &str,
    mut candidate_suffix: impl FnMut() -> String,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    use std::io::Write as _;

    for _ in 0..MAX_PUBLISH_ATTEMPTS {
        let file_name = format!("{encoded_op}-{}.json", candidate_suffix());
        let file_path = format!("{dir_path}/{file_name}");
        // The staging name carries its own independent suffix so writers
        // that collide on the candidate name still stage separately;
        // `create_new` backstops even that.
        let staging_path = format!("{file_path}.{}.tmp", unique_file_suffix());

        let mut staging = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging_path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(Box::new(FileSystemSerdesError::new(format!(
                    "failed to write file '{staging_path}': {e}"
                ))));
            }
        };
        if let Err(e) = staging.write_all(json_str.as_bytes()) {
            drop(staging);
            let _ = std::fs::remove_file(&staging_path);
            return Err(Box::new(FileSystemSerdesError::new(format!(
                "failed to write file '{staging_path}': {e}"
            ))));
        }
        drop(staging);

        // No-clobber atomic publish. Whatever the outcome, the staging
        // file is no longer needed: the envelope never referenced it, so
        // leaving it behind is only clutter.
        let linked = std::fs::hard_link(&staging_path, &file_path);
        let _ = std::fs::remove_file(&staging_path);
        match linked {
            Ok(()) => return Ok(file_name),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                return Err(Box::new(FileSystemSerdesError::new(format!(
                    "failed to move file into place at '{file_path}': {e}"
                ))));
            }
        }
    }

    Err(Box::new(FileSystemSerdesError::new(format!(
        "failed to publish a unique payload file under '{dir_path}' after {MAX_PUBLISH_ATTEMPTS} attempts"
    ))))
}

fn unique_file_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{nanos:x}-{:x}-{count:x}", std::process::id())
}

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
/// now that all of them hand the serdes the operation's typed value directly.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{Serdes, SerdesContext};
    use serde::Serialize;
    use serde::de::DeserializeOwned;
    use std::sync::{Arc, Mutex};

    type BoxError = Box<dyn std::error::Error + Send + Sync>;

    /// A serdes whose wire form is deliberately NOT JSON: the compact JSON
    /// rendering of the value is hex-encoded behind a `HEX1:` marker.
    ///
    /// Two properties make this useful as a probe:
    ///
    /// 1. It receives the operation's typed value directly, so it exercises
    ///    the generic (blanket-`impl`) shape a type-agnostic custom serdes
    ///    uses.
    /// 2. Plain `serde_json` cannot parse a `HEX1:`-prefixed payload, so a
    ///    successful round-trip proves the transform was actually applied and
    ///    reversed rather than incidentally bypassed by a JSON-only path.
    #[derive(Debug)]
    pub(crate) struct HexEnvelopeSerdes;

    impl<T> Serdes<T> for HexEnvelopeSerdes
    where
        T: Serialize + DeserializeOwned + Send + 'static,
    {
        // reason: exercises the async-fn impl form user code writes
        #[allow(clippy::unused_async_trait_impl)]
        async fn serialize(&self, value: T, _context: SerdesContext) -> Result<String, BoxError> {
            Ok(hex_envelope(&serde_json::to_string(&value)?))
        }

        // reason: exercises the async-fn impl form user code writes
        #[allow(clippy::unused_async_trait_impl)]
        async fn deserialize(&self, wire: String, _context: SerdesContext) -> Result<T, BoxError> {
            Ok(serde_json::from_str(&hex_decode(&wire)?)?)
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
    /// pins the shape a path *provides*: the JSON rendering of the typed value
    /// passed to [`Serdes::serialize`] and of the value returned from
    /// [`Serdes::deserialize`] (recorded as parsed [`serde_json::Value`]s for
    /// order-insensitive assertions). It keeps the non-JSON hex wire form, so
    /// a path that bypassed the transform would fail rather than quietly pass.
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

    impl<T> Serdes<T> for RecordingSerdes
    where
        T: Serialize + DeserializeOwned + Send + 'static,
    {
        // reason: exercises the async-fn impl form user code writes
        #[allow(clippy::unused_async_trait_impl)]
        async fn serialize(&self, value: T, _context: SerdesContext) -> Result<String, BoxError> {
            let json = serde_json::to_string(&value)?;
            if let Ok(mut seen) = self.serialize_inputs.lock() {
                seen.push(serde_json::from_str(&json)?);
            }
            Ok(hex_envelope(&json))
        }

        // reason: exercises the async-fn impl form user code writes
        #[allow(clippy::unused_async_trait_impl)]
        async fn deserialize(&self, wire: String, _context: SerdesContext) -> Result<T, BoxError> {
            let json = hex_decode(&wire)?;
            if let Ok(mut seen) = self.deserialize_outputs.lock() {
                seen.push(serde_json::from_str(&json)?);
            }
            Ok(serde_json::from_str(&json)?)
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
        let ctx = SerdesContext::new(
            "step-1".to_owned(),
            "arn:aws:lambda:us-east-1:123:function:my-fn:1/durable-execution/exec-1/inv-1"
                .to_owned(),
        );

        let input = serde_json::json!({"value": 42, "name": "test"});
        let envelope = serdes
            .serialize_sync(&input, &ctx)
            .expect("serialize should succeed");

        // Envelope should be a file pointer
        assert!(envelope.contains(r#""file""#), "envelope: {envelope}");
        assert!(!envelope.contains(r#""data""#), "envelope: {envelope}");

        // Deserialize should return the original value
        let output = serdes
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, input);

        // Clean up
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// ISSUE #46: candidate file names are not trusted to be unique —
    /// across Lambda environments sharing EFS/S3 Files, two writers can
    /// draw the same PID, counter, and clock reading. When their candidate
    /// names collide, publication must not clobber the already-published
    /// file (which an accepted checkpoint may reference); the loser must
    /// retry under a fresh name.
    #[test]
    fn publish_never_clobbers_on_candidate_name_collision() {
        let tmp = std::env::temp_dir().join(format!(
            "fs_serdes_publish_collision_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).expect("create temp dir");
        let dir = tmp.to_string_lossy().into_owned();

        // Writer A publishes under candidate suffix "dup".
        let name_a = publish_unique_payload_file(&dir, "op", r#""writer-a""#, || "dup".to_owned())
            .expect("writer A publishes");
        assert_eq!(name_a, "op-dup.json");

        // Writer B draws the SAME candidate suffix first, then a fresh one.
        let mut draws = vec!["fresh".to_owned(), "dup".to_owned()];
        let name_b = publish_unique_payload_file(&dir, "op", r#""writer-b""#, move || {
            draws.pop().expect("enough draws")
        })
        .expect("writer B retries under a fresh name");
        assert_eq!(name_b, "op-fresh.json");

        // The collision did not corrupt or replace writer A's file.
        assert_eq!(
            std::fs::read_to_string(tmp.join("op-dup.json")).expect("read A"),
            r#""writer-a""#,
            "an existing published file must never be overwritten"
        );
        assert_eq!(
            std::fs::read_to_string(tmp.join("op-fresh.json")).expect("read B"),
            r#""writer-b""#
        );

        // A suffix source stuck on a taken name errors out instead of
        // clobbering, and leaves the published file untouched.
        let err = publish_unique_payload_file(&dir, "op", r#""writer-c""#, || "dup".to_owned())
            .expect_err("exhausted retries must fail");
        assert!(err.to_string().contains("attempts"), "got: {err}");
        assert_eq!(
            std::fs::read_to_string(tmp.join("op-dup.json")).expect("read A again"),
            r#""writer-a""#
        );

        // No staging litter left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&tmp)
            .expect("list dir")
            .filter_map(Result::ok)
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "staging files leaked: {leftovers:?}");

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
        let ctx = SerdesContext::new(
            "step-2".to_owned(),
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e/i".to_owned(),
        );

        let small = serde_json::json!({"x": 1});
        let envelope = serdes
            .serialize_sync(&small, &ctx)
            .expect("serialize should succeed");

        // Small value should be inline, under the versioned marker
        assert!(
            envelope.contains(r#""__aws_durable_serdes""#),
            "envelope: {envelope}"
        );
        assert!(envelope.contains(r#""value""#), "envelope: {envelope}");
        assert!(!envelope.contains(r#""file""#), "envelope: {envelope}");

        // Deserialize returns the original
        let output = serdes
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
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
        let ctx = SerdesContext::new(
            "step-3".to_owned(),
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e/i".to_owned(),
        );

        // This will exceed the 50-byte threshold once wrapped in {"data":...}
        let large =
            serde_json::json!({"big": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"});
        let envelope = serdes
            .serialize_sync(&large, &ctx)
            .expect("serialize should succeed");

        // Large value should overflow to file
        assert!(envelope.contains(r#""file""#), "envelope: {envelope}");

        // Deserialize reads from file
        let output = serdes
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
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
        let ctx = SerdesContext::new(
            "step/with:special".to_owned(),
            "some-weird-arn/with/slashes".to_owned(),
        );

        let value = serde_json::json!("hello");
        let envelope = serdes
            .serialize_sync(&value, &ctx)
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
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, value);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_error_on_missing_file() {
        let tmp = std::env::temp_dir().join("fs_serdes_test_missing_file");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new("x".to_owned(), "y".to_owned());

        // Versioned envelope naming a file that does not exist under base.
        let envelope = file_envelope("does-not-exist.json");
        let result = serdes.deserialize_sync::<serde_json::Value>(&envelope, &ctx);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("failed to resolve file"), "err: {err_msg}");

        // Legacy envelope naming a missing file under base fails the same way.
        let legacy = format!(
            r#"{{"file":"{}/does-not-exist.json"}}"#,
            tmp.to_string_lossy()
        );
        let result = serdes.deserialize_sync::<serde_json::Value>(&legacy, &ctx);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("failed to resolve file"), "err: {err_msg}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn filesystem_serdes_pointer_format_stability() {
        // Verify the envelope format is stable for wire compatibility
        let tmp = std::env::temp_dir().join("fs_serdes_test_format");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new(
            "op-1".to_owned(),
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec/inv".to_owned(),
        );

        let envelope = serdes
            .serialize_sync(&serde_json::json!(42), &ctx)
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
        let ctx = SerdesContext::new(
            "step-1".to_owned(),
            "arn:aws:lambda:us-west-2:999:function:my-function:42/durable-execution/my-exec/my-inv"
                .to_owned(),
        );

        let envelope = serdes
            .serialize_sync(&serde_json::json!("data"), &ctx)
            .expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        let path = parsed.get("file").and_then(|v| v.as_str()).unwrap();

        // URI mode: functionName/executionName/invocationId/opId-<unique>.json,
        // relative to base_path.
        assert!(
            !std::path::Path::new(path).is_absolute(),
            "path must be relative to base_path: {path}"
        );
        assert!(path.starts_with("my-function/"), "path: {path}");
        assert!(path.contains("my-exec"), "path: {path}");
        assert!(path.contains("my-inv"), "path: {path}");
        // The file name keeps the operation ID prefix but carries a unique
        // suffix, so a retried attempt never reuses the same path.
        let file_name = path.rsplit('/').next().unwrap();
        assert!(file_name.starts_with("step-1-"), "file name: {file_name}");
        #[allow(clippy::case_sensitive_file_extension_comparisons)]
        // reason: we generate the extension; always lowercase
        let correct_suffix = file_name.ends_with(".json");
        assert!(correct_suffix, "file name: {file_name}");

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
            .deserialize_sync::<serde_json::Value>(r#"{"data":"hello"}"#, &ctx)
            .expect("should parse inline");
        assert_eq!(result, serde_json::json!("hello"));

        let result = serdes
            .deserialize_sync::<serde_json::Value>(r#"{"data":{"value":42}}"#, &ctx)
            .expect("should parse inline object");
        assert_eq!(result, serde_json::json!({"value": 42}));

        // Neither 'file' nor 'data': a loud error, never a silent passthrough.
        assert!(
            serdes
                .deserialize_sync::<serde_json::Value>(r#"{"value":42}"#, &ctx)
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
            .deserialize_sync::<serde_json::Value>(r#"{"raw":"not json at all"}"#, &ctx)
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
        let envelope = serdes.serialize_sync(&value, &ctx).expect("serialize");
        assert_eq!(
            envelope,
            r#"{"__aws_durable_serdes":{"version":1,"kind":"inline"},"value":"plain text"}"#
        );
        assert_eq!(
            serdes
                .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
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

        let ctx = SerdesContext::new(
            "replay-op".to_owned(),
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e1/i1".to_owned(),
        );

        // First invocation: serialize
        let serdes1 = FileSystemSerdes::new(base.clone());
        let input = serde_json::json!({"items": [1, 2, 3]});
        let envelope = serdes1.serialize_sync(&input, &ctx).expect("serialize");

        // Second invocation (replay): deserialize from a fresh instance
        let serdes2 = FileSystemSerdes::new(base);
        let output = serdes2
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
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
        let ctx = SerdesContext::new(
            "step-1".to_owned(),
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec/inv".to_owned(),
        );

        let input = serde_json::json!({"value": "test"});
        let envelope = serdes
            .serialize_sync(&input, &ctx)
            .expect("serialize should succeed");

        // The envelope must be valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&envelope).expect("envelope must be valid JSON");
        assert!(parsed.get("file").is_some());

        // And round-trips back
        let output = serdes
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
            .expect("deserialize should succeed");
        assert_eq!(output, input);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Engine-wiring tests: the generic async trait drives the file path ──

    /// `Serdes::serialize` on `FileSystemSerdes` activates the context-aware
    /// file-writing path (the engine code path) through one blocking task.
    #[tokio::test]
    async fn serialize_via_trait_writes_file() {
        let tmp = std::env::temp_dir().join("fs_serdes_trait_ctx_ser");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = tmp.to_string_lossy().to_string();

        let serdes = FileSystemSerdes::new(base.clone());
        let ctx = SerdesContext::new(
            "step-result-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec-1/inv-1",
        );

        let input = serde_json::json!({"total": 42});
        let envelope = serdes
            .serialize(input, ctx)
            .await
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

    /// `Serdes::deserialize` on `FileSystemSerdes` reads back what
    /// `Serdes::serialize` wrote, through the same generic trait surface.
    #[tokio::test]
    async fn deserialize_via_trait_reads_file() {
        let tmp = std::env::temp_dir().join("fs_serdes_trait_ctx_deser");
        let _ = std::fs::remove_dir_all(&tmp);
        let base = tmp.to_string_lossy().to_string();

        let serdes = FileSystemSerdes::new(base.clone());
        let ctx = SerdesContext::new(
            "op-deser-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/exec-1/inv-1",
        );

        // Write via the trait.
        let input = serde_json::json!({"key": "value"});
        let envelope = serdes
            .serialize(input.clone(), ctx.clone())
            .await
            .expect("serialize");

        // Read back via the trait (same instance).
        let output: serde_json::Value = serdes
            .deserialize(envelope, ctx)
            .await
            .expect("Serdes::deserialize should succeed");
        assert_eq!(output, input);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The shared probe serdes must be a valid `Serdes` while implementing
    /// only the two value methods, and its transform must be exactly
    /// reversible. It is the fixture the operation-path equivalence tests
    /// depend on, so it gets its own round-trip check.
    #[tokio::test]
    async fn hex_envelope_probe_serdes_round_trips() {
        use super::test_support::{HexEnvelopeSerdes, hex_envelope_of};

        let ctx = SerdesContext::new("op-1", "arn:test");

        let value = serde_json::json!({"label": "a\"b\\c\nd ☃", "nested": [[1, -2], []]});
        let wire = HexEnvelopeSerdes
            .serialize(value.clone(), ctx.clone())
            .await
            .expect("serialize");
        assert_eq!(wire, hex_envelope_of(&value));
        // The wire form must not be parseable as JSON — that is what makes it
        // a useful probe for "was the transform actually applied?".
        assert!(serde_json::from_str::<serde_json::Value>(&wire).is_err());

        let back: serde_json::Value = HexEnvelopeSerdes
            .deserialize(wire, ctx.clone())
            .await
            .expect("deserialize");
        assert_eq!(back, value);

        // A payload that never went through the transform must be rejected
        // rather than silently passed through.
        assert!(
            Serdes::<serde_json::Value>::deserialize(&HexEnvelopeSerdes, value.to_string(), ctx)
                .await
                .is_err()
        );
    }

    /// `JsonSerdes` behaves exactly like plain `serde_json`: compact
    /// rendering out, JSON parse in.
    #[tokio::test]
    async fn json_serdes_is_plain_json() {
        let ctx = SerdesContext::new("op-1", "arn:test");

        let value = serde_json::json!({"a": [1, "two"], "b": null});
        let wire = JsonSerdes
            .serialize(value.clone(), ctx.clone())
            .await
            .expect("serialize");
        assert_eq!(wire, value.to_string());
        let back: serde_json::Value = JsonSerdes
            .deserialize(wire, ctx)
            .await
            .expect("deserialize");
        assert_eq!(back, value);
    }

    /// A custom `Serdes` receives the operation's actual typed value — a
    /// `String` value arrives as `String`, with no JSON quoting to strip.
    #[tokio::test]
    async fn custom_serdes_receives_the_typed_value() {
        struct UpperSerdes;

        impl Serdes<String> for UpperSerdes {
            // reason: exercises the async-fn impl form user code writes
            #[allow(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(value.to_uppercase())
            }

            // reason: exercises the async-fn impl form user code writes
            #[allow(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(wire.to_lowercase())
            }
        }

        let ctx = SerdesContext::new("op-1", "arn:test");

        let wire = UpperSerdes
            .serialize("hello".to_owned(), ctx.clone())
            .await
            .expect("serialize");
        assert_eq!(wire, "HELLO");

        let back = UpperSerdes
            .deserialize("WORLD".to_owned(), ctx)
            .await
            .expect("deserialize");
        assert_eq!(back, "world");
    }

    /// `7i128 << 100` survives a custom-serdes round trip: the serdes
    /// receives the real `i128`, a value no `serde_json::Value` intermediary
    /// could represent (it exceeds both the `i64` and `u64` ranges).
    #[tokio::test]
    async fn i128_survives_custom_serdes_round_trip() {
        struct DecimalSerdes;

        impl Serdes<i128> for DecimalSerdes {
            // reason: exercises the async-fn impl form user code writes
            #[allow(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: i128,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(value.to_string())
            }

            // reason: exercises the async-fn impl form user code writes
            #[allow(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<i128, BoxError> {
                Ok(wire.parse()?)
            }
        }

        let ctx = SerdesContext::new("op-1", "arn:test");
        let value = 7_i128 << 100;

        let wire = DecimalSerdes
            .serialize(value, ctx.clone())
            .await
            .expect("serialize");
        let back = DecimalSerdes
            .deserialize(wire, ctx)
            .await
            .expect("deserialize");
        assert_eq!(back, value);
    }

    /// Struct field declaration order survives a custom-serdes round trip:
    /// the serdes sees the typed value's own `Serialize` output, not a
    /// key-sorted DOM.
    #[tokio::test]
    async fn struct_field_declaration_order_survives_custom_serdes() {
        // Deliberately anti-alphabetical declaration order.
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Ordered {
            zebra: u8,
            mango: u8,
            apple: u8,
        }

        /// A pass-through custom serdes: renders the typed value itself.
        struct PassThrough;

        impl Serdes<Ordered> for PassThrough {
            // reason: exercises the async-fn impl form user code writes
            #[allow(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: Ordered,
                _context: SerdesContext,
            ) -> Result<String, BoxError> {
                Ok(serde_json::to_string(&value)?)
            }

            // reason: exercises the async-fn impl form user code writes
            #[allow(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<Ordered, BoxError> {
                Ok(serde_json::from_str(&wire)?)
            }
        }

        let ctx = SerdesContext::new("op-1", "arn:test");
        let value = Ordered {
            zebra: 1,
            mango: 2,
            apple: 3,
        };

        let wire = PassThrough
            .serialize(value, ctx.clone())
            .await
            .expect("serialize");
        // Declaration order, not alphabetical order.
        assert_eq!(wire, r#"{"zebra":1,"mango":2,"apple":3}"#);

        let back = PassThrough
            .deserialize(wire, ctx)
            .await
            .expect("deserialize");
        assert_eq!(
            back,
            Ordered {
                zebra: 1,
                mango: 2,
                apple: 3
            }
        );
    }

    /// One `Arc<S>` instance works across output types supported by `S`,
    /// through the forwarding `impl<T, S> Serdes<T> for Arc<S>`.
    #[tokio::test]
    async fn arc_serdes_shared_across_output_types() {
        let shared = std::sync::Arc::new(JsonSerdes);
        let ctx = SerdesContext::new("op-1", "arn:test");

        let s = Serdes::<String>::serialize(&shared, "x".to_owned(), ctx.clone())
            .await
            .expect("string serialize");
        assert_eq!(s, "\"x\"");

        let n = Serdes::<u32>::serialize(&shared, 7_u32, ctx.clone())
            .await
            .expect("u32 serialize");
        assert_eq!(n, "7");

        let back: u32 = shared
            .deserialize("7".to_owned(), ctx)
            .await
            .expect("u32 deserialize");
        assert_eq!(back, 7);
    }

    /// A `Send` but non-`Sync` value round-trips through `FileSystemSerdes`:
    /// owned inputs let the whole operation move into the blocking task, so
    /// the returned future is `Send` without requiring `T: Sync`.
    #[tokio::test]
    async fn send_not_sync_value_round_trips_through_filesystem() {
        // `Cell` is `Send` but not `Sync`.
        #[derive(Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
        struct NotSync {
            value: String,
            #[serde(skip)]
            cell: std::cell::Cell<u8>,
        }

        fn assert_send<F: Send>(f: F) -> F {
            f
        }

        let tmp = std::env::temp_dir().join("fs_serdes_not_sync");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new(
            "op-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e/i",
        );

        let input = NotSync {
            value: "send-not-sync".to_owned(),
            cell: std::cell::Cell::new(3),
        };
        let envelope = assert_send(serdes.serialize(input, ctx.clone()))
            .await
            .expect("serialize");
        let back: NotSync = assert_send(serdes.deserialize(envelope, ctx))
            .await
            .expect("deserialize");
        assert_eq!(back.value, "send-not-sync");

        let _ = std::fs::remove_dir_all(&tmp);
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
        let ctx = SerdesContext::new("step-1".to_owned(), arn);

        let input = serde_json::json!({"data": "test"});
        let result = serdes.serialize_sync(&input, &ctx);

        // Either serialize succeeds and writes inside base_path, or it errors.
        match result {
            Ok(envelope) => {
                // If it succeeded, the file MUST be inside base_path.
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file =
                    std::fs::canonicalize(tmp.join(file_path)).expect("file should exist");
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
        let ctx = SerdesContext::new("step-1".to_owned(), arn);

        let input = serde_json::json!({"data": "test"});
        let result = serdes.serialize_sync(&input, &ctx);

        match result {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file =
                    std::fs::canonicalize(tmp.join(file_path)).expect("file should exist");
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
        let ctx = SerdesContext::new("step-1".to_owned(), arn);

        let input = serde_json::json!({"data": "test"});
        let result = serdes.serialize_sync(&input, &ctx);

        match result {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file =
                    std::fs::canonicalize(tmp.join(file_path)).expect("file should exist");
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
        let ctx = SerdesContext::new("step-1".to_owned(), arn);

        let input = serde_json::json!({"round": "trip", "value": 99});
        let envelope = serdes
            .serialize_sync(&input, &ctx)
            .expect("serialize should succeed with encoded components");

        let output = serdes
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
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
        let ctx_fn = SerdesContext::new("step-1".to_owned(), arn_fn);
        let input = serde_json::json!({"test": "bare_dotdot"});
        match serdes.serialize_sync(&input, &ctx_fn) {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file =
                    std::fs::canonicalize(tmp.join(file_path)).expect("file should exist");
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
        let ctx_exec = SerdesContext::new("step-1".to_owned(), arn_exec);
        match serdes.serialize_sync(&input, &ctx_exec) {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file =
                    std::fs::canonicalize(tmp.join(file_path)).expect("file should exist");
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
        let ctx_inv = SerdesContext::new("step-1".to_owned(), arn_inv);
        match serdes.serialize_sync(&input, &ctx_inv) {
            Ok(envelope) => {
                let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
                let file_path = parsed["file"].as_str().unwrap();
                let canonical_file =
                    std::fs::canonicalize(tmp.join(file_path)).expect("file should exist");
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
        let ctx = SerdesContext::new(
            "step-1".to_owned(),
            traversal_arn("my-fn", "exec-1", "inv-1"),
        );

        let result = serdes.serialize_sync(&serde_json::json!({"data": "test"}), &ctx);
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
        let ctx = SerdesContext::new(
            "step-1".to_owned(),
            traversal_arn("my-fn", "exec-1", "inv-1"),
        );

        let envelope = serdes
            .serialize_sync(&serde_json::json!({"data": "test"}), &ctx)
            .expect("write through a symlinked base path must succeed");
        let parsed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        let file_path = parsed["file"].as_str().unwrap();
        let canonical_file =
            std::fs::canonicalize(link_base.join(file_path)).expect("file should exist");
        let canonical_base = std::fs::canonicalize(&real_base).unwrap();
        assert!(
            canonical_file.starts_with(&canonical_base),
            "file {canonical_file:?} must be inside base {canonical_base:?}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    // ================================================================
    // Issue #46: immutable unique files, versioned envelope, origin
    // gating, strict resolution under base_path
    // ================================================================

    /// Every serialize call must write a NEW file: a retried attempt must
    /// never mutate the file an already-accepted checkpoint references.
    #[test]
    fn serialize_writes_unique_immutable_files() {
        let tmp = std::env::temp_dir().join("fs_serdes_unique_files");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        // Same operation, same context — models a retry of the same op.
        let ctx = SerdesContext::new(
            "step-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e/i",
        );

        let first = serdes
            .serialize_sync(&serde_json::json!(1), &ctx)
            .expect("first serialize");
        let second = serdes
            .serialize_sync(&serde_json::json!(2), &ctx)
            .expect("second serialize");

        let first_path = serde_json::from_str::<serde_json::Value>(&first).unwrap()["file"]
            .as_str()
            .unwrap()
            .to_owned();
        let second_path = serde_json::from_str::<serde_json::Value>(&second).unwrap()["file"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_ne!(
            first_path, second_path,
            "a second serialize of the same operation must not reuse the path"
        );

        // The first envelope still reads the ORIGINAL value: the second
        // write did not touch the file the first envelope references.
        let replayed: serde_json::Value = serdes
            .deserialize_sync(&first, &ctx)
            .expect("first envelope must stay readable");
        assert_eq!(replayed, serde_json::json!(1));
        let second_read: serde_json::Value =
            serdes.deserialize_sync(&second, &ctx).expect("second");
        assert_eq!(second_read, serde_json::json!(2));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// The atomic-rename staging file must not survive a successful write.
    #[test]
    fn serialize_leaves_no_staging_files_behind() {
        fn assert_no_tmp(dir: &std::path::Path) {
            for entry in std::fs::read_dir(dir).unwrap().filter_map(Result::ok) {
                let path = entry.path();
                if path.is_dir() {
                    assert_no_tmp(&path);
                } else {
                    assert!(
                        path.extension().is_none_or(|ext| ext != "tmp"),
                        "staging file left behind: {path:?}"
                    );
                }
            }
        }

        let tmp = std::env::temp_dir().join("fs_serdes_no_staging");
        let _ = std::fs::remove_dir_all(&tmp);

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new(
            "step-1",
            "arn:aws:lambda:us-east-1:123:function:fn:1/durable-execution/e/i",
        );
        serdes
            .serialize_sync(&serde_json::json!({"k": "v"}), &ctx)
            .expect("serialize");

        assert_no_tmp(&tmp);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// An externally delivered payload must NEVER be honored as a file
    /// reference — neither in the legacy shape nor with a spoofed
    /// versioned marker. It parses directly as the value.
    #[test]
    fn external_origin_never_resolves_file_references() {
        let tmp = std::env::temp_dir().join("fs_serdes_external_no_file");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        // Plant a real, readable file under base_path: even a reference an
        // SDK record COULD legitimately resolve must not resolve when the
        // payload came from an external caller.
        let secret = tmp.join("secret.json");
        std::fs::write(&secret, r#""the file contents""#).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new("cb-1", "arn:test").with_origin(PayloadOrigin::External);

        // Legacy-shaped reference.
        let legacy = format!(r#"{{"file":"{}"}}"#, secret.to_string_lossy());
        let out: serde_json::Value = serdes
            .deserialize_sync(&legacy, &ctx)
            .expect("external payload parses as its own JSON");
        assert_eq!(
            out,
            serde_json::json!({"file": secret.to_string_lossy()}),
            "the payload must come back as data, not as the file's contents"
        );

        // Spoofed versioned marker.
        let spoofed = file_envelope("secret.json");
        let out: serde_json::Value = serdes
            .deserialize_sync(&spoofed, &ctx)
            .expect("external payload parses as its own JSON");
        assert!(
            out.get("__aws_durable_serdes").is_some(),
            "spoofed marker must come back as plain data: {out}"
        );
        assert_ne!(out, serde_json::json!("the file contents"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// A realistic external payload that happens to contain a `file` key
    /// (the misparse from the issue) deserializes as the full value.
    #[test]
    fn external_inline_payload_with_file_key_parses_as_value() {
        let serdes = FileSystemSerdes::new("/unused-base");
        let ctx = SerdesContext::new("cb-1", "arn:test").with_origin(PayloadOrigin::External);

        let payload = r#"{"file": "/tmp/result.json", "data": {"status": "ready"}}"#;
        let out: serde_json::Value = serdes
            .deserialize_sync(payload, &ctx)
            .expect("must parse as the value");
        assert_eq!(
            out,
            serde_json::json!({"file": "/tmp/result.json", "data": {"status": "ready"}})
        );
    }

    /// An SDK-written INLINE value containing `file`/`data` keys cannot
    /// collide with the envelope: the reserved marker key disambiguates.
    #[test]
    fn inline_value_containing_file_key_round_trips() {
        let tmp = std::env::temp_dir().join("fs_serdes_inline_file_key");
        let _ = std::fs::remove_dir_all(&tmp);

        let config = FileSystemSerdesConfig::builder()
            .storage_mode(FileSystemSerdesMode::Overflow)
            .overflow_threshold_bytes(4096)
            .build();
        let serdes = FileSystemSerdes::with_config(tmp.to_string_lossy().to_string(), config);
        let ctx = SerdesContext::new("op", "arn:test");

        let value = serde_json::json!({"file": "/tmp/result.json", "data": {"status": "ready"}});
        let envelope = serdes.serialize_sync(&value, &ctx).expect("serialize");
        assert!(
            envelope.contains(r#""__aws_durable_serdes""#),
            "inline envelope must carry the marker: {envelope}"
        );

        let out: serde_json::Value = serdes
            .deserialize_sync(&envelope, &ctx)
            .expect("deserialize");
        assert_eq!(
            out, value,
            "the value's own 'file' key must not be read as a file reference"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Versioned file references must be relative and must stay under
    /// `base_path`: absolute paths and `..` escapes are rejected.
    #[test]
    fn file_reference_escaping_base_rejected() {
        let tmp = std::env::temp_dir().join("fs_serdes_ref_escape");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new("op", "arn:test");

        // Absolute path in a versioned envelope.
        let absolute = file_envelope("/etc/hostname");
        let err = serdes
            .deserialize_sync::<serde_json::Value>(&absolute, &ctx)
            .unwrap_err();
        assert!(
            err.to_string().contains("must be relative"),
            "unexpected error: {err}"
        );

        // Traversal escape in a versioned envelope.
        let escape = file_envelope("../outside.json");
        let err = serdes
            .deserialize_sync::<serde_json::Value>(&escape, &ctx)
            .unwrap_err();
        assert!(
            err.to_string().contains("escapes base_path"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Legacy (marker-less) file pointers from SDK records written by
    /// earlier versions still read — but only from under `base_path`.
    #[test]
    fn legacy_file_envelope_strictly_contained() {
        let tmp = std::env::temp_dir().join("fs_serdes_legacy_containment");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let serdes = FileSystemSerdes::new(tmp.to_string_lossy().to_string());
        let ctx = SerdesContext::new("op", "arn:test");

        // A genuine legacy record: absolute path under base_path.
        let inside = tmp.join("legacy-op.json");
        std::fs::write(&inside, r#"{"legacy": true}"#).unwrap();
        let legacy_ok = format!(r#"{{"file":"{}"}}"#, inside.to_string_lossy());
        let out: serde_json::Value = serdes
            .deserialize_sync(&legacy_ok, &ctx)
            .expect("legacy record under base_path must stay readable");
        assert_eq!(out, serde_json::json!({"legacy": true}));

        // A legacy-shaped record pointing outside base_path is rejected,
        // even though the file exists and is readable.
        let outside_dir = std::env::temp_dir().join("fs_serdes_legacy_outside");
        let _ = std::fs::remove_dir_all(&outside_dir);
        std::fs::create_dir_all(&outside_dir).unwrap();
        let outside = outside_dir.join("outside.json");
        std::fs::write(&outside, r#""outside""#).unwrap();
        let legacy_escape = format!(r#"{{"file":"{}"}}"#, outside.to_string_lossy());
        let err = serdes
            .deserialize_sync::<serde_json::Value>(&legacy_escape, &ctx)
            .unwrap_err();
        assert!(
            err.to_string().contains("escapes base_path"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&tmp);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    /// A symlink under `base_path` must not let a FILE READ escape: the
    /// final component is canonicalized before containment is judged.
    #[cfg(unix)]
    #[test]
    fn file_read_through_symlink_escape_rejected() {
        let root = std::env::temp_dir().join("fs_serdes_read_symlink");
        let _ = std::fs::remove_dir_all(&root);
        let base = root.join("base");
        let outside = root.join("outside");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("target.json"), r#""escaped""#).unwrap();
        std::os::unix::fs::symlink(outside.join("target.json"), base.join("link.json")).unwrap();

        let serdes = FileSystemSerdes::new(base.to_string_lossy().to_string());
        let ctx = SerdesContext::new("op", "arn:test");

        let envelope = file_envelope("link.json");
        let err = serdes
            .deserialize_sync::<serde_json::Value>(&envelope, &ctx)
            .unwrap_err();
        assert!(
            err.to_string().contains("escapes base_path"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    /// An envelope from a FUTURE format version fails loudly instead of
    /// being misread.
    #[test]
    fn unknown_envelope_version_rejected() {
        let serdes = FileSystemSerdes::new("/unused-base");
        let ctx = SerdesContext::new("op", "arn:test");

        let future = r#"{"__aws_durable_serdes":{"version":2,"kind":"inline"},"value":1}"#;
        let err = serdes
            .deserialize_sync::<serde_json::Value>(future, &ctx)
            .unwrap_err();
        assert!(
            err.to_string().contains("unsupported envelope version 2"),
            "unexpected error: {err}"
        );
    }

    /// An unknown envelope kind fails loudly.
    #[test]
    fn unknown_envelope_kind_rejected() {
        let serdes = FileSystemSerdes::new("/unused-base");
        let ctx = SerdesContext::new("op", "arn:test");

        let chunked = r#"{"__aws_durable_serdes":{"version":1,"kind":"chunked"},"value":1}"#;
        let err = serdes
            .deserialize_sync::<serde_json::Value>(chunked, &ctx)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported envelope kind 'chunked'"),
            "unexpected error: {err}"
        );
    }
}
