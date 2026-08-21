//! Error types for durable operations.
//!
//! All error types are hand-written (no `thiserror`), `#[non_exhaustive]`,
//! and implement [`std::error::Error`]. Use the `kind()` accessor pattern
//! for matching error variants, and [`source()`](std::error::Error::source)
//! to reach the underlying cause.
//!
//! # `Display` convention
//!
//! Each layer's `Display` writes **one frame** — its own description, never
//! its source's text. Walking [`source()`](std::error::Error::source)
//! therefore never repeats text. To print the whole chain in one line, use
//! the alternate form `{:#}`, which walks the source chain from that layer
//! down. The message recorded on the wire is built from that same single
//! flattening site.
//!
//! # Kinds classify, `source()` carries
//!
//! The `*ErrorKind` enums are pure classification: empty variants are unit
//! variants, and variants that retain structural facts (an attempt count,
//! a failed index) are newtype variants wrapping a payload struct with
//! accessors. The escaping error that caused the failure is never
//! stringified into a kind — it is carried by the error struct and
//! returned from `source()`.

use std::error::Error;
use std::fmt;

/// The boxed source type every error struct carries.
///
/// The `Send + Sync` bounds are load-bearing: all public error types are
/// `Send + Sync` (asserted at compile time below), and a bare
/// `Box<dyn Error>` would silently revoke that.
pub(crate) type Source = Box<dyn Error + Send + Sync + 'static>;

/// Maximum number of source-chain links walked when extracting wire
/// identity (`error_data`, `stack_trace`) from an escaping error.
///
/// Matches the JS SDK's ten-link cause-chain walk, so a payload survives
/// the same nesting depth across SDKs.
const CHAIN_WALK_LIMIT: usize = 10;

/// Maximum number of captured backtrace frames written to the wire.
const STACK_TRACE_FRAME_LIMIT: usize = 64;

// ────────────────────────────────────────────────────────────────────────────
// The single flattening site
// ────────────────────────────────────────────────────────────────────────────

/// Writes `frame` followed by every frame of `source`'s chain, separated
/// by `": "`.
///
/// This is the **single flattening site**: every alternate (`{:#}`)
/// `Display` implementation in this module and the wire-message builder
/// ([`wire_error_for`]) funnel through it. No other code path joins an
/// error's chain into one string.
pub(crate) fn write_chain(
    f: &mut fmt::Formatter<'_>,
    frame: &dyn fmt::Display,
    mut source: Option<&(dyn Error + 'static)>,
) -> fmt::Result {
    write!(f, "{frame}")?;
    while let Some(s) = source {
        write!(f, ": {s}")?;
        source = s.source();
    }
    Ok(())
}

/// Flattens an error's chain into one string via [`write_chain`].
pub(crate) fn chain_string(err: &(dyn Error + 'static)) -> String {
    struct Chain<'a>(&'a (dyn Error + 'static));
    impl fmt::Display for Chain<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write_chain(f, &FrameOnly(self.0), self.0.source())
        }
    }
    /// Adapter printing only the error's own `Display` (non-alternate),
    /// so foreign errors are not double-walked.
    struct FrameOnly<'a>(&'a (dyn Error + 'static));
    impl fmt::Display for FrameOnly<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    Chain(err).to_string()
}

// ────────────────────────────────────────────────────────────────────────────
// WireError — the named wire failure record
// ────────────────────────────────────────────────────────────────────────────

/// The failure fields of a wire `ErrorObject`, as a named record.
///
/// Carries the four wire error fields: `error_type`, `error_message`,
/// `error_data`, and `stack_trace`. Reachable from a failed operation via
/// [`OperationError::wire`], and from a replayed failure's synthetic
/// source via [`ReplayedFailure::wire`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct WireError {
    error_type: Option<String>,
    error_message: Option<String>,
    error_data: Option<String>,
    stack_trace: Vec<String>,
}

impl WireError {
    /// Creates a wire error record from its type and message fields.
    pub(crate) fn new(
        error_type: Option<impl Into<String>>,
        error_message: Option<impl Into<String>>,
    ) -> Self {
        Self {
            error_type: error_type.map(Into::into),
            error_message: error_message.map(Into::into),
            error_data: None,
            stack_trace: Vec::new(),
        }
    }

    /// Sets the opaque `error_data` payload.
    pub(crate) fn with_error_data(mut self, error_data: Option<impl Into<String>>) -> Self {
        self.error_data = error_data.map(Into::into);
        self
    }

    /// Sets the stack trace frames.
    pub(crate) fn with_stack_trace(mut self, stack_trace: Vec<String>) -> Self {
        self.stack_trace = stack_trace;
        self
    }

    /// The wire `ErrorType` — the name of the error type that failed the
    /// operation, as recorded in the execution history.
    #[must_use]
    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    /// The wire `ErrorMessage` — the flattened, human-readable message.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// The wire `ErrorData` — an opaque payload attached to the failure.
    ///
    /// The SDK writes this field and passes it through boundaries, but
    /// **never deserializes it and never dispatches on it**: the wire
    /// error fields can be supplied by an external caller (via
    /// `SendDurableExecutionCallbackFailure`), so typed reconstruction
    /// from an attacker-influenced payload is decode surface the SDK
    /// refuses to have. Interpret it yourself if you know its shape.
    #[must_use]
    pub fn error_data(&self) -> Option<&str> {
        self.error_data.as_deref()
    }

    /// The wire `StackTrace` frames, stored and exposed verbatim.
    ///
    /// A Rust [`std::backtrace::Backtrace`] cannot be reconstructed from
    /// recorded frames, so these are store-and-expose only.
    #[must_use]
    pub fn stack_trace(&self) -> &[String] {
        &self.stack_trace
    }

    /// Converts this record into the SDK `ErrorObject` wire shape
    /// (internal).
    pub(crate) fn to_error_object(&self) -> aws_sdk_lambda::types::ErrorObject {
        let mut builder = aws_sdk_lambda::types::ErrorObject::builder();
        if let Some(t) = self.error_type() {
            builder = builder.error_type(t);
        }
        if let Some(m) = self.error_message() {
            builder = builder.error_message(m);
        }
        if let Some(d) = self.error_data() {
            builder = builder.error_data(d);
        }
        for frame in self.stack_trace() {
            builder = builder.stack_trace(frame);
        }
        builder.build()
    }
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.error_type.as_deref(), self.error_message.as_deref()) {
            (Some(t), Some(m)) => write!(f, "{t}: {m}"),
            (None, Some(m)) => write!(f, "{m}"),
            (Some(t), None) => write!(f, "{t}"),
            (None, None) => write!(f, "unknown error"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ReplayedFailure — the synthetic source on replay
// ────────────────────────────────────────────────────────────────────────────

/// The synthetic source attached to an error rebuilt from a checkpointed
/// failure on replay.
///
/// On the live path an operation error's `source()` is the escaping error
/// itself; after a replay only the wire record survives, so `source()`
/// returns this type carrying the recorded `error_type` and
/// `error_message` (plus `error_data` and `stack_trace` when recorded).
/// Downcast to it to read the wire fields programmatically.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ReplayedFailure {
    wire: WireError,
}

impl ReplayedFailure {
    /// Creates a replayed failure from the recorded wire fields.
    pub(crate) fn new(wire: WireError) -> Self {
        Self { wire }
    }

    /// Boxes a replayed failure as an error source.
    pub(crate) fn source_from(wire: WireError) -> Source {
        Box::new(Self::new(wire))
    }

    /// The recorded wire `ErrorType`.
    #[must_use]
    pub fn error_type(&self) -> Option<&str> {
        self.wire.error_type()
    }

    /// The recorded wire `ErrorMessage`.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.wire.error_message()
    }

    /// The full recorded wire failure record.
    #[must_use]
    pub fn wire(&self) -> &WireError {
        &self.wire
    }
}

impl fmt::Display for ReplayedFailure {
    /// Writes the recorded message as this frame.
    ///
    /// The recorded `error_type` is deliberately NOT folded into the
    /// text — it is data, answered by [`error_type()`](Self::error_type) —
    /// so a failure observed live and the same failure observed on replay
    /// render identically.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.wire.error_message(), self.wire.error_type()) {
            (Some(m), _) => write!(f, "{m}"),
            (None, Some(t)) => write!(f, "{t}"),
            (None, None) => write!(f, "unknown error"),
        }
    }
}

impl Error for ReplayedFailure {}

// ────────────────────────────────────────────────────────────────────────────
// ContextualError — crate-internal frame + cause carrier
// ────────────────────────────────────────────────────────────────────────────

/// Crate-internal error that adds one context frame on top of a cause,
/// without stringifying the cause into the frame.
#[derive(Debug)]
pub(crate) struct ContextualError {
    context: String,
    source: Source,
}

impl ContextualError {
    /// Boxes a context frame over `source` as an error source.
    pub(crate) fn source_from(context: impl Into<String>, source: impl Into<Source>) -> Source {
        Box::new(Self {
            context: context.into(),
            source: source.into(),
        })
    }
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.context, self.source())
        } else {
            write!(f, "{}", self.context)
        }
    }
}

impl Error for ContextualError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&*self.source as &(dyn Error + 'static))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TypedError — explicit error-type naming for the wire
// ────────────────────────────────────────────────────────────────────────────

/// An error wrapper that records its inner error's concrete type name for
/// the wire `ErrorType`.
///
/// Rust erases an error's concrete type the moment it is boxed into a
/// [`BoxError`](crate::BoxError) — inside the SDK there is no runtime name
/// to recover, so an unwrapped error is recorded with the generic type
/// `"Error"`. Wrapping the error at the one place its type is still known
/// (its creation site) is the explicit alternative:
///
/// ```
/// use aws_durable_execution_sdk_rust::{BoxError, TypedError};
///
/// #[derive(Debug)]
/// struct PaymentDeclined;
/// # impl std::fmt::Display for PaymentDeclined {
/// #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
/// #         write!(f, "payment declined")
/// #     }
/// # }
/// # impl std::error::Error for PaymentDeclined {}
///
/// let err: BoxError = Box::new(TypedError::new(PaymentDeclined));
/// // The step that fails with `err` records ErrorType "PaymentDeclined".
/// ```
///
/// The wrapper is one chain layer: its `Display` frame is the recorded
/// type name, and its [`source()`](std::error::Error::source) is the
/// wrapped error itself — so the concrete error stays reachable through
/// the ordinary `source()` downcast walk, and a flattened chain reads
/// `"TransientError: temporary failure"`.
#[derive(Debug)]
#[non_exhaustive]
pub struct TypedError {
    error_type: String,
    inner: Source,
}

impl TypedError {
    /// Wraps `err`, recording its concrete type's name (without module
    /// path) as the wire `ErrorType`.
    #[must_use]
    pub fn new<E: Error + Send + Sync + 'static>(err: E) -> Self {
        Self {
            error_type: short_type_name(std::any::type_name::<E>()),
            inner: Box::new(err),
        }
    }

    /// Wraps `err` under an explicitly chosen wire `ErrorType`.
    #[must_use]
    pub fn with_type(error_type: impl Into<String>, err: impl Into<Source>) -> Self {
        Self {
            error_type: error_type.into(),
            inner: err.into(),
        }
    }

    /// The wire `ErrorType` recorded for this error.
    #[must_use]
    pub fn error_type(&self) -> &str {
        &self.error_type
    }

    /// The wrapped error.
    #[must_use]
    pub fn inner(&self) -> &(dyn Error + 'static) {
        &*self.inner
    }
}

impl fmt::Display for TypedError {
    /// Writes the recorded type name as this layer's frame; the wrapped
    /// error is the next frame in the chain.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.error_type, self.source())
        } else {
            write!(f, "{}", self.error_type)
        }
    }
}

impl Error for TypedError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&*self.inner as &(dyn Error + 'static))
    }
}

/// Reduces `std::any::type_name` output to bare type names: every module
/// path segment is dropped, including inside generic arguments
/// (`a::B<c::D>` becomes `B<D>`).
fn short_type_name(full: &str) -> String {
    let mut result = String::with_capacity(full.len());
    let mut segment = String::new();
    for c in full.chars() {
        match c {
            ':' => segment.clear(),
            '<' | '>' | ',' | ' ' | '(' | ')' | '[' | ']' | '&' | ';' => {
                result.push_str(&segment);
                segment.clear();
                result.push(c);
            }
            _ => segment.push(c),
        }
    }
    result.push_str(&segment);
    result
}

// ────────────────────────────────────────────────────────────────────────────
// Wire failure derivation (write path)
// ────────────────────────────────────────────────────────────────────────────

/// Derives the wire failure record for an escaping error, explicitly —
/// no `Debug` scraping.
///
/// - `error_message` is the error's flattened source chain, built by the
///   module's single flattening site ([`chain_string`]).
/// - `error_type` is taken from structured identity when the error *is*
///   one: an [`OperationError`] re-records its attached wire record's
///   type when it carries one (or a [`TypedError`]'s name from its
///   chain), falling back to its kind's registry name — so an explicitly
///   supplied user type survives re-recording across child-context and
///   map boundaries. A [`ReplayedFailure`] re-records its original type.
///   A plain user error has no recoverable type name in Rust — the
///   concrete type is erased when the caller boxes it — so it records
///   `fallback_type`.
/// - `error_data` is passed through from the first error in the source
///   chain (up to [`CHAIN_WALK_LIMIT`] links) that carries one, so an
///   externally supplied payload survives child-context and map
///   boundaries. It is never synthesized and never parsed.
/// - `stack_trace` is passed through from the chain when recorded, and
///   captured fresh at this site otherwise.
pub(crate) fn wire_error_for(err: &(dyn Error + 'static), fallback_type: &str) -> WireError {
    let error_type = if let Some(op) = err.downcast_ref::<OperationError>() {
        // An operation error's own recorded identity wins: the attached
        // wire record already names the user's type when one was
        // supplied, and a `TypedError` still in the chain names it
        // directly. Only an error with neither records its kind's
        // registry name.
        op.wire()
            .and_then(WireError::error_type)
            .map(str::to_owned)
            .or_else(|| typed_error_name(err))
            .unwrap_or_else(|| op.kind().wire_type_name().to_owned())
    } else if let Some(replayed) = err.downcast_ref::<ReplayedFailure>() {
        // Re-recording a replayed failure preserves its original type.
        replayed.error_type().unwrap_or(fallback_type).to_owned()
    } else {
        // A `TypedError` anywhere in the chain names the user's type
        // explicitly — the one non-erased source of a concrete name.
        typed_error_name(err).unwrap_or_else(|| fallback_type.to_owned())
    };
    wire_error_with_type(err, &error_type)
}

/// Walks the source chain (up to [`CHAIN_WALK_LIMIT`] links) for the first
/// [`TypedError`] and returns its recorded type name.
fn typed_error_name(err: &(dyn Error + 'static)) -> Option<String> {
    let mut link: Option<&(dyn Error + 'static)> = Some(err);
    for _ in 0..CHAIN_WALK_LIMIT {
        let e = link?;
        if let Some(typed) = e.downcast_ref::<TypedError>() {
            return Some(typed.error_type().to_owned());
        }
        link = e.source();
    }
    None
}

/// Like [`wire_error_for`], but with a caller-fixed wire `error_type`
/// (used where the type is a protocol discriminator, e.g. the combinator
/// replay markers). The message flattening and `error_data`/`stack_trace`
/// chain walk are identical.
pub(crate) fn wire_error_with_type(err: &(dyn Error + 'static), error_type: &str) -> WireError {
    let mut error_data = None;
    let mut stack_trace = Vec::new();
    let mut link: Option<&(dyn Error + 'static)> = Some(err);
    let mut walked = 0;
    while let Some(e) = link {
        if walked >= CHAIN_WALK_LIMIT {
            break;
        }
        if let Some(w) = wire_identity(e) {
            if error_data.is_none() {
                error_data = w.error_data().map(str::to_owned);
            }
            if stack_trace.is_empty() {
                stack_trace = w.stack_trace().to_vec();
            }
            if error_data.is_some() && !stack_trace.is_empty() {
                break;
            }
        }
        link = e.source();
        walked += 1;
    }

    if stack_trace.is_empty() {
        stack_trace = capture_stack_trace();
    }

    WireError::new(Some(error_type), Some(chain_string(err)))
        .with_error_data(error_data)
        .with_stack_trace(stack_trace)
}

/// Creates a live failure record from a caller-fixed `error_type` and
/// message, capturing the stack at the construction site.
///
/// This is the constructor for live failure records that have no
/// escaping error to walk — protocol-discriminator failures such as a
/// combinator's empty input, a wait-for-condition strategy exhaustion,
/// or a callback timeout. A bare [`WireError::new`] at such a site would
/// emit no `stack_trace`; routing through this helper centralizes the
/// capture so every live-written record carries one.
pub(crate) fn wire_error_manual(error_type: &str, message: impl Into<String>) -> WireError {
    WireError::new(Some(error_type), Some(message.into())).with_stack_trace(capture_stack_trace())
}

/// The wire `ErrorType` recorded when a checkpoint write itself failed —
/// on the terminal `FAIL` record a permanent rejection persists for the
/// operation, and on the `FAILED` envelope that then ends the execution.
/// A handler never observes this type: checkpoint failures unwind the
/// handler through the unrecoverable path (issue #43).
pub(crate) const CHECKPOINT_FAILED_ERROR_TYPE: &str = "CheckpointFailedError";

/// The wire `ErrorType` recorded on the terminal `FAIL` an operation
/// persists when its result (or carried state) failed LOCAL serialization
/// before the outcome write (issue #43).
///
/// This type is the replay discriminator for the serialization
/// classification: the live path yields a `SerializationFailed`-kinded
/// error after persisting this record, and replay reconstructs the SAME
/// kind by matching this type, so a handler that branches on the kind
/// takes the same path live and replayed. It is a protocol discriminator
/// written via [`wire_error_with_type`], never derived from a user error's
/// own identity.
pub(crate) const SERIALIZATION_FAILED_ERROR_TYPE: &str = "SerializationError";

/// Builds the wire record a terminal `FAIL` carries when the operation's
/// result failed local serialization before its outcome write (issue #43).
///
/// The fixed [`SERIALIZATION_FAILED_ERROR_TYPE`] is what lets replay
/// reconstruct the serialization classification (see the constant's doc);
/// message flattening and the `error_data`/`stack_trace` chain walk are
/// the standard [`wire_error_with_type`] behavior.
pub(crate) fn serialization_failure_wire(err: &(dyn Error + 'static)) -> WireError {
    wire_error_with_type(err, SERIALIZATION_FAILED_ERROR_TYPE)
}

/// Builds the small wire record a terminal `FAIL` carries when the
/// operation's own outcome write was permanently rejected (issue #43).
///
/// Deliberately derived from the *checkpoint failure*, not from the
/// operation's payload: the whole point of this record is that it is a
/// few hundred bytes and goes through on a channel that rejected only
/// the payload, ending the re-execution loop after one lap.
pub(crate) fn checkpoint_failure_wire(err: &impl fmt::Display) -> WireError {
    wire_error_manual(
        CHECKPOINT_FAILED_ERROR_TYPE,
        format!("checkpoint write failed: {err}"),
    )
}

/// Returns the wire record carried by `err`, when `err` is one of the
/// SDK's own wire-record-carrying types.
fn wire_identity<'a>(err: &'a (dyn Error + 'static)) -> Option<&'a WireError> {
    if let Some(op) = err.downcast_ref::<OperationError>() {
        return op.wire();
    }
    if let Some(replayed) = err.downcast_ref::<ReplayedFailure>() {
        return Some(replayed.wire());
    }
    None
}

/// Captures the current backtrace as wire frames, one line per frame,
/// capped at [`STACK_TRACE_FRAME_LIMIT`].
///
/// Uses `force_capture` so the failure record is populated regardless of
/// `RUST_BACKTRACE`; the cost is paid on the failure path only.
fn capture_stack_trace() -> Vec<String> {
    let backtrace = std::backtrace::Backtrace::force_capture();
    format!("{backtrace}")
        .lines()
        .take(STACK_TRACE_FRAME_LIMIT)
        .map(str::to_owned)
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// OperationError
// ────────────────────────────────────────────────────────────────────────────

/// The top-level error type returned by [`DurableFuture`](crate::DurableFuture).
///
/// Wraps a typed per-operation cause accessible via [`kind()`](Self::kind).
/// Match on the kind to determine the specific failure mode, and walk
/// [`source()`](std::error::Error::source) to reach the underlying cause —
/// on the live path that is the escaping error itself (downcastable to its
/// concrete type), and after a replay it is a [`ReplayedFailure`] carrying
/// the recorded wire fields.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{OperationError, OperationErrorKind};
///
/// fn handle_error(err: &OperationError) {
///     match err.kind() {
///         OperationErrorKind::Step(step_err) => {
///             // The step's escaping error is reachable through source().
///             if let Some(cause) = std::error::Error::source(step_err) {
///                 tracing::error!(%cause, "step failed");
///             }
///         }
///         // `{:#}` prints the full chain; `{}` prints one frame.
///         _ => tracing::error!(error = format!("{err:#}"), "operation failed"),
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct OperationError {
    kind: OperationErrorKind,
    /// Boxed so the failure context does not widen every
    /// `Result<_, OperationError>` on the happy path.
    context: Option<Box<OperationContext>>,
}

/// The failure's wire context: the operation it belongs to and the wire
/// record it produced or replayed.
#[derive(Debug, Default)]
struct OperationContext {
    operation_id: Option<String>,
    status: Option<String>,
    wire: Option<WireError>,
}

impl OperationError {
    /// Returns the specific error kind for this operation failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::{OperationError, OperationErrorKind, StepErrorKind};
    ///
    /// let err = OperationError::__test_error();
    /// if let OperationErrorKind::Step(step_err) = err.kind() {
    ///     assert!(matches!(step_err.kind(), StepErrorKind::ExecutionFailed { .. }));
    /// }
    /// ```
    #[must_use]
    pub fn kind(&self) -> &OperationErrorKind {
        &self.kind
    }

    /// The wire ID of the operation that failed, when known.
    #[must_use]
    pub fn operation_id(&self) -> Option<&str> {
        self.context.as_ref()?.operation_id.as_deref()
    }

    /// The operation's wire status (for example `FAILED` or `TIMED_OUT`),
    /// when known.
    #[must_use]
    pub fn status(&self) -> Option<&str> {
        self.context.as_ref()?.status.as_deref()
    }

    /// The wire failure record for this error, when one exists.
    ///
    /// On a replayed failure this is the record read back from the
    /// execution history; on a live failure it is the record the SDK
    /// wrote. `None` for errors that never reached the wire (for example
    /// a serialization failure before checkpointing).
    #[must_use]
    pub fn wire(&self) -> Option<&WireError> {
        self.context.as_ref()?.wire.as_ref()
    }

    /// Creates an `OperationError` from its kind (internal).
    pub(crate) fn from_kind(kind: OperationErrorKind) -> Self {
        Self {
            kind,
            context: None,
        }
    }

    /// Attaches the failing operation's wire ID and status (internal).
    pub(crate) fn with_operation(
        mut self,
        operation_id: impl Into<String>,
        status: impl Into<String>,
    ) -> Self {
        let context = self.context.get_or_insert_default();
        context.operation_id = Some(operation_id.into());
        context.status = Some(status.into());
        self
    }

    /// Attaches the wire failure record (internal).
    pub(crate) fn with_wire(mut self, wire: WireError) -> Self {
        self.context.get_or_insert_default().wire = Some(wire);
        self
    }

    /// Creates a test error (doc-hidden, for doctests only).
    #[doc(hidden)]
    #[must_use]
    pub fn __test_error() -> Self {
        Self::from_kind(OperationErrorKind::Step(StepError::new(
            StepErrorKind::ExecutionFailed,
            Some(Source::from("test")),
        )))
    }

    /// The operation family name used in this error's `Display` frame.
    fn family(&self) -> &'static str {
        match &self.kind {
            OperationErrorKind::Step(_) => "step",
            OperationErrorKind::Wait(_) => "wait",
            OperationErrorKind::Invoke(_) => "invoke",
            OperationErrorKind::Callback(_) => "callback",
            OperationErrorKind::WaitForCondition(_) => "wait_for_condition",
            OperationErrorKind::ChildContext(_) => "child_context",
            OperationErrorKind::Combinator(_) => "combinator",
            OperationErrorKind::NonDeterministicExecution(_) => "non_deterministic_execution",
        }
    }
}

impl fmt::Display for OperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        struct Frame<'a>(&'a OperationError);
        impl fmt::Display for Frame<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "operation error: {}", self.0.family())
            }
        }
        if f.alternate() {
            write_chain(f, &Frame(self), self.source())
        } else {
            write!(f, "{}", Frame(self))
        }
    }
}

impl Error for OperationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            OperationErrorKind::Step(e) => Some(e),
            OperationErrorKind::Wait(e) => Some(e),
            OperationErrorKind::Invoke(e) => Some(e),
            OperationErrorKind::Callback(e) => Some(e),
            OperationErrorKind::WaitForCondition(e) => Some(e),
            OperationErrorKind::ChildContext(e) => Some(e),
            OperationErrorKind::Combinator(e) => Some(e),
            OperationErrorKind::NonDeterministicExecution(e) => Some(e),
        }
    }
}

/// Classification of operation errors by operation type.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{OperationError, OperationErrorKind};
///
/// fn is_retryable(err: &OperationError) -> bool {
///     // Callback timeouts are external; everything else is treated as
///     // internal in this example.
///     matches!(err.kind(), OperationErrorKind::Callback(_))
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum OperationErrorKind {
    /// A step operation failed.
    Step(StepError),
    /// A wait operation failed.
    Wait(WaitError),
    /// An invoke operation failed.
    Invoke(InvokeError),
    /// A callback operation failed.
    Callback(CallbackError),
    /// A wait-for-condition operation failed.
    WaitForCondition(WaitForConditionError),
    /// A child context operation failed.
    ChildContext(ChildContextError),
    /// A combinator operation failed.
    Combinator(CombinatorError),
    /// The handler produced operations in a different order than the
    /// checkpointed history — the execution is non-deterministic.
    NonDeterministicExecution(NonDeterministicExecutionError),
}

impl OperationErrorKind {
    /// The wire `ErrorType` name recorded for this kind when the failing
    /// error carries no more specific identity.
    pub(crate) fn wire_type_name(&self) -> &'static str {
        match self {
            Self::Step(_) => "StepError",
            Self::Wait(_) => "WaitError",
            Self::Invoke(_) => "InvokeError",
            Self::Callback(_) => "CallbackError",
            Self::WaitForCondition(_) => "WaitForConditionError",
            Self::ChildContext(_) => "ChildContextError",
            Self::Combinator(_) => "PromiseCombinatorError",
            Self::NonDeterministicExecution(_) => "NonDeterministicExecutionError",
        }
    }
}

impl fmt::Display for OperationErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Step(e) => write!(f, "step: {e}"),
            Self::Wait(e) => write!(f, "wait: {e}"),
            Self::Invoke(e) => write!(f, "invoke: {e}"),
            Self::Callback(e) => write!(f, "callback: {e}"),
            Self::WaitForCondition(e) => write!(f, "wait_for_condition: {e}"),
            Self::ChildContext(e) => write!(f, "child_context: {e}"),
            Self::Combinator(e) => write!(f, "combinator: {e}"),
            Self::NonDeterministicExecution(e) => write!(f, "non_deterministic_execution: {e}"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StepError
// ────────────────────────────────────────────────────────────────────────────

/// Error from a step operation.
///
/// Use [`kind()`](Self::kind) to determine the failure mode and
/// [`source()`](std::error::Error::source) to reach the escaping error.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{StepError, StepErrorKind};
///
/// fn describe(err: &StepError) -> String {
///     match err.kind() {
///         StepErrorKind::RetriesExhausted(details) => {
///             format!("gave up after {} attempts", details.attempts())
///         }
///         // `{:#}` flattens the frame and its source chain.
///         _ => format!("{err:#}"),
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct StepError {
    kind: StepErrorKind,
    source: Option<Source>,
}

impl StepError {
    /// Returns the specific kind of step error.
    #[must_use]
    pub fn kind(&self) -> &StepErrorKind {
        &self.kind
    }

    /// Creates a `StepError` from its kind and source (internal).
    pub(crate) fn new(kind: StepErrorKind, source: Option<Source>) -> Self {
        Self { kind, source }
    }

    /// Moves the source out of this error (internal).
    pub(crate) fn into_source(self) -> Option<Source> {
        self.source
    }
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.kind, self.source())
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl Error for StepError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|s| s as &(dyn Error + 'static))
    }
}

/// Specific kinds of step failures.
///
/// Kinds classify; the escaping error is carried by [`StepError`] and
/// returned from its [`source()`](std::error::Error::source).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::StepErrorKind;
///
/// fn attempts_used(kind: &StepErrorKind) -> Option<u32> {
///     match kind {
///         StepErrorKind::RetriesExhausted(details) => Some(details.attempts()),
///         _ => None,
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum StepErrorKind {
    /// The step body returned an error.
    #[non_exhaustive]
    ExecutionFailed,
    /// The step exceeded its maximum retry attempts.
    RetriesExhausted(RetriesExhausted),
    /// A serialization or deserialization error occurred.
    #[non_exhaustive]
    SerializationFailed,
}

impl fmt::Display for StepErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExecutionFailed => write!(f, "execution failed"),
            Self::RetriesExhausted(details) => write!(f, "{details}"),
            Self::SerializationFailed => write!(f, "serialization failed"),
        }
    }
}

/// Details of a [`StepErrorKind::RetriesExhausted`] failure.
#[derive(Debug)]
#[non_exhaustive]
pub struct RetriesExhausted {
    attempts: u32,
}

impl RetriesExhausted {
    /// Creates the payload (internal).
    pub(crate) fn new(attempts: u32) -> Self {
        Self { attempts }
    }

    /// The number of attempts made before giving up.
    #[must_use]
    pub fn attempts(&self) -> u32 {
        self.attempts
    }
}

impl fmt::Display for RetriesExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "retries exhausted after {} attempts", self.attempts)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// WaitError
// ────────────────────────────────────────────────────────────────────────────

/// Error from a wait operation.
///
/// Use [`kind()`](Self::kind) to determine the failure mode.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{WaitError, WaitErrorKind};
///
/// fn describe(err: &WaitError) -> String {
///     match err.kind() {
///         WaitErrorKind::UnexpectedStatus(details) => {
///             format!("cannot resume from status {}", details.status())
///         }
///         _ => format!("{err:#}"),
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct WaitError {
    kind: WaitErrorKind,
    source: Option<Source>,
}

impl WaitError {
    /// Returns the specific kind of wait error.
    #[must_use]
    pub fn kind(&self) -> &WaitErrorKind {
        &self.kind
    }

    /// Creates a `WaitError` from its kind and source (internal).
    pub(crate) fn new(kind: WaitErrorKind, source: Option<Source>) -> Self {
        Self { kind, source }
    }
}

impl fmt::Display for WaitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.kind, self.source())
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl Error for WaitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|s| s as &(dyn Error + 'static))
    }
}

/// Specific kinds of wait failures.
///
/// A checkpoint write failure is deliberately NOT a kind here: a failed
/// wait START write unwinds the handler through the unrecoverable path
/// (issue #43), so user code never observes it as a `WaitError`.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::WaitErrorKind;
///
/// fn is_unexpected_status(kind: &WaitErrorKind) -> bool {
///     matches!(kind, WaitErrorKind::UnexpectedStatus(_))
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum WaitErrorKind {
    /// The wait's checkpointed record carries a status the SDK cannot
    /// resume from.
    UnexpectedStatus(UnexpectedStatus),
}

impl fmt::Display for WaitErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedStatus(details) => write!(f, "{details}"),
        }
    }
}

/// Details of a [`WaitErrorKind::UnexpectedStatus`] failure.
#[derive(Debug)]
#[non_exhaustive]
pub struct UnexpectedStatus {
    status: String,
}

impl UnexpectedStatus {
    /// Creates the payload (internal).
    pub(crate) fn new(status: impl Into<String>) -> Self {
        Self {
            status: status.into(),
        }
    }

    /// The unexpected status found in the checkpoint log.
    #[must_use]
    pub fn status(&self) -> &str {
        &self.status
    }
}

impl fmt::Display for UnexpectedStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unexpected checkpointed status: {}", self.status)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// InvokeError
// ────────────────────────────────────────────────────────────────────────────

/// Error from an invoke operation.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{InvokeError, InvokeErrorKind};
///
/// fn missing_function(err: &InvokeError) -> Option<&str> {
///     match err.kind() {
///         InvokeErrorKind::FunctionNotFound(details) => Some(details.function_id()),
///         _ => None,
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct InvokeError {
    kind: InvokeErrorKind,
    source: Option<Source>,
}

impl InvokeError {
    /// Returns the specific kind of invoke error.
    #[must_use]
    pub fn kind(&self) -> &InvokeErrorKind {
        &self.kind
    }

    /// Creates an `InvokeError` from its kind and source (internal).
    pub(crate) fn new(kind: InvokeErrorKind, source: Option<Source>) -> Self {
        Self { kind, source }
    }
}

impl fmt::Display for InvokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.kind, self.source())
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl Error for InvokeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|s| s as &(dyn Error + 'static))
    }
}

/// Specific kinds of invoke failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::InvokeErrorKind;
///
/// fn failed_in_function(kind: &InvokeErrorKind) -> bool {
///     // Unit kind variants are `#[non_exhaustive]`: match them with
///     // `{ .. }` so a future field cannot break the pattern.
///     matches!(kind, InvokeErrorKind::FunctionFailed { .. })
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum InvokeErrorKind {
    /// The invoked function returned an error.
    #[non_exhaustive]
    FunctionFailed,
    /// The function could not be found or accessed.
    FunctionNotFound(FunctionNotFound),
    /// A serialization or deserialization error occurred.
    #[non_exhaustive]
    SerializationFailed,
}

impl fmt::Display for InvokeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FunctionFailed => write!(f, "function failed"),
            Self::FunctionNotFound(details) => write!(f, "{details}"),
            Self::SerializationFailed => write!(f, "serialization failed"),
        }
    }
}

/// Details of an [`InvokeErrorKind::FunctionNotFound`] failure.
#[derive(Debug)]
#[non_exhaustive]
pub struct FunctionNotFound {
    function_id: String,
}

impl FunctionNotFound {
    /// Creates the payload (internal).
    ///
    /// No production path constructs this variant today (the invoke
    /// replay path cannot distinguish a missing function from a failed
    /// one on the wire); tests and future classification use it.
    #[allow(dead_code)] // reason: constructed by tests; kept for future invoke classification
    pub(crate) fn new(function_id: impl Into<String>) -> Self {
        Self {
            function_id: function_id.into(),
        }
    }

    /// The function identifier that was not found.
    #[must_use]
    pub fn function_id(&self) -> &str {
        &self.function_id
    }
}

impl fmt::Display for FunctionNotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "function not found: {}", self.function_id)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CallbackError
// ────────────────────────────────────────────────────────────────────────────

/// Error from a callback operation.
///
/// Follows the graded external/internal split: external errors are
/// caused by the caller's system (timeout, invalid data), internal
/// errors are SDK/service failures. For an
/// [`ExternalFailure`](CallbackErrorKind::ExternalFailure), the wire
/// fields the external caller supplied are reachable through the error's
/// [`source()`](std::error::Error::source) (a [`ReplayedFailure`]) and
/// through [`OperationError::wire`].
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{CallbackError, CallbackErrorKind};
///
/// fn describe(err: &CallbackError) -> String {
///     match err.kind() {
///         CallbackErrorKind::TimedOut { .. } => "callback timed out".to_owned(),
///         // The externally reported fields travel on `source()`.
///         CallbackErrorKind::ExternalFailure { .. } => format!("{err:#}"),
///         _ => format!("{err:#}"),
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct CallbackError {
    kind: CallbackErrorKind,
    source: Option<Source>,
}

impl CallbackError {
    /// Returns the specific kind of callback error.
    #[must_use]
    pub fn kind(&self) -> &CallbackErrorKind {
        &self.kind
    }

    /// Creates a `CallbackError` from its kind and source (internal).
    pub(crate) fn new(kind: CallbackErrorKind, source: Option<Source>) -> Self {
        Self { kind, source }
    }
}

impl fmt::Display for CallbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.kind, self.source())
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl Error for CallbackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|s| s as &(dyn Error + 'static))
    }
}

/// Specific kinds of callback failures.
///
/// The external/internal split separates user-attributable errors
/// (timeout, invalid payload) from SDK/service failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::CallbackErrorKind;
///
/// fn is_external(kind: &CallbackErrorKind) -> bool {
///     matches!(
///         kind,
///         CallbackErrorKind::TimedOut { .. }
///             | CallbackErrorKind::HeartbeatTimedOut { .. }
///             | CallbackErrorKind::ExternalFailure { .. }
///     )
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum CallbackErrorKind {
    /// The callback timed out waiting for completion.
    #[non_exhaustive]
    TimedOut,
    /// The callback heartbeat timed out (no keep-alive received in time).
    #[non_exhaustive]
    HeartbeatTimedOut,
    /// The callback was completed with an external failure.
    ///
    /// The failure fields the external caller reported travel as wire
    /// data: read them from the error's
    /// [`source()`](std::error::Error::source) (a [`ReplayedFailure`]) or
    /// from [`OperationError::wire`].
    #[non_exhaustive]
    ExternalFailure,
    /// The callback payload could not be deserialized.
    #[non_exhaustive]
    DeserializationFailed,
    /// An internal SDK or service error occurred.
    #[non_exhaustive]
    Internal,
}

impl fmt::Display for CallbackErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedOut => write!(f, "callback timed out"),
            Self::HeartbeatTimedOut => write!(f, "callback heartbeat timed out"),
            Self::ExternalFailure => write!(f, "external failure"),
            Self::DeserializationFailed => write!(f, "deserialization failed"),
            Self::Internal => write!(f, "internal error"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// WaitForConditionError
// ────────────────────────────────────────────────────────────────────────────

/// Error from a wait-for-condition operation.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{WaitForConditionError, WaitForConditionErrorKind};
///
/// fn checks_used(err: &WaitForConditionError) -> Option<u32> {
///     match err.kind() {
///         WaitForConditionErrorKind::MaxChecksExceeded(details) => Some(details.checks()),
///         _ => None,
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct WaitForConditionError {
    kind: WaitForConditionErrorKind,
    source: Option<Source>,
}

impl WaitForConditionError {
    /// Returns the specific kind of wait-for-condition error.
    #[must_use]
    pub fn kind(&self) -> &WaitForConditionErrorKind {
        &self.kind
    }

    /// Creates a `WaitForConditionError` from its kind and source (internal).
    pub(crate) fn new(kind: WaitForConditionErrorKind, source: Option<Source>) -> Self {
        Self { kind, source }
    }
}

impl fmt::Display for WaitForConditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.kind, self.source())
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl Error for WaitForConditionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|s| s as &(dyn Error + 'static))
    }
}

/// Specific kinds of wait-for-condition failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::WaitForConditionErrorKind;
///
/// fn check_function_failed(kind: &WaitForConditionErrorKind) -> bool {
///     matches!(kind, WaitForConditionErrorKind::CheckFailed { .. })
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum WaitForConditionErrorKind {
    /// The check function returned an error.
    #[non_exhaustive]
    CheckFailed,
    /// The maximum number of checks was exceeded.
    MaxChecksExceeded(MaxChecksExceeded),
    /// A serialization error occurred on the state.
    #[non_exhaustive]
    SerializationFailed,
}

impl fmt::Display for WaitForConditionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CheckFailed => write!(f, "check failed"),
            Self::MaxChecksExceeded(details) => write!(f, "{details}"),
            Self::SerializationFailed => write!(f, "serialization failed"),
        }
    }
}

/// Details of a [`WaitForConditionErrorKind::MaxChecksExceeded`] failure.
#[derive(Debug)]
#[non_exhaustive]
pub struct MaxChecksExceeded {
    checks: u32,
}

impl MaxChecksExceeded {
    /// Creates the payload (internal).
    pub(crate) fn new(checks: u32) -> Self {
        Self { checks }
    }

    /// The number of checks performed.
    #[must_use]
    pub fn checks(&self) -> u32 {
        self.checks
    }
}

impl fmt::Display for MaxChecksExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "max checks exceeded: {}", self.checks)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ChildContextError
// ────────────────────────────────────────────────────────────────────────────

/// Error from a child context (sub-orchestration) operation.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{ChildContextError, ChildContextErrorKind};
///
/// fn child_body_failed(err: &ChildContextError) -> bool {
///     matches!(err.kind(), ChildContextErrorKind::ChildFailed { .. })
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct ChildContextError {
    kind: ChildContextErrorKind,
    source: Option<Source>,
}

impl ChildContextError {
    /// Returns the specific kind of child context error.
    #[must_use]
    pub fn kind(&self) -> &ChildContextErrorKind {
        &self.kind
    }

    /// Creates a `ChildContextError` from its kind and source (internal).
    pub(crate) fn new(kind: ChildContextErrorKind, source: Option<Source>) -> Self {
        Self { kind, source }
    }
}

impl fmt::Display for ChildContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.kind, self.source())
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl Error for ChildContextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|s| s as &(dyn Error + 'static))
    }
}

/// Specific kinds of child context failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::ChildContextErrorKind;
///
/// fn is_internal(kind: &ChildContextErrorKind) -> bool {
///     matches!(kind, ChildContextErrorKind::Internal { .. })
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum ChildContextErrorKind {
    /// The child function returned an error.
    #[non_exhaustive]
    ChildFailed,
    /// An internal error occurred setting up the child context.
    #[non_exhaustive]
    Internal,
}

impl fmt::Display for ChildContextErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChildFailed => write!(f, "child failed"),
            Self::Internal => write!(f, "internal error"),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CombinatorError
// ────────────────────────────────────────────────────────────────────────────

/// Error from a combinator operation (`try_join_all`, `join_all`,
/// `select_ok`, `race`).
///
/// The combinator preserves the losing futures' errors as errors:
/// [`source()`](std::error::Error::source) returns the first loser, and
/// [`failures()`](Self::failures) returns every loser (in input order for
/// [`CombinatorErrorKind::AllFailed`]).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{CombinatorError, CombinatorErrorKind};
///
/// fn report(err: &CombinatorError) {
///     if let CombinatorErrorKind::AllFailed { .. } = err.kind() {
///         for loser in err.failures() {
///             tracing::error!(error = format!("{loser}"), "branch failed");
///         }
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct CombinatorError {
    kind: CombinatorErrorKind,
    failures: Vec<Source>,
}

impl CombinatorError {
    /// Creates a `CombinatorError` from a kind and its losing errors
    /// (internal).
    pub(crate) fn new(kind: CombinatorErrorKind, failures: Vec<Source>) -> Self {
        Self { kind, failures }
    }

    /// Returns the specific kind of combinator error.
    #[must_use]
    pub fn kind(&self) -> &CombinatorErrorKind {
        &self.kind
    }

    /// The underlying failures, preserved as errors.
    ///
    /// For [`CombinatorErrorKind::JoinFailed`] and
    /// [`CombinatorErrorKind::FirstSettledFailed`] this holds the single
    /// losing error; for [`CombinatorErrorKind::AllFailed`] it holds every
    /// loser in input order. After a replay, each entry is a
    /// [`ReplayedFailure`] rebuilt from the recorded wire fields.
    #[must_use]
    pub fn failures(&self) -> &[Box<dyn Error + Send + Sync + 'static>] {
        &self.failures
    }
}

impl fmt::Display for CombinatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &self.kind, self.source())
        } else {
            write!(f, "{}", self.kind)
        }
    }
}

impl Error for CombinatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.failures
            .first()
            .map(|s| &**s as &(dyn Error + 'static))
    }
}

/// Specific kinds of combinator failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::CombinatorErrorKind;
///
/// fn first_failed_index(kind: &CombinatorErrorKind) -> Option<usize> {
///     match kind {
///         CombinatorErrorKind::JoinFailed(details) => Some(details.failed_index()),
///         _ => None,
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum CombinatorErrorKind {
    /// One or more futures in `try_join_all` failed.
    JoinFailed(JoinFailed),
    /// All futures in `select_ok` failed.
    ///
    /// Every loser's error is reachable through
    /// [`CombinatorError::failures`].
    #[non_exhaustive]
    AllFailed,
    /// The first settled future in `race` was a failure.
    ///
    /// `race` propagates the first outcome to settle, success or failure.
    /// When that outcome is a failure, the losing future's error is
    /// carried as the error's source.
    #[non_exhaustive]
    FirstSettledFailed,
    /// The combinator was called with no futures.
    ///
    /// Returned by `race` and `select_ok`, which cannot produce a winner
    /// from an empty input. `try_join_all` and `join_all` instead resolve
    /// successfully to an empty collection.
    #[non_exhaustive]
    EmptyInput,
    /// An internal error occurred.
    #[non_exhaustive]
    Internal,
}

impl fmt::Display for CombinatorErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JoinFailed(details) => write!(f, "{details}"),
            Self::AllFailed => write!(f, "all futures failed"),
            Self::FirstSettledFailed => write!(f, "first settled future failed"),
            Self::EmptyInput => write!(f, "combinator called with no futures"),
            Self::Internal => write!(f, "internal error"),
        }
    }
}

/// Details of a [`CombinatorErrorKind::JoinFailed`] failure.
#[derive(Debug)]
#[non_exhaustive]
pub struct JoinFailed {
    failed_index: usize,
}

impl JoinFailed {
    /// Creates the payload (internal).
    pub(crate) fn new(failed_index: usize) -> Self {
        Self { failed_index }
    }

    /// The index of the first failed future in the combinator's input.
    #[must_use]
    pub fn failed_index(&self) -> usize {
        self.failed_index
    }
}

impl fmt::Display for JoinFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "join failed at index {}", self.failed_index)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// NonDeterministicExecutionError
// ────────────────────────────────────────────────────────────────────────────

/// Error raised when a replay operation does not match the checkpointed
/// history — the handler produced operations in a different order between
/// invocations.
///
/// This is a fatal, non-recoverable error: the execution must be failed and
/// cannot be retried without resetting the checkpoint log.
///
/// There is no foreign cause to carry — the mismatch report *is* the
/// error — so this type has no source; its structural facts live behind
/// the [`OperationMismatch`] payload's accessors.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{
///     NonDeterministicExecutionError, NonDeterministicExecutionErrorKind,
/// };
///
/// fn report(err: &NonDeterministicExecutionError) {
///     match err.kind() {
///         NonDeterministicExecutionErrorKind::OperationMismatch(details) => {
///             tracing::error!(
///                 wire_id = details.wire_id(),
///                 expected = details.expected(),
///                 actual = details.actual(),
///                 "non-deterministic replay detected"
///             );
///         }
///         // The enum is `#[non_exhaustive]`: new mismatch kinds may be
///         // added without a major version bump.
///         _ => tracing::error!("non-deterministic replay detected"),
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct NonDeterministicExecutionError {
    kind: NonDeterministicExecutionErrorKind,
}

impl NonDeterministicExecutionError {
    /// Returns the specific kind of non-determinism error.
    #[must_use]
    pub fn kind(&self) -> &NonDeterministicExecutionErrorKind {
        &self.kind
    }

    /// Creates a `NonDeterministicExecutionError` from its kind (internal).
    pub(crate) fn from_kind(kind: NonDeterministicExecutionErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for NonDeterministicExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl Error for NonDeterministicExecutionError {}

/// Specific kinds of non-deterministic execution failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::NonDeterministicExecutionErrorKind;
///
/// fn mismatch_slot(kind: &NonDeterministicExecutionErrorKind) -> Option<&str> {
///     match kind {
///         NonDeterministicExecutionErrorKind::OperationMismatch(details) => {
///             Some(details.wire_id())
///         }
///         _ => None,
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum NonDeterministicExecutionErrorKind {
    /// The claimed operation's identity (type, sub-type, or name) does not
    /// match the checkpointed record at the same positional slot.
    OperationMismatch(OperationMismatch),
}

impl fmt::Display for NonDeterministicExecutionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperationMismatch(details) => write!(f, "{details}"),
        }
    }
}

/// Details of a [`NonDeterministicExecutionErrorKind::OperationMismatch`]
/// failure.
#[derive(Debug)]
#[non_exhaustive]
pub struct OperationMismatch {
    wire_id: String,
    expected: String,
    actual: String,
}

impl OperationMismatch {
    /// Creates the payload (internal).
    pub(crate) fn new(
        wire_id: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self {
            wire_id: wire_id.into(),
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// The wire ID (SHA-256 hex) of the positional slot that mismatched.
    #[must_use]
    pub fn wire_id(&self) -> &str {
        &self.wire_id
    }

    /// Human-readable description of what the checkpoint log expected.
    #[must_use]
    pub fn expected(&self) -> &str {
        &self.expected
    }

    /// Human-readable description of what the handler claimed.
    #[must_use]
    pub fn actual(&self) -> &str {
        &self.actual
    }
}

impl fmt::Display for OperationMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "operation at wire id {} does not match checkpoint: \
             expected {}, got {}",
            self.wire_id, self.expected, self.actual
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ChildFnError
// ────────────────────────────────────────────────────────────────────────────

/// Crate-internal error carrier for `run_in_child_context`, `parallel`, and
/// `map` closure bodies.
///
/// Public closure boundaries take [`crate::BoxError`]; the SDK converts a
/// `BoxError` into this type at the boundary and grades it into
/// [`ChildContextErrorKind::ChildFailed`] when a child body fails. It
/// carries the escaping error as its source rather than flattening it, so
/// the child-boundary pass-through guarantee (`error_data`, downcastable
/// cause) holds. It is not part of the public API.
#[derive(Debug)]
pub(crate) struct ChildFnError {
    source: Source,
}

impl ChildFnError {
    /// Creates a new `ChildFnError` carrying the given source.
    #[must_use]
    pub(crate) fn new(source: impl Into<Source>) -> Self {
        Self {
            source: source.into(),
        }
    }

    /// Moves the carried error out (internal).
    pub(crate) fn into_source(self) -> Source {
        self.source
    }
}

impl fmt::Display for ChildFnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write_chain(f, &"child function error", self.source())
        } else {
            write!(f, "child function error")
        }
    }
}

impl Error for ChildFnError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&*self.source as &(dyn Error + 'static))
    }
}

impl From<OperationError> for ChildFnError {
    fn from(err: OperationError) -> Self {
        Self {
            source: Box::new(err),
        }
    }
}

impl From<crate::BoxError> for ChildFnError {
    /// Converts a user-facing [`crate::BoxError`] into the internal
    /// carrier, preserving the error itself as the source.
    fn from(err: crate::BoxError) -> Self {
        Self { source: err }
    }
}

// Static assertions: all public error types must be Send + Sync + 'static.
// These compile-time checks prevent accidental regressions — the `source`
// fields' `Send + Sync` bounds are what keep them true.
const _: () = {
    #[allow(dead_code)] // reason: compile-time assertion, never called at runtime
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    #[allow(dead_code)] // reason: compile-time assertion, existence proves bounds hold
    fn assert_bounds() {
        assert_send_sync_static::<OperationError>();
        assert_send_sync_static::<OperationErrorKind>();
        assert_send_sync_static::<StepError>();
        assert_send_sync_static::<StepErrorKind>();
        assert_send_sync_static::<RetriesExhausted>();
        assert_send_sync_static::<WaitError>();
        assert_send_sync_static::<WaitErrorKind>();
        assert_send_sync_static::<UnexpectedStatus>();
        assert_send_sync_static::<InvokeError>();
        assert_send_sync_static::<InvokeErrorKind>();
        assert_send_sync_static::<FunctionNotFound>();
        assert_send_sync_static::<CallbackError>();
        assert_send_sync_static::<CallbackErrorKind>();
        assert_send_sync_static::<WaitForConditionError>();
        assert_send_sync_static::<WaitForConditionErrorKind>();
        assert_send_sync_static::<MaxChecksExceeded>();
        assert_send_sync_static::<ChildContextError>();
        assert_send_sync_static::<ChildContextErrorKind>();
        assert_send_sync_static::<CombinatorError>();
        assert_send_sync_static::<CombinatorErrorKind>();
        assert_send_sync_static::<JoinFailed>();
        assert_send_sync_static::<NonDeterministicExecutionError>();
        assert_send_sync_static::<NonDeterministicExecutionErrorKind>();
        assert_send_sync_static::<OperationMismatch>();
        assert_send_sync_static::<WireError>();
        assert_send_sync_static::<TypedError>();
        assert_send_sync_static::<ReplayedFailure>();
        assert_send_sync_static::<ContextualError>();
        assert_send_sync_static::<ChildFnError>();
        assert_send_sync_static::<crate::serdes::FileSystemSerdesError>();
    }
};

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions with descriptive messages
mod tests {
    use super::*;
    /// A concrete user error type for downcast tests.
    #[derive(Debug)]
    struct UserBoom {
        detail: &'static str,
    }
    impl fmt::Display for UserBoom {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "user boom: {}", self.detail)
        }
    }
    impl Error for UserBoom {}

    fn step_error_with_user_source() -> OperationError {
        OperationError::from_kind(OperationErrorKind::Step(StepError::new(
            StepErrorKind::ExecutionFailed,
            Some(Box::new(UserBoom { detail: "flaky" })),
        )))
    }

    /// Walks the full `source()` chain and returns each layer's display.
    fn causal_chain(err: &dyn Error) -> Vec<String> {
        let mut chain = vec![err.to_string()];
        let mut current: &dyn Error = err;
        while let Some(next) = current.source() {
            chain.push(next.to_string());
            current = next;
        }
        chain
    }

    // ── Acceptance: live-path downcast ──────────────────────────────────

    #[test]
    fn live_failure_exposes_concrete_error_type_through_source_downcast() {
        let err = step_error_with_user_source();
        // Walk the chain until the caller's concrete type is found.
        let mut current: Option<&(dyn Error + 'static)> = err.source();
        let mut found = None;
        while let Some(e) = current {
            if let Some(user) = e.downcast_ref::<UserBoom>() {
                found = Some(user);
                break;
            }
            current = e.source();
        }
        let user = found.expect("caller's concrete error must be reachable via source()");
        assert_eq!(user.detail, "flaky");
    }

    // ── Acceptance: Display frames, alternate chain, no repeats ────────

    #[test]
    fn display_writes_one_frame_and_alternate_walks_chain() {
        let err = step_error_with_user_source();

        // `{}` yields one frame — no cause text.
        let plain = format!("{err}");
        assert_eq!(plain, "operation error: step");

        // `{:#}` yields the chain.
        let alternate = format!("{err:#}");
        assert_eq!(
            alternate,
            "operation error: step: execution failed: user boom: flaky"
        );
    }

    #[test]
    fn walking_source_does_not_repeat_text() {
        let err = step_error_with_user_source();
        let chain = causal_chain(&err);
        // OperationError -> StepError -> UserBoom.
        assert_eq!(
            chain,
            vec![
                "operation error: step".to_owned(),
                "execution failed".to_owned(),
                "user boom: flaky".to_owned(),
            ]
        );
        // No frame's text appears in another frame.
        for (i, frame) in chain.iter().enumerate() {
            for (j, other) in chain.iter().enumerate() {
                if i != j {
                    assert!(
                        !other.contains(frame.as_str()),
                        "frame {i} ({frame:?}) repeats inside frame {j} ({other:?})"
                    );
                }
            }
        }
    }

    // ── Replay source ────────────────────────────────────────────────────

    #[test]
    fn replayed_failure_carries_wire_fields() {
        let wire = WireError::new(Some("NotApproved"), Some("request was denied"))
            .with_error_data(Some("{\"code\":42}"));
        let err = OperationError::from_kind(OperationErrorKind::Step(StepError::new(
            StepErrorKind::ExecutionFailed,
            Some(ReplayedFailure::source_from(wire)),
        )));
        // kind() is meaningful after a replay.
        let OperationErrorKind::Step(step_err) = err.kind() else {
            unreachable!("constructed as a step error");
        };
        assert!(matches!(step_err.kind(), StepErrorKind::ExecutionFailed));
        // The synthetic source carries the wire error_type and message.
        let source = Error::source(step_err).expect("replay attaches a source");
        let replayed = source
            .downcast_ref::<ReplayedFailure>()
            .expect("replay source must be a ReplayedFailure");
        assert_eq!(replayed.error_type(), Some("NotApproved"));
        assert_eq!(replayed.error_message(), Some("request was denied"));
        assert_eq!(replayed.wire().error_data(), Some("{\"code\":42}"));
    }

    // ── Wire derivation ──────────────────────────────────────────────────

    #[test]
    fn wire_error_for_flattens_chain_once_and_uses_fallback_type() {
        let err = step_error_with_user_source();
        let wire = wire_error_for(&err, "Error");
        // The message is the flattened chain (single flattening site).
        assert_eq!(
            wire.error_message(),
            Some("operation error: step: execution failed: user boom: flaky")
        );
        // No structured identity in the chain — falls back to the kind's
        // wire name because the top error is an OperationError.
        assert_eq!(wire.error_type(), Some("StepError"));
        // A fresh failure captures a stack trace.
        assert!(!wire.stack_trace().is_empty());
        // No error_data anywhere in the chain — none synthesized.
        assert_eq!(wire.error_data(), None);
    }

    #[test]
    fn wire_error_for_plain_user_error_uses_fallback() {
        let err = UserBoom { detail: "plain" };
        let wire = wire_error_for(&err, "Error");
        assert_eq!(wire.error_type(), Some("Error"));
        assert_eq!(wire.error_message(), Some("user boom: plain"));
    }

    #[test]
    fn error_data_survives_child_boundary_via_cause_chain() {
        // An inner failure that carries wire data (e.g. an external
        // callback failure read back from the wire)...
        let inner_wire = WireError::new(Some("VendorError"), Some("denied"))
            .with_error_data(Some("opaque-payload"));
        let inner = OperationError::from_kind(OperationErrorKind::Callback(CallbackError::new(
            CallbackErrorKind::ExternalFailure,
            Some(ReplayedFailure::source_from(inner_wire.clone())),
        )))
        .with_wire(inner_wire);

        // ...escapes through a child-context boundary.
        let child_fn = ChildFnError::from(inner);
        let wire = wire_error_for(&child_fn, "ChildFnError");

        // The opaque payload survives the boundary via the cause chain.
        assert_eq!(wire.error_data(), Some("opaque-payload"));
        // It is passed through, never parsed: the value is byte-identical.
        assert_eq!(wire.error_data(), Some("opaque-payload"));
    }

    #[test]
    fn wire_error_re_records_replayed_failure_type() {
        // Re-recording a replayed failure directly preserves its type...
        let recorded = WireError::new(Some("MyDomainError"), Some("原因"));
        let replayed = ReplayedFailure::new(recorded.clone());
        let wire = wire_error_for(&replayed, "Error");
        assert_eq!(wire.error_type(), Some("MyDomainError"));

        // ...and an operation error wrapping it re-records its attached
        // wire identity — the explicitly supplied type survives the
        // boundary instead of degrading to the kind's registry name.
        let err = OperationError::from_kind(OperationErrorKind::Step(StepError::new(
            StepErrorKind::ExecutionFailed,
            Some(ReplayedFailure::source_from(recorded.clone())),
        )))
        .with_wire(recorded);
        let wire = wire_error_for(&err, "Error");
        assert_eq!(wire.error_type(), Some("MyDomainError"));
        assert_eq!(
            err.wire().and_then(WireError::error_type),
            Some("MyDomainError")
        );

        // An operation error with no attached wire and no typed identity
        // in its chain still falls back to the kind's registry name.
        let bare = OperationError::from_kind(OperationErrorKind::Step(StepError::new(
            StepErrorKind::ExecutionFailed,
            Some(Source::from("plain failure")),
        )));
        let wire = wire_error_for(&bare, "Error");
        assert_eq!(wire.error_type(), Some("StepError"));
    }

    // ── Structural facts behind accessors ────────────────────────────────

    #[test]
    fn step_error_kind_accessor() {
        let err = StepError::new(
            StepErrorKind::RetriesExhausted(RetriesExhausted::new(3)),
            Some(Source::from("fail")),
        );
        match err.kind() {
            StepErrorKind::RetriesExhausted(details) => assert_eq!(details.attempts(), 3),
            other => unreachable!("unexpected kind: {other:?}"),
        }
    }

    #[test]
    fn operation_mismatch_accessors() {
        let details = OperationMismatch::new("abc123", "Step/Step", "Wait/Wait");
        assert_eq!(details.wire_id(), "abc123");
        assert_eq!(details.expected(), "Step/Step");
        assert_eq!(details.actual(), "Wait/Wait");
        let err = NonDeterministicExecutionError::from_kind(
            NonDeterministicExecutionErrorKind::OperationMismatch(details),
        );
        // The mismatch report is the error: no source to carry.
        assert!(Error::source(&err).is_none());
        assert!(err.to_string().contains("abc123"));
    }

    // ── Combinators keep every loser ─────────────────────────────────────

    #[test]
    fn combinator_all_failed_keeps_every_loser() {
        let losers: Vec<Source> = vec![
            Box::new(UserBoom { detail: "a" }),
            Box::new(UserBoom { detail: "b" }),
        ];
        let err = CombinatorError::new(CombinatorErrorKind::AllFailed, losers);
        assert_eq!(err.failures().len(), 2);
        // source() returns the first loser.
        let first = Error::source(&err).expect("has a source");
        assert!(first.downcast_ref::<UserBoom>().is_some());
        // Every loser is reachable, as an error, not a string.
        let details: Vec<&str> = err
            .failures()
            .iter()
            .filter_map(|l| l.downcast_ref::<UserBoom>().map(|u| u.detail))
            .collect();
        assert_eq!(details, vec!["a", "b"]);
    }

    #[test]
    fn combinator_join_failed_keeps_loser_and_index() {
        let err = CombinatorError::new(
            CombinatorErrorKind::JoinFailed(JoinFailed::new(2)),
            vec![Box::new(UserBoom { detail: "loser" })],
        );
        match err.kind() {
            CombinatorErrorKind::JoinFailed(details) => assert_eq!(details.failed_index(), 2),
            other => unreachable!("unexpected kind: {other:?}"),
        }
        let source = Error::source(&err).expect("loser preserved");
        assert!(source.downcast_ref::<UserBoom>().is_some());
    }

    // ── ChildFnError carries the error ───────────────────────────────────

    #[test]
    fn child_fn_error_preserves_operation_error() {
        let op = step_error_with_user_source();
        let child_err = ChildFnError::from(op);
        let source = Error::source(&child_err).expect("carries the error");
        assert!(source.downcast_ref::<OperationError>().is_some());
        // Frame stays clean; the chain is reachable via {:#}.
        assert_eq!(child_err.to_string(), "child function error");
        assert!(format!("{child_err:#}").contains("user boom: flaky"));
    }

    // ── Display conventions across variants ──────────────────────────────

    #[test]
    fn every_operation_family_displays_its_frame() {
        let cases: Vec<(OperationError, &str)> = vec![
            (
                OperationError::from_kind(OperationErrorKind::Step(StepError::new(
                    StepErrorKind::ExecutionFailed,
                    None,
                ))),
                "step",
            ),
            (
                OperationError::from_kind(OperationErrorKind::Wait(WaitError::new(
                    WaitErrorKind::UnexpectedStatus(UnexpectedStatus::new("Failed")),
                    None,
                ))),
                "wait",
            ),
            (
                OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::new(
                    InvokeErrorKind::FunctionNotFound(FunctionNotFound::new("f")),
                    None,
                ))),
                "invoke",
            ),
            (
                OperationError::from_kind(OperationErrorKind::Callback(CallbackError::new(
                    CallbackErrorKind::TimedOut,
                    None,
                ))),
                "callback",
            ),
            (
                OperationError::from_kind(OperationErrorKind::WaitForCondition(
                    WaitForConditionError::new(
                        WaitForConditionErrorKind::MaxChecksExceeded(MaxChecksExceeded::new(2)),
                        None,
                    ),
                )),
                "wait_for_condition",
            ),
            (
                OperationError::from_kind(OperationErrorKind::ChildContext(
                    ChildContextError::new(ChildContextErrorKind::ChildFailed, None),
                )),
                "child_context",
            ),
            (
                OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec![Box::new(UserBoom { detail: "x" })],
                ))),
                "combinator",
            ),
            (
                OperationError::from_kind(OperationErrorKind::NonDeterministicExecution(
                    NonDeterministicExecutionError::from_kind(
                        NonDeterministicExecutionErrorKind::OperationMismatch(
                            OperationMismatch::new("abc123", "Step/Step", "Wait/Wait"),
                        ),
                    ),
                )),
                "non_deterministic_execution",
            ),
        ];
        for (err, family) in &cases {
            assert_eq!(format!("{err}"), format!("operation error: {family}"));
            // The chain never repeats text across frames.
            let chain = causal_chain(err);
            for (i, frame) in chain.iter().enumerate() {
                for (j, other) in chain.iter().enumerate() {
                    if i != j {
                        assert!(
                            !other.contains(frame.as_str()),
                            "family {family}: frame {i} repeats in frame {j}: {chain:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn callback_error_graded_split() {
        let timed_out = CallbackError::new(CallbackErrorKind::TimedOut, None);
        assert!(matches!(timed_out.kind(), CallbackErrorKind::TimedOut));

        let heartbeat = CallbackError::new(CallbackErrorKind::HeartbeatTimedOut, None);
        assert!(matches!(
            heartbeat.kind(),
            CallbackErrorKind::HeartbeatTimedOut
        ));

        let wire = WireError::new(Some("UserError"), Some("denied"));
        let external = CallbackError::new(
            CallbackErrorKind::ExternalFailure,
            Some(ReplayedFailure::source_from(wire)),
        );
        assert!(matches!(
            external.kind(),
            CallbackErrorKind::ExternalFailure
        ));
        // The wire fields the external caller reported are on the source.
        let source = Error::source(&external).expect("external carries wire source");
        let replayed = source.downcast_ref::<ReplayedFailure>().expect("replayed");
        assert_eq!(replayed.error_type(), Some("UserError"));

        let internal = CallbackError::new(CallbackErrorKind::Internal, Some(Source::from("oops")));
        assert!(matches!(internal.kind(), CallbackErrorKind::Internal));
    }

    #[test]
    fn operation_context_is_reachable() {
        let wire = WireError::new(Some("StepError"), Some("execution failed: boom"));
        let err = step_error_with_user_source()
            .with_operation("abc123", "FAILED")
            .with_wire(wire);
        assert_eq!(err.operation_id(), Some("abc123"));
        assert_eq!(err.status(), Some("FAILED"));
        assert_eq!(
            err.wire().and_then(WireError::error_type),
            Some("StepError")
        );
    }

    #[test]
    fn typed_error_names_the_users_type_on_the_wire() {
        let err = TypedError::new(UserBoom { detail: "typed" });
        assert_eq!(err.error_type(), "UserBoom");

        // The wire record derives the user's type from the wrapper...
        let wire = wire_error_for(&err, "Error");
        assert_eq!(wire.error_type(), Some("UserBoom"));
        assert_eq!(wire.error_message(), Some("UserBoom: user boom: typed"));

        // ...including through intermediate carriers (a child boundary).
        let child =
            ChildFnError::new(Box::new(TypedError::new(UserBoom { detail: "deep" })) as Source);
        let wire = wire_error_for(&child, "ChildFnError");
        assert_eq!(wire.error_type(), Some("UserBoom"));
    }

    #[test]
    fn typed_error_keeps_concrete_error_downcastable() {
        let err = TypedError::with_type("TransientError", UserBoom { detail: "x" });
        // The wrapped error is the next chain link — downcast reaches it.
        let source = Error::source(&err).expect("wraps the error");
        assert!(source.downcast_ref::<UserBoom>().is_some());
        // The chain names the type once, then the error's own frame.
        assert_eq!(format!("{err:#}"), "TransientError: user boom: x");
    }

    #[test]
    fn short_type_name_strips_module_paths() {
        assert_eq!(
            short_type_name("my_crate::errors::TransientError"),
            "TransientError"
        );
        assert_eq!(short_type_name("TransientError"), "TransientError");
        assert_eq!(
            short_type_name("wrapper::Holder<inner::Thing, other::T>"),
            "Holder<Thing, T>"
        );
    }

    #[test]
    fn contextual_error_adds_frame_without_flattening() {
        let source = ContextualError::source_from("checkpoint succeed", UserBoom { detail: "io" });
        assert_eq!(source.to_string(), "checkpoint succeed");
        let inner = source.source().expect("keeps cause");
        assert_eq!(inner.to_string(), "user boom: io");
    }
}
