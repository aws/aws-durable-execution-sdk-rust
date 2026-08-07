//! Error types for durable operations.
//!
//! All error types are hand-written (no `thiserror`), `#[non_exhaustive]`,
//! and implement [`std::error::Error`]. Use the `kind()` accessor pattern
//! for matching error variants.

use std::fmt;

/// The top-level error type returned by [`DurableFuture`](crate::DurableFuture).
///
/// Wraps a typed per-operation cause accessible via [`kind()`](Self::kind).
/// Match on the kind to determine the specific failure mode.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{OperationError, OperationErrorKind};
///
/// fn handle_error(err: OperationError) {
///     match err.kind() {
///         OperationErrorKind::Step(_) => tracing::error!("step failed"),
///         OperationErrorKind::Invoke(_) => tracing::error!("invoke failed"),
///         OperationErrorKind::Callback(_) => tracing::error!("callback failed"),
///         OperationErrorKind::WaitForCondition(_) => tracing::error!("condition failed"),
///         OperationErrorKind::ChildContext(_) => tracing::error!("child failed"),
///         OperationErrorKind::Combinator(_) => tracing::error!("combinator failed"),
///         _ => tracing::error!("other error: {err}"),
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct OperationError {
    kind: OperationErrorKind,
}

impl OperationError {
    /// Returns the specific error kind for this operation failure.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::{OperationError, OperationErrorKind};
    ///
    /// let err = OperationError::__test_error();
    /// match err.kind() {
    ///     OperationErrorKind::Step(_) => { /* handle step error */ }
    ///     _ => { /* handle other errors */ }
    /// }
    /// ```
    #[must_use]
    pub fn kind(&self) -> &OperationErrorKind {
        &self.kind
    }

    /// Creates an `OperationError` from its kind (internal).
    #[allow(dead_code)] // reason: used by enforce_task_ownership
    pub(crate) fn from_kind(kind: OperationErrorKind) -> Self {
        Self { kind }
    }

    /// Creates a test error (doc-hidden, for doctests only).
    #[doc(hidden)]
    #[must_use]
    pub fn __test_error() -> Self {
        Self {
            kind: OperationErrorKind::Step(StepError {
                kind: StepErrorKind::ExecutionFailed {
                    message: String::from("test"),
                },
            }),
        }
    }
}

impl fmt::Display for OperationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "operation error: {}", self.kind)
    }
}

impl std::error::Error for OperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Classification of operation errors by operation type.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::OperationErrorKind;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum OperationErrorKind {
    /// A step operation failed.
    Step(StepError),
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

impl fmt::Display for OperationErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Step(e) => write!(f, "step: {e}"),
            Self::Invoke(e) => write!(f, "invoke: {e}"),
            Self::Callback(e) => write!(f, "callback: {e}"),
            Self::WaitForCondition(e) => write!(f, "wait_for_condition: {e}"),
            Self::ChildContext(e) => write!(f, "child_context: {e}"),
            Self::Combinator(e) => write!(f, "combinator: {e}"),
            Self::NonDeterministicExecution(e) => write!(f, "non_deterministic_execution: {e}"),
        }
    }
}

impl std::error::Error for OperationErrorKind {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Step(e) => Some(e),
            Self::Invoke(e) => Some(e),
            Self::Callback(e) => Some(e),
            Self::WaitForCondition(e) => Some(e),
            Self::ChildContext(e) => Some(e),
            Self::Combinator(e) => Some(e),
            Self::NonDeterministicExecution(e) => Some(e),
        }
    }
}

// --- StepError ---

/// Error from a step operation.
///
/// Use [`kind()`](Self::kind) to determine the failure mode.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{StepError, StepErrorKind};
///
/// fn check_step_error(err: &StepError) {
///     match err.kind() {
///         StepErrorKind::ExecutionFailed { message } => {
///             tracing::error!(%message, "step execution failed");
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct StepError {
    kind: StepErrorKind,
}

impl StepError {
    /// Returns the specific kind of step error.
    #[must_use]
    pub fn kind(&self) -> &StepErrorKind {
        &self.kind
    }

    /// Creates a `StepError` from its kind (internal).
    #[allow(dead_code)] // reason: used by enforce_task_ownership
    pub(crate) fn from_kind(kind: StepErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for StepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Specific kinds of step failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::StepErrorKind;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum StepErrorKind {
    /// The step body returned an error.
    ExecutionFailed {
        /// The error message from the step body.
        message: String,
    },
    /// The step exceeded its maximum retry attempts.
    RetriesExhausted {
        /// Number of attempts made.
        attempts: u32,
        /// The last error message.
        last_error: String,
    },
    /// A serialization or deserialization error occurred.
    SerializationFailed {
        /// Description of the serialization failure.
        message: String,
    },
}

impl fmt::Display for StepErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExecutionFailed { message } => write!(f, "execution failed: {message}"),
            Self::RetriesExhausted {
                attempts,
                last_error,
            } => write!(
                f,
                "retries exhausted after {attempts} attempts: {last_error}"
            ),
            Self::SerializationFailed { message } => {
                write!(f, "serialization failed: {message}")
            }
        }
    }
}

impl std::error::Error for StepErrorKind {}

// --- InvokeError ---

/// Error from an invoke operation.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{InvokeError, InvokeErrorKind};
///
/// fn check_invoke_error(err: &InvokeError) {
///     match err.kind() {
///         InvokeErrorKind::FunctionFailed { .. } => {
///             tracing::error!("invoked function failed");
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct InvokeError {
    kind: InvokeErrorKind,
}

impl InvokeError {
    /// Returns the specific kind of invoke error.
    #[must_use]
    pub fn kind(&self) -> &InvokeErrorKind {
        &self.kind
    }

    /// Creates an `InvokeError` from its kind (internal).
    pub(crate) fn from_kind(kind: InvokeErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for InvokeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for InvokeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Specific kinds of invoke failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::InvokeErrorKind;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum InvokeErrorKind {
    /// The invoked function returned an error.
    FunctionFailed {
        /// Error message from the invoked function.
        message: String,
    },
    /// The function could not be found or accessed.
    FunctionNotFound {
        /// The function identifier that was not found.
        function_id: String,
    },
    /// A serialization or deserialization error occurred.
    SerializationFailed {
        /// Description of the serialization failure.
        message: String,
    },
}

impl fmt::Display for InvokeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FunctionFailed { message } => write!(f, "function failed: {message}"),
            Self::FunctionNotFound { function_id } => {
                write!(f, "function not found: {function_id}")
            }
            Self::SerializationFailed { message } => {
                write!(f, "serialization failed: {message}")
            }
        }
    }
}

impl std::error::Error for InvokeErrorKind {}

// --- CallbackError ---

/// Error from a callback operation.
///
/// Follows the graded external/internal split: external errors are
/// caused by the caller's system (timeout, invalid data), internal
/// errors are SDK/service failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{CallbackError, CallbackErrorKind};
///
/// fn check_callback_error(err: &CallbackError) {
///     match err.kind() {
///         CallbackErrorKind::TimedOut { .. } => {
///             tracing::warn!("callback timed out");
///         }
///         CallbackErrorKind::DeserializationFailed { .. } => {
///             tracing::error!("invalid callback payload");
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct CallbackError {
    kind: CallbackErrorKind,
}

impl CallbackError {
    /// Returns the specific kind of callback error.
    #[must_use]
    pub fn kind(&self) -> &CallbackErrorKind {
        &self.kind
    }

    /// Creates a `CallbackError` from its kind (internal).
    pub(crate) fn from_kind(kind: CallbackErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for CallbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for CallbackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
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
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum CallbackErrorKind {
    /// The callback timed out waiting for completion.
    TimedOut,
    /// The callback heartbeat timed out (no keep-alive received in time).
    HeartbeatTimedOut,
    /// The callback was completed with an external failure.
    ExternalFailure {
        /// The error type reported by the external system.
        error_type: String,
        /// The error message reported by the external system.
        message: String,
    },
    /// The callback payload could not be deserialized.
    DeserializationFailed {
        /// Description of the deserialization failure.
        message: String,
    },
    /// An internal SDK or service error occurred.
    Internal {
        /// Description of the internal failure.
        message: String,
    },
}

impl fmt::Display for CallbackErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedOut => write!(f, "callback timed out"),
            Self::HeartbeatTimedOut => write!(f, "callback heartbeat timed out"),
            Self::ExternalFailure {
                error_type,
                message,
            } => write!(f, "external failure ({error_type}): {message}"),
            Self::DeserializationFailed { message } => {
                write!(f, "deserialization failed: {message}")
            }
            Self::Internal { message } => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for CallbackErrorKind {}

// --- WaitForConditionError ---

/// Error from a wait-for-condition operation.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{WaitForConditionError, WaitForConditionErrorKind};
///
/// fn check_condition_error(err: &WaitForConditionError) {
///     match err.kind() {
///         WaitForConditionErrorKind::CheckFailed { .. } => {
///             tracing::error!("condition check function failed");
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct WaitForConditionError {
    kind: WaitForConditionErrorKind,
}

impl WaitForConditionError {
    /// Returns the specific kind of wait-for-condition error.
    #[must_use]
    pub fn kind(&self) -> &WaitForConditionErrorKind {
        &self.kind
    }

    /// Creates a `WaitForConditionError` from its kind (internal).
    pub(crate) fn from_kind(kind: WaitForConditionErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for WaitForConditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for WaitForConditionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Specific kinds of wait-for-condition failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::WaitForConditionErrorKind;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum WaitForConditionErrorKind {
    /// The check function returned an error.
    CheckFailed {
        /// The error message from the check function.
        message: String,
    },
    /// The maximum number of checks was exceeded.
    MaxChecksExceeded {
        /// Number of checks performed.
        checks: u32,
    },
    /// A serialization error occurred on the state.
    SerializationFailed {
        /// Description of the failure.
        message: String,
    },
}

impl fmt::Display for WaitForConditionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CheckFailed { message } => write!(f, "check failed: {message}"),
            Self::MaxChecksExceeded { checks } => {
                write!(f, "max checks exceeded: {checks}")
            }
            Self::SerializationFailed { message } => {
                write!(f, "serialization failed: {message}")
            }
        }
    }
}

impl std::error::Error for WaitForConditionErrorKind {}

// --- ChildContextError ---

/// Error from a child context (sub-orchestration) operation.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{ChildContextError, ChildContextErrorKind};
///
/// fn check_child_error(err: &ChildContextError) {
///     match err.kind() {
///         ChildContextErrorKind::ChildFailed { .. } => {
///             tracing::error!("child orchestration failed");
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct ChildContextError {
    kind: ChildContextErrorKind,
}

impl ChildContextError {
    /// Returns the specific kind of child context error.
    #[must_use]
    pub fn kind(&self) -> &ChildContextErrorKind {
        &self.kind
    }

    /// Creates a `ChildContextError` from its kind (internal).
    pub(crate) fn from_kind(kind: ChildContextErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for ChildContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for ChildContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Specific kinds of child context failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::ChildContextErrorKind;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum ChildContextErrorKind {
    /// The child function returned an error.
    ChildFailed {
        /// The error message from the child function.
        message: String,
    },
    /// An internal error occurred setting up the child context.
    Internal {
        /// Description of the internal failure.
        message: String,
    },
}

impl fmt::Display for ChildContextErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChildFailed { message } => write!(f, "child failed: {message}"),
            Self::Internal { message } => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for ChildContextErrorKind {}

// --- CombinatorError ---

/// Error from a combinator operation (`try_join_all`, `join_all`,
/// `select_ok`, `race`).
///
/// Wraps failures from the individual futures within the combinator.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{CombinatorError, CombinatorErrorKind};
///
/// fn check_combinator_error(err: &CombinatorError) {
///     match err.kind() {
///         CombinatorErrorKind::AllFailed { .. } => {
///             tracing::error!("all futures in select_ok failed");
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub struct CombinatorError {
    kind: CombinatorErrorKind,
}

impl CombinatorError {
    /// Creates a `CombinatorError` from a specific kind (internal).
    pub(crate) fn from_kind(kind: CombinatorErrorKind) -> Self {
        Self { kind }
    }

    /// Returns the specific kind of combinator error.
    #[must_use]
    pub fn kind(&self) -> &CombinatorErrorKind {
        &self.kind
    }
}

impl fmt::Display for CombinatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for CombinatorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Specific kinds of combinator failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::CombinatorErrorKind;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum CombinatorErrorKind {
    /// One or more futures in `try_join_all` failed.
    JoinFailed {
        /// The index of the first failed future.
        failed_index: usize,
        /// The error message.
        message: String,
    },
    /// All futures in `select_ok` failed.
    AllFailed {
        /// Error messages from each future.
        errors: Vec<String>,
    },
    /// An internal error occurred.
    Internal {
        /// Description of the internal failure.
        message: String,
    },
}

impl fmt::Display for CombinatorErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JoinFailed {
                failed_index,
                message,
            } => {
                write!(f, "join failed at index {failed_index}: {message}")
            }
            Self::AllFailed { errors } => {
                write!(f, "all failed ({} errors)", errors.len())
            }
            Self::Internal { message } => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for CombinatorErrorKind {}

// --- NonDeterministicExecutionError ---

/// Error raised when a replay operation does not match the checkpointed
/// history — the handler produced operations in a different order between
/// invocations.
///
/// This is a fatal, non-recoverable error: the execution must be failed and
/// cannot be retried without resetting the checkpoint log.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::{
///     NonDeterministicExecutionError, NonDeterministicExecutionErrorKind,
/// };
///
/// fn check_nondeterminism(err: &NonDeterministicExecutionError) {
///     match err.kind() {
///         NonDeterministicExecutionErrorKind::OperationMismatch { wire_id, .. } => {
///             tracing::error!(%wire_id, "non-deterministic replay detected");
///         }
///         _ => {}
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
    #[allow(dead_code)] // reason: used by validate_replay_identity
    pub(crate) fn from_kind(kind: NonDeterministicExecutionErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for NonDeterministicExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for NonDeterministicExecutionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.kind)
    }
}

/// Specific kinds of non-deterministic execution failures.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust::NonDeterministicExecutionErrorKind;
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum NonDeterministicExecutionErrorKind {
    /// The claimed operation's identity (type, sub-type, or name) does not
    /// match the checkpointed record at the same positional slot.
    OperationMismatch {
        /// The wire ID (SHA-256 hex) of the positional slot.
        wire_id: String,
        /// Human-readable description of what was expected (from the
        /// checkpoint log).
        expected: String,
        /// Human-readable description of what the handler claimed.
        actual: String,
    },
}

impl fmt::Display for NonDeterministicExecutionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OperationMismatch {
                wire_id,
                expected,
                actual,
            } => write!(
                f,
                "operation at wire id {wire_id} does not match checkpoint: \
                 expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for NonDeterministicExecutionErrorKind {}

// --- ChildFnError ---

/// Crate-internal error carrier for `run_in_child_context`, `parallel`, and
/// `map` closure bodies.
///
/// Public closure boundaries take [`crate::BoxError`]; the SDK converts a
/// `BoxError` into this type at the boundary and grades it into
/// [`ChildContextErrorKind::ChildFailed`] when a child body fails. It is not
/// part of the public API.
#[derive(Debug)]
pub(crate) struct ChildFnError {
    message: String,
}

impl ChildFnError {
    /// Creates a new `ChildFnError` with the given message.
    #[must_use]
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ChildFnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "child function error: {}", self.message)
    }
}

impl std::error::Error for ChildFnError {}

impl From<OperationError> for ChildFnError {
    fn from(err: OperationError) -> Self {
        Self {
            message: err.to_string(),
        }
    }
}

// Static assertions: all public error types must be Send + Sync + 'static.
// These compile-time checks prevent accidental regressions.
const _: () = {
    #[allow(dead_code)] // reason: compile-time assertion, never called at runtime
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    #[allow(dead_code)] // reason: compile-time assertion, existence proves bounds hold
    fn assert_bounds() {
        assert_send_sync_static::<OperationError>();
        assert_send_sync_static::<OperationErrorKind>();
        assert_send_sync_static::<StepError>();
        assert_send_sync_static::<StepErrorKind>();
        assert_send_sync_static::<InvokeError>();
        assert_send_sync_static::<InvokeErrorKind>();
        assert_send_sync_static::<CallbackError>();
        assert_send_sync_static::<CallbackErrorKind>();
        assert_send_sync_static::<WaitForConditionError>();
        assert_send_sync_static::<WaitForConditionErrorKind>();
        assert_send_sync_static::<ChildContextError>();
        assert_send_sync_static::<ChildContextErrorKind>();
        assert_send_sync_static::<CombinatorError>();
        assert_send_sync_static::<CombinatorErrorKind>();
        assert_send_sync_static::<NonDeterministicExecutionError>();
        assert_send_sync_static::<NonDeterministicExecutionErrorKind>();
        assert_send_sync_static::<ChildFnError>();
        assert_send_sync_static::<crate::FileSystemSerdesError>();
    }
};

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions with descriptive messages
mod tests {
    use super::*;

    #[test]
    fn operation_error_display_includes_kind() {
        let err = OperationError::from_kind(OperationErrorKind::Step(StepError::from_kind(
            StepErrorKind::ExecutionFailed {
                message: "boom".to_owned(),
            },
        )));
        let display = err.to_string();
        assert!(display.contains("step"), "display: {display}");
        assert!(display.contains("boom"), "display: {display}");
    }

    #[test]
    fn operation_error_source_chain() {
        use std::error::Error;
        let err = OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::from_kind(
            InvokeErrorKind::FunctionFailed {
                message: "not found".to_owned(),
            },
        )));
        let source = err.source().expect("should have source");
        // source is the OperationErrorKind
        assert!(source.to_string().contains("invoke"));
    }

    /// Walks the full `source()` chain and returns each layer's display.
    fn causal_chain(err: &dyn std::error::Error) -> Vec<String> {
        let mut chain = vec![err.to_string()];
        let mut current: &dyn std::error::Error = err;
        while let Some(next) = current.source() {
            chain.push(next.to_string());
            current = next;
        }
        chain
    }

    #[test]
    fn source_traverses_to_terminal_step_kind() {
        let err = OperationError::from_kind(OperationErrorKind::Step(StepError::from_kind(
            StepErrorKind::RetriesExhausted {
                attempts: 3,
                last_error: "flaky".to_owned(),
            },
        )));
        // OperationError -> OperationErrorKind -> StepError -> StepErrorKind.
        let chain = causal_chain(&err);
        assert_eq!(chain.len(), 4, "chain: {chain:?}");
        let terminal = chain.last().expect("chain is non-empty");
        assert!(
            terminal.contains("retries exhausted after 3 attempts"),
            "terminal cause must be the concrete kind: {chain:?}"
        );
    }

    #[test]
    fn source_traverses_every_operation_error_variant() {
        let cases: Vec<OperationError> = vec![
            OperationError::from_kind(OperationErrorKind::Step(StepError::from_kind(
                StepErrorKind::ExecutionFailed {
                    message: "s".to_owned(),
                },
            ))),
            OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::from_kind(
                InvokeErrorKind::FunctionNotFound {
                    function_id: "f".to_owned(),
                },
            ))),
            OperationError::from_kind(OperationErrorKind::Callback(CallbackError::from_kind(
                CallbackErrorKind::TimedOut,
            ))),
            OperationError::from_kind(OperationErrorKind::WaitForCondition(
                WaitForConditionError::from_kind(WaitForConditionErrorKind::MaxChecksExceeded {
                    checks: 2,
                }),
            )),
            OperationError::from_kind(OperationErrorKind::ChildContext(
                ChildContextError::from_kind(ChildContextErrorKind::ChildFailed {
                    message: "c".to_owned(),
                }),
            )),
            OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::from_kind(
                CombinatorErrorKind::Internal {
                    message: "x".to_owned(),
                },
            ))),
            OperationError::from_kind(OperationErrorKind::NonDeterministicExecution(
                NonDeterministicExecutionError::from_kind(
                    NonDeterministicExecutionErrorKind::OperationMismatch {
                        wire_id: "abc123".to_owned(),
                        expected: "Step/Step".to_owned(),
                        actual: "Wait/Wait".to_owned(),
                    },
                ),
            )),
        ];
        for err in &cases {
            let chain = causal_chain(err);
            assert_eq!(
                chain.len(),
                4,
                "every variant must expose the 4-layer causal chain: {chain:?}"
            );
        }
    }

    #[test]
    fn step_error_kind_accessor() {
        let err = StepError::from_kind(StepErrorKind::RetriesExhausted {
            attempts: 3,
            last_error: "fail".to_owned(),
        });
        assert!(matches!(
            err.kind(),
            StepErrorKind::RetriesExhausted { attempts: 3, .. }
        ));
    }

    #[test]
    fn callback_error_graded_split() {
        // External errors
        let timed_out = CallbackError::from_kind(CallbackErrorKind::TimedOut);
        assert!(matches!(timed_out.kind(), CallbackErrorKind::TimedOut));

        let heartbeat = CallbackError::from_kind(CallbackErrorKind::HeartbeatTimedOut);
        assert!(matches!(
            heartbeat.kind(),
            CallbackErrorKind::HeartbeatTimedOut
        ));

        let external = CallbackError::from_kind(CallbackErrorKind::ExternalFailure {
            error_type: "UserError".to_owned(),
            message: "denied".to_owned(),
        });
        assert!(matches!(
            external.kind(),
            CallbackErrorKind::ExternalFailure { .. }
        ));

        // Internal errors
        let internal = CallbackError::from_kind(CallbackErrorKind::Internal {
            message: "oops".to_owned(),
        });
        assert!(matches!(
            internal.kind(),
            CallbackErrorKind::Internal { .. }
        ));
    }

    #[test]
    fn child_fn_error_from_operation_error() {
        let op = OperationError::__test_error();
        let child_err = ChildFnError::from(op);
        assert!(child_err.to_string().contains("child function error"));
    }

    #[test]
    fn combinator_error_kinds() {
        let join_failed = CombinatorError::from_kind(CombinatorErrorKind::JoinFailed {
            failed_index: 2,
            message: "step failed".to_owned(),
        });
        assert!(matches!(
            join_failed.kind(),
            CombinatorErrorKind::JoinFailed {
                failed_index: 2,
                ..
            }
        ));

        let all_failed = CombinatorError::from_kind(CombinatorErrorKind::AllFailed {
            errors: vec!["a".to_owned(), "b".to_owned()],
        });
        assert!(all_failed.to_string().contains("2 errors"));
    }

    #[test]
    fn wait_for_condition_error_display() {
        let err = WaitForConditionError::from_kind(WaitForConditionErrorKind::MaxChecksExceeded {
            checks: 10,
        });
        assert!(err.to_string().contains("10"));
    }

    #[test]
    fn invoke_error_function_not_found() {
        let err = InvokeError::from_kind(InvokeErrorKind::FunctionNotFound {
            function_id: "my-fn".to_owned(),
        });
        assert!(err.to_string().contains("my-fn"));
    }

    #[test]
    fn all_error_types_implement_std_error() {
        use std::error::Error;

        // Verify Error trait is implemented (these would not compile otherwise)
        fn check_error<E: Error + Send + Sync + 'static>(_e: &E) {}

        let op_err = OperationError::__test_error();
        check_error(&op_err);

        let step_err = StepError::from_kind(StepErrorKind::ExecutionFailed {
            message: "x".to_owned(),
        });
        check_error(&step_err);

        let invoke_err = InvokeError::from_kind(InvokeErrorKind::FunctionFailed {
            message: "x".to_owned(),
        });
        check_error(&invoke_err);

        let cb_err = CallbackError::from_kind(CallbackErrorKind::TimedOut);
        check_error(&cb_err);

        let wfc_err = WaitForConditionError::from_kind(WaitForConditionErrorKind::CheckFailed {
            message: "x".to_owned(),
        });
        check_error(&wfc_err);

        let child_err = ChildContextError::from_kind(ChildContextErrorKind::ChildFailed {
            message: "x".to_owned(),
        });
        check_error(&child_err);

        let comb_err = CombinatorError::from_kind(CombinatorErrorKind::Internal {
            message: "x".to_owned(),
        });
        check_error(&comb_err);

        let child_fn_err = ChildFnError::new("x");
        check_error(&child_fn_err);
    }
}
