//! The kind and status of a durable operation, as a test or a plugin sees them.

use std::fmt;

/// The kind of a durable operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OperationKind {
    /// A step, created by `DurableContext::step`.
    Step,
    /// A timer, created by `DurableContext::wait`.
    Wait,
    /// A child context, such as a `map` item, a `parallel` branch or a `with_retry` block.
    Context,
    /// A callback that an external system completes.
    Callback,
    /// A call to another durable function, created by `DurableContext::invoke`.
    ChainedInvoke,
    /// The execution itself.
    Execution,
}

impl OperationKind {
    /// Returns the token the service history and JS plugins use for this kind,
    /// such as `STEP` or `CHAINED_INVOKE`.
    #[must_use]
    pub fn wire_str(self) -> &'static str {
        match self {
            Self::Step => "STEP",
            Self::Wait => "WAIT",
            Self::Context => "CONTEXT",
            Self::Callback => "CALLBACK",
            Self::ChainedInvoke => "CHAINED_INVOKE",
            Self::Execution => "EXECUTION",
        }
    }

    /// Converts a service model operation type, or returns `None` for a type
    /// this enum does not list.
    pub(crate) fn from_wire(t: &aws_sdk_lambda::types::OperationType) -> Option<Self> {
        use aws_sdk_lambda::types::OperationType;
        match t {
            OperationType::Step => Some(Self::Step),
            OperationType::Wait => Some(Self::Wait),
            OperationType::Context => Some(Self::Context),
            OperationType::Callback => Some(Self::Callback),
            OperationType::ChainedInvoke => Some(Self::ChainedInvoke),
            OperationType::Execution => Some(Self::Execution),
            _ => None,
        }
    }
}

impl fmt::Display for OperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Step => "Step",
            Self::Wait => "Wait",
            Self::Context => "Context",
            Self::Callback => "Callback",
            Self::ChainedInvoke => "ChainedInvoke",
            Self::Execution => "Execution",
        })
    }
}

/// The status of a durable operation.
///
/// This is not `lambda_types::OperationStatus`, the service model type of the
/// same name that the `ExecutionClient` transport trait uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum OperationStatus {
    /// The operation started and has no outcome yet.
    Started,
    /// The operation waits for a retry delay to elapse.
    Pending,
    /// The retry delay elapsed and the operation can run its next attempt.
    Ready,
    /// The operation completed successfully.
    Succeeded,
    /// The operation completed with a failure.
    Failed,
    /// The operation was cancelled.
    Cancelled,
    /// The operation reached its timeout.
    TimedOut,
    /// The operation was stopped.
    Stopped,
}

impl OperationStatus {
    /// Returns the token the service history and JS plugins use for this
    /// status, such as `SUCCEEDED` or `TIMED_OUT`.
    #[must_use]
    pub fn wire_str(self) -> &'static str {
        match self {
            Self::Started => "STARTED",
            Self::Pending => "PENDING",
            Self::Ready => "READY",
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
            Self::Cancelled => "CANCELLED",
            Self::TimedOut => "TIMED_OUT",
            Self::Stopped => "STOPPED",
        }
    }
}

impl fmt::Display for OperationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Started => "Started",
            Self::Pending => "Pending",
            Self::Ready => "Ready",
            Self::Succeeded => "Succeeded",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
            Self::TimedOut => "TimedOut",
            Self::Stopped => "Stopped",
        })
    }
}

/// How one invocation of a durable function ended, as a plugin sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InvocationStatus {
    /// The execution completed successfully in this invocation.
    Succeeded,
    /// The execution failed in this invocation.
    Failed,
    /// The invocation suspended, and the service invokes the function
    /// again when an operation it waits for is due.
    Pending,
    /// The invocation failed, and the service invokes the function again.
    Retrying,
}

impl InvocationStatus {
    /// Returns the token JS plugins use for this status, such as
    /// `SUCCEEDED` or `RETRYING`.
    #[must_use]
    pub fn wire_str(self) -> &'static str {
        match self {
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
            Self::Pending => "PENDING",
            Self::Retrying => "RETRYING",
        }
    }
}

impl fmt::Display for InvocationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Succeeded => "Succeeded",
            Self::Failed => "Failed",
            Self::Pending => "Pending",
            Self::Retrying => "Retrying",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{InvocationStatus, OperationKind, OperationStatus};

    /// Each invocation status displays its variant name and returns the
    /// token JS plugins use.
    #[test]
    fn invocation_status_displays_variant_names() {
        let all = [
            InvocationStatus::Succeeded,
            InvocationStatus::Failed,
            InvocationStatus::Pending,
            InvocationStatus::Retrying,
        ];
        let shown: Vec<String> = all.iter().map(ToString::to_string).collect();
        assert_eq!(shown, ["Succeeded", "Failed", "Pending", "Retrying"]);
        let wire: Vec<&str> = all.iter().map(|s| s.wire_str()).collect();
        assert_eq!(wire, ["SUCCEEDED", "FAILED", "PENDING", "RETRYING"]);
    }

    const KINDS: [OperationKind; 6] = [
        OperationKind::Step,
        OperationKind::Wait,
        OperationKind::Context,
        OperationKind::Callback,
        OperationKind::ChainedInvoke,
        OperationKind::Execution,
    ];

    const STATUSES: [OperationStatus; 8] = [
        OperationStatus::Started,
        OperationStatus::Pending,
        OperationStatus::Ready,
        OperationStatus::Succeeded,
        OperationStatus::Failed,
        OperationStatus::Cancelled,
        OperationStatus::TimedOut,
        OperationStatus::Stopped,
    ];

    #[test]
    fn operation_enums_display_as_variant_names() {
        let kinds: Vec<String> = KINDS.iter().map(ToString::to_string).collect();
        assert_eq!(
            kinds,
            [
                "Step",
                "Wait",
                "Context",
                "Callback",
                "ChainedInvoke",
                "Execution"
            ]
        );
        let statuses: Vec<String> = STATUSES.iter().map(ToString::to_string).collect();
        assert_eq!(
            statuses,
            [
                "Started",
                "Pending",
                "Ready",
                "Succeeded",
                "Failed",
                "Cancelled",
                "TimedOut",
                "Stopped"
            ]
        );
    }

    #[test]
    fn operation_enums_wire_spelling() {
        let kinds: Vec<&str> = KINDS.iter().map(|k| k.wire_str()).collect();
        assert_eq!(
            kinds,
            [
                "STEP",
                "WAIT",
                "CONTEXT",
                "CALLBACK",
                "CHAINED_INVOKE",
                "EXECUTION"
            ]
        );
        let statuses: Vec<&str> = STATUSES.iter().map(|s| s.wire_str()).collect();
        assert_eq!(
            statuses,
            [
                "STARTED",
                "PENDING",
                "READY",
                "SUCCEEDED",
                "FAILED",
                "CANCELLED",
                "TIMED_OUT",
                "STOPPED"
            ]
        );
    }
}
