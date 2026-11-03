//! AWS Durable Execution SDK for Rust.
//!
//! This crate provides a Rust implementation of the AWS Lambda Durable
//! Functions SDK, enabling long-running orchestrations that survive Lambda
//! invocation timeouts through automatic checkpointing and deterministic
//! replay.
//!
//! # Overview
//!
//! A durable function is a Lambda function whose progress is automatically
//! checkpointed. If the function is interrupted, it restarts and replays
//! recorded results instead of re-executing operations. The SDK guarantees
//! deterministic replay as long as operations are created in a consistent
//! order across invocations.
//!
//! # Quick start
//!
//! ```no_run
//! use aws_durable_execution_sdk as durable;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Deserialize, Serialize)]
//! struct Order { id: String }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), lambda_runtime::Error> {
//!     durable::run(|event: Order, ctx: durable::DurableContext| async move {
//!         let result = ctx.step(|_step_ctx| async move {
//!             Ok(format!("processed {}", event.id))
//!         }).name("process")
//!           .await?;
//!         Ok(result)
//!     }).await
//! }
//! ```
//!
//! # Determinism contract
//!
//! Each durable operation has an **operation ID**: the identity that ties a
//! call site to its recorded result. The SDK mints IDs synchronously at the
//! call site, in the order the handler creates operations, and replay hands
//! each recorded result to the operation that claims the same ID. Replay is
//! therefore correct only if the handler creates the same operations in the
//! same order on every invocation:
//!
//! 1. Keep the control flow that decides which operations to create a pure
//!    function of the event and of results the SDK already recorded.
//! 2. Never create durable operations while iterating a `HashMap` or
//!    `HashSet`; their iteration order changes between runs. Sort into a
//!    `Vec` first, or use a collection with deterministic iteration order
//!    such as `BTreeMap`, `BTreeSet`, or the `indexmap` crate's `IndexMap`.
//! 3. Do not use `tokio::select!` over durable futures; use
//!    [`DurableContext::race`] or [`DurableContext::select_ok`], which
//!    record which branch won.
//! 4. Do not use `tokio::spawn` to run a durable operation in the
//!    background; use `.spawn()` on the operation's builder, which keeps
//!    the spawned work inside the execution's replay accounting.
//! 5. On suspension, the user future is dropped, so do not rely on `Drop`
//!    ordering for correctness between durable operations. For example, a
//!    guard that unlocks a resource in `Drop` runs the moment the execution
//!    suspends on a `ctx.wait(...)`, not after the wait completes; release
//!    resources in a step instead of a destructor.
//!
//! # Observability
//!
//! The SDK instruments the operation lifecycle through the [`tracing`]
//! facade: spans around the handler and each live step body, and `DEBUG`
//! events at operation start/finish/replay/retry and execution
//! start/resume/suspend. The span names, event names, and field names are a
//! documented, stable contract, including how to bridge it to
//! OpenTelemetry, in the [`observability`] module.

// Test code asserts with `unwrap`/`expect` pervasively; hoisting the
// override here keeps the panic-policy lints (deny in workspace lints)
// focused on production code without repeating the allow at every test
// module.
#![cfg_attr(test, allow(clippy::unwrap_used))]
#![cfg_attr(test, allow(clippy::expect_used))]

// Compiles the README's Rust examples as doctests, so a README example that
// stops compiling fails `cargo test` instead of rotting silently.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}

pub mod builders;
pub(crate) mod callback;
pub(crate) mod checkpoint_coalescer;
pub(crate) mod child;
pub(crate) mod client;
pub(crate) mod combinator;
mod context;
pub(crate) mod coordinator;
pub(crate) mod driver;
mod engine;
mod error;
mod future;
pub(crate) mod invoke;
pub(crate) mod map_parallel; // public types re-exported from crate::builders::map_parallel
pub mod observability;
mod operation;
mod options;
pub mod serdes;
pub(crate) mod step;
mod task;
#[cfg(feature = "test-util")]
pub mod test_util;
pub(crate) mod tracing_layer;
pub(crate) mod wait;
pub(crate) mod wait_for_condition;
pub(crate) mod with_retry;

#[cfg(feature = "replay-filter")]
pub use self::tracing_layer::ReplayFilterLayer;

// When users run `cargo test` without `--features replay-filter`, the type is
// still compiled (via `#[cfg(any(test, ...))]`) inside the `pub(crate)` module.
// Without this re-export the `unreachable_pub` lint fires. The guard ensures
// only one `pub use` is active at a time.
#[cfg(all(test, not(feature = "replay-filter")))]
pub use self::tracing_layer::ReplayFilterLayer;

pub use self::client::{
    CheckpointOutput, ExecutionClient, ExecutionClientError, ExecutionClientErrorScope,
    GetStateOutput, LambdaExecutionClient,
};
pub use self::context::{DurableContext, StepContext};
pub use self::error::{
    CallbackError, CallbackErrorKind, ChildContextError, ChildContextErrorKind, CombinatorError,
    CombinatorErrorKind, InvokeError, InvokeErrorKind, JoinFailed, MaxChecksExceeded,
    NonDeterministicExecutionError, NonDeterministicExecutionErrorKind, OperationError,
    OperationErrorKind, OperationMismatch, ReplayedFailure, StepError, StepErrorKind, TypedError,
    UnexpectedStatus, UnrecognizedStatus, WaitError, WaitErrorKind, WaitForConditionError,
    WaitForConditionErrorKind, WireError,
};
pub use self::future::{Branch, BranchHandle, BranchSet, DurableFuture};
pub use self::operation::{OperationKind, OperationStatus};
pub use self::options::{Options, OptionsBuilder, OptionsValidationError};
pub use self::serdes::{JsonSerdes, PassthroughSerdes, Serdes};
pub use self::step::StepSemantics;

// Re-export rule: every foreign type in the public surface is re-exported.
pub use aws_config::SdkConfig;
pub use aws_sdk_lambda::Client as LambdaClient;
pub use lambda_runtime::{self, Context as LambdaContext};

/// The service model types of the checkpoint transport.
///
/// [`ExecutionClient`] speaks the service model types of `aws-sdk-lambda`
/// on purpose: the SDK already depends on that crate, and a transport for
/// another backend builds the same records the service returns. This
/// module re-exports exactly the types that [`Operation`] and
/// [`OperationUpdate`] name in their fields, directly or nested, and
/// [`DateTime`], the type of their timestamp fields.
///
/// With the `test-util` feature, the module also re-exports `Event`, the
/// service's history event that `test_util::TestResult::history_events`
/// returns, and every type that `Event` names in its fields, directly or
/// nested.
///
/// These types are those of the `aws-sdk-lambda` release this crate
/// depends on. They change when the crate moves to a new `aws-sdk-lambda`
/// release.
///
/// [`Operation`]: lambda_types::Operation
/// [`OperationUpdate`]: lambda_types::OperationUpdate
/// [`DateTime`]: lambda_types::DateTime
pub mod lambda_types {
    pub use aws_sdk_lambda::primitives::DateTime;
    pub use aws_sdk_lambda::types::{
        CallbackDetails, CallbackOptions, ChainedInvokeDetails, ChainedInvokeOptions,
        ContextDetails, ContextOptions, ErrorObject, ExecutionDetails, Operation, OperationAction,
        OperationStatus, OperationType, OperationUpdate, StepDetails, StepOptions, WaitDetails,
        WaitOptions,
    };
    #[cfg(feature = "test-util")]
    pub use aws_sdk_lambda::types::{
        CallbackFailedDetails, CallbackStartedDetails, CallbackSucceededDetails,
        CallbackTimedOutDetails, ChainedInvokeFailedDetails, ChainedInvokeStartedDetails,
        ChainedInvokeStoppedDetails, ChainedInvokeSucceededDetails, ChainedInvokeTimedOutDetails,
        ContextFailedDetails, ContextStartedDetails, ContextSucceededDetails, Event, EventError,
        EventInput, EventResult, EventType, ExecutionFailedDetails, ExecutionStartedDetails,
        ExecutionStoppedDetails, ExecutionSucceededDetails, ExecutionTimedOutDetails,
        InvocationCompletedDetails, RetryDetails, StepFailedDetails, StepStartedDetails,
        StepSucceededDetails, WaitCancelledDetails, WaitStartedDetails, WaitSucceededDetails,
    };
}

use serde::{Deserialize, Serialize};
use std::future::Future;
use std::time::Duration;
use tracing::Instrument as _;

/// Boxed error type: `Box<dyn std::error::Error + Send + Sync>`.
///
/// This is the error type handler and step closures return.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
///
/// async fn handler(
///     _event: serde_json::Value,
///     ctx: durable::DurableContext,
/// ) -> Result<String, durable::BoxError> {
///     let value = ctx.step(|_| async { Ok(42) }).await?;
///     Ok(format!("got {value}"))
/// }
/// ```
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Retry decision returned by a retry-strategy closure.
///
/// Tells the engine whether to retry a failed step and, if so, how long
/// to wait before the next attempt. Retry strategies are installed with
/// [`StepBuilder::retry_strategy`](crate::builders::StepBuilder::retry_strategy) and
/// [`WaitForCallbackBuilder::submitter_retry`](crate::builders::WaitForCallbackBuilder::submitter_retry).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk::RetryDecision;
/// use std::time::Duration;
///
/// let retry = RetryDecision::Retry {
///     delay: Duration::from_secs(1),
/// };
/// let stop = RetryDecision::Stop;
/// # drop(retry);
/// # drop(stop);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RetryDecision {
    /// Retry after the specified delay.
    Retry {
        /// Duration to wait before retrying.
        delay: Duration,
    },
    /// Do not retry; propagate the error.
    Stop,
}

/// A boxed retry strategy that decides whether to retry a failed step.
///
/// Receives the step error and the attempt number (starting from 1), and
/// returns a [`RetryDecision`].
///
/// Crate-internal: the boxing is an implementation detail. Public setters
/// ([`StepBuilder::retry_strategy`], [`WaitForCallbackBuilder::submitter_retry`])
/// take a generic closure and box it here.
pub(crate) type RetryStrategy = Box<dyn Fn(&StepError, u32) -> RetryDecision + Send + Sync>;

/// Starts the durable function runtime with the given handler.
///
/// This is the primary entry point. It configures the Lambda runtime with
/// durable execution support using default [`Options`], then runs the
/// handler for each invocation. Equivalent to calling [`run_with_options`]
/// with [`Options::default`].
///
/// To customize the runtime, keep this entry point's shape but swap the
/// function: for custom [`Options`] — a preconfigured Lambda client, your
/// own [`SdkConfig`], or checkpoint tuning — use [`run_with_options`]; to
/// add middleware around the service function itself, use [`wrap`], which
/// returns the service function without starting the runtime. Custom
/// serialization is configured per operation, not through [`Options`]:
/// pass a [`Serdes`] to an operation builder's `.serdes(...)`, sharing one
/// implementation across operations behind an `Arc<S>` (which itself
/// implements the trait) when they should serialize alike.
///
/// The handler closure is called once per invocation. It receives the
/// deserialized event and a [`DurableContext`] for performing durable
/// operations. Per invocation, the runtime parses the durable envelope into
/// a checkpoint log, constructs a [`DurableContext`] seeded with that log,
/// and drives the handler closure so that completed operations replay from
/// the log instead of re-executing.
///
/// # How handler failures are reported
///
/// When the handler returns `Err`, the runtime reports the failure *inside a
/// successful Lambda invocation response*, as a `FAILED` status envelope
/// that the durable execution service reads. The invocation itself does not
/// error. This is required by the durable service protocol, and it inverts
/// the usual Lambda observability signals:
///
/// - the `CloudWatch` `Errors` metric for the function does not fire,
/// - dead-letter queues and `OnFailure` destinations do not trigger,
/// - X-Ray does not mark the trace as an error.
///
/// Handler failures surface through the durable execution status instead:
/// poll `GetDurableExecution` for a `FAILED` status, or alarm on the
/// durable-execution metrics (for example, executions that reach a failed
/// terminal state) rather than on Lambda invocation errors.
///
/// # Errors
///
/// Returns an error if the Lambda runtime fails to start or encounters an
/// unrecoverable error during execution.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct MyEvent { name: String }
///
/// #[tokio::main]
/// async fn main() -> Result<(), lambda_runtime::Error> {
///     durable::run(|event: MyEvent, ctx: durable::DurableContext| async move {
///         Ok(format!("Hello, {}!", event.name))
///     }).await
/// }
/// ```
pub async fn run<F, E, Fut, O>(handler: F) -> Result<(), lambda_runtime::Error>
where
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    E: for<'de> Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    run_with_options(handler, Options::default()).await
}

/// Starts the durable function runtime with the given handler and options.
///
/// Like [`run`], but applies the supplied [`Options`], for example a
/// preconfigured Lambda client or your own [`SdkConfig`], to every
/// invocation. Equivalent to registering [`wrap`] with the Lambda
/// runtime yourself:
/// `lambda_runtime::run(lambda_runtime::service_fn(wrap(handler, options)))`.
/// Serialization is not an [`Options`] concern: set a [`Serdes`] per
/// operation via the builders' `.serdes(...)`, sharing an `Arc<S>` across
/// operations that should serialize alike.
///
/// The handler closure is called once per invocation. It receives the
/// deserialized event and a [`DurableContext`] for performing durable
/// operations.
///
/// Handler failures are reported the same way as [`run`]: inside a
/// successful Lambda invocation response, surfacing through the durable
/// execution status (`GetDurableExecution`) rather than as Lambda
/// invocation errors. See [`run`] for the observability implications.
///
/// # Errors
///
/// Returns an error if the Lambda runtime fails to start or encounters an
/// unrecoverable error during execution.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct MyEvent { name: String }
///
/// #[tokio::main]
/// async fn main() -> Result<(), lambda_runtime::Error> {
///     let options = durable::Options::default();
///     durable::run_with_options(
///         |event: MyEvent, ctx: durable::DurableContext| async move {
///             Ok(format!("Hello, {}!", event.name))
///         },
///         options,
///     ).await
/// }
/// ```
pub async fn run_with_options<F, E, Fut, O>(
    handler: F,
    options: Options,
) -> Result<(), lambda_runtime::Error>
where
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    E: for<'de> Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    lambda_runtime::run(lambda_runtime::service_fn(wrap(handler, options))).await
}

/// Parsed and validated durable invocation envelope fields.
#[derive(Debug)]
struct InvocationEnvelope {
    execution_arn: String,
    checkpoint_token: String,
}

/// Returns `true` when the payload looks like a durable invocation envelope
/// (contains at least one of the expected top-level keys). Used to distinguish
/// "the service sent an envelope but something is wrong" (an error naming the
/// bad field) from "this payload has no envelope shape at all" (rejected at
/// the entry points with a message describing the expected envelope).
fn has_envelope_shape(payload: &serde_json::Value) -> bool {
    payload.get("DurableExecutionArn").is_some()
        || payload.get("CheckpointToken").is_some()
        || payload.get("InitialExecutionState").is_some()
}

/// Parses and validates the durable invocation envelope.
///
/// When the envelope shape is present (any of the expected top-level keys
/// exist), this function requires `DurableExecutionArn` and
/// `CheckpointToken` to be present and to be strings. A missing or
/// mistyped field is an immediate error naming the field, rather than
/// silently defaulting to an empty string.
///
/// When the envelope shape is absent (none of the expected keys), returns
/// `None`: callers decide whether that's acceptable.
fn parse_envelope(
    payload: &serde_json::Value,
) -> Result<Option<InvocationEnvelope>, lambda_runtime::Error> {
    if !has_envelope_shape(payload) {
        return Ok(None);
    }

    let execution_arn = match payload.get("DurableExecutionArn") {
        Some(v) => v
            .as_str()
            .ok_or_else(|| {
                lambda_runtime::Error::from(
                    "malformed invocation envelope: \"DurableExecutionArn\" is present but is not \
                     a string"
                        .to_owned(),
                )
            })?
            .to_owned(),
        None => {
            return Err(lambda_runtime::Error::from(
                "malformed invocation envelope: required field \"DurableExecutionArn\" is missing"
                    .to_owned(),
            ));
        }
    };

    let checkpoint_token = match payload.get("CheckpointToken") {
        Some(v) => v
            .as_str()
            .ok_or_else(|| {
                lambda_runtime::Error::from(
                    "malformed invocation envelope: \"CheckpointToken\" is present but is not a \
                     string"
                        .to_owned(),
                )
            })?
            .to_owned(),
        None => {
            return Err(lambda_runtime::Error::from(
                "malformed invocation envelope: required field \"CheckpointToken\" is missing"
                    .to_owned(),
            ));
        }
    };

    Ok(Some(InvocationEnvelope {
        execution_arn,
        checkpoint_token,
    }))
}

/// Reads the input payload of the inline EXECUTION operation.
///
/// The service embeds the customer payload as a JSON string in
/// `ExecutionDetails.InputPayload` of the EXECUTION operation in
/// `InitialExecutionState.Operations`. This function reads the first
/// operation of that list whose `Type` is `EXECUTION`.
///
/// - No such operation returns `PageInput::NoExecutionOperation`. The
///   operation may be on a later page, which
///   [`client::resolve_bootstrap_log`] fetches.
/// - An absent or JSON `null` `InputPayload` returns
///   `PageInput::Execution(None)`.
/// - A string `InputPayload` returns `PageInput::Execution(Some(value))`.
/// - Any other `InputPayload` fails the invocation. So a malformed
///   envelope fails before pagination or the handler runs.
fn inline_input_payload(
    payload: &serde_json::Value,
) -> Result<client::PageInput, lambda_runtime::Error> {
    let execution_op = payload
        .get("InitialExecutionState")
        .and_then(|s| s.get("Operations"))
        .and_then(serde_json::Value::as_array)
        .and_then(|ops| {
            ops.iter().find(|op| {
                op.get("Type")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|t| t.eq_ignore_ascii_case("EXECUTION"))
            })
        });
    let Some(op) = execution_op else {
        return Ok(client::PageInput::NoExecutionOperation);
    };
    match op
        .get("ExecutionDetails")
        .and_then(|d| d.get("InputPayload"))
    {
        None | Some(serde_json::Value::Null) => Ok(client::PageInput::Execution(None)),
        Some(serde_json::Value::String(value)) => {
            Ok(client::PageInput::Execution(Some(value.clone())))
        }
        Some(_) => Err(lambda_runtime::Error::from(
            "malformed invocation envelope: InputPayload is not a string".to_owned(),
        )),
    }
}

/// Parses the customer's original event from the execution input payload.
///
/// `input_payload` is the `InputPayload` of the first EXECUTION operation
/// of the whole execution state, which [`client::resolve_bootstrap_log`]
/// returns after it follows the pagination marker. An absent payload parses
/// as `{}`, as the JS SDK does. A payload that does not parse fails the
/// invocation.
///
/// The envelope is always required: [`run`] and [`wrap`] reject an
/// envelope-free payload in `parse_envelope`, before this function runs.
/// Local testing goes through [`LocalRunner`](test_util::LocalRunner),
/// which invokes the [`wrap`]-produced service with synthesized envelopes,
/// so this function runs on the local path exactly as it does in
/// production.
fn extract_customer_input<E>(input_payload: Option<&str>) -> Result<E, lambda_runtime::Error>
where
    E: for<'de> Deserialize<'de>,
{
    serde_json::from_str(input_payload.unwrap_or("{}"))
        .map_err(|e| lambda_runtime::Error::from(format!("deserialize customer input: {e}")))
}

/// Derives the wire failure record from a handler-level `BoxError`.
///
/// Attempts to downcast to `OperationError` for structured extraction.
/// Any other error records the type `Error`, as the JS SDK records a
/// plain `Error`, unless a `TypedError` in its chain names the type. The
/// message is the error's flattened chain, built by the module-wide
/// single flattening site (see [`error::wire_error_for`]).
fn wire_error_from_box_error(err: BoxError) -> WireError {
    match err.downcast::<OperationError>() {
        Ok(op_err) => wire_error_from_operation_error(&op_err),
        Err(other) => error::wire_error_for(&*other, "Error"),
    }
}

/// Serializes the handler's `Ok` value for the `SUCCEEDED` envelope.
///
/// A value that fails to serialize records the type `Error` and the
/// `serde_json::Error` message, the same rule as an untyped handler
/// error.
fn serialize_handler_result<T: Serialize>(result: &T) -> Result<String, WireError> {
    serde_json::to_string(result).map_err(|e| error::wire_error_for(&e, "Error"))
}

/// Derives the wire failure record from an `OperationError`.
///
/// The wire type comes from the error's recorded wire identity when it
/// has one (preserving the original type across boundaries), else from
/// the kind's registry name. For callback external failures, the wire
/// message is the externally reported message (not the full chain).
fn wire_error_from_operation_error(err: &OperationError) -> WireError {
    if let OperationErrorKind::Callback(cb_err) = err.kind()
        && matches!(cb_err.kind(), CallbackErrorKind::ExternalFailure)
        && let Some(wire) = err.wire()
    {
        // Report the external caller's own failure fields verbatim.
        return wire.clone();
    }
    error::wire_error_for(err, err.kind().wire_type_name())
}

/// Parses a wire `StackTrace` array (of strings) from an error object.
fn parse_stack_trace(error: &serde_json::Value) -> Option<Vec<String>> {
    let frames: Vec<String> = error
        .get("StackTrace")?
        .as_array()?
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(String::from)
        .collect();
    if frames.is_empty() {
        None
    } else {
        Some(frames)
    }
}

/// Parses inline operations from the durable invocation envelope into a
/// checkpoint log.
///
/// The service embeds the execution state in
/// `InitialExecutionState.Operations` as a JSON array of operation objects.
/// Each has `Id`, `Type`, `Status`, and type-specific details (e.g.,
/// `StepDetails`). On first invocation the array is empty or contains only
/// the execution-start operation; on re-invocation it contains all prior
/// checkpointed operations.
fn parse_inline_operations(payload: &serde_json::Value) -> (engine::CheckpointLog, Option<String>) {
    let initial_state = payload.get("InitialExecutionState");

    // Check for a pagination marker indicating more pages of operations.
    // Extracted independently of `Operations`: the service may omit the
    // Operations array on the first page (e.g. when a large customer
    // payload displaces it) while still supplying a marker, and the
    // remaining pages must still be fetched.
    let next_marker = initial_state
        .and_then(|s| s.get("NextMarker"))
        .and_then(serde_json::Value::as_str)
        .filter(|m| !m.is_empty())
        .map(String::from);

    // A missing or non-array `Operations` field is an empty first page.
    // Skip the first operation (Execution type: the invocation context)
    // and parse remaining step/wait/etc. operations into records.
    let records: Vec<(String, engine::CheckpointRecord)> = initial_state
        .and_then(|s| s.get("Operations"))
        .and_then(serde_json::Value::as_array)
        .map(|ops| ops.iter().filter_map(parse_single_operation).collect())
        .unwrap_or_default();

    (engine::CheckpointLog::from_records(records), next_marker)
}

/// Parses a timestamp of the invocation payload.
///
/// The payload carries a timestamp as an ISO-8601 string. An integer is
/// read as milliseconds since the Unix epoch, as the Python SDK writes it.
/// Any other JSON type, a string that does not parse, or a value outside
/// the range of `SystemTime` gives `None`.
pub(crate) fn parse_envelope_timestamp(value: &serde_json::Value) -> Option<std::time::SystemTime> {
    use aws_sdk_lambda::primitives::{DateTime, DateTimeFormat};
    let parsed = match value {
        serde_json::Value::String(s) => DateTime::from_str(s, DateTimeFormat::DateTime).ok()?,
        serde_json::Value::Number(n) => DateTime::from_millis(n.as_i64()?),
        _ => return None,
    };
    to_system_time(parsed)
}

/// Converts a service timestamp to a `SystemTime`. A value outside the
/// range of `SystemTime` on this platform gives `None`, so an out-of-range
/// timestamp reads as a missing one.
pub(crate) fn to_system_time<T: TryInto<std::time::SystemTime>>(
    timestamp: T,
) -> Option<std::time::SystemTime> {
    timestamp.try_into().ok()
}

/// Parses a single operation JSON object into a checkpoint record.
#[expect(clippy::too_many_lines)] // reason: sequential detail extraction reads better as one flow
fn parse_single_operation(op: &serde_json::Value) -> Option<(String, engine::CheckpointRecord)> {
    let id = op.get("Id").and_then(serde_json::Value::as_str)?;
    let op_type = op.get("Type").and_then(serde_json::Value::as_str)?;
    // Skip the Execution context operation (wire format: "EXECUTION").
    if op_type.eq_ignore_ascii_case("Execution") {
        return None;
    }
    let status_str = op
        .get("Status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("STARTED");
    // The backend sends status in UPPER_CASE (wire format). `TIMED_OUT` is
    // the only wire spelling for a timeout, per the smithy model and the
    // Python and JS SDKs; a `TIMEDOUT` value therefore falls to `Unknown`
    // like any other unrecognized status (issue #45), carrying the raw
    // value as received so replay can name it when it fails the execution.
    let status = match status_str.to_ascii_uppercase().as_str() {
        "STARTED" => engine::CheckpointStatus::Started,
        "SUCCEEDED" => engine::CheckpointStatus::Succeeded,
        "FAILED" => engine::CheckpointStatus::Failed,
        "PENDING" => engine::CheckpointStatus::Pending,
        "READY" => engine::CheckpointStatus::Ready,
        "CANCELLED" => engine::CheckpointStatus::Cancelled,
        "TIMED_OUT" => engine::CheckpointStatus::TimedOut,
        "STOPPED" => engine::CheckpointStatus::Stopped,
        _ => engine::CheckpointStatus::Unknown(status_str.to_owned()),
    };

    // Extract step details.
    let step_details = op.get("StepDetails");
    let result = step_details
        .and_then(|d| d.get("Result"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let error = step_details.and_then(|d| d.get("Error"));
    let error_type = error
        .and_then(|e| e.get("ErrorType"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let error_message = error
        .and_then(|e| e.get("ErrorMessage"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let error_data = error
        .and_then(|e| e.get("ErrorData"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let stack_trace = error.and_then(parse_stack_trace);
    #[expect(clippy::cast_possible_truncation)] // reason: attempt ≤ MAX_ATTEMPTS (small)
    #[expect(clippy::cast_sign_loss)] // reason: clamped to non-negative
    let attempt = step_details
        .and_then(|d| d.get("Attempt"))
        .and_then(serde_json::Value::as_i64)
        .map_or(0, |a| a.clamp(0, i64::from(u32::MAX)) as u32);

    // A wait's scheduled end, or a step's next attempt.
    let due_at = op
        .get("WaitDetails")
        .and_then(|d| d.get("ScheduledEndTimestamp"))
        .or_else(|| step_details.and_then(|d| d.get("NextAttemptTimestamp")))
        .and_then(parse_envelope_timestamp);

    // Parse ChainedInvokeDetails (for invoke operations).
    let invoke_details = op.get("ChainedInvokeDetails");
    let invoke_result = invoke_details
        .and_then(|d| d.get("Result"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let invoke_error = invoke_details.and_then(|d| d.get("Error"));
    let invoke_error_type = invoke_error
        .and_then(|e| e.get("ErrorType"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let invoke_error_message = invoke_error
        .and_then(|e| e.get("ErrorMessage"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let invoke_error_data = invoke_error
        .and_then(|e| e.get("ErrorData"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let invoke_stack_trace = invoke_error.and_then(parse_stack_trace);

    // Parse ContextDetails for child context operations.
    let context_details = op.get("ContextDetails");
    let replay_children = context_details
        .and_then(|d| d.get("ReplayChildren"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    // Parse CallbackDetails for callback operations.
    let callback_details = op.get("CallbackDetails");
    let callback_id = callback_details
        .and_then(|d| d.get("CallbackId"))
        .and_then(serde_json::Value::as_str)
        .map(String::from);

    // Also check for result in ContextDetails (child context success payload).
    let result = result.or_else(|| {
        context_details
            .and_then(|d| d.get("Result"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });

    // Also check for errors in ContextDetails (child context failure).
    let context_error = context_details.and_then(|d| d.get("Error"));
    let error_type = error_type.or_else(|| {
        context_error
            .and_then(|e| e.get("ErrorType"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });
    let error_message = error_message.or_else(|| {
        context_error
            .and_then(|e| e.get("ErrorMessage"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });
    let error_data = error_data.or_else(|| {
        context_error
            .and_then(|e| e.get("ErrorData"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });
    let stack_trace = stack_trace.or_else(|| context_error.and_then(parse_stack_trace));

    // Also check for result in CallbackDetails (callback success payload).
    let result = result.or_else(|| {
        callback_details
            .and_then(|d| d.get("Result"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });

    // Also check for errors in CallbackDetails (callback failure).
    let callback_error = callback_details.and_then(|d| d.get("Error"));
    let error_type = error_type.or_else(|| {
        callback_error
            .and_then(|e| e.get("ErrorType"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });
    let error_message = error_message.or_else(|| {
        callback_error
            .and_then(|e| e.get("ErrorMessage"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });
    let error_data = error_data.or_else(|| {
        callback_error
            .and_then(|e| e.get("ErrorData"))
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    });
    let stack_trace = stack_trace.or_else(|| callback_error.and_then(parse_stack_trace));

    Some((
        id.to_owned(),
        engine::CheckpointRecord {
            id: id.to_owned(),
            status,
            result,
            error_type,
            error_message,
            error_data,
            stack_trace,
            attempt,
            invoke_result,
            invoke_error_type,
            invoke_error_message,
            invoke_error_data,
            invoke_stack_trace,
            replay_children,
            callback_id,
            op_type: Some(op_type.to_owned()),
            sub_type: op
                .get("SubType")
                .and_then(serde_json::Value::as_str)
                .map(String::from),
            op_name: op
                .get("Name")
                .and_then(serde_json::Value::as_str)
                .map(String::from),
            due_at,
        },
    ))
}

/// The raw invocation payload `lambda_runtime` hands to the service
/// functions [`wrap`] produces.
///
/// Opaque by design: the payload is the durable invocation envelope the
/// service sends, and its JSON representation is an implementation detail
/// between the runtime and the SDK. Callers never construct or inspect
/// one: `lambda_runtime::service_fn` deserializes it, and the service
/// function consumes it. Keeping the inner value private is what keeps
/// `serde_json` out of the crate's public API.
#[derive(Debug, serde::Deserialize)]
pub struct InvocationPayload(pub(crate) serde_json::Value);

/// The durable response envelope one invocation of a [`wrap`]-produced
/// service function resolves with.
///
/// Opaque by design: the envelope carries the invocation's status
/// (`SUCCEEDED`, `FAILED`, or suspension) for the durable execution
/// service to read, and its JSON representation is an implementation
/// detail between the SDK and the service. Callers hand it back to
/// `lambda_runtime` unexamined.
#[derive(Debug, serde::Serialize)]
pub struct InvocationResponse(pub(crate) serde_json::Value);

/// Boxed, pinned future of one Lambda invocation of a wrapped durable
/// handler: the return type of the service functions [`wrap`] produces.
///
/// Crate-internal: rustdoc renders the alias transparently, and callers
/// only ever name the `impl Fn` the wrappers return.
type BoxedInvocationFuture = std::pin::Pin<
    Box<dyn Future<Output = Result<InvocationResponse, lambda_runtime::Error>> + Send>,
>;

/// Creates a Lambda service function with durable execution support.
///
/// Unlike [`run`], this does not start the runtime: it returns a service
/// function suitable for passing to `lambda_runtime::run`. Use this for
/// composable setups where you need additional middleware or custom
/// runtime configuration.
///
/// The service function reports handler failures inside a *successful*
/// Lambda invocation response, a `FAILED` status envelope the durable
/// execution service reads, never as a Lambda invocation error. Middleware
/// wrapped around this service therefore sees `Ok` for failed handlers, and
/// Lambda-level error signals (the `Errors` metric, DLQs and `OnFailure`
/// destinations, X-Ray error status) do not fire. Monitor the durable
/// execution status (`GetDurableExecution`) instead; see [`run`] for
/// details.
///
/// # Errors
///
/// Returns an error if configuration is invalid.
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk as durable;
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Deserialize)]
/// struct MyEvent { name: String }
///
/// #[tokio::main]
/// async fn main() -> Result<(), lambda_runtime::Error> {
///     let service = durable::wrap(
///         |event: MyEvent, ctx: durable::DurableContext| async move {
///             Ok(format!("Hello, {}!", event.name))
///         },
///         durable::Options::default(),
///     );
///     lambda_runtime::run(lambda_runtime::service_fn(service)).await
/// }
/// ```
pub fn wrap<F, E, Fut, O>(
    handler: F,
    options: Options,
) -> impl Fn(lambda_runtime::LambdaEvent<InvocationPayload>) -> BoxedInvocationFuture + Send + Sync
where
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    E: for<'de> Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    use std::sync::Arc as StdArc;

    // Consume Options once, at wrap time. The execution client is resolved a
    // single time here and reused across every invocation (cold-start best
    // practice).
    let Options {
        sdk_config,
        lambda_client,
        execution_client,
        checkpoint_delay,
        checkpoint_batching,
    } = options;
    // The checkpoint buffer window: a configured delay is the coalescing
    // window; batching without a delay buffers with a zero window (writes
    // batch behind the single-writer lock but are never held back); batching
    // off without a delay means immediate writes.
    let checkpoint_buffer_window = match (checkpoint_delay, checkpoint_batching) {
        (Some(delay), _) => Some(delay),
        (None, true) => Some(Duration::ZERO),
        (None, false) => None,
    };
    // A supplied transport replaces the Lambda transport; `build()` already
    // rejected it together with `lambda_client` or `sdk_config`.
    let preset_client: Option<StdArc<dyn ExecutionClient>> = execution_client.or_else(|| {
        base_lambda_client_from_options(sdk_config, lambda_client)
            .map(|c| StdArc::new(LambdaExecutionClient::new(c)) as StdArc<dyn ExecutionClient>)
    });
    wrap_with_provider(
        handler,
        ClientProvider::new(preset_client),
        checkpoint_buffer_window,
    )
}

/// Body of [`wrap`]: builds the per-invocation service function on top of
/// an already-resolved [`ClientProvider`]. The `test-util` `LocalRunner`
/// calls [`wrap`] with its
/// in-memory backend as the execution client, so it runs the same envelope
/// parsing, pagination, driver, and error-mapping code as production.
fn wrap_with_provider<F, E, Fut, O>(
    handler: F,
    provider: ClientProvider,
    checkpoint_buffer_window: Option<Duration>,
) -> impl Fn(lambda_runtime::LambdaEvent<InvocationPayload>) -> BoxedInvocationFuture + Send + Sync
where
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    E: for<'de> Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    use std::sync::Arc as StdArc;

    let handler = StdArc::new(handler);
    let provider = StdArc::new(provider);

    move |event: lambda_runtime::LambdaEvent<InvocationPayload>| -> std::pin::Pin<
        Box<dyn Future<Output = Result<InvocationResponse, lambda_runtime::Error>> + Send>,
    > {
        let handler = StdArc::clone(&handler);
        let provider = StdArc::clone(&provider);
        Box::pin(async move {
            let (InvocationPayload(raw_payload), lambda_ctx) = event.into_parts();

            // Parse and validate the durable invocation envelope.
            let envelope = parse_envelope(&raw_payload)?.ok_or_else(|| {
                lambda_runtime::Error::from(
                    "invocation payload is not a durable execution envelope \
                         (missing DurableExecutionArn, CheckpointToken, and \
                         InitialExecutionState)"
                        .to_owned(),
                )
            })?;
            let execution_arn = envelope.execution_arn;
            let checkpoint_token = envelope.checkpoint_token;

            // Read the inline input and the inline page. A non-string
            // `InputPayload` fails here, before pagination or the handler.
            let first_page_input = inline_input_payload(&raw_payload)?;
            let (checkpoint_log, initial_marker) = parse_inline_operations(&raw_payload);

            // Reuse the execution client resolved once at wrap time (built
            // from the ambient default at most once when no client was
            // supplied via Options).
            let exec_client = provider.get().await;

            // If the initial state was paginated, fetch remaining pages. A
            // retryable error fails the invocation, so the service
            // re-invokes. A non-retryable error (a KMS misconfiguration)
            // fails the same way on every invocation, so it fails the
            // execution before the handler runs, as in JS.
            let (checkpoint_log, input_payload) = match client::resolve_bootstrap_log(
                exec_client.as_ref(),
                &execution_arn,
                &checkpoint_token,
                checkpoint_log,
                first_page_input,
                initial_marker.as_deref(),
            )
            .await
            {
                Ok((log, input_payload)) => (StdArc::new(log), input_payload),
                Err(e) if e.is_retryable() => {
                    return Err(lambda_runtime::Error::from(format!(
                        "failed to paginate initial state: {e}"
                    )));
                }
                Err(e) => {
                    return Ok(failed_envelope(&WireError::new(
                        Some(client::bootstrap_error_type(&e)),
                        Some(client::client_error_record_message(&e)),
                    )));
                }
            };

            // The input is parsed after pagination, because it may be on a
            // later page. An absent payload parses as `{}`.
            let customer_input: E = extract_customer_input(input_payload.as_deref())?;

            let ctx = DurableContext::new_root_with_client_and_defaults(
                execution_arn,
                lambda_ctx,
                checkpoint_log,
                exec_client,
                checkpoint_token,
                checkpoint_buffer_window,
            );

            // Retained past the handler move so the post-outcome flush can
            // drain the checkpoint coalescing buffer (a cheap Arc clone).
            let flush_ctx = ctx.clone();
            let coordinator = StdArc::clone(ctx.coordinator());
            let remaining = ctx.remaining_time();
            let replay_span = ctx.replay_span();

            // Lifecycle event: exactly one of `execution_started` /
            // `execution_resumed` per invocation, emitted while the
            // handler's `durable_execution` span is entered so it is a
            // span event of that span (which is what lets the documented
            // tracing-opentelemetry bridge export it on the execution
            // span). See `crate::observability`.
            {
                let _execution_scope = replay_span.enter();
                tracing_layer::invocation_begin_event(
                    ctx.is_replaying(),
                    ctx.execution_arn(),
                    &ctx.lambda_context().request_id,
                );
            }

            // Run the handler through the driver which handles suspension.
            // The handler future is instrumented with the handler-level span
            // so user log events between operations carry the execution ARN
            // and the live `isReplay` flag.
            //
            // The handler runs as the root context's body, so an operation
            // called on a child context that escaped its closure fails the
            // execution with a context usage error.
            //
            // The root adapter owns everything it captures (the handler, the
            // event, the context and the span), so it is `'static` and the
            // driver can run it as the invocation's root task, while the
            // handler's own future needs no `'static` bound.
            let task_ownership = StdArc::clone(ctx.task_ownership());
            let root = context::run_body(context::ActiveBody::Context(String::new()), async move {
                match (handler)(customer_input, ctx).await {
                    Ok(result) => serialize_handler_result(&result),
                    Err(e) => Err(wire_error_from_box_error(e)),
                }
            })
            .instrument(replay_span);
            // An invocation-failing panic (of the handler, of the
            // checkpoint client, or of any task the SDK started) returns
            // here only after the driver cleaned up every task, and leaves
            // the SDK with its original payload. When several panic, the
            // first one recorded is resumed. A recorded fatal error takes
            // precedence over a panic, so the driver then returns the fatal
            // outcome instead. Nothing runs between the cleanup and the
            // resume.
            let outcome = match driver::drive_invocation(
                root,
                task_ownership,
                coordinator,
                flush_ctx.clone(),
                remaining,
            )
            .await
            {
                Ok(outcome) => outcome,
                Err(payload) => std::panic::resume_unwind(payload),
            };

            // Unconditional flush point of the checkpoint buffering
            // contract (`checkpoint_delay` / `checkpoint_batching`):
            // whatever the outcome, suspension (PENDING), completion
            // (SUCCEEDED), or failure (FAILED), every buffered checkpoint
            // is written, and every in-flight batched write is awaited,
            // BEFORE the envelope reports the invocation's state to the
            // service, so buffering can never hold a checkpoint past the
            // end of the invocation. A no-op without configured buffering.
            //
            // Checkpoint-failure interplay (issue #43): a `Fault` outcome
            // skips the flush of pending writes: the write channel
            // already failed retryably; a follow-up write would fail the
            // same way, and the contract is "fails the invocation with no
            // further writes". But failures already RETAINED by detached
            // batch flushes (every contributor dropped: a lost
            // `race`/`select_ok` branch) are still drained: a retained
            // NON-retryable failure is deterministic, so dropping it here
            // would leave its operations `Started` and re-executing on
            // every future invocation, exactly the loop the flush point
            // exists to end. Such a failure routes through the same
            // terminalize-then-fail-the-execution path as a non-retryable
            // flush failure; retryable-only retained failures are dropped
            // with the invocation (the service re-invokes and the affected
            // operations re-run under the interruption contract). Every
            // other outcome flushes, and a flush failure routes through
            // the same classification as any other checkpoint write
            // failure (see `flush_failure_response`): retryable fails the
            // invocation, non-retryable persists terminal `FAIL` records
            // for the affected operations and fails the execution.
            if matches!(outcome, driver::InvocationOutcome::Fault { .. }) {
                // No flush of pending writes on a failed channel, but
                // classify retained failures (see above).
                if let Some(retained) = flush_ctx.take_retained_flush_failures().await
                    && retained.any_non_retryable()
                {
                    return flush_failure_response(&retained, outcome, &flush_ctx).await;
                }
            } else if let Err(flush) = flush_ctx.flush_pending_checkpoints().await {
                return flush_failure_response(&flush, outcome, &flush_ctx).await;
            }

            // Convert outcome to the durable response envelope.
            //
            // ENVELOPE CONTRACT: do not "fix" the `Ok` below. Every outcome,
            // including a handler failure, is reported inside a *successful*
            // Lambda invocation response: the durable execution service reads
            // the `Status` field of this envelope to record the execution
            // result, and it can only do that when the invocation itself
            // succeeds. Returning `Err` here would make the service treat the
            // invocation as a runtime fault and retry it, rather than marking
            // the execution FAILED with the handler's error.
            //
            // The one deliberate exception is `Fault`, a retryable
            // checkpoint failure, where retry-as-a-runtime-fault is
            // exactly the intended recovery, so it maps to `Err` inside
            // `outcome_envelope`.
            //
            // The observable consequence, which is intentional: a handler
            // failure does not increment the Lambda `Errors` metric, does not
            // route to a DLQ or OnFailure destination, and does not mark the
            // X-Ray trace as an error. Operators must monitor the durable
            // execution status (`GetDurableExecution` /
            // `ListDurableExecutionsByFunction`) instead. See the rustdoc on
            // [`run`] and [`wrap`].
            outcome_envelope(outcome, &flush_ctx)
        })
    }
}

/// Applies the issue #43 checkpoint-failure classification to a failed
/// end-of-invocation flush of buffered checkpoints, or, on the `Fault`
/// path, to failures retained by detached batch flushes (nothing is
/// flushed there; see the flush point in [`wrap`]).
///
/// - **Retryable** (every failure transient): the invocation fails with no
///   further writes: the channel is down, and the service re-invokes,
///   which is the same recovery as an interruption. This applies whatever
///   the driver outcome was: reporting SUCCEEDED or PENDING would claim
///   records the service never received, and the re-invocation converges
///   (dropped contributors' operations re-run under the documented
///   `AtLeastOncePerRetry` interruption contract).
/// - **Non-retryable** (any failure permanent): re-invoking would replay
///   into the same deterministic rejection, an infinite loop. Instead, a
///   small terminal `FAIL` is persisted for every operation whose buffered
///   outcome was lost (see
///   [`DurableContext::terminalize_unwritten_outcomes`]), and the
///   execution fails. A handler outcome of SUCCEEDED or PENDING is
///   overridden, completing while an operation record claims less than
///   what executed would violate the #43 invariant, and a `Fault`
///   outcome (the retained-failure drain path) likewise becomes an
///   execution failure carrying the retained error, while a handler that
///   already FAILED keeps its own error as the execution result (the
///   execution dies either way, and the handler's failure is the more
///   meaningful root cause).
async fn flush_failure_response(
    flush: &context::FlushFailure,
    outcome: driver::InvocationOutcome,
    ctx: &DurableContext,
) -> Result<InvocationResponse, lambda_runtime::Error> {
    let error = flush.primary_error();
    if !flush.any_non_retryable() {
        return Err(lambda_runtime::Error::from(format!(
            "failed to flush coalesced checkpoints at invocation end: {error}"
        )));
    }

    tracing::error!(
        error = %error,
        "non-retryable failure flushing coalesced checkpoints; recording terminal \
         FAILs for the affected operations and failing the execution"
    );
    ctx.terminalize_unwritten_outcomes(flush).await;

    let failed = match outcome {
        driver::InvocationOutcome::Failed { .. } => outcome,
        _ => driver::InvocationOutcome::Failed {
            error: error::checkpoint_execution_failure_wire(error),
        },
    };
    outcome_envelope(failed, ctx)
}

/// Converts the driver's invocation outcome into the durable response
/// envelope, emitting the `execution_suspended` lifecycle event on the
/// suspension path (see [`crate::observability`]).
///
/// The FAILED status deliberately travels in the envelope, not as a Lambda
/// invocation error: see the envelope contract note at the call site. The
/// `Fault` outcome is the deliberate inverse: a retryable checkpoint
/// failure returns `Err`, failing the Lambda invocation itself so the
/// durable service re-invokes (issue #43).
fn outcome_envelope(
    outcome: driver::InvocationOutcome,
    ctx: &DurableContext,
) -> Result<InvocationResponse, lambda_runtime::Error> {
    match outcome {
        driver::InvocationOutcome::Complete(serialized) => {
            Ok(InvocationResponse(serde_json::json!({
                "Status": "SUCCEEDED",
                "Result": serialized
            })))
        }
        driver::InvocationOutcome::Pending => {
            // Emitted while the handler's `durable_execution` span is
            // entered, the instrumented handler future has already been
            // dropped, but the context still holds the span handle, so
            // the event is a span event of the execution span, matching
            // the documented OpenTelemetry bridge (see
            // `crate::observability`).
            let execution_span = ctx.replay_span();
            let _execution_scope = execution_span.enter();
            tracing_layer::execution_suspended_event(
                ctx.execution_arn(),
                &ctx.lambda_context().request_id,
            );
            Ok(InvocationResponse(serde_json::json!({
                "Status": "PENDING"
            })))
        }
        driver::InvocationOutcome::Fault { message } => {
            // A retryable checkpoint failure: fail the invocation itself
            // (no envelope, no further writes). The service re-invokes,
            // which is the same recovery as an interruption.
            Err(lambda_runtime::Error::from(message))
        }
        driver::InvocationOutcome::Failed { error } => Ok(failed_envelope(&error)),
    }
}

/// Builds the `FAILED` envelope that records `error` as the execution's
/// failure.
fn failed_envelope(error: &WireError) -> InvocationResponse {
    let mut error_map = serde_json::Map::new();
    error_map.insert(
        "ErrorType".to_owned(),
        serde_json::Value::String(error.error_type().unwrap_or("Error").to_owned()),
    );
    error_map.insert(
        "ErrorMessage".to_owned(),
        serde_json::Value::String(error.error_message().unwrap_or_default().to_owned()),
    );
    if let Some(data) = error.error_data() {
        error_map.insert(
            "ErrorData".to_owned(),
            serde_json::Value::String(data.to_owned()),
        );
    }
    if !error.stack_trace().is_empty() {
        error_map.insert(
            "StackTrace".to_owned(),
            serde_json::json!(error.stack_trace()),
        );
    }
    InvocationResponse(serde_json::json!({
        "Status": "FAILED",
        "Error": serde_json::Value::Object(error_map)
    }))
}

/// Resolves the base Lambda client from the caller's [`Options`].
///
/// Precedence: a supplied `lambda_client` is used directly; otherwise a
/// supplied `sdk_config` builds one; otherwise `None`, which defers to the
/// ambient default resolved once at first use by [`ClientProvider`].
pub(crate) fn base_lambda_client_from_options(
    sdk_config: Option<SdkConfig>,
    lambda_client: Option<aws_sdk_lambda::Client>,
) -> Option<aws_sdk_lambda::Client> {
    match (lambda_client, sdk_config) {
        (Some(client), _) => Some(client),
        (None, Some(config)) => Some(aws_sdk_lambda::Client::new(&config)),
        (None, None) => None,
    }
}

/// Connect timeout of the Lambda client the SDK builds by default, as in
/// the JS SDK.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Read timeout of the Lambda client the SDK builds by default, as in the
/// JS SDK.
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(50);
/// Attempt timeout of the Lambda client the SDK builds by default, as the
/// JS SDK's request timeout.
const DEFAULT_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(55);

/// Builds the Lambda client the SDK uses when the caller supplies neither
/// a client nor a config.
///
/// The client takes every setting from `config` and adds three timeouts.
/// A read timeout restarts on every byte received, so it does not bound a
/// response that arrives slowly. The attempt timeout bounds it. So a lost
/// or slow checkpoint response fails the call after 55 s, rather than
/// blocking until the Lambda function times out.
fn default_lambda_client(config: &SdkConfig) -> aws_sdk_lambda::Client {
    use aws_sdk_lambda::config::timeout::TimeoutConfig;
    let conf = aws_sdk_lambda::config::Builder::from(config)
        .timeout_config(
            TimeoutConfig::builder()
                .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
                .read_timeout(DEFAULT_READ_TIMEOUT)
                .operation_attempt_timeout(DEFAULT_ATTEMPT_TIMEOUT)
                .build(),
        )
        .build();
    aws_sdk_lambda::Client::from_conf(conf)
}

/// Supplies the execution client for every invocation of a [`wrap`]-ed
/// handler, building it at most once and reusing it thereafter.
///
/// When [`Options`] supplied a client (or an SDK config), it is captured as
/// `preset` and returned on every call. Otherwise the ambient default config
/// is loaded lazily on the first invocation and the resulting client is
/// cached, so no per-invocation client construction or config load occurs.
pub(crate) struct ClientProvider {
    preset: Option<std::sync::Arc<dyn ExecutionClient>>,
    default_cell: tokio::sync::OnceCell<std::sync::Arc<dyn ExecutionClient>>,
}

impl ClientProvider {
    /// Creates a provider. `preset` is the client resolved from `Options`, or
    /// `None` to defer to the ambient default on first use.
    pub(crate) fn new(preset: Option<std::sync::Arc<dyn ExecutionClient>>) -> Self {
        Self {
            preset,
            default_cell: tokio::sync::OnceCell::new(),
        }
    }

    /// Returns the shared execution client, building the ambient-default one
    /// exactly once when no client was preset.
    pub(crate) async fn get(&self) -> std::sync::Arc<dyn ExecutionClient> {
        use std::sync::Arc as StdArc;
        if let Some(preset) = &self.preset {
            return StdArc::clone(preset);
        }
        let client = self
            .default_cell
            .get_or_init(|| async {
                let aws_config =
                    aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
                let lambda_client = default_lambda_client(&aws_config);
                StdArc::new(LambdaExecutionClient::new(lambda_client))
                    as StdArc<dyn ExecutionClient>
            })
            .await;
        StdArc::clone(client)
    }
}

#[cfg(test)]
mod tests {
    use std::future::IntoFuture;

    use super::*;

    #[test]
    fn due_at_from_inline_timestamps() {
        let payload = serde_json::json!({
            "InitialExecutionState": {
                "Operations": [
                    { "Id": "w1", "Type": "WAIT", "Status": "STARTED",
                      "WaitDetails": { "ScheduledEndTimestamp": "2026-01-01T00:00:00Z" } },
                    { "Id": "s1", "Type": "STEP", "Status": "PENDING",
                      "StepDetails": { "NextAttemptTimestamp": "2026-01-01T00:00:05.5Z" } },
                    { "Id": "w2", "Type": "WAIT", "Status": "STARTED",
                      "WaitDetails": { "ScheduledEndTimestamp": 1_767_225_600_000_i64 } },
                    { "Id": "w3", "Type": "WAIT", "Status": "STARTED",
                      "WaitDetails": { "ScheduledEndTimestamp": "not a date" } },
                    { "Id": "w4", "Type": "WAIT", "Status": "STARTED",
                      "WaitDetails": { "ScheduledEndTimestamp": 1.5 } },
                    { "Id": "w5", "Type": "WAIT", "Status": "STARTED",
                      "WaitDetails": { "ScheduledEndTimestamp": true } },
                ]
            }
        });
        let (log, _) = parse_inline_operations(&payload);
        let due = |id: &str| log.get(id).expect("parsed").due_at;
        let new_year = std::time::UNIX_EPOCH + Duration::from_hours(490_896);
        assert_eq!(due("w1"), Some(new_year));
        assert_eq!(due("s1"), Some(new_year + Duration::from_millis(5_500)));
        assert_eq!(due("w2"), Some(new_year), "epoch milliseconds");
        assert_eq!(due("w3"), None, "a string that does not parse");
        assert_eq!(due("w4"), None, "a number that is not an integer");
        assert_eq!(due("w5"), None, "another JSON type");
    }

    /// A timestamp that no platform's `SystemTime` can hold.
    struct OutOfRange;

    impl TryFrom<OutOfRange> for std::time::SystemTime {
        type Error = &'static str;

        fn try_from(_: OutOfRange) -> Result<Self, Self::Error> {
            Err("outside the range of SystemTime")
        }
    }

    /// Both `due_at` mappings convert through `to_system_time`. A timestamp
    /// in range converts, and one outside the range of `SystemTime` gives
    /// `None`, as a missing timestamp does.
    #[test]
    fn to_system_time_maps_an_out_of_range_timestamp_to_none() {
        let millis = aws_sdk_lambda::primitives::DateTime::from_millis(1_500);
        assert_eq!(
            to_system_time(millis),
            Some(std::time::UNIX_EPOCH + Duration::from_millis(1_500))
        );
        assert_eq!(to_system_time(OutOfRange), None);
    }

    #[derive(Debug)]
    struct PlainUserError;

    impl std::fmt::Display for PlainUserError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "plain user failure")
        }
    }

    impl std::error::Error for PlainUserError {}

    #[test]
    fn handler_error_records_no_stack_trace() {
        let wire = wire_error_from_box_error(Box::new(PlainUserError));
        assert_eq!(wire.error_type(), Some("Error"));
        assert_eq!(wire.error_message(), Some("plain user failure"));
        assert_eq!(wire.stack_trace(), [] as [String; 0]);
    }

    /// A value whose `Serialize` implementation always fails.
    struct Refuse;

    impl Serialize for Refuse {
        fn serialize<S: serde::Serializer>(&self, _serializer: S) -> Result<S::Ok, S::Error> {
            Err(<S::Error as serde::ser::Error>::custom("refused"))
        }
    }

    #[test]
    fn handler_result_serialization_failure_records_error() {
        let Err(wire) = serialize_handler_result(&Refuse) else {
            unreachable!("a refusing value must not serialize");
        };
        assert_eq!(wire.error_type(), Some("Error"));
        assert!(
            wire.error_message().unwrap_or_default().contains("refused"),
            "{:?}",
            wire.error_message()
        );
        assert_eq!(wire.stack_trace(), [] as [String; 0]);
    }

    /// Verifies that `tokio::join!` accepts `IntoFuture` operands directly.
    ///
    /// Since tokio 1.23+, `tokio::join!` desugars through `.await` which
    /// uses `IntoFuture`. This means operation builders can be passed
    /// directly to `tokio::join!` without calling `.future()` first.
    #[tokio::test]
    async fn tokio_join_accepts_into_future() {
        // Verify compilation: IntoFuture is accepted by tokio::join!
        fn check_into_future<T: IntoFuture>(_t: T) {}
        let ctx = DurableContext::test_context();
        check_into_future(ctx.step(|_| async { Ok(1i32) }));
        // NOTE: cannot actually tokio::join! the builders because they
        // todo!() at runtime, but the type-level verification above
        // plus the external rustc test confirms IntoFuture acceptance.
    }

    /// Verifies that `wrap()` produces a service function compatible with
    /// `lambda_runtime::service_fn`. This is a compile-time + type-level
    /// test: the returned closure has the correct signature.
    #[test]
    fn wrap_returns_callable_service_function() {
        fn assert_send_sync<T: Send + Sync>(_t: &T) {}

        // Verify that wrap() compiles and returns something Send + Sync.
        let service = wrap(
            |_event: serde_json::Value, _ctx: DurableContext| async move {
                Ok::<String, BoxError>("hello".to_owned())
            },
            Options::default(),
        );

        // The service must be Send + Sync (required by lambda_runtime::run).
        assert_send_sync(&service);

        // Verify the closure can be called (type-level check; we cannot
        // actually invoke without a real Lambda event envelope but the
        // fact that `service` is accepted by `service_fn` is proven by
        // the Send + Sync + correct return type checks above).
        drop(service);
    }

    // Service-level entry-point envelope tests
    //
    // These invoke the `wrap`-produced service end to end, covering the
    // entry-point envelope handling that the `parse_envelope` unit tests
    // alone cannot reach.

    /// Offline `Options`: a Lambda client built from a static config so the
    /// service never loads ambient AWS configuration. The client is only
    /// exercised by the happy-path test, which makes no AWS calls (single
    /// inline state page, no checkpointed operations).
    fn offline_options() -> Options {
        let conf = aws_sdk_lambda::config::Config::builder()
            .behavior_version(aws_sdk_lambda::config::BehaviorVersion::latest())
            .region(aws_sdk_lambda::config::Region::new("us-east-1"))
            .build();
        Options::builder()
            .lambda_client(aws_sdk_lambda::Client::from_conf(conf))
            .build()
            .expect("offline options build")
    }

    /// Invokes the `wrap`-produced echo service with the given payload.
    async fn invoke_wrap_service(
        payload: serde_json::Value,
    ) -> Result<serde_json::Value, lambda_runtime::Error> {
        let service = wrap(
            |event: serde_json::Value, _ctx: DurableContext| async move {
                Ok::<serde_json::Value, BoxError>(event)
            },
            offline_options(),
        );
        let event = lambda_runtime::LambdaEvent::new(
            InvocationPayload(payload),
            lambda_runtime::Context::default(),
        );
        service(event).await.map(|response| response.0)
    }

    #[tokio::test]
    async fn wrap_service_missing_arn_fails() {
        let payload = serde_json::json!({
            "CheckpointToken": "token-abc",
            "InitialExecutionState": { "Operations": [] }
        });
        let err = invoke_wrap_service(payload).await.expect_err("must fail");
        assert!(
            err.to_string().contains("DurableExecutionArn"),
            "error should name the missing field, got: {err}"
        );
    }

    #[tokio::test]
    async fn wrap_service_missing_token_fails() {
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "InitialExecutionState": { "Operations": [] }
        });
        let err = invoke_wrap_service(payload).await.expect_err("must fail");
        assert!(
            err.to_string().contains("CheckpointToken"),
            "error should name the missing field, got: {err}"
        );
    }

    #[tokio::test]
    async fn wrap_service_envelope_free_payload_fails() {
        // An envelope-free payload fails fast at the entry point. There is
        // no raw-payload fallback on the service paths, with or without
        // `test-util`; local testing goes through `LocalRunner` instead.
        let payload = serde_json::json!({ "count": 42 });
        let err = invoke_wrap_service(payload).await.expect_err("must fail");
        assert!(
            err.to_string().contains("not a durable execution envelope"),
            "error should describe the missing envelope, got: {err}"
        );
    }

    #[tokio::test]
    async fn wrap_service_valid_envelope_succeeds() {
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:aws:lambda:us-east-1:123456789012:function:test",
            "CheckpointToken": "token-abc",
            "InitialExecutionState": {
                "Operations": [{
                    "Id": "root",
                    "Type": "Execution",
                    "Status": "STARTED",
                    "ExecutionDetails": { "InputPayload": "{\"count\":42}" }
                }]
            }
        });
        let response = invoke_wrap_service(payload).await.expect("must succeed");
        assert_eq!(
            response.get("Status").and_then(serde_json::Value::as_str),
            Some("SUCCEEDED"),
            "unexpected response: {response}"
        );
        let result_json = response
            .get("Result")
            .and_then(serde_json::Value::as_str)
            .expect("Result should be a serialized JSON string");
        let echoed: serde_json::Value = serde_json::from_str(result_json).unwrap();
        assert_eq!(echoed, serde_json::json!({ "count": 42 }));
    }

    /// A handler panic leaves the service function with its original
    /// payload, after the SDK cleaned up its tasks, and no envelope is
    /// produced.
    #[tokio::test]
    // reason: the test calls the service on its own task so the panic is
    // observable as that task's `JoinError`.
    #[expect(clippy::disallowed_methods)]
    async fn handler_panic_leaves_sdk_with_original_payload() {
        #[expect(clippy::panic)] // reason: the handler panic under test
        fn boom() -> Result<(), BoxError> {
            panic!("root boom")
        }
        let client: std::sync::Arc<dyn ExecutionClient> =
            std::sync::Arc::new(client::InMemoryExecutionClient::new(Vec::new()));
        let service = wrap_with_provider(
            |_event: serde_json::Value, _ctx: DurableContext| async move { boom() },
            ClientProvider::new(Some(client)),
            None,
        );
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:test",
            "CheckpointToken": "token-abc",
            "InitialExecutionState": {
                "Operations": [{
                    "Id": "root",
                    "Type": "Execution",
                    "Status": "STARTED",
                    "ExecutionDetails": { "InputPayload": "{}" }
                }]
            }
        });
        let event = lambda_runtime::LambdaEvent::new(
            InvocationPayload(payload),
            lambda_runtime::Context::default(),
        );
        let join_err = tokio::spawn(service(event))
            .await
            .expect_err("the handler panic leaves the service, and no envelope is produced");
        assert!(join_err.is_panic(), "a panic, not a cancellation");
        let payload = join_err.into_panic();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"root boom"));
    }

    // Bootstrap state read errors

    /// An envelope with the input operation and a `NextMarker`, so the
    /// service must read the next page with `GetDurableExecutionState`.
    fn paginated_envelope() -> serde_json::Value {
        serde_json::json!({
            "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:test",
            "CheckpointToken": "token-abc",
            "InitialExecutionState": {
                "Operations": [{
                    "Id": "root",
                    "Type": "Execution",
                    "Status": "STARTED",
                    "ExecutionDetails": { "InputPayload": "{}" }
                }],
                "NextMarker": "page-2"
            }
        })
    }

    /// Invokes a `wrap` service whose Lambda client is `sdk_client` with
    /// [`paginated_envelope`]. Returns the result and whether the handler
    /// ran.
    async fn invoke_paginated(
        sdk_client: aws_sdk_lambda::Client,
    ) -> (Result<serde_json::Value, lambda_runtime::Error>, bool) {
        invoke_paginated_with(
            Options::builder()
                .lambda_client(sdk_client)
                .build()
                .expect("options build"),
        )
        .await
    }

    /// Invokes a `wrap` service built from `options` with
    /// [`paginated_envelope`]. Returns the result and whether the handler
    /// ran.
    async fn invoke_paginated_with(
        options: Options,
    ) -> (Result<serde_json::Value, lambda_runtime::Error>, bool) {
        use std::sync::atomic::{AtomicBool, Ordering};

        let ran = std::sync::Arc::new(AtomicBool::new(false));
        let handler_ran = std::sync::Arc::clone(&ran);
        let service = wrap(
            move |_event: serde_json::Value, _ctx: DurableContext| {
                handler_ran.store(true, Ordering::SeqCst);
                async move { Ok::<(), BoxError>(()) }
            },
            options,
        );
        let event = lambda_runtime::LambdaEvent::new(
            InvocationPayload(paginated_envelope()),
            lambda_runtime::Context::default(),
        );
        let result = service(event).await.map(|response| response.0);
        (result, ran.load(Ordering::SeqCst))
    }

    /// A KMS error on the bootstrap state read fails the execution with a
    /// `FAILED` envelope, as in JS, and the handler does not run.
    #[tokio::test]
    async fn bootstrap_kms_error_returns_failed_envelope() {
        use aws_smithy_mocks::{mock, mock_client};

        let rule =
            mock!(aws_sdk_lambda::Client::get_durable_execution_state).then_http_response(|| {
                client::mock_aws_error_response(400, "KMSAccessDeniedException", "kms denied")
            });
        let (result, ran) = invoke_paginated(mock_client!(aws_sdk_lambda, [&rule])).await;

        let response = result.expect("a KMS error returns an envelope");
        assert_eq!(
            response.get("Status").and_then(serde_json::Value::as_str),
            Some("FAILED"),
            "unexpected response: {response}"
        );
        let error = response
            .get("Error")
            .expect("a FAILED envelope has an Error");
        assert_eq!(
            error.get("ErrorType").and_then(serde_json::Value::as_str),
            Some("KMSAccessDeniedException")
        );
        assert_eq!(
            error
                .get("ErrorMessage")
                .and_then(serde_json::Value::as_str),
            Some("kms denied")
        );
        assert!(!ran, "the handler must not run");
    }

    /// The KMS scenario through the public `LambdaExecutionClient` passed as
    /// `execution_client`: the `FAILED` record carries the AWS error code
    /// and the service's message, read from the `SdkError` source.
    #[tokio::test]
    async fn bootstrap_kms_error_records_aws_error_code() {
        use aws_smithy_mocks::{mock, mock_client};

        let rule =
            mock!(aws_sdk_lambda::Client::get_durable_execution_state).then_http_response(|| {
                client::mock_aws_error_response(400, "KMSAccessDeniedException", "kms denied")
            });
        let client = mock_client!(aws_sdk_lambda, [&rule]);
        let (result, ran) = invoke_paginated_with(
            Options::builder()
                .execution_client(std::sync::Arc::new(LambdaExecutionClient::new(client)))
                .build()
                .expect("options build"),
        )
        .await;

        let response = result.expect("a KMS error returns an envelope");
        assert_eq!(
            response.get("Status").and_then(serde_json::Value::as_str),
            Some("FAILED"),
            "unexpected response: {response}"
        );
        let error = response
            .get("Error")
            .expect("a FAILED envelope has an Error");
        assert_eq!(
            error.get("ErrorType").and_then(serde_json::Value::as_str),
            Some("KMSAccessDeniedException")
        );
        assert_eq!(
            error
                .get("ErrorMessage")
                .and_then(serde_json::Value::as_str),
            Some("kms denied")
        );
        assert!(!ran, "the handler must not run");
    }

    /// A non-retryable checkpoint rejection through the public
    /// `LambdaExecutionClient` records the execution's error as
    /// `CheckpointUnrecoverableExecutionError` with the service's message.
    #[tokio::test]
    async fn lambda_checkpoint_rejection_records_the_service_message() {
        use aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionOutput;
        use aws_smithy_mocks::{RuleMode, mock, mock_client};

        let rejected = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
            .then_http_response(|| {
                client::mock_aws_error_response(
                    400,
                    "InvalidParameterValueException",
                    "bad payload",
                )
            });
        let accepted =
            mock!(aws_sdk_lambda::Client::checkpoint_durable_execution).then_output(|| {
                CheckpointDurableExecutionOutput::builder()
                    .checkpoint_token("tok-2")
                    .build()
            });
        let client = mock_client!(aws_sdk_lambda, RuleMode::Sequential, [&rejected, &accepted]);
        let service = wrap(
            |_event: serde_json::Value, ctx: DurableContext| async move {
                let value: u32 = ctx.step(|_| async { Ok(7_u32) }).name("one").await?;
                Ok::<u32, BoxError>(value)
            },
            Options::builder()
                .execution_client(std::sync::Arc::new(LambdaExecutionClient::new(client)))
                .build()
                .expect("options build"),
        );
        let event = lambda_runtime::LambdaEvent::new(
            InvocationPayload(serde_json::json!({
                "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:test",
                "CheckpointToken": "token-abc",
                "InitialExecutionState": {
                    "Operations": [{
                        "Id": "root",
                        "Type": "Execution",
                        "Status": "STARTED",
                        "ExecutionDetails": { "InputPayload": "{}" }
                    }]
                }
            })),
            lambda_runtime::Context::default(),
        );
        let response = service(event)
            .await
            .map(|response| response.0)
            .expect("a rejected checkpoint returns an envelope");

        assert_eq!(
            response.get("Status").and_then(serde_json::Value::as_str),
            Some("FAILED"),
            "unexpected response: {response}"
        );
        let error = response
            .get("Error")
            .expect("a FAILED envelope has an Error");
        assert_eq!(
            error.get("ErrorType").and_then(serde_json::Value::as_str),
            Some("CheckpointUnrecoverableExecutionError")
        );
        assert_eq!(
            error
                .get("ErrorMessage")
                .and_then(serde_json::Value::as_str),
            Some("[Unrecoverable Execution] Checkpoint failed: bad payload")
        );
    }

    /// A retryable error on the bootstrap state read, a persistent 503 or a
    /// plain 403, fails the invocation, as in JS.
    #[tokio::test]
    async fn bootstrap_retryable_error_fails_the_invocation() {
        use aws_smithy_mocks::{mock, mock_client};

        for status in [503, 403] {
            let rule = mock!(aws_sdk_lambda::Client::get_durable_execution_state)
                .sequence()
                .http_status(status, None)
                .repeatedly()
                .build();
            let (result, ran) = invoke_paginated(mock_client!(aws_sdk_lambda, [&rule])).await;

            let err = result.expect_err("a retryable error fails the invocation");
            assert!(
                err.to_string().contains("failed to paginate initial state"),
                "status {status}: {err}"
            );
            assert!(!ran, "the handler must not run");
        }
    }

    // Status parsing tests (issue #45)

    /// Builds a minimal step operation JSON with the given wire status.
    fn step_op_with_status(status: &str) -> serde_json::Value {
        serde_json::json!({
            "Id": "op-1",
            "Type": "Step",
            "Status": status,
            "StepDetails": {}
        })
    }

    #[test]
    fn parse_status_timed_out_wire_value_maps_to_timed_out() {
        let (_, record) = parse_single_operation(&step_op_with_status("TIMED_OUT")).unwrap();
        assert_eq!(record.status, engine::CheckpointStatus::TimedOut);
    }

    /// `TIMEDOUT` is not a wire value, the smithy model and the Python and
    /// JS SDKs spell it `TIMED_OUT` only, so the parser must not accept it
    /// (the removed pre-#45 arm did).
    #[test]
    fn parse_status_timedout_without_underscore_is_unrecognized() {
        let (_, record) = parse_single_operation(&step_op_with_status("TIMEDOUT")).unwrap();
        assert_eq!(
            record.status,
            engine::CheckpointStatus::Unknown("TIMEDOUT".to_owned())
        );
    }

    /// An unrecognized status maps to `Unknown` carrying the raw value as
    /// received (original casing), never to a guessed known status.
    #[test]
    fn parse_unknown_status_carries_raw_value() {
        let (_, record) = parse_single_operation(&step_op_with_status("Paused")).unwrap();
        assert_eq!(
            record.status,
            engine::CheckpointStatus::Unknown("Paused".to_owned())
        );
    }

    /// Every modeled wire status still maps explicitly, and the parser
    /// stays case-insensitive for them.
    #[test]
    fn parse_known_statuses_map_explicitly() {
        let cases = [
            ("STARTED", engine::CheckpointStatus::Started),
            ("PENDING", engine::CheckpointStatus::Pending),
            ("READY", engine::CheckpointStatus::Ready),
            ("SUCCEEDED", engine::CheckpointStatus::Succeeded),
            ("FAILED", engine::CheckpointStatus::Failed),
            ("CANCELLED", engine::CheckpointStatus::Cancelled),
            ("TIMED_OUT", engine::CheckpointStatus::TimedOut),
            ("STOPPED", engine::CheckpointStatus::Stopped),
            ("Succeeded", engine::CheckpointStatus::Succeeded),
        ];
        for (wire, expected) in cases {
            let (_, record) = parse_single_operation(&step_op_with_status(wire)).unwrap();
            assert_eq!(record.status, expected, "wire status {wire:?}");
        }
    }

    // CallbackDetails parsing tests

    #[test]
    fn parse_callback_details_extracts_result() {
        let op = serde_json::json!({
            "Id": "abc123",
            "Type": "Callback",
            "Status": "SUCCEEDED",
            "CallbackDetails": {
                "CallbackId": "cb-42",
                "Result": "\"hello from callback\""
            }
        });

        let parsed = parse_single_operation(&op);
        assert!(parsed.is_some());
        let (id, record) = parsed.unwrap();
        assert_eq!(id, "abc123");
        assert_eq!(record.status, engine::CheckpointStatus::Succeeded);
        assert_eq!(record.callback_id.as_deref(), Some("cb-42"));
        assert_eq!(record.result.as_deref(), Some("\"hello from callback\""));
    }

    #[test]
    fn parse_callback_details_extracts_error() {
        let op = serde_json::json!({
            "Id": "abc456",
            "Type": "Callback",
            "Status": "FAILED",
            "CallbackDetails": {
                "CallbackId": "cb-99",
                "Error": {
                    "ErrorType": "NotApproved",
                    "ErrorMessage": "request was denied"
                }
            }
        });

        let parsed = parse_single_operation(&op);
        assert!(parsed.is_some());
        let (id, record) = parsed.unwrap();
        assert_eq!(id, "abc456");
        assert_eq!(record.status, engine::CheckpointStatus::Failed);
        assert_eq!(record.callback_id.as_deref(), Some("cb-99"));
        assert_eq!(record.error_type.as_deref(), Some("NotApproved"));
        assert_eq!(record.error_message.as_deref(), Some("request was denied"));
    }

    #[test]
    fn parse_callback_details_result_does_not_override_step_result() {
        // StepDetails.Result takes priority; CallbackDetails.Result is
        // only a fallback for callback-type operations without step data.
        let op = serde_json::json!({
            "Id": "abc789",
            "Type": "Callback",
            "Status": "SUCCEEDED",
            "StepDetails": {
                "Result": "\"from step\""
            },
            "CallbackDetails": {
                "CallbackId": "cb-1",
                "Result": "\"from callback\""
            }
        });

        let parsed = parse_single_operation(&op);
        assert!(parsed.is_some());
        let (_, record) = parsed.unwrap();
        assert_eq!(record.result.as_deref(), Some("\"from step\""));
    }

    #[test]
    fn parse_inline_operations_handles_callback_succeeded() {
        let payload = serde_json::json!({
            "InitialExecutionState": {
                "Operations": [
                    {
                        "Id": "exec-0",
                        "Type": "Execution",
                        "Status": "STARTED"
                    },
                    {
                        "Id": "wire-id-1",
                        "Type": "Callback",
                        "Status": "SUCCEEDED",
                        "CallbackDetails": {
                            "CallbackId": "cb-id-123",
                            "Result": "\"payload\""
                        }
                    }
                ]
            }
        });

        let (log, marker) = parse_inline_operations(&payload);
        assert!(marker.is_none());
        let record = log.get("wire-id-1");
        assert!(record.is_some());
        let record = record.unwrap();
        assert_eq!(record.callback_id.as_deref(), Some("cb-id-123"));
        assert_eq!(record.result.as_deref(), Some("\"payload\""));
    }

    /// When `InitialExecutionState` includes a `NextMarker`, the parser
    /// returns it alongside the parsed operations so the caller can
    /// paginate.
    #[test]
    fn parse_inline_operations_returns_next_marker() {
        let payload = serde_json::json!({
            "InitialExecutionState": {
                "Operations": [
                    {
                        "Id": "exec-0",
                        "Type": "Execution",
                        "Status": "STARTED"
                    },
                    {
                        "Id": "wire-id-1",
                        "Type": "Step",
                        "Status": "SUCCEEDED",
                        "StepDetails": {
                            "Attempt": 1,
                            "Result": "\"hello\""
                        }
                    }
                ],
                "NextMarker": "page-token-2"
            }
        });

        let (log, marker) = parse_inline_operations(&payload);
        // The first page's operation is still parsed.
        let record = log.get("wire-id-1");
        assert!(record.is_some());
        // The marker signals that more pages are available.
        assert_eq!(marker, Some("page-token-2".to_owned()));
    }

    /// An empty `NextMarker` is treated as no marker (no pagination needed).
    #[test]
    fn parse_inline_operations_ignores_empty_marker() {
        let payload = serde_json::json!({
            "InitialExecutionState": {
                "Operations": [
                    {
                        "Id": "exec-0",
                        "Type": "Execution",
                        "Status": "STARTED"
                    }
                ],
                "NextMarker": ""
            }
        });

        let (_log, marker) = parse_inline_operations(&payload);
        assert_eq!(marker, None);
    }

    /// A payload with a `NextMarker` but no `Operations` array still
    /// reports the marker: the service may omit `Operations` on the first
    /// page (e.g. when a large customer payload displaces it), and the
    /// remaining pages must still be fetched rather than silently skipped.
    #[test]
    fn parse_inline_operations_keeps_marker_without_operations() {
        let payload = serde_json::json!({
            "InitialExecutionState": {
                "NextMarker": "page-token-1"
            }
        });

        let (log, marker) = parse_inline_operations(&payload);
        // No operations yet: the first page is empty.
        assert!(!log.has_records());
        // But the marker must survive so bootstrap pagination runs.
        assert_eq!(marker, Some("page-token-1".to_owned()));
    }

    /// Helper to build a Step operation for `resolve_bootstrap_log` tests.
    fn make_test_step_op(id: &str, result: &str) -> aws_sdk_lambda::types::Operation {
        aws_sdk_lambda::types::Operation::builder()
            .id(id)
            .r#type(aws_sdk_lambda::types::OperationType::Step)
            .status(aws_sdk_lambda::types::OperationStatus::Succeeded)
            .start_timestamp(aws_sdk_lambda::primitives::DateTime::from_secs(0))
            .step_details(
                aws_sdk_lambda::types::StepDetails::builder()
                    .result(result)
                    .build(),
            )
            .build()
            .unwrap()
    }

    /// Helper to build an Execution operation carrying `input`.
    fn make_test_execution_op(input: &str) -> aws_sdk_lambda::types::Operation {
        aws_sdk_lambda::types::Operation::builder()
            .id("execution")
            .r#type(aws_sdk_lambda::types::OperationType::Execution)
            .status(aws_sdk_lambda::types::OperationStatus::Started)
            .start_timestamp(aws_sdk_lambda::primitives::DateTime::from_secs(0))
            .execution_details(
                aws_sdk_lambda::types::ExecutionDetails::builder()
                    .input_payload(input)
                    .build(),
            )
            .build()
            .unwrap()
    }

    /// When `initial_marker` is `Some`, `resolve_bootstrap_log` calls
    /// `get_state` (count == 1) and returns a log built from the full
    /// paginated state, and the input of the fetched Execution operation.
    #[tokio::test]
    async fn resolve_bootstrap_log_paginates_when_marker_present() {
        let all_ops = vec![
            make_test_execution_op("{\"fetched\":true}"),
            make_test_step_op("step-1", "\"r1\""),
            make_test_step_op("step-2", "\"r2\""),
        ];
        let client = client::InMemoryExecutionClient::new(all_ops);

        // The inline page is empty and holds no Execution operation.
        let inline_log = engine::CheckpointLog::empty();

        let result = client::resolve_bootstrap_log(
            &client,
            "arn:test",
            "token",
            inline_log,
            client::PageInput::NoExecutionOperation,
            Some("page-2-marker"),
        )
        .await;

        assert!(result.is_ok());
        let (log, input) = result.unwrap();
        assert_eq!(
            input.as_deref(),
            Some("{\"fetched\":true}"),
            "the input must come from the fetched Execution operation"
        );
        assert!(
            log.get("execution").is_none(),
            "the Execution operation must not enter the log"
        );
        // Full state from get_state is used.
        assert!(log.get("step-1").is_some(), "step-1 must be in the log");
        assert!(
            log.get("step-2").is_some(),
            "step-2 must be in the log (from page 2)"
        );

        // get_state was called exactly once.
        let count = *client.get_state_call_count.lock().unwrap();
        assert_eq!(
            count, 1,
            "get_state must be called exactly once when marker is present"
        );
    }

    /// When `initial_marker` is `None`, `resolve_bootstrap_log` does NOT
    /// call `get_state` (count == 0) and returns the inline log as-is.
    #[tokio::test]
    async fn resolve_bootstrap_log_skips_pagination_when_no_marker() {
        let client = client::InMemoryExecutionClient::new(Vec::new());

        let inline_log = engine::CheckpointLog::empty();
        // Insert a record to prove the inline log is returned unchanged.
        inline_log.insert(
            "existing-op".to_owned(),
            engine::CheckpointRecord {
                id: "existing-op".to_owned(),
                status: engine::CheckpointStatus::Succeeded,
                result: Some("\"inline\"".to_owned()),
                error_type: None,
                error_message: None,
                error_data: None,
                stack_trace: None,
                attempt: 1,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                invoke_error_data: None,
                invoke_stack_trace: None,
                replay_children: false,
                callback_id: None,
                op_type: None,
                sub_type: None,
                due_at: None,
                op_name: None,
            },
        );

        let result = client::resolve_bootstrap_log(
            &client,
            "arn:test",
            "token",
            inline_log,
            client::PageInput::Execution(Some("\"inline-input\"".to_owned())),
            None,
        )
        .await;

        assert!(result.is_ok());
        let (log, input) = result.unwrap();
        // The inline input is returned unchanged.
        assert_eq!(input.as_deref(), Some("\"inline-input\""));
        // The inline log is returned as-is.
        assert!(
            log.get("existing-op").is_some(),
            "inline op must be preserved"
        );

        // get_state was NOT called.
        let count = *client.get_state_call_count.lock().unwrap();
        assert_eq!(
            count, 0,
            "get_state must not be called when no marker is present"
        );
    }

    // Options consumption: client resolution + reuse

    /// A supplied `sdk_config` measurably alters client construction: the
    /// resolved Lambda client carries the region from that config.
    #[test]
    fn sdk_config_measurably_alters_client_construction() {
        let sdk_config = SdkConfig::builder()
            .behavior_version(aws_config::BehaviorVersion::latest())
            .region(aws_sdk_lambda::config::Region::new("eu-west-1"))
            .build();
        let client = base_lambda_client_from_options(Some(sdk_config), None)
            .expect("sdk_config must yield a client");
        assert_eq!(
            client.config().region().map(ToString::to_string),
            Some("eu-west-1".to_owned()),
            "the supplied sdk_config's region must flow into the built client"
        );
    }

    /// A supplied `lambda_client` is the one used (not a default-constructed
    /// one): the resolved client preserves the supplied client's region.
    #[test]
    fn supplied_lambda_client_is_the_one_used() {
        let conf = aws_sdk_lambda::config::Config::builder()
            .behavior_version(aws_sdk_lambda::config::BehaviorVersion::latest())
            .region(aws_sdk_lambda::config::Region::new("ap-south-1"))
            .build();
        let supplied = aws_sdk_lambda::Client::from_conf(conf);
        let resolved = base_lambda_client_from_options(None, Some(supplied))
            .expect("lambda_client must be returned");
        assert_eq!(
            resolved.config().region().map(ToString::to_string),
            Some("ap-south-1".to_owned()),
            "the supplied lambda_client must be used verbatim, not replaced"
        );
    }

    /// With neither `sdk_config` nor `lambda_client`, resolution defers to the
    /// ambient default (returns `None` so `ClientProvider` builds it lazily).
    #[test]
    fn no_options_defers_client_to_ambient_default() {
        assert!(base_lambda_client_from_options(None, None).is_none());
    }

    /// `ClientProvider` reuses a preset execution client across calls rather
    /// than rebuilding one per invocation: two `get()` calls return the same
    /// `Arc` allocation.
    #[tokio::test]
    async fn client_provider_reuses_preset_across_invocations() {
        use crate::client::InMemoryExecutionClient;
        use std::sync::Arc as StdArc;

        let preset: StdArc<dyn ExecutionClient> =
            StdArc::new(InMemoryExecutionClient::new(Vec::new()));
        let provider = ClientProvider::new(Some(StdArc::clone(&preset)));

        let first = provider.get().await;
        let second = provider.get().await;
        assert!(
            StdArc::ptr_eq(&first, &second),
            "the client must be reused, not rebuilt, across invocations"
        );
        assert!(
            StdArc::ptr_eq(&first, &preset),
            "the reused client must be exactly the one supplied via Options"
        );
    }

    // Envelope validation tests

    #[test]
    fn parse_envelope_valid_payload() {
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:aws:lambda:us-west-2:123456789012:function:test",
            "CheckpointToken": "tok-abc",
            "InitialExecutionState": { "Operations": [] }
        });
        let result = parse_envelope(&payload);
        assert!(result.is_ok());
        let envelope = result.unwrap().expect("envelope should be Some");
        assert_eq!(
            envelope.execution_arn,
            "arn:aws:lambda:us-west-2:123456789012:function:test"
        );
        assert_eq!(envelope.checkpoint_token, "tok-abc");
    }

    #[test]
    fn parse_envelope_missing_arn_errors() {
        let payload = serde_json::json!({
            "CheckpointToken": "tok-abc",
            "InitialExecutionState": { "Operations": [] }
        });
        let err = parse_envelope(&payload).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("DurableExecutionArn") && msg.contains("missing"),
            "error should name the missing field, got: {msg}"
        );
    }

    #[test]
    fn parse_envelope_missing_token_errors() {
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:test",
            "InitialExecutionState": { "Operations": [] }
        });
        let err = parse_envelope(&payload).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("CheckpointToken") && msg.contains("missing"),
            "error should name the missing field, got: {msg}"
        );
    }

    #[test]
    fn parse_envelope_arn_wrong_type_errors() {
        let payload = serde_json::json!({
            "DurableExecutionArn": 12345,
            "CheckpointToken": "tok-abc",
            "InitialExecutionState": { "Operations": [] }
        });
        let err = parse_envelope(&payload).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("DurableExecutionArn") && msg.contains("not a string"),
            "error should note the type mismatch, got: {msg}"
        );
    }

    #[test]
    fn parse_envelope_token_wrong_type_errors() {
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:test",
            "CheckpointToken": ["not", "a", "string"],
            "InitialExecutionState": { "Operations": [] }
        });
        let err = parse_envelope(&payload).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("CheckpointToken") && msg.contains("not a string"),
            "error should note the type mismatch, got: {msg}"
        );
    }

    #[test]
    fn parse_envelope_no_envelope_returns_none() {
        // A plain customer event with no envelope fields.
        let payload = serde_json::json!({ "order_id": "abc-123" });
        let result = parse_envelope(&payload).unwrap();
        assert!(result.is_none(), "non-envelope payload should return None");
    }

    #[test]
    fn extract_customer_input_from_valid_envelope() {
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:test",
            "CheckpointToken": "tok",
            "InitialExecutionState": {
                "Operations": [{
                    "Id": "execution",
                    "Type": "EXECUTION",
                    "ExecutionDetails": {
                        "InputPayload": "\"hello\""
                    }
                }]
            }
        });
        let input = inline_input_payload(&payload).unwrap().into_payload();
        let result: Result<String, _> = extract_customer_input(input.as_deref());
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn extract_customer_input_defaults_to_empty_object() {
        // An envelope whose Execution operation has no InputPayload parses
        // as `{}`, as the JS SDK does.
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:test",
            "CheckpointToken": "tok",
            "InitialExecutionState": {
                "Operations": [{ "Id": "execution", "Type": "EXECUTION" }]
            }
        });
        let input = inline_input_payload(&payload).unwrap();
        assert_eq!(input, client::PageInput::Execution(None));
        let result: serde_json::Value =
            extract_customer_input(input.into_payload().as_deref()).unwrap();
        assert_eq!(result, serde_json::json!({}));
    }

    /// The `InputPayload` cases of `inline_input_payload`: no Execution
    /// operation, a `null`, a string, and a non-string value.
    #[test]
    fn inline_input_payload_cases() {
        let envelope = |ops: serde_json::Value| {
            serde_json::json!({
                "DurableExecutionArn": "arn:test",
                "CheckpointToken": "tok",
                "InitialExecutionState": { "Operations": ops }
            })
        };
        let step_only = envelope(serde_json::json!([{ "Id": "s1", "Type": "STEP" }]));
        assert_eq!(
            inline_input_payload(&step_only).unwrap(),
            client::PageInput::NoExecutionOperation
        );
        let no_operations = serde_json::json!({
            "DurableExecutionArn": "arn:test",
            "CheckpointToken": "tok",
            "InitialExecutionState": { "NextMarker": "m-1" }
        });
        assert_eq!(
            inline_input_payload(&no_operations).unwrap(),
            client::PageInput::NoExecutionOperation
        );
        // The first EXECUTION operation is read, wherever it sits.
        let later = envelope(serde_json::json!([
            { "Id": "s1", "Type": "STEP", "ExecutionDetails": { "InputPayload": "\"wrong\"" } },
            { "Id": "e", "Type": "EXECUTION", "ExecutionDetails": { "InputPayload": "\"right\"" } }
        ]));
        assert_eq!(
            inline_input_payload(&later).unwrap(),
            client::PageInput::Execution(Some("\"right\"".to_owned()))
        );
        let null = envelope(serde_json::json!([
            { "Id": "e", "Type": "EXECUTION", "ExecutionDetails": { "InputPayload": null } }
        ]));
        assert_eq!(
            inline_input_payload(&null).unwrap(),
            client::PageInput::Execution(None)
        );
        let number = envelope(serde_json::json!([
            { "Id": "e", "Type": "EXECUTION", "ExecutionDetails": { "InputPayload": 5 } }
        ]));
        let err = inline_input_payload(&number).unwrap_err();
        assert!(
            err.to_string().contains("InputPayload is not a string"),
            "got: {err}"
        );
    }

    #[test]
    fn extract_customer_input_parse_failure_errors() {
        let err = extract_customer_input::<serde_json::Value>(Some("not json")).unwrap_err();
        assert!(
            err.to_string().contains("deserialize customer input"),
            "got: {err}"
        );
    }

    // Bootstrap input cases, driven through `wrap`.

    /// A test transport that returns configured pages by marker and records
    /// each `get_state` call's marker.
    struct PagedTransport {
        pages: std::collections::HashMap<String, GetStateOutput>,
        markers: std::sync::Mutex<Vec<Option<String>>>,
    }

    impl PagedTransport {
        fn new(pages: Vec<(&str, Vec<aws_sdk_lambda::types::Operation>, Option<&str>)>) -> Self {
            Self {
                pages: pages
                    .into_iter()
                    .map(|(marker, ops, next)| {
                        (
                            marker.to_owned(),
                            GetStateOutput::new(ops, next.map(str::to_owned)),
                        )
                    })
                    .collect(),
                markers: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn markers(&self) -> Vec<Option<String>> {
            self.markers.lock().unwrap().clone()
        }
    }

    impl ExecutionClient for PagedTransport {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _updates: Vec<aws_sdk_lambda::types::OperationUpdate>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>,
        > {
            Box::pin(async {
                Ok(CheckpointOutput::new(
                    Some("token-next".to_owned()),
                    Vec::new(),
                    None,
                ))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
        > {
            self.markers.lock().unwrap().push(marker.map(str::to_owned));
            let page = marker.and_then(|m| self.pages.get(m)).map_or_else(
                || GetStateOutput::new(Vec::new(), None),
                |p| GetStateOutput::new(p.operations.clone(), p.next_marker().map(str::to_owned)),
            );
            Box::pin(async move { Ok(page) })
        }
    }

    /// Drives the `wrap`-produced echo service over `transport` and
    /// reports the response and whether the handler ran.
    async fn invoke_echo_over(
        payload: serde_json::Value,
        transport: std::sync::Arc<dyn ExecutionClient>,
    ) -> (
        Result<serde_json::Value, lambda_runtime::Error>,
        std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) {
        let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ran_in_handler = std::sync::Arc::clone(&ran);
        let options = Options::builder()
            .execution_client(transport)
            .build()
            .expect("options build");
        let service = wrap(
            move |event: serde_json::Value, _ctx: DurableContext| {
                let ran = std::sync::Arc::clone(&ran_in_handler);
                async move {
                    ran.store(true, std::sync::atomic::Ordering::SeqCst);
                    Ok::<serde_json::Value, BoxError>(event)
                }
            },
            options,
        );
        let event = lambda_runtime::LambdaEvent::new(
            InvocationPayload(payload),
            lambda_runtime::Context::default(),
        );
        (service(event).await.map(|response| response.0), ran)
    }

    fn bootstrap_envelope(state: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:test",
            "CheckpointToken": "token-abc",
            "InitialExecutionState": state
        })
    }

    fn assert_succeeded_with(response: &serde_json::Value, expected: &serde_json::Value) {
        assert_eq!(
            response.get("Status").and_then(serde_json::Value::as_str),
            Some("SUCCEEDED"),
            "unexpected response: {response}"
        );
        let result: serde_json::Value = serde_json::from_str(
            response
                .get("Result")
                .and_then(serde_json::Value::as_str)
                .expect("a Result string"),
        )
        .expect("the Result parses");
        assert_eq!(&result, expected, "unexpected response: {response}");
    }

    /// Rows 1 and 5: an absent or `null` `InputPayload` runs the handler
    /// on `{}`.
    #[tokio::test]
    async fn absent_input_payload_runs_on_empty_object() {
        for details in [
            serde_json::json!({}),
            serde_json::json!({ "InputPayload": null }),
        ] {
            let payload = bootstrap_envelope(&serde_json::json!({
                "Operations": [{
                    "Id": "execution",
                    "Type": "EXECUTION",
                    "Status": "STARTED",
                    "ExecutionDetails": details
                }]
            }));
            let transport = std::sync::Arc::new(PagedTransport::new(Vec::new()));
            let (response, _) =
                invoke_echo_over(payload, std::sync::Arc::clone(&transport) as _).await;
            assert_succeeded_with(&response.expect("must succeed"), &serde_json::json!({}));
            assert!(transport.markers().is_empty(), "no marker, no get_state");
        }
    }

    /// Row 2: the input is read from a later page.
    #[tokio::test]
    async fn input_payload_found_after_pagination() {
        let payload = bootstrap_envelope(&serde_json::json!({ "NextMarker": "m-1" }));
        let transport = std::sync::Arc::new(PagedTransport::new(vec![(
            "m-1",
            vec![make_test_execution_op("{\"count\":1}")],
            None,
        )]));
        let (response, _) = invoke_echo_over(payload, std::sync::Arc::clone(&transport) as _).await;
        assert_succeeded_with(
            &response.expect("must succeed"),
            &serde_json::json!({ "count": 1 }),
        );
        assert_eq!(transport.markers(), vec![Some("m-1".to_owned())]);
    }

    /// Row 3: an inline input is kept when later pages are fetched.
    #[tokio::test]
    async fn paginated_input_matches_first_page() {
        let payload = bootstrap_envelope(&serde_json::json!({
            "Operations": [{
                "Id": "execution",
                "Type": "EXECUTION",
                "Status": "STARTED",
                "ExecutionDetails": { "InputPayload": "{\"a\":1}" }
            }],
            "NextMarker": "m-1"
        }));
        let transport = std::sync::Arc::new(PagedTransport::new(vec![(
            "m-1",
            vec![make_test_step_op("s1", "\"r1\"")],
            None,
        )]));
        let (response, _) = invoke_echo_over(payload, std::sync::Arc::clone(&transport) as _).await;
        assert_succeeded_with(
            &response.expect("must succeed"),
            &serde_json::json!({ "a": 1 }),
        );
        assert_eq!(transport.markers(), vec![Some("m-1".to_owned())]);
    }

    /// Rows 4 and 6: an input that does not parse, and a non-string input,
    /// fail the invocation before the handler runs.
    #[tokio::test]
    async fn malformed_input_payload_fails_the_invocation() {
        for (input, expected) in [
            (serde_json::json!("not json"), "deserialize customer input"),
            (serde_json::json!(5), "InputPayload is not a string"),
        ] {
            let payload = bootstrap_envelope(&serde_json::json!({
                "Operations": [{
                    "Id": "execution",
                    "Type": "EXECUTION",
                    "Status": "STARTED",
                    "ExecutionDetails": { "InputPayload": input }
                }]
            }));
            let transport = std::sync::Arc::new(PagedTransport::new(Vec::new()));
            let (response, ran) =
                invoke_echo_over(payload, std::sync::Arc::clone(&transport) as _).await;
            let err = response.expect_err("must fail");
            assert!(err.to_string().contains(expected), "got: {err}");
            assert!(
                !ran.load(std::sync::atomic::Ordering::SeqCst),
                "the handler must not run"
            );
            assert!(transport.markers().is_empty(), "no marker, no get_state");
        }
    }

    /// Row 1's envelope without a `CheckpointToken` fails in
    /// `parse_envelope`, before the input defaults to `{}`.
    #[tokio::test]
    async fn missing_checkpoint_token_fails_before_input_default() {
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:test",
            "InitialExecutionState": {
                "Operations": [{
                    "Id": "execution",
                    "Type": "EXECUTION",
                    "Status": "STARTED",
                    "ExecutionDetails": {}
                }]
            }
        });
        let transport = std::sync::Arc::new(PagedTransport::new(Vec::new()));
        let (response, ran) = invoke_echo_over(payload, transport).await;
        let err = response.expect_err("must fail");
        assert!(err.to_string().contains("CheckpointToken"), "got: {err}");
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "the handler must not run"
        );
    }

    // Default client timeouts

    #[test]
    fn default_client_sets_connect_and_read_timeouts() {
        let config = SdkConfig::builder()
            .behavior_version(aws_config::BehaviorVersion::latest())
            .build();
        let client = default_lambda_client(&config);
        let timeouts = client
            .config()
            .timeout_config()
            .expect("the default client has a timeout config");
        assert_eq!(timeouts.connect_timeout(), Some(Duration::from_secs(5)));
        assert_eq!(timeouts.read_timeout(), Some(Duration::from_secs(50)));
        assert_eq!(
            timeouts.operation_attempt_timeout(),
            Some(Duration::from_secs(55))
        );
    }

    #[test]
    fn supplied_config_keeps_its_timeouts() {
        use aws_sdk_lambda::config::timeout::TimeoutConfig;
        let config = SdkConfig::builder()
            .behavior_version(aws_config::BehaviorVersion::latest())
            .timeout_config(
                TimeoutConfig::builder()
                    .connect_timeout(Duration::from_secs(1))
                    .build(),
            )
            .build();
        let client = base_lambda_client_from_options(Some(config), None)
            .expect("sdk_config must yield a client");
        let timeouts = client.config().timeout_config().expect("a timeout config");
        assert_eq!(timeouts.connect_timeout(), Some(Duration::from_secs(1)));
        assert_eq!(timeouts.read_timeout(), None);

        let conf = aws_sdk_lambda::config::Config::builder()
            .behavior_version(aws_sdk_lambda::config::BehaviorVersion::latest())
            .timeout_config(
                TimeoutConfig::builder()
                    .read_timeout(Duration::from_secs(7))
                    .build(),
            )
            .build();
        let supplied = aws_sdk_lambda::Client::from_conf(conf);
        let client = base_lambda_client_from_options(None, Some(supplied))
            .expect("lambda_client must be returned");
        let timeouts = client.config().timeout_config().expect("a timeout config");
        assert_eq!(timeouts.read_timeout(), Some(Duration::from_secs(7)));
        assert_eq!(timeouts.connect_timeout(), None);
        assert_eq!(timeouts.operation_attempt_timeout(), None);
    }
}
