//! Tracing integration: replay-aware filter layer, span helpers, and
//! lifecycle-event emitters.
//!
//! This module implements the instrumentation whose public, documented
//! contract lives in [`crate::observability`] — span names, lifecycle event
//! names, and field names are all specified there.
//!
//! The SDK creates two kinds of [`tracing::Span`]:
//!
//! - A `durable_execution` span per NAMESPACE. The handler-level span wraps
//!   each invocation of the user's handler; each child namespace (a
//!   `run_in_child_context` body, a map/parallel branch, a
//!   `wait_for_callback` body) gets its own detached span wrapping its body.
//!   Every one carries the execution ARN, the request ID, and an `isReplay`
//!   flag that the SDK keeps current as that namespace crosses its own
//!   replay high-water mark: the flag starts as the namespace's initial
//!   replay status and is re-recorded after every operation claim minted
//!   through it.
//! - A per-operation `durable_operation` span wrapping each live step body,
//!   carrying the full structured-log field contract below.
//!
//! When these spans are rendered by a JSON subscriber (e.g.,
//! `lambda_runtime`'s `init_default_subscriber()` with
//! `AWS_LAMBDA_LOG_FORMAT=JSON`), the fields appear as top-level JSON keys —
//! matching the `CloudWatch` Logs Insights query:
//!
//! ```text
//! filter coalesce(durableExecutionArn, executionArn) like "<arn>"
//! ```
//!
//! # Structured-log field contract
//!
//! | Field | Description |
//! |-------|-------------|
//! | `executionArn` | Durable execution ARN |
//! | `requestId` | Lambda invocation request ID |
//! | `operationId` | Wire operation ID (SHA-256 hex) |
//! | `attempt` | Current attempt number (1-based) |
//! | `isReplay` | Whether the span covers replayed work |
//!
//! # Replay filter
//!
//! When an execution resumes, the handler re-runs from the top and the SDK
//! replays recorded results. Handler code between operations executes again,
//! so its log statements would re-emit on every resume. The `isReplay` flag
//! on the handler span lets a per-layer filter suppress those events,
//! avoiding duplicate `CloudWatch` log lines. The SDK does not install a
//! subscriber automatically.
//!
//! Enable the **`replay-filter`** feature to get [`ReplayFilterLayer`], a
//! ready-made per-layer filter that implements this suppression. Install it
//! on a `tracing-subscriber` layer like any other filter:
//!
//! ```ignore
//! use aws_durable_execution_sdk_rust::ReplayFilterLayer;
//! use tracing_subscriber::Layer as _;
//! use tracing_subscriber::layer::SubscriberExt;
//! use tracing_subscriber::util::SubscriberInitExt;
//!
//! tracing_subscriber::registry()
//!     .with(tracing_subscriber::fmt::layer().with_filter(ReplayFilterLayer))
//!     .init();
//! ```
//!
//! Without the feature, applications can build their own filter by walking
//! the span scope and checking the `isReplay` field directly.

#[cfg(any(test, feature = "replay-filter"))]
use tracing::Id;
#[cfg(any(test, feature = "replay-filter"))]
use tracing::span::Attributes;

/// Field name constants for the structured-log contract.
///
/// The canonical, documented definitions live in
/// [`crate::observability::field_names`]; this alias keeps the internal
/// spelling short.
pub(crate) use crate::observability::field_names as fields;

/// Creates a `tracing::Span` for a durable operation with the structured-log
/// field contract.
///
/// The span carries:
/// - `executionArn`: the durable execution ARN
/// - `requestId`: the Lambda request ID
/// - `operationId`: the wire operation ID (SHA-256 hex)
/// - `attempt`: the current attempt (1-based)
/// - `isReplay`: whether this operation is replayed
///
/// User `tracing::info!` calls inside step bodies inherit these fields
/// automatically through span nesting.
#[must_use]
pub(crate) fn operation_span(
    execution_arn: &str,
    request_id: &str,
    operation_id: &str,
    attempt: u32,
    is_replay: bool,
) -> tracing::Span {
    tracing::info_span!(
        "durable_operation",
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        { fields::OPERATION_ID } = operation_id,
        { fields::ATTEMPT } = attempt,
        { fields::IS_REPLAY } = is_replay,
    )
}

/// Creates the handler-level `tracing::Span` wrapping one invocation of the
/// user's handler.
///
/// The span carries:
/// - `executionArn`: the durable execution ARN
/// - `requestId`: the Lambda request ID
/// - `isReplay`: whether the execution is currently replaying
///
/// Handler-level `tracing::info!` calls (user code between operations)
/// inherit these fields automatically through span nesting, which is what
/// lets [`ReplayFilterLayer`] suppress them during replay. The `isReplay`
/// field is dynamic: [`DurableContext::mint_id`] re-records it after every
/// operation claim, so it flips to `false` the moment the invocation crosses
/// the replay high-water mark.
///
/// [`DurableContext::mint_id`]: crate::DurableContext
#[must_use]
pub(crate) fn execution_span(
    execution_arn: &str,
    request_id: &str,
    is_replay: bool,
) -> tracing::Span {
    tracing::info_span!(
        "durable_execution",
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        { fields::IS_REPLAY } = is_replay,
    )
}

/// Creates a DETACHED `durable_execution` span for a child namespace — a
/// `run_in_child_context` body, a map/parallel branch, or a
/// `wait_for_callback` body.
///
/// Each child context owns an [`crate::engine::EngineState`] with its own ID
/// counter and therefore its own replay high-water mark: nested operations
/// can still be replaying while the parent namespace is already live (or
/// vice versa). Giving each namespace its own span — whose `isReplay` flag
/// that namespace's mints keep current — is what lets a per-layer filter
/// suppress a branch's pre-wait log lines on resume without consulting (or
/// clobbering) the parent's state.
///
/// The span is created with `parent: None` so an event inside the child
/// scope resolves its replay status against the child namespace alone: a
/// parent span that is still replaying must not suppress a branch that has
/// gone live, and a live parent must not un-suppress a branch that is still
/// replaying.
#[must_use]
pub(crate) fn scoped_execution_span(
    execution_arn: &str,
    request_id: &str,
    is_replay: bool,
) -> tracing::Span {
    tracing::info_span!(
        parent: None,
        "durable_execution",
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        { fields::IS_REPLAY } = is_replay,
    )
}

// ────────────────────────────────────────────────────────────────────────────
// Lifecycle events (see `crate::observability` for the documented contract)
// ────────────────────────────────────────────────────────────────────────────

use crate::observability::{TARGET, event_names};

/// Identity fields shared by every operation lifecycle event.
///
/// Groups the per-operation fields of the documented contract so each
/// emitter takes one argument instead of seven.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OperationIdentity<'a> {
    /// Durable execution ARN.
    pub(crate) execution_arn: &'a str,
    /// Lambda invocation request ID.
    pub(crate) request_id: &'a str,
    /// Wire operation ID (SHA-256 hex digest).
    pub(crate) operation_id: &'a str,
    /// User-supplied `.name("...")`, when present.
    pub(crate) operation_name: Option<&'a str>,
    /// Operation type (`Step`, `Wait`, `Context`, `Callback`,
    /// `ChainedInvoke`).
    pub(crate) operation_type: &'a str,
    /// Wire sub-type, when present.
    pub(crate) operation_sub_type: Option<&'a str>,
    /// 1-based attempt number the event describes.
    pub(crate) attempt: u32,
}

/// Maps the wire [`OperationType`] to the contract's `operationType` value.
///
/// The contract uses the same `PascalCase` spellings the SDK's replay-identity
/// checks use (`Step`, not the wire enum's `STEP`).
fn operation_type_str(op_type: &aws_sdk_lambda::types::OperationType) -> &'static str {
    use aws_sdk_lambda::types::OperationType;
    match *op_type {
        OperationType::Step => "Step",
        OperationType::Wait => "Wait",
        OperationType::Context => "Context",
        OperationType::Callback => "Callback",
        OperationType::ChainedInvoke => "ChainedInvoke",
        OperationType::Execution => "Execution",
        _ => "Unknown",
    }
}

/// A record-transition lifecycle event captured from one checkpoint update
/// before the write, to be emitted only after the checkpoint that persists
/// the transition succeeds.
///
/// The checkpoint path is the single chokepoint every operation type's
/// `Start`/`Succeed`/`Fail`/`Retry` transition passes through, so
/// record-transition events cover steps, waits, invokes, callbacks,
/// `wait_for_condition`, child contexts, and map/parallel batches uniformly.
/// The metadata is captured eagerly (the checkpoint response mutates the
/// log the attempt number is derived from) and owned (the update itself is
/// consumed by the write), but nothing is emitted until the write that
/// persists the transition succeeds: a rejected checkpoint records no
/// transition, so it must produce no telemetry claiming one was recorded.
///
/// The event travels **with its update** into the write path (see
/// `crate::checkpoint_coalescer::TrackedUpdate`): when checkpoint buffering
/// is configured, the flush task — not the contributor future — owns and
/// emits it, so a contributor dropped after joining a batch (a lost `race`
/// or `select_ok` branch) cannot suppress the telemetry for a transition
/// the flush still persists. The capture also snapshots
/// [`tracing::Span::current()`], and emission re-enters that span, so the
/// event keeps the originating operation's span context even when emitted
/// from the detached flush task.
#[derive(Debug)]
pub(crate) struct PendingTransitionEvent {
    /// Wire operation ID.
    operation_id: String,
    /// User-supplied operation name, when present.
    operation_name: Option<String>,
    /// Contract spelling of the operation type.
    operation_type: &'static str,
    /// Wire sub-type, when present.
    operation_sub_type: Option<String>,
    /// 1-based attempt the transition describes.
    attempt: u32,
    /// The transition being recorded.
    action: aws_sdk_lambda::types::OperationAction,
    /// Retry delay (`operation_retry_scheduled` only).
    delay_seconds: i32,
    /// Error message (`operation_failed` / `operation_retry_scheduled`).
    error: String,
    /// Span current when the transition was captured; emission re-enters
    /// it so the event stays parented to the originating operation even
    /// when emitted from the detached batch-flush task.
    span: tracing::Span,
}

impl PendingTransitionEvent {
    /// Captures the event metadata from an update about to be checkpointed.
    pub(crate) fn capture(update: &aws_sdk_lambda::types::OperationUpdate, attempt: u32) -> Self {
        Self {
            span: tracing::Span::current(),
            operation_id: update.id().to_owned(),
            operation_name: update.name().map(str::to_owned),
            operation_type: operation_type_str(update.r#type()),
            operation_sub_type: update.sub_type().map(str::to_owned),
            attempt,
            action: update.action().clone(),
            delay_seconds: update
                .step_options()
                .and_then(aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds)
                .unwrap_or_default(),
            error: update
                .error()
                .and_then(|e| e.error_message())
                .unwrap_or_default()
                .to_owned(),
        }
    }

    /// Emits the captured lifecycle event. Call only after the checkpoint
    /// write that persists this transition has succeeded.
    ///
    /// The event is emitted inside the span captured at
    /// [`Self::capture`] time, preserving the originating operation's
    /// span context even when this runs on the detached batch-flush task.
    pub(crate) fn emit(&self, execution_arn: &str, request_id: &str) {
        use aws_sdk_lambda::types::OperationAction;

        let op = OperationIdentity {
            execution_arn,
            request_id,
            operation_id: &self.operation_id,
            operation_name: self.operation_name.as_deref(),
            operation_type: self.operation_type,
            operation_sub_type: self.operation_sub_type.as_deref(),
            attempt: self.attempt,
        };
        self.span.in_scope(|| match self.action {
            OperationAction::Start => operation_started_event(&op),
            OperationAction::Succeed => operation_succeeded_event(&op),
            OperationAction::Fail => operation_failed_event(&op, &self.error),
            OperationAction::Retry => {
                operation_retry_scheduled_event(&op, self.delay_seconds, &self.error);
            }
            _ => {}
        });
    }
}

/// Emits [`event_names::OPERATION_STARTED`].
fn operation_started_event(op: &OperationIdentity<'_>) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = op.execution_arn,
        { fields::REQUEST_ID } = op.request_id,
        { fields::OPERATION_ID } = op.operation_id,
        { fields::OPERATION_NAME } = op.operation_name,
        { fields::OPERATION_TYPE } = op.operation_type,
        { fields::OPERATION_SUB_TYPE } = op.operation_sub_type,
        { fields::ATTEMPT } = op.attempt,
        { fields::IS_REPLAY } = false,
        "{}",
        event_names::OPERATION_STARTED,
    );
}

/// Emits [`event_names::OPERATION_SUCCEEDED`].
fn operation_succeeded_event(op: &OperationIdentity<'_>) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = op.execution_arn,
        { fields::REQUEST_ID } = op.request_id,
        { fields::OPERATION_ID } = op.operation_id,
        { fields::OPERATION_NAME } = op.operation_name,
        { fields::OPERATION_TYPE } = op.operation_type,
        { fields::OPERATION_SUB_TYPE } = op.operation_sub_type,
        { fields::ATTEMPT } = op.attempt,
        { fields::IS_REPLAY } = false,
        "{}",
        event_names::OPERATION_SUCCEEDED,
    );
}

/// Emits [`event_names::OPERATION_FAILED`].
fn operation_failed_event(op: &OperationIdentity<'_>, error: &str) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = op.execution_arn,
        { fields::REQUEST_ID } = op.request_id,
        { fields::OPERATION_ID } = op.operation_id,
        { fields::OPERATION_NAME } = op.operation_name,
        { fields::OPERATION_TYPE } = op.operation_type,
        { fields::OPERATION_SUB_TYPE } = op.operation_sub_type,
        { fields::ATTEMPT } = op.attempt,
        { fields::IS_REPLAY } = false,
        { fields::ERROR } = error,
        "{}",
        event_names::OPERATION_FAILED,
    );
}

/// Emits [`event_names::OPERATION_RETRY_SCHEDULED`].
fn operation_retry_scheduled_event(op: &OperationIdentity<'_>, delay_seconds: i32, error: &str) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = op.execution_arn,
        { fields::REQUEST_ID } = op.request_id,
        { fields::OPERATION_ID } = op.operation_id,
        { fields::OPERATION_NAME } = op.operation_name,
        { fields::OPERATION_TYPE } = op.operation_type,
        { fields::OPERATION_SUB_TYPE } = op.operation_sub_type,
        { fields::ATTEMPT } = op.attempt,
        { fields::IS_REPLAY } = false,
        { fields::DELAY_SECONDS } = delay_seconds,
        { fields::ERROR } = error,
        "{}",
        event_names::OPERATION_RETRY_SCHEDULED,
    );
}

/// Emits [`event_names::OPERATION_REPLAYED`] — a recorded terminal outcome
/// was returned without re-running the operation.
pub(crate) fn operation_replayed_event(op: &OperationIdentity<'_>) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = op.execution_arn,
        { fields::REQUEST_ID } = op.request_id,
        { fields::OPERATION_ID } = op.operation_id,
        { fields::OPERATION_NAME } = op.operation_name,
        { fields::OPERATION_TYPE } = op.operation_type,
        { fields::OPERATION_SUB_TYPE } = op.operation_sub_type,
        { fields::ATTEMPT } = op.attempt,
        { fields::IS_REPLAY } = true,
        "{}",
        event_names::OPERATION_REPLAYED,
    );
}

/// Emits the invocation-begin lifecycle event: exactly one of
/// [`event_names::EXECUTION_RESUMED`] (the invocation begins with recorded
/// operations to replay) or [`event_names::EXECUTION_STARTED`].
pub(crate) fn invocation_begin_event(is_replaying: bool, execution_arn: &str, request_id: &str) {
    if is_replaying {
        execution_resumed_event(execution_arn, request_id);
    } else {
        execution_started_event(execution_arn, request_id);
    }
}

/// Emits [`event_names::EXECUTION_STARTED`].
pub(crate) fn execution_started_event(execution_arn: &str, request_id: &str) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        "{}",
        event_names::EXECUTION_STARTED,
    );
}

/// Emits [`event_names::EXECUTION_RESUMED`].
pub(crate) fn execution_resumed_event(execution_arn: &str, request_id: &str) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        "{}",
        event_names::EXECUTION_RESUMED,
    );
}

/// Emits [`event_names::EXECUTION_SUSPENDED`].
pub(crate) fn execution_suspended_event(execution_arn: &str, request_id: &str) {
    tracing::event!(
        target: TARGET,
        tracing::Level::DEBUG,
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        "{}",
        event_names::EXECUTION_SUSPENDED,
    );
}

/// Visitor that extracts the `isReplay` boolean field from recorded values.
///
/// Shared by [`is_replay_event`] (span attributes at creation),
/// [`replay_flag_in_record`] (`span.record()` updates after creation), and
/// [`ReplayFilterLayer`]'s `event_enabled` (an event's own fields).
#[cfg(any(test, feature = "replay-filter"))]
struct ReplayVisitor {
    is_replay: Option<bool>,
}

#[cfg(any(test, feature = "replay-filter"))]
impl tracing::field::Visit for ReplayVisitor {
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        if field.name() == fields::IS_REPLAY {
            self.is_replay = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

/// Returns `true` if the given span attributes include `isReplay = true`.
///
/// Used by [`ReplayFilterLayer`] to detect replay spans at creation time and
/// by the test-only [`ReplayTracker`] for verification.
#[cfg(any(test, feature = "replay-filter"))]
pub(crate) fn is_replay_event(attrs: &Attributes<'_>) -> bool {
    let mut visitor = ReplayVisitor { is_replay: None };
    attrs.record(&mut visitor);
    visitor.is_replay == Some(true)
}

/// Extracts the `isReplay` value from a `span.record()` update, if present.
///
/// Used by [`ReplayFilterLayer`] to track the handler span's dynamic replay
/// flag, which the SDK re-records after every operation claim.
#[cfg(any(test, feature = "replay-filter"))]
fn replay_flag_in_record(values: &tracing::span::Record<'_>) -> Option<bool> {
    let mut visitor = ReplayVisitor { is_replay: None };
    values.record(&mut visitor);
    visitor.is_replay
}

// ────────────────────────────────────────────────────────────────────────────
// Replay-aware subscriber wrapper (for use in test infrastructure)
// ────────────────────────────────────────────────────────────────────────────

/// A subscriber wrapper that tracks which spans have `isReplay = true` and
/// provides a method to check if events should be suppressed.
///
/// This is used in the SDK's test infrastructure to verify replay-filter
/// behavior. Production handlers use `tracing-subscriber` layers directly.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ReplayTracker {
    /// Set of span IDs that are marked as replay.
    replay_spans: std::sync::RwLock<std::collections::HashSet<u64>>,
}

#[cfg(test)]
impl ReplayTracker {
    /// Creates a new replay tracker.
    pub(crate) fn new() -> Self {
        Self {
            replay_spans: std::sync::RwLock::new(std::collections::HashSet::new()),
        }
    }

    /// Returns whether the span (or any ancestor) is in replay mode.
    ///
    /// For a full implementation this would walk the span stack; for
    /// unit tests we check if the given span ID is directly marked.
    pub(crate) fn is_replay(&self, id: u64) -> bool {
        self.replay_spans.read().is_ok_and(|set| set.contains(&id))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Replay filter layer (public under the `replay-filter` feature;
// `tracing-subscriber` is an optional dependency)
// ────────────────────────────────────────────────────────────────────────────

/// A per-layer filter that suppresses events describing replayed work:
/// events carrying their own `isReplay = true` field (the SDK's lifecycle
/// events — see [`crate::observability`]) and events emitted inside spans
/// marked `isReplay = true` (application log lines during replay).
///
/// When an execution resumes, the handler re-runs from the top and replays
/// recorded results, so handler code between operations executes — and logs —
/// again. The SDK wraps each invocation in a span whose `isReplay` flag
/// tracks the live replay status; this filter drops events while that flag
/// is `true`, so a log line is written once, on the invocation that first
/// executed it, not again on every resume.
///
/// For events that carry an explicit `isReplay` field, that field is
/// authoritative and the span scope is not consulted: the execution span's
/// flag tracks the *next* operation claim, so at the replay high-water
/// boundary the final `operation_replayed` lifecycle event can fire under
/// a span already flipped to live. Inspecting the event's own field keeps
/// suppression exact at that boundary.
///
/// # Usage
///
/// Enable the **`replay-filter`** feature:
///
/// ```toml
/// [dependencies]
/// aws-durable-execution-sdk-rust = { git = "https://github.com/aws/aws-durable-execution-sdk-rust", branch = "alpha", features = ["replay-filter"] }
/// tracing-subscriber = { version = "0.3", features = ["registry"] }
/// ```
///
/// Then install the filter on a subscriber layer:
///
/// ```
/// use aws_durable_execution_sdk_rust::ReplayFilterLayer;
/// use tracing_subscriber::Layer as _;
/// use tracing_subscriber::layer::SubscriberExt;
///
/// let subscriber = tracing_subscriber::registry().with(
///     tracing_subscriber::fmt::layer()
///         .json()
///         .with_filter(ReplayFilterLayer),
/// );
///
/// // In a Lambda binary, install it globally instead:
/// // `tracing_subscriber::util::SubscriberInitExt::init(subscriber)`.
/// tracing::subscriber::with_default(subscriber, || {
///     tracing::info!("emitted normally — not inside a replay span");
/// });
/// ```
#[cfg(any(test, feature = "replay-filter"))]
#[derive(Debug, Clone)]
pub struct ReplayFilterLayer;

#[cfg(any(test, feature = "replay-filter"))]
impl<S> tracing_subscriber::layer::Filter<S> for ReplayFilterLayer
where
    S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
{
    fn enabled(
        &self,
        _meta: &tracing::Metadata<'_>,
        _ctx: &tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        // Always allow span creation — we only filter events.
        true
    }

    fn event_enabled(
        &self,
        event: &tracing::Event<'_>,
        ctx: &tracing_subscriber::layer::Context<'_, S>,
    ) -> bool {
        // An event carrying its own `isReplay` field (every SDK lifecycle
        // event does — see `crate::observability`) is authoritative: the
        // execution span's flag tracks the NEXT operation claim, so at the
        // replay high-water boundary the final `operation_replayed` event
        // can fire under a span already flipped to `isReplay = false`.
        // Trusting the event's field suppresses exactly the events that
        // describe replayed work, wherever they fire.
        let mut visitor = ReplayVisitor { is_replay: None };
        event.record(&mut visitor);
        if let Some(is_replay) = visitor.is_replay {
            return !is_replay;
        }

        // Otherwise (application log lines carry no `isReplay` field), walk
        // the span scope to check if any ancestor has isReplay=true.
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope {
                let extensions = span.extensions();
                if let Some(fields) = extensions.get::<ReplaySpanFields>()
                    && fields.is_replay
                {
                    return false; // Suppress event.
                }
            }
        }
        true
    }

    fn on_new_span(
        &self,
        attrs: &Attributes<'_>,
        id: &Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // Record isReplay field when a new span is created.
        if let Some(span) = ctx.span(id)
            && is_replay_event(attrs)
        {
            span.extensions_mut()
                .insert(ReplaySpanFields { is_replay: true });
        }
    }

    fn on_record(
        &self,
        id: &Id,
        values: &tracing::span::Record<'_>,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        // The SDK re-records the handler span's isReplay flag after every
        // operation claim (see `execution_span`). Track those updates so
        // suppression follows the live replay status, not just the value
        // the span was created with.
        if let Some(is_replay) = replay_flag_in_record(values)
            && let Some(span) = ctx.span(id)
        {
            let mut extensions = span.extensions_mut();
            if let Some(fields) = extensions.get_mut::<ReplaySpanFields>() {
                fields.is_replay = is_replay;
            } else {
                extensions.insert(ReplaySpanFields { is_replay });
            }
        }
    }
}

/// Storage for the replay flag in span extensions.
#[cfg(any(test, feature = "replay-filter"))]
#[derive(Debug)]
struct ReplaySpanFields {
    is_replay: bool,
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operation_span_has_correct_fields() {
        // Install a no-op subscriber so spans are not disabled.
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish());

        let span = operation_span(
            "arn:aws:lambda:us-east-1:123:function:test:durable:abc",
            "req-123",
            "deadbeef01234567890abcdef0123456789abcdef0123456789abcdef012345",
            1,
            false,
        );
        // The span should be valid (non-disabled) when a subscriber exists.
        assert!(!span.is_disabled());
    }

    #[test]
    fn replay_tracker_marks_replay_spans() {
        let tracker = ReplayTracker::new();
        // Non-existent span should not be marked as replay.
        assert!(!tracker.is_replay(999));
    }

    #[test]
    fn field_constants_match_cross_sdk_contract() {
        // Verify field name constants match the structured-log contract
        // and the CloudWatch Logs Insights query.
        assert_eq!(fields::EXECUTION_ARN, "executionArn");
        assert_eq!(fields::REQUEST_ID, "requestId");
        assert_eq!(fields::OPERATION_ID, "operationId");
        assert_eq!(fields::ATTEMPT, "attempt");
        assert_eq!(fields::IS_REPLAY, "isReplay");
    }

    /// Verifies that a JSON subscriber produces log lines containing the
    /// structured-log field contract fields when events fire inside an operation
    /// span.
    #[test]
    fn json_output_contains_cross_sdk_fields() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::fmt::MakeWriter;

        /// A writer that captures output in a shared buffer.
        #[derive(Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Ok(mut inner) = self.0.lock() {
                    inner.extend_from_slice(buf);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> MakeWriter<'a> for BufWriter {
            type Writer = BufWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = BufWriter(Arc::clone(&buffer));

        let subscriber = tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_writer(writer)
            .with_span_list(false)
            .finish();

        let _guard = tracing::subscriber::set_default(subscriber);

        let test_arn = "arn:aws:lambda:us-east-1:123456789012:function:myFunc:durable:exec-123";
        let test_request_id = "req-abc-def-456";
        let test_op_id = "aabbccdd00112233445566778899aabbccddeeff00112233445566778899aabb";

        let span = operation_span(test_arn, test_request_id, test_op_id, 2, false);
        let _entered = span.enter();
        tracing::info!("Greeting step started for: TestUser");

        // Parse the captured JSON output.
        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );

        // The output should contain the structured fields.
        assert!(
            output.contains(test_arn),
            "JSON output must contain executionArn. Got: {output}"
        );
        assert!(
            output.contains(test_request_id),
            "JSON output must contain requestId. Got: {output}"
        );
        assert!(
            output.contains(test_op_id),
            "JSON output must contain operationId. Got: {output}"
        );
        assert!(
            output.contains("\"attempt\":2"),
            "JSON output must contain attempt. Got: {output}"
        );
        assert!(
            output.contains("\"isReplay\":false"),
            "JSON output must contain isReplay. Got: {output}"
        );
        assert!(
            output.contains("Greeting step started for: TestUser"),
            "JSON output must contain the user message. Got: {output}"
        );
    }

    /// Verifies that events inside a replay span are emitted with
    /// `isReplay = true` — enabling downstream filters to suppress them.
    #[test]
    fn replay_span_marks_events_with_is_replay_true() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::fmt::MakeWriter;

        #[derive(Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Ok(mut inner) = self.0.lock() {
                    inner.extend_from_slice(buf);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> MakeWriter<'a> for BufWriter {
            type Writer = BufWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = BufWriter(Arc::clone(&buffer));

        let subscriber = tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_writer(writer)
            .with_span_list(false)
            .finish();

        let _guard = tracing::subscriber::set_default(subscriber);

        // Create a replay span (isReplay = true).
        let span = operation_span("arn:test", "req-1", "op-1", 1, true);
        let _entered = span.enter();
        tracing::info!("this event is inside a replay span");

        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );

        assert!(
            output.contains("\"isReplay\":true"),
            "Replay span must emit isReplay=true. Got: {output}"
        );
    }

    /// Verifies that the replay filter (implemented as a
    /// `tracing_subscriber` layer) suppresses events in replay spans.
    #[test]
    fn replay_filter_suppresses_replay_events() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::fmt::MakeWriter;
        use tracing_subscriber::layer::SubscriberExt;

        #[derive(Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Ok(mut inner) = self.0.lock() {
                    inner.extend_from_slice(buf);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> MakeWriter<'a> for BufWriter {
            type Writer = BufWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = BufWriter(Arc::clone(&buffer));

        // Build a subscriber with a replay-aware filter layer that checks
        // span fields for isReplay=true and suppresses events.
        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_span_list(false)
            .with_writer(writer)
            .with_filter(ReplayFilterLayer);

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        // Event in a live span — should pass through.
        {
            let span = operation_span("arn:test", "req-1", "op-1", 1, false);
            let _entered = span.enter();
            tracing::info!("live event");
        }

        // Event in a replay span — should be suppressed.
        {
            let span = operation_span("arn:test", "req-1", "op-2", 1, true);
            let _entered = span.enter();
            tracing::info!("replay event");
        }

        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );

        assert!(
            output.contains("live event"),
            "Live events must pass through the replay filter. Got: {output}"
        );
        assert!(
            !output.contains("replay event"),
            "Replay events must be suppressed. Got: {output}"
        );
    }

    /// REGRESSION (replay high-water boundary): the execution span's
    /// `isReplay` flag tracks the NEXT operation claim (`mint_id` re-records
    /// it before the claimed operation resolves), so the final replay hit
    /// fires its `operation_replayed` event — which carries `isReplay =
    /// true` — under a span already flipped to `isReplay = false`. The
    /// filter must trust the event's own field, not just the span scope,
    /// so that event is still suppressed; and conversely an event carrying
    /// an explicit `isReplay = false` must pass even under a span still
    /// marked as replaying.
    #[test]
    fn replay_filter_trusts_event_own_is_replay_field_at_boundary() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::fmt::MakeWriter;
        use tracing_subscriber::layer::SubscriberExt;

        #[derive(Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Ok(mut inner) = self.0.lock() {
                    inner.extend_from_slice(buf);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> MakeWriter<'a> for BufWriter {
            type Writer = BufWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = BufWriter(Arc::clone(&buffer));

        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_span_list(false)
            .with_writer(writer)
            .with_filter(ReplayFilterLayer);

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        // The boundary: the span starts replaying, then the next mint flips
        // it to live, and only afterwards does the final replay hit emit its
        // event (carrying isReplay = true).
        {
            let span = execution_span("arn:test", "req-1", true);
            span.record(fields::IS_REPLAY, false); // mint_id: next op is live
            let _entered = span.enter();
            tracing::info!({ fields::IS_REPLAY } = true, "boundary replay hit");
        }

        // The converse: a live-work event under a span still marked as
        // replaying must pass — its own field is authoritative.
        {
            let span = execution_span("arn:test", "req-1", true);
            let _entered = span.enter();
            tracing::info!({ fields::IS_REPLAY } = false, "live work under replay span");
        }

        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );

        assert!(
            !output.contains("boundary replay hit"),
            "an event carrying isReplay=true must be suppressed even when its \
             span has already flipped to live. Got: {output}"
        );
        assert!(
            output.contains("live work under replay span"),
            "an event carrying isReplay=false must pass even under a span \
             still marked as replaying. Got: {output}"
        );
    }

    /// Verifies that SDK lifecycle events (events NOT inside an operation
    /// span) are unaffected by the replay filter.
    #[test]
    fn replay_filter_allows_sdk_lifecycle_events() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::fmt::MakeWriter;
        use tracing_subscriber::layer::SubscriberExt;

        #[derive(Clone)]
        struct BufWriter(Arc<Mutex<Vec<u8>>>);

        impl std::io::Write for BufWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if let Ok(mut inner) = self.0.lock() {
                    inner.extend_from_slice(buf);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> MakeWriter<'a> for BufWriter {
            type Writer = BufWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = BufWriter(Arc::clone(&buffer));

        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_span_list(false)
            .with_writer(writer)
            .with_filter(ReplayFilterLayer);

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        // Event outside any operation span (SDK lifecycle).
        tracing::info!("sdk lifecycle event");

        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );

        assert!(
            output.contains("sdk lifecycle event"),
            "SDK lifecycle events must pass through. Got: {output}"
        );
    }

    /// A `MakeWriter` that captures output in a shared buffer, for the
    /// dynamic-replay tests below.
    #[derive(Clone)]
    struct CaptureWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CaptureWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if let Ok(mut inner) = self.0.lock() {
                inner.extend_from_slice(buf);
            }
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
        type Writer = CaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Verifies that the filter follows `span.record()` updates to
    /// `isReplay`: events are suppressed while the flag is `true` and
    /// emitted again once it is re-recorded as `false` — the mechanism the
    /// SDK uses on the handler span as the invocation crosses the replay
    /// high-water mark.
    #[test]
    fn replay_filter_follows_dynamic_is_replay_updates() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::layer::SubscriberExt;

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = CaptureWriter(Arc::clone(&buffer));

        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_span_list(false)
            .with_writer(writer)
            .with_filter(ReplayFilterLayer);
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        let span = execution_span("arn:test", "req-1", true);
        {
            let _entered = span.enter();
            tracing::info!("suppressed while replaying");
            span.record(fields::IS_REPLAY, false);
            tracing::info!("emitted after replay ends");
            span.record(fields::IS_REPLAY, true);
            tracing::info!("suppressed again");
        }

        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );

        assert!(
            !output.contains("suppressed while replaying"),
            "Events while isReplay=true must be suppressed. Got: {output}"
        );
        assert!(
            output.contains("emitted after replay ends"),
            "Events after isReplay flips to false must pass. Got: {output}"
        );
        assert!(
            !output.contains("suppressed again"),
            "Events after isReplay flips back to true must be suppressed. Got: {output}"
        );
    }

    /// Verifies the end-to-end handler-level fix: log events emitted by
    /// handler code BETWEEN operations (not inside any operation span) are
    /// suppressed while the execution replays recorded results, and emitted
    /// once it passes the replay high-water mark.
    #[tokio::test]
    async fn replay_filter_suppresses_handler_level_replay_logs() {
        use std::sync::{Arc, Mutex};
        use std::time::Duration;
        use tracing::Instrument as _;
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::layer::SubscriberExt;

        use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let writer = CaptureWriter(Arc::clone(&buffer));

        let fmt_layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_span_list(false)
            .with_writer(writer)
            .with_filter(ReplayFilterLayer);
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = tracing::subscriber::set_default(subscriber);

        // Seed the checkpoint log with a completed wait at position "1":
        // this invocation is a resume that replays the wait.
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = crate::DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );

        // Mirror production (`run`/`wrap`): the handler future is
        // instrumented with the handler-level span.
        let replay_span = ctx.replay_span();
        async move {
            tracing::info!("handler line before the wait");
            let _ = ctx.wait(Duration::from_secs(5)).await;
            tracing::info!("handler line after the wait");
        }
        .instrument(replay_span)
        .await;

        let output = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );

        assert!(
            !output.contains("handler line before the wait"),
            "Handler-level events during replay must be suppressed. Got: {output}"
        );
        assert!(
            output.contains("handler line after the wait"),
            "Handler-level events past the high-water mark must be emitted. Got: {output}"
        );
    }
}
