//! Tracing integration: replay-aware filter layer, span helpers, and
//! lifecycle-event emitters.
//!
//! This module implements the instrumentation whose public, documented
//! contract lives in [`crate::observability`]: span names, lifecycle event
//! names, and field names are all specified there.
//!
//! The SDK creates two kinds of [`tracing::Span`]:
//!
//! - A `durable_execution` span per NAMESPACE. The handler-level span wraps
//!   each invocation of the user's handler; each child namespace (a
//!   `run_in_child_context` body, a map/parallel branch, a
//!   `wait_for_callback` body) gets its own span wrapping its body, a child of
//!   the span current when the child context is created.
//!   Every one carries the execution ARN, the request ID, the tenant ID
//!   when the invocation has one, and an `isReplay` flag that the SDK keeps current as that namespace crosses its own
//!   replay high-water mark: the flag starts as the namespace's initial
//!   replay status and is re-recorded after every operation claim minted
//!   through it. A child span also carries the child operation's
//!   `operationId` and, when the operation has a recorded name, its
//!   `operationName`.
//! - A per-operation `durable_operation` span wrapping each live step body
//!   and each `wait_for_condition` check, carrying the full structured-log
//!   field contract below.
//!
//! The JSON record of [`crate::init_default_subscriber`] and
//! [`crate::init_default_subscriber_with_writer`] writes the correlation
//! fields `requestId`, `executionArn`, `tenantId`, `operationId`,
//! `operationName` and `attempt` as top-level keys. Each comes from the
//! event, else from the nearest SDK span that carries it. The record also
//! writes `timestamp`, `level`, the other fields, and `message` last, and
//! omits `isReplay`. So the `CloudWatch` Logs Insights query below matches
//! it:
//!
//! ```text
//! filter coalesce(durableExecutionArn, executionArn) like "<arn>"
//! ```
//!
//! `lambda_runtime`'s own subscriber and a stock `fmt::layer().json()`
//! write the event's fields under `fields` and the span fields under `span`
//! and `spans`, so the correlation fields are not top-level keys there.
//!
//! # Structured-log field contract
//!
//! | Field | Description | Root `durable_execution` | Child `durable_execution` | `durable_operation` |
//! |-------|-------------|------|-------|-----------|
//! | `executionArn` | Durable execution ARN | yes | yes | yes |
//! | `requestId` | Lambda invocation request ID | yes | yes | yes |
//! | `tenantId` | Lambda tenant ID; omitted when the invocation has none | yes | yes | yes |
//! | `isReplay` | Whether the span covers replayed work | yes | yes | yes |
//! | `operationId` | Wire operation ID (SHA-256 hex) | no | yes | yes |
//! | `operationName` | The operation's recorded name: its `.name(...)`, or a generated batch item name such as `map-item-0`; omitted when it has none | no | yes | yes |
//! | `attempt` | Current attempt number (1-based) | no | no | yes |
//!
//! # Replay filter
//!
//! When an execution resumes, the handler re-runs from the top and the SDK
//! replays recorded results. Handler code between operations executes again,
//! so its log statements would re-emit on every resume. The `isReplay` flag
//! on the handler span lets a per-layer filter suppress those events,
//! avoiding duplicate `CloudWatch` log lines.
//!
//! [`ReplayFilterLayer`] is that filter. [`crate::init_default_subscriber`]
//! installs `lambda_runtime`'s default subscriber settings with it. The SDK
//! never installs a subscriber implicitly: only an explicit call to that
//! helper, or the application's own subscriber, does.

use std::str::FromStr as _;

use lambda_runtime::tracing::subscriber;
use subscriber::Layer as _;
use subscriber::filter::{EnvFilter, LevelFilter};
use subscriber::fmt::format::{FormatEvent, FormatFields as _, JsonFields, Writer};
use subscriber::fmt::{FmtContext, FormattedFields, MakeWriter};
use subscriber::layer::SubscriberExt as _;
use tracing::Id;
use tracing::span::Attributes;

/// Field name constants for the structured-log contract.
///
/// The canonical, documented definitions live in
/// [`crate::observability::field_names`]; this alias keeps the internal
/// spelling short.
pub(crate) use crate::observability::field_names as fields;

/// Records the optional correlation fields on `span`.
///
/// `tracing` writes a field declared as `Empty` only after it is recorded.
/// So a `None` value leaves its field out of the span.
fn record_optional(span: &tracing::Span, tenant_id: Option<&str>, operation_name: Option<&str>) {
    if let Some(tenant_id) = tenant_id {
        span.record(fields::TENANT_ID, tenant_id);
    }
    if let Some(operation_name) = operation_name {
        span.record(fields::OPERATION_NAME, operation_name);
    }
}

/// Creates a `tracing::Span` for a durable operation with the structured-log
/// field contract.
///
/// The span carries:
/// - `executionArn`: the durable execution ARN
/// - `requestId`: the Lambda request ID
/// - `tenantId`: the Lambda tenant ID, when the invocation has one
/// - `operationId`: the wire operation ID (SHA-256 hex)
/// - `operationName`: the operation's recorded name, when it has one
/// - `attempt`: the current attempt (1-based)
/// - `isReplay`: whether this operation is replayed
///
/// User `tracing::info!` calls inside step bodies and `wait_for_condition`
/// checks inherit these fields automatically through span nesting.
#[must_use]
pub(crate) fn operation_span(
    execution_arn: &str,
    request_id: &str,
    tenant_id: Option<&str>,
    operation_id: &str,
    operation_name: Option<&str>,
    attempt: u32,
    is_replay: bool,
) -> tracing::Span {
    let span = tracing::info_span!(
        "durable_operation",
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        { fields::TENANT_ID } = tracing::field::Empty,
        { fields::OPERATION_ID } = operation_id,
        { fields::OPERATION_NAME } = tracing::field::Empty,
        { fields::ATTEMPT } = attempt,
        { fields::IS_REPLAY } = is_replay,
    );
    record_optional(&span, tenant_id, operation_name);
    span
}

/// Creates the handler-level `tracing::Span` wrapping one invocation of the
/// user's handler.
///
/// The span carries:
/// - `executionArn`: the durable execution ARN
/// - `requestId`: the Lambda request ID
/// - `tenantId`: the Lambda tenant ID, when the invocation has one
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
    tenant_id: Option<&str>,
    is_replay: bool,
) -> tracing::Span {
    let span = tracing::info_span!(
        "durable_execution",
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        { fields::TENANT_ID } = tracing::field::Empty,
        { fields::IS_REPLAY } = is_replay,
    );
    record_optional(&span, tenant_id, None);
    span
}

/// Creates the `durable_execution` span of a child namespace: a
/// `run_in_child_context` body, a map/parallel branch, or a
/// `wait_for_callback` body, recorded, virtual or flat.
///
/// Each child context owns an [`crate::engine::EngineState`] with its own ID
/// counter and therefore its own replay high-water mark: nested operations
/// can still be replaying while the parent namespace is already live (or
/// vice versa). Giving each namespace its own span, whose `isReplay` flag
/// that namespace's mints keep current, is what lets a per-layer filter
/// suppress a branch's pre-wait log lines on resume without consulting (or
/// clobbering) the parent's state.
///
/// The span is a child of the span current when the child context is
/// created, so a record inside the child body also carries the spans
/// around it, such as a plugin's span. [`ReplayFilterLayer`] resolves a
/// record's replay status against its nearest execution namespace: it
/// combines the `isReplay` flags of the SDK's `durable_execution` and
/// `durable_operation` spans from the record up to and including the
/// nearest `durable_execution` span, and suppresses the record when any of
/// them replays. So a parent that is still replaying does not suppress a
/// branch that has gone live, a live parent does not enable a branch that
/// is still replaying, and a replaying operation inside a live branch
/// still suppresses its records.
///
/// The span carries the root's fields itself (`executionArn`, `requestId`,
/// `tenantId`, `isReplay`), and adds the child operation's `operationId`
/// and, when it has one, `operationName`.
#[must_use]
pub(crate) fn scoped_execution_span(
    execution_arn: &str,
    request_id: &str,
    tenant_id: Option<&str>,
    operation_id: &str,
    operation_name: Option<&str>,
    is_replay: bool,
) -> tracing::Span {
    let span = tracing::info_span!(
        "durable_execution",
        { fields::EXECUTION_ARN } = execution_arn,
        { fields::REQUEST_ID } = request_id,
        { fields::TENANT_ID } = tracing::field::Empty,
        { fields::OPERATION_ID } = operation_id,
        { fields::OPERATION_NAME } = tracing::field::Empty,
        { fields::IS_REPLAY } = is_replay,
    );
    record_optional(&span, tenant_id, operation_name);
    span
}

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
/// is configured, the flush task, not the contributor future, owns and
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

/// Emits [`event_names::OPERATION_REPLAYED`]: a recorded terminal outcome
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
struct ReplayVisitor {
    is_replay: Option<bool>,
}

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
/// Used by [`ReplayFilterLayer`] to read an SDK span's initial replay flag
/// at creation time.
pub(crate) fn is_replay_event(attrs: &Attributes<'_>) -> bool {
    let mut visitor = ReplayVisitor { is_replay: None };
    attrs.record(&mut visitor);
    visitor.is_replay == Some(true)
}

/// Extracts the `isReplay` value from a `span.record()` update, if present.
///
/// Used by [`ReplayFilterLayer`] to track the dynamic replay flag of an SDK
/// namespace span, which the SDK re-records after every operation claim.
fn replay_flag_in_record(values: &tracing::span::Record<'_>) -> Option<bool> {
    let mut visitor = ReplayVisitor { is_replay: None };
    values.record(&mut visitor);
    visitor.is_replay
}

/// A subscriber wrapper that tracks which spans have `isReplay = true` and
/// provides a method to check if events should be suppressed.
///
/// This is used in the SDK's test infrastructure to verify replay filter
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

    /// Returns whether the span with this ID is itself marked as replay.
    /// It does not look at other spans; [`ReplayFilterLayer`] is what
    /// resolves a record's nearest execution namespace.
    pub(crate) fn is_replay(&self, id: u64) -> bool {
        self.replay_spans.read().is_ok_and(|set| set.contains(&id))
    }
}

/// A per-layer filter that suppresses events describing replayed work:
/// events carrying their own `isReplay = true` field (the SDK's lifecycle
/// events: see [`crate::observability`]) and application log lines emitted
/// while their nearest execution namespace replays.
///
/// When an execution resumes, the handler re-runs from the top and replays
/// recorded results, so handler code between operations executes, and logs,
/// again. The SDK wraps each invocation, and each child namespace, in a
/// `durable_execution` span whose `isReplay` flag tracks that namespace's
/// live replay status, and each step attempt and condition check in a
/// `durable_operation` span. So a log line is written once, on the
/// invocation that first executed it, not again on every resume.
///
/// For an event without its own `isReplay` field, the filter walks from the
/// event towards the root through the SDK spans only, up to and including
/// the first `durable_execution` span, and suppresses the event when any of
/// those spans has `isReplay = true`. Spans outside that namespace decide
/// nothing: a live child namespace is not suppressed by a replaying parent,
/// and a replaying child is not enabled by a live parent. A span that the
/// SDK did not create decides nothing either, even when it declares an
/// `isReplay` field, so a plugin's span around `next()` keeps the
/// suppression of the SDK span it is entered inside.
///
/// For events that carry an explicit `isReplay` field, that field is
/// authoritative and the span scope is not consulted: the execution span's
/// flag tracks the *next* operation claim, so at the replay high-water
/// boundary the final `operation_replayed` lifecycle event can fire under
/// a span already flipped to live. Inspecting the event's own field keeps
/// suppression exact at that boundary.
///
/// Suppression is global by design: the subscriber, not the handler, decides
/// which records are written. [`crate::init_default_subscriber`] installs a
/// subscriber with this filter. A custom subscriber adds the filter to its
/// own layer to keep each record between operations written once per
/// execution.
///
/// The filter only decides which records are written. A custom subscriber
/// keeps its formatter's record shape: a stock `fmt::layer().json()`, as
/// below, writes the event's fields under `fields` and the span fields
/// under `span` and `spans`, not the top-level record of
/// [`crate::init_default_subscriber`].
///
/// # Usage
///
/// The filter implements the `Filter` trait of `tracing-subscriber`, which
/// `lambda_runtime` re-exports as `lambda_runtime::tracing::subscriber`, so
/// no extra dependency is needed:
///
/// ```
/// use aws_durable_execution_sdk::ReplayFilterLayer;
/// use aws_durable_execution_sdk::lambda_runtime::tracing::subscriber;
/// use subscriber::Layer as _;
/// use subscriber::layer::SubscriberExt as _;
///
/// let subscriber = subscriber::registry().with(
///     subscriber::fmt::layer()
///         .json()
///         .with_filter(ReplayFilterLayer::new()),
/// );
///
/// // In a Lambda binary, install it globally instead:
/// // `subscriber::util::SubscriberInitExt::init(subscriber)`.
/// tracing::subscriber::with_default(subscriber, || {
///     tracing::info!("emitted normally, not inside a replay span");
/// });
/// ```
///
/// # Keeping one target during replay
///
/// A record of the target `audit` below is written on every invocation,
/// replay included. Every other record is written once per execution. The
/// first layer writes only `audit` records and has no replay filter. The
/// second layer writes every other target through the replay filter.
///
/// ```
/// use aws_durable_execution_sdk::ReplayFilterLayer;
/// use aws_durable_execution_sdk::lambda_runtime::tracing::subscriber;
/// use subscriber::Layer as _;
/// use subscriber::filter::{FilterExt as _, LevelFilter, Targets};
/// use subscriber::layer::SubscriberExt as _;
///
/// let subscriber = subscriber::registry()
///     .with(
///         subscriber::fmt::layer()
///             .with_filter(Targets::new().with_target("audit", LevelFilter::TRACE)),
///     )
///     .with(
///         subscriber::fmt::layer().with_filter(
///             ReplayFilterLayer::new().and(
///                 Targets::new()
///                     .with_default(LevelFilter::TRACE)
///                     .with_target("audit", LevelFilter::OFF),
///             ),
///         ),
///     );
///
/// tracing::subscriber::with_default(subscriber, || {
///     tracing::info!(target: "audit", "written on every invocation");
///     tracing::info!("written once per execution");
/// });
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ReplayFilterLayer;

impl ReplayFilterLayer {
    /// Creates a replay filter.
    ///
    /// The filter carries no configuration today; construct it with this
    /// method (or [`Default`]) rather than the bare struct name, so that a
    /// future field addition stays compatible.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl<S> subscriber::layer::Filter<S> for ReplayFilterLayer
where
    S: tracing::Subscriber + for<'lookup> subscriber::registry::LookupSpan<'lookup>,
{
    fn enabled(
        &self,
        _meta: &tracing::Metadata<'_>,
        _ctx: &subscriber::layer::Context<'_, S>,
    ) -> bool {
        // Always allow span creation: we only filter events.
        true
    }

    fn event_enabled(
        &self,
        event: &tracing::Event<'_>,
        ctx: &subscriber::layer::Context<'_, S>,
    ) -> bool {
        // An event carrying its own `isReplay` field (every SDK lifecycle
        // event does: see `crate::observability`) is authoritative: the
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

        // Otherwise (application log lines carry no `isReplay` field), the
        // event's nearest execution namespace decides: walk from the event
        // towards the root through the SDK spans, and stop at the first
        // `durable_execution` span. Any replaying SDK span on the way
        // suppresses the event; a span outside the namespace does not.
        if let Some(scope) = ctx.event_scope(event) {
            for span in scope {
                let extensions = span.extensions();
                let Some(fields) = extensions.get::<ReplaySpanFields>() else {
                    continue;
                };
                if fields.is_replay {
                    return false;
                }
                if fields.kind == SdkSpan::Execution {
                    break;
                }
            }
        }
        true
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: subscriber::layer::Context<'_, S>) {
        // Keep the kind and the initial `isReplay` value of each SDK span.
        // Span extensions are shared by every layer of the subscriber, so
        // with this filter on two layers the second call finds the first
        // one's value: `replace` stores the same value again where
        // `insert` would panic.
        if let Some(kind) = sdk_span(attrs.metadata())
            && let Some(span) = ctx.span(id)
        {
            span.extensions_mut().replace(ReplaySpanFields {
                kind,
                is_replay: is_replay_event(attrs),
            });
        }
    }

    fn on_record(
        &self,
        id: &Id,
        values: &tracing::span::Record<'_>,
        ctx: subscriber::layer::Context<'_, S>,
    ) {
        // The SDK re-records a namespace span's isReplay flag after every
        // operation claim (see `execution_span`). Track those updates so
        // suppression follows the live replay status, not just the value
        // the span was created with.
        if let Some(is_replay) = replay_flag_in_record(values)
            && let Some(span) = ctx.span(id)
            && let Some(fields) = span.extensions_mut().get_mut::<ReplaySpanFields>()
        {
            fields.is_replay = is_replay;
        }
    }
}

/// The kind of an SDK span that carries the replay state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SdkSpan {
    /// A `durable_execution` span: one execution namespace.
    Execution,
    /// A `durable_operation` span: one step attempt or condition check.
    Operation,
}

/// The target of the spans this module creates.
const SPAN_TARGET: &str = module_path!();

/// The kind of the SDK span `meta` describes, or `None` for any other span
/// or for an event. Only the spans this module creates count, whatever
/// fields another span declares.
fn sdk_span(meta: &tracing::Metadata<'_>) -> Option<SdkSpan> {
    if !meta.is_span() || meta.target() != SPAN_TARGET {
        return None;
    }
    match meta.name() {
        crate::observability::span_names::EXECUTION => Some(SdkSpan::Execution),
        crate::observability::span_names::OPERATION => Some(SdkSpan::Operation),
        _ => None,
    }
}

/// Storage for an SDK span's kind and current replay flag in span
/// extensions.
#[derive(Debug)]
struct ReplaySpanFields {
    kind: SdkSpan,
    is_replay: bool,
}

/// The level, the `EnvFilter` directives and the format that the default
/// subscriber is built with.
///
/// [`LogSettings::from_lookup`] reads them from a lookup function, so tests
/// can supply the variables without changing the process environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LogSettings {
    /// The default level: `AWS_LAMBDA_LOG_LEVEL`, else `RUST_LOG`, else
    /// `INFO`.
    pub(crate) level: String,
    /// The `EnvFilter` directives: `RUST_LOG`, else empty.
    pub(crate) directives: String,
    /// The format: `AWS_LAMBDA_LOG_FORMAT`, else empty.
    pub(crate) format: String,
}

impl LogSettings {
    /// Reads the settings from the process environment.
    pub(crate) fn from_env() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Reads the settings with `lookup`, which returns the value of the
    /// variable it is given, or `None` when the variable is unset.
    ///
    /// These are the rules of
    /// `lambda_runtime::tracing::init_default_subscriber_with_writer`.
    /// `RUST_LOG` serves twice: as the level when `AWS_LAMBDA_LOG_LEVEL` is
    /// unset, and as the `EnvFilter` directives, which `from_env_lossy`
    /// reads there.
    pub(crate) fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let directives = lookup("RUST_LOG");
        let level = lookup("AWS_LAMBDA_LOG_LEVEL")
            .or_else(|| directives.clone())
            .unwrap_or_else(|| "INFO".to_owned());
        Self {
            level,
            directives: directives.unwrap_or_default(),
            format: lookup("AWS_LAMBDA_LOG_FORMAT").unwrap_or_default(),
        }
    }
}

/// The filter of the default subscriber's `fmt` layer: the configured
/// `EnvFilter`, then [`ReplayFilterLayer`].
///
/// The SDK creates its replay-tracking spans at `INFO`. An `EnvFilter` with a
/// level above `INFO`, or a directive that turns the SDK's target off,
/// disables those spans. A disabled span is never created, so
/// [`ReplayFilterLayer`] cannot see its `isReplay` flag. A record that the
/// `EnvFilter` allows would then be written again on every replay. So this
/// filter enables the replay-tracking spans whatever the `EnvFilter` says,
/// and applies the `EnvFilter` to everything else. It also raises the level
/// hint to at least `INFO`, because `tracing` skips a span above the global
/// maximum level before any filter sees it.
///
/// `F` is the `EnvFilter` in the subscriber. Tests use another filter, to
/// observe the hooks this filter forwards.
struct DefaultFilter<F> {
    env: F,
    replay: ReplayFilterLayer,
}

impl<S, F> subscriber::layer::Filter<S> for DefaultFilter<F>
where
    S: tracing::Subscriber + for<'lookup> subscriber::registry::LookupSpan<'lookup>,
    F: subscriber::layer::Filter<S>,
{
    fn enabled(
        &self,
        meta: &tracing::Metadata<'_>,
        cx: &subscriber::layer::Context<'_, S>,
    ) -> bool {
        sdk_span(meta).is_some() || self.env.enabled(meta, cx)
    }

    fn callsite_enabled(
        &self,
        meta: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        // `sometimes`, never `always`: `tracing` skips `enabled` for an
        // `always` callsite, and `enabled` is what resets the per-layer
        // filter state that `event_enabled` leaves behind when it drops a
        // replayed record. `ReplayFilterLayer` alone returns `sometimes`
        // for the same reason.
        //
        // The `EnvFilter` is asked for every callsite, replay-tracking spans
        // included. Its `callsite_enabled` registers the matcher of a span
        // directive such as `[durable_execution]=debug`, and `on_new_span`
        // finds a span's matcher only when that registration happened. So
        // skipping the call for the SDK's spans would disable those
        // directives.
        let env = self.env.callsite_enabled(meta);
        if sdk_span(meta).is_none() && env.is_never() {
            tracing::subscriber::Interest::never()
        } else {
            tracing::subscriber::Interest::sometimes()
        }
    }

    fn event_enabled(
        &self,
        event: &tracing::Event<'_>,
        cx: &subscriber::layer::Context<'_, S>,
    ) -> bool {
        subscriber::layer::Filter::<S>::event_enabled(&self.replay, event, cx)
    }

    fn max_level_hint(&self) -> Option<LevelFilter> {
        self.env
            .max_level_hint()
            .map(|hint| hint.max(LevelFilter::INFO))
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: subscriber::layer::Context<'_, S>) {
        self.env.on_new_span(attrs, id, ctx.clone());
        subscriber::layer::Filter::<S>::on_new_span(&self.replay, attrs, id, ctx);
    }

    fn on_record(
        &self,
        id: &Id,
        values: &tracing::span::Record<'_>,
        ctx: subscriber::layer::Context<'_, S>,
    ) {
        self.env.on_record(id, values, ctx.clone());
        subscriber::layer::Filter::<S>::on_record(&self.replay, id, values, ctx);
    }

    fn on_enter(&self, id: &Id, ctx: subscriber::layer::Context<'_, S>) {
        self.env.on_enter(id, ctx);
    }

    fn on_exit(&self, id: &Id, ctx: subscriber::layer::Context<'_, S>) {
        self.env.on_exit(id, ctx);
    }

    fn on_close(&self, id: Id, ctx: subscriber::layer::Context<'_, S>) {
        self.env.on_close(id, ctx);
    }
}

/// Builds `lambda_runtime`'s default subscriber with replayed records
/// suppressed.
///
/// The settings are those of
/// `lambda_runtime::tracing::init_default_subscriber_with_writer`:
///
/// 1. An `EnvFilter` built from `directives`, ignoring invalid ones. When no
///    directive is valid, the filter uses its default directive, `level`. A
///    `level` that does not parse as a level filter falls back to `INFO`.
///    The caller passes the value of `RUST_LOG`, which is what
///    `from_env_lossy` would read.
/// 2. An `fmt` layer writing to `writer`. When `format` is `json` in any
///    case, it writes one [`TopLevelJson`] record per event: a JSON object
///    with the fields at the top level and a `timestamp`, the UTC time to
///    the millisecond. Otherwise it writes text without the target and the
///    time, and [`SpanFieldsInPlace`] keeps each span field in that text
///    once, with the value recorded last.
///
/// The `fmt` layer is filtered by [`DefaultFilter`]: the `EnvFilter`, then
/// [`ReplayFilterLayer`], with the SDK's replay-tracking spans always
/// enabled. The caller reads the environment, so tests pass every setting
/// directly. The box lets the JSON and text branches return one type.
pub(crate) fn default_subscriber<W>(
    level: &str,
    directives: &str,
    format: &str,
    writer: W,
) -> Box<dyn tracing::Subscriber + Send + Sync>
where
    W: for<'w> MakeWriter<'w> + Send + Sync + 'static,
{
    let level = LevelFilter::from_str(level).unwrap_or(LevelFilter::INFO);
    let filter = DefaultFilter {
        env: EnvFilter::builder()
            .with_default_directive(level.into())
            .parse_lossy(directives),
        replay: ReplayFilterLayer::new(),
    };
    let registry = subscriber::registry();
    if format.eq_ignore_ascii_case("json") {
        let layer = subscriber::fmt::layer()
            .fmt_fields(JsonFields::new())
            .event_format(TopLevelJson)
            .with_writer(writer);
        Box::new(registry.with(layer.with_filter(filter)))
    } else {
        let layer = subscriber::fmt::layer()
            .fmt_fields(TextFields::default())
            .with_target(false)
            .without_time()
            .with_writer(writer)
            .and_then(SpanFieldsInPlace);
        Box::new(registry.with(layer.with_filter(filter)))
    }
}

/// The field formatter of the default subscriber's text branch.
///
/// It formats fields as `DefaultFields` does, but adds nothing to a span's
/// formatted fields when a value is recorded after the span was created:
/// [`SpanFieldsInPlace`], which runs after the `fmt` layer, rewrites them
/// instead. So a line shows each field of a span once.
#[derive(Debug, Default)]
struct TextFields(subscriber::fmt::format::DefaultFields);

impl<'w> subscriber::fmt::format::FormatFields<'w> for TextFields {
    fn format_fields<R: subscriber::field::RecordFields>(
        &self,
        writer: Writer<'w>,
        fields: R,
    ) -> std::fmt::Result {
        self.0.format_fields(writer, fields)
    }

    fn add_fields(
        &self,
        _current: &'w mut FormattedFields<Self>,
        _fields: &tracing::span::Record<'_>,
    ) -> std::fmt::Result {
        Ok(())
    }
}

/// Each formatted field of a span of the text branch, by name, in the
/// order the span first showed them.
struct TextSpanFields(Vec<(&'static str, String)>);

/// Keeps one entry per field in the formatted fields of each span of the
/// default subscriber's text branch.
///
/// `tracing-subscriber`'s default field formatter appends a value recorded
/// after a span's creation, so the SDK's `isReplay` record would show
/// `isReplay=true isReplay=false`. This layer stores each field's formatted
/// text when the span is created, replaces a field's text when a value is
/// recorded, at the position the span first showed it, and rewrites the
/// span's formatted fields from the stored texts.
///
/// A field's `Debug` or `Display` implementation can return an error. When
/// the `fmt` layer could not format a span's fields, it stores no formatted
/// fields, and this layer leaves the span alone. When this layer cannot
/// format a field at the span's creation, it stores no texts, so the span
/// keeps the formatted fields of the `fmt` layer and later records change
/// nothing. A recorded value that fails to format leaves its field's
/// earlier text in place.
struct SpanFieldsInPlace;

impl<S> subscriber::Layer<S> for SpanFieldsInPlace
where
    S: tracing::Subscriber + for<'a> subscriber::registry::LookupSpan<'a>,
{
    #[expect(clippy::expect_used)]
    // reason: the registry calls this hook for a span it stores.
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: subscriber::layer::Context<'_, S>) {
        let span = ctx.span(id).expect("the registry stores the new span");
        let mut extensions = span.extensions_mut();
        let Some(formatted) = extensions.get_mut::<FormattedFields<TextFields>>() else {
            return;
        };
        let mut fields = Vec::new();
        let mut texts = FieldTexts {
            formatted,
            fields: &mut fields,
            result: Ok(()),
        };
        attrs.record(&mut texts);
        if texts.result.is_ok() {
            extensions.insert(TextSpanFields(fields));
        }
    }

    #[expect(clippy::expect_used)]
    // reason: the registry calls this hook for a span it stores, and
    // `on_new_span` stores the span's texts only beside formatted fields,
    // which nothing removes.
    fn on_record(
        &self,
        id: &Id,
        values: &tracing::span::Record<'_>,
        ctx: subscriber::layer::Context<'_, S>,
    ) {
        let span = ctx.span(id).expect("the registry stores the recorded span");
        let mut extensions = span.extensions_mut();
        let Some(TextSpanFields(mut fields)) = extensions.remove::<TextSpanFields>() else {
            return;
        };
        let formatted = extensions
            .get_mut::<FormattedFields<TextFields>>()
            .expect("the span's texts are stored beside its formatted fields");
        // A value that fails to format stores no text, so its field keeps
        // its earlier text, and the other recorded values still apply.
        values.record(&mut FieldTexts {
            formatted,
            fields: &mut fields,
            result: Ok(()),
        });
        let joined: Vec<&str> = fields.iter().map(|(_, text)| text.as_str()).collect();
        formatted.fields = joined.join(" ");
        extensions.insert(TextSpanFields(fields));
    }
}

/// Formats each visited field alone, as `DefaultFields` does, and stores
/// its text in `fields`: in place of an earlier text of the same name, or
/// at the end.
///
/// The text is written through `formatted`'s writer, so it has the ANSI
/// settings the span's fields were formatted with. `formatted`'s own text
/// is put back after each field. A field whose formatting fails stores no
/// text and sets `result` to the error.
struct FieldTexts<'a> {
    formatted: &'a mut FormattedFields<TextFields>,
    fields: &'a mut Vec<(&'static str, String)>,
    result: std::fmt::Result,
}

impl FieldTexts<'_> {
    fn store(
        &mut self,
        field: &tracing::field::Field,
        write: impl FnOnce(&mut subscriber::fmt::format::DefaultVisitor<'_>),
    ) {
        use subscriber::field::VisitOutput as _;

        let saved = std::mem::take(&mut self.formatted.fields);
        let mut visitor =
            subscriber::fmt::format::DefaultVisitor::new(self.formatted.as_writer(), true);
        write(&mut visitor);
        let result = visitor.finish();
        let text = std::mem::replace(&mut self.formatted.fields, saved);
        if let Err(error) = result {
            self.result = Err(error);
            return;
        }
        match self
            .fields
            .iter_mut()
            .find(|(name, _)| *name == field.name())
        {
            Some((_, old)) => *old = text,
            None => self.fields.push((field.name(), text)),
        }
    }
}

impl tracing::field::Visit for FieldTexts<'_> {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.store(field, |v| v.record_str(field, value));
    }

    fn record_error(
        &mut self,
        field: &tracing::field::Field,
        value: &(dyn std::error::Error + 'static),
    ) {
        self.store(field, |v| v.record_error(field, value));
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.store(field, |v| v.record_debug(field, value));
    }
}

/// The event format of the default subscriber's JSON branch.
///
/// Each record is one JSON object on one line. Its keys are, in order:
/// `requestId`, `timestamp`, `level`, `executionArn`, `tenantId`,
/// `operationId`, `operationName`, `attempt`, the ordinary fields, then
/// `message`, as the JavaScript SDK's default logger writes them. A key
/// that has no value is left out, except `timestamp`, `level` and
/// `message`, which every record has.
///
/// A correlation key takes the event's value, else the value of the
/// nearest SDK span that carries it. A span the SDK did not create never
/// supplies one. An ordinary field is any other recorded field except
/// `timestamp`, `level`, `isReplay` and `message`; it takes the event's
/// value, else the value of the nearest span that carries it. `isReplay`
/// is never written. `message` is the event's own field, or `""`.
///
/// Values have the representation `JsonFields` gives them: the event's
/// fields are formatted with the layer's `JsonFields`, and the span fields
/// are the `FormattedFields<JsonFields>` that the `fmt` layer stores.
#[derive(Debug, Clone, Copy)]
struct TopLevelJson;

/// The correlation keys after `requestId`, `timestamp` and `level`, in the
/// order the record writes them.
const TRAILING_CORRELATION_KEYS: [&str; 5] = [
    fields::EXECUTION_ARN,
    fields::TENANT_ID,
    fields::OPERATION_ID,
    fields::OPERATION_NAME,
    fields::ATTEMPT,
];

/// The key of the record's time.
const TIMESTAMP_KEY: &str = "timestamp";
/// The key of the record's level.
const LEVEL_KEY: &str = "level";
/// The key of the record's message.
const MESSAGE_KEY: &str = "message";

/// Whether `key` is written from a place other than the ordinary fields.
fn is_reserved_key(key: &str) -> bool {
    key == fields::REQUEST_ID
        || TRAILING_CORRELATION_KEYS.contains(&key)
        || key == TIMESTAMP_KEY
        || key == LEVEL_KEY
        || key == fields::IS_REPLAY
        || key == MESSAGE_KEY
}

/// The fields of one source of a record: the event, or one span.
type FieldMap = std::collections::BTreeMap<String, serde_json::Value>;

/// Parses fields that `JsonFields` formatted as one JSON object.
fn parse_fields(formatted: &str) -> Result<FieldMap, std::fmt::Error> {
    serde_json::from_str(formatted).map_err(|_| std::fmt::Error)
}

/// The fields of one span in an event's scope.
struct SpanFields {
    /// Whether the SDK created the span, so it may supply correlation keys.
    is_sdk: bool,
    fields: FieldMap,
}

impl<S> FormatEvent<S, JsonFields> for TopLevelJson
where
    S: tracing::Subscriber + for<'lookup> subscriber::registry::LookupSpan<'lookup>,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, JsonFields>,
        mut writer: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        // The time is sampled first, so a time outside the four-digit years
        // fails the record before any of it is written.
        let timestamp = js_timestamp(std::time::SystemTime::now())?;

        let mut formatted = String::new();
        ctx.field_format()
            .format_fields(Writer::new(&mut formatted), event)?;
        let event_fields = parse_fields(&formatted)?;

        // Leaf to root, so the first span found is the nearest one.
        let mut spans = Vec::new();
        if let Some(scope) = ctx.event_scope() {
            for span in scope {
                let extensions = span.extensions();
                let fields = match extensions.get::<FormattedFields<JsonFields>>() {
                    Some(stored) => parse_fields(&stored.fields)?,
                    None => FieldMap::new(),
                };
                spans.push(SpanFields {
                    is_sdk: sdk_span(span.metadata()).is_some(),
                    fields,
                });
            }
        }

        let correlation = |key: &str| {
            event_fields.get(key).or_else(|| {
                spans
                    .iter()
                    .filter(|span| span.is_sdk)
                    .find_map(|span| span.fields.get(key))
            })
        };
        let mut ordinary = std::collections::BTreeMap::new();
        for source in std::iter::once(&event_fields).chain(spans.iter().map(|span| &span.fields)) {
            for (key, value) in source {
                if !is_reserved_key(key) {
                    ordinary.entry(key.as_str()).or_insert(value);
                }
            }
        }
        let empty_message = serde_json::Value::String(String::new());
        let message = event_fields.get(MESSAGE_KEY).unwrap_or(&empty_message);
        let level = serde_json::Value::String(event.metadata().level().as_str().to_owned());
        let timestamp = serde_json::Value::String(timestamp);

        let mut entries: Vec<(&str, &serde_json::Value)> = Vec::new();
        if let Some(value) = correlation(fields::REQUEST_ID) {
            entries.push((fields::REQUEST_ID, value));
        }
        entries.push((TIMESTAMP_KEY, &timestamp));
        entries.push((LEVEL_KEY, &level));
        for key in TRAILING_CORRELATION_KEYS {
            if let Some(value) = correlation(key) {
                entries.push((key, value));
            }
        }
        entries.extend(ordinary);
        entries.push((MESSAGE_KEY, message));

        // `Value`'s `Display` writes compact JSON, and a string value
        // escapes the key as a JSON string.
        writer.write_char('{')?;
        for (index, (key, value)) in entries.into_iter().enumerate() {
            if index > 0 {
                writer.write_char(',')?;
            }
            write!(writer, "{}:{value}", serde_json::Value::from(key))?;
        }
        writer.write_str("}\n")
    }
}

/// Formats `time` as the JavaScript SDK's records do, which is
/// `Date.prototype.toISOString`: UTC, `YYYY-MM-DDTHH:MM:SS.sssZ`, with the
/// time cut to whole milliseconds rather than rounded.
///
/// Returns an error when the UTC year is outside `0000..=9999`, which the
/// format cannot write.
fn js_timestamp(time: std::time::SystemTime) -> Result<String, std::fmt::Error> {
    const MILLIS_PER_DAY: i128 = 86_400_000;
    // Milliseconds since the epoch, rounded towards earlier times.
    let millis = match time.duration_since(std::time::UNIX_EPOCH) {
        Ok(after) => i128::from(after.as_secs()) * 1000 + i128::from(after.subsec_millis()),
        Err(before) => {
            let before = before.duration();
            let nanos =
                i128::from(before.as_secs()) * 1_000_000_000 + i128::from(before.subsec_nanos());
            -((nanos + 999_999) / 1_000_000)
        }
    };
    let (year, month, day) = civil_from_days(millis.div_euclid(MILLIS_PER_DAY));
    if !(0..=9999).contains(&year) {
        return Err(std::fmt::Error);
    }
    let of_day = millis.rem_euclid(MILLIS_PER_DAY);
    let (hours, minutes) = (of_day / 3_600_000, of_day / 60_000 % 60);
    let (seconds, subsec) = (of_day / 1000 % 60, of_day % 1000);
    Ok(format!(
        "{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}.{subsec:03}Z"
    ))
}

/// The proleptic Gregorian year, month and day of `days` since
/// 1970-01-01, by Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(days: i128) -> (i128, i128, i128) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    // Months counted from March, so February is last and takes leap days.
    let march_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * march_month + 2) / 5 + 1;
    let month = if march_month < 10 {
        march_month + 3
    } else {
        march_month - 9
    };
    let year = year_of_era + era * 400 + i128::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn operation_span_has_correct_fields() {
        // Install a no-op subscriber so spans are not disabled.
        let _guard = test_tracing::install_test_subscriber(
            tracing_subscriber::fmt().with_test_writer().finish(),
        );

        let span = operation_span(
            "arn:aws:lambda:us-east-1:123:function:test:durable:abc",
            "req-123",
            None,
            "deadbeef01234567890abcdef0123456789abcdef0123456789abcdef012345",
            None,
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

        let _guard = test_tracing::install_test_subscriber(subscriber);

        let test_arn = "arn:aws:lambda:us-east-1:123456789012:function:myFunc:durable:exec-123";
        let test_request_id = "req-abc-def-456";
        let test_op_id = "aabbccdd00112233445566778899aabbccddeeff00112233445566778899aabb";

        let span = operation_span(test_arn, test_request_id, None, test_op_id, None, 2, false);
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
    /// `isReplay = true`: enabling downstream filters to suppress them.
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

        let _guard = test_tracing::install_test_subscriber(subscriber);

        // Create a replay span (isReplay = true).
        let span = operation_span("arn:test", "req-1", None, "op-1", None, 1, true);
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
            .with_filter(ReplayFilterLayer::new());

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = test_tracing::install_test_subscriber(subscriber);

        // Event in a live span: should pass through.
        {
            let span = operation_span("arn:test", "req-1", None, "op-1", None, 1, false);
            let _entered = span.enter();
            tracing::info!("live event");
        }

        // Event in a replay span: should be suppressed.
        {
            let span = operation_span("arn:test", "req-1", None, "op-2", None, 1, true);
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
    /// fires its `operation_replayed` event, which carries `isReplay =
    /// true`, under a span already flipped to `isReplay = false`. The
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
            .with_filter(ReplayFilterLayer::new());

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = test_tracing::install_test_subscriber(subscriber);

        // The boundary: the span starts replaying, then the next mint flips
        // it to live, and only afterwards does the final replay hit emit its
        // event (carrying isReplay = true).
        {
            let span = execution_span("arn:test", "req-1", None, true);
            span.record(fields::IS_REPLAY, false); // mint_id: next op is live
            let _entered = span.enter();
            tracing::info!({ fields::IS_REPLAY } = true, "boundary replay hit");
        }

        // The converse: a live-work event under a span still marked as
        // replaying must pass: its own field is authoritative.
        {
            let span = execution_span("arn:test", "req-1", None, true);
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
            .with_filter(ReplayFilterLayer::new());

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = test_tracing::install_test_subscriber(subscriber);

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

    impl<'a> MakeWriter<'a> for CaptureWriter {
        type Writer = CaptureWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Verifies that the filter follows `span.record()` updates to
    /// `isReplay`: events are suppressed while the flag is `true` and
    /// emitted again once it is re-recorded as `false`: the mechanism the
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
            .with_filter(ReplayFilterLayer::new());
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = test_tracing::install_test_subscriber(subscriber);

        let span = execution_span("arn:test", "req-1", None, true);
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
            .with_filter(ReplayFilterLayer::new());
        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        let _guard = test_tracing::install_test_subscriber(subscriber);

        // Seed the checkpoint log with a completed wait at position "1":
        // this invocation is a resume that replays the wait.
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
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
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = crate::DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );

        // Production (`run`/`wrap`) polls and drops the handler future
        // inside the root task's `TaskContext`, which enters the
        // handler-level span; instrumenting the future with that span
        // enters it at the same points for this test.
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

    /// Reads the text a [`CaptureWriter`] captured, without the ANSI color
    /// sequences the text format writes.
    fn captured(buffer: &std::sync::Mutex<Vec<u8>>) -> String {
        let raw = buffer.lock().map_or_else(
            |_| String::new(),
            |b| String::from_utf8_lossy(&b).to_string(),
        );
        let mut text = String::with_capacity(raw.len());
        let mut chars = raw.chars();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                // An SGR sequence is `ESC [ ... m`; skip through the `m`.
                for skipped in chars.by_ref() {
                    if skipped == 'm' {
                        break;
                    }
                }
            } else {
                text.push(c);
            }
        }
        text
    }

    /// Counts the captured text lines whose message is exactly `message`.
    /// The text format ends each line with the message, so the last word
    /// is compared. A message that contains another one (`audit-before`
    /// contains `before`) is not counted for both.
    fn count_message(output: &str, message: &str) -> usize {
        output
            .lines()
            .filter(|line| line.split_whitespace().last() == Some(message))
            .count()
    }

    /// How the handler of [`run_logging_handler`] logs `before` and `after`.
    #[derive(Clone, Copy)]
    enum LogStyle {
        /// `info!` with the default target.
        Info,
        /// `warn!` with the default target.
        Warn,
        /// `info!` with the target `app`.
        AppTarget,
    }

    fn log_line(style: LogStyle, message: &str) {
        match style {
            LogStyle::Info => tracing::info!("{message}"),
            LogStyle::Warn => tracing::warn!("{message}"),
            LogStyle::AppTarget => tracing::info!(target: "app", "{message}"),
        }
    }

    /// Runs a handler through `LocalRunner` under `subscriber`. The handler
    /// logs `before`, waits one second, and logs `after`. With
    /// `log_audit`, it also logs `audit-before` with the target `audit`
    /// before the wait. The run uses a current-thread runtime, because
    /// `with_default` scopes the subscriber to one thread.
    fn run_logging_handler<S>(subscriber: S, log_audit: bool) -> usize
    where
        S: tracing::Subscriber + Send + Sync + 'static,
    {
        run_logging_handler_with(subscriber, log_audit, LogStyle::Info)
    }

    /// [`run_logging_handler`], with `before` and `after` logged in `style`.
    fn run_logging_handler_with<S>(subscriber: S, log_audit: bool, style: LogStyle) -> usize
    where
        S: tracing::Subscriber + Send + Sync + 'static,
    {
        use std::time::Duration;

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                let result = crate::test_util::LocalRunner::new()
                    .run(
                        move |_event: serde_json::Value, ctx: crate::DurableContext| async move {
                            if log_audit {
                                tracing::info!(target: "audit", "audit-before");
                            }
                            log_line(style, "before");
                            ctx.wait(Duration::from_secs(1)).name("pause").await?;
                            log_line(style, "after");
                            Ok::<_, crate::BoxError>(())
                        },
                        serde_json::Value::Null,
                    )
                    .await
                    .expect("runner");
                result.invocation_count()
            })
        }
    }

    #[test]
    fn default_subscriber_writes_each_record_once_across_replay() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "INFO",
            "",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        let invocations = run_logging_handler(subscriber, false);

        let output = captured(&buffer);
        assert!(invocations >= 2, "the wait must suspend: {invocations}");
        assert_eq!(count_message(&output, "before"), 1, "{output}");
        assert_eq!(count_message(&output, "after"), 1, "{output}");
    }

    /// The text format shows a field recorded after the span's creation
    /// once, with its latest value, where the span first showed it.
    #[test]
    fn default_text_replaces_recorded_span_fields_in_place() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "INFO",
            "",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let span = tracing::info_span!(
                "phase",
                stage = "one",
                stable = "keep",
                late = tracing::field::Empty
            );
            span.record("stage", "two");
            span.record("late", "set");
            let _entered = span.enter();
            tracing::info!("inside");
        }

        let output = captured(&buffer);
        let line = output
            .lines()
            .find(|line| line.contains("inside"))
            .expect("the line is written");
        assert!(line.contains(r#"stage="two""#), "{line}");
        assert!(!line.contains(r#"stage="one""#), "{line}");
        assert_eq!(line.matches("stable").count(), 1, "{line}");
        assert!(
            line.find("stage").expect("stage") < line.find("stable").expect("stable"),
            "{line}"
        );
        // A field first recorded after creation goes after the others.
        assert!(
            line.find("stable").expect("stable") < line.find(r#"late="set""#).expect("late"),
            "{line}"
        );
    }

    /// A `Debug` value whose formatting fails on the calls `fail_on` names,
    /// counted from 1, and writes `ok` on the others.
    struct FailingDebug {
        calls: std::sync::atomic::AtomicUsize,
        fail_on: &'static [usize],
    }

    impl FailingDebug {
        fn new(fail_on: &'static [usize]) -> Self {
            Self {
                calls: std::sync::atomic::AtomicUsize::new(0),
                fail_on,
            }
        }
    }

    impl std::fmt::Debug for FailingDebug {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let call = self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1;
            if self.fail_on.contains(&call) {
                return Err(std::fmt::Error);
            }
            f.write_str("ok")
        }
    }

    /// Runs `body` under the default text subscriber and returns the line
    /// that holds the message `inside`.
    fn text_line(body: impl FnOnce()) -> String {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "INFO",
            "",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            body();
        }
        let output = captured(&buffer);
        output
            .lines()
            .find(|line| line.contains("inside"))
            .expect("the line is written")
            .to_owned()
    }

    /// When the `fmt` layer cannot format a span's fields, it stores none,
    /// and the text format still writes the span's records.
    #[test]
    fn default_text_writes_a_span_whose_fields_fail_to_format() {
        let line = text_line(|| {
            let value = FailingDebug::new(&[1]);
            let span = tracing::info_span!("phase", flaky = ?value, stage = "one");
            span.record("stage", "two");
            let _entered = span.enter();
            tracing::info!("inside");
        });
        assert!(line.contains("inside"), "{line}");
    }

    /// When a span field fails to format alone at the span's creation, the
    /// span keeps the fields the `fmt` layer formatted, and a later record
    /// does not rewrite them.
    #[test]
    fn default_text_keeps_span_fields_that_fail_to_format_alone() {
        let line = text_line(|| {
            let value = FailingDebug::new(&[2]);
            let span = tracing::info_span!("phase", flaky = ?value, stage = "one");
            span.record("stage", "two");
            let _entered = span.enter();
            tracing::info!("inside");
        });
        assert!(line.contains(r#"stage="one""#), "{line}");
        assert!(!line.contains(r#"stage="two""#), "{line}");
        assert!(line.contains("flaky=ok"), "{line}");
    }

    /// A recorded value that fails to format adds no text, and a later
    /// record still replaces its field in place.
    #[test]
    fn default_text_ignores_a_recorded_value_that_fails_to_format() {
        let line = text_line(|| {
            let span = tracing::info_span!("phase", stage = "one", late = tracing::field::Empty);
            span.record("stage", "two");
            span.record("late", tracing::field::debug(FailingDebug::new(&[1])));
            span.record("stage", "three");
            let _entered = span.enter();
            tracing::info!("inside");
        });
        assert!(line.contains(r#"stage="three""#), "{line}");
        assert_eq!(line.matches("stage").count(), 1, "{line}");
        assert!(!line.contains("late"), "{line}");
    }

    /// A `WARN` threshold disables `INFO` records, and the SDK creates its
    /// replay-tracking spans at `INFO`. The default subscriber keeps those
    /// spans enabled, so a warning before the wait is written once.
    #[test]
    fn default_subscriber_suppresses_replay_above_the_span_level() {
        for level in ["WARN", "ERROR"] {
            let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let subscriber =
                default_subscriber(level, "", "", CaptureWriter(std::sync::Arc::clone(&buffer)));

            let invocations = run_logging_handler_with(subscriber, false, LogStyle::Warn);

            let output = captured(&buffer);
            assert!(invocations >= 2, "the wait must suspend: {invocations}");
            let expected = usize::from(level == "WARN");
            assert_eq!(
                count_message(&output, "before"),
                expected,
                "{level}: {output}"
            );
            assert_eq!(
                count_message(&output, "after"),
                expected,
                "{level}: {output}"
            );
        }
    }

    /// A directive that turns the SDK's target off still leaves the
    /// replay-tracking spans enabled, so an application record before the
    /// wait is written once.
    #[test]
    fn default_subscriber_suppresses_replay_when_the_sdk_target_is_off() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "INFO",
            "aws_durable_execution_sdk=off,app=info",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        let invocations = run_logging_handler_with(subscriber, false, LogStyle::AppTarget);

        let output = captured(&buffer);
        assert!(invocations >= 2, "the wait must suspend: {invocations}");
        assert_eq!(count_message(&output, "before"), 1, "{output}");
        assert_eq!(count_message(&output, "after"), 1, "{output}");
    }

    /// Valid directives replace the default level: a target they enable is
    /// written below the level, and a record they do not name is not.
    #[test]
    fn default_subscriber_applies_directives() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "WARN",
            "app=debug,[=bogus",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::debug!(target: "app", "app-line");
            tracing::warn!("other-line");
        };

        let output = captured(&buffer);
        assert_eq!(output.lines().count(), 1, "{output}");
        assert_eq!(count_message(&output, "app-line"), 1, "{output}");
    }

    /// When no directive is valid, the default level applies.
    #[test]
    fn default_subscriber_ignores_invalid_directives() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "WARN",
            "[=bogus",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::info!("info-line");
            tracing::warn!("warn-line");
        };

        let output = captured(&buffer);
        assert_eq!(output.lines().count(), 1, "{output}");
        assert_eq!(count_message(&output, "warn-line"), 1, "{output}");
    }

    /// A span directive enables records inside the span only. This needs
    /// the `EnvFilter` to see the span's creation, entry and exit.
    #[test]
    fn default_subscriber_applies_span_directives() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "WARN",
            "[outer]=debug",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::info_span!("outer").in_scope(|| tracing::debug!("inside-line"));
            tracing::debug!("outside-line");
        };

        let output = captured(&buffer);
        assert_eq!(output.lines().count(), 1, "{output}");
        assert_eq!(count_message(&output, "inside-line"), 1, "{output}");
    }

    /// A span field directive follows a value recorded after the span was
    /// created, which needs the `EnvFilter` to see the record.
    #[test]
    fn default_subscriber_tracks_span_records() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "WARN",
            "[outer{flag=true}]=debug",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let span = tracing::info_span!("outer", flag = false);
            span.in_scope(|| tracing::debug!("before-record"));
            span.record("flag", true);
            span.in_scope(|| tracing::debug!("after-record"));
        };

        let output = captured(&buffer);
        assert_eq!(output.lines().count(), 1, "{output}");
        assert_eq!(count_message(&output, "after-record"), 1, "{output}");
    }

    /// A span directive that names an SDK replay-tracking span enables the
    /// records inside it. This needs the `EnvFilter` to register the
    /// directive when `tracing` registers the span's callsite.
    #[test]
    fn default_subscriber_applies_span_directives_to_sdk_spans() {
        for (directives, which) in [
            ("[durable_execution]=debug", "execution"),
            ("[durable_operation{attempt=1}]=debug", "operation"),
        ] {
            let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let subscriber = default_subscriber(
                "WARN",
                directives,
                "",
                CaptureWriter(std::sync::Arc::clone(&buffer)),
            );

            {
                let _guard = test_tracing::install_test_subscriber(subscriber);
                let span = if which == "execution" {
                    execution_span("arn", "request", None, false)
                } else {
                    operation_span("arn", "request", None, "op", None, 1, false)
                };
                span.in_scope(|| tracing::debug!("inside-line"));
                tracing::debug!("outside-line");
            };

            let output = captured(&buffer);
            assert_eq!(output.lines().count(), 1, "{directives}: {output}");
            assert_eq!(
                count_message(&output, "inside-line"),
                1,
                "{directives}: {output}"
            );
        }
    }

    /// A filter that enables everything and counts the span closes it sees.
    struct CloseCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    impl<S> tracing_subscriber::layer::Filter<S> for CloseCounter {
        fn enabled(
            &self,
            _meta: &tracing::Metadata<'_>,
            _cx: &tracing_subscriber::layer::Context<'_, S>,
        ) -> bool {
            true
        }

        fn on_close(&self, _id: Id, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// `DefaultFilter` forwards a span close to the `EnvFilter`, which
    /// releases the state it keeps for the span.
    #[test]
    fn default_filter_forwards_span_close() {
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::layer::SubscriberExt as _;

        let closes = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let filter = DefaultFilter {
            env: CloseCounter(std::sync::Arc::clone(&closes)),
            replay: ReplayFilterLayer::new(),
        };
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::sink)
                .with_filter(filter),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            drop(tracing::info_span!("closed"));
        };

        assert_eq!(closes.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    /// A lookup over a fixed list of variables, for `LogSettings::from_lookup`.
    fn lookup_in(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let vars: Vec<(String, String)> = vars
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        move |name| {
            vars.iter()
                .find(|(var, _)| var == name)
                .map(|(_, value)| value.clone())
        }
    }

    fn settings(level: &str, directives: &str, format: &str) -> LogSettings {
        LogSettings {
            level: level.to_owned(),
            directives: directives.to_owned(),
            format: format.to_owned(),
        }
    }

    #[test]
    fn log_settings_default_to_info_text_and_no_directives() {
        assert_eq!(
            LogSettings::from_lookup(lookup_in(&[])),
            settings("INFO", "", "")
        );
    }

    #[test]
    fn log_settings_take_the_level_from_aws_lambda_log_level_first() {
        let lookup = lookup_in(&[("AWS_LAMBDA_LOG_LEVEL", "ERROR"), ("RUST_LOG", "debug")]);
        assert_eq!(
            LogSettings::from_lookup(lookup),
            settings("ERROR", "debug", "")
        );
    }

    #[test]
    fn log_settings_fall_back_to_rust_log_for_the_level() {
        let lookup = lookup_in(&[("RUST_LOG", "app=debug")]);
        assert_eq!(
            LogSettings::from_lookup(lookup),
            settings("app=debug", "app=debug", "")
        );
    }

    #[test]
    fn log_settings_read_the_format() {
        let lookup = lookup_in(&[("AWS_LAMBDA_LOG_FORMAT", "JSON")]);
        assert_eq!(
            LogSettings::from_lookup(lookup),
            settings("INFO", "", "JSON")
        );
    }

    /// The JSON branch of `default_subscriber` filters its layer through
    /// `ReplayFilterLayer` too, so the pre-wait record is written once.
    #[test]
    fn default_subscriber_json_format_writes_each_record_once_across_replay() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "INFO",
            "",
            "json",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        let invocations = run_logging_handler(subscriber, false);

        let output = captured(&buffer);
        assert!(invocations >= 2, "the wait must suspend: {invocations}");
        let messages: Vec<String> = output
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|record| {
                record
                    .pointer("/message")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        let count = |message: &str| messages.iter().filter(|m| *m == message).count();
        assert_eq!(count("before"), 1, "{output}");
        assert_eq!(count("after"), 1, "{output}");
    }

    #[test]
    fn default_subscriber_honors_level() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "WARN",
            "",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::info!("info-line");
            tracing::warn!("warn-line");
        };

        let output = captured(&buffer);
        assert_eq!(output.lines().count(), 1, "{output}");
        assert_eq!(count_message(&output, "warn-line"), 1, "{output}");
    }

    #[test]
    fn default_subscriber_unknown_level_falls_back_to_info() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "bogus",
            "",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::debug!("debug-line");
            tracing::info!("info-line");
        };

        let output = captured(&buffer);
        assert_eq!(output.lines().count(), 1, "{output}");
        assert_eq!(count_message(&output, "info-line"), 1, "{output}");
    }

    #[test]
    fn default_subscriber_json_format() {
        for format in ["JSON", "json"] {
            let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            let subscriber = default_subscriber(
                "INFO",
                "",
                format,
                CaptureWriter(std::sync::Arc::clone(&buffer)),
            );

            {
                let _guard = test_tracing::install_test_subscriber(subscriber);
                tracing::info!("json-line");
            };

            let output = captured(&buffer);
            assert_eq!(output.lines().count(), 1, "{format}: {output}");
            for line in output.lines() {
                let parsed: serde_json::Value =
                    serde_json::from_str(line).expect("each line is JSON");
                assert!(parsed.is_object(), "{format}: {line}");
            }
        }
    }

    #[test]
    fn default_subscriber_text_format() {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "INFO",
            "",
            "",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );

        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::info!("text-line");
        };

        let output = captured(&buffer);
        assert_eq!(output.lines().count(), 1, "{output}");
        for line in output.lines() {
            assert!(
                serde_json::from_str::<serde_json::Value>(line).is_err(),
                "a text line must not parse as JSON: {line}"
            );
        }
    }

    /// The two-layer subscriber of the `ReplayFilterLayer` rustdoc keeps
    /// every `audit` record during replay and writes every other record
    /// once per execution.
    #[test]
    fn kept_target_survives_replay() {
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::filter::{FilterExt as _, LevelFilter, Targets};
        use tracing_subscriber::layer::SubscriberExt as _;

        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(CaptureWriter(std::sync::Arc::clone(&buffer)))
                    .with_filter(Targets::new().with_target("audit", LevelFilter::TRACE)),
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(CaptureWriter(std::sync::Arc::clone(&buffer)))
                    .with_filter(
                        ReplayFilterLayer::new().and(
                            Targets::new()
                                .with_default(LevelFilter::TRACE)
                                .with_target("audit", LevelFilter::OFF),
                        ),
                    ),
            );

        let invocations = run_logging_handler(subscriber, true);

        let output = captured(&buffer);
        assert!(invocations >= 2, "the wait must suspend: {invocations}");
        assert_eq!(count_message(&output, "audit-before"), 2, "{output}");
        assert_eq!(count_message(&output, "before"), 1, "{output}");
    }

    // Correlation fields on the spans of user-code bodies.

    /// Builds a JSON `fmt` subscriber that writes each record's current
    /// span and span list into `buffer`.
    pub(crate) fn json_span_subscriber(
        buffer: &std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    ) -> impl tracing::Subscriber + Send + Sync + 'static {
        tracing_subscriber::fmt()
            .json()
            .flatten_event(true)
            .with_current_span(true)
            .with_span_list(true)
            .with_writer(CaptureWriter(std::sync::Arc::clone(buffer)))
            .finish()
    }

    /// Parses the JSON records in `buffer`, one per line.
    pub(crate) fn json_records(buffer: &std::sync::Mutex<Vec<u8>>) -> Vec<serde_json::Value> {
        captured(buffer)
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    /// The span list of the first record whose message is `message`.
    pub(crate) fn spans_of(records: &[serde_json::Value], message: &str) -> Vec<serde_json::Value> {
        let record = records
            .iter()
            .find(|r| r.get("message").and_then(serde_json::Value::as_str) == Some(message))
            .expect("the record is in the captured output");
        record
            .get("spans")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default()
    }

    /// Whether one span of `spans` carries `field` with the value `value`.
    pub(crate) fn has_field(
        spans: &[serde_json::Value],
        field: &str,
        value: &serde_json::Value,
    ) -> bool {
        spans.iter().any(|span| span.get(field) == Some(value))
    }

    /// Whether any span of `spans` carries `field`.
    pub(crate) fn any_field(spans: &[serde_json::Value], field: &str) -> bool {
        spans.iter().any(|span| span.get(field).is_some())
    }

    /// Runs `handler` through `LocalRunner` under a JSON span subscriber,
    /// on a current-thread runtime, and returns the records and the result.
    fn run_capturing<F, Fut>(
        runner: crate::test_util::LocalRunner,
        handler: F,
    ) -> (Vec<serde_json::Value>, crate::test_util::TestResult<()>)
    where
        F: Fn(serde_json::Value, crate::DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), crate::BoxError>> + Send,
    {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = json_span_subscriber(&buffer);
        let result = {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                runner
                    .run(handler, serde_json::Value::Null)
                    .await
                    .expect("runner")
            })
        };
        assert!(result.is_success(), "{:?}", result.error());
        (json_records(&buffer), result)
    }

    fn json_str(value: &str) -> serde_json::Value {
        serde_json::Value::String(value.to_owned())
    }

    #[test]
    fn child_body_record_has_operation_id_and_name() {
        let (records, result) = run_capturing(
            crate::test_util::LocalRunner::new(),
            |_event, ctx| async move {
                ctx.run_in_child_context(|_child| async move {
                    tracing::info!("in-child");
                    Ok(())
                })
                .name("enrich")
                .await?;
                Ok(())
            },
        );
        let child = result.operation("enrich").expect("child operation");
        let spans = spans_of(&records, "in-child");
        assert!(
            has_field(&spans, fields::OPERATION_ID, &json_str(child.id())),
            "{spans:?}"
        );
        assert!(
            has_field(&spans, fields::OPERATION_NAME, &json_str("enrich")),
            "{spans:?}"
        );
    }

    #[test]
    fn map_item_record_has_operation_id() {
        let (records, result) = run_capturing(
            crate::test_util::LocalRunner::new(),
            |_event, ctx| async move {
                ctx.map(vec![7_u32], |_child, _item, _idx| async move {
                    tracing::info!("in-item");
                    Ok(())
                })
                .name("items")
                .await?;
                Ok(())
            },
        );
        let batch = result.operation("items").expect("map operation");
        let item = result
            .children(batch)
            .find(|op| op.kind() == crate::OperationKind::Context)
            .expect("item child");
        // The map is "1", and its items are numbered inside it.
        assert_eq!(item.id(), crate::engine::compute_wire_id_public("1-1"));
        let spans = spans_of(&records, "in-item");
        assert!(
            has_field(&spans, fields::OPERATION_ID, &json_str(item.id())),
            "{spans:?}"
        );
    }

    /// A flat map item has no child record of its own, but its span still
    /// carries the virtual child's wire ID and the item's recorded name.
    /// The virtual child has the same ID a normal item has, `1-1`, so the
    /// expected ID comes from a normal run of the same handler.
    #[test]
    fn flat_map_item_record_has_operation_id_and_name() {
        type Boxed = std::pin::Pin<Box<dyn Future<Output = Result<(), crate::BoxError>> + Send>>;

        fn handler(
            nesting: crate::map_parallel::NestingMode,
        ) -> impl Fn(serde_json::Value, crate::DurableContext) -> Boxed + Send + Sync + 'static
        {
            move |_event, ctx| {
                Box::pin(async move {
                    ctx.map(vec![7_u32], |_child, _item, _idx| async move {
                        tracing::info!("in-item");
                        Ok(())
                    })
                    .name("items")
                    .nesting(nesting)
                    .await?;
                    Ok(())
                })
            }
        }
        let (_records, normal) = run_capturing(
            crate::test_util::LocalRunner::new(),
            handler(crate::map_parallel::NestingMode::Normal),
        );
        let batch = normal.operation("items").expect("map operation");
        let item = normal
            .children(batch)
            .find(|op| op.kind() == crate::OperationKind::Context)
            .expect("item child");
        assert_eq!(item.id(), crate::engine::compute_wire_id_public("1-1"));

        let (records, _flat) = run_capturing(
            crate::test_util::LocalRunner::new(),
            handler(crate::map_parallel::NestingMode::Flat),
        );
        let spans = spans_of(&records, "in-item");
        assert!(
            has_field(&spans, fields::OPERATION_ID, &json_str(item.id())),
            "{spans:?}"
        );
        assert!(
            has_field(&spans, fields::OPERATION_NAME, &json_str("map-item-0")),
            "{spans:?}"
        );
    }

    #[test]
    fn parallel_branch_record_has_operation_id_and_name() {
        let (records, result) = run_capturing(
            crate::test_util::LocalRunner::new(),
            |_event, ctx| async move {
                let branches: Vec<crate::Branch<()>> =
                    vec![crate::Branch::new("left", |_child| async move {
                        tracing::info!("in-branch");
                        Ok(())
                    })];
                ctx.parallel(branches).name("fan-out").await?;
                Ok(())
            },
        );
        let branch = result.operation("left").expect("branch operation");
        // The parallel is "1", and its branches are numbered inside it.
        assert_eq!(branch.id(), crate::engine::compute_wire_id_public("1-1"));
        let spans = spans_of(&records, "in-branch");
        assert!(
            has_field(&spans, fields::OPERATION_ID, &json_str(branch.id())),
            "{spans:?}"
        );
        assert!(
            has_field(&spans, fields::OPERATION_NAME, &json_str("left")),
            "{spans:?}"
        );
    }

    #[test]
    fn condition_check_record_has_operation_id_and_attempt() {
        let (records, result) = run_capturing(
            crate::test_util::LocalRunner::new(),
            |_event, ctx| async move {
                ctx.wait_for_condition(
                    |_step, state: u32| async move {
                        tracing::info!("in-check");
                        Ok(state + 1)
                    },
                    0,
                )
                .wait_strategy_fn(|_state, _attempt| {
                    crate::builders::wait_for_condition::WaitDecision::complete()
                })
                .name("poll")
                .await?;
                Ok(())
            },
        );
        let poll = result.operation("poll").expect("condition operation");
        let spans = spans_of(&records, "in-check");
        assert!(
            has_field(&spans, fields::OPERATION_ID, &json_str(poll.id())),
            "{spans:?}"
        );
        assert!(
            has_field(&spans, fields::OPERATION_NAME, &json_str("poll")),
            "{spans:?}"
        );
        assert!(
            has_field(&spans, fields::ATTEMPT, &serde_json::json!(1)),
            "{spans:?}"
        );
    }

    #[test]
    fn submitter_record_has_operation_id() {
        let reply = crate::test_util::CallbackReply::success(&"approved").expect("reply");
        let (records, result) = run_capturing(
            crate::test_util::LocalRunner::new().callback_for("approve", reply),
            |_event, ctx| async move {
                let _value: String = ctx
                    .wait_for_callback(|_step, _callback_id| async move {
                        tracing::info!("in-submitter");
                        Ok(())
                    })
                    .name("approve")
                    .await?;
                Ok(())
            },
        );
        let wfcb = result
            .operation("approve")
            .expect("wait_for_callback operation");
        let submitter = result
            .children(wfcb)
            .find(|op| op.kind() == crate::OperationKind::Step)
            .expect("submitter step");
        let spans = spans_of(&records, "in-submitter");
        assert!(
            has_field(&spans, fields::OPERATION_ID, &json_str(submitter.id())),
            "{spans:?}"
        );
    }

    #[test]
    fn named_step_record_has_operation_name() {
        let (records, _result) = run_capturing(
            crate::test_util::LocalRunner::new(),
            |_event, ctx| async move {
                ctx.step(|_| async {
                    tracing::info!("in-step");
                    Ok(())
                })
                .name("fetch")
                .await?;
                Ok(())
            },
        );
        let spans = spans_of(&records, "in-step");
        assert!(
            has_field(&spans, fields::OPERATION_NAME, &json_str("fetch")),
            "{spans:?}"
        );
    }

    #[test]
    fn unnamed_step_record_has_no_operation_name() {
        let (records, _result) = run_capturing(
            crate::test_util::LocalRunner::new(),
            |_event, ctx| async move {
                ctx.step(|_| async {
                    tracing::info!("in-step");
                    Ok(())
                })
                .await?;
                Ok(())
            },
        );
        let spans = spans_of(&records, "in-step");
        assert!(any_field(&spans, fields::OPERATION_ID), "{spans:?}");
        assert!(!any_field(&spans, fields::OPERATION_NAME), "{spans:?}");
    }

    /// The messages a capture filtered by [`ReplayFilterLayer`] wrote.
    fn filtered_messages(run: impl FnOnce()) -> Vec<String> {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::layer::SubscriberExt as _;

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_writer(CaptureWriter(Arc::clone(&buffer)))
            .with_filter(ReplayFilterLayer::new());
        {
            let _guard =
                test_tracing::install_test_subscriber(tracing_subscriber::registry().with(layer));
            run();
        }
        let output = String::from_utf8_lossy(&buffer.lock().expect("not poisoned")).to_string();
        output
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter_map(|v| v.get("message")?.as_str().map(str::to_owned))
            .collect()
    }

    /// A record without its own `isReplay` field follows its nearest
    /// `durable_execution` span and the SDK spans inside it; a parent
    /// namespace and a non-SDK span decide nothing.
    #[test]
    fn replay_filter_uses_the_nearest_execution_namespace() {
        let emitted = filtered_messages(|| {
            let child = |parent: &tracing::Span, replay: bool| {
                let _entered = parent.enter();
                scoped_execution_span("arn", "req", None, "child", None, replay)
            };
            let parent = execution_span("arn", "req", None, true);
            let live_child = child(&parent, false);
            live_child.in_scope(|| {
                tracing::info!("parent replay, child live");
                tracing::info!(
                    { fields::IS_REPLAY } = true,
                    "explicit replay in live child"
                );
                tracing::info_span!("not_sdk", isReplay = true)
                    .in_scope(|| tracing::info!("non-SDK replay span in live child"));
                operation_span("arn", "req", None, "op", None, 1, true)
                    .in_scope(|| tracing::info!("replaying operation in live child"));
            });
            let parent = execution_span("arn", "req", None, false);
            child(&parent, true).in_scope(|| {
                tracing::info!("parent live, child replay");
                tracing::info!(
                    { fields::IS_REPLAY } = false,
                    "explicit live in replaying child"
                );
                operation_span("arn", "req", None, "op", None, 1, false)
                    .in_scope(|| tracing::info!("live operation in replaying child"));
                // Another target's span of the same name is no namespace.
                tracing::info_span!("durable_execution", isReplay = false)
                    .in_scope(|| tracing::info!("foreign namespace in replaying child"));
            });
        });
        assert_eq!(
            emitted,
            [
                "parent replay, child live",
                "non-SDK replay span in live child",
                "explicit live in replaying child",
            ]
        );
    }

    /// The filter on two layers of one subscriber: both store the replay
    /// state of an SDK span in the span's shared extensions, so the second
    /// store must not panic, and both layers follow the span's later
    /// `isReplay` updates.
    #[test]
    fn replay_filter_on_two_layers_shares_span_state() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::layer::SubscriberExt as _;

        let layer = |buffer: &Arc<Mutex<Vec<u8>>>| {
            tracing_subscriber::fmt::layer()
                .json()
                .flatten_event(true)
                .with_writer(CaptureWriter(Arc::clone(buffer)))
                .with_filter(ReplayFilterLayer::new())
                .boxed()
        };
        let (first, second) = (Arc::default(), Arc::default());
        {
            let _guard = test_tracing::install_test_subscriber(
                tracing_subscriber::registry().with(vec![layer(&first), layer(&second)]),
            );
            let span = execution_span("arn", "req", None, true);
            span.in_scope(|| tracing::info!("while replaying"));
            span.record(fields::IS_REPLAY, false);
            span.in_scope(|| tracing::info!("after replay"));
        }
        for buffer in [first, second] {
            let output = String::from_utf8_lossy(&buffer.lock().expect("not poisoned")).to_string();
            let messages: Vec<String> = output
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .filter_map(|v| v.get("message")?.as_str().map(str::to_owned))
                .collect();
            assert_eq!(messages, ["after replay"], "{output}");
        }
    }

    /// A child namespace's span is a child of the span current when it is
    /// created, and keeps its fields.
    #[test]
    #[expect(clippy::indexing_slicing, reason = "test assertions on JSON records")]
    fn child_execution_span_is_a_child_of_the_current_span() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::layer::SubscriberExt as _;

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_writer(CaptureWriter(Arc::clone(&buffer)));
        {
            let _guard =
                test_tracing::install_test_subscriber(tracing_subscriber::registry().with(layer));
            let current = tracing::info_span!("current");
            let child = current.in_scope(|| {
                scoped_execution_span("arn", "req", Some("tenant"), "op-1", Some("named"), true)
            });
            let unnamed =
                current.in_scope(|| scoped_execution_span("arn", "req", None, "op-2", None, false));
            // Entered outside `current`: only the stored parent links them.
            child.in_scope(|| tracing::info!("in child"));
            unnamed.in_scope(|| tracing::info!("in unnamed child"));
        }
        let output = String::from_utf8_lossy(&buffer.lock().expect("not poisoned")).to_string();
        let records: Vec<serde_json::Value> = output
            .lines()
            .map(|line| serde_json::from_str(line).expect("JSON"))
            .collect();
        let [named, unnamed] = records.as_slice() else {
            unreachable!("two records: {output}");
        };
        let child_of = |record: &serde_json::Value| {
            let spans = record["spans"].as_array().expect("a span list").clone();
            let chain: Vec<&str> = spans.iter().filter_map(|s| s["name"].as_str()).collect();
            assert_eq!(chain, ["current", "durable_execution"], "{record}");
            spans[1].clone()
        };
        let child = child_of(named);
        assert_eq!(child[fields::EXECUTION_ARN], "arn", "{child}");
        assert_eq!(child[fields::REQUEST_ID], "req", "{child}");
        assert_eq!(child[fields::TENANT_ID], "tenant", "{child}");
        assert_eq!(child[fields::OPERATION_ID], "op-1", "{child}");
        assert_eq!(child[fields::OPERATION_NAME], "named", "{child}");
        assert_eq!(child[fields::IS_REPLAY], true, "{child}");
        let child = child_of(unnamed);
        assert_eq!(child[fields::OPERATION_ID], "op-2", "{child}");
        assert_eq!(child[fields::IS_REPLAY], false, "{child}");
        assert!(child.get(fields::TENANT_ID).is_none(), "{child}");
        assert!(child.get(fields::OPERATION_NAME).is_none(), "{child}");
    }

    /// Only the SDK's `durable_execution` and `durable_operation` spans
    /// carry the replay state. A span of the SDK's target with another name
    /// neither suppresses a record nor bounds a namespace, and a non-SDK
    /// span that records `isReplay = true` after its creation decides
    /// nothing.
    #[test]
    fn replay_filter_ignores_other_spans_of_the_sdk_target() {
        let emitted = filtered_messages(|| {
            execution_span("arn", "req", None, false).in_scope(|| {
                tracing::info_span!(target: SPAN_TARGET, "not_a_namespace", isReplay = true)
                    .in_scope(|| tracing::info!("sdk-target replay span in live namespace"));
                let late = tracing::info_span!("not_sdk", isReplay = tracing::field::Empty);
                late.record(fields::IS_REPLAY, true);
                late.in_scope(|| tracing::info!("non-SDK span recorded as replaying"));
            });
            execution_span("arn", "req", None, true).in_scope(|| {
                tracing::info_span!(target: SPAN_TARGET, "not_an_operation", isReplay = false)
                    .in_scope(|| tracing::info!("sdk-target live span in replaying namespace"));
            });
        });
        assert_eq!(
            emitted,
            [
                "sdk-target replay span in live namespace",
                "non-SDK span recorded as replaying",
            ]
        );
    }

    /// An `EnvFilter` stand-in that enables only the `probe` target.
    struct OnlyProbe;

    impl<S> tracing_subscriber::layer::Filter<S> for OnlyProbe {
        fn enabled(
            &self,
            meta: &tracing::Metadata<'_>,
            _cx: &tracing_subscriber::layer::Context<'_, S>,
        ) -> bool {
            meta.target() == "probe"
        }

        fn callsite_enabled(
            &self,
            meta: &'static tracing::Metadata<'static>,
        ) -> tracing::subscriber::Interest {
            if meta.target() == "probe" {
                tracing::subscriber::Interest::sometimes()
            } else {
                tracing::subscriber::Interest::never()
            }
        }
    }

    /// `DefaultFilter` enables the SDK's replay-tracking spans that its
    /// `EnvFilter` disables, and no other span of the SDK's target.
    #[test]
    fn default_filter_enables_only_the_replay_tracking_spans() {
        use std::sync::{Arc, Mutex};
        use tracing_subscriber::Layer as _;
        use tracing_subscriber::layer::{Filter, SubscriberExt as _};

        let buffer = Arc::new(Mutex::new(Vec::new()));
        let filter = || DefaultFilter {
            env: OnlyProbe,
            replay: ReplayFilterLayer::new(),
        };
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_writer(CaptureWriter(Arc::clone(&buffer)))
            .with_filter(filter());
        let (other, execution) = {
            let _guard =
                test_tracing::install_test_subscriber(tracing_subscriber::registry().with(layer));
            let other = tracing::info_span!(target: SPAN_TARGET, "not_a_namespace");
            other.in_scope(|| tracing::info!(target: "probe", "in other span"));
            let execution = execution_span("arn", "req", None, false);
            execution.in_scope(|| tracing::info!(target: "probe", "in execution span"));
            (
                other.metadata().expect("the registry enables every span"),
                execution
                    .metadata()
                    .expect("the registry enables every span"),
            )
        };

        let records = json_records(&buffer);
        let names = |message: &str| -> Vec<String> {
            spans_of(&records, message)
                .iter()
                .filter_map(|span| span.get("name")?.as_str().map(str::to_owned))
                .collect()
        };
        assert!(names("in other span").is_empty(), "{records:?}");
        assert_eq!(
            names("in execution span"),
            ["durable_execution"],
            "{records:?}"
        );

        let interest = |meta| {
            <DefaultFilter<OnlyProbe> as Filter<tracing_subscriber::Registry>>::callsite_enabled(
                &filter(),
                meta,
            )
        };
        assert!(interest(other).is_never());
        assert!(interest(execution).is_sometimes());
    }

    // The default subscriber's JSON record.

    /// One JSON record with its top-level keys in the order they were
    /// written. Parsing it rejects a key that occurs twice.
    #[derive(Debug)]
    struct OrderedRecord(Vec<(String, serde_json::Value)>);

    impl<'de> serde::Deserialize<'de> for OrderedRecord {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct RecordVisitor;

            impl<'de> serde::de::Visitor<'de> for RecordVisitor {
                type Value = OrderedRecord;

                fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str("a JSON object")
                }

                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> Result<OrderedRecord, A::Error> {
                    let mut entries: Vec<(String, serde_json::Value)> = Vec::new();
                    while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                        if entries.iter().any(|(seen, _)| *seen == key) {
                            return Err(serde::de::Error::custom(format!("repeated key {key}")));
                        }
                        entries.push((key, value));
                    }
                    Ok(OrderedRecord(entries))
                }
            }

            deserializer.deserialize_map(RecordVisitor)
        }
    }

    impl OrderedRecord {
        fn keys(&self) -> Vec<&str> {
            self.0.iter().map(|(key, _)| key.as_str()).collect()
        }

        fn get(&self, key: &str) -> Option<&serde_json::Value> {
            self.0
                .iter()
                .find(|(seen, _)| seen == key)
                .map(|(_, value)| value)
        }
    }

    /// Parses every captured line as an [`OrderedRecord`], failing on a line
    /// that is not one.
    fn ordered_records(buffer: &std::sync::Mutex<Vec<u8>>) -> Vec<OrderedRecord> {
        captured(buffer)
            .lines()
            .map(|line| {
                serde_json::from_str(line).expect("each line is one record with unique keys")
            })
            .collect()
    }

    /// The first record whose `message` is `message`.
    fn record<'a>(records: &'a [OrderedRecord], message: &str) -> &'a OrderedRecord {
        records
            .iter()
            .find(|r| r.get("message") == Some(&json_str(message)))
            .expect("the record is in the captured output")
    }

    /// Whether `value` is a string holding a valid UTC time of the form
    /// `YYYY-MM-DDTHH:MM:SS.sssZ`.
    fn is_js_timestamp(value: &serde_json::Value) -> bool {
        let Some(text) = value.as_str() else {
            return false;
        };
        let shape_ok = text.len() == 24
            && text.char_indices().all(|(i, c)| match i {
                4 | 7 => c == '-',
                10 => c == 'T',
                13 | 16 => c == ':',
                19 => c == '.',
                23 => c == 'Z',
                _ => c.is_ascii_digit(),
            });
        let number = |range: std::ops::Range<usize>| -> Option<u32> {
            text.get(range).and_then(|digits| digits.parse().ok())
        };
        let (Some(year), Some(month), Some(day)) = (number(0..4), number(5..7), number(8..10))
        else {
            return false;
        };
        let (Some(hour), Some(minute), Some(second)) =
            (number(11..13), number(14..16), number(17..19))
        else {
            return false;
        };
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let month_days = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 if leap => 29,
            2 => 28,
            _ => return false,
        };
        shape_ok && (1..=month_days).contains(&day) && hour <= 23 && minute <= 59 && second <= 59
    }

    #[test]
    fn is_js_timestamp_accepts_only_valid_times() {
        for valid in [
            "1970-01-01T00:00:00.000Z",
            "2024-02-29T23:59:59.999Z",
            "9999-12-31T23:59:59.999Z",
        ] {
            assert!(is_js_timestamp(&json_str(valid)), "{valid}");
        }
        for invalid in [
            "9999-99-99T99:99:99.999Z",
            "2023-02-29T00:00:00.000Z",
            "2024-13-01T00:00:00.000Z",
            "2024-04-31T00:00:00.000Z",
            "2024-01-00T00:00:00.000Z",
            "2024-01-01T24:00:00.000Z",
            "2024-01-01T00:60:00.000Z",
            "2024-01-01T00:00:60.000Z",
            "2024-01-01T00:00:00.000+00:00",
            "2024-01-01 00:00:00.000Z",
        ] {
            assert!(!is_js_timestamp(&json_str(invalid)), "{invalid}");
        }
        assert!(!is_js_timestamp(&serde_json::json!(0)));
    }

    /// The JSON default subscriber at `INFO` plus `directives`, writing
    /// into a new buffer.
    fn default_json(
        directives: &str,
    ) -> (
        std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
        Box<dyn tracing::Subscriber + Send + Sync>,
    ) {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = default_subscriber(
            "INFO",
            directives,
            "json",
            CaptureWriter(std::sync::Arc::clone(&buffer)),
        );
        (buffer, subscriber)
    }

    /// Runs `handler` through `runner` under the JSON default subscriber
    /// with `directives`, on a current-thread runtime. Returns the records,
    /// the result and the run's execution ARN.
    fn run_default_json<F, Fut>(
        runner: crate::test_util::LocalRunner,
        directives: &str,
        handler: F,
    ) -> (Vec<OrderedRecord>, crate::test_util::TestResult<()>, String)
    where
        F: Fn(serde_json::Value, crate::DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), crate::BoxError>> + Send,
    {
        let (buffer, subscriber) = default_json(directives);
        let arn = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let seen = std::sync::Arc::clone(&arn);
        let handler = std::sync::Arc::new(handler);
        let result = {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                runner
                    .run(
                        move |event, ctx: crate::DurableContext| {
                            *seen.lock().expect("arn") = ctx.execution_arn().to_owned();
                            handler(event, ctx)
                        },
                        serde_json::Value::Null,
                    )
                    .await
                    .expect("runner")
            })
        };
        assert!(result.is_success(), "{:?}", result.error());
        let arn = arn.lock().expect("arn").clone();
        assert_ne!(arn, "");
        (ordered_records(&buffer), result, arn)
    }

    /// The keys a record of an operation's user code has, before its
    /// ordinary fields and `message`.
    const OPERATION_KEYS: [&str; 7] = [
        "requestId",
        "timestamp",
        "level",
        "executionArn",
        "operationId",
        "operationName",
        "attempt",
    ];

    #[test]
    fn default_json_step_record_has_top_level_fields() {
        let (records, result, arn) = run_default_json(
            crate::test_util::LocalRunner::new(),
            "",
            |_event, ctx| async move {
                ctx.step(|_| async {
                    tracing::info!(order = 5_u32, "in-step");
                    Ok(())
                })
                .name("fetch")
                .await?;
                Ok(())
            },
        );
        let step = result.operation("fetch").expect("step operation");
        let record = record(&records, "in-step");
        let mut expected = OPERATION_KEYS.to_vec();
        expected.extend(["order", "message"]);
        assert_eq!(record.keys(), expected, "{record:?}");
        assert_eq!(
            record.get("requestId"),
            Some(&json_str("local-invocation-1"))
        );
        assert!(is_js_timestamp(record.get("timestamp").expect("timestamp")));
        assert_eq!(record.get("level"), Some(&json_str("INFO")));
        assert_eq!(record.get("executionArn"), Some(&json_str(&arn)));
        assert_eq!(record.get("operationId"), Some(&json_str(step.id())));
        assert_eq!(record.get("operationName"), Some(&json_str("fetch")));
        assert_eq!(record.get("attempt"), Some(&serde_json::json!(1)));
        assert_eq!(record.get("order"), Some(&serde_json::json!(5)));
        assert_eq!(record.get("message"), Some(&json_str("in-step")));
    }

    #[test]
    fn default_json_tenant_id_is_optional_and_ordered() {
        let (buffer, subscriber) = default_json("");
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            execution_span("arn:run", "req-1", Some("tenant-1"), false)
                .in_scope(|| tracing::info!("with-tenant"));
            execution_span("arn:run", "req-1", None, false)
                .in_scope(|| tracing::info!("without-tenant"));
        }
        let records = ordered_records(&buffer);
        let with = record(&records, "with-tenant");
        assert_eq!(
            with.keys(),
            [
                "requestId",
                "timestamp",
                "level",
                "executionArn",
                "tenantId",
                "message"
            ],
            "{with:?}"
        );
        assert_eq!(with.get("tenantId"), Some(&json_str("tenant-1")));
        let without = record(&records, "without-tenant");
        assert_eq!(
            without.keys(),
            ["requestId", "timestamp", "level", "executionArn", "message"],
            "{without:?}"
        );
    }

    #[test]
    fn default_json_child_record_has_top_level_fields() {
        let (records, result, arn) = run_default_json(
            crate::test_util::LocalRunner::new(),
            "",
            |_event, ctx| async move {
                ctx.run_in_child_context(|_child| async move {
                    tracing::info!("in-child");
                    Ok(())
                })
                .name("enrich")
                .await?;
                Ok(())
            },
        );
        let child = result.operation("enrich").expect("child operation");
        let record = record(&records, "in-child");
        assert_eq!(
            record.keys(),
            [
                "requestId",
                "timestamp",
                "level",
                "executionArn",
                "operationId",
                "operationName",
                "message"
            ],
            "{record:?}"
        );
        assert_eq!(
            record.get("requestId"),
            Some(&json_str("local-invocation-1"))
        );
        assert_eq!(record.get("executionArn"), Some(&json_str(&arn)));
        assert_eq!(record.get("operationId"), Some(&json_str(child.id())));
        assert_eq!(record.get("operationName"), Some(&json_str("enrich")));
    }

    #[test]
    fn default_json_map_item_record_has_top_level_fields() {
        let (records, result, arn) = run_default_json(
            crate::test_util::LocalRunner::new(),
            "",
            |_event, ctx| async move {
                ctx.map(vec![7_u32], |_child, _item, _idx| async move {
                    tracing::info!("in-item");
                    Ok(())
                })
                .name("items")
                .await?;
                Ok(())
            },
        );
        let batch = result.operation("items").expect("map operation");
        let item = result
            .children(batch)
            .find(|op| op.kind() == crate::OperationKind::Context)
            .expect("item child");
        let record = record(&records, "in-item");
        assert_eq!(
            record.keys(),
            [
                "requestId",
                "timestamp",
                "level",
                "executionArn",
                "operationId",
                "operationName",
                "message"
            ],
            "{record:?}"
        );
        assert_eq!(
            record.get("requestId"),
            Some(&json_str("local-invocation-1"))
        );
        assert_eq!(record.get("executionArn"), Some(&json_str(&arn)));
        assert_eq!(record.get("operationId"), Some(&json_str(item.id())));
        assert_eq!(record.get("operationName"), Some(&json_str("map-item-0")));
    }

    #[test]
    fn default_json_step_in_child_uses_nearest_sdk_span() {
        let (records, result, _arn) = run_default_json(
            crate::test_util::LocalRunner::new(),
            "",
            |_event, ctx| async move {
                ctx.run_in_child_context(|child| async move {
                    child
                        .step(|_| async {
                            tracing::info!("in-inner-step");
                            Ok(())
                        })
                        .name("inner")
                        .await?;
                    tracing::info!("in-outer-child");
                    Ok(())
                })
                .name("outer")
                .await?;
                Ok(())
            },
        );
        let outer = result.operation("outer").expect("child operation");
        let inner = result.operation("inner").expect("step operation");
        let in_step = record(&records, "in-inner-step");
        assert_eq!(in_step.get("operationId"), Some(&json_str(inner.id())));
        assert_eq!(in_step.get("operationName"), Some(&json_str("inner")));
        assert_eq!(in_step.get("attempt"), Some(&serde_json::json!(1)));
        let in_child = record(&records, "in-outer-child");
        assert_eq!(in_child.get("operationId"), Some(&json_str(outer.id())));
        assert_eq!(in_child.get("operationName"), Some(&json_str("outer")));
        assert_eq!(in_child.get("attempt"), None, "{in_child:?}");
    }

    /// A span that is not the SDK's, though it has the name of the SDK's
    /// operation span and carries correlation keys.
    fn impostor_span() -> tracing::Span {
        tracing::info_span!(
            "durable_operation",
            requestId = "impostor-request",
            operationId = "impostor-operation",
            trace_id = "trace-1",
        )
    }

    /// A plugin whose user-function wrap enters [`impostor_span`].
    #[derive(Clone)]
    struct Impostor;

    impl crate::plugin::Plugin for Impostor {
        fn wrap_user_function(&self, _info: &crate::plugin::OperationInfo, next: &mut dyn FnMut()) {
            impostor_span().in_scope(next);
        }
    }

    impl crate::plugin::PluginFactory for Impostor {
        fn create_plugin(
            &self,
            _info: &crate::plugin::InvocationInfo,
        ) -> Result<Option<Box<dyn crate::plugin::Plugin>>, crate::BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    #[test]
    fn default_json_correlation_uses_only_events_and_sdk_spans() {
        let options = crate::Options::builder()
            .plugin(Impostor)
            .build()
            .expect("options");
        let (records, result, _arn) = run_default_json(
            crate::test_util::LocalRunner::new().options(options),
            "",
            |_event, ctx| async move {
                ctx.step(|_| async {
                    tracing::info!("in-step");
                    tracing::info!(
                        requestId = "event-request",
                        operationId = "event-operation",
                        "explicit"
                    );
                    Ok(())
                })
                .name("fetch")
                .await?;
                Ok(())
            },
        );
        let step = result.operation("fetch").expect("step operation");
        let in_step = record(&records, "in-step");
        assert_eq!(
            in_step.get("trace_id"),
            Some(&json_str("trace-1")),
            "{in_step:?}"
        );
        assert_eq!(
            in_step.get("requestId"),
            Some(&json_str("local-invocation-1"))
        );
        assert_eq!(in_step.get("operationId"), Some(&json_str(step.id())));
        let explicit = record(&records, "explicit");
        assert_eq!(explicit.get("requestId"), Some(&json_str("event-request")));
        assert_eq!(
            explicit.get("operationId"),
            Some(&json_str("event-operation"))
        );
        assert_eq!(explicit.get("trace_id"), Some(&json_str("trace-1")));

        let (buffer, subscriber) = default_json("");
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            impostor_span().in_scope(|| tracing::info!("outside"));
        }
        let records = ordered_records(&buffer);
        let outside = record(&records, "outside");
        assert_eq!(
            outside.keys(),
            ["timestamp", "level", "trace_id", "message"],
            "{outside:?}"
        );
    }

    #[test]
    fn default_json_field_collisions_have_one_winner() {
        let (buffer, subscriber) = default_json("");
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let outer = tracing::info_span!(
                "outer",
                shared = "outer",
                nested = "outer",
                outer_only = "outer",
                "timestamp" = "span-timestamp",
                "level" = "span-level",
                "message" = "span-message",
            );
            outer.in_scope(|| {
                // Created inside `outer`, so `outer` is its parent.
                let inner = tracing::info_span!(
                    "inner",
                    shared = "inner",
                    nested = "inner",
                    isReplay = false,
                );
                inner.in_scope(|| {
                    tracing::info!(
                        shared = "event",
                        "timestamp" = "event-timestamp",
                        "level" = "event-level",
                        isReplay = false,
                        "fields" = "user-fields",
                        "span" = "user-span",
                        "spans" = "user-spans",
                        "target" = "user-target",
                        "name" = "user-name",
                        "collide"
                    );
                });
            });
        }
        let records = ordered_records(&buffer);
        let record = record(&records, "collide");
        let keys = record.keys();
        assert_eq!(keys.first(), Some(&"timestamp"), "{record:?}");
        assert_eq!(keys.get(1), Some(&"level"), "{record:?}");
        assert_eq!(keys.last(), Some(&"message"), "{record:?}");
        assert!(is_js_timestamp(record.get("timestamp").expect("timestamp")));
        assert_eq!(record.get("level"), Some(&json_str("INFO")));
        assert_eq!(record.get("message"), Some(&json_str("collide")));
        assert_eq!(record.get("shared"), Some(&json_str("event")));
        assert_eq!(record.get("nested"), Some(&json_str("inner")));
        assert_eq!(
            record.get("outer_only"),
            Some(&json_str("outer")),
            "{record:?}"
        );
        assert_eq!(record.get("isReplay"), None, "{record:?}");
        for (key, value) in [
            ("fields", "user-fields"),
            ("span", "user-span"),
            ("spans", "user-spans"),
            ("target", "user-target"),
            ("name", "user-name"),
        ] {
            assert_eq!(keys.iter().filter(|k| **k == key).count(), 1, "{key}");
            assert_eq!(record.get(key), Some(&json_str(value)), "{key}");
        }
        assert_eq!(keys.len(), 11, "{record:?}");
    }

    #[test]
    fn default_json_values_match_json_fields() {
        /// Renders differently with `Display` and `Debug`.
        #[derive(Debug)]
        struct Shown;

        impl std::fmt::Display for Shown {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("shown-display")
            }
        }

        impl std::error::Error for Shown {}

        let (buffer, subscriber) = default_json("");
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            let error: &(dyn std::error::Error + 'static) = &Shown;
            tracing::info_span!(
                "values",
                span_u64 = u64::MAX,
                span_i128 = i128::MAX,
                span_display = %Shown,
                span_debug = ?Shown,
            )
            .in_scope(|| {
                tracing::info!(
                    event_u64 = u64::MAX,
                    event_i64 = i64::MIN,
                    event_i128 = i128::MAX,
                    event_u128 = u128::MAX,
                    display = %Shown,
                    debug = ?Shown,
                    finite = 1.5_f64,
                    infinite = f64::INFINITY,
                    nan = f64::NAN,
                    flag = true,
                    text = "plain",
                    bytes = &b"ab"[..],
                    error = error,
                    "values"
                );
            });
        }
        let records = ordered_records(&buffer);
        let record = record(&records, "values");
        let expected = [
            ("span_u64", serde_json::json!(u64::MAX)),
            (
                "span_i128",
                json_str("170141183460469231731687303715884105727"),
            ),
            ("span_display", json_str("shown-display")),
            ("span_debug", json_str("Shown")),
            ("event_u64", serde_json::json!(u64::MAX)),
            ("event_i64", serde_json::json!(i64::MIN)),
            (
                "event_i128",
                json_str("170141183460469231731687303715884105727"),
            ),
            (
                "event_u128",
                json_str("340282366920938463463374607431768211455"),
            ),
            ("display", json_str("shown-display")),
            ("debug", json_str("Shown")),
            ("finite", serde_json::json!(1.5)),
            ("infinite", serde_json::Value::Null),
            ("nan", serde_json::Value::Null),
            ("flag", serde_json::json!(true)),
            ("text", json_str("plain")),
            ("bytes", serde_json::json!([97, 98])),
            ("error", json_str("shown-display")),
        ];
        for (key, value) in expected {
            assert_eq!(record.get(key), Some(&value), "{key}: {record:?}");
        }
    }

    #[test]
    fn default_json_message_is_last_and_defaults_to_empty() {
        let (buffer, subscriber) = default_json("");
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::info!(marker = "no-message");
            tracing::info!(message = 7_u64);
        }
        let records = ordered_records(&buffer);
        assert_eq!(records.len(), 2, "{records:?}");
        let last = |record: &OrderedRecord| record.0.last().cloned();
        let first = records.first().expect("first record");
        assert_eq!(first.get("marker"), Some(&json_str("no-message")));
        assert_eq!(last(first), Some(("message".to_owned(), json_str(""))));
        let second = records.get(1).expect("second record");
        assert_eq!(
            last(second),
            Some(("message".to_owned(), serde_json::json!(7)))
        );
    }

    #[test]
    fn default_json_records_have_unique_keys() {
        let directives = format!("info,{TARGET}=debug");
        let (records, _result, arn) = run_default_json(
            crate::test_util::LocalRunner::new(),
            &directives,
            |_event, ctx| async move {
                ctx.step(|_| async { Ok(()) }).name("s").await?;
                ctx.run_in_child_context(|_child| async move { Ok(()) })
                    .name("c")
                    .await?;
                ctx.map(vec![1_u32], |_child, _item, _idx| async move { Ok(()) })
                    .name("m")
                    .await?;
                Ok(())
            },
        );
        let lifecycle: Vec<&OrderedRecord> = records
            .iter()
            .filter(|r| {
                r.get("message")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|m| m.starts_with("operation_") || m.starts_with("execution_"))
            })
            .collect();
        assert!(
            lifecycle
                .iter()
                .any(|r| r.get("message") == Some(&json_str("operation_started"))),
            "{records:?}"
        );
        for record in &records {
            assert_eq!(record.get("isReplay"), None, "{record:?}");
        }
        for record in lifecycle {
            assert_eq!(
                record.get("executionArn"),
                Some(&json_str(&arn)),
                "{record:?}"
            );
            assert!(record.get("requestId").is_some(), "{record:?}");
            let is_operation = record
                .get("message")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|m| m.starts_with("operation_"));
            if is_operation {
                assert!(record.get("operationId").is_some(), "{record:?}");
            }
        }
    }

    #[test]
    fn ordered_json_record_rejects_duplicate_keys() {
        assert!(serde_json::from_str::<OrderedRecord>(r#"{"a":1,"b":2}"#).is_ok());
        assert!(serde_json::from_str::<OrderedRecord>(r#"{"a":1,"b":2,"a":3}"#).is_err());
    }

    #[test]
    fn js_timestamp_matches_js() {
        use std::time::{Duration, UNIX_EPOCH};

        let at = |secs: u64, nanos: u32| UNIX_EPOCH + Duration::new(secs, nanos);
        let ok = |time| js_timestamp(time).expect("in range");
        assert_eq!(ok(UNIX_EPOCH), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            ok(at(1_700_000_000, 123_456_789)),
            "2023-11-14T22:13:20.123Z"
        );
        assert_eq!(
            ok(at(1_709_251_199, 999_999_999)),
            "2024-02-29T23:59:59.999Z"
        );
        assert_eq!(ok(at(1_709_251_200, 0)), "2024-03-01T00:00:00.000Z");
        // The last instant of year 9999, then the first of year 10000.
        assert_eq!(
            ok(at(253_402_300_799, 999_000_000)),
            "9999-12-31T23:59:59.999Z"
        );
        assert!(js_timestamp(at(253_402_300_800, 0)).is_err());

        // Before the epoch the time is cut towards earlier times, as JS's
        // millisecond clock is.
        let before = |secs: u64, nanos: u32| UNIX_EPOCH - Duration::new(secs, nanos);
        assert_eq!(ok(before(0, 1)), "1969-12-31T23:59:59.999Z");
        assert_eq!(ok(before(1, 0)), "1969-12-31T23:59:59.000Z");
        assert_eq!(ok(before(62_167_219_200, 0)), "0000-01-01T00:00:00.000Z");
        assert!(js_timestamp(before(62_167_219_200, 1)).is_err());
    }

    #[test]
    fn parse_fields_rejects_text_that_is_not_an_object() {
        assert_eq!(
            parse_fields(r#"{"a":1}"#).expect("an object").get("a"),
            Some(&serde_json::json!(1))
        );
        assert!(parse_fields("a=1").is_err());
    }

    /// A layer that removes the fields the `fmt` layer stored on each new
    /// span, or replaces them with text that is not JSON.
    struct TamperFields {
        replace: bool,
    }

    impl<S> tracing_subscriber::Layer<S> for TamperFields
    where
        S: tracing::Subscriber + for<'lookup> tracing_subscriber::registry::LookupSpan<'lookup>,
    {
        fn on_new_span(
            &self,
            _attrs: &Attributes<'_>,
            id: &Id,
            ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if let Some(span) = ctx.span(id) {
                let mut extensions = span.extensions_mut();
                extensions.remove::<FormattedFields<JsonFields>>();
                if self.replace {
                    extensions.insert(FormattedFields::<JsonFields>::new("a=1".to_owned()));
                }
            }
        }
    }

    /// The captured output of `info!("tampered")` in a span with the field
    /// `gone`, when [`TamperFields`] runs after the record's `fmt` layer.
    fn tampered_output(replace: bool) -> String {
        use tracing_subscriber::layer::SubscriberExt as _;

        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .fmt_fields(JsonFields::new())
                    .event_format(TopLevelJson)
                    .with_writer(CaptureWriter(std::sync::Arc::clone(&buffer))),
            )
            .with(TamperFields { replace });
        {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            tracing::info_span!("tampered_span", gone = 1_u32)
                .in_scope(|| tracing::info!("tampered"));
        }
        captured(&buffer)
    }

    /// A span without stored fields supplies none; a span whose stored
    /// fields are not JSON fails the record, which is then not written.
    #[test]
    fn top_level_json_handles_spans_without_json_fields() {
        let records: Vec<OrderedRecord> = tampered_output(false)
            .lines()
            .map(|line| serde_json::from_str(line).expect("a record"))
            .collect();
        let record = record(&records, "tampered");
        assert_eq!(
            record.keys(),
            ["timestamp", "level", "message"],
            "{record:?}"
        );

        assert_eq!(tampered_output(true), "");
    }
}
