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
//! When these spans are rendered by a JSON subscriber (e.g.,
//! `lambda_runtime`'s `init_default_subscriber()` with
//! `AWS_LAMBDA_LOG_FORMAT=JSON`), the fields appear as top-level JSON keys:
//! matching the `CloudWatch` Logs Insights query:
//!
//! ```text
//! filter coalesce(durableExecutionArn, executionArn) like "<arn>"
//! ```
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
use subscriber::fmt::MakeWriter;
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
/// 2. An `fmt` layer without the target and the time, writing to `writer`.
///    It writes JSON when `format` is `json` in any case, and text otherwise.
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
    let layer = subscriber::fmt::layer()
        .with_target(false)
        .without_time()
        .with_writer(writer);
    let registry = subscriber::registry();
    if format.eq_ignore_ascii_case("json") {
        Box::new(registry.with(layer.json().with_filter(filter)))
    } else {
        Box::new(registry.with(layer.with_filter(filter)))
    }
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
                    .pointer("/fields/message")
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
}
