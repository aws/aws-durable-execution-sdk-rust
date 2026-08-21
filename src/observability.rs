//! The SDK's observability contract: documented, stable `tracing` spans and
//! events describing the operation lifecycle.
//!
//! The SDK instruments itself through the [`tracing`] facade only: it never
//! installs a subscriber, adds no observability dependency, and works with
//! whatever subscriber the application provides. This module documents the
//! names and fields the SDK emits and exports them as constants, so a
//! subscriber, filter, or exporter can match on them without hard-coding
//! strings.
//!
//! # Stability
//!
//! Everything named in this module is a public, semver-meaningful contract:
//!
//! - Renaming or removing a span name, event name, or field name listed here
//!   is a **breaking change**.
//! - Adding a new event, or adding a new field to an existing span or event,
//!   is a **minor** (non-breaking) change. Write consumers to tolerate
//!   unknown fields and unknown events.
//!
//! # Spans
//!
//! | Span name | Wraps | Fields |
//! |-----------|-------|--------|
//! | [`span_names::EXECUTION`] (`durable_execution`) | one invocation of the handler; each child namespace (a `run_in_child_context` body, a map/parallel branch, a `wait_for_callback` submitter) gets its own detached span | `executionArn`, `requestId`, `isReplay` |
//! | [`span_names::OPERATION`] (`durable_operation`) | each live step body | `executionArn`, `requestId`, `operationId`, `attempt`, `isReplay` |
//!
//! Both spans are created at the `INFO` level. The `isReplay` field on the
//! `durable_execution` span is dynamic: the SDK re-records it after every
//! operation claim, so it flips to `false` the moment the invocation crosses
//! the replay high-water mark. The `replay-filter` feature's
//! `ReplayFilterLayer` is built on exactly this contract.
//!
//! # Events
//!
//! Lifecycle events are emitted at the `DEBUG` level with the target
//! [`TARGET`] (`aws_durable_execution_sdk_rust::lifecycle`). The event's
//! `message` is its stable name.
//!
//! Operation events fire when the SDK records an operation transition
//! (`operation_started`, `operation_succeeded`, `operation_failed`,
//! `operation_retry_scheduled`) or short-circuits one from recorded state
//! (`operation_replayed`). The record-transition events cover **every**
//! operation type the SDK checkpoints (steps, waits, invokes, callbacks,
//! `wait_for_condition` polls, child contexts, and map/parallel batches and
//! items), and each one is emitted only **after** the checkpoint write that
//! persists the transition succeeds: a rejected checkpoint records nothing,
//! so it emits nothing. The converse holds too: every transition the
//! service records emits its event: the events fire the moment the service
//! accepts the write, before any fallible follow-up work (such as fetching
//! paginated execution state), so a failure *after* the write still leaves
//! the persisted transitions' events emitted. The code path that performs
//! the write owns the event, so under checkpoint buffering
//! ([`checkpoint_delay`](crate::OptionsBuilder::checkpoint_delay) /
//! [`checkpoint_batching`](crate::OptionsBuilder::checkpoint_batching)) a
//! caller cancelled while awaiting a batched write (a dropped `race` or
//! `select_ok` loser) does not suppress the events for updates the flush
//! still persists, and when a large batch is split into several requests,
//! each request's events are emitted as soon as that request succeeds:
//! even if a later request in the same batch fails.
//!
//! `operation_replayed` likewise covers every operation type: steps, waits,
//! invokes, callbacks, `wait_for_condition`, child contexts, map/parallel
//! batches, and the combinators (`try_join_all`, `join_all`, `select_ok`,
//! `race`). It is emitted only after the recorded outcome has actually been
//! reconstructed: a corrupt payload or a failing custom serdes surfaces as
//! an error **without** the event, because no recorded terminal outcome was
//! returned. One exception is deliberate: a child context or batch recorded
//! in *replay-children* mode (its result was too large to checkpoint)
//! re-executes its body rather than short-circuiting, so it emits no
//! `operation_replayed` of its own: the operations inside it emit theirs.
//!
//! | Event name | Emitted when | Fields |
//! |------------|--------------|--------|
//! | [`event_names::OPERATION_STARTED`] | the SDK records the start of a live operation | identity fields, `attempt`, `isReplay = false` |
//! | [`event_names::OPERATION_SUCCEEDED`] | the SDK records an operation's success | identity fields, `attempt`, `isReplay = false` |
//! | [`event_names::OPERATION_FAILED`] | the SDK records an operation's permanent failure | identity fields, `attempt`, `isReplay = false`, `error` |
//! | [`event_names::OPERATION_RETRY_SCHEDULED`] | a retry strategy schedules another attempt | identity fields, `attempt` (the attempt that failed), `isReplay = false`, `delaySeconds`, `error` |
//! | [`event_names::OPERATION_REPLAYED`] | a recorded terminal outcome is returned without re-running the operation (steps, waits, invokes, callbacks, `wait_for_condition`, child contexts, map/parallel batches, combinators) | identity fields, `attempt`, `isReplay = true` |
//! | [`event_names::EXECUTION_STARTED`] | an invocation begins with no recorded operations to replay | `executionArn`, `requestId` |
//! | [`event_names::EXECUTION_RESUMED`] | an invocation begins with recorded operations to replay | `executionArn`, `requestId` |
//! | [`event_names::EXECUTION_SUSPENDED`] | the invocation ends by suspending (reported `PENDING`) | `executionArn`, `requestId` |
//!
//! Exactly one of `execution_started` / `execution_resumed` is emitted per
//! invocation, and `execution_suspended` at most once per invocation. All
//! three are emitted while the handler's `durable_execution` span is
//! entered, so they are span events of that span: a subscriber that
//! groups events by span (including the OpenTelemetry bridge below) sees
//! them on the execution span rather than as orphans.
//!
//! # Fields
//!
//! | Field | Type | Meaning |
//! |-------|------|---------|
//! | [`field_names::EXECUTION_ARN`] (`executionArn`) | string | durable execution ARN |
//! | [`field_names::REQUEST_ID`] (`requestId`) | string | Lambda invocation request ID |
//! | [`field_names::OPERATION_ID`] (`operationId`) | string | wire operation ID (SHA-256 hex digest of the positional ID) |
//! | [`field_names::OPERATION_NAME`] (`operationName`) | string, omitted when the operation is unnamed | the `.name("...")` label |
//! | [`field_names::OPERATION_TYPE`] (`operationType`) | string | one of `Step`, `Wait`, `Context`, `Callback`, `ChainedInvoke` |
//! | [`field_names::OPERATION_SUB_TYPE`] (`operationSubType`) | string, omitted when absent | wire sub-type (e.g. `Step`, `Wait`, `Map`, `Parallel`, `RunInChildContext`, `WaitForCondition`, `WaitForCallback`) |
//! | [`field_names::ATTEMPT`] (`attempt`) | integer | 1-based attempt number the event describes |
//! | [`field_names::IS_REPLAY`] (`isReplay`) | boolean | whether the event describes replayed (already recorded) work |
//! | [`field_names::DELAY_SECONDS`] (`delaySeconds`) | integer | delay before the next attempt (`operation_retry_scheduled` only) |
//! | [`field_names::ERROR`] (`error`) | string | error message (`operation_failed` and `operation_retry_scheduled` only) |
//!
//! # Consuming the contract with `tracing-subscriber`
//!
//! Lifecycle events are `DEBUG`-level, so a default `INFO` subscriber does
//! not print them. Enable the lifecycle target explicitly:
//!
//! ```
//! use tracing_subscriber::EnvFilter;
//! use tracing_subscriber::layer::SubscriberExt as _;
//!
//! // INFO everywhere, plus every SDK lifecycle event.
//! let filter = EnvFilter::new(format!(
//!     "info,{}=debug",
//!     aws_durable_execution_sdk_rust::observability::TARGET,
//! ));
//! let subscriber = tracing_subscriber::registry()
//!     .with(tracing_subscriber::fmt::layer().json())
//!     .with(filter);
//! # drop(subscriber);
//! ```
//!
//! # OpenTelemetry bridge
//!
//! Because the contract is plain `tracing` instrumentation, the standard
//! [`tracing-opentelemetry`](https://docs.rs/tracing-opentelemetry) bridge
//! exports it without SDK support: `durable_execution` and
//! `durable_operation` become OpenTelemetry spans carrying the fields above
//! as attributes, and lifecycle events become span events. The bridge crates
//! are the application's dependencies, not this SDK's: the SDK's dependency
//! allowlist stays closed. A typical Lambda `main` (versions current at the
//! time of writing; the example is illustrative, not compiled):
//!
//! ```ignore
//! // [dependencies]
//! // opentelemetry = "0.30"
//! // opentelemetry-otlp = "0.30"
//! // opentelemetry_sdk = "0.30"
//! // tracing-opentelemetry = "0.31"
//! // tracing-subscriber = { version = "0.3", features = ["env-filter"] }
//! use aws_durable_execution_sdk_rust as durable;
//! use opentelemetry::trace::TracerProvider as _;
//! use tracing_subscriber::layer::SubscriberExt as _;
//! use tracing_subscriber::util::SubscriberInitExt as _;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), lambda_runtime::Error> {
//!     let exporter = opentelemetry_otlp::SpanExporter::builder()
//!         .with_tonic()
//!         .build()?;
//!     let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
//!         .with_batch_exporter(exporter)
//!         .build();
//!     let tracer = provider.tracer("my-durable-function");
//!
//!     // Export the SDK's spans and lifecycle events to OTel; keep
//!     // CloudWatch logs at INFO.
//!     tracing_subscriber::registry()
//!         .with(tracing_opentelemetry::layer().with_tracer(tracer))
//!         .with(tracing_subscriber::fmt::layer().json())
//!         .with(tracing_subscriber::EnvFilter::new(format!(
//!             "info,{}=debug",
//!             durable::observability::TARGET,
//!         )))
//!         .init();
//!
//!     durable::run(handler).await
//! }
//! ```
//!
//! Replay awareness carries over: an exporter that must not re-emit
//! replayed work can drop events where `isReplay` is `true`. The
//! `replay-filter` feature's `ReplayFilterLayer` implements exactly that
//! rule for events that carry the field (and additionally suppresses
//! application log lines emitted inside spans marked `isReplay = true`,
//! which carry no field of their own).

/// The `tracing` target of every lifecycle event in this contract.
///
/// Use it to enable or route lifecycle events independently of the rest of
/// the application's logs, e.g. `EnvFilter::new("info,{TARGET}=debug")`.
pub const TARGET: &str = "aws_durable_execution_sdk_rust::lifecycle";

/// Names of the spans the SDK creates.
pub mod span_names {
    /// Wraps one invocation of the handler (and, detached, each child
    /// namespace's body). Fields: `executionArn`, `requestId`, `isReplay`.
    pub const EXECUTION: &str = "durable_execution";

    /// Wraps each live step body. Fields: `executionArn`, `requestId`,
    /// `operationId`, `attempt`, `isReplay`.
    pub const OPERATION: &str = "durable_operation";
}

/// Stable names (the `message` field) of the lifecycle events.
pub mod event_names {
    /// The SDK recorded the start of a live operation.
    pub const OPERATION_STARTED: &str = "operation_started";

    /// The SDK recorded an operation's success.
    pub const OPERATION_SUCCEEDED: &str = "operation_succeeded";

    /// The SDK recorded an operation's permanent failure.
    pub const OPERATION_FAILED: &str = "operation_failed";

    /// A retry strategy scheduled another attempt; the execution suspends
    /// for the recorded delay.
    pub const OPERATION_RETRY_SCHEDULED: &str = "operation_retry_scheduled";

    /// A recorded terminal outcome was returned without re-running the
    /// operation.
    pub const OPERATION_REPLAYED: &str = "operation_replayed";

    /// An invocation began with no recorded operations to replay.
    pub const EXECUTION_STARTED: &str = "execution_started";

    /// An invocation began with recorded operations to replay.
    pub const EXECUTION_RESUMED: &str = "execution_resumed";

    /// The invocation ended by suspending (reported `PENDING`).
    pub const EXECUTION_SUSPENDED: &str = "execution_suspended";
}

/// Names of the structured fields carried by the spans and events.
///
/// When rendered by a JSON subscriber (e.g. `lambda_runtime`'s
/// `init_default_subscriber()` with `AWS_LAMBDA_LOG_FORMAT=JSON`), these
/// appear as top-level JSON keys, matching the `CloudWatch` Logs Insights
/// query `filter coalesce(durableExecutionArn, executionArn) like "<arn>"`.
pub mod field_names {
    /// Durable execution ARN: identifies the orchestration instance.
    pub const EXECUTION_ARN: &str = "executionArn";

    /// Lambda invocation request ID.
    pub const REQUEST_ID: &str = "requestId";

    /// Wire operation ID (SHA-256 hex digest of the positional ID).
    pub const OPERATION_ID: &str = "operationId";

    /// User-supplied operation name (`.name("...")`); omitted when unnamed.
    pub const OPERATION_NAME: &str = "operationName";

    /// Operation type: `Step`, `Wait`, `Context`, `Callback`, or
    /// `ChainedInvoke`.
    pub const OPERATION_TYPE: &str = "operationType";

    /// Wire sub-type (e.g. `Map`, `RunInChildContext`); omitted when absent.
    pub const OPERATION_SUB_TYPE: &str = "operationSubType";

    /// 1-based attempt number the event describes.
    pub const ATTEMPT: &str = "attempt";

    /// Whether the span or event describes replayed work (`true`/`false`).
    pub const IS_REPLAY: &str = "isReplay";

    /// Delay in seconds before the next attempt
    /// (`operation_retry_scheduled` only).
    pub const DELAY_SECONDS: &str = "delaySeconds";

    /// Error message (`operation_failed` / `operation_retry_scheduled`).
    pub const ERROR: &str = "error";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_names_are_stable() {
        // These literals are the public contract; a change here is a
        // semver-relevant event that must be deliberate.
        assert_eq!(TARGET, "aws_durable_execution_sdk_rust::lifecycle");
        assert_eq!(span_names::EXECUTION, "durable_execution");
        assert_eq!(span_names::OPERATION, "durable_operation");
        assert_eq!(event_names::OPERATION_STARTED, "operation_started");
        assert_eq!(event_names::OPERATION_SUCCEEDED, "operation_succeeded");
        assert_eq!(event_names::OPERATION_FAILED, "operation_failed");
        assert_eq!(
            event_names::OPERATION_RETRY_SCHEDULED,
            "operation_retry_scheduled"
        );
        assert_eq!(event_names::OPERATION_REPLAYED, "operation_replayed");
        assert_eq!(event_names::EXECUTION_STARTED, "execution_started");
        assert_eq!(event_names::EXECUTION_RESUMED, "execution_resumed");
        assert_eq!(event_names::EXECUTION_SUSPENDED, "execution_suspended");
        assert_eq!(field_names::EXECUTION_ARN, "executionArn");
        assert_eq!(field_names::REQUEST_ID, "requestId");
        assert_eq!(field_names::OPERATION_ID, "operationId");
        assert_eq!(field_names::OPERATION_NAME, "operationName");
        assert_eq!(field_names::OPERATION_TYPE, "operationType");
        assert_eq!(field_names::OPERATION_SUB_TYPE, "operationSubType");
        assert_eq!(field_names::ATTEMPT, "attempt");
        assert_eq!(field_names::IS_REPLAY, "isReplay");
        assert_eq!(field_names::DELAY_SECONDS, "delaySeconds");
        assert_eq!(field_names::ERROR, "error");
    }
}
