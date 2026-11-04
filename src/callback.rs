//! Callback operation execution engine.
//!
//! Implements `create_callback` (deferred external completion) and
//! `wait_for_callback` (child-context wrapper around callback + submitter
//! step).
//!
//! Wire shape:
//! - `create_callback`: `OperationType::Callback`, `SubType` `"Callback"`
//! - `wait_for_callback`: `OperationType::Context`, `SubType` `"WaitForCallback"`
//! - Actions: Start → (Succeed | Fail)
//! - `CallbackOptions { timeout_seconds, heartbeat_timeout_seconds }`
//! - `CallbackDetails { callback_id, result }`

use std::future::Future;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::Instrument as _;

use crate::BoxError;
use crate::context::{DurableContext, StepContext, SubmittedWrite};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    CallbackError, CallbackErrorKind, ChildContextError, ChildContextErrorKind, OperationError,
    OperationErrorKind,
};
use crate::future::Callback;
use crate::serdes::{PayloadOrigin, Serdes, SerdesContext};
use crate::start::StartStep;

/// Wire sub-type for callback operations.
pub(crate) const CALLBACK_SUB_TYPE: &str = "Callback";

/// Wire sub-type for wait-for-callback context operations.
pub(crate) const WFCB_SUB_TYPE: &str = "WaitForCallback";

/// Internal state for `create_callback` execution passed from the builder.
pub(crate) struct CreateCallbackExecution<O, S> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) heartbeat: Option<Duration>,
    /// The resolved subtype: the builder's `sub_type`, or `Callback`.
    pub(crate) sub_type: crate::OperationSubType,
    /// The name the plugin hooks report instead of `name`: the callback of
    /// a named `wait_for_callback` reports `<name>-callback`. `None`
    /// reports `name`.
    pub(crate) plugin_name: Option<String>,
    pub(crate) serdes: S,
    pub(crate) _marker: std::marker::PhantomData<O>,
}

impl<O, S> CreateCallbackExecution<O, S>
where
    O: Send + 'static,
    S: Serdes<O>,
{
    /// Executes the create-callback operation: replay or live path.
    pub(crate) async fn execute(self) -> Result<Callback<O>, OperationError> {
        let started = self.start();
        self.execute_after_start(started).await
    }

    /// Runs the callback's start step now, on the caller's task, and
    /// returns the future of the rest of the operation (see
    /// [`crate::start`]).
    pub(crate) fn execute_started(
        self,
    ) -> impl Future<Output = Result<Callback<O>, OperationError>> + Send + 'static {
        let started = self.start();
        self.execute_after_start(started)
    }

    /// The start step: admission, task ownership, the replay identity
    /// check, and the urgent START of the first invocation (see
    /// `crate::start`). An in-flight record with its callback ID fires the
    /// replay start hook.
    fn start(&self) -> Result<StartStep, OperationError> {
        let positional_id = self.op_id.positional();
        let wire_id = self.op_id.wire();
        crate::start::start_step(
            &self.ctx,
            &crate::start::StartSpec {
                op_id: &self.op_id,
                operation_type: "Callback",
                sub_type: self.sub_type.as_str(),
                name: self.name.as_deref(),
                order: crate::start::StartOrder::AwaitUrgentResponse,
            },
            |view| {
                let in_flight = matches!(
                    view.status,
                    CheckpointStatus::Started | CheckpointStatus::Pending
                );
                // A record without a callback ID fails below, before any
                // hook, so it does not resume.
                if in_flight && self.recorded_callback_id(positional_id).is_some() {
                    crate::start::Cycle::Resume
                } else {
                    crate::start::Cycle::Record
                }
            },
            || self.start_update(wire_id),
            |plugins, _| {
                let info = self.info(wire_id).with_replay(true);
                crate::plugin::operation_started(plugins, &info);
            },
        )
    }

    /// The rest of the operation after its start step `started`.
    async fn execute_after_start(
        self,
        started: Result<StartStep, OperationError>,
    ) -> Result<Callback<O>, OperationError> {
        let started = started?;
        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // 2. Replay: a recorded callback returns its ID, and its outcome
        // when it has one.
        let start = match started {
            StartStep::Refused => return std::future::pending().await,
            StartStep::Queued(write) => write,
            StartStep::Recorded(view) => {
                // The backend assigns the callback ID in the START response,
                // so every checkpointed callback record must carry one. A
                // record without it is an invariant violation: fail at the
                // fault instead of continuing with an empty ID (issue #47;
                // the JS and Python SDKs fail the same way).
                let callback_id = self
                    .recorded_callback_id(&positional_id)
                    .ok_or_else(|| missing_callback_id_error(&wire_id, view.status.wire_str()))?;
                // Terminal statuses replay the recorded outcome without
                // waiting on the external system again (see
                // `crate::observability`).
                if matches!(
                    view.status,
                    CheckpointStatus::Started | CheckpointStatus::Pending
                ) {
                    // Callback is in flight: return pending (`result()`
                    // waits for the record). The start step fired the
                    // replay start hook.
                    return Ok(self.into_pending(callback_id));
                }
                // A terminal record fires the end hook when the callback is
                // created, as JS does. It is live only in the first
                // invocation after the service completed the callback.
                if let Some(plugins) = self.ctx.plugins()
                    && is_terminal_callback_status(&view.status)
                {
                    let info = self
                        .info(&wire_id)
                        .with_replay(!plugins.is_updated(&wire_id));
                    crate::plugin::operation_ended(plugins, &info);
                }
                let outcome = callback_outcome(
                    &self.ctx,
                    &positional_id,
                    self.name.as_deref(),
                    self.sub_type.as_str(),
                    &self.serdes,
                    &view,
                    true,
                )
                .await?;
                return Ok(Callback::new_settled(callback_id, outcome));
            }
        };

        // 3. First invocation: await the START response. Callback creation
        // is a flush point of the checkpoint-delay contract: the backend
        // assigns the callback_id in this response, so the start step
        // queued the write as urgent.
        if let Err(err) = start.await {
            // Audit (#43): callback-creation START: no user code ran (the
            // callback ID does not exist yet), so no terminal FAIL is
            // needed; re-invocation reconverges on the same write.
            return self
                .ctx
                .checkpoint_failure_unrecoverable(&wire_id, err, None)
                .await;
        }

        // After checkpointing START, the backend assigns a callback_id.
        // Read it from the (now-updated) checkpoint log. A record without
        // one is an invariant violation: fail at the fault instead of
        // handing the caller an empty ID (issue #47).
        let callback_id = self.recorded_callback_id(&positional_id).ok_or_else(|| {
            missing_callback_id_error(&wire_id, CheckpointStatus::Started.wire_str())
        })?;

        if let Some(plugins) = self.ctx.plugins() {
            let info = self
                .info(&wire_id)
                .with_status(crate::OperationStatus::Started)
                .with_replay(false);
            crate::plugin::operation_started(plugins, &info);
        }

        // Return pending: `result()` waits for the record.
        Ok(self.into_pending(callback_id))
    }

    /// The non-empty callback ID the record of `positional_id` carries.
    fn recorded_callback_id(&self, positional_id: &str) -> Option<String> {
        self.ctx
            .checkpoint_callback_id(positional_id)
            .filter(|id| !id.is_empty())
    }

    /// The START update of this callback, with its callback options.
    fn start_update(&self, wire_id: &str) -> OperationUpdate {
        let mut builder = OperationUpdate::builder()
            .id(wire_id.to_owned())
            .r#type(OperationType::Callback)
            .sub_type(self.sub_type.as_str())
            .action(OperationAction::Start);

        if let Some(n) = &self.name {
            builder = builder.name(n.clone());
        }
        if let Some(parent_wire) = self.ctx.parent_wire_id() {
            builder = builder.parent_id(parent_wire);
        }

        // Build callback options if timeout or heartbeat is set.
        let cb_opts = build_callback_options(self.timeout, self.heartbeat);
        if let Some(opts) = cb_opts {
            builder = builder.callback_options(opts);
        }

        #[expect(clippy::expect_used)] // reason: all required fields set above
        builder
            .build()
            .expect("all required OperationUpdate fields set")
    }

    /// The plugin info of this callback, read from its current record.
    fn info(&self, wire_id: &str) -> crate::plugin::OperationInfo {
        callback_info(
            &self.ctx,
            wire_id,
            self.sub_type.as_str(),
            self.name.as_deref(),
            self.plugin_name.as_deref(),
        )
    }

    /// A pending callback whose `result()` waits for the record to settle,
    /// then builds the outcome from it as the create-time replay does.
    fn into_pending(self, callback_id: String) -> Callback<O> {
        let Self {
            ctx,
            op_id,
            name,
            sub_type,
            plugin_name,
            serdes,
            ..
        } = self;
        let identity = crate::future::CallbackIdentity {
            name: name.clone(),
            plugin_name,
            sub_type: sub_type.as_str().to_owned(),
        };
        let wire_id = op_id.wire().to_owned();
        let positional_id = op_id.positional().to_owned();
        let resolver_ctx = ctx.clone();
        let resolver: crate::future::CallbackResolver<O> = Box::new(move || {
            Box::pin(async move {
                #[expect(clippy::expect_used)]
                // reason: `result()` calls the resolver only after the record
                // left STARTED and PENDING, and the checkpoint log never
                // removes a record during an invocation.
                let view = resolver_ctx
                    .checkpoint_status_view(&positional_id)
                    .expect("a settled callback record is still in the log");
                callback_outcome(
                    &resolver_ctx,
                    &positional_id,
                    name.as_deref(),
                    sub_type.as_str(),
                    &serdes,
                    &view,
                    false,
                )
                .await
                .and_then(|outcome| outcome)
            })
        });
        Callback::new_pending(callback_id, ctx, wire_id, identity, resolver)
    }
}

/// Whether a callback record with `status` holds the callback's outcome:
/// `SUCCEEDED`, `FAILED` or `TIMED_OUT`.
pub(crate) fn is_terminal_callback_status(status: &CheckpointStatus) -> bool {
    matches!(
        status,
        CheckpointStatus::Succeeded | CheckpointStatus::Failed | CheckpointStatus::TimedOut
    )
}

/// The plugin info of the callback `wire_id`, created in `ctx`, read from
/// its current record. A `plugin_name` replaces the name the info reports.
pub(crate) fn callback_info(
    ctx: &DurableContext,
    wire_id: &str,
    sub_type: &str,
    name: Option<&str>,
    plugin_name: Option<&str>,
) -> crate::plugin::OperationInfo {
    let info = crate::plugin::OperationInfo::for_operation(
        ctx,
        wire_id,
        crate::OperationKind::Callback,
        sub_type,
        name,
    );
    match plugin_name {
        Some(plugin_name) => info.with_name(Some(plugin_name.to_owned())),
        None => info,
    }
}

/// The outcome a settled callback record holds: the create-time replay
/// (`emit_replayed` true) and the result of a pending callback whose record
/// settled during the invocation (`emit_replayed` false).
///
/// The outer error fails the operation that reads the record (a payload
/// the serdes cannot decode, an unexpected status); the inner result is the
/// callback's own outcome.
async fn callback_outcome<O, S: Serdes<O>>(
    ctx: &DurableContext,
    positional_id: &str,
    name: Option<&str>,
    sub_type: &str,
    serdes: &S,
    view: &crate::engine::CheckpointStatusView,
    emit_replayed: bool,
) -> Result<Result<O, OperationError>, OperationError>
where
    O: Send + 'static,
{
    let wire_id = crate::engine::compute_wire_id_public(positional_id);
    let emit = || {
        if emit_replayed {
            ctx.emit_operation_replayed(&wire_id, name, "Callback", Some(sub_type), view.attempt);
        }
    };
    match view.status {
        CheckpointStatus::Succeeded => {
            // Callback completed successfully. The payload was written by an
            // external caller, so only the deserialize side of the serdes
            // acts on it. Decode FIRST, then emit `operation_replayed`: a
            // corrupt payload or failing serdes surfaces as an error without
            // claiming a recorded outcome was returned.
            let payload = ctx.checkpoint_result_payload(positional_id);
            let result_str = payload.unwrap_or_else(|| "null".to_owned());
            let serdes_ctx = SerdesContext::new(&wire_id, ctx.execution_arn());
            let value: O =
                deserialize_callback_result(ctx, name, serdes, result_str, serdes_ctx).await?;
            emit();
            Ok(Ok(value))
        }
        CheckpointStatus::Failed => {
            // Callback failed externally.
            emit();
            let wire = ctx.checkpoint_wire_error(positional_id).unwrap_or_default();
            Ok(Err(replayed_callback_error(
                wire,
                &wire_id,
                view.status.wire_str(),
            )))
        }
        CheckpointStatus::TimedOut => {
            // Callback timed out. The wire record, when the backend attached
            // one, still travels on the error. A missed heartbeat also ends
            // in `TimedOut`; its record carries the `Callback.Heartbeat`
            // type.
            emit();
            let wire = ctx.checkpoint_wire_error(positional_id).unwrap_or_default();
            let kind = if wire.error_type() == Some("Callback.Heartbeat") {
                CallbackErrorKind::HeartbeatTimedOut
            } else {
                CallbackErrorKind::TimedOut
            };
            Ok(Err(OperationError::from_kind(
                OperationErrorKind::Callback(CallbackError::new(kind, None)),
            )
            .with_operation(&wire_id, view.status.wire_str())
            .with_wire(wire)))
        }
        _ => {
            // Unexpected status: treat as internal error.
            Err(callback_internal_error(&format!(
                "unexpected checkpointed status: {:?}",
                view.status
            )))
        }
    }
}

/// Internal state for `wait_for_callback` execution passed from the builder.
///
/// Generic over the submitter closure `F` (and, through `F`'s output, its
/// future), so the submitter runs **without type erasure**. The one
/// erasure point is the builder's `.future()` / `into_future`, which boxes
/// the whole execution future once inside
/// [`DurableFuture`](crate::DurableFuture).
pub(crate) struct WaitForCallbackExecution<O, F, S> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) heartbeat: Option<Duration>,
    pub(crate) submitter: F,
    pub(crate) submitter_retry: Option<crate::RetryStrategy>,
    pub(crate) serdes: S,
    pub(crate) _marker: std::marker::PhantomData<O>,
}

impl<O, F, Fut, S> WaitForCallbackExecution<O, F, S>
where
    O: DeserializeOwned + Serialize + Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    /// Executes the wait-for-callback operation: replay or live path.
    ///
    /// Thin generic wrapper: the ONLY code monomorphized per call site.
    /// The replay/checkpoint state machine lives in the non-generic
    /// [`WfcbCore`] / [`WfcbAfter`] halves (generic over the result type
    /// `O` only); this wrapper just runs the inner body (whose own
    /// checkpoint plumbing, callback creation and the submitter step,
    /// is likewise non-generic) between them.
    ///
    /// Tests call it; the SDK builds every operation future with the start
    /// step form (see [`crate::future::DurableFuture::for_start_step`]).
    #[cfg(test)]
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        let (core, rest) = self.split();
        let started = core.start();
        Self::run(core, rest, started).await
    }

    /// Runs the operation's start step now, on the caller's task, and
    /// returns the future of the rest of it (see [`crate::start`]).
    pub(crate) fn execute_started(
        self,
    ) -> impl Future<Output = Result<O, OperationError>> + Send + 'static {
        let (core, rest) = self.split();
        let started = core.start();
        Self::run(core, rest, started)
    }

    /// Splits the execution into its non-generic core and the inputs of
    /// its body.
    fn split(self) -> (WfcbCore<O>, WfcbBody<F, S>) {
        let Self {
            ctx,
            op_id,
            name,
            timeout,
            heartbeat,
            submitter,
            submitter_retry,
            serdes,
            ..
        } = self;
        // The plugin names of the inner callback and submitter step.
        let plugin_names = InnerPluginNames::of(name.as_deref());
        let core = WfcbCore {
            ctx,
            op_id,
            name,
            _marker: std::marker::PhantomData,
        };
        let rest = WfcbBody {
            plugin_names,
            timeout,
            heartbeat,
            submitter,
            submitter_retry,
            serdes,
        };
        (core, rest)
    }

    /// The rest of the operation after its start step `started`.
    async fn run(
        core: WfcbCore<O>,
        rest: WfcbBody<F, S>,
        started: Result<StartStep, OperationError>,
    ) -> Result<O, OperationError> {
        let WfcbBody {
            plugin_names,
            timeout,
            heartbeat,
            submitter,
            submitter_retry,
            serdes,
        } = rest;
        match core.before(started?).await? {
            WfcbPrelude::Done(result) => result,
            WfcbPrelude::Run {
                child_ctx,
                start,
                after,
            } => {
                // Run inner body: create_callback + submitter step + await
                // result. Instrumented with the child namespace's
                // replay-aware span so a resumed body's log lines are
                // suppressed while its nested operations replay.
                let child_span = child_ctx.replay_span();
                let body_result = run_wfcb_body::<O, _, _, _>(
                    child_ctx,
                    plugin_names,
                    timeout,
                    heartbeat,
                    submitter,
                    submitter_retry,
                    serdes,
                )
                .instrument(child_span)
                .await;
                // The body returned, so the child is finished: abandon it,
                // as a `run_in_child_context` child is (see `Admission`).
                // The terminal update below goes through the parent
                // context, so it is admitted.
                after.ctx.admission().abandon([after.wire_id.as_str()]);
                after.settle(start, body_result).await
            }
        }
    }
}

/// The inputs of a `wait_for_callback` body.
struct WfcbBody<F, S> {
    plugin_names: InnerPluginNames,
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
    submitter: F,
    submitter_retry: Option<crate::RetryStrategy>,
    serdes: S,
}

/// The names the plugin hooks report for the inner operations of a
/// `wait_for_callback` named `n`: `n-callback` and `n-submitter`. An
/// unnamed `wait_for_callback` reports unnamed inner operations.
struct InnerPluginNames {
    callback: Option<String>,
    submitter: Option<String>,
}

impl InnerPluginNames {
    fn of(name: Option<&str>) -> Self {
        Self {
            callback: name.map(|n| format!("{n}-callback")),
            submitter: name.map(|n| format!("{n}-submitter")),
        }
    }
}

/// The plugin info of the `wait_for_callback` context `wire_id`, created
/// in `ctx`, read from its current record.
fn wfcb_info(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
) -> crate::plugin::OperationInfo {
    crate::plugin::OperationInfo::for_operation(
        ctx,
        wire_id,
        crate::OperationKind::Context,
        WFCB_SUB_TYPE,
        name,
    )
}

/// The pre-body half of `wait_for_callback`: task-ownership check, replay
/// resolution, and the START checkpoint. Generic only over the result type
/// `O`, no user closure reaches this state machine, so its replay and
/// checkpoint logic compiles once per result type instead of once per call
/// site.
struct WfcbCore<O> {
    ctx: DurableContext,
    op_id: OperationId,
    name: Option<String>,
    _marker: std::marker::PhantomData<fn() -> O>,
}

/// What [`WfcbCore::before`] decided: the operation is already resolved
/// from the checkpoint log, or the inner body must run in the prepared
/// child context.
enum WfcbPrelude<O> {
    /// Resolved without running the body (replayed success or failure).
    Done(Result<O, OperationError>),
    /// The body must run in `child_ctx`; `after` settles the outcome.
    Run {
        /// The fresh child context (chained prefix) the body runs in.
        child_ctx: DurableContext,
        /// The `START` write of this invocation, still in flight. The
        /// outcome write waits for it. `None` when the record existed.
        start: Option<SubmittedWrite>,
        /// The post-body half that checkpoints the outcome.
        after: WfcbAfter<O>,
    },
}

/// The post-body half of `wait_for_callback`: outcome checkpointing
/// (`ContextSucceeded` / `ContextFailed`). Generic only over the result type.
struct WfcbAfter<O> {
    ctx: DurableContext,
    wire_id: String,
    name: Option<String>,
    _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O> WfcbCore<O>
where
    O: DeserializeOwned + Serialize + Send + 'static,
{
    /// The start step of the outer context (see [`crate::start`]):
    /// admission, task ownership, the replay identity check, and, without
    /// a record, the `START`, queued, then `on_operation_start`, then
    /// released, as JS orders them. A record that re-enters the body fires
    /// the replay start hook here.
    fn start(&self) -> Result<StartStep, OperationError> {
        let wire_id = self.op_id.wire();
        crate::start::start_step(
            &self.ctx,
            &crate::start::StartSpec {
                op_id: &self.op_id,
                operation_type: "Context",
                sub_type: WFCB_SUB_TYPE,
                name: self.name.as_deref(),
                order: crate::start::StartOrder::QueueThenHook,
            },
            |view| match view.status {
                CheckpointStatus::Succeeded | CheckpointStatus::Failed => {
                    crate::start::Cycle::Record
                }
                // Started/Pending/Ready: re-enter the body.
                _ => crate::start::Cycle::Resume,
            },
            || {
                build_wfcb_update(
                    wire_id,
                    self.name.as_deref(),
                    OperationAction::Start,
                    &self.ctx,
                )
            },
            |plugins, is_replay| {
                let info = wfcb_info(&self.ctx, wire_id, self.name.as_deref());
                let info = if is_replay {
                    info.with_replay(true)
                } else {
                    info.with_status(crate::OperationStatus::Started)
                        .with_replay(false)
                };
                crate::plugin::operation_started(plugins, &info);
            },
        )
    }

    /// Runs everything that follows the start step `started` and precedes
    /// the inner body: the replay path, or the child context of the body.
    /// The body does not wait for a queued `START`: the outcome write does.
    async fn before(self, started: StartStep) -> Result<WfcbPrelude<O>, OperationError> {
        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();
        let Self { ctx, name, .. } = self;

        let start = match started {
            StartStep::Refused => return std::future::pending().await,
            StartStep::Queued(write) => Some(write),
            StartStep::Recorded(view) => match view.status {
                CheckpointStatus::Succeeded => {
                    // WaitForCallback context completed: deserialize the
                    // result FIRST, then emit `operation_replayed`: a corrupt
                    // payload surfaces as an error without claiming a
                    // recorded outcome was returned.
                    let payload = ctx.checkpoint_result_payload(&positional_id);
                    let result_str = payload.as_deref().unwrap_or("null");
                    let value: O = serde_json::from_str(result_str)
                        .map_err(|e| wfcb_internal_error(&format!("deserialize result: {e}")))?;
                    ctx.emit_operation_replayed(
                        &wire_id,
                        name.as_deref(),
                        "Context",
                        Some(WFCB_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(WfcbPrelude::Done(Ok(value)));
                }
                CheckpointStatus::Failed => {
                    // Context failed: classify the error.
                    ctx.emit_operation_replayed(
                        &wire_id,
                        name.as_deref(),
                        "Context",
                        Some(WFCB_SUB_TYPE),
                        view.attempt,
                    );
                    let wire = ctx
                        .checkpoint_wire_error(&positional_id)
                        .unwrap_or_default();
                    return Ok(WfcbPrelude::Done(Err(wfcb_failed_error(
                        wire,
                        &wire_id,
                        view.status.wire_str(),
                    ))));
                }
                // Started/Pending/Ready: execute/resume. The start step
                // fired the replay start hook.
                _ => None,
            },
        };

        // 4. Create child context with chained prefix.
        let child_ctx = ctx.new_child(&positional_id, name.as_deref());

        Ok(WfcbPrelude::Run {
            child_ctx,
            start,
            after: WfcbAfter {
                ctx,
                wire_id,
                name,
                _marker: std::marker::PhantomData,
            },
        })
    }
}

impl<O> WfcbAfter<O>
where
    O: DeserializeOwned + Serialize + Send + 'static,
{
    /// Settles the inner body's outcome: checkpoints `ContextSucceeded` with
    /// the serialized result, or `ContextFailed` with the classified error.
    ///
    /// The outcome write first waits for `start`, the retained `START`
    /// write. When that write fails, the invocation ends and no outcome is
    /// written.
    async fn settle(
        self,
        start: Option<SubmittedWrite>,
        body_result: Result<O, OperationError>,
    ) -> Result<O, OperationError> {
        let Self {
            ctx, wire_id, name, ..
        } = self;
        match body_result {
            Ok(value) => {
                // Success: serialize and checkpoint ContextSucceeded.
                let serialized = serde_json::to_string(&value)
                    .map_err(|e| wfcb_internal_error(&format!("serialize result: {e}")))?;
                let mut builder = OperationUpdate::builder()
                    .id(wire_id.clone())
                    .r#type(OperationType::Context)
                    .sub_type(WFCB_SUB_TYPE.to_owned())
                    .action(OperationAction::Succeed)
                    .payload(serialized.clone());
                if let Some(n) = &name {
                    builder = builder.name(n.clone());
                }
                if let Some(parent_wire) = ctx.parent_wire_id() {
                    builder = builder.parent_id(parent_wire);
                }
                #[expect(clippy::expect_used)] // reason: all required fields set above
                let update = builder
                    .build()
                    .expect("all required OperationUpdate fields set");

                ctx.await_retained_start(&wire_id, start, OperationAction::Succeed)
                    .await?;
                if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
                    // Audit (#43): wait-for-callback SUCCEED: the
                    // submitter ran and the external system completed the
                    // callback, so those side effects need a recorded
                    // outcome. A permanent rejection persists a small
                    // terminal FAIL before the execution fails.
                    let cwire = crate::error::checkpoint_failure_wire(&err);
                    let terminal = build_wfcb_fail_update(&wire_id, name.as_deref(), &ctx, &cwire);
                    return ctx
                        .checkpoint_failure_unrecoverable(&wire_id, err, Some(terminal))
                        .await;
                }
                if let Some(plugins) = ctx.plugins() {
                    let info = wfcb_info(&ctx, &wire_id, name.as_deref())
                        .with_status(crate::OperationStatus::Succeeded)
                        .with_result(Some(serialized.clone()))
                        .with_replay(false);
                    crate::plugin::operation_ended(plugins, &info);
                }

                // Round-trip deserialize for consistency.
                let out: O = serde_json::from_str(&serialized)
                    .map_err(|e| wfcb_internal_error(&format!("deserialize result: {e}")))?;
                Ok(out)
            }
            Err(err) => {
                // Failure: checkpoint ContextFailed.
                let wire = extract_wire_error(&err);
                let mut builder = OperationUpdate::builder()
                    .id(wire_id.clone())
                    .r#type(OperationType::Context)
                    .sub_type(WFCB_SUB_TYPE.to_owned())
                    .action(OperationAction::Fail);
                if let Some(n) = &name {
                    builder = builder.name(n.clone());
                }
                if let Some(parent_wire) = ctx.parent_wire_id() {
                    builder = builder.parent_id(parent_wire);
                }
                builder = builder.error(wire.to_error_object());
                #[expect(clippy::expect_used)] // reason: all required fields set above
                let update = builder
                    .build()
                    .expect("all required OperationUpdate fields set");

                // Checkpoint the failure so a replay of this record
                // reconstructs the same error: operation id, status, and
                // the wire record are present either way. A rejected
                // write routes unrecoverable: discarding it would leave
                // the record claiming less than what executed (#43).
                ctx.await_retained_start(&wire_id, start, OperationAction::Fail)
                    .await?;
                if let Err(client_err) = ctx.checkpoint_updates(vec![update]).await {
                    // Audit (#43): wait-for-callback FAIL: the submitter
                    // (and possibly the external system) ran; the failed
                    // FAIL write routes unrecoverable with a minimal
                    // terminal FAIL retry.
                    let cwire = crate::error::checkpoint_failure_wire(&client_err);
                    let terminal = build_wfcb_fail_update(&wire_id, name.as_deref(), &ctx, &cwire);
                    return ctx
                        .checkpoint_failure_unrecoverable(&wire_id, client_err, Some(terminal))
                        .await;
                }
                if let Some(plugins) = ctx.plugins() {
                    let info = wfcb_info(&ctx, &wire_id, name.as_deref())
                        .with_status(crate::OperationStatus::Failed)
                        .with_error(Some(wire.clone()))
                        .with_replay(false);
                    crate::plugin::operation_ended(plugins, &info);
                }
                Err(err
                    .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                    .with_wire(wire))
            }
        }
    }
}

/// Inner body of `wait_for_callback`: create callback, run submitter step,
/// await result.
///
/// The submitter is executed as a proper step operation (producing
/// `StepStarted`/`StepSucceeded`/`StepFailed` checkpoint events).
async fn run_wfcb_body<O, F, Fut, S>(
    child_ctx: DurableContext,
    plugin_names: InnerPluginNames,
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
    submitter: F,
    submitter_retry: Option<crate::RetryStrategy>,
    serdes: S,
) -> Result<O, OperationError>
where
    O: Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Clone + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    // Step 1: create the inner callback (no name, per wire spec). The per-op
    // serdes flows through to the inner callback's decode so a serdes set on
    // wait_for_callback actually reaches the delivered payload.
    let cb_exec: CreateCallbackExecution<O, S> = CreateCallbackExecution {
        ctx: child_ctx.clone(),
        op_id: child_ctx.mint_id(),
        name: None,
        timeout,
        heartbeat,
        sub_type: crate::OperationSubType::CALLBACK,
        plugin_name: plugin_names.callback,
        serdes,
        _marker: std::marker::PhantomData,
    };
    let cb = cb_exec.execute().await?;

    // Step 2: run the submitter as a proper step operation.
    // This produces StepStarted/StepSucceeded (or StepFailed) events in the
    // checkpoint log, matching the wire protocol expectation. The step is
    // unnamed (empty-string equivalent).
    let callback_id = cb.id().to_owned();
    let step_exec = crate::step::StepExecution {
        ctx: child_ctx.clone(),
        op_id: child_ctx.mint_id(),
        name: None,
        retry_strategy: submitter_retry,
        serdes: crate::serdes::JsonSerdes,
        semantics: crate::step::StepSemantics::default(),
        sub_type: crate::OperationSubType::STEP,
        plugin_name: plugin_names.submitter,
        closure: move |step_ctx: StepContext| async move {
            (submitter)(step_ctx, callback_id).await?;
            Ok(())
        },
        _marker: std::marker::PhantomData,
    };
    let step_result = step_exec.execute().await;
    if let Err(e) = step_result {
        // Map step errors to child-context errors for consistent error
        // propagation through the WaitForCallback context boundary.
        return Err(OperationError::from_kind(OperationErrorKind::ChildContext(
            ChildContextError::new(ChildContextErrorKind::ChildFailed, Some(Box::new(e))),
        )));
    }

    // Step 3: await callback result.
    cb.result().await
}

/// Builds the wire `CallbackOptions` from timeout/heartbeat durations.
/// Returns `None` when neither is set.
fn build_callback_options(
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
) -> Option<aws_sdk_lambda::types::CallbackOptions> {
    #[expect(clippy::cast_possible_truncation)] // reason: duration ≤ i32::MAX for practical timers
    let timeout_secs = timeout.map_or(0, |d| {
        (d.as_secs_f64().ceil() as i64).min(i64::from(i32::MAX)) as i32
    });
    #[expect(clippy::cast_possible_truncation)] // reason: duration ≤ i32::MAX for practical timers
    let heartbeat_secs = heartbeat.map_or(0, |d| {
        (d.as_secs_f64().ceil() as i64).min(i64::from(i32::MAX)) as i32
    });

    if timeout_secs == 0 && heartbeat_secs == 0 {
        return None;
    }

    Some(
        aws_sdk_lambda::types::CallbackOptions::builder()
            .timeout_seconds(timeout_secs)
            .heartbeat_timeout_seconds(heartbeat_secs)
            .build(),
    )
}

/// Builds a `WaitForCallback` context operation update.
fn build_wfcb_update(
    wire_id: &str,
    name: Option<&str>,
    action: OperationAction,
    ctx: &DurableContext,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(WFCB_SUB_TYPE.to_owned())
        .action(action);
    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }
    #[expect(clippy::expect_used)] // reason: all required fields set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Builds the terminal `FAIL` update a wait-for-callback context persists
/// when its own outcome write was permanently rejected (issue #43).
fn build_wfcb_fail_update(
    wire_id: &str,
    name: Option<&str>,
    ctx: &DurableContext,
    wire: &crate::error::WireError,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(WFCB_SUB_TYPE.to_owned())
        .action(OperationAction::Fail)
        .error(wire.to_error_object());
    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }
    #[expect(clippy::expect_used)] // reason: all required fields set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Classifies a callback error kind from the wire `error_type`.
///
/// The registry is scoped to the SDK's own wire discriminators
/// (`Callback.Timeout`, `Callback.Heartbeat`); any other type is an
/// external failure, whose reported fields travel as wire data on the
/// error's source rather than as kind fields.
fn classify_callback_error(error_type: Option<&str>) -> CallbackErrorKind {
    match error_type {
        Some("Callback.Timeout") => CallbackErrorKind::TimedOut,
        Some("Callback.Heartbeat") => CallbackErrorKind::HeartbeatTimedOut,
        _ => CallbackErrorKind::ExternalFailure,
    }
}

/// Rebuilds a callback `OperationError` from a recorded wire failure.
///
/// The kind classifies; the recorded fields travel on the synthetic
/// source and the attached wire record.
fn replayed_callback_error(
    wire: crate::error::WireError,
    wire_id: &str,
    status: &str,
) -> OperationError {
    let kind = classify_callback_error(wire.error_type());
    let source = match kind {
        // Timeouts carry no foreign cause; the kind is the whole story.
        CallbackErrorKind::TimedOut | CallbackErrorKind::HeartbeatTimedOut => None,
        _ => Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    };
    OperationError::from_kind(OperationErrorKind::Callback(CallbackError::new(
        kind, source,
    )))
    .with_operation(wire_id, status)
    .with_wire(wire)
}

/// Constructs the appropriate error for a failed `WaitForCallback` context.
fn wfcb_failed_error(wire: crate::error::WireError, wire_id: &str, status: &str) -> OperationError {
    // Propagate callback errors directly (they bubble through the
    // child-context wrapping on the wire).
    if matches!(
        wire.error_type(),
        Some("Callback.Timeout" | "Callback.Heartbeat" | "CallbackError")
    ) {
        return replayed_callback_error(wire, wire_id, status);
    }

    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::ChildFailed,
        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    )))
    .with_operation(wire_id, status)
    .with_wire(wire)
}

/// Derives the wire failure record for an `OperationError` crossing the
/// wait-for-callback context boundary.
///
/// Callback kinds map to their dedicated wire discriminators; an external
/// failure re-reports the fields the external caller supplied (from the
/// error's attached wire record). Everything else goes through the
/// standard wire derivation, flattening once.
fn extract_wire_error(err: &OperationError) -> crate::error::WireError {
    if let OperationErrorKind::Callback(cb_err) = err.kind() {
        match cb_err.kind() {
            CallbackErrorKind::TimedOut => {
                return crate::error::WireError::new(
                    Some("Callback.Timeout"),
                    Some("callback timed out"),
                );
            }
            CallbackErrorKind::HeartbeatTimedOut => {
                return crate::error::WireError::new(
                    Some("Callback.Heartbeat"),
                    Some("heartbeat timed out"),
                );
            }
            CallbackErrorKind::ExternalFailure => {
                // Re-report the external caller's own fields verbatim.
                if let Some(wire) = err.wire() {
                    return wire.clone();
                }
            }
            CallbackErrorKind::DeserializationFailed => {
                return crate::error::wire_error_with_type(cb_err, "DeserializationError");
            }
            CallbackErrorKind::Internal => {
                return crate::error::wire_error_with_type(cb_err, "InternalError");
            }
        }
    }
    crate::error::wire_error_for(err, "Error")
}

/// Creates a callback deserialization error carrying its cause.
fn callback_deser_error(boundary: &str, e: BoxError) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Callback(CallbackError::new(
        CallbackErrorKind::DeserializationFailed,
        Some(crate::error::ContextualError::source_from(boundary, e)),
    )))
}

/// Decodes an externally-delivered callback payload into the target type.
///
/// The payload is produced by an external caller, so only the deserialize
/// side of the serdes is meaningful: the configured serdes turns the wire
/// payload directly into `O`. The context is marked
/// [`PayloadOrigin::External`] here, at the boundary, so a serdes with
/// storage indirection (e.g. `FileSystemSerdes`) never honors a file
/// reference an external caller delivered.
///
/// A marked error ends the invocation
/// ([`DurableContext::route_serdes_error`]); any other error becomes
/// `DeserializationFailed`.
pub(crate) async fn deserialize_callback_result<O, S: Serdes<O>>(
    ctx: &DurableContext,
    name: Option<&str>,
    serdes: &S,
    payload: String,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    let operation_id = serdes_ctx.operation_id().to_owned();
    match ctx
        .hold_activity(serdes.deserialize(payload, serdes_ctx.with_origin(PayloadOrigin::External)))
        .await
    {
        Ok(value) => Ok(value),
        Err(err) => {
            let e = ctx
                .route_serdes_error("deserialize", &operation_id, name, err)
                .await;
            Err(callback_deser_error("callback serdes", e))
        }
    }
}

/// Creates a callback internal error; the message becomes the source
/// frame, keeping the kind a pure classification.
fn callback_internal_error(msg: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Callback(CallbackError::new(
        CallbackErrorKind::Internal,
        Some(msg.to_owned().into()),
    )))
}

/// Creates the error for a checkpointed callback record that carries no
/// backend-assigned callback ID (issue #47). The backend assigns the ID
/// in the START response, so its absence after START is an invariant
/// violation; the message names the operation, and the operation context
/// carries the wire ID and recorded status.
fn missing_callback_id_error(wire_id: &str, status: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Callback(CallbackError::new(
        CallbackErrorKind::Internal,
        Some(format!("no callback ID found for started callback: {wire_id}").into()),
    )))
    .with_operation(wire_id, status)
}

/// Creates a wait-for-callback internal error; the message becomes the
/// source frame, keeping the kind a pure classification.
fn wfcb_internal_error(msg: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
        ChildContextErrorKind::Internal,
        Some(msg.to_owned().into()),
    )))
}

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::client::InMemoryExecutionClient;
    use crate::engine::{CheckpointLog, CheckpointRecord};
    use std::sync::Arc;

    /// Helper: create a context with a preloaded checkpoint log.
    fn ctx_with_log(records: Vec<(String, CheckpointRecord)>) -> DurableContext {
        let log = CheckpointLog::from_records(records);
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            client,
            "token-1".to_owned(),
        )
    }

    /// Helper: build a checkpoint record for callback operations.
    fn callback_record(
        wire_id: &str,
        status: CheckpointStatus,
        result: Option<&str>,
        error_type: Option<&str>,
        error_message: Option<&str>,
        callback_id: Option<&str>,
    ) -> (String, CheckpointRecord) {
        (
            wire_id.to_owned(),
            CheckpointRecord {
                id: wire_id.to_owned(),
                status,
                result: result.map(String::from),
                error_type: error_type.map(String::from),
                error_message: error_message.map(String::from),
                error_data: None,
                stack_trace: None,
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                invoke_error_data: None,
                invoke_stack_trace: None,
                replay_children: false,
                callback_id: callback_id.map(String::from),
                op_type: None,
                sub_type: None,
                due_at: None,
                op_name: None,
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
            },
        )
    }

    #[tokio::test]
    async fn replay_succeeded() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#""hello""#),
            None,
            None,
            Some("cb-123"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("test".to_owned()),
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should succeed");
        assert_eq!(cb.id(), "cb-123");
        let value = cb.result().await.expect("should have value");
        assert_eq!(value, "hello");
    }

    /// A callback result is a [`DurableFuture`], so it participates in the
    /// durable combinators like any other durable operation. Here a settled
    /// callback wins a `race` against a slow future.
    #[tokio::test]
    async fn result_participates_in_race_combinator() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#""callback-wins""#),
            None,
            None,
            Some("cb-race"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("racer".to_owned()),
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should succeed");

        let slow = crate::future::DurableFuture::from_async(async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok("slow".to_owned())
        });

        let winner = ctx
            .race([cb.result(), slow])
            .await
            .expect("race should resolve with the callback value");
        assert_eq!(winner, "callback-wins");
    }

    /// A settled callback result composes with `try_join_all` alongside
    /// other durable futures.
    #[tokio::test]
    async fn result_participates_in_try_join_all_combinator() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#""from-callback""#),
            None,
            None,
            Some("cb-join"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("joiner".to_owned()),
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should succeed");

        let other = crate::future::DurableFuture::from_async(async { Ok("plain".to_owned()) });

        let values = ctx
            .try_join_all([cb.result(), other])
            .await
            .expect("try_join_all should resolve");
        assert_eq!(values, vec!["from-callback".to_owned(), "plain".to_owned()]);
    }

    #[tokio::test]
    async fn replay_timed_out() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::TimedOut,
            None,
            None,
            None,
            Some("cb-456"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: Some(Duration::from_secs(5)),
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should return callback");
        let err = cb.result().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::TimedOut)
        ));
    }

    #[tokio::test]
    async fn replay_heartbeat_timed_out() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Failed,
            None,
            Some("Callback.Heartbeat"),
            Some("heartbeat timed out"),
            Some("cb-789"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: Some(Duration::from_secs(10)),
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should return callback");
        let err = cb.result().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::HeartbeatTimedOut)
        ));
    }

    #[tokio::test]
    async fn replay_timed_out_on_heartbeat() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::TimedOut,
            None,
            Some("Callback.Heartbeat"),
            Some("Callback timed out on heartbeat"),
            Some("cb-790"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: Some(Duration::from_secs(10)),
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should return callback");
        let err = cb.result().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::HeartbeatTimedOut)
        ));
    }

    #[tokio::test]
    async fn replay_external_failure() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Failed,
            None,
            Some("ValidationError"),
            Some("invalid input"),
            Some("cb-fail"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should return callback");
        let err = cb.result().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::ExternalFailure)
        ));
    }

    #[tokio::test]
    async fn live_returns_assigned_id() {
        // Empty checkpoint log: first invocation.
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("my-cb".to_owned()),
            timeout: Some(Duration::from_mins(1)),
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should succeed");
        // The in-memory client assigns backend-style callback IDs.
        assert_eq!(cb.id(), "in-mem-cb-1");
    }

    /// A context on the records `records` whose one plugin is `hooks`.
    fn ctx_with_hooks(
        records: Vec<(String, CheckpointRecord)>,
        hooks: &crate::plugin::test_support::OpHooks,
    ) -> DurableContext {
        DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:123:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(records)),
            Arc::new(InMemoryExecutionClient::new(vec![])),
            "token-1".to_owned(),
            None,
            Some(hooks.plugins()),
        )
    }

    /// Creates the callback `cb` in `ctx` and returns the outcome.
    async fn create_cb(ctx: &DurableContext) -> Result<Callback<String>, OperationError> {
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id: ctx.mint_id(),
            name: Some("cb".to_owned()),
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        exec.execute().await
    }

    /// `(hook, is_replay, status)` of every hook call `hooks` recorded.
    fn hook_shapes(
        hooks: &crate::plugin::test_support::OpHooks,
    ) -> Vec<(&'static str, bool, Option<crate::OperationStatus>)> {
        hooks
            .records()
            .iter()
            .map(|r| (r.hook, r.replay, r.status))
            .collect()
    }

    /// A `STARTED` callback record without a callback ID returns the
    /// missing-ID error and fires no hook.
    #[tokio::test]
    async fn callback_missing_id_fires_no_hook() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_hooks(
            vec![callback_record(
                &wire,
                CheckpointStatus::Started,
                None,
                None,
                None,
                None,
            )],
            &hooks,
        );
        let err = create_cb(&ctx).await.expect_err("the ID is missing");
        assert_missing_callback_id_error(&err, &wire);
        assert!(hooks.records().is_empty(), "{:?}", hooks.records());
    }

    /// A live callback whose START response carries no callback ID returns
    /// the missing-ID error and fires no hook.
    #[tokio::test]
    async fn live_callback_missing_id_fires_no_hook() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        client.enqueue_checkpoint_response(crate::client::TestResponse::Success(vec![]));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-1".to_owned(),
            None,
            Some(hooks.plugins()),
        );
        let wire = crate::engine::compute_wire_id_public("1");
        let err = create_cb(&ctx).await.expect_err("the ID is missing");
        assert_missing_callback_id_error(&err, &wire);
        assert!(hooks.records().is_empty(), "{:?}", hooks.records());
    }

    /// A replayed `STARTED` or `PENDING` callback with an ID fires
    /// `on_operation_start` with `is_replay` `true`.
    #[tokio::test]
    async fn running_callback_record_starts_replayed() {
        for status in [CheckpointStatus::Started, CheckpointStatus::Pending] {
            let hooks = crate::plugin::test_support::OpHooks::default();
            let wire = crate::engine::compute_wire_id_public("1");
            let ctx = ctx_with_hooks(
                vec![callback_record(
                    &wire,
                    status.clone(),
                    None,
                    None,
                    None,
                    Some("cb-1"),
                )],
                &hooks,
            );
            create_cb(&ctx).await.expect("the callback is running");
            let records = hooks.of("cb");
            assert_eq!(records.len(), 1, "{status:?}: {records:?}");
            let record = records.first().expect("one record");
            assert_eq!((record.hook, record.replay), ("start", true), "{status:?}");
        }
    }

    /// A replayed `FAILED` or `TIMED_OUT` callback that the service did not
    /// change since the previous invocation fires `on_operation_end` with
    /// `is_replay` `true` when it is created.
    #[tokio::test]
    async fn failed_callback_record_ends_replayed() {
        for (status, expected) in [
            (CheckpointStatus::Failed, crate::OperationStatus::Failed),
            (CheckpointStatus::TimedOut, crate::OperationStatus::TimedOut),
        ] {
            let hooks = crate::plugin::test_support::OpHooks::default();
            let wire = crate::engine::compute_wire_id_public("1");
            let ctx = ctx_with_hooks(
                vec![callback_record(
                    &wire,
                    status.clone(),
                    None,
                    Some("Declined"),
                    Some("no"),
                    Some("cb-1"),
                )],
                &hooks,
            );
            let cb = create_cb(&ctx).await.expect("the record replays");
            assert_eq!(
                hook_shapes(&hooks),
                [("end", true, Some(expected))],
                "{status:?}"
            );
            cb.result().await.expect_err("the callback did not succeed");
            assert_eq!(hooks.records().len(), 1, "result() fires no second hook");
        }
    }

    /// A callback record with an ID and a status other than `STARTED`,
    /// `PENDING`, `SUCCEEDED`, `FAILED` or `TIMED_OUT` fires no hook.
    #[tokio::test]
    async fn cancelled_callback_record_fires_no_hook() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_hooks(
            vec![callback_record(
                &wire,
                CheckpointStatus::Cancelled,
                None,
                None,
                None,
                Some("cb-1"),
            )],
            &hooks,
        );
        create_cb(&ctx)
            .await
            .expect_err("a cancelled callback is unexpected");
        assert!(hooks.records().is_empty(), "{:?}", hooks.records());
    }

    /// Asserts the error is the typed `CallbackError` for a missing
    /// callback ID, naming the failing operation (issue #47).
    fn assert_missing_callback_id_error(err: &OperationError, wire: &str) {
        assert!(
            matches!(
                err.kind(),
                OperationErrorKind::Callback(e)
                    if matches!(e.kind(), CallbackErrorKind::Internal)
            ),
            "expected Callback/Internal error, got {err:#}"
        );
        assert_eq!(err.operation_id(), Some(wire));
        let chain = format!("{err:#}");
        assert!(
            chain.contains(wire),
            "error should name the operation, got: {chain}"
        );
    }

    #[tokio::test]
    async fn replay_started_missing_callback_id_fails() {
        // A STARTED callback record without a backend-assigned callback ID
        // is an invariant violation: the SDK must fail loudly instead of
        // continuing with an empty ID (issue #47).
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Started,
            None,
            None,
            None,
            None,
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("no-id".to_owned()),
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.expect_err("must fail on missing id");
        assert_missing_callback_id_error(&err, &wire);
        assert_eq!(err.status(), Some("STARTED"));
    }

    #[tokio::test]
    async fn replay_started_empty_callback_id_fails() {
        // An empty-string callback ID is as unusable as a missing one and
        // fails the same way (issue #47; same principle as #31).
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Started,
            None,
            None,
            None,
            Some(""),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.expect_err("must fail on empty id");
        assert_missing_callback_id_error(&err, &wire);
    }

    #[tokio::test]
    async fn replay_succeeded_missing_callback_id_fails() {
        // The invariant covers every checkpointed callback record, not just
        // in-flight ones: the backend assigns the ID at START, so a
        // terminal record without one is equally corrupt (issue #47).
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#""hello""#),
            None,
            None,
            None,
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.expect_err("must fail on missing id");
        assert_missing_callback_id_error(&err, &wire);
    }

    #[tokio::test]
    async fn live_missing_callback_id_fails() {
        // Live path: the backend's START response is expected to carry the
        // assigned callback ID. When it does not (forced here with an
        // explicit empty checkpoint response), the SDK fails with the typed
        // error instead of returning an empty ID (issue #47).
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        client.enqueue_checkpoint_response(crate::client::TestResponse::Success(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();
        let wire = op_id.wire().to_owned();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("no-id-live".to_owned()),
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.expect_err("must fail on missing id");
        assert_missing_callback_id_error(&err, &wire);
        assert_eq!(err.status(), Some("STARTED"));
    }

    #[tokio::test]
    async fn id_stable_across_replay() {
        // Verify the operation ID is stable.
        let wire1 = crate::engine::compute_wire_id_public("1");
        let wire2 = crate::engine::compute_wire_id_public("1");
        assert_eq!(wire1, wire2);
    }

    #[tokio::test]
    async fn wfcb_replay_succeeded() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#""done""#),
            None,
            None,
            None,
        )]);
        let op_id = ctx.mint_id();
        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("wfcb-test".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let result = exec.execute().await.expect("should succeed");
        assert_eq!(result, "done");
    }

    #[tokio::test]
    async fn wfcb_replay_timed_out() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Failed,
            None,
            Some("Callback.Timeout"),
            Some("callback timed out"),
            None,
        )]);
        let op_id = ctx.mint_id();
        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::TimedOut)
        ));
    }

    #[tokio::test]
    async fn wfcb_replay_heartbeat_timed_out() {
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Failed,
            None,
            Some("Callback.Heartbeat"),
            Some("heartbeat timed out"),
            None,
        )]);
        let op_id = ctx.mint_id();
        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::HeartbeatTimedOut)
        ));
    }

    #[tokio::test]
    async fn wfcb_submitter_failure_propagates() {
        // When the submitter closure returns an error, the step execution
        // path uses the default retry strategy. On first failure, it
        // schedules a retry (checkpoint + suspend), and the error propagates
        // as a ChildContext/ChildFailed wrapping the step error.
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();
        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("sub-fail-test".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async {
                Err::<(), BoxError>("submitter exploded".into())
            },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        // The submitter step fails; the default retry strategy schedules a
        // retry (backend-owned timer), which suspends the invocation instead
        // of surfacing a fabricated error to the caller.
        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            exec.execute(),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn create_callback_spawn_executes() {
        // Verify that spawning a CreateCallbackBuilder returns a future
        // that resolves (live path: will checkpoint then return pending).
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-1".to_owned(),
        );
        // Use the builder's spawn path.
        let future = ctx.create_callback::<String>().name("spawn-test").spawn();
        let cb = future.await.expect("spawn should produce callback");
        // Live path returns the backend-assigned callback_id.
        assert!(!cb.id().is_empty(), "expected an assigned callback id");
    }

    #[tokio::test]
    async fn create_callback_ownership_rejects_foreign_task() {
        // Verify that calling from an unblessed task triggers an ownership
        // error (same pattern as invoke/step ownership tests).
        let result = tokio::spawn(async {
            let client = Arc::new(InMemoryExecutionClient::new(vec![]));
            let log = Arc::new(CheckpointLog::empty());
            let ctx = DurableContext::new_root_with_client(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                log,
                client,
                "token-1".to_owned(),
            );

            // Spawn a DIFFERENT (non-blessed) task.
            let ctx_clone = ctx.clone();
            let handle = tokio::spawn(async move {
                let op_id = ctx_clone.mint_id();
                let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
                    ctx: ctx_clone,
                    op_id,
                    name: None,
                    timeout: None,
                    heartbeat: None,
                    sub_type: crate::OperationSubType::CALLBACK,
                    plugin_name: None,
                    serdes: crate::serdes::JsonSerdes,
                    _marker: std::marker::PhantomData,
                };
                exec.execute().await
            });

            handle.await.unwrap()
        })
        .await
        .unwrap();

        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("task") || err_msg.contains("owner"),
            "expected ownership error, got: {err_msg}"
        );
    }

    // Submitter-as-step tests (verifies the fix for missing StepStarted/
    //    StepSucceeded events in the WaitForCallback child context)

    #[tokio::test]
    async fn wfcb_submitter_executes_as_step_with_checkpoints() {
        // The submitter must go through StepExecution, producing checkpoint
        // calls for StepStarted and StepSucceeded. With the child context's
        // callback START checkpoint, we expect at least 4 checkpoint calls:
        //   1. ContextStarted (from WaitForCallbackExecution)
        //   2. CallbackStarted (from CreateCallbackExecution)
        //   3. StepStarted (from StepExecution wrapping submitter)
        //   4. StepSucceeded (from StepExecution wrapping submitter)
        // Plus ContextSucceeded for the outer context.
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();

        let submitted = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let submitted_clone = Arc::clone(&submitted);

        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("step-check".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: move |_sc: StepContext, _id: String| {
                submitted_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                async { Ok::<(), BoxError>(()) }
            },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };

        // The callback result will suspend (pending), so the execution
        // will error at cb.result(). We just need to verify that the
        // submitter ran and checkpoints were produced.
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            Arc::clone(ctx.coordinator()),
            exec.execute(),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        // Verify submitter was called.
        assert!(
            submitted.load(std::sync::atomic::Ordering::SeqCst),
            "submitter should have been invoked"
        );

        // Verify checkpoint calls were made (at minimum: ContextStart,
        // CallbackStart, StepStart, StepSucceed).
        let call_count = *client
            .checkpoint_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            call_count >= 4,
            "expected at least 4 checkpoint calls (ContextStart + CallbackStart + \
             StepStart + StepSucceed), got {call_count}"
        );
    }

    #[tokio::test]
    #[expect(clippy::too_many_lines)] // reason: three full checkpoint-record literals read better inline
    async fn wfcb_submitter_replay_skips_re_execution() {
        // When the submitter step is already checkpointed as Succeeded,
        // replay should NOT re-invoke the submitter closure.
        use std::sync::atomic::{AtomicBool, Ordering};

        // Build a checkpoint log that has:
        //   pos "1" → ContextStarted (WaitForCallback context) [Status: Started]
        //   child prefix = "1", so child ops are:
        //   pos "1-1" → CallbackStarted [Status: Started, callback_id set]
        //   pos "1-2" → StepSucceeded  [the submitter step]
        //   Then the callback is still pending (no terminal status) so the
        //   execution suspends, but the submitter must NOT re-run.
        let wfcb_wire = crate::engine::compute_wire_id_public("1");
        let cb_wire = crate::engine::compute_wire_id_public("1-1");
        let step_wire = crate::engine::compute_wire_id_public("1-2");

        let ctx = ctx_with_log(vec![
            // WaitForCallback context: Started (non-terminal, fall through).
            (
                wfcb_wire.clone(),
                CheckpointRecord {
                    id: wfcb_wire,
                    status: CheckpointStatus::Started,
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
                },
            ),
            // Callback: Started (pending: will suspend on .result()).
            (
                cb_wire.clone(),
                CheckpointRecord {
                    id: cb_wire,
                    status: CheckpointStatus::Started,
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
                    callback_id: Some("cb-replay-test".to_owned()),
                    op_type: None,
                    sub_type: None,
                    due_at: None,
                    op_name: None,
                    parent_id: None,
                    start_timestamp: None,
                    end_timestamp: None,
                },
            ),
            // Submitter step: Succeeded (replay: should not re-invoke).
            (
                step_wire.clone(),
                CheckpointRecord {
                    id: step_wire,
                    status: CheckpointStatus::Succeeded,
                    result: Some("null".to_owned()),
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
                },
            ),
        ]);

        let submitter_called = Arc::new(AtomicBool::new(false));
        let submitter_called_clone = Arc::clone(&submitter_called);

        let op_id = ctx.mint_id();
        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("replay-test".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: move |_sc: StepContext, _id: String| {
                submitter_called_clone.store(true, Ordering::SeqCst);
                async { Ok::<(), BoxError>(()) }
            },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };

        // Execution should suspend at callback.result() since the callback
        // is pending. But the submitter must NOT have been re-invoked.
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            Arc::clone(ctx.coordinator()),
            exec.execute(),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        assert!(
            !submitter_called.load(Ordering::SeqCst),
            "submitter should NOT be re-invoked during replay of a succeeded step"
        );
    }

    #[tokio::test]
    async fn wfcb_submitter_failure_checkpoints_step_and_propagates() {
        // When the submitter fails, the step should checkpoint failure and
        // the error should propagate as ChildFailed through the context.
        // Verify that checkpoints are produced (StepStart + StepFail/Retry).
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();

        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("fail-check".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async {
                Err::<(), BoxError>("submission failed".into())
            },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };

        // The submitter step fails; the default retry strategy schedules a
        // retry (backend-owned timer), which suspends the invocation instead
        // of surfacing a fabricated error to the caller.
        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            exec.execute(),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        // Verify checkpoint calls were made for the step execution path:
        //   1. ContextStarted
        //   2. CallbackStarted
        //   3. StepStarted
        //   4. StepRetry or StepFail (depends on default retry strategy)
        //   5+. Possible ContextFailed
        let call_count = *client
            .checkpoint_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            call_count >= 4,
            "expected at least 4 checkpoint calls for failed submitter path, got {call_count}"
        );
    }

    #[tokio::test]
    async fn wfcb_live_failure_carries_checkpointed_context() {
        // A live wait-for-callback failure that reaches the terminal
        // ContextFailed checkpoint must return an error carrying the
        // checkpointed context: the same operation id, status, and wire
        // record a replay of that record reconstructs.
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();
        let expected_wire_id = op_id.wire().to_owned();

        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("live-fail-context".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async {
                Err::<(), BoxError>("submission failed".into())
            },
            // No retries: the step failure is terminal, so the body's
            // error reaches the ContextFailed settle path live.
            submitter_retry: Some(Box::new(|_err: &crate::StepError, _attempt: u32| {
                crate::RetryDecision::Stop
            })),
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };

        let err = exec.execute().await.unwrap_err();
        // The body returned, so the finished child is abandoned: a later
        // write from inside it would be rejected.
        assert!(
            !ctx.admission()
                .admits(std::slice::from_ref(&expected_wire_id)),
            "a finished wait_for_callback child is abandoned"
        );
        // Live/replay parity: the checkpointed context is attached.
        assert_eq!(err.operation_id(), Some(expected_wire_id.as_str()));
        assert_eq!(err.status(), Some("FAILED"));
        let wire = err.wire().expect("live wfcb failure carries wire record");
        assert!(
            wire.error_message()
                .is_some_and(|m| m.contains("submission failed")),
            "wire message flattens the body failure: {wire:?}"
        );
    }

    #[tokio::test]
    async fn wfcb_submitter_receives_step_context_with_attempt() {
        // Verify the submitter closure receives a StepContext with
        // the correct attempt number (from the step execution path).
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();

        let received_attempt = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let received_attempt_clone = Arc::clone(&received_attempt);

        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("attempt-check".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: move |sc: StepContext, _id: String| {
                received_attempt_clone.store(sc.attempt(), std::sync::atomic::Ordering::SeqCst);
                async { Ok::<(), BoxError>(()) }
            },
            submitter_retry: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };

        // Execute: will suspend at callback result, but submitter runs.
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            Arc::clone(ctx.coordinator()),
            exec.execute(),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        // First invocation: attempt should be 1 (1-based from StepExecution).
        let attempt = received_attempt.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(
            attempt, 1,
            "submitter should receive attempt=1 on first invocation, got {attempt}"
        );
    }

    // Callback payload serdes tests

    /// Test serdes that expects a non-JSON `MARK:` prefix on the wire payload
    /// and strips it on the deserialize side. Plain `serde_json` cannot decode
    /// a `MARK:`-prefixed payload, so a successful decode proves this serdes
    /// was actually consulted on the callback path.
    #[derive(Debug)]
    struct MarkerSerdes;

    impl<T> Serdes<T> for MarkerSerdes
    where
        T: Serialize + DeserializeOwned + Send + 'static,
    {
        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn serialize(&self, value: T, _context: SerdesContext) -> Result<String, BoxError> {
            Ok(format!("MARK:{}", serde_json::to_string(&value)?))
        }

        // reason: exercises the async-fn impl form user code writes
        #[expect(clippy::unused_async_trait_impl)]
        async fn deserialize(&self, wire: String, _context: SerdesContext) -> Result<T, BoxError> {
            let body = wire
                .strip_prefix("MARK:")
                .ok_or_else(|| -> BoxError { "missing MARK: prefix".into() })?;
            Ok(serde_json::from_str(body)?)
        }
    }

    #[tokio::test]
    async fn callback_per_op_serdes_decodes_non_json_payload() {
        // A per-op serdes decodes a payload that plain JSON cannot parse.
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#"MARK:"hello""#),
            None,
            None,
            Some("cb-marker"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: MarkerSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should settle");
        let value = cb.result().await.expect("per-op serdes should decode");
        assert_eq!(value, "hello");
    }

    /// The shared probe serdes, the same type the map/parallel item-serdes
    /// equivalence test uses, must decode a callback payload too. One
    /// implementation, every operation path: that is the point of the
    /// normalized serialization model.
    #[tokio::test]
    async fn shared_probe_serdes_decodes_callback_payload() {
        use crate::serdes::test_support::{HexEnvelopeSerdes, hex_envelope};

        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        struct Doc {
            label: String,
            nested: Vec<Vec<i64>>,
        }

        let doc = Doc {
            label: "quote:\" backslash:\\ newline:\n tab:\t ünïcodé ☃".to_owned(),
            nested: vec![vec![1, -2, i64::MIN], Vec::new()],
        };
        let wire = hex_envelope(&serde_json::to_string(&doc).expect("doc is JSON-able"));
        // Control: the stored payload is not JSON, so a decode can only
        // succeed if the serdes transform was reversed.
        assert!(serde_json::from_str::<Doc>(&wire).is_err());

        let op_wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &op_wire,
            CheckpointStatus::Succeeded,
            Some(&wire),
            None,
            None,
            Some("cb-probe"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<Doc, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: HexEnvelopeSerdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should settle");
        let value = cb.result().await.expect("probe serdes should decode");
        assert_eq!(value, doc);
    }

    /// ISSUE #46: an externally delivered callback payload must never be
    /// honored as a `FileSystemSerdes` file reference. A payload shaped
    /// like a file pointer, even one naming a real, readable file under
    /// `base_path`, decodes as plain data, and a realistic inline payload
    /// containing a `file` key is not misparsed.
    #[tokio::test]
    #[expect(clippy::indexing_slicing)] // reason: test assertions on known JSON keys
    async fn callback_payload_never_resolves_file_references() {
        use crate::serdes::FileSystemSerdes;

        let tmp = std::env::temp_dir().join(format!(
            "cb_no_file_refs_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).expect("create temp base");
        std::fs::write(tmp.join("secret.json"), r#""stolen contents""#).expect("plant file");

        let filesystem_serdes = FileSystemSerdes::new(tmp.to_string_lossy().into_owned());

        // The external caller delivered a payload that LOOKS like a file
        // pointer (an unversioned file pointer). It must come back as data.
        let payload = format!(
            r#"{{"file":"{}","data":{{"status":"ready"}}}}"#,
            tmp.join("secret.json").to_string_lossy()
        );
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(&payload),
            None,
            None,
            Some("cb-ext"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<serde_json::Value, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: filesystem_serdes,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should settle");
        let value = cb.result().await.expect("external payload decodes as data");

        let _ = std::fs::remove_dir_all(&tmp);

        assert_eq!(
            value["data"],
            serde_json::json!({"status": "ready"}),
            "inline payload with a 'file' key must not be misparsed"
        );
        assert_eq!(
            value["file"],
            serde_json::json!(tmp.join("secret.json").to_string_lossy()),
            "the 'file' key is plain data"
        );
        assert_ne!(
            value,
            serde_json::json!("stolen contents"),
            "an external payload must never trigger a local file read"
        );
    }

    #[tokio::test]
    async fn callback_marker_payload_fails_without_serdes() {
        // Control: the same MARK:-prefixed payload is NOT valid JSON, so with
        // no serdes (and no execution-wide serdes) the decode must fail. This
        // proves the serdes is what makes the override/fallback tests pass.
        let wire = crate::engine::compute_wire_id_public("1");
        let ctx = ctx_with_log(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#"MARK:"hello""#),
            None,
            None,
            Some("cb-marker"),
        )]);
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String, _> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            sub_type: crate::OperationSubType::CALLBACK,
            plugin_name: None,
            serdes: crate::serdes::JsonSerdes,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::DeserializationFailed)
        ));
    }

    #[tokio::test]
    async fn wfcb_per_op_serdes_threads_to_inner_decode() {
        // A serdes set on wait_for_callback must reach the inner callback's
        // decode of the delivered payload. The MARK:-prefixed payload is not
        // valid JSON, so a correct result proves the per-op serdes was
        // threaded through to the inner decode (the Go non-wiring quirk is
        // deliberately not replicated).
        let wfcb_wire = crate::engine::compute_wire_id_public("1");
        let cb_wire = crate::engine::compute_wire_id_public("1-1");
        let step_wire = crate::engine::compute_wire_id_public("1-2");

        let ctx = ctx_with_log(vec![
            // WaitForCallback context: Started (non-terminal, fall through).
            callback_record(
                &wfcb_wire,
                CheckpointStatus::Started,
                None,
                None,
                None,
                None,
            ),
            // Inner callback: Succeeded with a non-JSON marker payload.
            callback_record(
                &cb_wire,
                CheckpointStatus::Succeeded,
                Some(r#"MARK:"wfcb""#),
                None,
                None,
                Some("cb-wfcb"),
            ),
            // Submitter step: Succeeded (replay, not re-invoked).
            callback_record(
                &step_wire,
                CheckpointStatus::Succeeded,
                Some("null"),
                None,
                None,
                None,
            ),
        ]);
        let op_id = ctx.mint_id();
        let exec = WaitForCallbackExecution::<String, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("wfcb-serdes".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: MarkerSerdes,
            _marker: std::marker::PhantomData,
        };
        let result = exec
            .execute()
            .await
            .expect("wfcb should decode via threaded serdes");
        assert_eq!(result, "wfcb");
    }
}

#[cfg(test)]
mod coordinator_tests {
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{
        ScriptedClient, drive_handler, named, operation_like, scripted_ctx, wire_err,
    };
    use aws_sdk_lambda::types::{
        CallbackDetails, ErrorObject, Operation, OperationAction, OperationStatus, OperationType,
        OperationUpdate,
    };
    use std::sync::Arc;

    /// The record a callback START gets: `STARTED` with its callback ID.
    fn started(start: &OperationUpdate) -> Operation {
        operation_like(start, OperationStatus::Started)
            .callback_details(CallbackDetails::builder().callback_id("cb-1").build())
            .build()
            .expect("all required fields set")
    }

    /// A client that answers the callback START with its ID, and carries
    /// `completed` for the callback on the call that carries step `s`'s
    /// SUCCEED.
    fn client_completing(
        completed: impl Fn(&OperationUpdate) -> Operation + Send + Sync + 'static,
    ) -> ScriptedClient {
        ScriptedClient::new(move |updates, history| {
            let mut ops = Vec::new();
            for update in updates {
                if update.r#type() == &OperationType::Callback
                    && update.action() == &OperationAction::Start
                {
                    ops.push(started(update));
                }
            }
            if named(updates, "s", &OperationAction::Succeed).is_some()
                && let Some(start) = named(history, "cb", &OperationAction::Start)
            {
                ops.push(completed(start));
            }
            ops
        })
    }

    fn succeeded(start: &OperationUpdate) -> Operation {
        operation_like(start, OperationStatus::Succeeded)
            .callback_details(
                CallbackDetails::builder()
                    .callback_id("cb-1")
                    .result("\"ok\"")
                    .build(),
            )
            .build()
            .expect("all required fields set")
    }

    fn declined(start: &OperationUpdate) -> Operation {
        operation_like(start, OperationStatus::Failed)
            .callback_details(
                CallbackDetails::builder()
                    .callback_id("cb-1")
                    .error(
                        ErrorObject::builder()
                            .error_type("Declined")
                            .error_message("no")
                            .build(),
                    )
                    .build(),
            )
            .build()
            .expect("all required fields set")
    }

    /// Runs the callback scenario against `client` and returns the
    /// callback's outcome.
    async fn join_callback_with_step(
        client: &Arc<ScriptedClient>,
    ) -> (
        InvocationOutcome,
        Option<Result<String, crate::OperationError>>,
    ) {
        let ctx = scripted_ctx(client);
        let seen = Arc::new(std::sync::Mutex::new(None));
        let handler_seen = Arc::clone(&seen);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let (result, step) = tokio::join!(
                cb.result(),
                ctx.step(|_| async { Ok(2_u32) }).name("s").future()
            );
            step.map_err(wire_err)?;
            *handler_seen.lock().expect("lock") = Some(result);
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        let result = seen.lock().expect("lock").take();
        (outcome, result)
    }

    #[tokio::test(start_paused = true)]
    async fn callback_resumes_on_sibling_response() {
        let (events, _guard) = crate::driver::test_support::EventMessages::install();
        let client = Arc::new(client_completing(succeeded));
        let (outcome, result) = join_callback_with_step(&client).await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        let value = result
            .expect("the handler finished")
            .expect("the callback succeeded");
        assert_eq!(value, "ok");
        assert_eq!(
            events.replayed(),
            0,
            "a resumed operation emits no replay event"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn resumed_callback_failure_matches_replay() {
        let client = Arc::new(client_completing(declined));
        let (outcome, result) = join_callback_with_step(&client).await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        let live = result
            .expect("the handler finished")
            .expect_err("the callback was declined");

        // The create-time replay of the same record.
        let start = named(&client.updates(), "cb", &OperationAction::Start)
            .cloned()
            .expect("the callback started");
        let log = crate::client::operations_to_checkpoint_log(&[declined(&start)]);
        let replay_ctx = crate::DurableContext::new_root(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
        );
        let replayed = replay_ctx
            .create_callback::<String>()
            .name("cb")
            .await
            .expect("the record replays")
            .result()
            .await
            .expect_err("the replayed callback was declined");
        assert_eq!(
            format!("{:?}", live.kind()),
            format!("{:?}", replayed.kind())
        );
        assert_eq!(live.wire(), replayed.wire());
    }

    #[tokio::test(start_paused = true)]
    async fn spawn_result_settled_and_direct_result() {
        // (a) A settled callback: the stored outcome, and no task.
        let start = OperationUpdate::builder()
            .id(crate::engine::compute_wire_id_public("1"))
            .r#type(OperationType::Callback)
            .sub_type(super::CALLBACK_SUB_TYPE)
            .action(OperationAction::Start)
            .name("cb")
            .build()
            .expect("all required fields set");
        let log = crate::client::operations_to_checkpoint_log(&[succeeded(&start)]);
        let ctx = crate::DurableContext::new_root(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
        );
        let tasks = Arc::clone(ctx.coordinator().tasks());
        let cb = ctx
            .create_callback::<String>()
            .name("cb")
            .await
            .expect("the record replays");
        let handle = cb.spawn_result();
        assert_eq!(tasks.live(), 0, "a settled callback starts no task");
        let value = handle.await.expect("the callback succeeded");
        assert_eq!(value, "ok");
        assert_eq!(tasks.live(), 0);

        // (b) A pending callback awaited directly is refreshed without a
        // first-poll flag.
        let client = Arc::new(crate::driver::test_support::RefreshClient::new(|req| {
            let mut ops = crate::driver::test_support::callback_starts(req.updates);
            if req.empty_number == 1
                && let Some(start) = named(req.history, "cb", &OperationAction::Start)
            {
                ops.push(succeeded(start));
            }
            Ok(ops)
        }));
        let ctx = crate::driver::test_support::refresh_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    Ok(1_u32)
                })
                .name("s")
                .future();
            let (result, step) = tokio::join!(cb.result(), step);
            step.map_err(wire_err)?;
            result.map_err(wire_err)
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("ok".to_owned()));
        crate::driver::test_support::assert_times(
            &client.empty_calls(),
            &[std::time::Duration::from_secs(1)],
        );
    }

    /// `spawn_result()` on a thread with no Tokio runtime resolves to the
    /// error `.spawn()` returns.
    #[tokio::test]
    async fn spawn_result_outside_runtime_returns_error() {
        let client = Arc::new(ScriptedClient::new(|updates, _| {
            updates
                .iter()
                .filter(|u| u.r#type() == &OperationType::Callback)
                .map(started)
                .collect()
        }));
        let ctx = scripted_ctx(&client);
        let cb = ctx
            .create_callback::<String>()
            .name("cb")
            .await
            .expect("the callback starts");
        let polled =
            crate::task::test_support::poll_once_without_runtime(move || cb.spawn_result());
        let std::task::Poll::Ready(Err(err)) = polled else {
            unreachable!("expected an immediate error, got {polled:?}");
        };
        assert!(
            format!("{err:#}").contains(crate::task::NO_RUNTIME_MESSAGE),
            "unexpected error: {err:#}"
        );
    }

    /// The first unnamed step update with `action` among `updates`: the
    /// submitter step of a `wait_for_callback`.
    fn submitter<'a>(
        updates: &'a [OperationUpdate],
        action: &OperationAction,
    ) -> Option<&'a OperationUpdate> {
        updates.iter().find(|u| {
            u.r#type() == &OperationType::Step && u.name().is_none() && u.action() == action
        })
    }

    /// The submitter step record of `update` with `status` after attempt 1.
    fn submitter_record(
        req: &crate::driver::test_support::RefreshRequest<'_>,
        update: &OperationUpdate,
        status: OperationStatus,
        next_in: Option<std::time::Duration>,
    ) -> Operation {
        let mut details = aws_sdk_lambda::types::StepDetails::builder().attempt(1);
        if let Some(next_in) = next_in {
            details = details.next_attempt_timestamp(req.timestamp_in(next_in));
        }
        operation_like(update, status)
            .step_details(details.build())
            .build()
            .expect("all required fields set")
    }

    #[tokio::test(start_paused = true)]
    async fn submitter_retry_runs_in_invocation() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::time::Duration;
        let client = Arc::new(crate::driver::test_support::RefreshClient::new(|req| {
            let mut ops = crate::driver::test_support::callback_starts(req.updates);
            if let Some(retry) = submitter(req.updates, &OperationAction::Retry) {
                ops.push(submitter_record(
                    req,
                    retry,
                    OperationStatus::Pending,
                    Some(Duration::from_secs(1)),
                ));
            }
            if req.empty_number == 1
                && let Some(retry) = submitter(req.history, &OperationAction::Retry)
            {
                ops.push(submitter_record(req, retry, OperationStatus::Ready, None));
            }
            if req.empty_number >= 2
                && let Some(start) = req.history.iter().find(|u| {
                    u.r#type() == &OperationType::Callback && u.action() == &OperationAction::Start
                })
            {
                ops.push(succeeded(start));
            }
            Ok(ops)
        }));
        let ctx = crate::driver::test_support::refresh_ctx(&client);
        let runs = Arc::new(AtomicU32::new(0));
        let submitter_runs = Arc::clone(&runs);
        let outcome = drive_handler(ctx, None, move |ctx| async move {
            let wfcb = ctx
                .wait_for_callback::<String, _, _>(move |_, _callback_id| {
                    let runs = Arc::clone(&submitter_runs);
                    async move {
                        if runs.fetch_add(1, Ordering::SeqCst) == 0 {
                            return Err("transient".into());
                        }
                        Ok(())
                    }
                })
                .name("wfcb")
                .submitter_retry(|_err, attempt| {
                    if attempt < 2 {
                        crate::RetryDecision::Retry {
                            delay: Duration::from_secs(1),
                        }
                    } else {
                        crate::RetryDecision::Stop
                    }
                })
                .future();
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    Ok(1_u32)
                })
                .name("s")
                .future();
            let (result, step) = tokio::join!(wfcb, step);
            step.map_err(wire_err)?;
            result.map_err(wire_err)
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("ok".to_owned()));
        assert_eq!(runs.load(Ordering::SeqCst), 2, "the submitter ran twice");
    }

    /// A refresh client that answers the callback `cb`'s START with its
    /// ID, and answers the first refresh with `completed` for it.
    fn refresh_completing(
        completed: impl Fn(&OperationUpdate) -> Operation + Send + Sync + 'static,
    ) -> crate::driver::test_support::RefreshClient {
        crate::driver::test_support::RefreshClient::new(move |req| {
            if let Some(start) = named(req.updates, "cb", &OperationAction::Start) {
                return Ok(vec![started(start)]);
            }
            if req.empty_number == 1
                && let Some(start) = named(req.history, "cb", &OperationAction::Start)
            {
                return Ok(vec![completed(start)]);
            }
            Ok(Vec::new())
        })
    }

    /// Runs `create_callback` named `cb` with a recording plugin, and
    /// joins `cb.result()` with a step that sleeps 3 s. Returns the
    /// outcome, the callback's result, and the number of hook records of
    /// `cb` when `result()` returned.
    async fn join_result_with_sleeping_step(
        client: crate::driver::test_support::RefreshClient,
        hooks: &crate::plugin::test_support::OpHooks,
    ) -> (
        InvocationOutcome,
        Option<(Result<String, crate::OperationError>, usize)>,
    ) {
        let ctx = crate::DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::new(client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            None,
            Some(hooks.plugins()),
        );
        let seen = Arc::new(std::sync::Mutex::new(None));
        let handler_seen = Arc::clone(&seen);
        let handler_hooks = hooks.clone();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let cb = ctx
                .create_callback::<String>()
                .name("cb")
                .await
                .map_err(wire_err)?;
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    Ok(1_u32)
                })
                .name("s")
                .spawn();
            let result = async {
                let result = cb.result().await;
                (result, handler_hooks.of("cb").len())
            };
            let (result, step) = tokio::join!(result, step);
            step.map_err(wire_err)?;
            *handler_seen.lock().expect("lock") = Some(result);
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        let result = seen.lock().expect("lock").take();
        (outcome, result)
    }

    /// A callback that the first refresh completes while `result()` waits
    /// fires its end hook live, before `result()` returns.
    #[tokio::test(start_paused = true)]
    async fn callback_resumed_in_invocation_ends_live() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let (outcome, result) =
            join_result_with_sleeping_step(refresh_completing(succeeded), &hooks).await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        let (value, hooks_at_return) = result.expect("the handler finished");
        assert_eq!(value.expect("the callback succeeded"), "ok");
        let shapes: Vec<_> = hooks
            .of("cb")
            .iter()
            .map(|r| (r.hook, r.replay, r.status))
            .collect();
        assert_eq!(
            shapes,
            [
                ("start", false, Some(crate::OperationStatus::Started)),
                ("end", false, Some(crate::OperationStatus::Succeeded)),
            ]
        );
        assert_eq!(
            hooks_at_return, 2,
            "the end hook ran before result() returned"
        );
    }

    /// A callback that a refresh fails while `result()` waits fires its end
    /// hook live with the status `Failed`.
    #[tokio::test(start_paused = true)]
    async fn callback_failed_in_invocation_ends_live() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let (outcome, result) =
            join_result_with_sleeping_step(refresh_completing(declined), &hooks).await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        let (value, _) = result.expect("the handler finished");
        value.expect_err("the callback was declined");
        let shapes: Vec<_> = hooks
            .of("cb")
            .iter()
            .map(|r| (r.hook, r.replay, r.status))
            .collect();
        assert_eq!(
            shapes,
            [
                ("start", false, Some(crate::OperationStatus::Started)),
                ("end", false, Some(crate::OperationStatus::Failed)),
            ]
        );
    }

    /// A callback whose record a refresh changes to `CANCELLED` while
    /// `result()` waits fires no end hook.
    #[tokio::test(start_paused = true)]
    async fn callback_cancelled_in_invocation_fires_no_end() {
        let hooks = crate::plugin::test_support::OpHooks::default();
        let cancelled = |start: &OperationUpdate| {
            operation_like(start, OperationStatus::Cancelled)
                .callback_details(CallbackDetails::builder().callback_id("cb-1").build())
                .build()
                .expect("all required fields set")
        };
        let (outcome, result) =
            join_result_with_sleeping_step(refresh_completing(cancelled), &hooks).await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        let (value, _) = result.expect("the handler finished");
        let err = value.expect_err("a cancelled callback is unexpected");
        assert!(
            format!("{err:?}").contains("unexpected checkpointed status"),
            "{err:?}"
        );
        let hooks_of_cb: Vec<_> = hooks.of("cb").iter().map(|r| r.hook).collect();
        assert_eq!(hooks_of_cb, ["start"]);
    }
}

/// The start order of `wait_for_callback`: its outcome waits for the
/// retained outer `START`, and its body starts before the `START`
/// response.
#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions
mod start_order_tests {
    use std::future::{Future, IntoFuture as _};
    use std::sync::Arc;
    use std::time::Duration;

    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    use super::WfcbAfter;
    use crate::client::InMemoryExecutionClient;
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use crate::error::{CallbackError, CallbackErrorKind, OperationError, OperationErrorKind};

    /// One submitted update the recorder saw: its name, wire ID and
    /// action.
    type Submission = (String, String, OperationAction);

    /// A context with direct writes over `client`, whose submissions
    /// `submissions` records, and with the plugin `hooks`.
    fn recorded_context(
        client: Arc<dyn crate::client::ExecutionClient>,
        submissions: &Arc<std::sync::Mutex<Vec<Submission>>>,
        hooks: Option<&crate::plugin::test_support::OpHooks>,
    ) -> DurableContext {
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(Vec::new())),
            client,
            "token-0".to_owned(),
            None,
            hooks.map(crate::plugin::test_support::OpHooks::plugins),
        );
        let recorded = Arc::clone(submissions);
        ctx.set_submission_recorder(Arc::new(move |updates, _flag| {
            let mut recorded = recorded.lock().unwrap();
            for u in updates {
                recorded.push((
                    u.name().unwrap_or("-").to_owned(),
                    u.id().to_owned(),
                    u.action().clone(),
                ));
            }
        }));
        ctx
    }

    /// Waits up to two seconds for `ready`, failing the test with `what`.
    async fn wait_until(what: &str, mut ready: impl FnMut() -> bool) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while !ready() {
            assert!(tokio::time::Instant::now() < deadline, "timed out: {what}");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Runs test 27: the `wait_for_callback` settler with a retained START
    /// that fails, after a body that returned `body_result`.
    async fn wfcb_outcome_waits_for_failed_start(
        body_result: Result<String, OperationError>,
        action: OperationAction,
    ) {
        let submissions: Arc<std::sync::Mutex<Vec<Submission>>> = Arc::default();
        let ctx = recorded_context(
            Arc::new(InMemoryExecutionClient::new(Vec::new())),
            &submissions,
            None,
        );
        let entries: Arc<std::sync::Mutex<Vec<(String, OperationAction)>>> = Arc::default();
        let entered = Arc::clone(&entries);
        ctx.set_settler_entry_recorder(Arc::new(move |wire_id, action| {
            entered.lock().unwrap().push((wire_id.to_owned(), action));
        }));
        let wire_id = crate::engine::compute_wire_id_public("1");
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let after: WfcbAfter<String> = WfcbAfter {
            ctx: ctx.clone(),
            wire_id: wire_id.clone(),
            name: Some("wcb".to_owned()),
            _marker: std::marker::PhantomData,
        };
        let settle = after.settle(
            Some(crate::context::SubmittedWrite::from_receiver(receiver)),
            body_result,
        );
        let outcomes = |submissions: &[Submission]| {
            submissions
                .iter()
                .filter(|(_, id, action)| {
                    *id == wire_id
                        && matches!(action, OperationAction::Succeed | OperationAction::Fail)
                })
                .count()
        };
        let observe = async {
            wait_until("the settler-entry marker", || {
                entries
                    .lock()
                    .unwrap()
                    .contains(&(wire_id.clone(), action.clone()))
            })
            .await;
            assert_eq!(outcomes(&submissions.lock().unwrap()), 0, "no outcome yet");
            let rejection = crate::client::ExecutionClientError::new(
                crate::client::ExecutionClientErrorScope::Execution,
                "START rejected",
            );
            let wire = crate::error::checkpoint_execution_failure_wire(&rejection);
            sender.send(Err(rejection)).expect("the settler waits");
            wait_until("the invocation's fatal error", || {
                ctx.coordinator().fatal_error().is_some()
            })
            .await;
            let fatal = ctx.coordinator().fatal_error().expect("checked above");
            assert_eq!(fatal.error_type, "CheckpointUnrecoverableExecutionError");
            assert_eq!(Some(fatal.error_message.as_str()), wire.error_message());
            for _ in 0..10 {
                tokio::task::yield_now().await;
            }
            assert_eq!(outcomes(&submissions.lock().unwrap()), 0, "no outcome");
        };
        tokio::select! {
            _ = settle => panic!("the settler resolved after a failed START"),
            () = observe => {}
        }
    }

    #[tokio::test]
    async fn wait_for_callback_success_waits_for_failed_start() {
        wfcb_outcome_waits_for_failed_start(Ok("ok".to_owned()), OperationAction::Succeed).await;
    }

    #[tokio::test]
    async fn wait_for_callback_failure_waits_for_failed_start() {
        let failed = Err(OperationError::from_kind(OperationErrorKind::Callback(
            CallbackError::new(CallbackErrorKind::ExternalFailure, Some("declined".into())),
        )));
        wfcb_outcome_waits_for_failed_start(failed, OperationAction::Fail).await;
    }

    /// A client that logs `response <name>` for each `START` it accepts
    /// into `log`, and answers a callback `START` with its callback ID.
    struct ResponseLog(Arc<std::sync::Mutex<Vec<String>>>);

    impl crate::client::ExecutionClient for ResponseLog {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> std::pin::Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::client::CheckpointOutput,
                            crate::client::ExecutionClientError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            let mut records = Vec::new();
            {
                let mut log = self.0.lock().unwrap();
                for u in updates
                    .iter()
                    .filter(|u| u.action() == &OperationAction::Start)
                {
                    log.push(format!("response {}", u.name().unwrap_or("-")));
                    if u.r#type() == &OperationType::Callback {
                        records.push(
                            aws_sdk_lambda::types::Operation::builder()
                                .id(u.id())
                                .r#type(OperationType::Callback)
                                .status(aws_sdk_lambda::types::OperationStatus::Started)
                                .start_timestamp(aws_smithy_types::DateTime::from_secs(0))
                                .callback_details(
                                    aws_sdk_lambda::types::CallbackDetails::builder()
                                        .callback_id("cb-id")
                                        .build(),
                                )
                                .build()
                                .expect("every required Operation field is set"),
                        );
                    }
                }
            }
            Box::pin(async move {
                Ok(crate::client::CheckpointOutput::new(
                    Some("t".to_owned()),
                    records,
                    None,
                ))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> std::pin::Pin<
            Box<
                dyn Future<
                        Output = Result<
                            crate::client::GetStateOutput,
                            crate::client::ExecutionClientError,
                        >,
                    > + Send
                    + '_,
            >,
        > {
            Box::pin(async { Ok(crate::client::GetStateOutput::new(Vec::new(), None)) })
        }
    }

    /// The body of a `wait_for_callback` starts before its outer `START`
    /// response: while the outer `START` is held, the body submits the
    /// inner callback's `START`, and the order is hook, marker, response.
    #[tokio::test]
    async fn wait_for_callback_body_starts_before_start_response() {
        let log: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
        let probe_log = Arc::clone(&log);
        // Each hook record carries the log length when the hook fired.
        let hooks = crate::plugin::test_support::OpHooks::probing(move || {
            probe_log.lock().unwrap().len().to_string()
        });
        let submissions: Arc<std::sync::Mutex<Vec<Submission>>> = Arc::default();
        let ctx = recorded_context(
            Arc::new(ResponseLog(Arc::clone(&log))),
            &submissions,
            Some(&hooks),
        );
        let marker_log = Arc::clone(&log);
        let recorded = Arc::clone(&submissions);
        ctx.set_submission_recorder(Arc::new(move |updates, _flag| {
            for u in updates {
                if u.action() == &OperationAction::Start && u.r#type() == &OperationType::Callback {
                    marker_log
                        .lock()
                        .unwrap()
                        .push("inner callback START submitted".to_owned());
                }
                recorded.lock().unwrap().push((
                    u.name().unwrap_or("-").to_owned(),
                    u.id().to_owned(),
                    u.action().clone(),
                ));
            }
        }));
        let gate = Arc::new(tokio::sync::Notify::new());
        let held = Arc::clone(&gate);
        ctx.set_pre_lock_gate(Arc::new(move |updates: &[OperationUpdate]| {
            updates
                .iter()
                .any(|u| u.action() == &OperationAction::Start && u.name() == Some("wcb"))
                .then(|| Arc::clone(&held))
        }));
        let operation = ctx
            .wait_for_callback::<String, _, _>(|_, _id| async { Ok(()) })
            .name("wcb")
            .into_future();
        let marker = "inner callback START submitted";
        let observe = async {
            wait_until("the inner callback START", || {
                log.lock().unwrap().iter().any(|e| e == marker)
            })
            .await;
            assert!(
                !log.lock().unwrap().iter().any(|e| e == "response wcb"),
                "the body ran while the outer START was held: {:?}",
                log.lock().unwrap()
            );
            gate.notify_one();
            wait_until("the outer START response", || {
                log.lock().unwrap().iter().any(|e| e == "response wcb")
            })
            .await;
        };
        tokio::select! {
            _ = operation => {}
            () = observe => {}
        }
        let entries = log.lock().unwrap().clone();
        let index = |entry: &str| {
            entries
                .iter()
                .position(|e| e == entry)
                .unwrap_or_else(|| panic!("{entry:?} is missing from {entries:?}"))
        };
        let hook: usize = hooks
            .of("wcb")
            .into_iter()
            .find(|r| r.hook == "start")
            .expect("the start hook of wcb fired")
            .probe
            .parse()
            .expect("the probe is a log length");
        assert!(
            hook <= index(marker) && index(marker) < index("response wcb"),
            "hook at {hook}: {entries:?}"
        );
    }
}
