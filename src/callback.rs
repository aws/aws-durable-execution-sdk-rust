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
use std::sync::Arc;
use std::time::Duration;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tracing::Instrument as _;

use crate::BoxError;
use crate::context::{DurableContext, StepContext};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    CallbackError, CallbackErrorKind, ChildContextError, ChildContextErrorKind, OperationError,
    OperationErrorKind,
};
use crate::future::Callback;
use crate::serdes::{Serdes, SerdesContext};

/// Wire sub-type for callback operations.
pub(crate) const CALLBACK_SUB_TYPE: &str = "Callback";

/// Wire sub-type for wait-for-callback context operations.
pub(crate) const WFCB_SUB_TYPE: &str = "WaitForCallback";

// ────────────────────────────────────────────────────────────────────────────
// CreateCallbackExecution
// ────────────────────────────────────────────────────────────────────────────

/// Internal state for `create_callback` execution passed from the builder.
pub(crate) struct CreateCallbackExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) heartbeat: Option<Duration>,
    pub(crate) serdes: Option<Arc<dyn Serdes>>,
    pub(crate) _marker: std::marker::PhantomData<O>,
}

impl<O: DeserializeOwned + Send + 'static> CreateCallbackExecution<O> {
    /// Executes the create-callback operation: replay or live path.
    #[allow(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
    pub(crate) async fn execute(self) -> Result<Callback<O>, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // 2. Check checkpoint log for replay. Every branch consumes the
        // backend-assigned callback ID; the terminal branches additionally
        // fetch only the field they consume.
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Callback",
            Some(CALLBACK_SUB_TYPE),
            self.name.as_deref(),
        )? {
            let callback_id = self
                .ctx
                .checkpoint_callback_id(&positional_id)
                .unwrap_or_default();
            // Terminal statuses replay the recorded outcome without waiting
            // on the external system again (see `crate::observability`).
            let emit_replayed = || {
                self.ctx.emit_operation_replayed(
                    &wire_id,
                    self.name.as_deref(),
                    "Callback",
                    Some(CALLBACK_SUB_TYPE),
                    view.attempt,
                );
            };
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Callback completed successfully during a previous
                    // invocation — return settled Callback with the result.
                    // The payload was written by an external caller, so only
                    // the deserialize side of the serdes acts on it. Per-op
                    // serdes wins; otherwise the execution-wide serdes decodes
                    // it (default JSON). Decode FIRST, then emit
                    // `operation_replayed`: a corrupt payload or failing
                    // serdes surfaces as an error without claiming a recorded
                    // outcome was returned.
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let result_str = payload.as_deref().unwrap_or("null");
                    let serdes_ctx =
                        SerdesContext::new(self.op_id.wire(), self.ctx.execution_arn());
                    let value: O = deserialize_callback_result(
                        self.serdes.as_ref().or_else(|| self.ctx.default_serdes()),
                        result_str,
                        &serdes_ctx,
                    )
                    .await?;
                    emit_replayed();
                    return Ok(Callback::new_settled(callback_id, Ok(value)));
                }
                CheckpointStatus::Failed => {
                    // Callback failed externally.
                    emit_replayed();
                    let (error_type, error_message) = self
                        .ctx
                        .checkpoint_error_parts(&positional_id)
                        .unwrap_or_default();
                    let error_type = error_type.unwrap_or_else(|| "Error".to_owned());
                    let error_message =
                        error_message.unwrap_or_else(|| "callback failed".to_owned());
                    let kind = classify_callback_error(&error_type, &error_message);
                    let err = OperationError::from_kind(OperationErrorKind::Callback(
                        CallbackError::from_kind(kind),
                    ));
                    return Ok(Callback::new_settled(callback_id, Err(err)));
                }
                CheckpointStatus::TimedOut => {
                    // Callback timed out.
                    emit_replayed();
                    let kind = CallbackErrorKind::TimedOut;
                    let err = OperationError::from_kind(OperationErrorKind::Callback(
                        CallbackError::from_kind(kind),
                    ));
                    return Ok(Callback::new_settled(callback_id, Err(err)));
                }
                CheckpointStatus::Started | CheckpointStatus::Pending => {
                    // Callback is in flight — return pending (will suspend on
                    // .result()).
                    return Ok(Callback::new_pending(callback_id, self.ctx.clone()));
                }
                _ => {
                    // Unexpected status — treat as internal error.
                    return Err(callback_internal_error(&format!(
                        "unexpected checkpointed status: {:?}",
                        view.status
                    )));
                }
            }
        }

        // 3. First invocation — checkpoint START with callback options.
        let mut builder = OperationUpdate::builder()
            .id(wire_id.clone())
            .r#type(OperationType::Callback)
            .sub_type(CALLBACK_SUB_TYPE.to_owned())
            .action(OperationAction::Start);

        if let Some(n) = &self.name {
            builder = builder.name(n.clone());
        }
        if let Some(parent_wire) = self.ctx.parent_wire_id_computed() {
            builder = builder.parent_id(parent_wire);
        }

        // Build callback options if timeout or heartbeat is set.
        let cb_opts = build_callback_options(self.timeout, self.heartbeat);
        if let Some(opts) = cb_opts {
            builder = builder.callback_options(opts);
        }

        #[allow(clippy::expect_used)] // reason: all required fields set above
        let update = builder
            .build()
            .expect("all required OperationUpdate fields set");

        // Callback creation is a flush point of the checkpoint-delay
        // contract: the backend assigns the callback_id in this response,
        // so the write cannot wait out a coalescing window.
        self.ctx
            .checkpoint_updates_urgent(vec![update])
            .await
            .map_err(|e| callback_internal_error(&format!("checkpoint start: {e}")))?;

        // After checkpointing START, the backend assigns a callback_id.
        // Read it from the (now-updated) checkpoint log.
        let callback_id = self
            .ctx
            .checkpoint_callback_id(&positional_id)
            .unwrap_or_default();

        // Return pending — Result() will fire suspend.
        Ok(Callback::new_pending(callback_id, self.ctx.clone()))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// WaitForCallbackExecution
// ────────────────────────────────────────────────────────────────────────────

/// Internal state for `wait_for_callback` execution passed from the builder.
///
/// Generic over the submitter closure `F` (and, through `F`'s output, its
/// future), so the submitter runs **without type erasure**. The one
/// erasure point is the builder's `.future()` / `into_future`, which boxes
/// the whole execution future once inside
/// [`DurableFuture`](crate::DurableFuture).
pub(crate) struct WaitForCallbackExecution<O, F> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) timeout: Option<Duration>,
    pub(crate) heartbeat: Option<Duration>,
    pub(crate) submitter: F,
    pub(crate) submitter_retry: Option<crate::RetryStrategy>,
    pub(crate) serdes: Option<Arc<dyn Serdes>>,
    pub(crate) _marker: std::marker::PhantomData<O>,
}

impl<O, F, Fut> WaitForCallbackExecution<O, F>
where
    O: DeserializeOwned + Serialize + Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    /// Executes the wait-for-callback operation: replay or live path.
    ///
    /// Thin generic wrapper — the ONLY code monomorphized per call site.
    /// The replay/checkpoint state machine lives in the non-generic
    /// [`WfcbCore`] / [`WfcbAfter`] halves (generic over the result type
    /// `O` only); this wrapper just runs the inner body (whose own
    /// checkpoint plumbing — callback creation and the submitter step —
    /// is likewise non-generic) between them.
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
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
        let core = WfcbCore {
            ctx,
            op_id,
            name,
            _marker: std::marker::PhantomData,
        };
        match core.before().await? {
            WfcbPrelude::Done(result) => result,
            WfcbPrelude::Run { child_ctx, after } => {
                // Run inner body: create_callback + submitter step + await
                // result. Instrumented with the child namespace's
                // replay-aware span so a resumed body's log lines are
                // suppressed while its nested operations replay.
                let child_span = child_ctx.replay_span();
                let body_result = run_wfcb_body::<O, _, _>(
                    child_ctx,
                    timeout,
                    heartbeat,
                    submitter,
                    submitter_retry,
                    serdes,
                )
                .instrument(child_span)
                .await;
                after.settle(body_result).await
            }
        }
    }
}

/// The pre-body half of `wait_for_callback`: task-ownership check, replay
/// resolution, and the START checkpoint. Generic only over the result type
/// `O` — no user closure reaches this state machine, so its replay and
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
    /// Runs everything that precedes the inner body: replay path, or the
    /// live-path preamble ending at the START checkpoint.
    async fn before(self) -> Result<WfcbPrelude<O>, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();
        let Self { ctx, name, .. } = self;

        // 2. Check checkpoint log for replay (context-level terminal). The
        // validated view covers the non-terminal branches without cloning.
        if let Some(view) = ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(WFCB_SUB_TYPE),
            name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // WaitForCallback context completed — deserialize the
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
                    // Context failed — classify the error.
                    ctx.emit_operation_replayed(
                        &wire_id,
                        name.as_deref(),
                        "Context",
                        Some(WFCB_SUB_TYPE),
                        view.attempt,
                    );
                    let (error_type, error_message) = ctx
                        .checkpoint_error_parts(&positional_id)
                        .unwrap_or_default();
                    return Ok(WfcbPrelude::Done(Err(wfcb_failed_error(
                        error_type.as_deref(),
                        error_message.as_deref(),
                    ))));
                }
                // Started/Pending/Ready: fall through to execute/resume.
                _ => {}
            }
        } else {
            // 3. First invocation — checkpoint ContextStarted.
            let update = build_wfcb_update(&wire_id, name.as_deref(), OperationAction::Start, &ctx);
            ctx.checkpoint_updates(vec![update])
                .await
                .map_err(|e| wfcb_internal_error(&format!("checkpoint start: {e}")))?;
        }

        // 4. Create child context with chained prefix.
        let child_ctx = ctx.new_child(&positional_id);

        Ok(WfcbPrelude::Run {
            child_ctx,
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
    async fn settle(self, body_result: Result<O, OperationError>) -> Result<O, OperationError> {
        let Self {
            ctx, wire_id, name, ..
        } = self;
        match body_result {
            Ok(value) => {
                // Success — serialize and checkpoint ContextSucceeded.
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
                if let Some(parent_wire) = ctx.parent_wire_id_computed() {
                    builder = builder.parent_id(parent_wire);
                }
                #[allow(clippy::expect_used)] // reason: all required fields set above
                let update = builder
                    .build()
                    .expect("all required OperationUpdate fields set");

                ctx.checkpoint_updates(vec![update])
                    .await
                    .map_err(|e| wfcb_internal_error(&format!("checkpoint succeed: {e}")))?;

                // Round-trip deserialize for consistency.
                let out: O = serde_json::from_str(&serialized)
                    .map_err(|e| wfcb_internal_error(&format!("deserialize result: {e}")))?;
                Ok(out)
            }
            Err(err) => {
                // Failure — checkpoint ContextFailed.
                let (err_type, err_msg) = extract_error_info(&err);
                let mut builder = OperationUpdate::builder()
                    .id(wire_id.clone())
                    .r#type(OperationType::Context)
                    .sub_type(WFCB_SUB_TYPE.to_owned())
                    .action(OperationAction::Fail);
                if let Some(n) = &name {
                    builder = builder.name(n.clone());
                }
                if let Some(parent_wire) = ctx.parent_wire_id_computed() {
                    builder = builder.parent_id(parent_wire);
                }
                builder = builder.error(
                    aws_sdk_lambda::types::ErrorObject::builder()
                        .error_type(err_type)
                        .error_message(err_msg)
                        .build(),
                );
                #[allow(clippy::expect_used)] // reason: all required fields set above
                let update = builder
                    .build()
                    .expect("all required OperationUpdate fields set");

                // Best-effort checkpoint — if it fails, return original error.
                let _ = ctx.checkpoint_updates(vec![update]).await;
                Err(err)
            }
        }
    }
}

/// Inner body of `wait_for_callback`: create callback, run submitter step,
/// await result.
///
/// The submitter is executed as a proper step operation (producing
/// `StepStarted`/`StepSucceeded`/`StepFailed` checkpoint events).
async fn run_wfcb_body<O, F, Fut>(
    child_ctx: DurableContext,
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
    submitter: F,
    submitter_retry: Option<crate::RetryStrategy>,
    serdes: Option<Arc<dyn Serdes>>,
) -> Result<O, OperationError>
where
    O: DeserializeOwned + Send + 'static,
    F: FnOnce(StepContext, String) -> Fut + Send + 'static,
    Fut: Future<Output = Result<(), BoxError>> + Send + 'static,
{
    // Step 1: create the inner callback (no name, per wire spec). The per-op
    // serdes flows through to the inner callback's decode so a serdes set on
    // wait_for_callback actually reaches the delivered payload.
    let cb_exec: CreateCallbackExecution<O> = CreateCallbackExecution {
        ctx: child_ctx.clone(),
        op_id: child_ctx.mint_id(),
        name: None,
        timeout,
        heartbeat,
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
        serdes: None,
        semantics: crate::step::StepSemantics::default(),
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
            ChildContextError::from_kind(ChildContextErrorKind::ChildFailed {
                message: e.to_string(),
            }),
        )));
    }

    // Step 3: await callback result.
    cb.result().await
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Builds the wire `CallbackOptions` from timeout/heartbeat durations.
/// Returns `None` when neither is set.
fn build_callback_options(
    timeout: Option<Duration>,
    heartbeat: Option<Duration>,
) -> Option<aws_sdk_lambda::types::CallbackOptions> {
    #[allow(clippy::cast_possible_truncation)] // reason: duration ≤ i32::MAX for practical timers
    #[allow(clippy::cast_sign_loss)] // reason: ceil is non-negative
    let timeout_secs = timeout.map_or(0, |d| {
        (d.as_secs_f64().ceil() as i64).min(i64::from(i32::MAX)) as i32
    });
    #[allow(clippy::cast_possible_truncation)] // reason: duration ≤ i32::MAX for practical timers
    #[allow(clippy::cast_sign_loss)] // reason: ceil is non-negative
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
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }
    #[allow(clippy::expect_used)] // reason: all required fields set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Classifies a callback error from the wire error type/message.
fn classify_callback_error(error_type: &str, error_message: &str) -> CallbackErrorKind {
    match error_type {
        "Callback.Timeout" => CallbackErrorKind::TimedOut,
        "Callback.Heartbeat" => CallbackErrorKind::HeartbeatTimedOut,
        _ => CallbackErrorKind::ExternalFailure {
            error_type: error_type.to_owned(),
            message: error_message.to_owned(),
        },
    }
}

/// Constructs the appropriate error for a failed `WaitForCallback` context.
fn wfcb_failed_error(error_type: Option<&str>, error_message: Option<&str>) -> OperationError {
    let err_type = error_type.unwrap_or("Error");
    let err_msg = error_message.unwrap_or("wait-for-callback failed");

    // Propagate callback errors directly (they bubble through the
    // child-context wrapping on the wire).
    if err_type == "Callback.Timeout"
        || err_type == "Callback.Heartbeat"
        || err_type == "CallbackError"
    {
        let kind = classify_callback_error(err_type, err_msg);
        return OperationError::from_kind(OperationErrorKind::Callback(CallbackError::from_kind(
            kind,
        )));
    }

    OperationError::from_kind(OperationErrorKind::ChildContext(
        ChildContextError::from_kind(ChildContextErrorKind::ChildFailed {
            message: err_msg.to_owned(),
        }),
    ))
}

/// Extracts error type and message from an `OperationError`.
fn extract_error_info(err: &OperationError) -> (String, String) {
    match err.kind() {
        OperationErrorKind::Callback(cb_err) => match cb_err.kind() {
            CallbackErrorKind::TimedOut => (
                "Callback.Timeout".to_owned(),
                "callback timed out".to_owned(),
            ),
            CallbackErrorKind::HeartbeatTimedOut => (
                "Callback.Heartbeat".to_owned(),
                "heartbeat timed out".to_owned(),
            ),
            CallbackErrorKind::ExternalFailure {
                error_type,
                message,
            } => (error_type.clone(), message.clone()),
            CallbackErrorKind::DeserializationFailed { message } => {
                ("DeserializationError".to_owned(), message.clone())
            }
            CallbackErrorKind::Internal { message } => {
                ("InternalError".to_owned(), message.clone())
            }
        },
        _ => ("Error".to_owned(), err.to_string()),
    }
}

/// Creates a callback deserialization error.
fn callback_deser_error(e: &serde_json::Error) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Callback(CallbackError::from_kind(
        CallbackErrorKind::DeserializationFailed {
            message: e.to_string(),
        },
    )))
}

/// Creates a callback deserialization error from a message.
fn callback_deser_error_msg(message: String) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Callback(CallbackError::from_kind(
        CallbackErrorKind::DeserializationFailed { message },
    )))
}

/// Decodes an externally-delivered callback payload into the target type.
///
/// The payload is produced by an external caller, so only the deserialize
/// side of the serdes is meaningful: `serdes` (when present) turns the wire
/// payload back into a `serde_json::Value`, which is then deserialized into
/// `O`. With no serdes the payload is parsed as JSON directly.
pub(crate) async fn deserialize_callback_result<O: DeserializeOwned>(
    serdes: Option<&Arc<dyn Serdes>>,
    payload: &str,
    serdes_ctx: &SerdesContext,
) -> Result<O, OperationError> {
    let Some(s) = serdes else {
        return serde_json::from_str(payload).map_err(|e| callback_deser_error(&e));
    };
    // Custom serdes may block (e.g. filesystem I/O): run off the runtime.
    let json_value = crate::serdes::deserialize_off_runtime(s, payload.to_owned(), serdes_ctx)
        .await
        .map_err(|e| callback_deser_error_msg(format!("callback serdes: {e}")))?;
    serde_json::from_value(json_value).map_err(|e| callback_deser_error(&e))
}

/// Creates a callback internal error.
fn callback_internal_error(msg: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Callback(CallbackError::from_kind(
        CallbackErrorKind::Internal {
            message: msg.to_owned(),
        },
    )))
}

/// Creates a wait-for-callback internal error.
fn wfcb_internal_error(msg: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::ChildContext(
        ChildContextError::from_kind(ChildContextErrorKind::Internal {
            message: msg.to_owned(),
        }),
    ))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
#[allow(clippy::panic)] // reason: test assertions with descriptive messages
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
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: callback_id.map(String::from),
                op_type: None,
                sub_type: None,
                op_name: None,
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("test".to_owned()),
            timeout: None,
            heartbeat: None,
            serdes: None,
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("racer".to_owned()),
            timeout: None,
            heartbeat: None,
            serdes: None,
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("joiner".to_owned()),
            timeout: None,
            heartbeat: None,
            serdes: None,
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: Some(Duration::from_secs(5)),
            heartbeat: None,
            serdes: None,
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: Some(Duration::from_secs(10)),
            serdes: None,
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            serdes: None,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should return callback");
        let err = cb.result().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::ExternalFailure { .. })
        ));
    }

    #[tokio::test]
    async fn live_returns_assigned_id() {
        // Empty checkpoint log — first invocation.
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client,
            "token-1".to_owned(),
        );
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("my-cb".to_owned()),
            timeout: Some(Duration::from_mins(1)),
            heartbeat: None,
            serdes: None,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should succeed");
        // The InMemoryClient doesn't assign callback IDs, so it's empty.
        assert_eq!(cb.id(), "");
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
        let exec = WaitForCallbackExecution::<String, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("wfcb-test".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: None,
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
        let exec = WaitForCallbackExecution::<String, _> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: None,
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
        let exec = WaitForCallbackExecution::<String, _> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: None,
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
        let exec = WaitForCallbackExecution::<String, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("sub-fail-test".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async {
                Err::<(), BoxError>("submitter exploded".into())
            },
            submitter_retry: None,
            serdes: None,
            _marker: std::marker::PhantomData,
        };
        // The submitter step fails; the default retry strategy schedules a
        // retry (backend-owned timer), which suspends the invocation instead
        // of surfacing a fabricated error to the caller.
        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn create_callback_spawn_executes() {
        // Verify that spawning a CreateCallbackBuilder returns a future
        // that resolves (live path — will checkpoint then return pending).
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
        // Live path returns empty callback_id from InMemoryClient.
        assert_eq!(cb.id(), "");
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
                let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
                    ctx: ctx_clone,
                    op_id,
                    name: None,
                    timeout: None,
                    heartbeat: None,
                    serdes: None,
                    _marker: std::marker::PhantomData,
                };
                exec.execute().await
            });

            handle.await.unwrap()
        })
        .await
        .unwrap();

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("task") || err_msg.contains("owner"),
            "expected ownership error, got: {err_msg}"
        );
    }

    // ── Submitter-as-step tests (verifies the fix for missing StepStarted/
    //    StepSucceeded events in the WaitForCallback child context) ──────

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

        let exec = WaitForCallbackExecution::<String, _> {
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
            serdes: None,
            _marker: std::marker::PhantomData,
        };

        // The callback result will suspend (pending), so the execution
        // will error at cb.result(). We just need to verify that the
        // submitter ran and checkpoints were produced.
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.suspension_signal()),
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
        //   execution suspends — but the submitter must NOT re-run.
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
                    attempt: 0,
                    invoke_result: None,
                    invoke_error_type: None,
                    invoke_error_message: None,
                    replay_children: false,
                    callback_id: None,
                    op_type: None,
                    sub_type: None,
                    op_name: None,
                },
            ),
            // Callback: Started (pending — will suspend on .result()).
            (
                cb_wire.clone(),
                CheckpointRecord {
                    id: cb_wire,
                    status: CheckpointStatus::Started,
                    result: None,
                    error_type: None,
                    error_message: None,
                    attempt: 0,
                    invoke_result: None,
                    invoke_error_type: None,
                    invoke_error_message: None,
                    replay_children: false,
                    callback_id: Some("cb-replay-test".to_owned()),
                    op_type: None,
                    sub_type: None,
                    op_name: None,
                },
            ),
            // Submitter step: Succeeded (replay — should not re-invoke).
            (
                step_wire.clone(),
                CheckpointRecord {
                    id: step_wire,
                    status: CheckpointStatus::Succeeded,
                    result: Some("null".to_owned()),
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
                },
            ),
        ]);

        let submitter_called = Arc::new(AtomicBool::new(false));
        let submitter_called_clone = Arc::clone(&submitter_called);

        let op_id = ctx.mint_id();
        let exec = WaitForCallbackExecution::<String, _> {
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
            serdes: None,
            _marker: std::marker::PhantomData,
        };

        // Execution should suspend at callback.result() since the callback
        // is pending. But the submitter must NOT have been re-invoked.
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.suspension_signal()),
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

        let exec = WaitForCallbackExecution::<String, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("fail-check".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async {
                Err::<(), BoxError>("submission failed".into())
            },
            submitter_retry: None,
            serdes: None,
            _marker: std::marker::PhantomData,
        };

        // The submitter step fails; the default retry strategy schedules a
        // retry (backend-owned timer), which suspends the invocation instead
        // of surfacing a fabricated error to the caller.
        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
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

        let exec = WaitForCallbackExecution::<String, _> {
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
            serdes: None,
            _marker: std::marker::PhantomData,
        };

        // Execute — will suspend at callback result, but submitter runs.
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.suspension_signal()),
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

    // ── Callback payload serdes tests ───────────────────────────────────

    /// Test serdes that expects a non-JSON `MARK:` prefix on the wire payload
    /// and strips it on the deserialize side. Plain `serde_json` cannot decode
    /// a `MARK:`-prefixed payload, so a successful decode proves this serdes
    /// was actually consulted on the callback path.
    #[derive(Debug)]
    struct MarkerSerdes;

    impl Serdes for MarkerSerdes {
        fn deserialize(
            &self,
            data: &str,
            _context: &SerdesContext,
        ) -> Result<serde_json::Value, BoxError> {
            let body = data
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            serdes: Some(Arc::new(MarkerSerdes)),
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should settle");
        let value = cb.result().await.expect("per-op serdes should decode");
        assert_eq!(value, "hello");
    }

    /// The shared probe serdes — the same type the map/parallel item-serdes
    /// equivalence test uses — must decode a callback payload too. One
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
        let exec: CreateCallbackExecution<Doc> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            serdes: Some(Arc::new(HexEnvelopeSerdes)),
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should settle");
        let value = cb.result().await.expect("probe serdes should decode");
        assert_eq!(value, doc);
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
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            serdes: None,
            _marker: std::marker::PhantomData,
        };
        let err = exec.execute().await.unwrap_err();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::Callback(e)
                if matches!(e.kind(), CallbackErrorKind::DeserializationFailed { .. })
        ));
    }

    #[tokio::test]
    async fn callback_falls_back_to_execution_wide_serdes() {
        // With no per-op serdes, the decode falls back to the execution-wide
        // serdes from Options.
        let wire = crate::engine::compute_wire_id_public("1");
        let log = CheckpointLog::from_records(vec![callback_record(
            &wire,
            CheckpointStatus::Succeeded,
            Some(r#"MARK:"world""#),
            None,
            None,
            Some("cb-fallback"),
        )]);
        let client = Arc::new(InMemoryExecutionClient::new(vec![]));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:123:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            client,
            "token-1".to_owned(),
            Some(Arc::new(MarkerSerdes)),
            None,
        );
        let op_id = ctx.mint_id();
        let exec: CreateCallbackExecution<String> = CreateCallbackExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            timeout: None,
            heartbeat: None,
            serdes: None,
            _marker: std::marker::PhantomData,
        };
        let cb = exec.execute().await.expect("should settle via fallback");
        let value = cb
            .result()
            .await
            .expect("execution-wide serdes should decode");
        assert_eq!(value, "world");
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
            // Submitter step: Succeeded (replay — not re-invoked).
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
        let exec = WaitForCallbackExecution::<String, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("wfcb-serdes".to_owned()),
            timeout: None,
            heartbeat: None,
            submitter: |_sc: StepContext, _id: String| async { Ok::<(), BoxError>(()) },
            submitter_retry: None,
            serdes: Some(Arc::new(MarkerSerdes)),
            _marker: std::marker::PhantomData,
        };
        let result = exec
            .execute()
            .await
            .expect("wfcb should decode via threaded serdes");
        assert_eq!(result, "wfcb");
    }
}
