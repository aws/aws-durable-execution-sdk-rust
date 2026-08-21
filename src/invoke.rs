//! Invoke operation execution engine.
//!
//! Implements the chained-invoke model: checkpoint `ChainedInvokeStarted`
//! with the target function name and serialized input, then suspend. The
//! backend performs the actual child invocation; on resume the SDK reads the
//! result from the checkpoint log.

use aws_sdk_lambda::types::{
    ChainedInvokeOptions, OperationAction, OperationType, OperationUpdate,
};

use crate::Serdes;
use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{InvokeError, InvokeErrorKind, OperationError, OperationErrorKind};
use crate::serdes::{PayloadOrigin, SerdesContext};

/// Wire sub-type for chained invoke operations.
pub(crate) const CHAINED_INVOKE_SUB_TYPE: &str = "ChainedInvoke";

/// Internal state for invoke execution passed from the builder.
///
/// Generic over the input type `I` (carried typed — the payload serdes
/// receives the owned input directly, a write-only transfer) and the two
/// serdes implementations: `PS` serializes the input payload, `RS`
/// deserializes the target function's result.
pub(crate) struct InvokeExecution<O, I, PS, RS> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) function_id: String,
    pub(crate) input: I,
    pub(crate) payload_serdes: PS,
    pub(crate) result_serdes: RS,
    pub(crate) tenant_id: Option<String>,
    pub(crate) _marker: std::marker::PhantomData<O>,
}

impl<O, I, PS, RS> InvokeExecution<O, I, PS, RS>
where
    O: Send + 'static,
    I: Send + 'static,
    PS: Serdes<I>,
    RS: Serdes<O>,
{
    /// Executes the invoke operation: replay path or live path.
    #[expect(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();
        let serdes_ctx = SerdesContext::new(&wire_id, self.ctx.execution_arn());

        // 2. Check checkpoint log for replay. The validated view covers the
        // non-terminal branches without cloning; the terminal branches fetch
        // only the invoke fields they consume.
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "ChainedInvoke",
            Some(CHAINED_INVOKE_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Invoke succeeded — deserialize the result from invoke
                    // details FIRST, then emit `operation_replayed`: a corrupt
                    // payload or failing serdes surfaces as an error without
                    // claiming a recorded outcome was returned.
                    let payload = self
                        .ctx
                        .with_checkpoint_record(&positional_id, |record| {
                            record.invoke_result.clone()
                        })
                        .flatten();
                    let value = deserialize_invoke_result(
                        &self.result_serdes,
                        payload.unwrap_or_else(|| "null".to_owned()),
                        serdes_ctx.clone(),
                    )
                    .await?;
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "ChainedInvoke",
                        Some(CHAINED_INVOKE_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(value);
                }
                CheckpointStatus::Failed
                | CheckpointStatus::TimedOut
                | CheckpointStatus::Stopped => {
                    // Invoke failed — reconstruct InvokeError from details.
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "ChainedInvoke",
                        Some(CHAINED_INVOKE_SUB_TYPE),
                        view.attempt,
                    );
                    let (error_type, error_message, error_data, stack_trace) = self
                        .ctx
                        .with_checkpoint_record(&positional_id, |record| {
                            (
                                record.invoke_error_type.clone(),
                                record.invoke_error_message.clone(),
                                record.invoke_error_data.clone(),
                                record.invoke_stack_trace.clone(),
                            )
                        })
                        .unwrap_or_default();
                    return Err(invoke_error_from_record(
                        &self.function_id,
                        error_type,
                        error_message,
                        error_data,
                        stack_trace,
                        &wire_id,
                        view.status.wire_str(),
                    ));
                }
                CheckpointStatus::Started | CheckpointStatus::Pending | CheckpointStatus::Ready => {
                    // Invoke has not settled yet — suspend.
                    return self.ctx.suspend_now().await;
                }
                CheckpointStatus::Cancelled => {
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "ChainedInvoke",
                        Some(CHAINED_INVOKE_SUB_TYPE),
                        view.attempt,
                    );
                    return Err(OperationError::from_kind(OperationErrorKind::Invoke(
                        InvokeError::new(
                            InvokeErrorKind::FunctionFailed,
                            Some("invoke cancelled".into()),
                        ),
                    ))
                    .with_operation(&wire_id, view.status.wire_str()));
                }
                CheckpointStatus::Unknown(ref raw) => {
                    // Unreachable in production — `checkpoint_view_validated`
                    // already failed the execution (issue #45). Kept as a
                    // typed arm so a future bypass cannot suspend forever on
                    // a status that may be terminal.
                    return Err(self.ctx.unrecognized_status_error(&wire_id, raw));
                }
            }
        }

        // 3. Live path: serialize the typed input through the payload
        // serdes, then checkpoint. The input transfers by ownership — a
        // write-only payload has no round-trip to preserve, and owning it
        // lets the serdes move it into a blocking task when it needs one.
        let wire_payload =
            serialize_invoke_input(&self.payload_serdes, self.input, &serdes_ctx).await?;

        // Checkpoint ChainedInvokeStarted then suspend.
        let update = build_chained_invoke_start(
            &wire_id,
            self.name.as_deref(),
            self.ctx.parent_wire_id(),
            &self.function_id,
            &wire_payload,
            self.tenant_id.as_deref(),
        );
        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
            // Audit (#43) — chained-invoke START: no user code ran (the
            // input was serialized, but nothing external happened), so
            // there is no side effect needing a recorded outcome. No
            // terminal FAIL: re-invocation reconverges on the same write.
            return self
                .ctx
                .checkpoint_failure_unrecoverable(&wire_id, err, None)
                .await;
        }

        // Suspend — the backend owns the child invocation.
        self.ctx.suspend_now().await
    }
}

// ── Serialization helpers ───────────────────────────────────────────────

/// Serializes the invoke input payload through the configured serdes.
///
/// Ownership of the input transfers to the serdes (write-only payloads
/// have no round-trip to preserve); the serdes decides where its work
/// runs.
async fn serialize_invoke_input<I, PS: Serdes<I>>(
    payload_serdes: &PS,
    input: I,
    serdes_ctx: &SerdesContext,
) -> Result<String, OperationError> {
    payload_serdes
        .serialize(input, serdes_ctx.clone())
        .await
        .map_err(|e| invoke_serialization_error("payload serdes", e))
}

/// Deserializes the invoke result payload through the configured serdes.
///
/// The payload is returned by the external target function — it never
/// passed through this SDK's serialize path. The context is marked
/// [`PayloadOrigin::External`] here, at the boundary, so a serdes with
/// storage indirection (e.g. `FileSystemSerdes`) never honors a file
/// reference the target function's output happens to contain.
async fn deserialize_invoke_result<O, RS: Serdes<O>>(
    result_serdes: &RS,
    payload: String,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    result_serdes
        .deserialize(payload, serdes_ctx.with_origin(PayloadOrigin::External))
        .await
        .map_err(|e| invoke_serialization_error("result serdes", e))
}

/// Wraps a serdes error as an invoke `SerializationFailed` operation
/// error, keeping the serdes failure as the source under a boundary
/// context frame.
fn invoke_serialization_error(boundary: &str, e: crate::BoxError) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::new(
        InvokeErrorKind::SerializationFailed,
        Some(crate::error::ContextualError::source_from(boundary, e)),
    )))
}

/// Reconstructs an `InvokeError` from a failed checkpoint record.
///
/// The recorded failure fields travel on the synthetic source rather
/// than being folded into a message, so `kind()` stays meaningful and
/// the recorded `error_type` stays recoverable after a replay. All four
/// wire fields are preserved: `error_data` and `stack_trace` pass
/// through verbatim (store-and-expose — never captured fresh on replay),
/// so the record attached to the [`crate::error::ReplayedFailure`] is
/// complete.
fn invoke_error_from_record(
    function_id: &str,
    error_type: Option<String>,
    error_message: Option<String>,
    error_data: Option<String>,
    stack_trace: Option<Vec<String>>,
    wire_id: &str,
    status: &str,
) -> OperationError {
    let message = error_message.unwrap_or_else(|| format!("invoked function {function_id} failed"));
    let wire = crate::error::WireError::new(error_type, Some(message))
        .with_error_data(error_data)
        .with_stack_trace(stack_trace.unwrap_or_default());
    OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::new(
        InvokeErrorKind::FunctionFailed,
        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
    )))
    .with_operation(wire_id, status)
    .with_wire(wire)
}

// ── Update builder ──────────────────────────────────────────────────────

fn build_chained_invoke_start(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    function_name: &str,
    payload: &str,
    tenant_id: Option<&str>,
) -> OperationUpdate {
    let mut invoke_opts_builder = ChainedInvokeOptions::builder().function_name(function_name);
    if let Some(tid) = tenant_id {
        invoke_opts_builder = invoke_opts_builder.tenant_id(tid);
    }
    #[expect(clippy::expect_used)] // reason: function_name is always set
    let invoke_opts = invoke_opts_builder
        .build()
        .expect("ChainedInvokeOptions: function_name is set");

    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::ChainedInvoke)
        .sub_type(CHAINED_INVOKE_SUB_TYPE)
        .action(OperationAction::Start)
        .payload(payload)
        .chained_invoke_options(invoke_opts);

    if let Some(n) = name {
        builder = builder.name(n);
    }

    if let Some(parent) = parent_wire_id {
        builder = builder.parent_id(parent);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions with descriptive messages
#[expect(clippy::indexing_slicing)] // reason: test assertions on known-populated vectors
mod tests {
    use super::*;
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord};
    use std::marker::PhantomData;
    use std::sync::Arc;

    // ── Replay tests ────────────────────────────────────────────────────

    #[tokio::test]
    async fn invoke_replay_success_deserializes_result() {
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
            invoke_result: Some(r#""hello from target""#.to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<String, _, _, _> {
            ctx,
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            input: serde_json::json!("input"),
            payload_serdes: crate::serdes::JsonSerdes,
            result_serdes: crate::serdes::JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), "hello from target");
    }

    /// ISSUE #46: a target function's result payload must never be honored
    /// as a `FileSystemSerdes` file reference. The result is produced by
    /// the external target function, not this SDK's serialize path, so a
    /// payload shaped like a legacy or versioned file pointer — even one
    /// naming a real, readable file under `base_path` — decodes as plain
    /// data, and a realistic inline payload containing a `file` key is not
    /// misparsed.
    #[tokio::test]
    async fn invoke_result_never_resolves_file_references() {
        use crate::serdes::FileSystemSerdes;

        let tmp = std::env::temp_dir().join(format!(
            "invoke_no_file_refs_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).expect("create temp base");
        std::fs::write(tmp.join("secret.json"), r#""stolen contents""#).expect("plant file");

        // Two attack shapes: the legacy file pointer and the versioned
        // file envelope, both naming the planted file.
        let legacy = format!(
            r#"{{"file":"{}","data":{{"status":"ready"}}}}"#,
            tmp.join("secret.json").to_string_lossy()
        );
        let versioned =
            r#"{"__aws_durable_serdes":{"version":1,"kind":"file"},"file":"secret.json"}"#
                .to_owned();

        for (label, payload) in [("legacy", legacy), ("versioned", versioned)] {
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
                invoke_result: Some(payload),
                invoke_error_type: None,
                invoke_error_message: None,
                invoke_error_data: None,
                invoke_stack_trace: None,
                replay_children: false,
                callback_id: None,
                op_type: None,
                sub_type: None,
                op_name: None,
            };
            let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
            let ctx = DurableContext::new_root(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                log,
            );
            let op_id = ctx.mint_id();

            let exec = InvokeExecution::<serde_json::Value, _, _, _> {
                ctx,
                op_id,
                name: None,
                function_id: "target-fn".to_owned(),
                input: serde_json::json!("input"),
                payload_serdes: crate::serdes::JsonSerdes,
                result_serdes: FileSystemSerdes::new(tmp.to_string_lossy().into_owned()),
                tenant_id: None,
                _marker: PhantomData,
            };

            let value = exec
                .execute()
                .await
                .expect("external result decodes as data");
            assert_ne!(
                value,
                serde_json::json!("stolen contents"),
                "{label}: an invoke result must never trigger a local file read"
            );
            assert!(
                value.get("file").is_some(),
                "{label}: the 'file' key is plain data"
            );
            if label == "legacy" {
                assert_eq!(
                    value["data"],
                    serde_json::json!({"status": "ready"}),
                    "inline payload with a 'file' key must not be misparsed"
                );
            }
        }

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn invoke_replay_failed_returns_invoke_error() {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Failed,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt: 0,
            invoke_result: None,
            invoke_error_type: Some("TargetError".to_owned()),
            invoke_error_message: Some("target function error".to_owned()),
            invoke_error_data: Some("{\"code\":7}".to_owned()),
            invoke_stack_trace: Some(vec!["frame-a".to_owned(), "frame-b".to_owned()]),
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<String, _, _, _> {
            ctx,
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            input: serde_json::json!("input"),
            payload_serdes: crate::serdes::JsonSerdes,
            result_serdes: crate::serdes::JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        // The recorded message renders in the chain; the recorded type is
        // data on the wire record, not display text.
        let err_msg = format!("{err:#}");
        assert!(err_msg.contains("target function error"), "got: {err_msg}");
        // All four recorded wire fields are preserved on the error's
        // attached record — nothing is discarded on replay.
        let wire = err.wire().unwrap();
        assert_eq!(wire.error_type(), Some("TargetError"));
        assert_eq!(wire.error_message(), Some("target function error"));
        assert_eq!(wire.error_data(), Some("{\"code\":7}"));
        assert_eq!(wire.stack_trace(), ["frame-a", "frame-b"]);
        // ...and the same complete record travels on the synthetic source.
        let mut link: Option<&(dyn std::error::Error + 'static)> = Some(&err);
        let replayed = loop {
            let Some(e) = link else {
                panic!("no ReplayedFailure in the chain");
            };
            if let Some(r) = e.downcast_ref::<crate::error::ReplayedFailure>() {
                break r;
            }
            link = e.source();
        };
        assert_eq!(replayed.wire(), wire);
        // Verify it's an InvokeError kind.
        assert!(matches!(err.kind(), OperationErrorKind::Invoke(_)));
    }

    #[tokio::test]
    async fn invoke_replay_started_suspends() {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
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
            op_name: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<String, _, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            input: serde_json::json!("input"),
            payload_serdes: crate::serdes::JsonSerdes,
            result_serdes: crate::serdes::JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn invoke_live_checkpoints_and_suspends() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<String, _, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("charge".to_owned()),
            function_id: "payment-fn".to_owned(),
            input: serde_json::json!({"amount": 100}),
            payload_serdes: crate::serdes::JsonSerdes,
            result_serdes: crate::serdes::JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn invoke_serdes_round_trip() {
        // Custom serdes: uppercases the JSON rendering on serialization,
        // lowercases the wire payload before parsing on deserialization.
        struct Upper;
        impl Serdes<String> for Upper {
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::to_string(&value)?.to_uppercase())
            }
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::from_str(&wire.to_lowercase())?)
            }
        }

        // Test payload serialization: the serdes receives the typed input.
        let ctx = SerdesContext::new("op-1", "arn:test");
        let serialized = serialize_invoke_input(&Upper, "hello".to_owned(), &ctx)
            .await
            .unwrap();
        assert_eq!(serialized, "\"HELLO\"");

        // Test result deserialization: "\"WORLD\"" -> "\"world\"" -> "world".
        let result: String = deserialize_invoke_result(&Upper, "\"WORLD\"".to_owned(), ctx)
            .await
            .unwrap();
        assert_eq!(result, "world");
    }

    #[tokio::test]
    async fn invoke_name_propagates() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<serde_json::Value, _, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: Some("my-invoke".to_owned()),
            function_id: "fn-arn".to_owned(),
            input: serde_json::Value::Null,
            payload_serdes: crate::serdes::JsonSerdes,
            result_serdes: crate::serdes::JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        };

        // Should not panic, should suspend.
        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn invoke_spawn_executes_on_blessed_task() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client,
            "token0".to_owned(),
        );

        // Use the public API: ctx.invoke().spawn() should succeed on a blessed task.
        let signal = Arc::clone(ctx.suspension_signal());
        let handle = ctx
            .invoke::<serde_json::Value, _>("fn-arn", "input".to_owned())
            .name("spawned-invoke")
            .spawn();
        // Blessed spawn: the op suspends (no ownership error), so the driver
        // reports Pending rather than Failed.
        let outcome = crate::driver::test_support::outcome_of(signal, handle).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn invoke_ownership_rejects_foreign_task() {
        use crate::client::InMemoryExecutionClient;

        // Must create ctx inside a spawned task where try_id() returns Some.
        let result = tokio::spawn(async {
            let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
            let log = Arc::new(CheckpointLog::empty());
            let ctx = DurableContext::new_root_with_client(
                "arn:test".to_owned(),
                lambda_runtime::Context::default(),
                log,
                client,
                "token0".to_owned(),
            );

            // Spawn a DIFFERENT (non-blessed) task.
            let ctx_clone = ctx.clone();
            let handle = tokio::spawn(async move {
                let op_id = ctx_clone.mint_id();
                let exec = InvokeExecution::<String, _, _, _> {
                    ctx: ctx_clone,
                    op_id,
                    name: None,
                    function_id: "fn".to_owned(),
                    input: serde_json::json!("x"),
                    payload_serdes: crate::serdes::JsonSerdes,
                    result_serdes: crate::serdes::JsonSerdes,
                    tenant_id: None,
                    _marker: PhantomData,
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

    #[tokio::test]
    async fn invoke_null_result_deserializes_to_option_none() {
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
            invoke_result: Some("null".to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<Option<String>, _, _, _> {
            ctx,
            op_id,
            name: None,
            function_id: "fn".to_owned(),
            input: serde_json::Value::Null,
            payload_serdes: crate::serdes::JsonSerdes,
            result_serdes: crate::serdes::JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), None);
    }

    #[tokio::test]
    async fn invoke_payload_serdes_applies_to_input() {
        use crate::client::InMemoryExecutionClient;

        // Custom payload serdes uppercases the input on serialization.
        struct UpperPayload;
        impl Serdes<String> for UpperPayload {
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::to_string(&value)?.to_uppercase())
            }
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::from_str(&wire)?)
            }
        }

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client.clone(),
            "token0".to_owned(),
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<String, _, _, _> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            input: "hello".to_owned(),
            payload_serdes: UpperPayload,
            result_serdes: crate::serdes::JsonSerdes,
            tenant_id: None,
            _marker: PhantomData,
        };

        // Execute — checkpoints then suspends (parks); drive through the
        // driver so it terminates as Pending instead of parking forever.
        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);

        // Verify that the checkpointed payload was uppercased.
        let updates = client.recorded_updates();
        assert!(!updates.is_empty(), "should have checkpointed");
        let first = &updates[0];
        // The payload on wire should be uppercased.
        let payload = first.payload().unwrap_or("");
        assert_eq!(
            payload, "\"HELLO\"",
            "payload_serdes should uppercase input"
        );
    }

    #[tokio::test]
    async fn invoke_result_serdes_applies_to_output() {
        // Custom result serdes lowercases output on deserialization.
        struct LowerResult;
        impl Serdes<String> for LowerResult {
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::to_string(&value)?)
            }
            // reason: exercises the async-fn impl form user code writes
            #[expect(clippy::unused_async_trait_impl)]
            async fn deserialize(
                &self,
                wire: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::from_str(&wire.to_lowercase())?)
            }
        }

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
            invoke_result: Some("\"WORLD\"".to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: None,
            sub_type: None,
            op_name: None,
        };
        let log = Arc::new(CheckpointLog::from_records(vec![(wire_key, record)]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<String, _, _, _> {
            ctx,
            op_id,
            name: None,
            function_id: "fn".to_owned(),
            input: serde_json::Value::Null,
            payload_serdes: crate::serdes::JsonSerdes,
            result_serdes: LowerResult,
            tenant_id: None,
            _marker: PhantomData,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), "world");
    }

    #[tokio::test]
    async fn invoke_input_serialization_failure_surfaces_error_and_skips_checkpoint() {
        use crate::client::InMemoryExecutionClient;
        use serde::Serializer;
        use serde::ser::Error as _;

        // A custom `Serialize` that always fails.
        struct FailingSerialize;
        impl serde::Serialize for FailingSerialize {
            fn serialize<S: Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
                Err(S::Error::custom("boom: cannot serialize"))
            }
        }
        impl<'de> serde::Deserialize<'de> for FailingSerialize {
            fn deserialize<D: serde::Deserializer<'de>>(_d: D) -> Result<Self, D::Error> {
                Ok(Self)
            }
        }

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client.clone(),
            "token0".to_owned(),
        );

        let result = ctx
            .invoke::<serde_json::Value, _>("target-fn", FailingSerialize)
            .name("charge")
            .await;

        // 1. The operation returns the serialization error (not a null-payload
        //    success or a suspend).
        let err = result.unwrap_err();
        let inner = match err.kind() {
            OperationErrorKind::Invoke(e) => e,
            other => panic!("expected Invoke error, got {other:?}"),
        };
        assert!(
            matches!(inner.kind(), InvokeErrorKind::SerializationFailed),
            "expected SerializationFailed, got {inner}"
        );
        assert!(
            format!("{err:#}").contains("boom"),
            "error must carry the original serde failure: {err:#}"
        );

        // 2. No invoke checkpoint was emitted.
        assert!(
            client.recorded_updates().is_empty(),
            "must not checkpoint when input serialization fails"
        );

        // 3. The `null` payload never reached the client.
        for u in client.recorded_updates() {
            assert_ne!(
                u.payload(),
                Some("null"),
                "null payload must never be substituted"
            );
        }
    }
}
