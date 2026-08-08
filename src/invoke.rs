//! Invoke operation execution engine.
//!
//! Implements the chained-invoke model: checkpoint `ChainedInvokeStarted`
//! with the target function name and serialized input, then suspend. The
//! backend performs the actual child invocation; on resume the SDK reads the
//! result from the checkpoint log.

use aws_sdk_lambda::types::{
    ChainedInvokeOptions, OperationAction, OperationType, OperationUpdate,
};

use std::sync::Arc;

use crate::Serdes;
use crate::SerdesContext;
use crate::client::ClientError;
use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{InvokeError, InvokeErrorKind, OperationError, OperationErrorKind};

#[cfg(test)]
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Wire sub-type for chained invoke operations.
pub(crate) const CHAINED_INVOKE_SUB_TYPE: &str = "ChainedInvoke";

/// Internal state for invoke execution passed from the builder.
pub(crate) struct InvokeExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) function_id: String,
    pub(crate) erased_input: Result<serde_json::Value, String>,
    pub(crate) payload_serdes: Option<Arc<dyn Serdes>>,
    pub(crate) result_serdes: Option<Arc<dyn Serdes>>,
    pub(crate) tenant_id: Option<String>,
    pub(crate) _marker: std::marker::PhantomData<O>,
}

impl<O: DeserializeOwned + Send + 'static> InvokeExecution<O> {
    /// Executes the invoke operation: replay path or live path.
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        // Surface an input-serialization failure now — before any replay
        // lookup, checkpoint, or execution-client call — so a failing
        // `Serialize` never invokes the target with a `null` payload and
        // never records an operation the caller did not request.
        let erased_input = match self.erased_input {
            Ok(input) => input,
            Err(msg) => {
                return Err(OperationError::from_kind(OperationErrorKind::Invoke(
                    InvokeError::from_kind(InvokeErrorKind::SerializationFailed {
                        message: format!("serialize invoke input: {msg}"),
                    }),
                )));
            }
        };

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
                    // Invoke succeeded — deserialize the result from invoke details.
                    let payload = self
                        .ctx
                        .with_checkpoint_record(&positional_id, |record| {
                            record.invoke_result.clone()
                        })
                        .flatten();
                    return deserialize_invoke_result(
                        self.result_serdes
                            .as_ref()
                            .or_else(|| self.ctx.default_serdes()),
                        payload.as_deref().unwrap_or("null"),
                        &serdes_ctx,
                    )
                    .await;
                }
                CheckpointStatus::Failed
                | CheckpointStatus::TimedOut
                | CheckpointStatus::Stopped => {
                    // Invoke failed — reconstruct InvokeError from details.
                    let (error_type, error_message) = self
                        .ctx
                        .with_checkpoint_record(&positional_id, |record| {
                            (
                                record.invoke_error_type.clone(),
                                record.invoke_error_message.clone(),
                            )
                        })
                        .unwrap_or_default();
                    return Err(invoke_error_from_record(
                        &self.function_id,
                        error_type.as_deref(),
                        error_message.as_deref(),
                    ));
                }
                CheckpointStatus::Started | CheckpointStatus::Pending | CheckpointStatus::Ready => {
                    // Invoke has not settled yet — suspend.
                    return self.ctx.suspend_now().await;
                }
                CheckpointStatus::Cancelled => {
                    return Err(OperationError::from_kind(OperationErrorKind::Invoke(
                        InvokeError::from_kind(InvokeErrorKind::FunctionFailed {
                            message: "invoke cancelled".to_owned(),
                        }),
                    )));
                }
            }
        }

        // 3. Live path: apply payload serdes then checkpoint.
        let effective_payload_serdes = self
            .payload_serdes
            .as_ref()
            .or_else(|| self.ctx.default_serdes());
        let wire_payload = if let Some(ps) = effective_payload_serdes {
            // The Value was erased at the call site (context.rs). The serdes
            // receives the same shape every other path provides — no re-parsing
            // needed. Custom serdes may block (e.g. filesystem I/O), so the
            // call runs off the async runtime.
            crate::serdes::serialize_off_runtime(ps, erased_input, &serdes_ctx)
                .await
                .map_err(|e| invoke_serialization_error(&format!("payload serdes: {e}")))?
        } else {
            // No custom serdes: render the Value to compact JSON for the wire.
            // Note: this is `to_string(&Value)`, not `to_string(&I)`. The two
            // can differ for edge cases (struct field order without
            // `preserve_order`, 128-bit integers outside i64/u64 range,
            // duplicate keys). Those cases fail at the `to_value` call site
            // rather than silently producing different wire bytes.
            serde_json::to_string(&erased_input)
                .map_err(|e| invoke_serialization_error(&format!("serialize invoke input: {e}")))?
        };

        // Checkpoint ChainedInvokeStarted then suspend.
        let update = build_chained_invoke_start(
            &wire_id,
            self.name.as_deref(),
            self.ctx.parent_wire_id(),
            &self.function_id,
            &wire_payload,
            self.tenant_id.as_deref(),
        );
        self.ctx
            .checkpoint_updates(vec![update])
            .await
            .map_err(|e| client_error_to_invoke_op_error(&e))?;

        // Suspend — the backend owns the child invocation.
        self.ctx.suspend_now().await
    }
}

// ── Serialization helpers ───────────────────────────────────────────────

/// Serializes the invoke input payload using the configured serdes.
///
/// The production path serializes through the pre-erased two-phase flow in
/// [`InvokeExecution::execute`] (`prepare_value` at builder time, then
/// `serialize_off_runtime`); this one-shot form is retained as the direct
/// unit-test harness for the payload-serdes behavior both share.
#[cfg(test)]
pub(crate) async fn serialize_invoke_input<I: Serialize>(
    payload_serdes: Option<&Arc<dyn Serdes>>,
    input: &I,
    serdes_ctx: &SerdesContext,
) -> Result<String, OperationError> {
    // Phase 1 (sync): consume the `&I` borrow before awaiting. No custom
    // serdes renders straight to the wire; a custom serdes receives the
    // input erased to `serde_json::Value` — the same shape every other
    // operation path provides.
    let prepared = crate::serdes::prepare_value(payload_serdes, input)
        .map_err(|e| invoke_serialization_error(&format!("serialize invoke input: {e}")))?;
    // Phase 2 (async): a custom serdes may block (e.g. filesystem I/O), so
    // the call runs off the async runtime.
    prepared
        .into_wire(serdes_ctx)
        .await
        .map_err(|e| invoke_serialization_error(&format!("payload serdes: {e}")))
}

/// Deserializes the invoke result payload using the configured serdes.
async fn deserialize_invoke_result<O: DeserializeOwned>(
    result_serdes: Option<&Arc<dyn Serdes>>,
    payload: &str,
    serdes_ctx: &SerdesContext,
) -> Result<O, OperationError> {
    let Some(s) = result_serdes else {
        return serde_json::from_str(payload)
            .map_err(|e| invoke_serialization_error(&format!("deserialize invoke result: {e}")));
    };
    // Custom serdes may block (e.g. filesystem I/O): run off the runtime.
    let json_value = crate::serdes::deserialize_off_runtime(s, payload.to_owned(), serdes_ctx)
        .await
        .map_err(|e| invoke_serialization_error(&format!("result serdes: {e}")))?;
    serde_json::from_value(json_value)
        .map_err(|e| invoke_serialization_error(&format!("deserialize invoke result: {e}")))
}

/// Wraps a message as an invoke `SerializationFailed` operation error.
fn invoke_serialization_error(message: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::from_kind(
        InvokeErrorKind::SerializationFailed {
            message: message.to_owned(),
        },
    )))
}

/// Reconstructs an `InvokeError` from a failed checkpoint record.
fn invoke_error_from_record(
    function_id: &str,
    error_type: Option<&str>,
    error_message: Option<&str>,
) -> OperationError {
    let msg = match (error_type, error_message) {
        (Some(t), Some(m)) => format!("{t}: {m}"),
        (None, Some(m)) => m.to_owned(),
        (Some(t), None) => t.to_owned(),
        (None, None) => format!("invoked function {function_id} failed"),
    };
    OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::from_kind(
        InvokeErrorKind::FunctionFailed { message: msg },
    )))
}

fn client_error_to_invoke_op_error(err: &ClientError) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Invoke(InvokeError::from_kind(
        InvokeErrorKind::FunctionFailed {
            message: err.to_string(),
        },
    )))
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
    #[allow(clippy::expect_used)] // reason: function_name is always set
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

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::panic)] // reason: test assertions with descriptive messages
#[allow(clippy::indexing_slicing)] // reason: test assertions on known-populated vectors
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
            attempt: 0,
            invoke_result: Some(r#""hello from target""#.to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
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

        let exec = InvokeExecution::<String> {
            ctx,
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            erased_input: Ok(serde_json::json!("input")),
            payload_serdes: None,
            result_serdes: None,
            tenant_id: None,
            _marker: PhantomData,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), "hello from target");
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
            attempt: 0,
            invoke_result: None,
            invoke_error_type: Some("TargetError".to_owned()),
            invoke_error_message: Some("target function error".to_owned()),
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

        let exec = InvokeExecution::<String> {
            ctx,
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            erased_input: Ok(serde_json::json!("input")),
            payload_serdes: None,
            result_serdes: None,
            tenant_id: None,
            _marker: PhantomData,
        };

        let result = exec.execute().await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("TargetError"), "got: {err_msg}");
        assert!(err_msg.contains("target function error"), "got: {err_msg}");
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
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let exec = InvokeExecution::<String> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            erased_input: Ok(serde_json::json!("input")),
            payload_serdes: None,
            result_serdes: None,
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

        let exec = InvokeExecution::<String> {
            ctx: ctx.clone(),
            op_id,
            name: Some("charge".to_owned()),
            function_id: "payment-fn".to_owned(),
            erased_input: Ok(serde_json::json!({"amount": 100})),
            payload_serdes: None,
            result_serdes: None,
            tenant_id: None,
            _marker: PhantomData,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn invoke_serdes_round_trip() {
        // Custom payload serdes uppercases input on serialization.
        struct Upper;
        impl std::fmt::Debug for Upper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Upper")
            }
        }
        impl Serdes for Upper {
            fn serialize(
                &self,
                value: &serde_json::Value,
                _context: &SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(value.to_string().to_uppercase())
            }
            fn deserialize(
                &self,
                data: &str,
                _context: &SerdesContext,
            ) -> Result<serde_json::Value, crate::BoxError> {
                Ok(serde_json::from_str(&data.to_lowercase())?)
            }
        }

        // Test payload serialization.
        let ctx = SerdesContext::new("op-1", "arn:test");
        let upper: Arc<dyn Serdes> = Arc::new(Upper);
        let serialized = serialize_invoke_input(Some(&upper), &"hello", &ctx)
            .await
            .unwrap();
        assert_eq!(serialized, "\"HELLO\"");

        // Test result deserialization.
        // Upper::deserialize lowercases the whole payload string (including
        // JSON quotes) and parses it into a Value, which is then deserialized
        // into `String`: "\"WORLD\"" -> "\"world\"" -> "world".
        let result: String = deserialize_invoke_result(Some(&upper), "\"WORLD\"", &ctx)
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

        let exec = InvokeExecution::<serde_json::Value> {
            ctx: ctx.clone(),
            op_id,
            name: Some("my-invoke".to_owned()),
            function_id: "fn-arn".to_owned(),
            erased_input: Ok(serde_json::Value::Null),
            payload_serdes: None,
            result_serdes: None,
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
            .invoke::<serde_json::Value, _>("fn-arn", "input")
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
                let exec = InvokeExecution::<String> {
                    ctx: ctx_clone,
                    op_id,
                    name: None,
                    function_id: "fn".to_owned(),
                    erased_input: Ok(serde_json::json!("x")),
                    payload_serdes: None,
                    result_serdes: None,
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
        let err_msg = result.unwrap_err().to_string();
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
            attempt: 0,
            invoke_result: Some("null".to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
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

        let exec = InvokeExecution::<Option<String>> {
            ctx,
            op_id,
            name: None,
            function_id: "fn".to_owned(),
            erased_input: Ok(serde_json::Value::Null),
            payload_serdes: None,
            result_serdes: None,
            tenant_id: None,
            _marker: PhantomData,
        };

        let result = exec.execute().await;
        assert_eq!(result.unwrap(), None);
    }

    #[tokio::test]
    async fn invoke_payload_serdes_applies_to_input() {
        use crate::client::InMemoryExecutionClient;

        // Custom payload serdes uppercases input on serialization.
        struct UpperPayload;
        impl std::fmt::Debug for UpperPayload {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("UpperPayload")
            }
        }
        impl Serdes for UpperPayload {
            fn serialize(
                &self,
                value: &serde_json::Value,
                _context: &SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(value.to_string().to_uppercase())
            }
            fn deserialize(
                &self,
                data: &str,
                _context: &SerdesContext,
            ) -> Result<serde_json::Value, crate::BoxError> {
                Ok(serde_json::from_str(data)?)
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

        let exec = InvokeExecution::<String> {
            ctx: ctx.clone(),
            op_id,
            name: None,
            function_id: "target-fn".to_owned(),
            erased_input: Ok(serde_json::json!("hello")),
            payload_serdes: Some(Arc::new(UpperPayload)),
            result_serdes: None,
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
        impl std::fmt::Debug for LowerResult {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("LowerResult")
            }
        }
        impl Serdes for LowerResult {
            fn serialize(
                &self,
                value: &serde_json::Value,
                _context: &SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(value.to_string())
            }
            fn deserialize(
                &self,
                data: &str,
                _context: &SerdesContext,
            ) -> Result<serde_json::Value, crate::BoxError> {
                Ok(serde_json::from_str(&data.to_lowercase())?)
            }
        }

        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Succeeded,
            result: None,
            error_type: None,
            error_message: None,
            attempt: 0,
            invoke_result: Some("\"WORLD\"".to_owned()),
            invoke_error_type: None,
            invoke_error_message: None,
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

        let exec = InvokeExecution::<String> {
            ctx,
            op_id,
            name: None,
            function_id: "fn".to_owned(),
            erased_input: Ok(serde_json::Value::Null),
            payload_serdes: None,
            result_serdes: Some(Arc::new(LowerResult)),
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
        impl Serialize for FailingSerialize {
            fn serialize<S: Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
                Err(S::Error::custom("boom: cannot serialize"))
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
            matches!(inner.kind(), InvokeErrorKind::SerializationFailed { .. }),
            "expected SerializationFailed, got {inner}"
        );
        assert!(
            err.to_string().contains("boom"),
            "error must carry the original serde failure: {err}"
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
