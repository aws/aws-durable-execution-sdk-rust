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
use crate::client::ClientError;
use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{InvokeError, InvokeErrorKind, OperationError, OperationErrorKind};
use crate::serdes::SerdesContext;

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
    #[allow(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
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
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "ChainedInvoke",
                        Some(CHAINED_INVOKE_SUB_TYPE),
                        view.attempt,
                    );
                    return Err(OperationError::from_kind(OperationErrorKind::Invoke(
                        InvokeError::from_kind(InvokeErrorKind::FunctionFailed {
                            message: "invoke cancelled".to_owned(),
                        }),
                    )));
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
        self.ctx
            .checkpoint_updates(vec![update])
            .await
            .map_err(|e| client_error_to_invoke_op_error(&e))?;

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
        .map_err(|e| invoke_serialization_error(&format!("payload serdes: {e}")))
}

/// Deserializes the invoke result payload through the configured serdes.
async fn deserialize_invoke_result<O, RS: Serdes<O>>(
    result_serdes: &RS,
    payload: String,
    serdes_ctx: SerdesContext,
) -> Result<O, OperationError> {
    result_serdes
        .deserialize(payload, serdes_ctx)
        .await
        .map_err(|e| invoke_serialization_error(&format!("result serdes: {e}")))
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
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::to_string(&value)?.to_uppercase())
            }
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
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::to_string(&value)?.to_uppercase())
            }
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
            async fn serialize(
                &self,
                value: String,
                _context: SerdesContext,
            ) -> Result<String, crate::BoxError> {
                Ok(serde_json::to_string(&value)?)
            }
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
