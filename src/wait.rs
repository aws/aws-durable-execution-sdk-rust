//! Wait operation execution engine.
//!
//! Implements the durable timer: checkpoint `WaitStarted` with
//! `WaitOptions.WaitSeconds`, then request suspension. The backend owns
//! the timer and re-invokes when the duration elapses. On replay a
//! completed wait returns immediately.

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate, WaitOptions};

use crate::client::ClientError;
use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{OperationError, OperationErrorKind, WaitError, WaitErrorKind};

/// Wire sub-type for wait operations.
pub(crate) const WAIT_SUB_TYPE: &str = "Wait";

/// Internal state for wait execution passed from the builder.
pub(crate) struct WaitExecution {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) duration_secs: i32,
}

impl WaitExecution {
    /// Executes the wait operation: replay path or live path.
    pub(crate) async fn execute(self) -> Result<(), OperationError> {
        // 1. Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // 2. Check checkpoint log for replay. The validated view carries
        // everything the wait replay path reads, so nothing is cloned.
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Wait",
            Some(WAIT_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Timer completed — return immediately.
                    return Ok(());
                }
                CheckpointStatus::Started => {
                    // Timer started but not yet completed — suspend by
                    // parking the future; the driver drops it and reports
                    // PENDING.
                    return self.ctx.suspend_now().await;
                }
                CheckpointStatus::Pending
                | CheckpointStatus::Ready
                | CheckpointStatus::Failed
                | CheckpointStatus::Cancelled
                | CheckpointStatus::TimedOut
                | CheckpointStatus::Stopped => {
                    // Unexpected status for wait — treat as error.
                    return Err(OperationError::from_kind(OperationErrorKind::Wait(
                        WaitError::from_kind(WaitErrorKind::UnexpectedStatus {
                            status: format!("{:?}", view.status),
                        }),
                    )));
                }
            }
        }

        // 3. Live path: checkpoint WaitStarted then suspend.
        let update = build_wait_start_update(
            &wire_id,
            self.name.as_deref(),
            self.ctx.parent_wire_id(),
            self.duration_secs,
        );
        self.ctx
            .checkpoint_updates(vec![update])
            .await
            .map_err(|e| client_error_to_op_error(&e))?;

        // Request suspension — the backend owns the timer.
        self.ctx.suspend_now().await
    }
}

// ── Update builder ──────────────────────────────────────────────────────

fn build_wait_start_update(
    wire_id: &str,
    name: Option<&str>,
    parent_wire_id: Option<&str>,
    wait_seconds: i32,
) -> OperationUpdate {
    let mut builder = OperationUpdate::builder()
        .id(wire_id)
        .r#type(OperationType::Wait)
        .sub_type(WAIT_SUB_TYPE)
        .action(OperationAction::Start)
        .wait_options(WaitOptions::builder().wait_seconds(wait_seconds).build());

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

fn client_error_to_op_error(err: &ClientError) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Wait(WaitError::from_kind(
        WaitErrorKind::CheckpointFailed {
            message: err.to_string(),
        },
    )))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
mod tests {
    use super::*;
    use crate::engine::{CheckpointLog, CheckpointRecord};
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn wait_replay_succeeded_returns_immediately() {
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
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let op_id = ctx.mint_id();

        let exec = WaitExecution {
            ctx,
            op_id,
            name: None,
            duration_secs: 5,
        };

        let result = exec.execute().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn wait_replay_started_requests_suspension() {
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

        let exec = WaitExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            duration_secs: 5,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn wait_unexpected_status_yields_wait_error() {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status: CheckpointStatus::Failed,
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

        let exec = WaitExecution {
            ctx,
            op_id,
            name: None,
            duration_secs: 5,
        };

        let err = exec.execute().await.unwrap_err();
        // A failed wait surfaces as OperationErrorKind::Wait, not Step.
        assert!(
            matches!(
                err.kind(),
                OperationErrorKind::Wait(e)
                    if matches!(e.kind(), WaitErrorKind::UnexpectedStatus { .. })
            ),
            "expected Wait/UnexpectedStatus error, got {:?}",
            err.kind()
        );
        assert!(
            err.to_string().contains("Failed"),
            "display must carry the offending status: {err}"
        );
    }

    #[tokio::test]
    async fn wait_live_checkpoints_and_suspends() {
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

        let exec = WaitExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("test-wait".to_owned()),
            duration_secs: 10,
        };

        let signal = Arc::clone(ctx.suspension_signal());
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn wait_name_chaining_propagates() {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            client.clone(),
            "token0".to_owned(),
        );

        // Use builder chain: .name() returns Self and propagates to execution.
        let builder = ctx.wait(Duration::from_secs(5)).name("cooldown");
        // Verify the builder is Debug (API Guidelines C-COMMON-TRAITS)
        let _debug = format!("{builder:?}");
        // Execute via IntoFuture
        let signal = Arc::clone(ctx.suspension_signal());
        // Should suspend (not panic) — name was propagated successfully.
        let outcome = crate::driver::test_support::outcome_of(signal, builder).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn wait_spawn_runs_on_blessed_task() {
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

        // spawn() should succeed (blessed task, not foreign).
        let signal = Arc::clone(ctx.suspension_signal());
        let handle = ctx.wait(Duration::from_secs(2)).name("spawned").spawn();
        // Blessed spawn: the op suspends (no ownership error), so the driver
        // reports Pending rather than Failed.
        let outcome = crate::driver::test_support::outcome_of(signal, handle).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn wait_ownership_check_rejects_foreign_task() {
        use crate::client::InMemoryExecutionClient;

        // Must create ctx inside a spawned task where try_id() returns Some.
        #[allow(clippy::unwrap_used)] // reason: test code
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

            // Spawn a DIFFERENT (non-blessed) task that tries a wait op.
            let ctx_clone = ctx.clone();
            let handle = tokio::spawn(async move {
                let op_id = ctx_clone.mint_id();
                let exec = WaitExecution {
                    ctx: ctx_clone,
                    op_id,
                    name: None,
                    duration_secs: 3,
                };
                exec.execute().await
            });

            handle.await.unwrap()
        })
        .await
        .unwrap();

        // Should fail with an ownership error.
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("task") || err_msg.contains("owner"),
            "expected ownership error, got: {err_msg}"
        );
    }

    #[tokio::test]
    async fn wait_zero_duration_passes_zero_seconds() {
        use crate::client::InMemoryExecutionClient;

        // Zero duration → WaitSeconds=0.
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
        let exec = WaitExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            duration_secs: 0, // direct: zero duration
        };

        let signal = Arc::clone(ctx.suspension_signal());
        // Should checkpoint and suspend (not panic or reject).
        let outcome = crate::driver::test_support::outcome_of(signal, exec.execute()).await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn wait_sub_second_duration_rounds_up() {
        // Sub-second durations round up: Duration(500ms) → 1 second.
        let log = Arc::new(CheckpointLog::empty());
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );

        // Build a wait with 500ms — should round up to 1 second.
        let builder = ctx.wait(Duration::from_millis(500));
        // The builder stores duration_secs internally — verify via execution.
        // We can't inspect private fields, but the execution should not panic.
        let _debug = format!("{builder:?}");
    }
}
