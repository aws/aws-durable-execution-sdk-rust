//! Wait operation execution engine.
//!
//! Implements the durable timer: checkpoint `WaitStarted` with
//! `WaitOptions.WaitSeconds`, then wait for the record to change. The
//! backend owns the timer. When a checkpoint response completes the wait
//! during the invocation, the wait continues at once; otherwise the
//! invocation suspends once no other SDK work can make progress, and the
//! backend invokes the function again when the duration elapses. On replay
//! a completed wait returns immediately.

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate, WaitOptions};

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
                    // Timer completed: return immediately.
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Wait",
                        Some(WAIT_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(());
                }
                CheckpointStatus::Started => {
                    // Timer started but not yet completed: wait for the
                    // record to change.
                    return self.resume(&positional_id, &wire_id).await;
                }
                status => return recorded_outcome(&self.ctx, &wire_id, &status),
            }
        }

        // 3. Live path: checkpoint WaitStarted then suspend.
        let update = build_wait_start_update(
            &wire_id,
            self.name.as_deref(),
            self.ctx.parent_wire_id(),
            self.duration_secs,
        );
        if let Err(err) = self.ctx.checkpoint_updates(vec![update]).await {
            // Audit (#43): wait START: no user code ran, so there is no
            // side effect needing a recorded outcome. No terminal FAIL:
            // the invocation dies, the service re-invokes, and replay
            // reaches this point and attempts the same write, losing only
            // one invocation.
            return self
                .ctx
                .checkpoint_failure_unrecoverable(&wire_id, err, None)
                .await;
        }

        // The backend owns the timer: wait for the record to change.
        self.resume(&positional_id, &wire_id).await
    }

    /// Waits until the wait's record is no longer `STARTED`, then returns
    /// the outcome the record holds. A wait that a checkpoint response
    /// completes continues in the same invocation; one that nothing
    /// completes keeps a wait registration, so the invocation suspends once
    /// no other SDK work can make progress. A resumed wait emits no
    /// `operation_replayed` event.
    async fn resume(&self, positional_id: &str, wire_id: &str) -> Result<(), OperationError> {
        let view = self
            .ctx
            .wait_for_record_view(positional_id, |status| *status != CheckpointStatus::Started)
            .await;
        recorded_outcome(&self.ctx, wire_id, &view.status)
    }
}

/// The outcome of a wait whose record is not `STARTED`: `Ok` for
/// `SUCCEEDED`, and an unexpected-status error for every other status.
fn recorded_outcome(
    ctx: &DurableContext,
    wire_id: &str,
    status: &CheckpointStatus,
) -> Result<(), OperationError> {
    match status {
        CheckpointStatus::Succeeded => Ok(()),
        CheckpointStatus::Unknown(raw) => {
            // Unreachable in production: `checkpoint_view_validated`
            // already failed the execution (issue #45). Kept as a typed arm
            // so a future bypass fails loudly instead of being demoted to a
            // catchable unexpected-status error.
            Err(ctx.unrecognized_status_error(wire_id, raw))
        }
        CheckpointStatus::Started
        | CheckpointStatus::Pending
        | CheckpointStatus::Ready
        | CheckpointStatus::Failed
        | CheckpointStatus::Cancelled
        | CheckpointStatus::TimedOut
        | CheckpointStatus::Stopped => Err(OperationError::from_kind(OperationErrorKind::Wait(
            WaitError::new(
                WaitErrorKind::UnexpectedStatus(crate::error::UnexpectedStatus::new(
                    status.wire_str(),
                )),
                None,
            ),
        ))
        .with_operation(wire_id, status.wire_str())),
    }
}

// Update builder

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

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
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

        let exec = WaitExecution {
            ctx: ctx.clone(),
            op_id,
            name: None,
            duration_secs: 5,
        };

        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.task_ownership()),
            signal,
            exec.execute(),
        )
        .await;
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
                    if matches!(e.kind(), WaitErrorKind::UnexpectedStatus(_))
            ),
            "expected Wait/UnexpectedStatus error, got {:?}",
            err.kind()
        );
        // The offending status is a structural fact behind the payload's
        // accessor, and renders in the flattened chain. (A let-chain here
        // trips 1.94.1's irrefutable-let lint, `WaitErrorKind` has one
        // variant, so extract through a single-arm match instead.)
        if let OperationErrorKind::Wait(e) = err.kind() {
            match e.kind() {
                WaitErrorKind::UnexpectedStatus(details) => {
                    assert_eq!(details.status(), "FAILED");
                }
            }
        }
        assert!(
            format!("{err:#}").contains("FAILED"),
            "the chain must carry the offending status: {err:#}"
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

        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.task_ownership()),
            signal,
            exec.execute(),
        )
        .await;
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
        let signal = Arc::clone(ctx.coordinator());
        // Should suspend (not panic): name was propagated successfully.
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.task_ownership()),
            signal,
            builder,
        )
        .await;
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
        let signal = Arc::clone(ctx.coordinator());
        let handle = ctx.wait(Duration::from_secs(2)).name("spawned").spawn();
        // Blessed spawn: the op suspends (no ownership error), so the driver
        // reports Pending rather than Failed.
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.task_ownership()),
            signal,
            handle,
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
    }

    #[tokio::test]
    async fn wait_ownership_check_rejects_foreign_task() {
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
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("task") || err_msg.contains("owner"),
            "expected ownership error, got: {err_msg}"
        );
    }

    /// Drives `ctx.wait(duration)` on a fresh root context and returns the
    /// `WaitSeconds` of the one START update it records. The wait must
    /// suspend the invocation (`Pending`), because no timer fires.
    async fn recorded_wait_seconds(duration: Duration) -> i32 {
        use crate::client::InMemoryExecutionClient;

        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client.clone(),
            "token0".to_owned(),
        );

        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            Arc::clone(ctx.task_ownership()),
            signal,
            ctx.wait(duration),
        )
        .await;
        assert_eq!(
            outcome,
            crate::driver::InvocationOutcome::Pending,
            "{duration:?}"
        );

        let updates = client.recorded_updates();
        let starts: Vec<_> = updates
            .iter()
            .filter(|u| {
                matches!(u.action(), OperationAction::Start)
                    && matches!(u.r#type(), OperationType::Wait)
            })
            .collect();
        assert_eq!(starts.len(), 1, "{duration:?}: {updates:?}");
        starts
            .first()
            .and_then(|u| u.wait_options())
            .and_then(WaitOptions::wait_seconds)
            .expect("the START update carries WaitSeconds")
    }

    /// The service rejects `WaitSeconds: 0`, so a zero wait records one
    /// second.
    #[tokio::test]
    async fn wait_zero_duration_records_one_second() {
        assert_eq!(recorded_wait_seconds(Duration::ZERO).await, 1);
    }

    /// A caller-given duration rounds up to whole seconds, so the wait
    /// never ends early.
    #[tokio::test]
    async fn wait_sub_second_duration_rounds_up() {
        for (duration, expected) in [
            (Duration::from_millis(500), 1),
            (Duration::from_millis(1500), 2),
            (Duration::from_secs(2), 2),
        ] {
            assert_eq!(
                recorded_wait_seconds(duration).await,
                expected,
                "{duration:?}"
            );
        }
    }
}
