//! Wait operation execution engine.
//!
//! Implements the durable timer: checkpoint `WaitStarted` with
//! `WaitOptions.WaitSeconds`, then wait for the record to change. The
//! backend owns the timer. When a checkpoint response completes the wait
//! during the invocation, the wait continues at once; otherwise the
//! invocation suspends once no other SDK work can make progress, and the
//! backend invokes the function again when the duration elapses. On replay
//! a completed wait returns immediately.

use std::future::Future;

use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate, WaitOptions};

use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{OperationError, OperationErrorKind, WaitError, WaitErrorKind};
use crate::start::StartStep;

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
    ///
    /// Tests call it; the SDK builds every operation future with the start
    /// step form (see [`crate::future::DurableFuture::for_start_step`]).
    #[cfg(test)]
    pub(crate) async fn execute(self) -> Result<(), OperationError> {
        let started = self.start();
        self.execute_after_start(started).await
    }

    /// Runs the wait's start step now, on the caller's task, and returns
    /// the future of the rest of the wait (see [`crate::start`]).
    pub(crate) fn execute_started(
        self,
    ) -> impl Future<Output = Result<(), OperationError>> + Send + 'static {
        let started = self.start();
        self.execute_after_start(started)
    }

    /// The start step: admission, task ownership, the replay identity
    /// check, and the START of the live path (see `crate::start`). A
    /// STARTED record resumes the timer, and fires the replay start hook.
    fn start(&self) -> Result<StartStep, OperationError> {
        let wire_id = self.op_id.wire();
        crate::start::start_step(
            &self.ctx,
            &crate::start::StartSpec {
                op_id: &self.op_id,
                operation_type: "Wait",
                sub_type: WAIT_SUB_TYPE,
                name: self.name.as_deref(),
                order: crate::start::StartOrder::AwaitResponse,
            },
            |view| match view.status {
                CheckpointStatus::Started => crate::start::Cycle::Resume,
                _ => crate::start::Cycle::Record,
            },
            || {
                build_wait_start_update(
                    wire_id,
                    self.name.as_deref(),
                    self.ctx.parent_wire_id(),
                    self.duration_secs,
                )
            },
            |plugins, _| {
                let info = self.info(wire_id).with_replay(true);
                crate::plugin::operation_started(plugins, &info);
            },
        )
    }

    /// The rest of the wait after its start step `started`.
    async fn execute_after_start(
        self,
        started: Result<StartStep, OperationError>,
    ) -> Result<(), OperationError> {
        let started = started?;
        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // 2. A recorded wait returns its outcome, or keeps waiting.
        let start = match started {
            StartStep::Refused => return std::future::pending().await,
            StartStep::Queued(write) => write,
            StartStep::Recorded(view) => {
                return match view.status {
                    CheckpointStatus::Succeeded => {
                        // Timer completed: return immediately.
                        self.ctx.emit_operation_replayed(
                            &wire_id,
                            self.name.as_deref(),
                            "Wait",
                            Some(WAIT_SUB_TYPE),
                            view.attempt,
                        );
                        // A replayed completion is live only in the first
                        // invocation after the service completed the wait.
                        self.ended(&wire_id, |plugins| !plugins.is_updated(&wire_id));
                        Ok(())
                    }
                    // Timer started but not yet completed: wait for the
                    // record to change.
                    CheckpointStatus::Started => self.resume(&positional_id, &wire_id).await,
                    status => recorded_outcome(&self.ctx, &wire_id, &status),
                };
            }
        };

        // 3. Live path: await the WaitStarted response, then suspend.
        if let Err(err) = start.await {
            return self.ctx.checkpoint_failure_unrecoverable(err).await;
        }

        self.started(&wire_id, |info| {
            info.with_status(crate::OperationStatus::Started)
                .with_replay(false)
        });

        // The backend owns the timer: wait for the record to change.
        self.resume(&positional_id, &wire_id).await
    }

    /// The plugin info of this wait, read from its current record.
    fn info(&self, wire_id: &str) -> crate::plugin::OperationInfo {
        crate::plugin::OperationInfo::for_operation(
            &self.ctx,
            wire_id,
            crate::OperationKind::Wait,
            WAIT_SUB_TYPE,
            self.name.as_deref(),
        )
    }

    /// Fires `on_operation_start` with the info `shape` makes of the
    /// current record's info.
    fn started(
        &self,
        wire_id: &str,
        shape: impl FnOnce(crate::plugin::OperationInfo) -> crate::plugin::OperationInfo,
    ) {
        if let Some(plugins) = self.ctx.plugins() {
            crate::plugin::operation_started(plugins, &shape(self.info(wire_id)));
        }
    }

    /// Fires `on_operation_end` with the current record's info, and the
    /// replay flag `is_replay` returns.
    fn ended(
        &self,
        wire_id: &str,
        is_replay: impl FnOnce(&crate::plugin::InvocationPlugins) -> bool,
    ) {
        if let Some(plugins) = self.ctx.plugins() {
            let info = self.info(wire_id).with_replay(is_replay(plugins));
            crate::plugin::operation_ended(plugins, &info);
        }
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
        // The wait waited in this invocation, so its completion is live.
        if view.status == CheckpointStatus::Succeeded {
            self.ended(wire_id, |_| false);
        }
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
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
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
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
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
            ctx.clone(),
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
            due_at: None,
            op_name: None,
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
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
            ctx.clone(),
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
            ctx.clone(),
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
            ctx.clone(),
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
            ctx.clone(),
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

    /// A wait record with `status` for the first operation of a root
    /// context.
    fn wait_record_with(status: CheckpointStatus) -> (String, CheckpointRecord) {
        let wire_key = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: wire_key.clone(),
            status,
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
        (wire_key, record)
    }

    /// A `FAILED` wait record returns its error and fires no hook.
    #[tokio::test]
    async fn failed_wait_record_fires_no_hook() {
        use crate::client::InMemoryExecutionClient;

        let hooks = crate::plugin::test_support::OpHooks::default();
        let log = Arc::new(CheckpointLog::from_records(vec![wait_record_with(
            CheckpointStatus::Failed,
        )]));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
            Arc::new(InMemoryExecutionClient::new(Vec::new())),
            "token0".to_owned(),
            None,
            0,
            Some(hooks.plugins()),
        );
        let err = ctx.wait(Duration::from_secs(5)).await.unwrap_err();
        assert!(
            matches!(err.kind(), OperationErrorKind::Wait(_)),
            "{:?}",
            err.kind()
        );
        assert!(hooks.records().is_empty(), "{:?}", hooks.records());
    }

    /// A wait that waited in the invocation and whose record became
    /// `FAILED` returns its error and fires no end hook.
    #[tokio::test(start_paused = true)]
    async fn wait_failed_in_invocation_fires_no_end() {
        use crate::driver::test_support::{RefreshClient, drive_handler, named, operation_like};
        use aws_sdk_lambda::types::OperationStatus;

        let client = Arc::new(RefreshClient::new(|req| {
            if req.empty_number == 1
                && let Some(start) = named(req.history, "w", &OperationAction::Start)
            {
                let record = operation_like(start, OperationStatus::Failed)
                    .build()
                    .expect("all required fields set");
                return Ok(vec![record]);
            }
            Ok(Vec::new())
        }));
        let hooks = crate::plugin::test_support::OpHooks::default();
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            None,
            0,
            Some(hooks.plugins()),
        );
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let wait = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    Ok(1_u32)
                })
                .name("s")
                .spawn();
            let (wait, _step) = tokio::join!(wait, step);
            Ok(format!("{}", wait.is_err()))
        })
        .await
        .expect("no panic");
        assert_eq!(
            outcome,
            crate::driver::InvocationOutcome::Complete("true".to_owned())
        );
        let hooks_of_w: Vec<_> = hooks.of("w").iter().map(|r| r.hook).collect();
        assert_eq!(hooks_of_w, ["start"]);
    }

    /// A new wait fires a live start hook with status `Started` once its
    /// START is written, even when the response carries no record.
    #[tokio::test]
    async fn live_wait_start_reports_started() {
        use crate::client::InMemoryExecutionClient;

        let hooks = crate::plugin::test_support::OpHooks::default();
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::new(InMemoryExecutionClient::new(Vec::new())),
            "token0".to_owned(),
            None,
            0,
            Some(hooks.plugins()),
        );
        let signal = Arc::clone(ctx.coordinator());
        let outcome = crate::driver::test_support::outcome_of(
            ctx.clone(),
            Arc::clone(ctx.task_ownership()),
            signal,
            ctx.wait(Duration::from_secs(5)).name("w"),
        )
        .await;
        assert_eq!(outcome, crate::driver::InvocationOutcome::Pending);
        let shapes: Vec<_> = hooks
            .records()
            .iter()
            .map(|r| (r.hook, r.replay, r.status))
            .collect();
        assert_eq!(
            shapes,
            [("start", false, Some(crate::OperationStatus::Started))]
        );
    }

    /// Without plugins, a replayed `SUCCEEDED` wait returns.
    #[tokio::test]
    async fn succeeded_wait_without_plugins_returns() {
        let log = Arc::new(CheckpointLog::from_records(vec![wait_record_with(
            CheckpointStatus::Succeeded,
        )]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        assert!(ctx.plugins().is_none());
        ctx.wait(Duration::from_secs(5)).await.unwrap();
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

#[cfg(test)]
mod coordinator_tests {
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{
        RefreshClient, drive_handler, named, operation_like, wire_err,
    };
    use aws_sdk_lambda::types::{OperationAction, OperationStatus, WaitDetails};
    use std::sync::Arc;
    use std::time::Duration;

    const fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    /// A wait that the first refresh completes, while a step sleeps 3 s
    /// beside it, fires its end hook live in the same invocation.
    #[tokio::test(start_paused = true)]
    async fn wait_resumed_in_invocation_ends_live() {
        let client = Arc::new(RefreshClient::new(|req| {
            if let Some(start) = named(req.updates, "w", &OperationAction::Start) {
                let record = operation_like(start, OperationStatus::Started)
                    .wait_details(
                        WaitDetails::builder()
                            .scheduled_end_timestamp(req.timestamp_in(secs(1)))
                            .build(),
                    )
                    .build()
                    .expect("all required fields set");
                return Ok(vec![record]);
            }
            if req.empty_number == 1
                && let Some(start) = named(req.history, "w", &OperationAction::Start)
            {
                let record = operation_like(start, OperationStatus::Succeeded)
                    .build()
                    .expect("all required fields set");
                return Ok(vec![record]);
            }
            Ok(Vec::new())
        }));
        let hooks = crate::plugin::test_support::OpHooks::default();
        let ctx = crate::DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            None,
            0,
            Some(hooks.plugins()),
        );
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let wait = ctx.wait(secs(1)).name("w").spawn();
            let step = ctx
                .step(|_| async {
                    tokio::time::sleep(secs(3)).await;
                    Ok(1_u32)
                })
                .name("s")
                .spawn();
            let (wait, step) = tokio::join!(wait, step);
            wait.map_err(wire_err)?;
            step.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        let shapes: Vec<_> = hooks
            .of("w")
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
    }
}
