//! The start step: the synchronous first phase of an operation.
//!
//! The JS SDK begins each non-combinator operation inside the call that
//! creates it. Its checkpoint manager queues an update synchronously, and
//! sends the queue only after the current synchronous code has returned.
//! [`start_step`] is the Rust form of that phase. It is a synchronous
//! function of the operation's execution state, so nothing is awaited
//! before it returns. In order, it:
//!
//! 1. returns [`StartStep::Refused`] when the context's admission refuses
//!    the operation, and queues nothing;
//! 2. runs the task-ownership check;
//! 3. runs the replay identity check and reads the operation's record;
//! 4. when the cycle needs a `START` (no record, or a record that begins a
//!    new attempt), builds it and fixes its position in the write order;
//! 5. for an operation whose JS order is "queue `START`, then hook", fires
//!    `on_operation_start` and then releases the `START`;
//! 6. for a record that resumes work without a new `START`, fires its
//!    replay start hook.
//!
//! The order of the `START` write and `on_operation_start` follows JS for
//! each kind ([`StartOrder`]):
//!
//! - A step under `AtLeastOncePerRetry` and `wait_for_condition` queue
//!   `START`, fire `on_operation_start`, and do not wait for the response.
//!   The `START` is reserved (see
//!   [`ReservedWrite`](crate::context::ReservedWrite)) until the hook
//!   dispatch returns, so the hook returns before the request is sent on
//!   any runtime, as JS's `setImmediate` send runs after the hook.
//! - A step under `AtMostOncePerRetry`, a wait and a callback queue
//!   `START` here, and the rest of the operation awaits the response
//!   before it fires `on_operation_start`, as JS awaits the `START`
//!   promise before the hook. A callback's `START` is urgent: it flushes
//!   the coalescing buffer at once, because the response assigns the
//!   callback ID.
//!
//! `on_attempt_start` is not part of the start step. The rest of the
//! operation fires it before the body, as JS fires
//! `onOperationAttemptStart` after `onOperationStart`.

use aws_sdk_lambda::types::OperationUpdate;

use crate::context::{DurableContext, SubmittedWrite};
use crate::engine::{CheckpointStatusView, OperationId};
use crate::error::OperationError;
use crate::plugin::InvocationPlugins;

/// When an operation fires `on_operation_start` relative to its `START`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StartOrder {
    /// Queue `START`, fire the hook, then release `START`. The rest of the
    /// operation does not wait for the response before its body.
    QueueThenHook,
    /// Queue `START`. The rest of the operation awaits the response, then
    /// fires the hook.
    AwaitResponse,
    /// As [`Self::AwaitResponse`], with an urgent `START` that flushes
    /// the coalescing buffer at once.
    AwaitUrgentResponse,
}

/// What the start step does for an operation that has a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cycle {
    /// The record begins a new attempt: queue its `START`. `is_replay` is
    /// the flag a [`StartOrder::QueueThenHook`] hook sees.
    Start { is_replay: bool },
    /// The record resumes work without a new `START`: fire the replay
    /// start hook.
    Resume,
    /// The record queues no `START` and fires no start hook: a terminal
    /// or waiting record, which the rest of the operation handles.
    Record,
}

/// The identity and start order of one operation.
pub(crate) struct StartSpec<'a> {
    pub(crate) op_id: &'a OperationId,
    /// The record type the replay identity check expects.
    pub(crate) operation_type: &'static str,
    pub(crate) sub_type: &'a str,
    pub(crate) name: Option<&'a str>,
    pub(crate) order: StartOrder,
}

/// The outcome of [`start_step`].
pub(crate) enum StartStep {
    /// Admission refused the operation. Nothing was queued, and the
    /// operation's future never resolves.
    Refused,
    /// The operation has a record, and this cycle queues no `START`.
    Recorded(CheckpointStatusView),
    /// The `START` write of a first or later attempt, already in the write
    /// order. For [`StartOrder::QueueThenHook`] it was released after the
    /// hook dispatch returned.
    Queued(SubmittedWrite),
}

/// Runs the start step of the operation `spec` on `ctx` (see the module
/// documentation).
///
/// `classify` decides what a record means for this cycle; without a
/// record the cycle starts a first attempt. `start_update` builds the
/// `START`. `start_hook` fires `on_operation_start` with the given replay
/// flag. The start step calls it for a [`StartOrder::QueueThenHook`]
/// `START` and for a [`Cycle::Resume`] record, and only when plugins are
/// registered.
///
/// # Errors
///
/// A task-ownership error, or a replay identity error. Nothing is queued
/// in either case.
pub(crate) fn start_step(
    ctx: &DurableContext,
    spec: &StartSpec<'_>,
    classify: impl FnOnce(&CheckpointStatusView) -> Cycle,
    start_update: impl FnOnce() -> OperationUpdate,
    start_hook: impl FnOnce(&InvocationPlugins, bool),
) -> Result<StartStep, OperationError> {
    if !ctx.admission().admits(ctx.owners()) {
        return Ok(StartStep::Refused);
    }
    ctx.enforce_task_ownership()?;
    let view = ctx.checkpoint_view_validated(
        spec.op_id.positional(),
        spec.op_id.wire(),
        spec.operation_type,
        Some(spec.sub_type),
        spec.name,
    )?;
    let is_replay = match view {
        None => false,
        Some(view) => match classify(&view) {
            Cycle::Start { is_replay } => is_replay,
            Cycle::Resume => {
                if let Some(plugins) = ctx.plugins() {
                    start_hook(plugins, true);
                }
                return Ok(StartStep::Recorded(view));
            }
            Cycle::Record => return Ok(StartStep::Recorded(view)),
        },
    };
    let update = start_update();
    let write = match spec.order {
        StartOrder::QueueThenHook => {
            let reserved = ctx.reserve_checkpoint_updates(vec![update]);
            if let Some(plugins) = ctx.plugins() {
                start_hook(plugins, is_replay);
            }
            reserved.release()
        }
        StartOrder::AwaitResponse => ctx.submit_checkpoint_updates(vec![update]),
        StartOrder::AwaitUrgentResponse => ctx.submit_checkpoint_updates_urgent(vec![update]),
    };
    Ok(StartStep::Queued(write))
}

#[cfg(test)]
#[expect(clippy::panic, clippy::indexing_slicing)] // reason: test assertions
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    use super::{Cycle, StartOrder, StartSpec, StartStep, start_step};
    use crate::client::{
        CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput,
        InMemoryExecutionClient,
    };
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus, OperationId};
    use crate::error::OperationError;
    use crate::plugin::test_support::OpHooks;
    use crate::step::{StepSemantics, step_cycle};

    /// The record of the first operation of the root, a step `a`.
    fn step_record(
        status: CheckpointStatus,
        attempt: u32,
        op_type: &str,
    ) -> (String, CheckpointRecord) {
        let id = crate::engine::compute_wire_id_public("1");
        let record = CheckpointRecord {
            id: id.clone(),
            status,
            result: None,
            error_type: None,
            error_message: None,
            error_data: None,
            stack_trace: None,
            attempt,
            invoke_result: None,
            invoke_error_type: None,
            invoke_error_message: None,
            invoke_error_data: None,
            invoke_stack_trace: None,
            replay_children: false,
            callback_id: None,
            op_type: Some(op_type.to_owned()),
            sub_type: Some("Step".to_owned()),
            due_at: None,
            op_name: Some("a".to_owned()),
            parent_id: None,
            start_timestamp: None,
            end_timestamp: None,
        };
        (id, record)
    }

    /// A root context over `records` and `client`, with the coalescer
    /// `window` (`None`: direct writes) and the plugin `hooks`.
    fn context(
        records: Vec<(String, CheckpointRecord)>,
        client: Arc<dyn ExecutionClient>,
        window: Option<Duration>,
        hooks: Option<&OpHooks>,
    ) -> DurableContext {
        DurableContext::new_root_with_client_and_defaults(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(records)),
            client,
            "token-0".to_owned(),
            window,
            hooks.map(OpHooks::plugins),
        )
    }

    fn in_memory() -> Arc<InMemoryExecutionClient> {
        Arc::new(InMemoryExecutionClient::new(Vec::new()))
    }

    fn start_update(op_id: &OperationId, r#type: OperationType, sub_type: &str) -> OperationUpdate {
        OperationUpdate::builder()
            .id(op_id.wire())
            .r#type(r#type)
            .sub_type(sub_type)
            .name("a")
            .action(OperationAction::Start)
            .build()
            .expect("all required OperationUpdate fields set")
    }

    /// Runs the start step of a step `a`, claimed on `ctx`, under
    /// `semantics`, as the step operation does.
    fn step_start(
        ctx: &DurableContext,
        semantics: StepSemantics,
    ) -> Result<StartStep, OperationError> {
        let op_id = ctx.mint_id();
        let order = match semantics {
            StepSemantics::AtLeastOncePerRetry => StartOrder::QueueThenHook,
            StepSemantics::AtMostOncePerRetry => StartOrder::AwaitResponse,
        };
        start_step(
            ctx,
            &StartSpec {
                op_id: &op_id,
                operation_type: "Step",
                sub_type: "Step",
                name: Some("a"),
                order,
            },
            |view| step_cycle(semantics, view),
            || start_update(&op_id, OperationType::Step, "Step"),
            |plugins, is_replay| {
                let info = crate::plugin::OperationInfo::for_operation(
                    ctx,
                    op_id.wire(),
                    crate::OperationKind::Step,
                    "Step",
                    Some("a"),
                )
                .with_status(crate::OperationStatus::Started)
                .with_replay(is_replay);
                crate::plugin::operation_started(plugins, &info);
            },
        )
    }

    /// Runs the start step of a wait or a callback `a` with no record.
    fn response_then_hook_start(
        ctx: &DurableContext,
        operation_type: &'static str,
        r#type: OperationType,
        order: StartOrder,
    ) -> StartStep {
        let op_id = ctx.mint_id();
        start_step(
            ctx,
            &StartSpec {
                op_id: &op_id,
                operation_type,
                sub_type: operation_type,
                name: Some("a"),
                order,
            },
            |_| Cycle::Record,
            || start_update(&op_id, r#type, operation_type),
            |_, _| unreachable!("no record resumes, and the hook follows the response"),
        )
        .expect("the start step succeeds")
    }

    /// The `(hook, replay)` of every start hook `hooks` recorded.
    fn starts(hooks: &OpHooks) -> Vec<bool> {
        hooks
            .records()
            .iter()
            .filter(|r| r.hook == "start")
            .map(|r| r.replay)
            .collect()
    }

    #[tokio::test]
    async fn start_step_queues_start_before_returning() {
        for window in [None, Some(Duration::ZERO)] {
            let client = in_memory();
            let ctx = context(Vec::new(), client.clone(), window, None);
            let started = step_start(&ctx, StepSemantics::AtLeastOncePerRetry).expect("no error");
            assert!(matches!(started, StartStep::Queued(_)), "{window:?}");
            assert_eq!(
                ctx.queued_update_ids(),
                [crate::engine::compute_wire_id_public("1")],
                "START is queued before anything is polled ({window:?})"
            );
            assert!(client.recorded_updates().is_empty(), "{window:?}");
            let StartStep::Queued(write) = started else {
                unreachable!("checked above")
            };
            write.await.expect("START is written");
            assert_eq!(client.recorded_updates().len(), 1, "{window:?}");
        }
    }

    #[tokio::test]
    async fn start_step_returns_record_without_queuing() {
        let ctx = context(
            vec![step_record(CheckpointStatus::Started, 0, "Step")],
            in_memory(),
            None,
            None,
        );
        let started = step_start(&ctx, StepSemantics::AtMostOncePerRetry).expect("no error");
        let StartStep::Recorded(view) = started else {
            panic!("a STARTED record queues no START");
        };
        assert_eq!(view.status, CheckpointStatus::Started);
        assert_eq!(ctx.queued_update_ids(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn start_step_refused_by_admission_queues_nothing() {
        let client = in_memory();
        let root = context(Vec::new(), client.clone(), Some(Duration::ZERO), None);
        let child = root.new_child("1", None);
        root.admission()
            .abandon([crate::engine::compute_wire_id_public("1").as_str()]);
        let started = step_start(&child, StepSemantics::AtLeastOncePerRetry).expect("no error");
        assert!(matches!(started, StartStep::Refused));
        assert_eq!(child.queued_update_ids(), Vec::<String>::new());

        // The operation's future never resolves, and writes nothing.
        let op_id = child.mint_id();
        let step = crate::step::StepExecution {
            ctx: child.clone(),
            op_id,
            name: Some("b".to_owned()),
            retry_strategy: None,
            serdes: crate::JsonSerdes,
            semantics: StepSemantics::AtLeastOncePerRetry,
            sub_type: crate::OperationSubType::STEP,
            plugin_name: None,
            closure: |_| async { Ok(1_u32) },
            _marker: std::marker::PhantomData,
        };
        assert!(
            tokio::time::timeout(Duration::from_millis(50), step.execute())
                .await
                .is_err(),
            "a refused operation never resolves"
        );
        assert_eq!(client.recorded_updates(), Vec::new());
    }

    #[tokio::test]
    async fn start_step_fires_live_hook_only_for_queue_then_hook() {
        let hooks = OpHooks::default();
        let ctx = context(Vec::new(), in_memory(), None, Some(&hooks));
        let started = step_start(&ctx, StepSemantics::AtLeastOncePerRetry).expect("no error");
        assert!(matches!(started, StartStep::Queued(_)));
        assert_eq!(
            starts(&hooks),
            [false],
            "a default step fires the live hook"
        );

        let hooks = OpHooks::default();
        let ctx = context(Vec::new(), in_memory(), None, Some(&hooks));
        let started =
            response_then_hook_start(&ctx, "Wait", OperationType::Wait, StartOrder::AwaitResponse);
        assert!(matches!(started, StartStep::Queued(_)));
        assert!(
            starts(&hooks).is_empty(),
            "a wait fires nothing in the start step"
        );
    }

    #[tokio::test]
    async fn start_step_fires_replay_hook_for_started_record() {
        let hooks = OpHooks::default();
        let ctx = context(
            vec![step_record(CheckpointStatus::Started, 0, "Step")],
            in_memory(),
            None,
            Some(&hooks),
        );
        let started = step_start(&ctx, StepSemantics::AtLeastOncePerRetry).expect("no error");
        assert!(matches!(started, StartStep::Recorded(_)));
        assert_eq!(
            starts(&hooks),
            [true],
            "the replay hook fired before the return"
        );
        assert_eq!(ctx.queued_update_ids(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn start_step_queues_callback_start_urgently_before_returning() {
        let client = in_memory();
        let ctx = context(
            Vec::new(),
            client.clone(),
            Some(Duration::from_hours(1)),
            None,
        );
        let started = response_then_hook_start(
            &ctx,
            "Callback",
            OperationType::Callback,
            StartOrder::AwaitUrgentResponse,
        );
        let StartStep::Queued(write) = started else {
            panic!("a callback without a record queues START");
        };
        assert_eq!(
            ctx.queued_update_ids(),
            [crate::engine::compute_wire_id_public("1")],
            "START joined the coalescer"
        );
        tokio::task::yield_now().await;
        assert_eq!(
            client.recorded_updates().len(),
            1,
            "the urgent flush reached the client without the delay"
        );
        write.await.expect("START is written");
    }

    #[tokio::test]
    async fn start_step_returns_replay_identity_error_without_queuing() {
        let ctx = context(
            vec![step_record(CheckpointStatus::Started, 0, "Wait")],
            in_memory(),
            None,
            None,
        );
        let Err(err) = step_start(&ctx, StepSemantics::AtLeastOncePerRetry) else {
            panic!("a record of another type fails the replay identity check");
        };
        assert!(err.to_string().contains("non_deterministic"), "{err}");
        assert_eq!(ctx.queued_update_ids(), Vec::<String>::new());
    }

    #[tokio::test]
    async fn start_step_queues_start_for_new_attempt_states() {
        for status in [
            CheckpointStatus::Ready,
            CheckpointStatus::Cancelled,
            CheckpointStatus::TimedOut,
            CheckpointStatus::Stopped,
        ] {
            let hooks = OpHooks::default();
            let client = in_memory();
            let ctx = context(
                vec![step_record(status.clone(), 1, "Step")],
                client.clone(),
                None,
                Some(&hooks),
            );
            let started = step_start(&ctx, StepSemantics::AtLeastOncePerRetry).expect("no error");
            assert_eq!(
                starts(&hooks),
                [true],
                "{status:?}: the retry's hook is a replay, fired before the return"
            );
            assert_eq!(ctx.queued_update_ids().len(), 1, "{status:?}");
            let StartStep::Queued(write) = started else {
                panic!("{status:?} queues START");
            };
            write.await.expect("START is written");
            let updates = client.recorded_updates();
            assert_eq!(updates.len(), 1, "{status:?}");
            assert_eq!(updates[0].action(), &OperationAction::Start, "{status:?}");
        }
        let hooks = OpHooks::default();
        let ctx = context(
            vec![step_record(CheckpointStatus::Started, 1, "Step")],
            in_memory(),
            None,
            Some(&hooks),
        );
        let started = step_start(&ctx, StepSemantics::AtLeastOncePerRetry).expect("no error");
        assert!(matches!(started, StartStep::Recorded(_)));
        assert_eq!(ctx.queued_update_ids(), Vec::<String>::new());
    }

    /// A client that logs `response <name>` for each `START` when its
    /// checkpoint future completes, after one yield.
    #[derive(Default)]
    struct LoggingClient(Arc<Mutex<Vec<String>>>);

    impl ExecutionClient for LoggingClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            let starts: Vec<String> = updates
                .iter()
                .filter(|u| u.action() == &OperationAction::Start)
                .map(|u| u.name().unwrap_or("-").to_owned())
                .collect();
            let log = Arc::clone(&self.0);
            Box::pin(async move {
                tokio::task::yield_now().await;
                for name in starts {
                    log.lock().unwrap().push(format!("response {name}"));
                }
                Ok(CheckpointOutput::new(
                    Some("t".to_owned()),
                    Vec::new(),
                    None,
                ))
            })
        }

        fn get_state(
            &self,
            _execution_arn: &str,
            _checkpoint_token: &str,
            _marker: Option<&str>,
        ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>>
        {
            Box::pin(async { Ok(GetStateOutput::new(Vec::new(), None)) })
        }
    }

    #[tokio::test]
    async fn at_most_once_retry_start_hook_fires_after_start_response() {
        let client = Arc::new(LoggingClient::default());
        let log = Arc::clone(&client.0);
        let hooks = OpHooks::probing(move || {
            log.lock()
                .unwrap()
                .iter()
                .any(|e| e == "response a")
                .to_string()
        });
        let ctx = context(
            vec![step_record(CheckpointStatus::Ready, 1, "Step")],
            client,
            None,
            Some(&hooks),
        );
        let op_id = ctx.mint_id();
        let value = crate::step::StepExecution {
            ctx: ctx.clone(),
            op_id,
            name: Some("a".to_owned()),
            retry_strategy: None,
            serdes: crate::JsonSerdes,
            semantics: StepSemantics::AtMostOncePerRetry,
            sub_type: crate::OperationSubType::STEP,
            plugin_name: None,
            closure: |_| async { Ok(1_u32) },
            _marker: std::marker::PhantomData,
        }
        .execute()
        .await
        .expect("the step succeeds");
        assert_eq!(value, 1);
        let start = hooks
            .records()
            .into_iter()
            .find(|r| r.hook == "start")
            .expect("the start hook fired");
        assert!(start.replay, "the retry's start hook is a replay");
        assert_eq!(start.probe, "true", "response a came before the hook");
    }
}
