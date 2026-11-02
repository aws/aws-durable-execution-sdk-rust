//!
//! The driver polls the user's handler future and controls the execution
//! lifecycle of one invocation:
//!
//! - **Suspension**: the invocation coordinator
//!   ([`crate::coordinator`]) decides when the invocation suspends. Local
//!   work that can make progress holds a permit: a polled or woken task, a
//!   step body, a `wait_for_condition` check, a serdes call, blocking work,
//!   and an admitted checkpoint write until its response is merged. An
//!   operation that waits on the service holds a wait registration. When no
//!   permit is held and at least one registration is live, the phase
//!   becomes `Suspending` and the driver ends the invocation with
//!   `PENDING`. The handler future is dropped at its current await point,
//!   so a suspension never surfaces to user code.
//!
//! - **Continuation inside the invocation**: a merged checkpoint response
//!   wakes the registrations of the operations it carries. A wait whose
//!   record is complete continues in the same invocation, before the
//!   response's checkpoint permit is released.
//!
//! - **Resume**: on a later invocation with existing checkpoint state, the
//!   engine reconstructs replay mode via [`CheckpointLog`] and re-runs the
//!   handler. Replayed operations return frozen results from the log
//!   without re-execution.
//!
//! - **Task-ownership**: each context records the `tokio::task::Id` of the
//!   task that created it. Durable operations invoked from a DIFFERENT task
//!   fail fast with a clear error. The engine's own tasks (the handler,
//!   `.spawn()`, batch items and combinator inputs) are exempt.
//!
//! Zero permits and zero registrations do not suspend: the handler awaits
//! work the SDK does not track, and the invocation runs until that work
//! finishes or the Lambda deadline passes. The driver logs a warning when
//! the deadline is reached in that state.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use crate::coordinator::{InvocationCoordinator, Phase};

// Imports used by tests: suppress dead-code warnings for the engine types
// which are consumed by operation execution.
#[cfg(test)]
use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus, EngineState};

/// The warning logged when the Lambda deadline is reached while nothing the
/// SDK tracks runs and no operation waits.
pub(crate) const UNTRACKED_DEADLINE_WARNING: &str = "the invocation reached its deadline while \
     the handler awaited work the SDK does not track; run that work in a step";

/// The outcome of driving a single invocation of the user's handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InvocationOutcome {
    /// The handler completed successfully with a serialized result.
    Complete(String),
    /// The handler suspended: an operation requires an external event
    /// before it can proceed. The engine drops the handler future and
    /// reports PENDING to the runtime.
    Pending,
    /// The handler returned an error with its wire failure record.
    Failed {
        /// The wire failure record (type, message, opaque data, stack
        /// trace) reported in the response envelope.
        error: crate::error::WireError,
    },
    /// The INVOCATION itself must fail, a Lambda runtime error rather
    /// than a `FAILED` envelope, so the durable service re-invokes, the
    /// same recovery as an interruption. Produced by a retryable
    /// checkpoint write failure that exhausted transport-level retries:
    /// the write channel is down, so nothing more is written and the
    /// execution stays alive for the retry.
    Fault {
        /// The message reported as the Lambda invocation error.
        message: String,
    },
}

/// An unrecoverable error recorded by an operation, together with the
/// scope that must die for the execution to converge.
///
/// Producers:
/// - non-determinism detection
///   ([`crate::context::DurableContext::validate_replay_identity`]): a
///   replay identity mismatch means no amount of re-invocation can make
///   progress: [`FatalSeverity::FailExecution`];
/// - checkpoint write failures
///   ([`crate::context::DurableContext::checkpoint_failure_unrecoverable`]):
///   a retryable failure fails the invocation so the service re-invokes
///   ([`FatalSeverity::FailInvocation`]); a permanent rejection fails the
///   execution ([`FatalSeverity::FailExecution`]).
/// - serdes faults ([`crate::context::DurableContext::serdes_fault`]): a
///   serdes error marked with
///   [`RetryableSerdesError`](crate::serdes::RetryableSerdesError) fails the
///   invocation so the service re-invokes
///   ([`FatalSeverity::FailInvocation`]).
///
/// The invocation driver checks this slot with priority over both
/// completion and suspension.
#[derive(Debug, Clone)]
pub(crate) struct FatalError {
    /// The wire error type (e.g. `NonDeterministicExecutionError`).
    pub(crate) error_type: String,
    /// The full error message.
    pub(crate) error_message: String,
    /// Which scope dies: the execution (terminal `FAILED` envelope) or
    /// only this invocation (Lambda runtime error; the service
    /// re-invokes).
    pub(crate) severity: FatalSeverity,
}

/// The recovery scope of a [`FatalError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FatalSeverity {
    /// The execution is over: report a `FAILED` envelope carrying the
    /// error, which the service records as the execution result.
    FailExecution,
    /// Only this invocation dies: report a Lambda runtime error (no
    /// envelope), and the service re-invokes: the interruption recovery
    /// path.
    FailInvocation,
}

impl FatalError {
    /// Converts the recorded fatal into the invocation outcome its
    /// severity demands.
    pub(crate) fn into_outcome(self) -> InvocationOutcome {
        match self.severity {
            FatalSeverity::FailExecution => InvocationOutcome::Failed {
                error: crate::error::WireError::new(
                    Some(self.error_type),
                    Some(self.error_message),
                ),
            },
            FatalSeverity::FailInvocation => InvocationOutcome::Fault {
                message: self.error_message,
            },
        }
    }
}

/// Identifies the owning task and tracks blessed (engine-spawned) task IDs.
///
/// Each `DurableContext` records the `tokio::task::Id` of the task that
/// created it. Durable operations check ownership before proceeding:
/// operations from a foreign task fail fast.
///
/// **Production topology**: `lambda_runtime` awaits the invocation future
/// inline under `block_on`, so `tokio::task::try_id()` returns `None` at
/// context-creation time. The guard handles this as an explicit
/// "root-context" marker: operations invoked from the same inline context
/// (where `try_id()` is also `None`) pass, but operations invoked from a
/// foreign `tokio::spawn` (where `try_id()` returns `Some`) are rejected
/// unless the task was blessed by the SDK.
///
/// The task gateway (`crate::task`) blesses every operation task it starts:
/// the handler itself, each `.spawn()`, each map item and parallel branch,
/// and each combinator input. This catches user `tokio::spawn` misuse while
/// allowing the SDK's own tasks.
#[derive(Debug)]
pub(crate) struct TaskOwnership {
    /// The task ID that owns this context. `None` if created outside a
    /// spawned task (e.g. in `block_on`: the production path). A `None`
    /// owner acts as a root-context marker: inline callers (also `None`)
    /// pass, but unblessed spawned tasks are rejected.
    owner_task_id: Option<tokio::task::Id>,
    /// The IDs of the running SDK operation tasks: these are exempt from
    /// the check. The task gateway (`crate::task`) inserts a task's ID
    /// before the task's first poll and removes it before the task exits,
    /// so tokio can reuse an ID only for a task that is not in the set.
    blessed_tasks: std::sync::Mutex<std::collections::HashSet<tokio::task::Id>>,
}

impl TaskOwnership {
    /// Creates ownership tracking anchored to the current task.
    ///
    /// If called outside a tokio spawned task (e.g. directly in
    /// `block_on`), the owner is recorded as `None`: the "root-context"
    /// marker. This is the production path: `lambda_runtime` awaits the
    /// handler inline under `block_on`, so `try_id()` returns `None`.
    /// The guard still activates: operations invoked from unblessed
    /// spawned tasks (where `try_id()` returns `Some`) are rejected.
    pub(crate) fn new_current() -> Self {
        Self {
            owner_task_id: tokio::task::try_id(),
            blessed_tasks: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Registers a task ID as blessed (engine-spawned).
    ///
    /// Blessed tasks pass the ownership check even though they are not the
    /// original owner.
    pub(crate) fn bless_task(&self, task_id: tokio::task::Id) {
        let mut blessed = self
            .blessed_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        blessed.insert(task_id);
    }

    /// Removes a task ID from the blessed set. The task gateway calls it
    /// before a blessed task exits.
    pub(crate) fn unbless_task(&self, task_id: tokio::task::Id) {
        self.blessed_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&task_id);
    }

    /// Returns the number of blessed task IDs.
    #[cfg(test)]
    pub(crate) fn blessed_count(&self) -> usize {
        self.blessed_tasks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Checks whether the calling task is authorized to perform durable
    /// operations on this context.
    ///
    /// Returns `Ok(())` if the caller is the owner or a blessed task.
    /// Returns `Err` with a descriptive message if unauthorized.
    ///
    /// **Root-context mode** (owner is `None`: production path): callers
    /// with `try_id() == None` (inline under the same `block_on`) pass.
    /// Callers with a task ID must be blessed; otherwise they are foreign
    /// spawned tasks and the operation is rejected.
    ///
    /// **Task-context mode** (owner is `Some`): callers must be the same
    /// task or a blessed task; otherwise rejected.
    pub(crate) fn check_current_task(&self) -> Result<(), String> {
        match self.owner_task_id {
            None => {
                // Root-context mode: owner was created inline under block_on
                // (the production lambda_runtime topology).
                let Some(current_id) = tokio::task::try_id() else {
                    // Caller is also inline (no task ID): same context.
                    return Ok(());
                };
                // Caller has a task ID: must be blessed to proceed.
                let blessed = self
                    .blessed_tasks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if blessed.contains(&current_id) {
                    return Ok(());
                }
                Err(format!(
                    "durable operation invoked from task {current_id:?}, but context is owned by \
                     the root handler (no task). Use .spawn() instead of tokio::spawn for durable \
                     fan-out"
                ))
            }
            Some(owner_id) => {
                // Task-context mode: owner was created inside a tokio::spawn.
                let Some(current_id) = tokio::task::try_id() else {
                    return Err("durable operation invoked outside a tokio task context".to_owned());
                };
                if current_id == owner_id {
                    return Ok(());
                }
                let blessed = self
                    .blessed_tasks
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if blessed.contains(&current_id) {
                    return Ok(());
                }
                Err(format!(
                    "durable operation invoked from task {current_id:?}, but context is owned by \
                     task {owner_id:?}. Use .spawn() instead of tokio::spawn for durable fan-out"
                ))
            }
        }
    }
}

/// How the drive of the root task ended.
enum DriveEnd<T> {
    /// The root task resolved or panicked.
    Completed(crate::task::TaskOutcome<T>),
    /// A fatal error was recorded, or the phase became `Suspending`.
    Interrupted,
}

/// Drives a single invocation of the user's handler future.
///
/// The handler runs as the invocation's root task, started through the
/// task gateway (`crate::task::spawn_root`), and polled by its
/// permit-aware wrapper. The drive ends in one of four ways:
///
/// 1. A fatal error is recorded: the fatal outcome, with precedence over
///    everything else.
/// 2. The phase becomes `Suspending`: `Pending`.
/// 3. The handler resolves: `Complete` or `Failed`. A wait whose handle the
///    handler dropped does not change that; its record can stay `STARTED`.
/// 4. The handler panics: `Err` with the original payload, which the caller
///    resumes.
///
/// When `remaining` is `Some(d)`, the driver also polls a sleep of `d`
/// directly, so a timer wake takes no permit. When it fires while the
/// phase is `Running`, no permit is held and no wait is live, it logs
/// [`UNTRACKED_DEADLINE_WARNING`] once.
///
/// Cleanup, after every ending: the phase becomes `Finished`, the root task
/// is aborted and awaited, every other tracked task is aborted, and the
/// driver waits until every tracked task has unwound. Dropping the futures
/// retires their wait registrations and releases their permits. So when
/// this function returns, no SDK task, permit or wait registration of this
/// invocation remains.
pub(crate) async fn drive_invocation<F>(
    handler_future: F,
    task_ownership: Arc<TaskOwnership>,
    coordinator: Arc<InvocationCoordinator>,
    remaining: Option<Duration>,
) -> Result<InvocationOutcome, Box<dyn std::any::Any + Send + 'static>>
where
    F: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
{
    // The handler runs as an operation task, so every SDK poll of user code
    // runs on a Tokio worker with the gateway wrapper.
    #[expect(clippy::expect_used)]
    // reason: this function awaits Tokio timers and tasks itself, so it is
    // always polled inside a Tokio runtime.
    let (mut root, root_join) =
        crate::task::spawn_root(&task_ownership, &coordinator, handler_future)
            .expect("the invocation driver runs inside a Tokio runtime");

    let mut deadline = remaining.map(|d| Box::pin(tokio::time::sleep(d)));
    let ended = std::future::poll_fn(|cx| {
        coordinator.register_driver_waker(cx.waker());
        if coordinator.fatal_error().is_some() {
            return Poll::Ready(DriveEnd::Interrupted);
        }
        if let Poll::Ready(outcome) = Pin::new(&mut root).poll(cx) {
            return Poll::Ready(DriveEnd::Completed(outcome));
        }
        if coordinator.phase() == Phase::Suspending {
            return Poll::Ready(DriveEnd::Interrupted);
        }
        if let Some(sleep) = deadline.as_mut()
            && sleep.as_mut().poll(cx).is_ready()
        {
            deadline = None;
            if coordinator.running_and_idle() {
                tracing::warn!("{UNTRACKED_DEADLINE_WARNING}");
            }
        }
        Poll::Pending
    })
    .await;

    // Cleanup, whatever ended the drive. No task is polled from here on.
    // The root task is aborted and awaited, so the handler has unwound and
    // dropped every handle it held before the other tasks are aborted.
    // Then the driver waits until every tracked task has unwound, so no SDK
    // task outlives the invocation.
    coordinator.finish();
    root.abort();
    drop(root);
    drop(root_join.await);
    let tasks = Arc::clone(coordinator.tasks());
    tasks.abort_all();
    tasks.wait_idle().await;

    // Fatal precedence: whatever ended the drive, a recorded fatal error
    // is the outcome. This read runs after every tracked task has unwound,
    // so a fatal that a task records during the cleanup also wins over a
    // handler panic.
    if let Some(fatal) = coordinator.fatal_error() {
        return Ok(fatal.into_outcome());
    }

    match ended {
        DriveEnd::Completed(crate::task::TaskOutcome::Panicked(payload)) => Err(payload),
        DriveEnd::Completed(crate::task::TaskOutcome::Value(Ok(result))) => {
            Ok(InvocationOutcome::Complete(result))
        }
        DriveEnd::Completed(crate::task::TaskOutcome::Value(Err(err))) => {
            Ok(InvocationOutcome::Failed { error: err })
        }
        DriveEnd::Interrupted => Ok(InvocationOutcome::Pending),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{InvocationOutcome, TaskOwnership, drive_invocation};
    use crate::client::{CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput};
    use crate::coordinator::InvocationCoordinator;
    use aws_sdk_lambda::types::{Operation, OperationStatus, OperationUpdate};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Drives `fut` through [`drive_invocation`] and resumes a handler
    /// panic, for tests that expect an outcome.
    pub(crate) async fn drive_ok<F>(
        fut: F,
        ownership: Arc<TaskOwnership>,
        coordinator: Arc<InvocationCoordinator>,
    ) -> InvocationOutcome
    where
        F: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
    {
        match drive_invocation(fut, ownership, coordinator, None).await {
            Ok(outcome) => outcome,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Drives an operation future through the real invocation driver and
    /// returns the invocation outcome. The operation's own result is
    /// discarded: a waiting operation yields
    /// [`InvocationOutcome::Pending`] exactly as in production, and a
    /// completing one yields `Complete`. Used by operation unit tests to
    /// drive a waiting future to an outcome instead of awaiting it directly
    /// (a direct await would wait forever).
    ///
    /// `ownership` is the task-ownership tracker of the context the
    /// operation runs on: the driver blesses its root task there.
    pub(crate) async fn outcome_of<F>(
        ownership: Arc<TaskOwnership>,
        coordinator: Arc<InvocationCoordinator>,
        fut: F,
    ) -> InvocationOutcome
    where
        F: IntoFuture + Send + 'static,
        F::IntoFuture: Send + 'static,
    {
        // Box the operation future: `drive_invocation` stack-pins its
        // argument, so a large operation future would otherwise inflate
        // every test future that awaits this helper (clippy
        // `large_futures`). Test-only; production handler futures are
        // driven unboxed.
        let fut = Box::pin(async move {
            let _ = fut.await;
            Ok::<_, crate::error::WireError>("ok".to_owned())
        });
        match drive_invocation(fut, ownership, coordinator, None).await {
            Ok(outcome) => outcome,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    /// Drives `fut` to its invocation outcome and asserts that it ended the
    /// invocation with a fault whose message contains `needle`.
    #[expect(clippy::panic)] // reason: test assertion on an unexpected outcome
    pub(crate) async fn expect_serdes_fault<F>(
        ownership: Arc<TaskOwnership>,
        coordinator: Arc<InvocationCoordinator>,
        fut: F,
        needle: &str,
    ) where
        F: IntoFuture + Send + 'static,
        F::IntoFuture: Send + 'static,
    {
        let outcome = tokio::time::timeout(
            Duration::from_secs(5),
            outcome_of(ownership, coordinator, fut),
        )
        .await
        .expect("the operation must reach an invocation outcome within 5 s");
        let InvocationOutcome::Fault { message } = outcome else {
            panic!("expected an invocation fault, got {outcome:?}");
        };
        assert!(
            message.contains(needle),
            "fault message {message:?} must contain {needle:?}"
        );
    }

    /// Decides the operations one checkpoint response carries, from the
    /// updates of the call and every update received so far (this call's
    /// included).
    type Responder = dyn Fn(&[OperationUpdate], &[OperationUpdate]) -> Vec<Operation> + Send + Sync;

    /// A test `ExecutionClient` that answers each checkpoint call with the
    /// operations its responder returns, optionally after a delay of tokio
    /// time, and records every call.
    pub(crate) struct ScriptedClient {
        respond: Box<Responder>,
        delay: Option<Duration>,
        /// Notified before the first call returns, when set.
        first_call_gate: Option<Arc<tokio::sync::Notify>>,
        calls: Mutex<Vec<(String, Vec<OperationUpdate>)>>,
        tokens: Mutex<u32>,
    }

    impl ScriptedClient {
        /// A client whose responses carry what `respond` returns.
        pub(crate) fn new(
            respond: impl Fn(&[OperationUpdate], &[OperationUpdate]) -> Vec<Operation>
            + Send
            + Sync
            + 'static,
        ) -> Self {
            Self {
                respond: Box::new(respond),
                delay: None,
                first_call_gate: None,
                calls: Mutex::new(Vec::new()),
                tokens: Mutex::new(0),
            }
        }

        /// A client whose responses carry nothing.
        pub(crate) fn empty() -> Self {
            Self::new(|_, _| Vec::new())
        }

        /// Answers each call after `delay` of tokio time.
        pub(crate) fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = Some(delay);
            self
        }

        /// Holds the first call's response until `gate` is notified.
        pub(crate) fn with_first_call_gate(mut self, gate: Arc<tokio::sync::Notify>) -> Self {
            self.first_call_gate = Some(gate);
            self
        }

        /// Every call's checkpoint token and updates, in call order.
        pub(crate) fn calls(&self) -> Vec<(String, Vec<OperationUpdate>)> {
            self.calls.lock().expect("lock").clone()
        }

        /// Every update received, in call order.
        pub(crate) fn updates(&self) -> Vec<OperationUpdate> {
            self.calls().into_iter().flat_map(|(_, u)| u).collect()
        }

        /// The number of checkpoint calls received.
        pub(crate) fn call_count(&self) -> usize {
            self.calls.lock().expect("lock").len()
        }

        /// Whether an update of `name` with `action` was received.
        pub(crate) fn received(
            &self,
            name: &str,
            action: &aws_sdk_lambda::types::OperationAction,
        ) -> bool {
            self.updates()
                .iter()
                .any(|u| u.name() == Some(name) && u.action() == action)
        }
    }

    impl ExecutionClient for ScriptedClient {
        fn checkpoint(
            &self,
            _execution_arn: &str,
            checkpoint_token: &str,
            updates: Vec<OperationUpdate>,
        ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_>>
        {
            let (operations, first) = {
                let mut calls = self.calls.lock().expect("lock");
                let mut history: Vec<OperationUpdate> =
                    calls.iter().flat_map(|(_, u)| u.iter().cloned()).collect();
                history.extend(updates.iter().cloned());
                let operations = (self.respond)(&updates, &history);
                calls.push((checkpoint_token.to_owned(), updates));
                (operations, calls.len() == 1)
            };
            let token = {
                let mut tokens = self.tokens.lock().expect("lock");
                *tokens += 1;
                format!("t{tokens}")
            };
            let delay = self.delay;
            let gate = if first {
                self.first_call_gate.clone()
            } else {
                None
            };
            Box::pin(async move {
                if let Some(gate) = gate {
                    gate.notified().await;
                }
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                Ok(CheckpointOutput::new(Some(token), operations, None))
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

    /// A root context on `client`, with an empty checkpoint log.
    pub(crate) fn scripted_ctx(client: &Arc<ScriptedClient>) -> crate::DurableContext {
        crate::DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(client) as Arc<dyn ExecutionClient>,
            "t0".to_owned(),
        )
    }

    /// An operation record with the identity of `update` (its ID, type,
    /// sub-type, name and parent) and `status`, as a checkpoint response
    /// carries it.
    pub(crate) fn operation_like(
        update: &OperationUpdate,
        status: OperationStatus,
    ) -> aws_sdk_lambda::types::builders::OperationBuilder {
        let mut builder = Operation::builder()
            .id(update.id())
            .r#type(update.r#type().clone())
            .status(status)
            .start_timestamp(aws_smithy_types::DateTime::from_secs(0));
        if let Some(sub_type) = update.sub_type() {
            builder = builder.sub_type(sub_type);
        }
        if let Some(name) = update.name() {
            builder = builder.name(name);
        }
        if let Some(parent) = update.parent_id() {
            builder = builder.parent_id(parent);
        }
        builder
    }

    /// The first update in `updates` named `name` with `action`.
    pub(crate) fn named<'a>(
        updates: &'a [OperationUpdate],
        name: &str,
        action: &aws_sdk_lambda::types::OperationAction,
    ) -> Option<&'a OperationUpdate> {
        updates
            .iter()
            .find(|u| u.name() == Some(name) && u.action() == action)
    }

    /// Drives `handler` on `ctx` through [`drive_invocation`], with
    /// `remaining` as the time left.
    pub(crate) async fn drive_handler<F, Fut>(
        ctx: crate::DurableContext,
        remaining: Option<Duration>,
        handler: F,
    ) -> Result<InvocationOutcome, Box<dyn std::any::Any + Send + 'static>>
    where
        F: FnOnce(crate::DurableContext) -> Fut,
        Fut: Future<Output = Result<String, crate::error::WireError>> + Send + 'static,
    {
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());
        let root = Box::pin(handler(ctx));
        drive_invocation(root, ownership, coordinator, remaining).await
    }

    /// Records the level and message of every event while installed as
    /// the thread's default subscriber.
    #[derive(Clone, Default)]
    pub(crate) struct EventMessages(Arc<Mutex<Vec<(tracing::Level, String)>>>);

    struct MessageVisitor(String);

    impl tracing::field::Visit for MessageVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for EventMessages {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            let mut visitor = MessageVisitor(String::new());
            event.record(&mut visitor);
            self.0
                .lock()
                .expect("lock")
                .push((*event.metadata().level(), visitor.0));
        }
    }

    impl EventMessages {
        /// Installs a capture as the thread's default subscriber until the
        /// returned guard drops.
        pub(crate) fn install() -> (Self, tracing::subscriber::DefaultGuard) {
            use tracing_subscriber::layer::SubscriberExt as _;
            let capture = Self::default();
            let subscriber = tracing_subscriber::registry().with(capture.clone());
            (capture, tracing::subscriber::set_default(subscriber))
        }

        /// The messages of the events at `level`, in emission order.
        pub(crate) fn at(&self, level: tracing::Level) -> Vec<String> {
            self.0
                .lock()
                .expect("lock")
                .iter()
                .filter(|(l, _)| *l == level)
                .map(|(_, m)| m.clone())
                .collect()
        }

        /// The number of `operation_replayed` events.
        pub(crate) fn replayed(&self) -> usize {
            self.at(tracing::Level::DEBUG)
                .iter()
                .filter(|m| *m == crate::observability::event_names::OPERATION_REPLAYED)
                .count()
        }
    }

    /// Turns an operation error into the handler's wire error.
    #[expect(clippy::needless_pass_by_value)]
    // reason: it is passed to `map_err`, which hands over the error.
    pub(crate) fn wire_err(err: crate::OperationError) -> crate::error::WireError {
        crate::error::WireError::new(None::<String>, Some(format!("{err:#}")))
    }
}

#[cfg(test)]
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::test_support::drive_ok;
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    // Suspension drops the future at the await point

    /// A guard that sets a flag when dropped: proves the future was
    /// dropped at the await point.
    struct DropGuard {
        dropped: Arc<AtomicBool>,
    }

    impl Drop for DropGuard {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    /// A root context whose checkpoint calls carry nothing back, so a wait
    /// started on it stays waiting.
    fn waiting_ctx() -> crate::DurableContext {
        let client = Arc::new(test_support::ScriptedClient::empty());
        test_support::scripted_ctx(&client)
    }

    #[tokio::test]
    async fn suspension_drops_future_at_await_point() {
        let ctx = waiting_ctx();
        let drop_observed = Arc::new(AtomicBool::new(false));
        let after_await_ran = Arc::new(AtomicBool::new(false));

        let drop_clone = Arc::clone(&drop_observed);
        let after_clone = Arc::clone(&after_await_ran);
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            async move {
                let _guard = DropGuard {
                    dropped: drop_clone,
                };

                // A wait that nothing completes: the invocation suspends
                // and the driver drops this future.
                let _ = ctx.wait(Duration::from_secs(30)).await;

                // This code must NEVER run: the future is dropped above.
                after_clone.store(true, Ordering::SeqCst);
                Ok("should not reach".to_owned())
            },
            ownership,
            coordinator,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            drop_observed.load(Ordering::SeqCst),
            "future was not dropped at the await point"
        );
        assert!(
            !after_await_ran.load(Ordering::SeqCst),
            "code after the suspension point must not run"
        );
    }

    // Suspension is unswallowable (no catchable error)

    #[tokio::test]
    async fn suspension_unswallowable_by_catch_all() {
        let ctx = waiting_ctx();
        let after_catch_ran = Arc::new(AtomicBool::new(false));
        let after_clone = Arc::clone(&after_catch_ran);
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            async move {
                // User tries a "catch-all" pattern: wrapping in a closure
                // that catches panics. This cannot catch future-drop.
                let result: Result<String, crate::error::WireError> = async {
                    let _ = ctx.wait(Duration::from_secs(30)).await;
                    Ok("survived".to_owned())
                }
                .await;

                // If somehow the suspension was catchable, this would run:
                after_clone.store(true, Ordering::SeqCst);
                result
            },
            ownership,
            coordinator,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !after_catch_ran.load(Ordering::SeqCst),
            "catch-all must not intercept suspension — future is dropped"
        );
    }

    // PENDING outcome surfaces to the driver caller

    #[tokio::test]
    async fn pending_outcome_surfaces_to_caller() {
        let ctx = waiting_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let coordinator = Arc::clone(ctx.coordinator());

        let outcome = drive_ok(
            async move {
                let _ = ctx.wait(Duration::from_secs(30)).await;
                Ok("unreachable".to_owned())
            },
            ownership,
            coordinator,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
    }

    // Normal completion

    #[tokio::test]
    async fn driver_returns_complete_on_success() {
        let signal = InvocationCoordinator::new();

        let outcome = drive_ok(
            async move { Ok::<_, crate::error::WireError>("done".to_owned()) },
            Arc::new(TaskOwnership::new_current()),
            Arc::clone(&signal),
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    #[tokio::test]
    async fn driver_returns_failed_on_error() {
        let signal = InvocationCoordinator::new();

        let outcome = drive_ok(
            async move {
                Err(crate::error::WireError::new(
                    Some("Error".to_owned()),
                    Some("boom".to_owned()),
                ))
            },
            Arc::new(TaskOwnership::new_current()),
            Arc::clone(&signal),
        )
        .await;

        let InvocationOutcome::Failed { error } = outcome else {
            unreachable!("expected Failed, got {outcome:?}");
        };
        assert_eq!(error.error_type(), Some("Error"));
        assert_eq!(error.error_message(), Some("boom"));
    }

    // Resume: replayed operations return frozen results

    #[tokio::test]
    async fn resume_replays_frozen_results_without_re_executing() {
        // Simulate a resume invocation: checkpoint log has a result for
        // operation "1" (keyed by its wire ID), so the engine should
        // return the frozen value without calling the step body.
        let execution_count = Arc::new(AtomicU32::new(0));
        let exec_clone = Arc::clone(&execution_count);

        let wire_key = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire_key.clone(),
            CheckpointRecord {
                id: wire_key,
                status: CheckpointStatus::Succeeded,
                result: Some(r#""frozen_value""#.to_owned()),
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
            },
        )]));

        let engine = EngineState::new_root(Arc::clone(&log));

        // Simulate what the engine does on resume: check if replaying at
        // the operation's position.
        let op_id = engine.mint_id();
        let is_replay = engine.is_replaying_at(op_id.positional());

        let result = if is_replay {
            // Replay path: return frozen result without executing.
            let wire = crate::engine::compute_wire_id_public(op_id.positional());
            let record = log.get(&wire);
            record.unwrap().result.clone()
        } else {
            // Live path: execute the body (should NOT happen here).
            exec_clone.fetch_add(1, Ordering::SeqCst);
            Some(r#""live_value""#.to_owned())
        };

        assert_eq!(result, Some(r#""frozen_value""#.to_owned()));
        assert_eq!(
            execution_count.load(Ordering::SeqCst),
            0,
            "step body must NOT execute during replay"
        );
    }

    // Task ownership: foreign task fails fast

    #[tokio::test]
    async fn ownership_check_rejects_foreign_task() {
        // Must create ownership inside a spawned task where try_id()
        // returns Some: the #[tokio::test] root runs in block_on which
        // has no task ID.
        let result = tokio::spawn(async {
            let ownership = Arc::new(TaskOwnership::new_current());
            let ownership_clone = Arc::clone(&ownership);

            // Spawn a DIFFERENT task and try the ownership check there.
            let handle = tokio::spawn(async move { ownership_clone.check_current_task() });

            handle.await.unwrap()
        })
        .await;

        let inner_result = result.unwrap();
        assert!(inner_result.is_err());
        let msg = inner_result.unwrap_err();
        assert!(
            msg.contains("Use .spawn()"),
            "error message should guide users: {msg}"
        );
    }

    // Task ownership: owning task succeeds

    #[tokio::test]
    async fn ownership_check_allows_owning_task() {
        // Run inside a spawned task so try_id() returns Some.
        let result = tokio::spawn(async {
            let ownership = TaskOwnership::new_current();
            ownership.check_current_task()
        })
        .await;

        let inner = result.unwrap();
        assert!(inner.is_ok(), "owning task should pass: {inner:?}");
    }

    // Task ownership: .spawn() exemption (blessed tasks)

    #[tokio::test]
    async fn ownership_spawn_exemption_blesses_task() {
        // Create ownership in a spawned task, then spawn a child and bless it.
        let result = tokio::spawn(async {
            let ownership = Arc::new(TaskOwnership::new_current());
            let ownership_clone = Arc::clone(&ownership);

            // Simulate what .spawn() does: launch a task and bless it.
            let handle = tokio::spawn(async move {
                let task_id = tokio::task::id();
                ownership_clone.bless_task(task_id);
                ownership_clone.check_current_task()
            });

            handle.await.unwrap()
        })
        .await;

        let inner = result.unwrap();
        assert!(
            inner.is_ok(),
            "blessed task should pass ownership check: {inner:?}"
        );
    }

    // Task ownership: production topology (inline under block_on)

    /// Reproduces the REAL production topology: handler awaited inline
    /// under `block_on` (NOT inside `tokio::spawn`). `#[tokio::test]`
    /// runs in `block_on`, so `try_id()` is `None`: matching production.
    #[tokio::test]
    async fn ownership_root_context_allows_inline_operations() {
        // Context created inline (try_id() == None): production path.
        let ownership = TaskOwnership::new_current();
        assert!(
            ownership.owner_task_id.is_none(),
            "sanity: #[tokio::test] root runs in block_on with no task ID"
        );
        // Operation invoked inline: should pass.
        let result = ownership.check_current_task();
        assert!(
            result.is_ok(),
            "inline operation in root context should pass: {result:?}"
        );
    }

    /// Production topology: a durable operation invoked from a bare user
    /// `tokio::spawn` (NOT blessed) MUST be rejected.
    #[tokio::test]
    async fn ownership_root_context_rejects_unblessed_spawn() {
        // Context created inline (try_id() == None): production path.
        let ownership = Arc::new(TaskOwnership::new_current());
        assert!(
            ownership.owner_task_id.is_none(),
            "sanity: root runs in block_on with no task ID"
        );
        let ownership_clone = Arc::clone(&ownership);

        // User tokio::spawn, NOT blessed.
        let handle = tokio::spawn(async move { ownership_clone.check_current_task() });

        let inner = handle.await.unwrap();
        assert!(
            inner.is_err(),
            "unblessed spawned task must be rejected in root-context mode"
        );
        let msg = inner.unwrap_err();
        assert!(
            msg.contains("Use .spawn()"),
            "error message should guide users: {msg}"
        );
    }

    /// Production topology: SDK-blessed spawns (`.spawn()`, combinators)
    /// still work in root-context mode.
    #[tokio::test]
    async fn ownership_root_context_allows_blessed_spawn() {
        // Context created inline (try_id() == None): production path.
        let ownership = Arc::new(TaskOwnership::new_current());
        assert!(
            ownership.owner_task_id.is_none(),
            "sanity: root runs in block_on with no task ID"
        );
        let ownership_clone = Arc::clone(&ownership);

        // Simulate what .spawn() / combinators do: spawn and bless.
        let handle = tokio::spawn(async move {
            let task_id = tokio::task::id();
            ownership_clone.bless_task(task_id);
            ownership_clone.check_current_task()
        });

        let inner = handle.await.unwrap();
        assert!(
            inner.is_ok(),
            "blessed task in root-context mode should pass: {inner:?}"
        );
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test module
mod suspension_containment_and_task_ownership {
    use super::InvocationOutcome;
    use super::test_support::drive_ok;
    use crate::client::{ExecutionClient, InMemoryExecutionClient, TestResponse};
    use crate::context::DurableContext;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use aws_sdk_lambda::types::OperationAction;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    fn succeeded_record(wire: &str) -> CheckpointRecord {
        CheckpointRecord {
            id: wire.to_owned(),
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
        }
    }

    // 1. Suspension never lets following user code run
    // Each op is awaited and its (never-produced) result ignored; the side
    // effect after the await MUST NOT run and the invocation MUST be Pending.

    #[tokio::test]
    async fn wait_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                let _ = ctx_h.wait(Duration::from_secs(30)).await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended wait ran"
        );
    }

    #[tokio::test]
    async fn invoke_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                let _ = ctx_h
                    .invoke::<serde_json::Value, _>("target-fn", "input".to_owned())
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended invoke ran"
        );
    }

    #[tokio::test]
    async fn step_retry_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                // Body fails; the default retry strategy schedules a retry,
                // which suspends (parks) on the backend timer.
                let _ = ctx_h
                    .step(|_| async { Err::<i32, crate::BoxError>("boom".into()) })
                    .name("retrying")
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended step retry ran"
        );
    }

    #[tokio::test]
    async fn wait_for_condition_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                let _ = ctx_h
                    .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
                    .wait_strategy_fn(|_state: i32, _attempt| {
                        crate::wait_for_condition::WaitDecision::continue_with(Duration::from_secs(
                            1,
                        ))
                    })
                    .await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended wfc ran"
        );
    }

    #[tokio::test]
    async fn callback_suspend_does_not_run_following_code() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                let Ok(cb) = ctx_h.create_callback::<serde_json::Value>().await else {
                    return Err(crate::error::WireError::new(
                        Some("E".to_owned()),
                        Some("create failed".to_owned()),
                    ));
                };
                // Awaiting a pending callback result suspends.
                let _ = cb.result().await;
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !ran.load(Ordering::SeqCst),
            "code after a suspended callback ran"
        );
    }

    // 2. A replayed completed wait returns Ok and following code runs

    #[tokio::test]
    async fn replayed_completed_wait_returns_ok_and_continues() {
        let wire = crate::engine::compute_wire_id_public("1");
        let log = Arc::new(CheckpointLog::from_records(vec![(
            wire.clone(),
            succeeded_record(&wire),
        )]));
        let ctx = DurableContext::new_root(
            "arn:test".to_owned(),
            lambda_runtime::Context::default(),
            log,
        );
        let signal = Arc::clone(ctx.coordinator());
        let ran = Arc::new(AtomicBool::new(false));
        let ran_c = Arc::clone(&ran);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                ctx_h
                    .wait(Duration::from_secs(30))
                    .await
                    .expect("replayed completed wait returns Ok");
                ran_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            ran.load(Ordering::SeqCst),
            "code after a replayed wait must run"
        );
    }

    // 3. Validation errors remain catchable; checkpoint failures do not

    #[tokio::test]
    async fn validation_error_is_catchable_err() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let signal = Arc::clone(ctx.coordinator());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                // max_concurrency(0) is a validation error, returned as Err.
                let r = ctx_h
                    .parallel(Vec::<crate::Branch<String>>::new())
                    .max_concurrency(0)
                    .await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, crate::error::WireError>("handled".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert_eq!(outcome, InvocationOutcome::Complete("handled".to_owned()));
        assert!(
            caught.load(Ordering::SeqCst),
            "validation error must be catchable"
        );
    }

    /// A checkpoint API failure is never yielded to the handler as a
    /// catchable error (issue #43): the handler is dropped at its await
    /// point exactly as `suspend_now` drops it, so a catch around the
    /// operation cannot observe the failure or branch on it. A
    /// non-retryable failure fails the EXECUTION with the
    /// checkpoint-failure error type.
    #[tokio::test]
    async fn non_retryable_checkpoint_failure_uncatchable_fails_execution() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::NonRetryableError("kaboom".to_owned()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                // The live wait's START checkpoint fails non-retryably.
                // Pre-#43 this surfaced as a catchable Err; now the future
                // parks and the driver drops the handler, so neither branch
                // below runs.
                let r = ctx_h.wait(Duration::from_secs(5)).await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, crate::error::WireError>("handled".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert!(
            !caught.load(Ordering::SeqCst),
            "a checkpoint API failure must not be observable by a catch"
        );
        match outcome {
            InvocationOutcome::Failed { error } => {
                assert_eq!(
                    error.error_type(),
                    Some(crate::error::CHECKPOINT_EXECUTION_FAILED_ERROR_TYPE),
                    "a non-retryable checkpoint failure fails the execution \
                     with the unrecoverable-execution type"
                );
                assert!(
                    error.error_message().is_some_and(|m| m.contains("kaboom")),
                    "the service's rejection reason is preserved: {error:?}"
                );
            }
            other => panic!("expected a FAILED execution outcome, got {other:?}"),
        }
    }

    /// The retryable counterpart: a checkpoint failure that exhausted the
    /// client's internal retries fails the INVOCATION (a Lambda runtime
    /// error, so the service re-invokes: the interruption recovery path)
    /// rather than the execution, and is equally unobservable by a catch.
    #[tokio::test]
    async fn retryable_checkpoint_failure_uncatchable_fails_invocation() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        client.enqueue_checkpoint_response(TestResponse::RetryableError("throttled".to_owned()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let caught = Arc::new(AtomicBool::new(false));
        let caught_c = Arc::clone(&caught);
        let ctx_h = ctx.clone();
        let outcome = drive_ok(
            async move {
                let r = ctx_h.wait(Duration::from_secs(5)).await;
                if r.is_err() {
                    caught_c.store(true, Ordering::SeqCst);
                }
                Ok::<_, crate::error::WireError>("handled".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;
        assert!(
            !caught.load(Ordering::SeqCst),
            "a checkpoint API failure must not be observable by a catch"
        );
        match outcome {
            InvocationOutcome::Fault { message } => {
                assert!(
                    message.contains("throttled"),
                    "the exhausted failure's reason is preserved: {message}"
                );
            }
            other => panic!("expected an invocation Fault outcome, got {other:?}"),
        }
    }

    // 5. Dropping a .spawn() handle cancels its task

    #[tokio::test]
    async fn dropping_spawn_handle_cancels_task() {
        let ctx = live_ctx(Arc::new(InMemoryExecutionClient::new(Vec::new())));
        let started = Arc::new(AtomicBool::new(false));
        let reached_end = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        {
            let s = Arc::clone(&started);
            let e = Arc::clone(&reached_end);
            let d = Arc::clone(&dropped);
            let handle = ctx
                .step(move |_| {
                    let s = Arc::clone(&s);
                    let e = Arc::clone(&e);
                    let d = Arc::clone(&d);
                    async move {
                        struct G(Arc<AtomicBool>);
                        impl Drop for G {
                            fn drop(&mut self) {
                                self.0.store(true, Ordering::SeqCst);
                            }
                        }
                        let _g = G(d);
                        s.store(true, Ordering::SeqCst);
                        std::future::pending::<()>().await;
                        e.store(true, Ordering::SeqCst);
                        Ok(1i32)
                    }
                })
                .name("bg")
                .spawn();
            // Let the spawned task run to its park point.
            for _ in 0..3 {
                tokio::task::yield_now().await;
            }
            assert!(
                started.load(Ordering::SeqCst),
                "spawned task should have started"
            );
            drop(handle);
        }
        // Let the runtime process the abort.
        for _ in 0..5 {
            tokio::task::yield_now().await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "dropped .spawn() handle must cancel the task"
        );
        assert!(
            !reached_end.load(Ordering::SeqCst),
            "cancelled task must not reach its end"
        );
    }

    // 4 + 6. A spawned straggler (non-durable `pending()` body) holds the
    // invocation until the Lambda timeout fires. The timeout drops the
    // `drive_invocation` future, so the driver's cleanup never runs. Dropping
    // it drops the root task's `TrackedTask` handle, which aborts the root
    // task. When tokio unwinds the root task, the handler drops the
    // straggler's `TrackedTask` handle, which aborts the straggler. No
    // SUCCEED checkpoint is made for the aborted straggler. This is Case G
    // from the S2a diagnosis: the bounded timeout simulates the Lambda
    // platform's execution timeout backstop.

    #[tokio::test]
    async fn straggler_is_aborted_by_timeout_no_succeed_checkpoint() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());

        let sib_started = Arc::new(AtomicBool::new(false));
        let sib_completed = Arc::new(AtomicBool::new(false));
        let sib_dropped = Arc::new(AtomicBool::new(false));

        let ctx_h = ctx.clone();
        let s = Arc::clone(&sib_started);
        let c = Arc::clone(&sib_completed);
        let d = Arc::clone(&sib_dropped);

        // Wrap in a timeout to simulate the Lambda execution timeout. The
        // straggler (a step body awaiting a non-durable pending future)
        // holds an activity permit, so the invocation does NOT report
        // Pending on its own. The Lambda timeout is the backstop that aborts
        // the straggler.
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            drive_ok(
                async move {
                    // Sibling task: starts, then blocks on non-durable
                    // pending(). Its body never returns, so it is a
                    // "straggler".
                    let sib = ctx_h
                        .step(move |_| {
                            let s = Arc::clone(&s);
                            let c = Arc::clone(&c);
                            let d = Arc::clone(&d);
                            async move {
                                struct G(Arc<AtomicBool>);
                                impl Drop for G {
                                    fn drop(&mut self) {
                                        self.0.store(true, Ordering::SeqCst);
                                    }
                                }
                                let _g = G(d);
                                s.store(true, Ordering::SeqCst);
                                std::future::pending::<()>().await;
                                c.store(true, Ordering::SeqCst);
                                Ok(1i32)
                            }
                        })
                        .name("sibling")
                        .spawn();
                    // Let the sibling run to its park point.
                    for _ in 0..3 {
                        tokio::task::yield_now().await;
                    }
                    // The owner waits. The straggler still holds a permit, so
                    // the handler stays at this await point until the timeout
                    // fires and drops us.
                    let _ = ctx_h.wait(Duration::from_secs(30)).await;
                    let _ = sib.await;
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await;

        // The timeout fired: the invocation never completed on its own
        // because the straggler held its permit.
        assert!(
            result.is_err(),
            "drive_invocation must NOT terminate on its own while a step body \
             runs; the Lambda timeout is the backstop"
        );

        // Let the runtime process the abort of the sibling task.
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        // Give tokio a moment to run the drop paths.
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(
            sib_started.load(Ordering::SeqCst),
            "sibling should have started"
        );
        assert!(
            !sib_completed.load(Ordering::SeqCst),
            "sibling must not complete (its body is non-durable pending)"
        );
        assert!(
            sib_dropped.load(Ordering::SeqCst),
            "sibling task must be aborted when the timeout drops the handler"
        );
        let recorded = client.recorded_updates();
        assert!(
            recorded
                .iter()
                .all(|u| u.action() != &OperationAction::Succeed),
            "no operation should have checkpointed SUCCEED — the straggler \
             was aborted before reaching its terminal"
        );
    }
}

#[cfg(test)]
mod scoped_suspension {
    use super::InvocationOutcome;
    use super::test_support::drive_ok;
    use crate::client::{ExecutionClient, InMemoryExecutionClient};
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use crate::future::Branch;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    /// Counts `WaitStarted` checkpoints (each carries `wait_options`).
    fn wait_starts(client: &InMemoryExecutionClient) -> usize {
        client
            .recorded_updates()
            .iter()
            .filter(|u| u.wait_options().is_some())
            .count()
    }

    // 1. Branch A waits while branch B has runnable work: B COMPLETES before
    //    the invocation reports Pending.
    #[tokio::test]
    async fn sibling_with_runnable_work_completes_before_invocation_suspends() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let b_ran = Arc::new(AtomicBool::new(false));
        let b_ran_c = Arc::clone(&b_ran);
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            async move {
                let flag = b_ran_c;
                let branches = vec![
                    Branch::new("A-wait", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }),
                    Branch::new("B-step", move |cc: DurableContext| {
                        Box::pin(async move {
                            let v = cc.step(|_| async { Ok(7_i32) }).await?;
                            flag.store(true, Ordering::SeqCst);
                            Ok(v)
                        })
                    }),
                ];
                let _ = ctx_h.parallel(branches).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            b_ran.load(Ordering::SeqCst),
            "sibling B must complete its runnable work before the invocation suspends"
        );
    }

    // 2. Two branches suspend: Pending only after BOTH parked (both reached
    //    WaitStarted).
    #[tokio::test]
    async fn two_branches_suspend_pending_after_both_parked() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            async move {
                let branches = vec![
                    Branch::new("a", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }),
                    Branch::new("b", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }),
                ];
                let _ = ctx_h.parallel(branches).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(
            wait_starts(&client),
            2,
            "both branches must checkpoint WaitStarted before the invocation suspends"
        );
    }

    // 3. A suspended branch retains its slot: with max_concurrency=2 and four
    //    waiting branches, peak concurrent bodies equals the cap (not 4), and
    //    only cap-many branches ever start.
    #[tokio::test]
    async fn suspended_branches_retain_slots_peak_bounded_by_cap() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let ctx_h = ctx.clone();
        let active_h = Arc::clone(&active);
        let peak_h = Arc::clone(&peak);

        let outcome = drive_ok(
            async move {
                let mut branches = Vec::new();
                for i in 0..4 {
                    let active = Arc::clone(&active_h);
                    let peak = Arc::clone(&peak_h);
                    branches.push(Branch::new(format!("w{i}"), move |cc: DurableContext| {
                        Box::pin(async move {
                            let cur = active.fetch_add(1, Ordering::SeqCst) + 1;
                            peak.fetch_max(cur, Ordering::SeqCst);
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(0_i32)
                        })
                    }));
                }
                let _ = ctx_h.parallel(branches).max_concurrency(2).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "peak concurrent branches must equal the cap — suspended branches hold their slots"
        );
        assert!(
            active.load(Ordering::SeqCst) <= 2,
            "no more than cap-many branches may enter while others hold suspended slots"
        );
        assert_eq!(
            wait_starts(&client),
            2,
            "only the two slot-holding branches start; the other two never do"
        );
    }

    // 4. An eligible unstarted branch STARTS once a terminal completion frees
    //    a slot (contrast with test 3, where suspended branches hold slots).
    #[tokio::test]
    async fn slot_freed_by_completion_lets_next_branch_start() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let second_ran = Arc::new(AtomicBool::new(false));
        let second_h = Arc::clone(&second_ran);
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            async move {
                let flag = second_h;
                let branches = vec![
                    Branch::new("first", |cc: DurableContext| async move {
                        let v = cc.step(|_| async { Ok(1_i32) }).await?;
                        Ok(v)
                    }),
                    Branch::new("second", move |cc: DurableContext| {
                        Box::pin(async move {
                            let v = cc.step(|_| async { Ok(2_i32) }).await?;
                            flag.store(true, Ordering::SeqCst);
                            Ok(v)
                        })
                    }),
                ];
                let r = ctx_h.parallel(branches).max_concurrency(1).await;
                Ok::<_, crate::error::WireError>(format!("{}", r.is_ok()))
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Complete("true".to_owned()));
        assert!(
            second_ran.load(Ordering::SeqCst),
            "second branch must start after the first frees its slot"
        );
    }

    // 5 + 8. Nested map/parallel/child suspension propagates through child
    //         scopes to the root and reaches Pending WITHOUT hanging.
    #[tokio::test]
    async fn nested_coordinator_suspension_propagates_to_root_without_hang() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let ctx_h = ctx.clone();

        let fut = drive_ok(
            async move {
                let outer = vec![Branch::new("outer", |cc: DurableContext| {
                    Box::pin(async move {
                        // A child context whose step feeds a nested map whose
                        // single item waits: exercises child + coordinator
                        // scope propagation in one branch.
                        let inner = vec![Branch::new("inner-wait", |icc: DurableContext| {
                            Box::pin(async move {
                                icc.wait(Duration::from_secs(30)).await?;
                                Ok(0_i32)
                            })
                        })];
                        cc.parallel(inner)
                            .await
                            .map(|v: Vec<i32>| v.into_iter().sum::<i32>())
                            .map_err(Into::into)
                    })
                })];
                let _ = ctx_h.parallel(outer).await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        );

        let outcome = tokio::time::timeout(Duration::from_secs(5), fut)
            .await
            .expect("nested suspension must reach Pending, not hang");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(
            wait_starts(&client),
            1,
            "the innermost wait must checkpoint WaitStarted"
        );
    }

    // 6. Early-completion threshold (min_successful) excludes a parked branch:
    //    the batch completes and the invocation does NOT suspend.
    #[tokio::test]
    async fn min_successful_completion_excludes_parked_branch() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            async move {
                let branches = vec![
                    Branch::new("ok", |cc: DurableContext| async move {
                        let v = cc.step(|_| async { Ok(1_i32) }).await?;
                        Ok(v)
                    }),
                    Branch::new("waiter", |cc: DurableContext| {
                        Box::pin(async move {
                            cc.wait(Duration::from_secs(30)).await?;
                            Ok(2_i32)
                        })
                    }),
                ];
                let r = ctx_h
                    .parallel(branches)
                    .completion(
                        crate::builders::map_parallel::CompletionConfig::builder()
                            .min_successful(1)
                            .build()
                            .expect("valid completion config"),
                    )
                    .await;
                Ok::<_, crate::error::WireError>(format!("ok={}", r.is_ok()))
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        // Threshold met by "ok" → batch completes; the parked "waiter" is
        // excluded and the invocation does NOT suspend.
        assert_eq!(outcome, InvocationOutcome::Complete("ok=true".to_owned()));
    }

    // 7. No regression: a ROOT-level wait suspends the invocation immediately
    //    and following code does not run.
    #[tokio::test]
    async fn root_level_wait_suspends_invocation_immediately() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let after = Arc::new(AtomicBool::new(false));
        let after_c = Arc::clone(&after);
        let ctx_h = ctx.clone();

        let outcome = drive_ok(
            async move {
                ctx_h.wait(Duration::from_secs(30)).await.map_err(|e| {
                    crate::error::WireError::new(Some("W".to_owned()), Some(e.to_string()))
                })?;
                after_c.store(true, Ordering::SeqCst);
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            Arc::clone(ctx.task_ownership()),
            signal,
        )
        .await;

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            !after.load(Ordering::SeqCst),
            "code after a root-level wait must not run"
        );
    }
}

//
// `.spawn()` used to let a waiting operation inside a top-level `.spawn()`
// suspend the invocation while a sibling spawned task was still running, which
// aborted the sibling after it had already checkpointed STARTED. The
// invocation coordinator now suspends only when no SDK work can make progress.
//
// These tests encode the CORRECT behaviour and run as part of `make check`.
#[cfg(test)]
mod spawn_scope_regressions {
    use super::InvocationOutcome;
    use super::test_support::drive_ok;
    use crate::client::{ExecutionClient, InMemoryExecutionClient};
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use aws_sdk_lambda::types::OperationAction;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Upper bound on either test: a suspension-scoping defect must show up as
    /// a FAILED assertion, never as a hung test run.
    const BOUND: Duration = Duration::from_secs(5);

    fn live_ctx(client: Arc<InMemoryExecutionClient>) -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            client as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    /// True if the client recorded a terminal (Succeed) checkpoint for the
    /// operation named `name`.
    fn succeeded(client: &InMemoryExecutionClient, name: &str) -> bool {
        client
            .recorded_updates()
            .iter()
            .any(|u| matches!(u.action(), OperationAction::Succeed) && u.name() == Some(name))
    }

    /// `wait.spawn()` beside `step.spawn()`: the spawned wait parks, but the
    /// spawned step is still runnable, so the invocation must stay alive until
    /// the step reaches its terminal checkpoint and only then report Pending.
    #[tokio::test]
    async fn spawned_step_reaches_terminal_checkpoint_before_pending() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let body_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&body_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                async move {
                    // Parks: no timer result is available on a first invocation.
                    let wait = ctx_h.wait(Duration::from_secs(10)).spawn();
                    // Runnable, and slower than the wait's park so the ordering
                    // under test is deterministic.
                    let work = ctx_h
                        .step(move |_| {
                            let done = Arc::clone(&done);
                            async move {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                                done.store(true, Ordering::SeqCst);
                                Ok(7_i32)
                            }
                        })
                        .name("work")
                        .spawn();
                    let _ = work.await;
                    let _ = wait.await;
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect("drive_invocation must terminate within the bound, not hang");

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            body_done.load(Ordering::SeqCst),
            "the spawned step body must run to completion before the invocation \
             suspends; a parked spawned sibling must not abort it"
        );
        assert!(
            succeeded(&client, "work"),
            "the spawned step must reach its terminal Succeed checkpoint before \
             the invocation reports Pending; otherwise its STARTED checkpoint \
             re-executes the body on resume and duplicates side effects"
        );
    }

    /// The documented composition: `tokio::join!` over a spawned wait and a
    /// spawned step. Same guarantee as above, reached through `join!`.
    #[tokio::test]
    async fn joined_spawned_wait_and_step_lets_the_step_finish() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let body_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&body_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                async move {
                    let wait = ctx_h.wait(Duration::from_secs(10)).spawn();
                    let work = ctx_h
                        .step(move |_| {
                            let done = Arc::clone(&done);
                            async move {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                                done.store(true, Ordering::SeqCst);
                                Ok(7_i32)
                            }
                        })
                        .name("work")
                        .spawn();
                    let (_timer, _result) = tokio::join!(wait, work);
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect("drive_invocation must terminate within the bound, not hang");

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            body_done.load(Ordering::SeqCst),
            "join! over a spawned wait and a spawned step must not abort the step"
        );
        assert!(
            succeeded(&client, "work"),
            "the joined spawned step must reach its terminal Succeed checkpoint \
             before the invocation reports Pending"
        );
    }

    /// A spawned wait waits, then the owner performs a non-spawned sequential
    /// step. The sequential step must complete and checkpoint Succeed. The
    /// handler then resolves, so the invocation completes with its result
    /// while the dropped wait's record stays `STARTED`.
    #[tokio::test]
    async fn spawned_wait_parks_then_owner_does_sequential_step() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let step_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&step_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                async move {
                    // Spawn a wait: it parks immediately (no timer result on
                    // first invocation). Hold the handle so dropping it
                    // does not abort the task, but never await it.
                    let _wait = ctx_h.wait(Duration::from_secs(10)).spawn();

                    // Yield once so the spawned wait task has a chance to run
                    // and register its wait. The yield runs under a held
                    // activity permit: a bare yield while an operation waits
                    // is a point where the invocation may suspend.
                    ctx_h.hold_activity(tokio::task::yield_now()).await;

                    // Now do a NON-SPAWNED sequential step. If the spawned
                    // wait already flagged ROOT, the driver will drop us here.
                    let _result = ctx_h
                        .step(move |_| {
                            let done = Arc::clone(&done);
                            async move {
                                done.store(true, Ordering::SeqCst);
                                Ok(42_i32)
                            }
                        })
                        .name("sequential_work")
                        .await;

                    // Handler completes normally: the invocation completes
                    // with its result.
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect("drive_invocation must terminate within the bound, not hang");

        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            step_done.load(Ordering::SeqCst),
            "the non-spawned step must complete before Pending; a parked spawn \
             must not pre-empt the owner's sequential work"
        );
        assert!(
            succeeded(&client, "sequential_work"),
            "the sequential step must reach its terminal Succeed checkpoint \
             before the invocation reports Pending"
        );
    }

    /// A spawned `try_join_all` containing a waiting wait input beside a
    /// runnable step sibling. The combinator is itself spawned. The waiting
    /// input keeps the combinator pending, the runnable sibling spawned
    /// alongside must still complete, and only then does the invocation
    /// suspend, without a deadlock.
    #[tokio::test]
    async fn spawned_combinator_with_parking_input_does_not_deadlock() {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = live_ctx(Arc::clone(&client));
        let signal = Arc::clone(ctx.coordinator());
        let sibling_done = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&sibling_done);
        let ctx_h = ctx.clone();

        let outcome = tokio::time::timeout(
            BOUND,
            drive_ok(
                async move {
                    // Parking input: a spawned wait (will park on first
                    // invocation since no timer has elapsed).
                    let parking_wait = ctx_h.wait(Duration::from_mins(1)).spawn();
                    // The combinator contains the parking input alongside a
                    // step that resolves immediately.
                    let instant_step = ctx_h.step(|_| async { Ok(()) }).name("inner").future();
                    // Spawn the combinator: this registers the combinator on
                    // root scope. The defect made the parking input also park
                    // root, causing a deadlock.
                    let combo = ctx_h.try_join_all([parking_wait, instant_step]).spawn();

                    // Runnable sibling spawned on root beside the combinator.
                    let sibling = ctx_h
                        .step(move |_| {
                            let d = Arc::clone(&done);
                            async move {
                                tokio::time::sleep(Duration::from_millis(100)).await;
                                d.store(true, Ordering::SeqCst);
                                Ok(42_i32)
                            }
                        })
                        .name("sibling")
                        .spawn();

                    let _ = tokio::join!(combo, sibling);
                    Ok::<_, crate::error::WireError>("done".to_owned())
                },
                Arc::clone(ctx.task_ownership()),
                signal,
            ),
        )
        .await
        .expect(
            "drive_invocation must terminate within the bound; \
             a deadlock from the spawned combinator defect would hang here",
        );

        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            sibling_done.load(Ordering::SeqCst),
            "the runnable sibling must complete before the invocation \
             suspends; the combinator's suspension must not abort it"
        );
        assert!(
            succeeded(&client, "sibling"),
            "the sibling must reach its terminal Succeed checkpoint \
             before the invocation reports Pending"
        );
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: the handler panics under test and test assertions
mod task_gateway {
    use super::{InvocationOutcome, drive_invocation};
    use crate::client::{ExecutionClient, InMemoryExecutionClient};
    use crate::context::DurableContext;
    use crate::engine::CheckpointLog;
    use crate::future::DurableFuture;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    fn live_ctx() -> DurableContext {
        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::empty()),
            Arc::new(InMemoryExecutionClient::new(Vec::new())) as Arc<dyn ExecutionClient>,
            "token0".to_owned(),
        )
    }

    /// Sets its flag when dropped.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Spawns a step whose body holds a drop guard and awaits `pending()`.
    fn spawn_holding_step(ctx: &DurableContext, dropped: &Arc<AtomicBool>) -> DurableFuture<()> {
        let dropped = Arc::clone(dropped);
        ctx.step(move |_| {
            let dropped = Arc::clone(&dropped);
            async move {
                let _guard = DropFlag(dropped);
                std::future::pending::<()>().await;
                Ok(())
            }
        })
        .name("hold")
        .spawn()
    }

    async fn yield_three_times() {
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
    }

    /// A spawned step body still mid-await when the handler resolves is
    /// dropped before `drive_invocation` returns, and the invocation
    /// completes with the handler's result.
    #[tokio::test]
    async fn aborted_spawned_body_drops_before_invocation_returns() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let dropped = Arc::new(AtomicBool::new(false));
        let handler_dropped = Arc::clone(&dropped);
        let outcome = drive_invocation(
            async move {
                let _wait = ctx.wait(Duration::from_secs(10)).name("timer").spawn();
                let _step = spawn_holding_step(&ctx, &handler_dropped);
                yield_three_times().await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            ownership,
            signal,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            dropped.load(Ordering::SeqCst),
            "the step body must be dropped before the invocation returns"
        );
    }

    /// A handler panic returns its payload only after every tracked task
    /// has unwound.
    #[tokio::test]
    async fn handler_panic_returns_payload_after_cleanup() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(signal.tasks());
        let dropped = Arc::new(AtomicBool::new(false));
        let handler_dropped = Arc::clone(&dropped);
        let outcome = drive_invocation(
            async move {
                let _step = spawn_holding_step(&ctx, &handler_dropped);
                yield_three_times().await;
                panic!("handler boom");
            },
            ownership,
            signal,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"handler boom"));
        assert!(dropped.load(Ordering::SeqCst), "the step body was dropped");
        assert_eq!(tasks.live(), 0);
    }

    /// After a handler panic, the driver also aborts a task whose handle
    /// the handler moved out of its own future.
    #[tokio::test]
    async fn handler_panic_aborts_task_whose_handle_left_the_handler() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(signal.tasks());
        let dropped = Arc::new(AtomicBool::new(false));
        let handler_dropped = Arc::clone(&dropped);
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let handler_escaped = Arc::clone(&escaped);
        let outcome = drive_invocation(
            async move {
                let step = spawn_holding_step(&ctx, &handler_dropped);
                *handler_escaped.lock().expect("lock") = Some(step);
                yield_three_times().await;
                panic!("handler boom");
            },
            ownership,
            signal,
            None,
        )
        .await;
        assert!(outcome.is_err(), "expected the handler panic");
        assert!(
            escaped.lock().expect("lock").is_some(),
            "the test still holds the handle"
        );
        assert!(dropped.load(Ordering::SeqCst), "the step body was dropped");
        assert_eq!(tasks.live(), 0);
    }

    /// A handler that panics while a `.spawn()`ed wait is registered returns
    /// the panic payload, not `Pending`.
    #[tokio::test]
    async fn handler_panic_with_parked_spawn_returns_payload() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(signal.tasks());
        let parked_signal = Arc::clone(&signal);
        let outcome = drive_invocation(
            async move {
                let _wait = ctx.wait(Duration::from_secs(10)).name("timer").spawn();
                // Held activity keeps the handler runnable while it waits
                // for the registration.
                ctx.hold_activity(async {
                    for _ in 0..1000 {
                        if parked_signal.live_waits() == 1 {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                })
                .await;
                assert_eq!(parked_signal.live_waits(), 1, "the wait is registered");
                panic!("handler boom");
            },
            ownership,
            signal,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"handler boom"));
        assert_eq!(tasks.live(), 0);
    }

    /// A handler that panics while a wait registration is live returns the
    /// panic payload.
    #[tokio::test]
    async fn handler_panic_after_suspend_request_returns_payload() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(signal.tasks());
        let handler_signal = Arc::clone(&signal);
        let outcome = drive_invocation(
            async move {
                let _ctx = ctx;
                let _registration = handler_signal.register_wait("op", std::task::Waker::noop());
                panic!("handler boom");
            },
            ownership,
            signal,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"handler boom"));
        assert_eq!(tasks.live(), 0);
    }

    /// A recorded fatal error takes precedence over a handler panic.
    #[tokio::test]
    async fn fatal_error_wins_over_handler_panic() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let handler_signal = Arc::clone(&signal);
        let outcome = drive_invocation(
            async move {
                let _ctx = ctx;
                handler_signal.record_fatal("Fatal".to_owned(), "mismatch".to_owned());
                panic!("handler boom");
            },
            ownership,
            signal,
            None,
        )
        .await
        .expect("the fatal error takes precedence over the panic");
        assert_eq!(
            outcome,
            InvocationOutcome::Failed {
                error: crate::error::WireError::new(
                    Some("Fatal".to_owned()),
                    Some("mismatch".to_owned()),
                ),
            }
        );
    }

    /// Records a fatal error on the invocation's coordinator when it is
    /// dropped.
    struct FatalOnDrop(Arc<crate::coordinator::InvocationCoordinator>);

    impl Drop for FatalOnDrop {
        fn drop(&mut self) {
            self.0
                .record_fatal("Fatal".to_owned(), "mismatch during cleanup".to_owned());
        }
    }

    /// A fatal error that a tracked task records while the driver cleans up
    /// after a handler panic still takes precedence over that panic. The
    /// handler moves the step's handle out of its own future, so only the
    /// driver's `abort_all` drops the step body, after the drive observed
    /// the panic. Dropping the body records the fatal error.
    #[tokio::test(flavor = "current_thread")]
    async fn fatal_recorded_during_panic_cleanup_wins_over_panic() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(signal.tasks());
        let body_signal = Arc::clone(&signal);
        let started = Arc::new(AtomicBool::new(false));
        let body_started = Arc::clone(&started);
        let escaped: Arc<Mutex<Option<DurableFuture<()>>>> = Arc::new(Mutex::new(None));
        let handler_escaped = Arc::clone(&escaped);
        let check_signal = Arc::clone(&signal);
        let outcome = drive_invocation(
            async move {
                let step = ctx
                    .step(move |_| {
                        let guard = FatalOnDrop(Arc::clone(&body_signal));
                        let started = Arc::clone(&body_started);
                        async move {
                            let _guard = guard;
                            started.store(true, Ordering::SeqCst);
                            std::future::pending::<()>().await;
                            Ok(())
                        }
                    })
                    .name("hold")
                    .spawn();
                *handler_escaped.lock().expect("lock") = Some(step);
                yield_three_times().await;
                assert!(check_signal.fatal_error().is_none(), "no fatal yet");
                panic!("handler boom");
            },
            ownership,
            signal,
            None,
        )
        .await
        .expect("the fatal error recorded during cleanup takes precedence over the panic");
        assert!(started.load(Ordering::SeqCst), "the step body started");
        assert_eq!(
            outcome,
            InvocationOutcome::Failed {
                error: crate::error::WireError::new(
                    Some("Fatal".to_owned()),
                    Some("mismatch during cleanup".to_owned()),
                ),
            }
        );
        assert_eq!(tasks.live(), 0);
    }

    /// A handler that returns a value while a wait registration is live
    /// completes with that value: the resolved handler decides the outcome.
    #[tokio::test]
    async fn handler_value_with_live_registration_completes() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let handler_signal = Arc::clone(&signal);
        let outcome = drive_invocation(
            async move {
                let _ctx = ctx;
                let _registration = handler_signal.register_wait("op", std::task::Waker::noop());
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            ownership,
            signal,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    /// The handler runs on a blessed task of its own, and its blessing is
    /// removed once the drive ends.
    #[tokio::test]
    async fn handler_runs_as_blessed_operation_task() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let outcome = drive_invocation(
            async move {
                let on_task = tokio::task::try_id().is_some();
                let value = ctx
                    .step(|_| async { Ok(1_i32) })
                    .name("s")
                    .await
                    .map_err(|e| {
                        crate::error::WireError::new(None::<String>, Some(e.to_string()))
                    })?;
                Ok::<_, crate::error::WireError>(format!("{on_task}:{value}"))
            },
            Arc::clone(&ownership),
            signal,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("true:1".to_owned()));
        assert_eq!(ownership.blessed_count(), 0);
    }

    /// The root task is not counted on the tracker, so a handler that waits
    /// for idle does not wait for itself.
    #[tokio::test]
    async fn wait_idle_in_handler_does_not_wait_for_handler() {
        let ctx = live_ctx();
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let tasks = Arc::clone(signal.tasks());
        let outcome = drive_invocation(
            async move {
                let step = ctx.step(|_| async { Ok(1_i32) }).name("quick").spawn();
                let _ = step.await;
                let idle = tokio::time::timeout(Duration::from_secs(1), tasks.wait_idle()).await;
                Ok::<_, crate::error::WireError>(idle.is_ok().to_string())
            },
            ownership,
            signal,
            None,
        )
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("true".to_owned()));
    }
}

#[cfg(test)]
#[expect(clippy::panic)] // reason: test assertions
mod coordinator_suspension {
    use super::test_support::{
        EventMessages, ScriptedClient, drive_handler, named, operation_like, scripted_ctx, wire_err,
    };
    use super::{InvocationOutcome, UNTRACKED_DEADLINE_WARNING};
    use crate::coordinator::{PermitKind, Phase};
    use crate::engine::CheckpointStatus;
    use aws_sdk_lambda::types::{OperationAction, OperationStatus};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    /// Waits, inside the handler's held activity, until `condition` holds,
    /// polling every 10 ms of tokio time, at most 1,000 times.
    async fn until(ctx: &crate::DurableContext, condition: impl Fn() -> bool) -> bool {
        ctx.hold_activity(async {
            for _ in 0..1000 {
                if condition() {
                    return true;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            false
        })
        .await
    }

    /// A client that, on the call carrying the step named `step`'s SUCCEED,
    /// also carries the operation named `waiting` with `status`.
    fn completes_on_step_succeed(
        step: &'static str,
        waiting: &'static str,
        status: OperationStatus,
    ) -> ScriptedClient {
        ScriptedClient::new(move |updates, history| {
            if named(updates, step, &OperationAction::Succeed).is_none() {
                return Vec::new();
            }
            named(history, waiting, &OperationAction::Start)
                .map(|start| {
                    operation_like(start, status.clone())
                        .build()
                        .expect("all required fields set")
                })
                .into_iter()
                .collect()
        })
    }

    /// A step body that returns once the client received the START of the
    /// operation named `waiting`, so the response that completes it comes
    /// after its registration.
    async fn after_start_of(
        client: &Arc<ScriptedClient>,
        waiting: &str,
    ) -> Result<u32, crate::BoxError> {
        for _ in 0..1000 {
            if client.received(waiting, &OperationAction::Start) {
                return Ok(2);
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        Err("the waiting operation never started".into())
    }

    #[tokio::test(start_paused = true)]
    async fn lone_wait_suspends_at_once() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            ctx.wait(Duration::from_secs(1)).await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_eq!(client.call_count(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn untracked_await_does_not_suspend_and_warns() {
        let (capture, _guard) = EventMessages::install();
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let drive = drive_handler(ctx, Some(Duration::from_secs(2)), |_ctx| async move {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok("done".to_owned())
        });
        let timed = tokio::time::timeout(Duration::from_secs(5), drive).await;
        assert!(timed.is_err(), "the untracked await does not suspend");
        assert_eq!(
            capture.at(tracing::Level::WARN),
            vec![UNTRACKED_DEADLINE_WARNING.to_owned()]
        );
    }

    /// A deadline that passes while SDK work runs logs nothing.
    #[tokio::test(start_paused = true)]
    async fn deadline_during_tracked_work_does_not_warn() {
        let (capture, _guard) = EventMessages::install();
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let outcome = drive_handler(ctx, Some(Duration::from_secs(1)), |ctx| async move {
            ctx.step(|_| async {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Ok(1_u32)
            })
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(capture.at(tracing::Level::WARN), Vec::<String>::new());
    }

    #[tokio::test(start_paused = true)]
    async fn sibling_response_completes_wait_in_invocation() {
        let (events, _guard) = EventMessages::install();
        let client = Arc::new(completes_on_step_succeed(
            "s",
            "w",
            OperationStatus::Succeeded,
        ));
        let ctx = scripted_ctx(&client);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            ctx.step(move |_| async move { after_start_of(&body_client, "w").await })
                .name("s")
                .await
                .map_err(wire_err)?;
            w.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(events.replayed(), 0, "a resumed wait emits no replay event");
    }

    #[tokio::test(start_paused = true)]
    async fn buffered_checkpoint_holds_permit_until_merged() {
        let client = Arc::new(completes_on_step_succeed(
            "s",
            "w",
            OperationStatus::Succeeded,
        ));
        let ctx = crate::DurableContext::new_root_with_client_and_coalescer(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(crate::engine::CheckpointLog::empty()),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::from_millis(50)),
        );
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            ctx.step(move |_| async move { after_start_of(&body_client, "w").await })
                .name("s")
                .await
                .map_err(wire_err)?;
            w.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
    }

    /// A root context on `client` whose writes wait out a coalescing window
    /// of one hour, with `log` as the checkpoint log.
    fn hour_window_ctx(
        client: &Arc<ScriptedClient>,
        log: crate::engine::CheckpointLog,
    ) -> crate::DurableContext {
        crate::DurableContext::new_root_with_client_and_coalescer(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            Arc::clone(client) as Arc<dyn crate::client::ExecutionClient>,
            "t0".to_owned(),
            crate::checkpoint_coalescer::CheckpointCoalescer::new(Duration::from_secs(3600)),
        )
    }

    #[tokio::test(start_paused = true)]
    async fn completion_preempts_window_of_dropped_contributor() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = hour_window_ctx(&client, crate::engine::CheckpointLog::empty());
        let coordinator = Arc::clone(ctx.coordinator());
        let started = tokio::time::Instant::now();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(10)).name("w").spawn();
            let buffered = ctx
                .hold_activity(async {
                    for _ in 0..1000 {
                        if ctx.coordinator().permits(PermitKind::Checkpoint) == 1 {
                            return true;
                        }
                        tokio::task::yield_now().await;
                    }
                    false
                })
                .await;
            if !buffered {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("the START of w was never buffered"),
                ));
            }
            drop(w);
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "completion waited {:?} of the window",
            started.elapsed()
        );
        assert!(
            client.received("w", &OperationAction::Start),
            "the buffered START is flushed"
        );
        assert_eq!(coordinator.tasks().live(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn waiting_operation_preempts_window() {
        let client = Arc::new(ScriptedClient::empty());
        let wait_id = crate::engine::compute_wire_id_public("1");
        let log = crate::engine::CheckpointLog::from_records(vec![(
            wait_id.clone(),
            crate::engine::CheckpointRecord::with_status(&wait_id, CheckpointStatus::Started),
        )]);
        let ctx = hour_window_ctx(&client, log);
        let started = tokio::time::Instant::now();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(10)).name("w").spawn();
            ctx.step(|_| async { Ok(1_u32) })
                .name("s")
                .await
                .map_err(wire_err)?;
            w.await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the step's writes waited {:?} of the window",
            started.elapsed()
        );
        assert!(client.received("s", &OperationAction::Succeed));
    }

    #[tokio::test(start_paused = true)]
    async fn wait_resumed_maps_terminal_status() {
        let client = Arc::new(completes_on_step_succeed("s", "w", OperationStatus::Failed));
        let ctx = scripted_ctx(&client);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(1)).name("w").spawn();
            ctx.step(move |_| async move { after_start_of(&body_client, "w").await })
                .name("s")
                .await
                .map_err(wire_err)?;
            let err = w.await.expect_err("the wait record failed");
            let crate::error::OperationErrorKind::Wait(wait_err) = err.kind() else {
                panic!("expected a wait error, got {err:?}");
            };
            let crate::error::WaitErrorKind::UnexpectedStatus(status) = wait_err.kind();
            Ok(status.status().to_owned())
        })
        .await
        .expect("the handler does not panic");
        assert_eq!(outcome, InvocationOutcome::Complete("FAILED".to_owned()));
    }

    #[tokio::test(start_paused = true)]
    async fn resolved_handler_completes_with_dropped_spawned_wait() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let handler_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let w = ctx.wait(Duration::from_secs(10)).name("w").spawn();
            let registered = ctx
                .hold_activity(async {
                    for _ in 0..1000 {
                        if handler_client.received("w", &OperationAction::Start)
                            && ctx.coordinator().live_waits() == 1
                        {
                            return true;
                        }
                        tokio::task::yield_now().await;
                    }
                    false
                })
                .await;
            if !registered {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("the wait never registered"),
                ));
            }
            drop(w);
            Ok(serde_json::to_string("done").expect("a string serializes"))
        })
        .await;
        assert!(
            matches!(&outcome, Ok(InvocationOutcome::Complete(v)) if v == "\"done\""),
            "unexpected outcome: {outcome:?}"
        );
        let w_updates: Vec<_> = client
            .updates()
            .into_iter()
            .filter(|u| u.name() == Some("w"))
            .collect();
        assert_eq!(w_updates.len(), 1, "only the START of w was written");
        assert_eq!(
            w_updates
                .first()
                .map(aws_sdk_lambda::types::OperationUpdate::action),
            Some(&OperationAction::Start)
        );
        assert_eq!(coordinator.tasks().live(), 0);
        assert_eq!(coordinator.live_waits(), 0);
    }

    /// The coordinator's counts after a drive: no task, permit or wait
    /// registration remains.
    fn assert_clean(coordinator: &crate::coordinator::InvocationCoordinator) {
        assert_eq!(coordinator.permits(PermitKind::Activity), 0);
        assert_eq!(coordinator.permits(PermitKind::Checkpoint), 0);
        assert_eq!(coordinator.live_waits(), 0);
        assert_eq!(coordinator.tasks().live(), 0);
    }

    /// Spawns a wait of an hour and a step whose body never returns.
    fn spawn_wait_and_stuck_step(
        ctx: &crate::DurableContext,
    ) -> (crate::DurableFuture<()>, crate::DurableFuture<()>) {
        let wait = ctx.wait(Duration::from_secs(3600)).name("w").spawn();
        let step = ctx
            .step(|_| async {
                std::future::pending::<()>().await;
                Ok(())
            })
            .name("stuck")
            .spawn();
        (wait, step)
    }

    #[tokio::test(start_paused = true)]
    async fn invocation_end_leaves_no_task_permit_or_registration() {
        // (a) The handler returns Ok.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _ops = spawn_wait_and_stuck_step(&ctx);
            ctx.hold_activity(tokio::time::sleep(Duration::from_millis(10)))
                .await;
            Ok("a".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("a".to_owned()));
        assert_clean(&coordinator);

        // (b) The handler awaits the wait alone after dropping the step.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let (wait, step) = spawn_wait_and_stuck_step(&ctx);
            drop(step);
            wait.await.map_err(wire_err)?;
            Ok("b".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert_clean(&coordinator);

        // (c) The handler returns Err.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _ops = spawn_wait_and_stuck_step(&ctx);
            ctx.hold_activity(tokio::time::sleep(Duration::from_millis(10)))
                .await;
            Err(crate::error::WireError::new(Some("E"), Some("c")))
        })
        .await
        .expect("no panic");
        assert!(matches!(outcome, InvocationOutcome::Failed { .. }));
        assert_clean(&coordinator);

        // (d) The handler panics.
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _ops = spawn_wait_and_stuck_step(&ctx);
            ctx.hold_activity(tokio::time::sleep(Duration::from_millis(10)))
                .await;
            panic!("d");
        })
        .await;
        let Err(payload) = outcome else {
            panic!("expected the handler panic, got {outcome:?}");
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"d"));
        assert_clean(&coordinator);
    }

    #[tokio::test(start_paused = true)]
    async fn unawaited_finished_spawn_does_not_block_suspension() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let start = tokio::time::Instant::now();
        let handler_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, Some(Duration::from_mins(15)), |ctx| async move {
            let _h = ctx.step(|_| async { Ok(1_u32) }).name("h").spawn();
            let tasks = Arc::clone(ctx.coordinator().tasks());
            let exited = until(&ctx, || {
                handler_client.received("h", &OperationAction::Succeed) && tasks.live() == 0
            })
            .await;
            if !exited {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("h never ended"),
                ));
            }
            ctx.wait(Duration::from_secs(3600))
                .await
                .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "the invocation suspended at once, not at the deadline"
        );
    }

    #[test]
    fn exhausted_budget_does_not_suspend_runnable_handler() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let client = Arc::new(ScriptedClient::empty());
            let ctx = scripted_ctx(&client);
            let outcome = drive_handler(ctx, None, |ctx| async move {
                let _wait = ctx.wait(Duration::from_secs(3600)).name("w").spawn();
                let coordinator = Arc::clone(ctx.coordinator());
                let registered = until(&ctx, || coordinator.live_waits() == 1).await;
                if !registered {
                    return Err(crate::error::WireError::new(
                        None::<String>,
                        Some("no wait"),
                    ));
                }
                let sibling = ctx.step(|_| async { Ok(1_u32) }).name("sibling").spawn();
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<u32>();
                for value in 0..20_000 {
                    tx.send(value).expect("the receiver is alive");
                }
                drop(tx);
                let mut received = 0_u32;
                while rx.recv().await.is_some() {
                    received += 1;
                }
                sibling.await.map_err(wire_err)?;
                ctx.step(|_| async { Ok(2_u32) })
                    .name("after")
                    .await
                    .map_err(wire_err)?;
                Ok(received.to_string())
            })
            .await
            .expect("no panic");
            assert_eq!(outcome, InvocationOutcome::Complete("20000".to_owned()));
            assert!(client.received("after", &OperationAction::Succeed));
        });
    }

    #[tokio::test(start_paused = true)]
    async fn detached_write_in_flight_at_finish_completes() {
        let client = Arc::new(
            ScriptedClient::new(|updates, _| {
                named(updates, "quick", &OperationAction::Succeed)
                    .map(|succeed| {
                        operation_like(succeed, OperationStatus::Succeeded)
                            .build()
                            .expect("all required fields set")
                    })
                    .into_iter()
                    .collect()
            })
            .with_delay(Duration::from_millis(500)),
        );
        let ctx = scripted_ctx(&client);
        let coordinator = Arc::clone(ctx.coordinator());
        let log_ctx = ctx.clone();
        let handler_client = Arc::clone(&client);
        let start = tokio::time::Instant::now();
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let h = ctx.step(|_| async { Ok(1_u32) }).name("quick").spawn();
            let sent = until(&ctx, || {
                handler_client.received("quick", &OperationAction::Succeed)
            })
            .await;
            if !sent {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("no SUCCEED"),
                ));
            }
            drop(h);
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert!(start.elapsed() >= Duration::from_millis(500));
        assert_eq!(
            log_ctx.checkpoint_status_view("1").map(|v| v.status),
            Some(CheckpointStatus::Succeeded),
            "the in-flight write's response was merged"
        );
        assert_eq!(coordinator.tasks().live(), 0);
    }

    /// Runs the scenario of `root_yield_now_is_a_boundary_on_every_runtime`
    /// once and returns its outcome and whether `after` ran.
    async fn yield_with_live_wait() -> (InvocationOutcome, bool) {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let after_ran = Arc::new(AtomicBool::new(false));
        let handler_ran = Arc::clone(&after_ran);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let _wait = ctx.wait(Duration::from_secs(3600)).name("w").spawn();
            let coordinator = Arc::clone(ctx.coordinator());
            let ready = ctx
                .hold_activity(async {
                    for _ in 0..10_000 {
                        if coordinator.live_waits() == 1
                            && coordinator.permits(PermitKind::Activity) == 2
                            && coordinator.permits(PermitKind::Checkpoint) == 0
                        {
                            return true;
                        }
                        tokio::task::yield_now().await;
                    }
                    false
                })
                .await;
            if !ready {
                return Err(crate::error::WireError::new(
                    None::<String>,
                    Some("not ready"),
                ));
            }
            tokio::task::yield_now().await;
            ctx.step(move |_| async move {
                handler_ran.store(true, Ordering::SeqCst);
                Ok(1_u32)
            })
            .name("after")
            .await
            .map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        (outcome, after_ran.load(Ordering::SeqCst))
    }

    #[test]
    fn root_yield_now_is_a_boundary_on_every_runtime() {
        let current = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let multi = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("runtime");
        for _ in 0..100 {
            let (outcome, after_ran) = current.block_on(yield_with_live_wait());
            assert_eq!(
                outcome,
                InvocationOutcome::Pending,
                "current-thread runtime"
            );
            assert!(!after_ran, "current-thread runtime");

            let (outcome, after_ran) = multi.block_on(yield_with_live_wait());
            assert_eq!(outcome, InvocationOutcome::Pending, "multi-thread block_on");
            assert!(!after_ran, "multi-thread block_on");

            let (outcome, after_ran) = multi
                .block_on(async { spawned(yield_with_live_wait()).await })
                .expect("the spawned scenario does not panic");
            assert_eq!(outcome, InvocationOutcome::Pending, "multi-thread task");
            assert!(!after_ran, "multi-thread task");
        }
    }

    /// Runs `future` on its own tokio task.
    #[expect(clippy::disallowed_methods)]
    // reason: the scenario must run on a task of the multi-thread runtime,
    // outside the SDK's gateway.
    fn spawned<F>(future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        tokio::spawn(future)
    }

    #[tokio::test(start_paused = true)]
    async fn phase_is_finished_after_every_drive() {
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let coordinator = Arc::clone(ctx.coordinator());
        let outcome = drive_handler(ctx, None, |_ctx| async move { Ok("done".to_owned()) })
            .await
            .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("done".to_owned()));
        assert_eq!(coordinator.phase(), Phase::Finished);
    }
}
