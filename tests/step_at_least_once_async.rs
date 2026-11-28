//! `StepSemantics::AtLeastOnceAsync` through the whole invocation.
//!
//! The step returns before its `START` and `SUCCEED` are answered. These
//! tests drive handlers through [`LocalRunner`] with a slow checkpoint
//! client and check that the invocation still saves the result before it
//! reports the execution's outcome, in checkpoint order, and that a failed
//! `START` or `SUCCEED` write does not report the execution as succeeded.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::lambda_types::EventType;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::test_util::{LocalRunner, TestResult};
use durable::{BoxError, StepSemantics};

/// The history event types of `result`, in order.
fn event_types<O>(result: &TestResult<O>) -> Vec<EventType> {
    result
        .history_events()
        .iter()
        .filter_map(|event| event.event_type().cloned())
        .collect()
}

/// The body delay after which the `START` of a step is answered under a
/// 50 ms checkpoint latency, so its `SUCCEED` goes in a later request.
const AFTER_START: Duration = Duration::from_millis(100);

/// Runs a handler with one async step named `s` that counts its body runs,
/// sleeps for `body_delay` and returns `"one"`, on `runner`. Returns the
/// result and the body runs.
async fn run_async_step(runner: LocalRunner, body_delay: Duration) -> (TestResult<String>, u32) {
    let runs = Arc::new(AtomicU32::new(0));
    let body_runs = Arc::clone(&runs);
    let result = runner
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs);
                async move {
                    let value = ctx
                        .step(move |_| {
                            let body_runs = Arc::clone(&body_runs);
                            async move {
                                body_runs.fetch_add(1, Ordering::SeqCst);
                                tokio::time::sleep(body_delay).await;
                                Ok("one".to_owned())
                            }
                        })
                        .name("s")
                        .semantics(StepSemantics::AtLeastOnceAsync)
                        .await?;
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    (result, runs.load(Ordering::SeqCst))
}

#[tokio::test]
async fn succeed_is_saved_before_the_execution_succeeds() {
    let (result, runs) = run_async_step(
        LocalRunner::new().checkpoint_latency(Duration::from_millis(50)),
        Duration::ZERO,
    )
    .await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(runs, 1);
    assert_eq!(result.invocation_count(), 1);
    let step = result.operation("s").expect("step s");
    assert!(step.succeeded(), "the step's result is saved");
    assert_eq!(step.result(), Some(r#""one""#));
    assert_eq!(
        event_types(&result),
        [
            EventType::ExecutionStarted,
            EventType::StepStarted,
            EventType::StepSucceeded,
            EventType::InvocationCompleted,
            EventType::ExecutionSucceeded,
        ]
    );
}

/// The same under a coalescing window: the window does not hold the
/// `SUCCEED` past the end of the invocation.
#[tokio::test]
async fn buffered_succeed_is_saved_before_the_execution_succeeds() {
    let options = durable::Options::builder()
        .checkpoint_delay(Duration::from_millis(200))
        .build()
        .expect("valid options");
    let (result, runs) = run_async_step(LocalRunner::new().options(options), Duration::ZERO).await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(runs, 1);
    assert_eq!(result.invocation_count(), 1);
    assert!(result.operation("s").expect("step s").succeeded());
}

#[tokio::test]
async fn child_step_succeed_precedes_the_context_succeed() {
    let result = LocalRunner::new()
        .checkpoint_latency(Duration::from_millis(50))
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                let value = ctx
                    .run_in_child_context(|child| async move {
                        let value = child
                            .step(|_| async { Ok(1_u32) })
                            .name("s")
                            .semantics(StepSemantics::AtLeastOnceAsync)
                            .await?;
                        Ok(value)
                    })
                    .name("child")
                    .await?;
                Ok::<_, BoxError>(value)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{result:?}");
    assert_eq!(
        event_types(&result),
        [
            EventType::ExecutionStarted,
            EventType::ContextStarted,
            EventType::StepStarted,
            EventType::StepSucceeded,
            EventType::ContextSucceeded,
            EventType::InvocationCompleted,
            EventType::ExecutionSucceeded,
        ]
    );
}

/// Records the info of every step's `on_operation_end`, and the order of
/// those hooks and `on_invocation_end` with the tokio time of each.
#[derive(Clone, Default)]
struct StepEnds {
    ends: Arc<Mutex<Vec<OperationInfo>>>,
    order: Arc<Mutex<Vec<(&'static str, tokio::time::Instant)>>>,
}

impl StepEnds {
    /// The hook names in call order: `step_end` and `invocation_end`.
    fn order(&self) -> Vec<&'static str> {
        self.order
            .lock()
            .expect("lock")
            .iter()
            .map(|(hook, _)| *hook)
            .collect()
    }
}

impl Plugin for StepEnds {
    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if info.kind() == Some(durable::OperationKind::Step) {
            self.ends.lock().expect("lock").push(info.clone());
            self.order
                .lock()
                .expect("lock")
                .push(("step_end", tokio::time::Instant::now()));
        }
        Ok(None)
    }

    fn on_invocation_end<'a>(
        &'a self,
        _info: &'a durable::plugin::InvocationEndInfo,
    ) -> durable::plugin::HookFuture<'a> {
        self.order
            .lock()
            .expect("lock")
            .push(("invocation_end", tokio::time::Instant::now()));
        Box::pin(std::future::ready(Ok(())))
    }
}

impl PluginFactory for StepEnds {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// Options with the plugin `ends`.
fn with_plugin(ends: &StepEnds) -> durable::Options {
    durable::Options::builder()
        .plugin(ends.clone())
        .build()
        .expect("valid options")
}

/// The end hook fires once, when the service acknowledged the `SUCCEED`,
/// before `on_invocation_end`. It reports `SUCCEEDED`, the result and both
/// timestamps.
#[tokio::test]
async fn end_hook_reports_succeeded() {
    let ends = StepEnds::default();
    let (result, _) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .options(with_plugin(&ends)),
        Duration::ZERO,
    )
    .await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(ends.order(), ["step_end", "invocation_end"]);
    let ends = ends.ends.lock().expect("lock");
    let [end] = ends.as_slice() else {
        unreachable!("expected one step end, got {ends:?}");
    };
    assert_eq!(end.status(), Some(durable::OperationStatus::Succeeded));
    assert_eq!(end.result(), Some(r#""one""#));
    assert!(end.start_timestamp().is_some(), "a start time");
    assert!(end.end_timestamp().is_some(), "an end time");
}

/// The step returns before its `SUCCEED` is answered, and the end hook
/// fires only after the answer: one checkpoint latency after the step
/// returned, or later.
#[tokio::test(start_paused = true)]
async fn end_hook_waits_for_the_succeed_answer() {
    let latency = Duration::from_millis(300);
    let ends = StepEnds::default();
    let returned = Arc::new(Mutex::new(None));
    let step_returned = Arc::clone(&returned);
    let result = LocalRunner::new()
        .checkpoint_latency(latency)
        .options(with_plugin(&ends))
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let step_returned = Arc::clone(&step_returned);
                async move {
                    let value = ctx
                        .step(|_| async { Ok(1_u32) })
                        .name("s")
                        .semantics(StepSemantics::AtLeastOnceAsync)
                        .await?;
                    *step_returned.lock().expect("lock") = Some(tokio::time::Instant::now());
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{result:?}");
    let returned = returned.lock().expect("lock").expect("the step returned");
    let order = ends.order.lock().expect("lock").clone();
    let [("step_end", ended), ("invocation_end", _)] = order.as_slice() else {
        unreachable!("expected a step end, then the invocation end, got {order:?}");
    };
    assert!(
        *ended >= returned + latency,
        "the end hook fired {:?} after the step returned",
        ended.duration_since(returned)
    );
}

/// A `SUCCEED` the service rejected fires no end hook. A permanent
/// rejection ends the execution. A retryable one fails the invocation,
/// and the next invocation's `SUCCEED` fires the only end hook.
#[tokio::test]
async fn failed_succeed_fires_no_end_hook() {
    let ends = StepEnds::default();
    let (result, _) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .fail_checkpoints_after(1, 1)
            .options(with_plugin(&ends)),
        AFTER_START,
    )
    .await;
    assert!(result.is_failure(), "{result:?}");
    assert_eq!(ends.order(), ["invocation_end"]);

    let ends = StepEnds::default();
    let (result, _) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .fail_checkpoints_after_retryable(1, 1)
            .options(with_plugin(&ends)),
        AFTER_START,
    )
    .await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(
        ends.order(),
        ["invocation_end", "step_end", "invocation_end"]
    );
    let ends = ends.ends.lock().expect("lock");
    assert_eq!(ends.len(), 1);
    assert_eq!(
        ends.first().and_then(OperationInfo::status),
        Some(durable::OperationStatus::Succeeded)
    );
}

/// The service permanently rejects the `SUCCEED`: the execution fails, and
/// it is not reported as succeeded.
#[tokio::test]
async fn rejected_succeed_fails_the_execution() {
    let (result, runs) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .fail_checkpoints_after(1, 1),
        AFTER_START,
    )
    .await;
    assert!(result.is_failure(), "{result:?}");
    assert_eq!(
        result.error_type(),
        Some("CheckpointUnrecoverableExecutionError")
    );
    assert_eq!(runs, 1);
    assert!(!event_types(&result).contains(&EventType::ExecutionSucceeded));
}

/// A retryable `SUCCEED` failure fails the invocation. The next invocation
/// finds the step `STARTED` and runs its body again.
#[tokio::test]
async fn retryable_succeed_failure_reruns_the_step() {
    let (result, runs) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .fail_checkpoints_after_retryable(1, 1),
        AFTER_START,
    )
    .await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(runs, 2, "the body runs again in the next invocation");
    assert_eq!(result.invocation_count(), 2);
    assert!(result.operation("s").expect("step s").succeeded());
}

/// The service permanently rejects the `START`, which the step did not
/// await. With `body_delay` zero the `SUCCEED` shares the request; with
/// a short delay the `SUCCEED` waits behind it and is never sent. The
/// execution fails, and it is not reported as succeeded.
async fn rejected_start_fails_the_execution(body_delay: Duration) {
    let (result, runs) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .fail_checkpoints_after(0, 1),
        body_delay,
    )
    .await;
    assert!(result.is_failure(), "{result:?}");
    assert_eq!(
        result.error_type(),
        Some("CheckpointUnrecoverableExecutionError")
    );
    assert_eq!(runs, 1);
    assert_eq!(result.invocation_count(), 1);
    assert!(result.operation("s").is_none(), "no record is saved");
    assert!(!event_types(&result).contains(&EventType::ExecutionSucceeded));
}

/// A `START` the service rejected while the `SUCCEED` waited behind it
/// fires no end hook, and the invocation still ends: the end hook task of
/// the `SUCCEED` that was never sent stops.
#[tokio::test]
async fn rejected_start_fires_no_end_hook() {
    for body_delay in [Duration::ZERO, Duration::from_millis(10)] {
        let ends = StepEnds::default();
        let (result, _) = tokio::time::timeout(
            Duration::from_secs(10),
            run_async_step(
                LocalRunner::new()
                    .checkpoint_latency(Duration::from_millis(50))
                    .fail_checkpoints_after(0, 1)
                    .options(with_plugin(&ends)),
                body_delay,
            ),
        )
        .await
        .expect("the invocation ends");
        assert!(result.is_failure(), "{result:?}");
        assert_eq!(ends.order(), ["invocation_end"], "{body_delay:?}");
    }
}

#[tokio::test]
async fn rejected_start_in_the_succeed_request_fails_the_execution() {
    rejected_start_fails_the_execution(Duration::ZERO).await;
}

#[tokio::test]
async fn rejected_start_before_the_succeed_fails_the_execution() {
    rejected_start_fails_the_execution(Duration::from_millis(10)).await;
}

/// A retryable `START` failure fails the invocation before the `SUCCEED`
/// is saved. The next invocation finds no record and runs the step again.
async fn retryable_start_failure_reruns_the_step(body_delay: Duration) {
    let (result, runs) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .fail_checkpoints_after_retryable(0, 1),
        body_delay,
    )
    .await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(runs, 2, "the body runs again in the next invocation");
    assert_eq!(result.invocation_count(), 2);
    assert!(result.operation("s").expect("step s").succeeded());
    assert_eq!(
        event_types(&result)
            .into_iter()
            .filter(|t| matches!(t, EventType::StepStarted | EventType::StepSucceeded))
            .collect::<Vec<_>>(),
        [EventType::StepStarted, EventType::StepSucceeded]
    );
}

#[tokio::test]
async fn retryable_start_failure_in_the_succeed_request_reruns_the_step() {
    retryable_start_failure_reruns_the_step(Duration::ZERO).await;
}

#[tokio::test]
async fn retryable_start_failure_before_the_succeed_reruns_the_step() {
    retryable_start_failure_reruns_the_step(Duration::from_millis(10)).await;
}

/// Runs one step under `own` semantics, on an execution whose
/// `default_step_semantics` is `AtLeastOnceAsync`, with a 300 ms checkpoint
/// latency. Returns the tokio time the step took.
async fn step_time(own: Option<StepSemantics>) -> Duration {
    let options = durable::Options::builder()
        .default_step_semantics(StepSemantics::AtLeastOnceAsync)
        .build()
        .expect("valid options");
    let took = Arc::new(Mutex::new(None));
    let step_took = Arc::clone(&took);
    let result = LocalRunner::new()
        .checkpoint_latency(Duration::from_millis(300))
        .options(options)
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let step_took = Arc::clone(&step_took);
                async move {
                    let begin = tokio::time::Instant::now();
                    let step = ctx.step(|_| async { Ok(1_u32) }).name("s");
                    let value = match own {
                        Some(semantics) => step.semantics(semantics).await?,
                        None => step.await?,
                    };
                    *step_took.lock().expect("lock") = Some(begin.elapsed());
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{result:?}");
    assert!(result.operation("s").expect("step s").succeeded());
    let took = *took.lock().expect("lock");
    took.expect("the step returned")
}

/// `default_step_semantics` reaches the steps of a wrapped handler, and a
/// step's own semantics wins: the async step waits for neither `START`
/// nor `SUCCEED`.
#[tokio::test(start_paused = true)]
async fn default_step_semantics_applies_to_steps() {
    assert_eq!(step_time(None).await, Duration::ZERO);
    assert_eq!(
        step_time(Some(StepSemantics::AtLeastOnce)).await,
        Duration::from_millis(600)
    );
}

/// Runs a handler that fails right after an async step, on `runner`, so
/// the handler returns while the step's `SUCCEED` is still in flight.
/// Returns the result and the body runs.
async fn fail_after_async_step(runner: LocalRunner) -> (TestResult<String>, u32) {
    let runs = Arc::new(AtomicU32::new(0));
    let body_runs = Arc::clone(&runs);
    let result = runner
        .checkpoint_latency(Duration::from_millis(50))
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs);
                async move {
                    ctx.step(move |_| {
                        let body_runs = Arc::clone(&body_runs);
                        async move {
                            body_runs.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(AFTER_START).await;
                            Ok("one".to_owned())
                        }
                    })
                    .name("s")
                    .semantics(StepSemantics::AtLeastOnceAsync)
                    .await?;
                    Err::<String, BoxError>("handler failed".into())
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    (result, runs.load(Ordering::SeqCst))
}

/// The handler fails, and the step's `SUCCEED` then fails with a retryable
/// error. The invocation fails, so the execution is not failed without the
/// step's result. The next invocation runs the step again and saves it.
#[tokio::test]
async fn handler_error_with_retryable_succeed_failure_retries_the_invocation() {
    let (result, runs) =
        fail_after_async_step(LocalRunner::new().fail_checkpoints_after_retryable(1, 1)).await;
    assert_eq!(result.invocation_count(), 2, "the first invocation fails");
    assert_eq!(runs, 2, "the step runs again in the next invocation");
    assert!(result.operation("s").expect("step s").succeeded());
    assert!(result.is_failure(), "{result:?}");
    assert_eq!(result.error_message(), Some("handler failed"));
}

/// The handler fails, and the step's `SUCCEED` then fails with an
/// execution-scope error, such as a validation error. The execution fails
/// in the same invocation.
#[tokio::test]
async fn handler_error_with_execution_succeed_failure_fails_the_execution() {
    let (result, runs) =
        fail_after_async_step(LocalRunner::new().fail_checkpoints_after(1, 1)).await;
    assert_eq!(result.invocation_count(), 1);
    assert_eq!(runs, 1);
    assert!(result.is_failure(), "{result:?}");
    assert_eq!(
        result.error_type(),
        Some("CheckpointUnrecoverableExecutionError")
    );
}

std::thread_local! {
    /// Set while a `wrap_invocation` of a [`Context`] plugin runs on this
    /// thread.
    static IN_INVOCATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Records, for each end hook of a step or a child context, the hook,
/// the operation kind, whether a `wrap_invocation` was entered and the
/// name of the current `tracing` span.
#[derive(Clone, Default)]
struct Context(Arc<Mutex<Vec<Seen>>>);

/// One end hook [`Context`] saw: the hook, the operation kind, whether a
/// `wrap_invocation` was entered and the current span's name.
type Seen = (&'static str, String, bool, Option<&'static str>);

impl Context {
    fn record(&self, hook: &'static str, info: &OperationInfo) {
        let kind = match info.kind() {
            Some(durable::OperationKind::Step) => "step",
            Some(durable::OperationKind::Context) => "context",
            _ => return,
        };
        let span = tracing::Span::current()
            .metadata()
            .map(tracing::Metadata::name);
        self.0.lock().expect("lock").push((
            hook,
            kind.to_owned(),
            IN_INVOCATION.with(std::cell::Cell::get),
            span,
        ));
    }

    fn seen(&self) -> Vec<Seen> {
        self.0.lock().expect("lock").clone()
    }
}

impl Plugin for Context {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        let outer = IN_INVOCATION.with(|flag| flag.replace(true));
        next();
        IN_INVOCATION.with(|flag| flag.set(outer));
    }

    fn on_attempt_end(&self, info: &durable::plugin::AttemptEndInfo) -> HookWork {
        self.record("attempt_end", info.operation());
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.record("operation_end", info);
        Ok(None)
    }
}

impl PluginFactory for Context {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// The end hooks of an async step run inside `wrap_invocation` and inside
/// the SDK's `tracing` span, as an awaited step's do. Current-thread, so
/// the thread-local subscriber and flag see every task.
#[tokio::test]
async fn end_hooks_run_inside_the_wraps_and_the_span() {
    let _guard = test_tracing::install_test_subscriber(tracing_subscriber::registry());
    let seen = Context::default();
    let (result, _) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .options(
                durable::Options::builder()
                    .plugin(seen.clone())
                    .build()
                    .expect("valid options"),
            ),
        Duration::ZERO,
    )
    .await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(
        seen.seen(),
        [
            (
                "attempt_end",
                "step".to_owned(),
                true,
                Some("durable_execution")
            ),
            (
                "operation_end",
                "step".to_owned(),
                true,
                Some("durable_execution")
            ),
        ]
    );
}

/// A child context holds an async step, and one response answers the
/// step's `SUCCEED` and the context's `SUCCEED`. The wait makes the next
/// invocation resume the context, which then writes no `START`, and the
/// coalescing window puts both `SUCCEED`s in one request. The step's end
/// hooks fire before the context's.
#[tokio::test]
async fn async_step_end_hooks_precede_the_child_context_end_hook() {
    let seen = Context::default();
    let result = LocalRunner::new()
        .options(
            durable::Options::builder()
                .checkpoint_delay(Duration::from_millis(200))
                .plugin(seen.clone())
                .build()
                .expect("valid options"),
        )
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                let value = ctx
                    .run_in_child_context(|child| async move {
                        child.wait(Duration::from_secs(1)).name("w").await?;
                        let value = child
                            .step(|_| async { Ok(1_u32) })
                            .name("s")
                            .semantics(StepSemantics::AtLeastOnceAsync)
                            .await?;
                        Ok(value)
                    })
                    .name("child")
                    .await?;
                Ok::<_, BoxError>(value)
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{result:?}");
    assert_eq!(result.invocation_count(), 2);
    let order: Vec<(&'static str, String)> = seen
        .seen()
        .into_iter()
        .map(|(hook, kind, _, _)| (hook, kind))
        .collect();
    assert_eq!(
        order,
        [
            ("attempt_end", "step".to_owned()),
            ("operation_end", "step".to_owned()),
            ("operation_end", "context".to_owned()),
        ]
    );
}
