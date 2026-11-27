//! `StepSemantics::AtLeastOnceAsync` through the whole invocation.
//!
//! The step returns before its `SUCCEED` is answered. These tests drive
//! handlers through [`LocalRunner`] with a slow checkpoint client and check
//! that the invocation still saves the result before it reports the
//! execution's outcome, in checkpoint order, and that a failed `SUCCEED`
//! write does not report the execution as succeeded.

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

/// Runs a handler with one async step named `s` that counts its body runs
/// and returns `"one"`, on `runner`. Returns the result and the body runs.
async fn run_async_step(runner: LocalRunner) -> (TestResult<String>, u32) {
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
    let (result, runs) =
        run_async_step(LocalRunner::new().checkpoint_latency(Duration::from_millis(50))).await;
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
    let (result, runs) = run_async_step(LocalRunner::new().options(options)).await;
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

/// Records the info of every step's `on_operation_end`.
#[derive(Clone, Default)]
struct StepEnds(Arc<Mutex<Vec<OperationInfo>>>);

impl Plugin for StepEnds {
    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if info.kind() == Some(durable::OperationKind::Step) {
            self.0.lock().expect("lock").push(info.clone());
        }
        Ok(None)
    }
}

impl PluginFactory for StepEnds {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// The end hook reports `SUCCEEDED`, the result and both timestamps before
/// the service answered the `SUCCEED`.
#[tokio::test]
async fn end_hook_reports_succeeded() {
    let ends = StepEnds::default();
    let options = durable::Options::builder()
        .plugin(ends.clone())
        .build()
        .expect("valid options");
    let (result, _) = run_async_step(
        LocalRunner::new()
            .checkpoint_latency(Duration::from_millis(50))
            .options(options),
    )
    .await;
    assert!(result.is_success(), "{result:?}");
    let ends = ends.0.lock().expect("lock");
    let [end] = ends.as_slice() else {
        unreachable!("expected one step end, got {ends:?}");
    };
    assert_eq!(end.status(), Some(durable::OperationStatus::Succeeded));
    assert_eq!(end.result(), Some(r#""one""#));
    assert!(end.start_timestamp().is_some(), "a start time");
    assert!(end.end_timestamp().is_some(), "an end time");
}

/// The service permanently rejects the `SUCCEED`: the execution fails, and
/// it is not reported as succeeded.
#[tokio::test]
async fn rejected_succeed_fails_the_execution() {
    let (result, runs) = run_async_step(LocalRunner::new().fail_checkpoints_after(1, 1)).await;
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
    let (result, runs) =
        run_async_step(LocalRunner::new().fail_checkpoints_after_retryable(1, 1)).await;
    assert!(result.is_success(), "{result:?}");
    assert_eq!(runs, 2, "the body runs again in the next invocation");
    assert_eq!(result.invocation_count(), 2);
    assert!(result.operation("s").expect("step s").succeeded());
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
/// step's own semantics wins: the async step waits only for `START`.
#[tokio::test(start_paused = true)]
async fn default_step_semantics_applies_to_steps() {
    assert_eq!(step_time(None).await, Duration::from_millis(300));
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
