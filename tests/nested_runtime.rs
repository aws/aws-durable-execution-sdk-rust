//! A handler that polls durable operations inside a second Tokio runtime,
//! and then drops that runtime.
//!
//! The SDK starts operation tasks, batch item tasks, combinator input tasks
//! and checkpoint writes on the invocation's runtime, so dropping the second
//! runtime drops none of them.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::pin::Pin;
use std::task::{Context, Waker};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{LocalRunner, TestResult};
use durable::{BoxError, DurableContext, OperationStatus};

/// The status of the operation named `name`, or `None` when nothing is
/// recorded under that name.
fn status<O>(result: &TestResult<O>, name: &str) -> Option<OperationStatus> {
    result
        .operation(name)
        .map(durable::test_util::TestOperation::status)
}

/// Calls `f` while a new current-thread runtime is current, keeps that
/// runtime running for `run_for` after `f` returns, then drops the runtime.
fn in_a_dropped_runtime<T>(run_for: Duration, f: impl FnOnce() -> T) -> Result<T, BoxError> {
    tokio::task::block_in_place(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let out = runtime.block_on(async {
            let out = f();
            tokio::time::sleep(run_for).await;
            out
        });
        drop(runtime);
        Ok(out)
    })
}

/// Polls `fut` once with a no-op waker, and returns whether it is still
/// pending.
fn poll_once<F: Future + Unpin>(fut: &mut F) -> bool {
    Pin::new(fut)
        .poll(&mut Context::from_waker(Waker::noop()))
        .is_pending()
}

/// Runs `handler` on `runner` with the unit event, and fails when the run
/// does not finish within 10 s.
async fn run_within_10s<O, F, Fut>(runner: LocalRunner, handler: F) -> TestResult<O>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    tokio::time::timeout(Duration::from_secs(10), runner.run(handler, ()))
        .await
        .expect("the run finishes within 10 s")
        .expect("the run finishes")
}

/// A step started with `.spawn()` while the second runtime is current runs
/// to completion after the handler drops that runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawned_step_started_in_a_dropped_second_runtime_runs() {
    async fn handler(_event: (), ctx: DurableContext) -> Result<u32, BoxError> {
        let spawned = in_a_dropped_runtime(Duration::ZERO, || {
            ctx.step(|_| async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(1_u32)
            })
            .name("spawned")
            .spawn()
        })?;
        Ok(spawned.await?)
    }
    let result = run_within_10s(LocalRunner::new(), handler).await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_eq!(
        status(&result, "spawned"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

/// A `map` whose first poll starts its items while the second runtime is
/// current finishes when the handler awaits it after dropping that runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_first_polled_in_a_dropped_second_runtime_finishes() {
    async fn handler(_event: (), ctx: DurableContext) -> Result<Vec<u32>, BoxError> {
        let mut map = Box::pin(
            ctx.map(vec![1_u32, 2], |child, item, _index| async move {
                Ok(child
                    .step(move |_| async move {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(item * 10)
                    })
                    .name("scale")
                    .await?)
            })
            .name("scale-all")
            .into_future(),
        );
        let pending = in_a_dropped_runtime(Duration::ZERO, || poll_once(&mut map))?;
        assert!(
            pending,
            "the first poll starts the items and waits for them"
        );
        Ok(map.await?)
    }
    let result = run_within_10s(LocalRunner::new(), handler).await;
    assert_eq!(result.output(), Some(&vec![10, 20]), "{result}");
    assert_eq!(
        status(&result, "scale-all"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

/// A combinator whose first poll starts its inputs while the second
/// runtime is current finishes when the handler awaits it after dropping
/// that runtime.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn combinator_first_polled_in_a_dropped_second_runtime_finishes() {
    async fn handler(_event: (), ctx: DurableContext) -> Result<Vec<u32>, BoxError> {
        let a = ctx
            .step(|_| async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(1_u32)
            })
            .name("a")
            .future();
        let b = ctx
            .step(|_| async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(2_u32)
            })
            .name("b")
            .future();
        let mut all = Box::pin(ctx.try_join_all([a, b]).name("all").into_future());
        let pending = in_a_dropped_runtime(Duration::ZERO, || poll_once(&mut all))?;
        assert!(
            pending,
            "the first poll starts the inputs and waits for them"
        );
        Ok(all.await?)
    }
    let result = run_within_10s(LocalRunner::new(), handler).await;
    assert_eq!(result.output(), Some(&vec![1, 2]), "{result}");
    assert_eq!(
        status(&result, "all"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

/// A checkpoint request that a step polled inside the second runtime
/// starts, and that is in flight when the handler drops that runtime, is
/// applied, so the step finishes in the same invocation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn checkpoint_write_in_flight_when_a_second_runtime_drops_is_applied() {
    async fn handler(_event: (), ctx: DurableContext) -> Result<u32, BoxError> {
        let mut step = Box::pin(
            ctx.step(|_| async { Ok(1_u32) })
                .name("written")
                .into_future(),
        );
        // The first poll queues the step's START. The second runtime then
        // runs for 50 ms, so the request that carries the START is in
        // flight, for 500 ms, when the runtime is dropped.
        let pending = in_a_dropped_runtime(Duration::from_millis(50), || poll_once(&mut step))?;
        assert!(pending, "the step waits for its START response");
        Ok(step.await?)
    }
    let runner = LocalRunner::new().checkpoint_latency(Duration::from_millis(500));
    let result = run_within_10s(runner, handler).await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_eq!(result.invocations().len(), 1, "{result}");
    assert_eq!(
        status(&result, "written"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}
