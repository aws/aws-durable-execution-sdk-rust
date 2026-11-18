//! A body that a later invocation runs again to rebuild a recorded
//! success.
//!
//! A child context or a map item whose result is over 256 KiB records
//! `ReplayChildren`, so each later invocation runs its body again to
//! rebuild the result. An operation in a body there returns its record
//! only when replay can return it without running anything, as a
//! `SUCCEEDED` record. Every other operation, and one without a record,
//! never resolves there and runs nothing, except that a `create_callback`
//! with a `STARTED` or `PENDING` record still returns its callback.
//!
//! In the tests, the body starts operations with `.spawn()` and returns
//! after 10 ms with a 300 KiB result, so the body's scope stops what still
//! runs. Two 1 s waits follow, so invocations 2 and 3 rebuild the body.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::NestingMode;
use durable::builders::wait_for_condition::WaitDecision;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::test_util::{LocalRunner, TestResult};
use durable::{
    BoxError, DurableContext, DurableFuture, OperationStatus, RetryDecision, StepSemantics,
};

/// The handles a body keeps beyond its scope.
type Holder = Arc<Mutex<Vec<DurableFuture<()>>>>;

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn count(counter: &AtomicUsize) -> usize {
    counter.load(Ordering::SeqCst)
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

fn secs(n: u64) -> Duration {
    Duration::from_secs(n)
}

/// A result over 256 KiB, so the context that returns it records
/// `ReplayChildren`.
fn oversized() -> String {
    "x".repeat(300 * 1024)
}

/// The status of the operation named `name`, or `None` when nothing is
/// recorded under that name.
fn status<O>(result: &TestResult<O>, name: &str) -> Option<OperationStatus> {
    result
        .operation(name)
        .map(durable::test_util::TestOperation::status)
}

/// Runs `body` as the handler, with a counter it can add to, a holder that
/// lives until the handler returns, and the 1-based number of the
/// invocation. Returns the result and the counter's value.
async fn run_counted<F, Fut>(body: F) -> (TestResult<usize>, usize)
where
    F: Fn(DurableContext, Arc<AtomicUsize>, Holder, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<usize, BoxError>> + Send + 'static,
{
    run_counted_with(LocalRunner::new(), body).await
}

/// [`run_counted`] on `runner`.
async fn run_counted_with<F, Fut>(runner: LocalRunner, body: F) -> (TestResult<usize>, usize)
where
    F: Fn(DurableContext, Arc<AtomicUsize>, Holder, usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<usize, BoxError>> + Send + 'static,
{
    let runs = counter();
    let invocations = counter();
    let handler_runs = Arc::clone(&runs);
    let body = Arc::new(body);
    let result = runner
        .run(
            move |(), ctx: DurableContext| {
                let holder: Holder = Arc::new(Mutex::new(Vec::new()));
                let invocation = invocations.fetch_add(1, Ordering::SeqCst) + 1;
                let fut = body(
                    ctx,
                    Arc::clone(&handler_runs),
                    Arc::clone(&holder),
                    invocation,
                );
                async move {
                    let output = fut.await;
                    drop(holder);
                    output
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    (result, count(&runs))
}

/// Waits `w1` and `w2`, 1 s each, then returns `len`.
async fn two_waits(ctx: &DurableContext, len: usize) -> Result<usize, BoxError> {
    ctx.wait(secs(1)).name("w1").await?;
    ctx.wait(secs(1)).name("w2").await?;
    Ok(len)
}

/// Starts the step `escaped` with `semantics` and `.spawn()`, keeps its
/// handle in `holder`, sleeps 10 ms and returns an oversized result. The
/// step adds 1 to `runs` and sleeps 50 ms, so the scope stops it in
/// invocation 1 and its record stays `STARTED`.
async fn escape(
    ctx: DurableContext,
    runs: Arc<AtomicUsize>,
    holder: Holder,
    semantics: StepSemantics,
) -> Result<String, BoxError> {
    let handle = ctx
        .step(move |_| async move {
            runs.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(ms(50)).await;
            Ok(())
        })
        .name("escaped")
        .semantics(semantics)
        .spawn();
    holder.lock().expect("lock").push(handle);
    tokio::time::sleep(ms(10)).await;
    Ok(oversized())
}

/// Where [`escape`] runs: a child context, a `with_retry` block, or the
/// one item of a `map` with the given nesting.
#[derive(Clone, Copy)]
enum Site {
    Child,
    Block,
    Item(NestingMode),
}

/// Runs [`escape`] at `site`, then [`two_waits`]. Returns the result and
/// the number of runs of the step body.
async fn escape_at(site: Site, semantics: StepSemantics) -> (TestResult<usize>, usize) {
    run_counted(move |ctx, runs, holder, _invocation| async move {
        let len = match site {
            Site::Child => ctx
                .run_in_child_context(move |child| escape(child, runs, holder, semantics))
                .name("child")
                .await?
                .len(),
            Site::Block => ctx
                .with_retry(move |block, _attempt| {
                    escape(block, Arc::clone(&runs), Arc::clone(&holder), semantics)
                })
                .name("block")
                .await?
                .len(),
            Site::Item(nesting) => ctx
                .map(vec![0_u32], move |child, _item, _idx| {
                    escape(child, Arc::clone(&runs), Arc::clone(&holder), semantics)
                })
                .name("m")
                .nesting(nesting)
                .await?
                .iter()
                .map(String::len)
                .sum(),
        };
        two_waits(&ctx, len).await
    })
    .await
}

/// The step body ran once in 3 invocations, and the record of `escaped`
/// stayed `STARTED`.
fn assert_ran_once(result: &TestResult<usize>, runs: usize) {
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 3, "{result}");
    assert_eq!(runs, 1, "{result}");
    assert_eq!(
        status(result, "escaped"),
        Some(OperationStatus::Started),
        "{result}"
    );
}

#[tokio::test(start_paused = true)]
async fn rebuilt_child_does_not_run_its_unfinished_spawned_step_again() {
    let (result, runs) = escape_at(Site::Child, StepSemantics::AtLeastOncePerRetry).await;
    assert_ran_once(&result, runs);
}

/// The block's own record and its `attempt-1` context both record
/// `ReplayChildren`, so each later invocation runs the attempt's body
/// again.
#[tokio::test(start_paused = true)]
async fn rebuilt_with_retry_block_does_not_run_its_unfinished_spawned_step_again() {
    let (result, runs) = escape_at(Site::Block, StepSemantics::AtLeastOncePerRetry).await;
    assert_ran_once(&result, runs);
}

#[tokio::test(start_paused = true)]
async fn rebuilt_map_item_does_not_run_its_unfinished_spawned_step_again() {
    let (result, runs) = escape_at(
        Site::Item(NestingMode::Normal),
        StepSemantics::AtLeastOncePerRetry,
    )
    .await;
    assert_ran_once(&result, runs);
}

#[tokio::test(start_paused = true)]
async fn rebuilt_flat_map_item_does_not_run_its_unfinished_spawned_step_again() {
    let (result, runs) = escape_at(
        Site::Item(NestingMode::Flat),
        StepSemantics::AtLeastOncePerRetry,
    )
    .await;
    assert_ran_once(&result, runs);
}

#[tokio::test(start_paused = true)]
async fn rebuilt_child_does_not_retry_its_unfinished_at_most_once_step() {
    let (result, runs) = escape_at(Site::Child, StepSemantics::AtMostOncePerRetry).await;
    assert_ran_once(&result, runs);
}

/// A `NestingMode::Normal` item whose own record succeeded with
/// `ReplayChildren` runs its body again in each invocation in which item 1
/// still waits, so the map's own record is not final yet.
#[tokio::test(start_paused = true)]
async fn rebuilt_item_of_a_running_map_does_not_run_its_unfinished_spawned_step_again() {
    let (result, runs) = run_counted(|ctx, runs, holder, _invocation| async move {
        let results = ctx
            .map(vec![0_u32, 1], move |child, item, _idx| {
                let runs = Arc::clone(&runs);
                let holder = Arc::clone(&holder);
                async move {
                    if item == 1 {
                        two_waits(&child, 0).await?;
                        return Ok(String::new());
                    }
                    escape(child, runs, holder, StepSemantics::AtLeastOncePerRetry).await
                }
            })
            .name("m")
            .await?;
        Ok(results.iter().map(String::len).sum())
    })
    .await;
    assert_ran_once(&result, runs);
}

/// Records the name of every operation whose `on_operation_start` fires.
struct StartHooks {
    names: Arc<Mutex<Vec<String>>>,
}

impl Plugin for StartHooks {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        let name = info.name().unwrap_or_default().to_owned();
        self.names.lock().expect("lock").push(name);
        Ok(None)
    }
}

/// Gives each invocation a [`StartHooks`] that shares one list.
struct StartHooksFactory {
    names: Arc<Mutex<Vec<String>>>,
}

impl PluginFactory for StartHooksFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(StartHooks {
            names: Arc::clone(&self.names),
        })))
    }
}

/// The step `escaped` fires its start hook once, in invocation 1. A rebuilt
/// body fires no hook for it.
#[tokio::test(start_paused = true)]
async fn rebuilt_child_fires_no_hook_for_its_unfinished_spawned_step() {
    let names = Arc::new(Mutex::new(Vec::new()));
    let options = durable::Options::builder()
        .plugin(StartHooksFactory {
            names: Arc::clone(&names),
        })
        .build()
        .expect("valid options");
    let (result, runs) = run_counted_with(
        LocalRunner::new().options(options),
        |ctx, runs, holder, _invocation| async move {
            let len = ctx
                .run_in_child_context(move |child| {
                    escape(child, runs, holder, StepSemantics::AtLeastOncePerRetry)
                })
                .name("child")
                .await?
                .len();
            two_waits(&ctx, len).await
        },
    )
    .await;
    let starts = names
        .lock()
        .expect("lock")
        .iter()
        .filter(|name| *name == "escaped")
        .count();
    assert_eq!(starts, 1, "{result}");
    assert_ran_once(&result, runs);
}

/// A child context that the body started with `.spawn()` and that its scope
/// stopped keeps its `STARTED` record, and its body does not run again.
#[tokio::test(start_paused = true)]
async fn rebuilt_child_does_not_run_its_unfinished_nested_child_again() {
    let (result, runs) = run_counted(|ctx, runs, holder, _invocation| async move {
        let len = ctx
            .run_in_child_context(move |child| async move {
                let handle = child
                    .run_in_child_context(move |inner| async move {
                        runs.fetch_add(1, Ordering::SeqCst);
                        inner
                            .step(|_| async {
                                tokio::time::sleep(ms(50)).await;
                                Ok(())
                            })
                            .name("inner-step")
                            .await?;
                        Ok(())
                    })
                    .name("inner")
                    .spawn();
                holder.lock().expect("lock").push(handle);
                tokio::time::sleep(ms(10)).await;
                Ok(oversized())
            })
            .name("child")
            .await?
            .len();
        two_waits(&ctx, len).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(runs, 1, "{result}");
    assert_eq!(
        status(&result, "inner"),
        Some(OperationStatus::Started),
        "{result}"
    );
}

/// A `wait_for_condition` that the body started with `.spawn()` is in its
/// first check, which sleeps 50 ms, when the scope stops it, so its record
/// stays `STARTED`. A rebuilt body runs no further check.
#[tokio::test(start_paused = true)]
async fn rebuilt_child_runs_no_check_of_its_unfinished_condition() {
    let (result, checks) = run_counted(|ctx, checks, _holder, _invocation| async move {
        let len = ctx
            .run_in_child_context(move |child| async move {
                drop(
                    child
                        .wait_for_condition(
                            move |_ctx, state: u32| {
                                let checks = Arc::clone(&checks);
                                async move {
                                    checks.fetch_add(1, Ordering::SeqCst);
                                    tokio::time::sleep(ms(50)).await;
                                    Ok(state + 1)
                                }
                            },
                            0,
                        )
                        .wait_strategy_fn(|_state: u32, _attempt| {
                            WaitDecision::continue_with(secs(1))
                        })
                        .name("cond")
                        .spawn(),
                );
                tokio::time::sleep(ms(10)).await;
                Ok(oversized())
            })
            .name("child")
            .await?
            .len();
        two_waits(&ctx, len).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 3, "{result}");
    assert_eq!(checks, 1, "{result}");
    assert_eq!(
        status(&result, "cond"),
        Some(OperationStatus::Started),
        "{result}"
    );
}

/// A step that the body started with `.spawn()` fails at once and waits
/// 1 s for its retry when the scope stops it, so its record is `PENDING`
/// and then `READY`. A rebuilt body does not retry it.
#[tokio::test(start_paused = true)]
async fn rebuilt_child_does_not_retry_its_unfinished_failing_step() {
    let (result, runs) = run_counted(|ctx, runs, holder, _invocation| async move {
        let len = ctx
            .run_in_child_context(move |child| async move {
                let flaky = child
                    .step(move |_| {
                        let runs = Arc::clone(&runs);
                        async move {
                            runs.fetch_add(1, Ordering::SeqCst);
                            Err::<(), BoxError>("flaky".into())
                        }
                    })
                    .name("flaky")
                    .retry_strategy(|_err, _attempt| RetryDecision::retry(secs(1)))
                    .spawn();
                holder.lock().expect("lock").push(flaky);
                escape(
                    child,
                    counter(),
                    Arc::clone(&holder),
                    StepSemantics::AtLeastOncePerRetry,
                )
                .await
            })
            .name("child")
            .await?
            .len();
        two_waits(&ctx, len).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 3, "{result}");
    assert_eq!(runs, 1, "{result}");
    assert_eq!(
        status(&result, "flaky"),
        Some(OperationStatus::Ready),
        "{result}"
    );
}

/// An invoke started with `.spawn()` starts on its own task. The body
/// returns without yielding in invocation 1, so the scope closes before
/// that task runs and the invoke has no record. In later invocations the
/// body sleeps 10 ms before it returns, so the task runs first, as it can
/// when a multi-threaded runtime schedules the task differently in each
/// invocation. The rebuilt body starts no chained invoke.
#[tokio::test(start_paused = true)]
async fn rebuilt_child_starts_no_unrecorded_invoke() {
    let calls = counter();
    let target_calls = Arc::clone(&calls);
    let invocations = counter();
    let result = LocalRunner::new()
        .register_function("target", move |(): ()| {
            let target_calls = Arc::clone(&target_calls);
            async move {
                target_calls.fetch_add(1, Ordering::SeqCst);
                Ok(1_u32)
            }
        })
        .run(
            move |(), ctx: DurableContext| {
                let invocation = invocations.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    let len = ctx
                        .run_in_child_context(move |child| async move {
                            drop(child.invoke::<u32>("target", ()).name("inv").spawn());
                            if invocation > 1 {
                                tokio::time::sleep(ms(10)).await;
                            }
                            Ok(oversized())
                        })
                        .name("child")
                        .await?
                        .len();
                    two_waits(&ctx, len).await
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(status(&result, "inv"), None, "{result}");
    assert_eq!(count(&calls), 0, "{result}");
    assert_eq!(result.invocation_count(), 3, "{result}");
}

/// A combinator started with `.spawn()` starts on its own task, as an
/// invoke does (see [`rebuilt_child_starts_no_unrecorded_invoke`]), so it
/// has no record after invocation 1. The rebuilt body runs neither the
/// combinator nor its input.
#[tokio::test(start_paused = true)]
async fn rebuilt_child_runs_no_unrecorded_combinator() {
    let (result, _) = run_counted(|ctx, _runs, _holder, invocation| async move {
        let len = ctx
            .run_in_child_context(move |child| async move {
                let input = child.step(|_| async { Ok(1_u32) }).name("input").future();
                drop(child.try_join_all([input]).name("all").spawn());
                if invocation > 1 {
                    tokio::time::sleep(ms(10)).await;
                }
                Ok(oversized())
            })
            .name("child")
            .await?
            .len();
        two_waits(&ctx, len).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(status(&result, "all"), None, "{result}");
    assert_eq!(status(&result, "input"), None, "{result}");
    assert_eq!(result.invocation_count(), 3, "{result}");
}

/// A rebuilt body still resolves a `create_callback` whose record is
/// `STARTED`, so a body that returns the callback ID returns. The timeout
/// turns a body that never returns into a failure.
#[tokio::test(start_paused = true)]
async fn rebuilt_child_resolves_its_started_callback() {
    let run = run_counted(|ctx, _runs, _holder, _invocation| async move {
        let id = ctx
            .run_in_child_context(|child| async move {
                let callback = child.create_callback::<String>().name("cb").await?;
                Ok(format!("{}{}", callback.id(), oversized()))
            })
            .name("child")
            .await?;
        ctx.wait(secs(1)).name("w1").await?;
        Ok(id.len())
    });
    let (result, _) = tokio::time::timeout(secs(60), run)
        .await
        .expect("the rebuilt body returns");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(
        status(&result, "cb"),
        Some(OperationStatus::Started),
        "{result}"
    );
}
