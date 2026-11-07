//! An operation claimed on a task or thread the SDK did not start.
//!
//! The SDK assigns operation IDs in the order the handler creates
//! operations. Code on a raw `tokio::spawn` task or a `std::thread` runs
//! beside the handler, so an operation it creates takes an ID at a moment
//! that depends on scheduling. Such a claim must fail the execution with a
//! context usage error on the first invocation, at the claim, whatever the
//! code does with the operation afterwards: drop it, start it with
//! `.spawn()`, hand its `.future()` to the handler, or await it. An
//! operation the handler creates and a raw task or thread then awaits or
//! spawns must fail the execution the same way, when it executes or
//! spawns, and a refused `.spawn()` must start nothing. Plugin hook work
//! runs on such a task too, so a claim it makes is refused the same way.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions
#![expect(clippy::disallowed_methods)] // reason: each test misuses tokio::spawn on purpose

use std::future::IntoFuture;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::test_util::{LocalRunner, TestOperation, TestResult};
use durable::{BoxError, DurableContext, OperationStatus, Options};

/// Runs `handler` with the unit event, under a 20 second limit.
async fn run<F, Fut>(handler: F) -> TestResult<u32>
where
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<u32, BoxError>> + Send + 'static,
{
    run_on(LocalRunner::new(), handler).await
}

/// Runs `handler` on `runner` with the unit event, under a 20 second limit.
async fn run_on<F, Fut>(runner: LocalRunner, handler: F) -> TestResult<u32>
where
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<u32, BoxError>> + Send + 'static,
{
    tokio::time::timeout(Duration::from_secs(20), runner.run(handler, ()))
        .await
        .expect("the run finishes within 20 seconds")
        .expect("the runner returns a result")
}

/// Asserts the refusal names the `spawn` method.
fn assert_refused_at_spawn(result: &TestResult<u32>) {
    assert_refused(result);
    assert!(
        result
            .error_message()
            .is_some_and(|m| m.starts_with("context usage error in \"spawn\"")),
        "expected the refusal at `.spawn()`; got {:?}",
        result.error_message()
    );
}

/// The step every handler runs after the foreign claim. With the claim
/// refused, the execution fails before this step matters.
async fn after(ctx: &DurableContext) -> Result<u32, BoxError> {
    Ok(ctx.step(|_| async { Ok(7_u32) }).name("after").await?)
}

/// Asserts the execution failed on the first invocation with a context
/// usage error. The message shows what the execution did instead.
fn assert_refused(result: &TestResult<u32>) {
    let after_id = result.operation("after").map(|op| op.id().to_owned());
    let shown = format!(
        "status {:?}, error {:?}, `after` has ID {after_id:?}\n{result}",
        result.status(),
        result.error_message(),
    );
    assert!(result.is_failure(), "expected FAILED; got {shown}");
    assert_eq!(result.error_type(), Some("Error"), "{shown}");
    assert!(
        result
            .error_message()
            .is_some_and(|m| m.starts_with("context usage error")),
        "expected a context usage error; got {shown}"
    );
    assert_eq!(result.invocation_count(), 1, "{shown}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_task_builder_dropped_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let foreign = ctx.clone();
        tokio::spawn(async move {
            drop(foreign.step(|_| async { Ok(1_u32) }).name("foreign"));
        })
        .await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thread_builder_dropped_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let foreign = ctx.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            drop(foreign.step(|_| async { Ok(1_u32) }).name("foreign"));
            let _ = tx.send(());
        });
        rx.await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_task_spawn_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let foreign = ctx.clone();
        tokio::spawn(async move {
            let _ = foreign
                .step(|_| async { Ok(1_u32) })
                .name("foreign")
                .spawn()
                .await;
        })
        .await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thread_spawn_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let foreign = ctx.clone();
        let runtime = tokio::runtime::Handle::current();
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            let _entered = runtime.enter();
            let handle = foreign
                .step(|_| async { Ok(1_u32) })
                .name("foreign")
                .spawn();
            let _ = runtime.block_on(handle);
            let _ = tx.send(());
        });
        rx.await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_task_future_awaited_by_handler_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let foreign = ctx.clone();
        #[expect(clippy::async_yields_async)] // reason: the raw task hands the future back
        let future = tokio::spawn(async move {
            foreign
                .step(|_| async { Ok(1_u32) })
                .name("foreign")
                .future()
        })
        .await?;
        future.await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_task_awaited_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let foreign = ctx.clone();
        let _ = tokio::spawn(
            async move { foreign.step(|_| async { Ok(1_u32) }).name("foreign").await },
        )
        .await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn thread_awaited_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let foreign = ctx.clone();
        let runtime = tokio::runtime::Handle::current();
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            let outcome = runtime.block_on(
                foreign
                    .step(|_| async { Ok(1_u32) })
                    .name("foreign")
                    .into_future(),
            );
            let _ = tx.send(outcome.is_ok());
        });
        let _ = rx.await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_builder_awaited_on_raw_task_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let builder = ctx.step(|_| async { Ok(1_u32) }).name("foreign");
        let _ = tokio::spawn(async move { builder.await }).await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_builder_awaited_on_thread_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let builder = ctx.step(|_| async { Ok(1_u32) }).name("foreign");
        let runtime = tokio::runtime::Handle::current();
        let (tx, rx) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            let outcome = runtime.block_on(builder.into_future());
            let _ = tx.send(outcome.is_ok());
        });
        let _ = rx.await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_builder_spawned_on_raw_task_starts_nothing() {
    let ran = Arc::new(AtomicBool::new(false));
    let ran_in = Arc::clone(&ran);
    // The raw task reports whether the handle resolved to an error. It is
    // not an SDK task, so it can outlive the run; the channel waits for it.
    let (report, handle_outcome) = std::sync::mpsc::channel::<bool>();
    let result = run(move |(), ctx: DurableContext| {
        let ran = Arc::clone(&ran_in);
        let report = report.clone();
        async move {
            let builder = ctx
                .step(move |_| async move {
                    ran.store(true, Ordering::SeqCst);
                    Ok(1_u32)
                })
                .name("foreign");
            tokio::spawn(async move {
                let outcome = builder.spawn().await;
                let _ = report.send(outcome.is_err());
            })
            .await?;
            after(&ctx).await
        }
    })
    .await;
    assert_refused_at_spawn(&result);
    let handle_failed = handle_outcome
        .recv_timeout(Duration::from_secs(5))
        .expect("the raw task reports the handle's outcome");
    assert!(handle_failed, "the refused handle resolves to an error");
    assert!(
        !ran.load(Ordering::SeqCst),
        "the refused step body never runs"
    );
    assert!(
        result.operation("foreign").is_none(),
        "nothing is recorded for `foreign`\n{result}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_task_combinator_polled_is_refused() {
    let result = run(|(), ctx: DurableContext| async move {
        let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
        let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
        let foreign = ctx.clone();
        let _ = tokio::spawn(async move { foreign.race([a, b]).name("foreign").await }).await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_body_resumed_on_raw_task_claim_is_refused() {
    // The handler starts the child, so its entry check passes, and the body
    // parks on a channel. A raw task then polls the child to resume the
    // body. The child's poll gate checks ownership on every poll, so the
    // raw task is refused at `execute` before the body resumes and claims
    // `inner`. A claim by a caller that does not own the context while the
    // matching body is active is refused too; the unit test
    // `claim_in_own_body_on_raw_task_records_fatal` covers that claim,
    // which no public API lets a raw task reach.
    let result = run(|(), ctx: DurableContext| async move {
        let (go, wait) = tokio::sync::oneshot::channel::<()>();
        let mut child = Box::pin(
            ctx.run_in_child_context(move |child| async move {
                let _ = wait.await;
                drop(child.step(|_| async { Ok(1_u32) }).name("inner"));
                Ok(0_u32)
            })
            .name("foreign")
            .into_future(),
        );
        let first =
            std::future::poll_fn(|cx| std::task::Poll::Ready(child.as_mut().poll(cx))).await;
        assert!(first.is_pending(), "the body parks on the channel");
        let _ = go.send(());
        let _ = tokio::spawn(child).await?;
        after(&ctx).await
    })
    .await;
    assert_refused_at_execute(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn combinator_built_on_raw_task_polled_by_handler_succeeds() {
    // Building a combinator claims no ID, so a raw task may build it. The
    // handler polls it first, so the combinator claims in the handler's
    // control flow and the execution succeeds.
    let result = run(|(), ctx: DurableContext| async move {
        let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
        let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
        let foreign = ctx.clone();
        let built = tokio::spawn(async move { foreign.race([a, b]).name("race") }).await?;
        let winner = built.await?;
        let last = after(&ctx).await?;
        Ok(winner + last)
    })
    .await;
    assert!(
        result.is_success(),
        "expected SUCCEEDED; got {:?} {:?}\n{result}",
        result.status(),
        result.error_message()
    );
    assert!(
        result.operation("race").is_some(),
        "the race is recorded\n{result}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_builder_spawned_on_thread_without_runtime_is_refused() {
    // The thread enters no runtime, so `.spawn()` can start no task. The
    // ownership check runs before the runtime check, so the call fails the
    // execution rather than only returning the no-runtime error.
    let result = run(|(), ctx: DurableContext| async move {
        let builder = ctx.step(|_| async { Ok(1_u32) }).name("foreign");
        std::thread::spawn(move || drop(builder.spawn()))
            .join()
            .map_err(|_| "the thread panicked")?;
        after(&ctx).await
    })
    .await;
    assert_refused_at_spawn(&result);
    assert!(
        result.operation("foreign").is_none(),
        "nothing is recorded for `foreign`\n{result}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_invoke_spawned_on_raw_task_is_refused() {
    // An invoke's `.spawn()` runs no start step on the caller, so the
    // refusal comes from the check before its task starts.
    let runner = LocalRunner::new()
        .register_function("target", |n: u32| async move { Ok::<u32, BoxError>(n) });
    let result = run_on(runner, |(), ctx: DurableContext| async move {
        let builder = ctx.invoke::<u32, _>("target", 1_u32).name("foreign");
        let _ = tokio::spawn(async move { builder.spawn().await }).await?;
        after(&ctx).await
    })
    .await;
    assert_refused_at_spawn(&result);
}

/// The context the handler shares with the plugin, and the channel on which
/// the hook work reports that it claimed.
#[derive(Clone, Default)]
struct HookShared {
    ctx: Arc<Mutex<Option<DurableContext>>>,
    claimed: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

/// A plugin whose work for the end of operation `first` claims a step on
/// the shared context and drops it.
struct ClaimingPlugin(HookShared);

impl Plugin for ClaimingPlugin {
    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if info.name() != Some("first") {
            return Ok(None);
        }
        let ctx = self.0.ctx.lock().expect("lock").take();
        let claimed = self.0.claimed.lock().expect("lock").take();
        Ok(Some(Box::pin(async move {
            if let Some(ctx) = ctx {
                drop(ctx.step(|_| async { Ok(1_u32) }).name("foreign"));
            }
            if let Some(claimed) = claimed {
                let _ = claimed.send(());
            }
            Ok(())
        })))
    }
}

impl PluginFactory for HookShared {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(ClaimingPlugin(self.clone()))))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plugin_hook_work_claim_is_refused() {
    // Hook work runs beside the handler, under the plugin guard, so a claim
    // it makes on a captured context fails the execution with a context
    // usage error.
    let shared = HookShared::default();
    let (claimed, on_claim) = tokio::sync::oneshot::channel();
    *shared.claimed.lock().expect("lock") = Some(claimed);
    let on_claim = Arc::new(Mutex::new(Some(on_claim)));
    let options = Options::builder()
        .plugin(shared.clone())
        .build()
        .expect("valid options");
    let result = run_on(
        LocalRunner::new().options(options),
        move |(), ctx: DurableContext| {
            let shared = shared.clone();
            let on_claim = on_claim.lock().expect("lock").take();
            async move {
                *shared.ctx.lock().expect("lock") = Some(ctx.clone());
                ctx.step(|_| async { Ok(0_u32) }).name("first").await?;
                if let Some(on_claim) = on_claim {
                    tokio::time::timeout(Duration::from_secs(5), on_claim).await??;
                }
                after(&ctx).await
            }
        },
    )
    .await;
    assert_refused(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_task_combinator_spawn_is_refused_at_spawn() {
    // The ownership check runs before the combinator claims its ID, so the
    // refusal names `spawn` and the combinator records nothing.
    let result = run(|(), ctx: DurableContext| async move {
        let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
        let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
        let foreign = ctx.clone();
        let _ =
            tokio::spawn(async move { foreign.race([a, b]).name("foreign").spawn().await }).await?;
        after(&ctx).await
    })
    .await;
    assert_refused_at_spawn(&result);
    assert!(
        result.operation("foreign").is_none(),
        "nothing is recorded for `foreign`\n{result}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_callback_result_awaited_on_raw_task_is_refused() {
    // The handler creates the callback, which stays pending. A raw task
    // awaits its result and gives up after a short wait.
    let result = run(|(), ctx: DurableContext| async move {
        let cb = ctx.create_callback::<u32>().name("approval").await?;
        let _ = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_millis(300), cb.result()).await
        })
        .await?;
        after(&ctx).await
    })
    .await;
    assert_refused(&result);
    assert!(
        result
            .error_message()
            .is_some_and(|m| m.starts_with("context usage error in \"execute\"")),
        "expected the refusal at execution; got {:?}",
        result.error_message()
    );
}

/// Starts an operation the handler built: calls its `.spawn()`.
type SpawnBuilt = Box<dyn FnOnce() -> durable::DurableFuture<u32> + Send>;

/// Runs a handler whose first invocation records step `foreign` at the
/// first position and suspends on a wait. Its second invocation builds the
/// operation `build` returns at that position, which the record
/// contradicts, and a raw task calls its `.spawn()`. The refusal of the
/// `.spawn()` must win over the replay mismatch.
async fn run_contradicted_spawn<B>(runner: LocalRunner, build: B) -> TestResult<u32>
where
    B: Fn(&DurableContext) -> SpawnBuilt + Send + Sync + 'static,
{
    let first = Arc::new(AtomicBool::new(true));
    let build = Arc::new(build);
    run_on(runner, move |(), ctx: DurableContext| {
        let first = Arc::clone(&first);
        let build = Arc::clone(&build);
        async move {
            if first.swap(false, Ordering::SeqCst) {
                ctx.step(|_| async { Ok(1_u32) }).name("foreign").await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                return after(&ctx).await;
            }
            let spawn = build(&ctx);
            let _ = tokio::spawn(async move { spawn().await }).await?;
            after(&ctx).await
        }
    })
    .await
}

/// Asserts the refusal at `.spawn()` on the second invocation, and that
/// it, not the replay mismatch, is the execution's error.
fn assert_refused_at_spawn_on_replay(result: &TestResult<u32>) {
    let shown = format!(
        "status {:?}, error {:?} {:?}\n{result}",
        result.status(),
        result.error_type(),
        result.error_message(),
    );
    assert!(result.is_failure(), "expected FAILED; got {shown}");
    assert_eq!(result.error_type(), Some("Error"), "{shown}");
    assert!(
        result
            .error_message()
            .is_some_and(|m| m.starts_with("context usage error in \"spawn\"")),
        "expected the refusal at `.spawn()`; got {shown}"
    );
    assert_eq!(result.invocation_count(), 2, "{shown}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_invoke_spawned_on_raw_task_with_replay_mismatch_is_refused() {
    // Building the invoke's future records the replay mismatch, so the
    // ownership check must run before the future is built.
    let runner = LocalRunner::new()
        .register_function("target", |n: u32| async move { Ok::<u32, BoxError>(n) });
    let result = run_contradicted_spawn(runner, |ctx: &DurableContext| {
        let builder = ctx.invoke::<u32, _>("target", 1_u32).name("foreign");
        Box::new(move || builder.spawn())
    })
    .await;
    assert_refused_at_spawn_on_replay(&result);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_condition_spawned_on_raw_task_with_replay_mismatch_is_refused() {
    // The same with a `wait_for_condition`, whose recorded subtype differs.
    let result = run_contradicted_spawn(LocalRunner::new(), |ctx: &DurableContext| {
        let builder = ctx
            .wait_for_condition(|_, state: u32| async move { Ok(state) }, 0_u32)
            .wait_strategy_fn(|_state, _attempt| {
                durable::builders::wait_for_condition::WaitDecision::complete()
            })
            .name("foreign");
        Box::new(move || builder.spawn())
    })
    .await;
    assert_refused_at_spawn_on_replay(&result);
}

/// Polls `future` once on the calling task and returns whether it is still
/// pending.
async fn poll_once<T>(future: &mut std::pin::Pin<Box<impl Future<Output = T>>>) -> bool {
    std::future::poll_fn(|cx| std::task::Poll::Ready(future.as_mut().poll(cx).is_pending())).await
}

/// Asserts the refusal is the execution check, `execute`.
fn assert_refused_at_execute(result: &TestResult<u32>) {
    assert_refused(result);
    assert!(
        result
            .error_message()
            .is_some_and(|m| m.starts_with("context usage error in \"execute\"")),
        "expected the refusal at execution; got {:?}",
        result.error_message()
    );
}

/// Asserts the operation named `name` was recorded and has no outcome.
fn assert_no_outcome(result: &TestResult<u32>, name: &str) {
    let status = result.operation(name).map(TestOperation::status);
    assert_eq!(status, Some(OperationStatus::Started), "{name:?}\n{result}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_wait_resumed_on_raw_task_is_refused() {
    // The handler polls the wait until it suspends, so its execution entry
    // check passes on the owner. A raw task then polls it again. Nothing
    // on the raw task claims an operation, so the resumed poll is the one
    // refusal point.
    let result = run(|(), ctx: DurableContext| async move {
        let mut wait = Box::pin(ctx.wait(Duration::from_secs(60)).name("w").into_future());
        assert!(poll_once(&mut wait).await, "the wait suspends");
        // Keep SDK activity live; invocation cleanup aborts this pending step.
        let _hold = ctx
            .step(|_| std::future::pending::<Result<u32, BoxError>>())
            .name("hold")
            .spawn();
        let _ = tokio::spawn(async move { poll_once(&mut wait).await }).await?;
        after(&ctx).await
    })
    .await;
    assert_refused_at_execute(&result);
    assert_no_outcome(&result, "w");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn handler_callback_result_resumed_on_raw_task_is_refused() {
    // The handler polls the pending callback result until it suspends, so
    // its execution entry check passes on the owner. A raw task then polls
    // it again, and claims nothing.
    let result = run(|(), ctx: DurableContext| async move {
        let cb = ctx.create_callback::<u32>().name("approval").await?;
        let mut pending = Box::pin(cb.result());
        assert!(poll_once(&mut pending).await, "the result suspends");
        // Keep SDK activity live; invocation cleanup aborts this pending step.
        let _hold = ctx
            .step(|_| std::future::pending::<Result<u32, BoxError>>())
            .name("hold")
            .spawn();
        let _ = tokio::spawn(async move { poll_once(&mut pending).await }).await?;
        after(&ctx).await
    })
    .await;
    assert_refused_at_execute(&result);
    assert_no_outcome(&result, "approval");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn child_future_first_polled_on_raw_task_after_scope_close_is_refused() {
    // The child body creates a step future and sends it out unpolled, then
    // returns, so its scope closes. A raw task then polls the future for
    // the first time. The abandoned scope makes the operation pend, so the
    // first poll must check ownership before the operation checks
    // admission.
    let result = run(|(), ctx: DurableContext| async move {
        let (tx, rx) = tokio::sync::oneshot::channel();
        ctx.run_in_child_context(move |child| async move {
            let future = child.step(|_| async { Ok(1_u32) }).name("foreign").future();
            let _ = tx.send(future);
            Ok(0_u32)
        })
        .name("scope")
        .await?;
        let mut future = Box::pin(rx.await?);
        let _ = tokio::spawn(async move { poll_once(&mut future).await }).await?;
        after(&ctx).await
    })
    .await;
    assert_refused_at_execute(&result);
}
