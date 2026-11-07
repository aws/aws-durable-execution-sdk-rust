//! How long an operation started with `.spawn()` runs.
//!
//! Dropping the handle of `.spawn()`, `.spawn_batch()` or
//! `Callback::spawn_result()` loses only the result. The operation runs
//! until it settles, until its scope closes, or until the drive ends.
//!
//! A combinator stops the `.future()` inputs it started, and lets an input
//! started with `.spawn()` run. A scope closes when a child context's body
//! returns, when a `with_retry` block with its own record ends, when an
//! attempt of any `with_retry` block ends (a block with
//! `.without_child_context()` has no scope of its own, only its attempts
//! do), and when a batch item or the whole batch completes. A closed scope stops the
//! tracked tasks still running inside it.
//!
//! The scope tests share [`escape`]. It starts a step named `escaped`
//! whose body sleeps 50 ms and then adds 1 to `effects`. It keeps the
//! handle in a holder the handler owns, so the handle outlives the scope.
//! A `keep` step then holds the drive open long enough for an escaped
//! body to finish if nothing stops it.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::panic)] // reason: test assertions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{CompletionConfig, NestingMode};
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::test_util::{LocalRunner, TestResult};
use durable::{BoxError, BranchSet, DurableContext, DurableFuture, OperationStatus, RetryDecision};

/// The handles [`escape`] keeps alive beyond the scope that started them.
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

/// Runs `handler` with the unit event and returns the result.
async fn run<O, F, Fut>(handler: F) -> TestResult<O>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    LocalRunner::new()
        .run(handler, ())
        .await
        .expect("the run finishes")
}

/// The status of the operation named `name`, or `None` when nothing is
/// recorded under that name.
fn status<O>(result: &TestResult<O>, name: &str) -> Option<OperationStatus> {
    result
        .operation(name)
        .map(durable::test_util::TestOperation::status)
}

/// Starts the step `escaped` on `ctx`, keeps its handle in `holder`, and
/// sleeps 10 ms, so the step body is at its sleep when the caller's scope
/// closes.
async fn escape(ctx: &DurableContext, effects: &Arc<AtomicUsize>, holder: &Holder) {
    let effects = Arc::clone(effects);
    let handle = ctx
        .step(move |_| async move {
            tokio::time::sleep(ms(50)).await;
            effects.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .name("escaped")
        .spawn();
    holder.lock().expect("lock").push(handle);
    tokio::time::sleep(ms(10)).await;
}

/// A step named `name` whose body sleeps `duration`.
async fn sleeping_step(
    ctx: &DurableContext,
    name: &str,
    duration: Duration,
) -> Result<(), BoxError> {
    ctx.step(move |_| async move {
        tokio::time::sleep(duration).await;
        Ok(())
    })
    .name(name)
    .await?;
    Ok(())
}

/// The step `s1` and the wait `tail` that follow a dropped handle.
async fn s1_and_tail(ctx: &DurableContext) -> Result<(), BoxError> {
    ctx.step(|_| async { Ok(2_u32) }).name("s1").await?;
    ctx.wait(secs(1)).name("tail").await?;
    Ok(())
}

/// A step named `slow` whose body sleeps 100 ms, adds 1 to `runs` and
/// returns 2. Started with `.spawn()` when `spawn` is true, and made with
/// `.future()` otherwise.
fn slow_step(ctx: &DurableContext, runs: &Arc<AtomicUsize>, spawn: bool) -> DurableFuture<u32> {
    let runs = Arc::clone(runs);
    let builder = ctx
        .step(move |_| async move {
            tokio::time::sleep(ms(100)).await;
            runs.fetch_add(1, Ordering::SeqCst);
            Ok(2_u32)
        })
        .name("slow");
    if spawn {
        builder.spawn()
    } else {
        builder.future()
    }
}

/// A `.future()` step named `fast` that fails at once and does not retry.
fn failing_fast(ctx: &DurableContext) -> DurableFuture<u32> {
    ctx.step(|_| async { Err::<u32, BoxError>("fast failed".into()) })
        .name("fast")
        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
        .future()
}

// ----- Dropped handles -----

#[tokio::test(start_paused = true)]
async fn dropped_spawn_handle_step_succeeds() {
    let result = run(|(), ctx: DurableContext| async move {
        drop(ctx.step(|_| async { Ok(1_u32) }).name("a").spawn());
        s1_and_tail(&ctx).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        status(&result, "a"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

#[tokio::test(start_paused = true)]
async fn dropped_combinator_spawn_handle_succeeds() {
    let result = run(|(), ctx: DurableContext| async move {
        let x = ctx.step(|_| async { Ok(1_u32) }).name("x").future();
        let y = ctx.step(|_| async { Ok(2_u32) }).name("y").future();
        drop(ctx.race([x, y]).name("r").spawn());
        s1_and_tail(&ctx).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        status(&result, "r"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

#[tokio::test(start_paused = true)]
async fn dropped_spawn_batch_handle_succeeds() {
    let result = run(|(), ctx: DurableContext| async move {
        drop(
            ctx.map(vec![1_u32, 2], |child, item, _idx| async move {
                Ok(child
                    .step(move |_| async move { Ok(item * 10) })
                    .name("scale")
                    .await?)
            })
            .name("m")
            .spawn_batch(),
        );
        s1_and_tail(&ctx).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        status(&result, "m"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

// ----- Combinator inputs -----

#[tokio::test(start_paused = true)]
async fn race_does_not_stop_spawned_input() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let slow = slow_step(&ctx, &runs, true);
            let fast = ctx.step(|_| async { Ok(1_u32) }).name("fast").future();
            let winner = ctx.race([fast, slow]).name("r").await?;
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(winner)
        }
    })
    .await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_eq!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 1);
}

#[tokio::test(start_paused = true)]
async fn race_stops_future_input() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let slow = slow_step(&ctx, &runs, false);
            let fast = ctx.step(|_| async { Ok(1_u32) }).name("fast").future();
            let winner = ctx.race([fast, slow]).name("r").await?;
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(winner)
        }
    })
    .await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_ne!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 0);
}

#[tokio::test(start_paused = true)]
async fn select_ok_does_not_stop_spawned_input() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let slow = slow_step(&ctx, &runs, true);
            let fast = ctx.step(|_| async { Ok(1_u32) }).name("fast").future();
            let winner = ctx.select_ok([fast, slow]).name("s").await?;
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(winner)
        }
    })
    .await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_eq!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 1);
}

#[tokio::test(start_paused = true)]
async fn try_join_all_does_not_stop_spawned_input() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let slow = slow_step(&ctx, &runs, true);
            let fast = failing_fast(&ctx);
            let joined = ctx.try_join_all([fast, slow]).name("j").await;
            assert!(joined.is_err(), "the failing input fails the join");
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(())
        }
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 1);
}

#[tokio::test(start_paused = true)]
async fn select_ok_stops_future_input() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let slow = slow_step(&ctx, &runs, false);
            let fast = ctx.step(|_| async { Ok(1_u32) }).name("fast").future();
            let winner = ctx.select_ok([fast, slow]).name("s").await?;
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(winner)
        }
    })
    .await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_ne!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 0);
}

#[tokio::test(start_paused = true)]
async fn try_join_all_stops_future_input() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let slow = slow_step(&ctx, &runs, false);
            let fast = failing_fast(&ctx);
            let joined = ctx.try_join_all([fast, slow]).name("j").await;
            assert!(joined.is_err(), "the failing input fails the join");
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(())
        }
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_ne!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 0);
}

/// Runs the race of test 11 or 12 on a two-worker runtime. `slow` signals
/// `started` and waits for `release`; `fast` waits for `started` and wins.
async fn multi_thread_race(spawn_slow: bool) -> (TestResult<u32>, usize) {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let run = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let (slow_started, slow_release) = (Arc::clone(&started), Arc::clone(&release));
            let slow = ctx
                .step(move |_| async move {
                    slow_started.notify_one();
                    slow_release.notified().await;
                    runs.fetch_add(1, Ordering::SeqCst);
                    Ok(2_u32)
                })
                .name("slow");
            let slow = if spawn_slow {
                slow.spawn()
            } else {
                slow.future()
            };
            let fast_started = Arc::clone(&started);
            let fast = ctx
                .step(move |_| async move {
                    fast_started.notified().await;
                    Ok(1_u32)
                })
                .name("fast")
                .future();
            let winner = ctx.race([fast, slow]).name("r").await?;
            release.notify_one();
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(winner)
        }
    });
    let result = tokio::time::timeout(secs(10), run)
        .await
        .expect("the run finishes within 10 s");
    (result, count(&runs))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn race_does_not_stop_spawned_input_on_multi_thread() {
    let (result, runs) = multi_thread_race(true).await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_eq!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(runs, 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn race_stops_future_input_on_multi_thread() {
    let (result, runs) = multi_thread_race(false).await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_ne!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(runs, 0);
}

#[tokio::test(start_paused = true)]
async fn join_all_waits_for_every_input() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let outcomes = Arc::new(Mutex::new(None));
    let handler_outcomes = Arc::clone(&outcomes);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        let outcomes = Arc::clone(&handler_outcomes);
        async move {
            let slow = slow_step(&ctx, &runs, false);
            let fast = failing_fast(&ctx);
            let joined = ctx.join_all([fast, slow]).name("j").await?;
            *outcomes.lock().expect("lock") = Some(
                joined
                    .iter()
                    .map(|outcome| outcome.as_ref().ok().copied())
                    .collect::<Vec<_>>(),
            );
            Ok(())
        }
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        outcomes.lock().expect("lock").clone(),
        Some(vec![None, Some(2)])
    );
    assert_eq!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 1);
}

#[tokio::test(start_paused = true)]
async fn race_replay_children_does_not_stop_spawned_input() {
    let second_attempts = counter();
    let handler_attempts = Arc::clone(&second_attempts);
    let result = run(move |(), ctx: DurableContext| {
        let attempts = Arc::clone(&handler_attempts);
        async move {
            let slow = ctx
                .step(move |step_ctx| async move {
                    if step_ctx.attempt() == 1 {
                        return Err::<String, BoxError>("first attempt fails".into());
                    }
                    attempts.fetch_add(1, Ordering::SeqCst);
                    Ok("slow".to_owned())
                })
                .name("slow")
                .retry_strategy(|_err, attempt| {
                    if attempt >= 2 {
                        RetryDecision::Stop
                    } else {
                        RetryDecision::Retry { delay: secs(2) }
                    }
                })
                .spawn();
            // A value larger than the checkpoint limit, so the child and the
            // race record `ReplayChildren`.
            let fast = ctx
                .run_in_child_context(|_child| async { Ok("x".repeat(300 * 1024)) })
                .name("fast")
                .future();
            let winner = ctx.race([fast, slow]).name("r").await?;
            assert_eq!(winner.len(), 300 * 1024, "the child wins");
            ctx.wait(secs(1)).name("w1").await?;
            ctx.wait(secs(2)).name("w2").await?;
            Ok(())
        }
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&second_attempts), 1);
}

#[tokio::test(start_paused = true)]
async fn race_stops_spawned_nested_combinator_in_future_child() {
    let started = counter();
    let runs = counter();
    let (handler_started, handler_runs) = (Arc::clone(&started), Arc::clone(&runs));
    let result = run(move |(), ctx: DurableContext| {
        let (started, runs) = (Arc::clone(&handler_started), Arc::clone(&handler_runs));
        async move {
            let fast = ctx
                .step(|_| async {
                    tokio::time::sleep(ms(20)).await;
                    Ok(1_u32)
                })
                .name("fast")
                .future();
            let c = ctx
                .run_in_child_context(move |child| async move {
                    let slow = child
                        .step(move |_| async move {
                            started.fetch_add(1, Ordering::SeqCst);
                            tokio::time::sleep(ms(100)).await;
                            runs.fetch_add(1, Ordering::SeqCst);
                            Ok(2_u32)
                        })
                        .name("slow")
                        .future();
                    let inner = child.race([slow]).name("inner").spawn();
                    Ok(inner.await?)
                })
                .name("c")
                .future();
            let winner = ctx.race([fast, c]).name("outer").await?;
            sleeping_step(&ctx, "keep", ms(200)).await?;
            Ok(winner)
        }
    })
    .await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_eq!(count(&started), 1);
    assert_ne!(
        status(&result, "inner"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_ne!(
        status(&result, "slow"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 0);
}

// ----- Closed scopes -----

/// Runs `body` as a handler that owns a fresh holder and the shared
/// `effects` counter, and returns the result with the final count.
async fn run_scoped<O, F, Fut>(body: F) -> (TestResult<O>, usize)
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: Fn(DurableContext, Arc<AtomicUsize>, Holder) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
{
    let effects = counter();
    let handler_effects = Arc::clone(&effects);
    let body = Arc::new(body);
    let result = run(move |(), ctx: DurableContext| {
        let holder: Holder = Arc::new(Mutex::new(Vec::new()));
        let fut = body(ctx, Arc::clone(&handler_effects), Arc::clone(&holder));
        async move {
            let output = fut.await;
            drop(holder);
            output
        }
    })
    .await;
    (result, count(&effects))
}

#[tokio::test(start_paused = true)]
async fn child_return_stops_escaped_task() {
    let (result, effects) = run_scoped(|ctx, effects, holder| async move {
        ctx.run_in_child_context(move |child| async move {
            escape(&child, &effects, &holder).await;
            Ok(())
        })
        .name("child")
        .await?;
        sleeping_step(&ctx, "keep", ms(200)).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(effects, 0, "{result}");
    assert_ne!(
        status(&result, "escaped"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

#[tokio::test(start_paused = true)]
async fn with_retry_attempt_end_stops_escaped_task() {
    let attempts = counter();
    let (result, effects) = run_scoped(move |ctx, effects, holder| {
        let attempts = Arc::clone(&attempts);
        async move {
            ctx.with_retry(move |attempt_ctx, _attempt| {
                let (effects, holder, attempts) = (
                    Arc::clone(&effects),
                    Arc::clone(&holder),
                    Arc::clone(&attempts),
                );
                async move {
                    if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                        escape(&attempt_ctx, &effects, &holder).await;
                        return Err::<(), BoxError>("first attempt fails".into());
                    }
                    Ok(())
                }
            })
            .name("block")
            .retry_strategy(|_err, attempt| {
                if attempt >= 2 {
                    RetryDecision::Stop
                } else {
                    RetryDecision::Retry { delay: secs(1) }
                }
            })
            .await?;
            Ok(())
        }
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(effects, 0, "{result}");
    assert_ne!(
        status(&result, "escaped"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

/// The map of tests 17 and 18: item 0 escapes and returns, item 1 keeps
/// the batch open for 200 ms.
async fn item_end_run(nesting: NestingMode) -> (TestResult<()>, usize) {
    run_scoped(move |ctx, effects, holder| async move {
        ctx.map(vec![0_u32, 1], move |child, item, _idx| {
            let (effects, holder) = (Arc::clone(&effects), Arc::clone(&holder));
            async move {
                if item == 0 {
                    escape(&child, &effects, &holder).await;
                } else {
                    sleeping_step(&child, "busy", ms(200)).await?;
                }
                Ok(())
            }
        })
        .name("m")
        .nesting(nesting)
        .await?;
        sleeping_step(&ctx, "keep", ms(200)).await
    })
    .await
}

#[tokio::test(start_paused = true)]
async fn batch_item_end_stops_escaped_task_before_batch_end() {
    let (result, effects) = item_end_run(NestingMode::Normal).await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(effects, 0, "{result}");
}

#[tokio::test(start_paused = true)]
async fn flat_item_end_stops_escaped_task_before_batch_end() {
    let (result, effects) = item_end_run(NestingMode::Flat).await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(effects, 0, "{result}");
}

/// The map of tests 19 and 20: item 0 escapes and returns, item 1 runs a
/// 1 s step, and the batch completes at its first success.
async fn early_settled_run(nesting: NestingMode) -> (TestResult<()>, usize) {
    run_scoped(move |ctx, effects, holder| async move {
        ctx.map(vec![0_u32, 1], move |child, item, _idx| {
            let (effects, holder) = (Arc::clone(&effects), Arc::clone(&holder));
            async move {
                if item == 0 {
                    escape(&child, &effects, &holder).await;
                } else {
                    sleeping_step(&child, "busy", secs(1)).await?;
                }
                Ok(())
            }
        })
        .name("m")
        .nesting(nesting)
        .completion(CompletionConfig::with_min_successful(1))
        .await?;
        sleeping_step(&ctx, "keep", ms(200)).await
    })
    .await
}

#[tokio::test(start_paused = true)]
async fn early_batch_completion_stops_settled_items_escaped_task() {
    let (result, effects) = early_settled_run(NestingMode::Normal).await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(effects, 0, "{result}");
}

#[tokio::test(start_paused = true)]
async fn early_batch_completion_stops_escaped_task_of_flat_item() {
    let (result, effects) = early_settled_run(NestingMode::Flat).await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(effects, 0, "{result}");
}

#[tokio::test(start_paused = true)]
async fn early_batch_completion_stops_running_items_nested_task() {
    let (result, effects) = run_scoped(|ctx, effects, holder| async move {
        ctx.map(vec![0_u32, 1], move |child, item, _idx| {
            let (effects, holder) = (Arc::clone(&effects), Arc::clone(&holder));
            async move {
                if item == 0 {
                    escape(&child, &effects, &holder).await;
                    sleeping_step(&child, "busy", secs(1)).await?;
                } else {
                    sleeping_step(&child, "quick", ms(30)).await?;
                }
                Ok(())
            }
        })
        .name("m")
        .completion(CompletionConfig::with_min_successful(1))
        .await?;
        sleeping_step(&ctx, "keep", ms(200)).await
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(effects, 0, "{result}");
}

#[tokio::test(start_paused = true)]
async fn parallel_mixed_completion_stops_escaped_task() {
    let (result, effects) = run_scoped(|ctx, effects, holder| async move {
        let mut branches = BranchSet::new();
        let a = branches.add("a", move |child| async move {
            escape(&child, &effects, &holder).await;
            Ok(1_u32)
        });
        let _b = branches.add("b", |child| async move {
            sleeping_step(&child, "busy", ms(200)).await?;
            Ok(0_u32)
        });
        ctx.parallel_mixed(branches).name("pm").await?;
        let value = a.into_result()?;
        sleeping_step(&ctx, "keep", ms(200)).await?;
        Ok(value)
    })
    .await;
    assert_eq!(result.output(), Some(&1), "{result}");
    assert_eq!(effects, 0, "{result}");
}

// ----- Suspension, drive end, errors and panics -----

#[tokio::test(start_paused = true)]
async fn detached_running_step_holds_the_invocation() {
    let runs = counter();
    let handler_runs = Arc::clone(&runs);
    let result = run(move |(), ctx: DurableContext| {
        let runs = Arc::clone(&handler_runs);
        async move {
            drop(
                ctx.step(move |_| async move {
                    runs.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(secs(2)).await;
                    Ok(())
                })
                .name("long")
                .spawn(),
            );
            ctx.wait(secs(3)).name("tail").await?;
            Ok(())
        }
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        status(&result, "long"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
    assert_eq!(count(&runs), 1);
}

#[tokio::test(start_paused = true)]
async fn detached_service_wait_lets_the_invocation_suspend() {
    let result = run(|(), ctx: DurableContext| async move {
        drop(ctx.wait(secs(60)).name("long").spawn());
        ctx.wait(secs(1)).name("tail").await?;
        Ok(())
    })
    .await;
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 2, "{result}");
    assert_eq!(
        status(&result, "tail"),
        Some(OperationStatus::Succeeded),
        "{result}"
    );
}

/// Counts every `on_operation_end` call. In `drive_end_stops_detached_work`
/// the step `late` is the handler's only operation, so the count is the
/// number of end hooks `late` fired.
struct EndHookCounter {
    ends: Arc<AtomicUsize>,
}

impl Plugin for EndHookCounter {
    fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
        self.ends.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }
}

/// Gives each invocation an [`EndHookCounter`] that shares one count.
struct EndHookCounterFactory {
    ends: Arc<AtomicUsize>,
}

impl PluginFactory for EndHookCounterFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(EndHookCounter {
            ends: Arc::clone(&self.ends),
        })))
    }
}

/// The handler drops the handle of the running step `late` and returns, so
/// the end of the invocation stops the step: its body has no effect, its
/// record stays `STARTED`, its end hook never fires, and the execution
/// ends with the handler's result.
#[tokio::test(start_paused = true)]
async fn drive_end_stops_detached_work() {
    let effects = counter();
    let late_end_hooks = counter();
    let options = durable::Options::builder()
        .plugin(EndHookCounterFactory {
            ends: Arc::clone(&late_end_hooks),
        })
        .build()
        .expect("valid options");
    let handler_effects = Arc::clone(&effects);
    let result = LocalRunner::new()
        .options(options)
        .run(
            move |(), ctx: DurableContext| {
                let effects = Arc::clone(&handler_effects);
                async move {
                    let started = Arc::new(tokio::sync::Notify::new());
                    let body_started = Arc::clone(&started);
                    drop(
                        ctx.step(move |_| async move {
                            body_started.notify_one();
                            tokio::time::sleep(secs(5)).await;
                            effects.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        })
                        .name("late")
                        .spawn(),
                    );
                    started.notified().await;
                    Ok(7_u32)
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    tokio::time::sleep(secs(10)).await;
    assert_eq!(result.output(), Some(&7_u32), "{result}");
    assert_eq!(count(&effects), 0, "{result}");
    assert_eq!(
        status(&result, "late"),
        Some(OperationStatus::Started),
        "{result}"
    );
    assert_eq!(count(&late_end_hooks), 0, "{result}");
}

#[tokio::test(start_paused = true)]
async fn detached_error_stays_in_its_record() {
    let result = run(|(), ctx: DurableContext| async move {
        drop(
            ctx.step(|_| async { Err::<u32, BoxError>("bad".into()) })
                .name("bad")
                .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                .spawn(),
        );
        s1_and_tail(&ctx).await?;
        Ok(7_u32)
    })
    .await;
    assert_eq!(result.output(), Some(&7), "{result}");
    assert_eq!(
        status(&result, "bad"),
        Some(OperationStatus::Failed),
        "{result}"
    );
}

#[tokio::test(start_paused = true)]
async fn detached_panic_fails_the_invocation() {
    #[expect(clippy::disallowed_methods)]
    // reason: the test observes the panic that the run resumes, which only
    // a task's JoinHandle reports.
    let handle = tokio::spawn(async {
        run(|(), ctx: DurableContext| async move {
            drop(
                ctx.step(|_| async {
                    panic!("detached body panics");
                    #[expect(unreachable_code)]
                    // reason: the closure's return type comes from this line
                    Ok::<u32, BoxError>(0)
                })
                .name("boom")
                .spawn(),
            );
            s1_and_tail(&ctx).await
        })
        .await
    });
    let joined = handle.await;
    assert!(
        joined.as_ref().is_err_and(tokio::task::JoinError::is_panic),
        "the detached panic fails the invocation"
    );
}
