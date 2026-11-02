//! Public-API tests for the refresh scheduler and `Callback::spawn_result`
//! (issue #104).
//!
//! `LocalRunner` moves its clock only between invocations. So a refresh it
//! answers during an invocation finds no change, and a waiting operation
//! still resumes on replay. These tests check that the refreshes sent
//! beside local work change no result and no recorded operation, and that
//! a callback result started with `spawn_result()` survives a suspension.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{CallbackReply, LocalRunner};
use durable::{OperationKind, OperationStatus};

/// A callback result started with `spawn_result()` is polled once, so the
/// scheduler refreshes it while a 3 s step runs. The callback completes
/// only between invocations. The execution suspends on the handle, and
/// replay delivers the payload to a new handle. The step body ran once.
#[tokio::test(start_paused = true)]
async fn spawned_callback_result_resolves_after_replay() {
    let step_runs = Arc::new(AtomicUsize::new(0));
    let handler_runs = Arc::clone(&step_runs);
    let result = LocalRunner::new()
        .callback_for(
            "approval",
            CallbackReply::success(&"ok".to_owned()).expect("serializable"),
        )
        .run(
            move |(), ctx: durable::DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move {
                    let cb = ctx.create_callback::<String>().name("approval").await?;
                    let handle = cb.spawn_result();
                    let mut handle = std::pin::pin!(handle);
                    // Poll the handle once, so its registration is refreshed.
                    // On replay the callback is settled, and that poll returns
                    // its value.
                    let first =
                        std::future::poll_fn(|cx| std::task::Poll::Ready(handle.as_mut().poll(cx)))
                            .await;
                    ctx.step(move |_| async move {
                        runs.fetch_add(1, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        Ok(())
                    })
                    .name("work")
                    .await?;
                    let value = match first {
                        std::task::Poll::Ready(value) => value?,
                        std::task::Poll::Pending => handle.await?,
                    };
                    Ok::<_, durable::BoxError>(value)
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output().map(String::as_str), Some("ok"));
    assert!(
        result.invocation_count() >= 2,
        "the callback completes only between invocations, got {}",
        result.invocation_count()
    );
    assert_eq!(step_runs.load(Ordering::SeqCst), 1, "replay ran the step");
    let approval = result.operation("approval").expect("the callback");
    assert_eq!(approval.kind(), OperationKind::Callback);
    assert_eq!(approval.status(), OperationStatus::Succeeded);
    assert_eq!(
        result
            .operation("work")
            .map(durable::test_util::TestOperation::status),
        Some(OperationStatus::Succeeded)
    );
}

/// `parallel` of a branch that waits 5 s and then runs a step, beside a
/// branch whose step runs 60 s. The wait's registration is refreshed
/// while the long step runs. Every operation succeeds, each step body ran
/// once, and the result is the same as without the refreshes.
#[tokio::test(start_paused = true)]
async fn parallel_wait_beside_long_step_succeeds_across_replay() {
    let step_runs = Arc::new(AtomicUsize::new(0));
    let handler_runs = Arc::clone(&step_runs);
    let result = LocalRunner::new()
        .run(
            move |(), ctx: durable::DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move {
                    let runs_a = Arc::clone(&runs);
                    let runs_b = Arc::clone(&runs);
                    let branches: Vec<durable::Branch<u32>> = vec![
                        durable::Branch::new("a", move |child| async move {
                            child.wait(Duration::from_secs(5)).name("pause").await?;
                            let value = child
                                .step(move |_| async move {
                                    runs_a.fetch_add(1, Ordering::SeqCst);
                                    Ok(1_u32)
                                })
                                .name("after-pause")
                                .await?;
                            Ok(value)
                        }),
                        durable::Branch::new("b", move |child| async move {
                            let value = child
                                .step(move |_| async move {
                                    runs_b.fetch_add(1, Ordering::SeqCst);
                                    tokio::time::sleep(Duration::from_mins(1)).await;
                                    Ok(2_u32)
                                })
                                .name("long")
                                .await?;
                            Ok(value)
                        }),
                    ];
                    let values = ctx.parallel(branches).name("fan-out").await?;
                    Ok::<_, durable::BoxError>(values.iter().sum::<u32>())
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output(), Some(&3));
    assert_eq!(
        step_runs.load(Ordering::SeqCst),
        2,
        "each step body ran once"
    );
    for name in ["fan-out", "pause", "after-pause", "long"] {
        assert_eq!(
            result
                .operation(name)
                .map(durable::test_util::TestOperation::status),
            Some(OperationStatus::Succeeded),
            "{name}"
        );
    }
}

/// The refresh is a checkpoint call of its own. The handler starts a
/// callback (call 1), polls its spawned result and starts a 3 s step
/// (call 2). The refresh at 1 s is call 3, and the runner rejects it
/// non-retryably. The failure fails the execution while the step body
/// still sleeps, so the body never finishes and the step's outcome is
/// never recorded. Without the refresh, call 3 would be the step's
/// outcome, which runs after the body finished.
#[tokio::test(start_paused = true)]
async fn rejected_refresh_fails_the_execution_while_local_work_runs() {
    let finished = Arc::new(AtomicUsize::new(0));
    let handler_finished = Arc::clone(&finished);
    let result = LocalRunner::new()
        .fail_checkpoints_after(2, 1)
        .run(
            move |(), ctx: durable::DurableContext| {
                let finished = Arc::clone(&handler_finished);
                async move {
                    let cb = ctx.create_callback::<String>().name("approval").await?;
                    let mut handle = std::pin::pin!(cb.spawn_result());
                    let _ =
                        std::future::poll_fn(|cx| std::task::Poll::Ready(handle.as_mut().poll(cx)))
                            .await;
                    ctx.step(move |_| async move {
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        finished.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    })
                    .name("work")
                    .await?;
                    Ok::<_, durable::BoxError>(handle.await?)
                }
            },
            (),
        )
        .await
        .expect("runner");

    assert!(
        !result.is_success(),
        "the rejected refresh fails the execution"
    );
    assert_eq!(result.invocation_count(), 1);
    assert_eq!(
        finished.load(Ordering::SeqCst),
        0,
        "the step body was cut off"
    );
    assert_eq!(
        result
            .operation("work")
            .map(durable::test_util::TestOperation::status),
        Some(OperationStatus::Started)
    );
}
