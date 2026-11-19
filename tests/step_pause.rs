//! Integration tests for pausing the execution from a retry decision: a
//! step pauses a `LocalRunner` run, the run waits for `resume`, and the
//! next attempt then runs as usual.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::{JitterStrategy, RetryStrategyConfig};
use durable::test_util::{ExecutionStatus, LocalRunner};
use durable::{BoxError, Branch, DurableContext, RetryDecision};

/// How long a test waits to see that a paused run does not finish.
const HELD: Duration = Duration::from_millis(200);

/// `pause_every(2)` pauses after attempt 2. The run waits for `resume`,
/// and attempt 3 then succeeds. The pause does not change the attempt
/// count.
#[tokio::test]
async fn pause_every_pauses_then_resume_succeeds() {
    let runs = Arc::new(AtomicU32::new(0));
    let body_runs = Arc::clone(&runs);
    let mut run = LocalRunner::new().run(
        move |_: serde_json::Value, ctx: DurableContext| {
            let runs = Arc::clone(&body_runs);
            async move {
                let value = ctx
                    .step(move |step_ctx| {
                        let runs = Arc::clone(&runs);
                        async move {
                            runs.fetch_add(1, Ordering::SeqCst);
                            let attempt = step_ctx.attempt();
                            if attempt < 3 {
                                return Err::<u32, BoxError>(format!("down {attempt}").into());
                            }
                            Ok(attempt)
                        }
                    })
                    .name("call-partner")
                    .retry_strategy_config(
                        RetryStrategyConfig::builder()
                            .max_attempts(5)
                            .initial_delay(Duration::from_secs(1))
                            .jitter(JitterStrategy::None)
                            .pause_every(2)
                            .build(),
                    )
                    .await?;
                Ok::<_, BoxError>(value)
            }
        },
        serde_json::json!({}),
    );
    assert!(
        tokio::time::timeout(HELD, &mut run).await.is_err(),
        "the paused run does not finish"
    );
    assert_eq!(runs.load(Ordering::SeqCst), 2, "attempt 3 waits for resume");

    run.resume().await.expect("resume");
    let result = run.await.expect("runner");
    assert_eq!(result.status(), ExecutionStatus::Succeeded);
    assert_eq!(result.output(), Some(&3));
    assert_eq!(runs.load(Ordering::SeqCst), 3);
}

/// A step that pauses in one `parallel` branch does not stop the other
/// branch at once. The other branch's next checkpoint is saved and answered
/// without a token, and the invocation ends there: the branch's next step
/// runs only after the resume.
#[tokio::test]
async fn pause_lets_other_branches_run_to_their_next_checkpoint() {
    let mut run = LocalRunner::new().run(
        |_: serde_json::Value, ctx: DurableContext| async move {
            let branches = vec![
                Branch::new("fails", |child: DurableContext| async move {
                    child
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() == 1 {
                                return Err::<u32, BoxError>("down".into());
                            }
                            Ok(1)
                        })
                        .name("fail-once")
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 3 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::retry(Duration::from_secs(1))
                                    .with_pause(attempt == 1)
                            }
                        })
                        .await
                        .map_err(BoxError::from)
                }),
                Branch::new("other", |child: DurableContext| async move {
                    child
                        .step(|_| async {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            Ok::<u32, BoxError>(2)
                        })
                        .name("slow")
                        .await?;
                    child
                        .step(|_| async { Ok::<u32, BoxError>(3) })
                        .name("after")
                        .await
                        .map_err(BoxError::from)
                }),
            ];
            let values = ctx.parallel(branches).name("branches").await?;
            Ok::<_, BoxError>(values)
        },
        serde_json::json!({}),
    );
    let slow = run.operation("slow");
    let after = run.operation("after");
    assert!(
        tokio::time::timeout(HELD, &mut run).await.is_err(),
        "the paused run does not finish"
    );
    assert!(
        slow.snapshot().is_some_and(|op| op.succeeded()),
        "the other branch's next checkpoint is saved"
    );
    assert!(
        after.snapshot().is_none(),
        "the invocation ended at that checkpoint"
    );

    run.resume().await.expect("resume");
    let result = run.await.expect("runner");
    assert_eq!(result.status(), ExecutionStatus::Succeeded);
    assert_eq!(result.output(), Some(&vec![1, 3]));
}
