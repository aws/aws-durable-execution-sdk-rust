//! Integration tests for the context usage check.
//!
//! An operation called on the wrong `DurableContext` must fail the
//! execution on the first invocation, with the error type `Error` and a
//! message that names the misuse. Operations called on the right context,
//! including the SDK's own composites inside a child body, must keep
//! working.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{CallbackReply, LocalRunner, TestResult};
use durable::{Branch, DurableContext};

/// Message A: an operation called on a context whose body is not running.
fn message_a(method: &str, calling: &str, active: &str) -> String {
    format!(
        "context usage error in \"{method}\": the operation was called on context \
         \"{calling}\", but the SDK is running the body of context \"{active}\". Inside \
         run_in_child_context, with_retry, map and parallel, call operations on the context \
         passed to the closure."
    )
}

/// Message B: an operation called inside a step body or a condition check.
fn message_b(method: &str, id: &str) -> String {
    format!(
        "context usage error in \"{method}\": the operation was called inside the body of \
         operation \"{id}\". A step body or a wait_for_condition check cannot call durable \
         operations; use run_in_child_context to group operations."
    )
}

/// Asserts the execution failed on the first invocation with the error type
/// `Error` and the expected message.
fn assert_context_usage_failure<O>(result: &TestResult<O>, expected: &str) {
    assert!(
        result.is_failure(),
        "expected FAILED, got error {:?}",
        result.error_message()
    );
    assert_eq!(result.error_type(), Some("Error"));
    assert_eq!(result.error_message(), Some(expected));
    assert_eq!(result.invocation_count(), 1);
}

#[tokio::test]
async fn parent_step_inside_child_body_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let parent = ctx.clone();
                let value = ctx
                    .run_in_child_context(move |_child| async move {
                        let v = parent.step(|_| async { Ok(1_u32) }).name("p").await?;
                        Ok(v)
                    })
                    .name("child")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_context_usage_failure(&result, &message_a("step", "root", "1"));
}

#[tokio::test]
async fn context_used_inside_step_body_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let cloned = ctx.clone();
                let value = ctx
                    .step(move |_| async move {
                        let v = cloned.step(|_| async { Ok(1_u32) }).name("inner").await?;
                        Ok(v)
                    })
                    .name("outer")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_context_usage_failure(&result, &message_b("step", "1"));
}

/// A capture whose `Clone` implementation claims a durable operation.
///
/// The SDK clones a step body before each attempt, and a terminal record
/// replays without cloning. So a claim inside `clone` must fail the
/// execution as a claim inside the body does.
struct CloneCallsWait(DurableContext);

impl Clone for CloneCallsWait {
    fn clone(&self) -> Self {
        drop(self.0.wait(Duration::from_secs(1)).name("in-clone"));
        Self(self.0.clone())
    }
}

#[tokio::test]
async fn context_used_inside_step_capture_clone_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let capture = CloneCallsWait(ctx.clone());
                let value = ctx
                    .step(move |_| async move {
                        let _capture = capture;
                        Ok(1_u32)
                    })
                    .name("outer")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_context_usage_failure(&result, &message_b("wait", "1"));
}

#[tokio::test]
#[expect(clippy::expect_used)] // reason: the queued reply encodes a string, which cannot fail
async fn context_used_inside_submitter_capture_clone_fails() {
    let result = LocalRunner::new()
        .callback(CallbackReply::success(&"approved".to_owned()).expect("encodes"))
        .run(
            |_event: (), ctx: DurableContext| async move {
                let capture = CloneCallsWait(ctx.clone());
                let value: String = ctx
                    .wait_for_callback::<String, _, _>(move |_step_ctx, _callback_id| async move {
                        let _capture = capture;
                        Ok(())
                    })
                    .name("approval")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    // wait_for_callback "1" runs its submitter step as "1-2".
    assert_context_usage_failure(&result, &message_b("wait", "1-2"));
}

#[tokio::test]
async fn context_used_inside_condition_check_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let cloned = ctx.clone();
                let value = ctx
                    .wait_for_condition(
                        move |_check_ctx, state: u32| {
                            let cloned = cloned.clone();
                            async move {
                                let v = cloned.step(|_| async { Ok(1_u32) }).name("inner").await?;
                                Ok(state + v)
                            }
                        },
                        0_u32,
                    )
                    .wait_strategy_fn(|_state, _attempt| {
                        durable::builders::wait_for_condition::WaitDecision::complete()
                    })
                    .name("poll")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_context_usage_failure(&result, &message_b("step", "1"));
}

#[tokio::test]
#[expect(clippy::expect_used)] // reason: the queued reply encodes a string, which cannot fail
async fn context_used_inside_submitter_fails() {
    let result = LocalRunner::new()
        .callback(CallbackReply::success(&"approved".to_owned()).expect("encodes"))
        .run(
            |_event: (), ctx: DurableContext| async move {
                let cloned = ctx.clone();
                let value: String = ctx
                    .wait_for_callback::<String, _, _>(move |_step_ctx, _callback_id| async move {
                        cloned.step(|_| async { Ok(1_u32) }).name("inner").await?;
                        Ok(())
                    })
                    .name("approval")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    // wait_for_callback "1" runs its callback as "1-1" and the submitter
    // step as "1-2".
    assert_context_usage_failure(&result, &message_b("step", "1-2"));
}

#[tokio::test]
async fn parent_step_inside_map_item_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let parent = ctx.clone();
                let values = ctx
                    .map(vec![1_u32], move |_item_ctx, item, _idx| {
                        let parent = parent.clone();
                        async move {
                            let v = parent
                                .step(move |_| async move { Ok(item) })
                                .name("p")
                                .await?;
                            Ok(v)
                        }
                    })
                    .name("items")
                    .await?;
                Ok::<_, durable::BoxError>(values)
            },
            (),
        )
        .await
        .expect("runner");
    // The map is "1". Its items are numbered inside the batch, so the
    // first item's context has the prefix "1-1".
    assert_context_usage_failure(&result, &message_a("step", "root", "1-1"));
}

/// Not in the issue's table: it covers the `run_body` wrapper of a parallel
/// branch, which no other test reaches, because a branch runs on its own
/// task, where no body is active without the wrapper.
#[tokio::test]
async fn parent_step_inside_parallel_branch_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let parent = ctx.clone();
                let branches: Vec<Branch<u32>> =
                    vec![Branch::new("only", move |_branch_ctx| async move {
                        let v = parent.step(|_| async { Ok(1_u32) }).name("p").await?;
                        Ok(v)
                    })];
                let values = ctx.parallel(branches).name("fan-out").await?;
                Ok::<_, durable::BoxError>(values)
            },
            (),
        )
        .await
        .expect("runner");
    // The parallel is "1"; its branch is "1-1", like a map item.
    assert_context_usage_failure(&result, &message_a("step", "root", "1-1"));
}

#[tokio::test]
async fn parent_child_inside_grandchild_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let value = ctx
                    .run_in_child_context(|child| async move {
                        let outer = child.clone();
                        let v = child
                            .run_in_child_context(move |_grandchild| async move {
                                let v = outer.step(|_| async { Ok(1_u32) }).name("p").await?;
                                Ok(v)
                            })
                            .name("grandchild")
                            .await?;
                        Ok(v)
                    })
                    .name("child")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_context_usage_failure(&result, &message_a("step", "1", "1-1"));
}

#[tokio::test]
async fn parent_step_in_child_closure_prelude_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let parent = ctx.clone();
                let value = ctx
                    .run_in_child_context(move |_child| {
                        // Claimed before the closure returns its future.
                        let fut = parent.step(|_| async { Ok(1_u32) }).name("p").future();
                        async move { Ok(fut.await?) }
                    })
                    .name("child")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_context_usage_failure(&result, &message_a("step", "root", "1"));
}

#[tokio::test]
async fn escaped_child_context_at_handler_level_fails() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let slot: Arc<Mutex<Option<DurableContext>>> = Arc::default();
                let store = Arc::clone(&slot);
                ctx.run_in_child_context(move |child| async move {
                    *store
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(child);
                    Ok(())
                })
                .name("child")
                .await?;
                let escaped = slot
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .take()
                    .ok_or("the child stored its context")?;
                let v = escaped.step(|_| async { Ok(1_u32) }).name("p").await?;
                Ok::<_, durable::BoxError>(v)
            },
            (),
        )
        .await
        .expect("runner");
    assert_context_usage_failure(&result, &message_a("step", "1", "root"));
}

#[tokio::test]
async fn correct_contexts_succeed() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let joined = ctx
                    .run_in_child_context(|child| async move {
                        let a = child.step(|_| async { Ok(1_u32) }).name("a").spawn();
                        let b = child.step(|_| async { Ok(2_u32) }).name("b").spawn();
                        let values = child.try_join_all([a, b]).name("join").await?;
                        Ok(values.iter().sum::<u32>())
                    })
                    .name("child")
                    .await?;
                let mapped = ctx
                    .map(vec![1_u32, 2], |item_ctx, item, _idx| async move {
                        let v = item_ctx
                            .step(move |_| async move { Ok(item * 10) })
                            .name("item")
                            .await?;
                        Ok(v)
                    })
                    .name("items")
                    .await?;
                let retried = ctx
                    .with_retry(|attempt_ctx, _attempt| async move {
                        let v = attempt_ctx
                            .step(|_| async { Ok(5_u32) })
                            .name("work")
                            .await?;
                        Ok(v)
                    })
                    .name("retried")
                    .await?;
                let branches: Vec<Branch<u32>> = vec![
                    Branch::new("first", |branch_ctx| async move {
                        let v = branch_ctx.step(|_| async { Ok(7_u32) }).name("s").await?;
                        Ok(v)
                    }),
                    Branch::new("second", |branch_ctx| async move {
                        branch_ctx
                            .wait(Duration::from_secs(1))
                            .name("pause")
                            .await?;
                        Ok(8_u32)
                    }),
                ];
                let parallel = ctx.parallel(branches).name("fan-out").await?;
                Ok::<_, durable::BoxError>(
                    joined + mapped.iter().sum::<u32>() + retried + parallel.iter().sum::<u32>(),
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(
        result.is_success(),
        "expected SUCCEEDED, got error {:?}",
        result.error_message()
    );
    assert_eq!(result.output(), Some(&(3 + 30 + 5 + 15)));
}

#[tokio::test]
#[expect(clippy::expect_used)] // reason: the queued reply encodes a string, which cannot fail
async fn composites_inside_child_body_succeed() {
    let result = LocalRunner::new()
        .callback(CallbackReply::success(&"approved".to_owned()).expect("encodes"))
        .run(
            |_event: (), ctx: DurableContext| async move {
                let value = ctx
                    .run_in_child_context(|child| async move {
                        let approval: String = child
                            .wait_for_callback::<String, _, _>(|_step_ctx, _callback_id| async {
                                Ok(())
                            })
                            .name("approval")
                            .await?;
                        let mapped = child
                            .map(vec![1_u32, 2], |item_ctx, item, _idx| async move {
                                let v = item_ctx
                                    .step(move |_| async move { Ok(item) })
                                    .name("item")
                                    .await?;
                                Ok(v)
                            })
                            .name("items")
                            .await?;
                        Ok(format!("{approval}:{}", mapped.iter().sum::<u32>()))
                    })
                    .name("child")
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert!(
        result.is_success(),
        "expected SUCCEEDED, got error {:?}",
        result.error_message()
    );
    assert_eq!(result.output().map(String::as_str), Some("approved:3"));
}
