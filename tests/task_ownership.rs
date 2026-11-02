//! A durable operation runs on every kind of task the SDK starts.
//!
//! `LocalRunner` creates the context where no task ID exists, as Lambda
//! does in production. So the task-ownership check rejects a durable
//! operation from any task the SDK did not bless. Each test runs a step on
//! one kind of SDK task: a `.spawn()`ed step, a map item, a parallel
//! branch, and the input of each combinator. Each step must succeed, which
//! it does only because the task gateway blesses every operation task
//! before its first poll.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use aws_durable_execution_sdk as durable;
use durable::test_util::{LocalRunner, TestResult};
use durable::{BoxError, Branch, DurableContext, OperationKind, OperationStatus};

/// Asserts that the execution succeeded and that every step succeeded.
fn assert_all_steps_succeeded<O>(result: &TestResult<O>) {
    assert!(result.is_success(), "{:?}", result.error_message());
    let steps: Vec<_> = result
        .operations()
        .iter()
        .filter(|op| op.kind() == OperationKind::Step)
        .collect();
    assert!(!steps.is_empty(), "the handler ran steps");
    for step in steps {
        assert_eq!(step.status(), OperationStatus::Succeeded, "{step:?}");
    }
}

#[tokio::test]
async fn durable_operation_in_spawned_step() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let value = ctx.step(|_| async { Ok(1_u32) }).spawn().await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_all_steps_succeeded(&result);
}

#[tokio::test]
async fn durable_operation_in_map_item() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let values = ctx
                    .map(vec![1_u32, 2], |child, item: u32, _idx| async move {
                        let value = child.step(move |_| async move { Ok(item) }).await?;
                        Ok(value)
                    })
                    .await?;
                Ok::<_, BoxError>(values)
            },
            (),
        )
        .await
        .expect("runner");
    assert_all_steps_succeeded(&result);
}

#[tokio::test]
async fn durable_operation_in_parallel_branch() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let branches = vec![
                    Branch::new("a", |child: DurableContext| async move {
                        Ok(child.step(|_| async { Ok(1_u32) }).await?)
                    }),
                    Branch::new("b", |child: DurableContext| async move {
                        Ok(child.step(|_| async { Ok(2_u32) }).await?)
                    }),
                ];
                let values = ctx.parallel(branches).await?;
                Ok::<_, BoxError>(values)
            },
            (),
        )
        .await
        .expect("runner");
    assert_all_steps_succeeded(&result);
}

#[tokio::test]
async fn durable_operation_in_try_join_all_input() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
                let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
                let values = ctx.try_join_all([a, b]).await?;
                Ok::<_, BoxError>(values)
            },
            (),
        )
        .await
        .expect("runner");
    assert_all_steps_succeeded(&result);
}

#[tokio::test]
async fn durable_operation_in_join_all_input() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
                let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
                let outcomes = ctx.join_all([a, b]).await?;
                let fulfilled = outcomes.iter().filter(|slot| slot.is_ok()).count();
                Ok::<_, BoxError>(fulfilled)
            },
            (),
        )
        .await
        .expect("runner");
    assert_all_steps_succeeded(&result);
    assert_eq!(result.output(), Some(&2), "both slots are fulfilled");
}

#[tokio::test]
async fn durable_operation_in_select_ok_input() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
                let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
                let value = ctx.select_ok([a, b]).await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_all_steps_succeeded(&result);
}

#[tokio::test]
async fn durable_operation_in_race_input() {
    let result = LocalRunner::new()
        .run(
            |(), ctx: DurableContext| async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
                let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
                let value = ctx.race([a, b]).await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_all_steps_succeeded(&result);
}
