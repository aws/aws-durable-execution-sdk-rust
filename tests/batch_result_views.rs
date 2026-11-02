//! Integration tests for the `BatchResult` views, `into_result` and
//! `spawn_batch`, through the public API only.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{BatchItem, CompletionConfig};
use durable::test_util::LocalRunner;
use durable::{BoxError, ChildContextErrorKind, DurableContext, OperationErrorKind};

fn indexes<'a, O: 'a>(items: impl Iterator<Item = &'a BatchItem<O>>) -> Vec<usize> {
    items.map(|item| item.index).collect()
}

#[tokio::test]
async fn views_and_into_result_are_public() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                let batch = ctx
                    .map(vec![1, 2, 3], |_child, i: i32, idx| async move {
                        if idx == 1 {
                            return Err::<i32, BoxError>("boom".into());
                        }
                        Ok(i)
                    })
                    .completion(CompletionConfig::with_tolerated_failure_count(1))
                    .await_batch()
                    .await?;
                let views = (
                    indexes(batch.succeeded()),
                    indexes(batch.failed()),
                    indexes(batch.started()),
                    batch.started_count(),
                );
                let child_failed = match batch.into_result() {
                    Err(err) => matches!(
                        err.kind(),
                        OperationErrorKind::ChildContext(child)
                            if matches!(child.kind(), ChildContextErrorKind::ChildFailed { .. })
                    ),
                    Ok(_) => false,
                };
                Ok::<_, BoxError>((views.0, views.1, views.2, views.3, child_failed))
            },
            (),
        )
        .await
        .expect("the run finishes");
    assert_eq!(
        result.output(),
        Some(&(vec![0, 2], vec![1], vec![], 0, true))
    );
}

#[tokio::test]
async fn started_items_are_public() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                let batch = ctx
                    .map(vec![0, 1, 2], |child, i: i32, _idx| async move {
                        if i != 0 {
                            child.wait(Duration::from_secs(10)).await?;
                        }
                        Ok(i)
                    })
                    .completion(CompletionConfig::with_min_successful(1))
                    .await_batch()
                    .await?;
                Ok::<_, BoxError>((indexes(batch.started()), batch.started_count()))
            },
            (),
        )
        .await
        .expect("the run finishes");
    assert_eq!(result.output(), Some(&(vec![1, 2], 2)));
}

#[tokio::test]
async fn spawn_batch_is_public() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                let batch = ctx
                    .map(vec![1, 2], |child, i: i32, _idx| async move {
                        Ok(child.step(move |_| async move { Ok(i * 10) }).await?)
                    })
                    .spawn_batch();
                ctx.wait(Duration::from_secs(1)).await?;
                let batch = batch.await?.into_result()?;
                Ok::<_, BoxError>(batch.results().into_iter().copied().collect::<Vec<i32>>())
            },
            (),
        )
        .await
        .expect("the run finishes");
    assert_eq!(result.output(), Some(&vec![10, 20]));
}
