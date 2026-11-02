//! A batch item whose serdes writes an empty string keeps its value across
//! replay, and the batch record holds each item value in JS's shape.
//!
//! Each handler runs one batch, records what it returned, then waits 1 s.
//! So the batch runs live in the first invocation and replays from its
//! record in the second.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::BatchItemStatus;
use durable::serdes::SerdesContext;
use durable::test_util::LocalRunner;
use durable::{Branch, Serdes};

/// A serdes for `()` that writes the empty string, and reads it back.
struct EmptyUnitSerdes;

impl Serdes<()> for EmptyUnitSerdes {
    fn serialize(
        &self,
        _value: (),
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, durable::BoxError>> + Send {
        std::future::ready(Ok(String::new()))
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<(), durable::BoxError>> + Send {
        std::future::ready(if wire.is_empty() {
            Ok(())
        } else {
            Err(format!("expected an empty encoding, got {wire:?}").into())
        })
    }
}

/// What each invocation saw, one entry per invocation.
type Seen<T> = Arc<Mutex<Vec<T>>>;

/// Runs `handler` through `LocalRunner`, asserts the execution succeeded,
/// and returns the recorded payload of the batch named `b`.
async fn run<F, Fut>(handler: F) -> String
where
    F: Fn(serde_json::Value, durable::DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), durable::BoxError>> + Send + 'static,
{
    let result = LocalRunner::new()
        .run(handler, serde_json::Value::Null)
        .await
        .expect("runner");
    assert!(result.is_success(), "the execution succeeds: {result}");
    result
        .operation("b")
        .and_then(durable::test_util::TestOperation::result)
        .expect("the batch records a payload")
        .to_owned()
}

#[tokio::test]
async fn map_empty_item_encoding_replays_its_value() {
    let seen: Seen<usize> = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let payload = run(move |_event, ctx| {
        let seen = Arc::clone(&handler_seen);
        async move {
            let values = ctx
                .map(
                    vec![1_u32, 2_u32],
                    |_child, _item, _idx| async move { Ok(()) },
                )
                .serdes(EmptyUnitSerdes)
                .name("b")
                .await?;
            seen.lock().expect("lock").push(values.len());
            ctx.wait(Duration::from_secs(1)).name("after").await?;
            Ok(())
        }
    })
    .await;
    assert_eq!(
        *seen.lock().expect("lock"),
        [2, 2],
        "the first run and the replay both return two values"
    );
    assert_eq!(
        payload.matches(r#""result":"""#).count(),
        2,
        "payload: {payload}"
    );
}

#[tokio::test]
async fn parallel_empty_branch_encoding_replays_its_value() {
    let seen: Seen<usize> = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let payload = run(move |_event, ctx| {
        let seen = Arc::clone(&handler_seen);
        async move {
            let branches = vec![
                Branch::new("x", |_child| async move { Ok(()) }),
                Branch::new("y", |_child| async move { Ok(()) }),
            ];
            let values = ctx
                .parallel(branches)
                .serdes(EmptyUnitSerdes)
                .name("b")
                .await?;
            seen.lock().expect("lock").push(values.len());
            ctx.wait(Duration::from_secs(1)).name("after").await?;
            Ok(())
        }
    })
    .await;
    assert_eq!(
        *seen.lock().expect("lock"),
        [2, 2],
        "the first run and the replay both return two values"
    );
    assert_eq!(
        payload.matches(r#""result":"""#).count(),
        2,
        "payload: {payload}"
    );
}

#[tokio::test]
async fn await_batch_reports_the_empty_encoded_value() {
    let seen: Seen<Vec<(BatchItemStatus, Option<()>)>> = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    run(move |_event, ctx| {
        let seen = Arc::clone(&handler_seen);
        async move {
            let batch = ctx
                .map(
                    vec![1_u32, 2_u32],
                    |_child, _item, _idx| async move { Ok(()) },
                )
                .serdes(EmptyUnitSerdes)
                .name("b")
                .await_batch()
                .await?;
            let items = batch
                .items
                .iter()
                .map(|item| (item.status, item.result))
                .collect();
            seen.lock().expect("lock").push(items);
            ctx.wait(Duration::from_secs(1)).name("after").await?;
            Ok(())
        }
    })
    .await;
    let expected = vec![
        (BatchItemStatus::Succeeded, Some(())),
        (BatchItemStatus::Succeeded, Some(())),
    ];
    assert_eq!(
        *seen.lock().expect("lock"),
        [expected.clone(), expected],
        "the first run and the replay report both values"
    );
}

#[tokio::test]
async fn map_unit_values_replay() {
    let seen: Seen<usize> = Arc::new(Mutex::new(Vec::new()));
    let handler_seen = Arc::clone(&seen);
    let payload = run(move |_event, ctx| {
        let seen = Arc::clone(&handler_seen);
        async move {
            let values: Vec<()> = ctx
                .map(
                    vec![1_u32, 2_u32],
                    |_child, _item, _idx| async move { Ok(()) },
                )
                .name("b")
                .await?;
            seen.lock().expect("lock").push(values.len());
            ctx.wait(Duration::from_secs(1)).name("after").await?;
            Ok(())
        }
    })
    .await;
    assert_eq!(
        *seen.lock().expect("lock"),
        [2, 2],
        "the first run and the replay both return two values"
    );
    assert_eq!(
        payload.matches(r#""result":null"#).count(),
        2,
        "payload: {payload}"
    );
}
