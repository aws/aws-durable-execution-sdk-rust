//! A failure that propagates another operation's error records that
//! operation's recorded message, as JS does: a step that fails with
//! `bad item` makes every operation that propagates its error, and the
//! execution, record `bad item`.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::unwrap_used, clippy::panic)] // reason: test assertions

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{CompletionConfig, NestingMode};
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::test_util::{LocalRunner, TestResult};
use durable::{
    BoxError, Branch, BranchSet, DurableContext, OperationError, OperationKind, Options,
    RetryDecision, WireError,
};

const MESSAGE: &str = "bad item";

/// The step every case propagates: it fails with [`MESSAGE`], and the
/// retry strategy stops.
async fn failing_step(ctx: &DurableContext) -> Result<u32, OperationError> {
    ctx.step(|_| async { Err::<u32, BoxError>(MESSAGE.into()) })
        .name("bad")
        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
        .await
}

/// The failed step's result, held across a one-second wait, so a later
/// invocation propagates the error rebuilt from the step's record.
async fn failing_step_after_wait(ctx: &DurableContext) -> Result<u32, BoxError> {
    let failed = failing_step(ctx).await;
    ctx.wait(Duration::from_secs(1)).name("pause").await?;
    Ok(failed?)
}

/// Runs `handler` once with `options`.
async fn run_with<O, F, Fut>(options: Options, handler: F) -> TestResult<O>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: Fn(DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
{
    LocalRunner::new()
        .options(options)
        .run(move |(), ctx: DurableContext| handler(ctx), ())
        .await
        .expect("runner")
}

/// Runs `handler` once.
async fn run<O, F, Fut>(handler: F) -> TestResult<O>
where
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
    F: Fn(DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
{
    run_with(Options::default(), handler).await
}

/// Every failure record of the run, and the execution when it failed,
/// carries [`MESSAGE`]. Returns the names of the failed operations.
fn assert_failure_messages<O>(result: &TestResult<O>, case: &str) -> Vec<String> {
    let failed: Vec<_> = result
        .operations()
        .iter()
        .filter(|op| op.failed())
        .collect();
    assert!(!failed.is_empty(), "{case}: no failure record");
    for op in &failed {
        assert_eq!(
            op.error_message(),
            Some(MESSAGE),
            "{case}: the record of {:?}",
            op.name()
        );
    }
    if result.is_failure() {
        assert_eq!(
            result.error_message(),
            Some(MESSAGE),
            "{case}: the execution"
        );
    }
    failed
        .iter()
        .filter_map(|op| op.name().map(str::to_owned))
        .collect()
}

/// Whether the run recorded a failed operation named `name`.
fn has_failed<O>(result: &TestResult<O>, name: &str) -> bool {
    result
        .operations_named(name)
        .any(durable::test_util::TestOperation::failed)
}

/// The failed item's messages in the serialized batch record of the
/// operation named `batch`: the item error's and its cause's.
fn serialized_item_messages<O>(result: &TestResult<O>, batch: &str) -> (String, String) {
    let record = result
        .operation(batch)
        .and_then(durable::test_util::TestOperation::result)
        .expect("the batch records its result");
    let record: serde_json::Value = serde_json::from_str(record).expect("a JSON batch record");
    let message = |pointer: &str| {
        record
            .pointer(pointer)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("{pointer} in {record}"))
            .to_owned()
    };
    (
        message("/all/0/error/ErrorMessage"),
        message("/all/0/error/Cause/ErrorMessage"),
    )
}

/// The batch views a batch case returns: the failed item's
/// `BatchItem::error_message` and `BatchError::message`.
fn batch_views<O>(batch: &durable::builders::map_parallel::BatchResult<O>) -> Vec<String> {
    let item = batch.failed().next().expect("the item failed");
    let error = batch
        .errors()
        .into_iter()
        .next()
        .expect("the batch has an error");
    vec![
        item.error_message
            .clone()
            .expect("a failed item has a message"),
        error.message.to_owned(),
    ]
}

#[tokio::test]
#[expect(clippy::too_many_lines)] // reason: one case per propagating operation, as the issue lists them
async fn propagated_step_failure_records_the_step_message() {
    // Normal and virtual child contexts.
    for virtual_child in [false, true] {
        let result = run(move |ctx| async move {
            let child = ctx
                .run_in_child_context(|child| async move { Ok(failing_step(&child).await?) })
                .name("child");
            let child = if virtual_child {
                child.virtual_context()
            } else {
                child
            };
            Ok::<_, BoxError>(child.await?)
        })
        .await;
        let case = format!("child, virtual {virtual_child}");
        assert!(result.is_failure(), "{case}");
        let failed = assert_failure_messages(&result, &case);
        assert_eq!(
            failed.contains(&"child".to_owned()),
            !virtual_child,
            "{case}"
        );
    }

    // Map, parallel and parallel_mixed, in both nesting modes.
    for mode in [NestingMode::Normal, NestingMode::Flat] {
        let result = run(move |ctx| async move {
            let batch = ctx
                .map(vec![1_u32], |child, _item, _idx| async move {
                    Ok(failing_step(&child).await?)
                })
                .name("batch")
                .item_namer(|_item, _idx| "item".to_owned())
                .nesting(mode)
                .completion(CompletionConfig::with_tolerated_failure_count(1))
                .await_batch()
                .await?;
            Ok::<_, BoxError>(batch_views(&batch))
        })
        .await;
        assert_batch_case(&result, &format!("map {mode:?}"), mode, true);

        let result = run(move |ctx| async move {
            let branches: Vec<Branch<u32>> = vec![Branch::new("item", |child| async move {
                Ok(failing_step(&child).await?)
            })];
            let batch = ctx
                .parallel(branches)
                .name("batch")
                .nesting(mode)
                .completion(CompletionConfig::with_tolerated_failure_count(1))
                .await_batch()
                .await?;
            Ok::<_, BoxError>(batch_views(&batch))
        })
        .await;
        assert_batch_case(&result, &format!("parallel {mode:?}"), mode, true);

        let result = run(move |ctx| async move {
            let mut branches = BranchSet::new();
            let handle = branches.add(
                "item",
                |child| async move { Ok(failing_step(&child).await?) },
            );
            ctx.parallel_mixed(branches)
                .name("batch")
                .nesting(mode)
                .completion(CompletionConfig::with_tolerated_failure_count(1))
                .await?;
            let err = handle.into_result().expect_err("the branch failed");
            Ok::<_, BoxError>(vec![
                err.wire()
                    .and_then(WireError::error_message)
                    .unwrap_or_default()
                    .to_owned(),
            ])
        })
        .await;
        assert_batch_case(&result, &format!("parallel_mixed {mode:?}"), mode, false);
        assert_mixed_batch_views(mode).await;
    }

    // A default wrapped `with_retry` block and its `attempt-1`.
    let result = run(|ctx| async move {
        Ok::<_, BoxError>(
            ctx.with_retry(|child, _attempt| async move { Ok(failing_step(&child).await?) })
                .name("block")
                .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                .await?,
        )
    })
    .await;
    let failed = assert_failure_messages(&result, "with_retry");
    assert!(failed.contains(&"block".to_owned()), "{failed:?}");
    assert!(failed.contains(&"attempt-1".to_owned()), "{failed:?}");

    // `try_join_all`.
    let result = run(|ctx| async move {
        let input = ctx
            .step(|_| async { Err::<u32, BoxError>(MESSAGE.into()) })
            .name("bad")
            .retry_strategy(|_err, _attempt| RetryDecision::Stop)
            .future();
        Ok::<_, BoxError>(ctx.try_join_all([input]).name("join").await?)
    })
    .await;
    let failed = assert_failure_messages(&result, "try_join_all");
    assert!(failed.contains(&"join".to_owned()), "{failed:?}");

    // A rejecting `race`.
    let result = run(|ctx| async move {
        let loser = ctx
            .step(|_| async { Err::<u32, BoxError>(MESSAGE.into()) })
            .name("bad")
            .retry_strategy(|_err, _attempt| RetryDecision::Stop)
            .future();
        let slow = ctx
            .step(|_| async {
                tokio::time::sleep(Duration::from_secs(30)).await;
                Ok(1_u32)
            })
            .name("slow")
            .future();
        Ok::<_, BoxError>(ctx.race([loser, slow]).name("race").await?)
    })
    .await;
    let failed = assert_failure_messages(&result, "race");
    assert!(failed.contains(&"race".to_owned()), "{failed:?}");

    // A `wait_for_callback` submitter.
    let result = run(|ctx| async move {
        Ok::<_, BoxError>(
            ctx.wait_for_callback::<String, _, _>(|_step_ctx, _callback_id| async {
                Err::<(), BoxError>(MESSAGE.into())
            })
            .name("cb")
            .submitter_retry(|_err, _attempt| RetryDecision::Stop)
            .await?,
        )
    })
    .await;
    let failed = assert_failure_messages(&result, "wait_for_callback");
    assert!(failed.contains(&"cb".to_owned()), "{failed:?}");
    // The submitter step's record carries no name.
    assert!(
        result
            .operations()
            .iter()
            .any(|op| op.kind() == OperationKind::Step && op.failed()),
        "the submitter step's failure record"
    );

    // Plugin end hooks of a normal and a virtual child.
    for virtual_child in [false, true] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let options = Options::builder()
            .plugin(EndErrors(Arc::clone(&events)))
            .build()
            .expect("options");
        let result = run_with(options, move |ctx| async move {
            let child = ctx
                .run_in_child_context(|child| async move { Ok(failing_step(&child).await?) })
                .name("child");
            let child = if virtual_child {
                child.virtual_context()
            } else {
                child
            };
            Ok::<_, BoxError>(child.await?)
        })
        .await;
        assert!(result.is_failure());
        let events = events.lock().unwrap().clone();
        for hook in ["operation_end", "user_function_end"] {
            assert!(
                events.contains(&(hook, Some(MESSAGE.to_owned()))),
                "virtual {virtual_child}: {hook} in {events:?}"
            );
        }
    }
}

/// The assertions of one batch case. `views` holds the failed item's
/// `BatchItem::error_message` and `BatchError::message` when the operation
/// exposes them, else the branch handle's recorded message.
fn assert_batch_case(
    result: &TestResult<Vec<String>>,
    case: &str,
    mode: NestingMode,
    has_batch_views: bool,
) {
    assert!(result.is_success(), "{case}: {:?}", result.error_message());
    let views = result.output().expect("the views");
    let expected = if has_batch_views { 2 } else { 1 };
    assert_eq!(views.len(), expected, "{case}");
    for view in views {
        assert_eq!(view, MESSAGE, "{case}: {views:?}");
    }
    let (item_message, cause_message) = serialized_item_messages(result, "batch");
    assert_eq!(item_message, MESSAGE, "{case}");
    assert_eq!(cause_message, MESSAGE, "{case}");
    assert_failure_messages(result, case);
    let item_record = result
        .operations_named("item")
        .any(|op| op.kind() == OperationKind::Context && op.failed());
    match mode {
        NestingMode::Normal => assert!(item_record, "{case}: an item failure record"),
        NestingMode::Flat => assert!(!item_record, "{case}: no item record"),
        _ => unreachable!("two nesting modes"),
    }
}

/// Asserts `BatchItem::error_message` and `BatchError::message` of a
/// failed `parallel_mixed` branch. `parallel_mixed` returns no
/// `BatchResult`, so the case reads it in `.summary(...)`, which the SDK
/// calls with the batch's `BatchResult` when the serialized result is over
/// 256 KiB. A second branch returns a 300 KiB string to make it so.
async fn assert_mixed_batch_views(mode: NestingMode) {
    let case = format!("parallel_mixed {mode:?}, batch views");
    let result = run(move |ctx| async move {
        let views = Arc::new(Mutex::new(Vec::new()));
        let mut branches = BranchSet::new();
        let _failed = branches.add(
            "item",
            |child| async move { Ok(failing_step(&child).await?) },
        );
        let _large = branches.add("large", |_child| async move {
            Ok::<_, BoxError>("x".repeat(300 * 1024))
        });
        let captured = Arc::clone(&views);
        ctx.parallel_mixed(branches)
            .name("batch")
            .nesting(mode)
            .completion(CompletionConfig::with_tolerated_failure_count(1))
            .summary(move |batch| {
                captured.lock().unwrap().extend(batch_views(batch));
                String::new()
            })
            .await?;
        let views = views.lock().unwrap().clone();
        Ok::<_, BoxError>(views)
    })
    .await;
    assert!(result.is_success(), "{case}: {:?}", result.error_message());
    let views = result.output().expect("the views");
    assert_eq!(
        views,
        &vec![MESSAGE.to_owned(), MESSAGE.to_owned()],
        "{case}: BatchItem::error_message and BatchError::message"
    );
    assert_failure_messages(&result, &case);
}

/// Records the error of each operation and user-function end hook of the
/// operation named `child`.
#[derive(Clone)]
struct EndErrors(Arc<Mutex<Vec<EndError>>>);

/// One end hook's name and its error's message.
type EndError = (&'static str, Option<String>);

impl PluginFactory for EndErrors {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl Plugin for EndErrors {
    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if info.name() == Some("child") {
            let message = info
                .error()
                .and_then(WireError::error_message)
                .map(str::to_owned);
            self.0.lock().unwrap().push(("operation_end", message));
        }
        Ok(None)
    }

    fn on_user_function_end(
        &self,
        info: &OperationInfo,
        outcome: Result<(), &WireError>,
    ) -> HookWork {
        if info.name() == Some("child") {
            let message = outcome
                .err()
                .and_then(WireError::error_message)
                .map(str::to_owned);
            self.0.lock().unwrap().push(("user_function_end", message));
        }
        Ok(None)
    }
}

#[tokio::test]
async fn replayed_step_failure_records_the_step_message() {
    let map = run(|ctx| async move {
        Ok::<_, BoxError>(
            ctx.map(vec![1_u32], |child, _item, _idx| async move {
                failing_step_after_wait(&child).await
            })
            .name("batch")
            .await?,
        )
    })
    .await;
    let parallel = run(|ctx| async move {
        let branches: Vec<Branch<u32>> = vec![Branch::new("item", |child| async move {
            failing_step_after_wait(&child).await
        })];
        Ok::<_, BoxError>(ctx.parallel(branches).name("batch").await?)
    })
    .await;
    let child = run(|ctx| async move {
        Ok::<_, BoxError>(
            ctx.run_in_child_context(|child| async move { failing_step_after_wait(&child).await })
                .name("child")
                .await?,
        )
    })
    .await;
    for (case, result) in [("map", &map), ("parallel", &parallel)] {
        assert_eq!(result.invocation_count(), 2, "{case}");
        assert!(result.is_failure(), "{case}");
        assert_failure_messages(result, case);
    }
    assert_eq!(child.invocation_count(), 2, "child");
    assert!(child.is_failure(), "child");
    let failed = assert_failure_messages(&child, "child");
    assert!(failed.contains(&"child".to_owned()), "{failed:?}");
}

#[tokio::test]
async fn handler_records_the_propagated_message() {
    let direct = run(|ctx| async move { Ok::<_, BoxError>(failing_step(&ctx).await?) }).await;
    assert_eq!(direct.invocation_count(), 1);
    let after_wait = run(|ctx| async move { failing_step_after_wait(&ctx).await }).await;
    assert_eq!(after_wait.invocation_count(), 2);
    let map = run(|ctx| async move {
        Ok::<_, BoxError>(
            ctx.map(vec![1_u32], |child, _item, _idx| async move {
                Ok(failing_step(&child).await?)
            })
            .name("batch")
            .await?,
        )
    })
    .await;
    for (case, result) in [("direct", direct), ("after a wait", after_wait)] {
        assert!(result.is_failure(), "{case}");
        assert_eq!(result.error_message(), Some(MESSAGE), "{case}");
    }
    assert!(map.is_failure(), "map");
    assert_eq!(map.error_message(), Some(MESSAGE), "map");
}

/// A user wrapper whose `Display` is `wrapper`.
#[derive(Debug)]
struct Wrapper(OperationError);

impl std::fmt::Display for Wrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "wrapper")
    }
}

impl std::error::Error for Wrapper {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[tokio::test]
async fn wrapped_failure_keeps_the_flattened_chain() {
    let result = run(|ctx| async move {
        Ok::<_, BoxError>(
            ctx.run_in_child_context(|child| async move {
                match failing_step(&child).await {
                    Ok(value) => Ok(value),
                    Err(err) => Err(Box::new(Wrapper(err)) as BoxError),
                }
            })
            .name("child")
            .await?,
        )
    })
    .await;
    let child = result.operation("child").expect("the child's record");
    assert_eq!(
        child.error_message(),
        Some("wrapper: operation error: step: execution failed: bad item")
    );
    assert!(has_failed(&result, "child"));

    let plain = run(|_ctx| async { Err::<u32, BoxError>("plain".into()) }).await;
    assert_eq!(plain.error_message(), Some("plain"));
}
