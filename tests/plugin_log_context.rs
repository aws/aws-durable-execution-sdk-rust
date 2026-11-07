//! A span a plugin enters around `next()` is a stored ancestor of the log
//! records of the code it wraps, and keeps the SDK's correlation fields and
//! replay suppression (#165).
//!
//! Each test captures records as JSON, whose `spans` list is the record's
//! stored span chain, outermost first.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::indexing_slicing, clippy::panic)] // reason: test assertions

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::{CompletionConfig, CompletionDecision, NestingMode};
use durable::builders::wait_for_condition::WaitDecision;
use durable::plugin::{InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::test_util::{CallbackReply, LocalRunner, TestResult};
use durable::{BoxError, Branch, BranchSet, DurableContext, Options, ReplayFilterLayer};
use serde_json::Value;
use tracing_subscriber::Layer as _;
use tracing_subscriber::layer::SubscriberExt as _;

// Example 2 of the README, with `BoxError` imported.

struct TraceIdFactory;

impl PluginFactory for TraceIdFactory {
    fn create_plugin(&self, info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let root = info.x_ray_trace_id().unwrap_or("no-trace").to_owned();
        Ok(Some(Box::new(TraceIdPlugin { root })))
    }
}

/// Holds the invocation's X-Ray trace header.
struct TraceIdPlugin {
    root: String,
}

impl Plugin for TraceIdPlugin {
    fn wrap_user_function(&self, info: &OperationInfo, next: &mut dyn FnMut()) {
        let id = format!("{}/{}", self.root, info.id());
        let span = tracing::info_span!("plugin_user_function", trace_id = %id);
        span.in_scope(next);
    }
}

/// The trace ID Example 2 gives the user function `id` under
/// [`LocalRunner`], which supplies no X-Ray trace header.
fn trace_id(id: &str) -> String {
    format!("no-trace/{id}")
}

/// A `MakeWriter` that appends to a shared buffer.
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("not poisoned").extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl Capture {
    /// Every captured record.
    fn records(&self) -> Vec<Value> {
        String::from_utf8_lossy(&self.0.lock().expect("not poisoned"))
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    /// The records whose message is `message`.
    fn all(&self, message: &str) -> Vec<Value> {
        self.records()
            .into_iter()
            .filter(|r| r["fields"]["message"] == message)
            .collect()
    }

    /// The first record whose message is `message`.
    fn first(&self, message: &str) -> Value {
        self.all(message)
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("no record {message:?}: {:#?}", self.records()))
    }
}

/// Installs a JSON capture on this thread, unfiltered or behind
/// [`ReplayFilterLayer`].
fn capture(replay_filter: bool) -> (Capture, tracing::subscriber::DefaultGuard) {
    let capture = Capture::default();
    let layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(capture.clone());
    let guard = if replay_filter {
        test_tracing::install_test_subscriber(
            tracing_subscriber::registry().with(layer.with_filter(ReplayFilterLayer::new())),
        )
    } else {
        test_tracing::install_test_subscriber(tracing_subscriber::registry().with(layer))
    };
    (capture, guard)
}

/// The stored span chain of `record`, outermost first.
fn spans(record: &Value) -> Vec<Value> {
    record["spans"].as_array().cloned().unwrap_or_default()
}

fn name(span: &Value) -> &str {
    span["name"].as_str().unwrap_or_default()
}

fn is_sdk(span: &Value) -> bool {
    matches!(name(span), "durable_execution" | "durable_operation")
}

/// The stored parent of the Example 2 span of the user function `id` in
/// `record`'s chain. Fails when that span is not an ancestor.
fn parent_of_plugin_span(record: &Value, id: &str) -> Value {
    let chain = spans(record);
    let index = chain
        .iter()
        .position(|s| name(s) == "plugin_user_function" && s["trace_id"] == trace_id(id))
        .unwrap_or_else(|| panic!("no plugin span of {id} in {record}"));
    assert!(index > 0, "the plugin span has a parent: {record}");
    chain[index - 1].clone()
}

/// The innermost SDK span of `record`.
fn innermost_sdk(record: &Value) -> Value {
    spans(record)
        .into_iter()
        .rev()
        .find(is_sdk)
        .unwrap_or_else(|| panic!("no SDK span in {record}"))
}

/// The wire ID of the recorded operation `name`.
fn id_of<O>(result: &TestResult<O>, name: &str) -> String {
    result
        .operation(name)
        .unwrap_or_else(|| panic!("no operation {name}"))
        .id()
        .to_owned()
}

/// Asserts that `span` is a `kind` SDK span with the namespace or operation
/// correlation fields of the operation `id` named `op_name`, and with
/// `attempt` exactly when it is an operation span.
fn assert_correlation<'n>(span: &Value, kind: &str, id: &str, op_name: impl Into<Option<&'n str>>) {
    assert_eq!(name(span), kind, "{span}");
    assert_eq!(span["operationId"], id, "{span}");
    match op_name.into() {
        Some(op_name) => assert_eq!(span["operationName"], op_name, "{span}"),
        None => assert!(span.get("operationName").is_none(), "{span}"),
    }
    assert!(span["executionArn"].is_string(), "{span}");
    assert!(span["requestId"].is_string(), "{span}");
    assert!(span["isReplay"].is_boolean(), "{span}");
    assert!(span.get("tenantId").is_none(), "{span}");
    if kind == "durable_operation" {
        assert_eq!(span["attempt"], 1, "{span}");
    } else {
        assert!(span.get("attempt").is_none(), "{span}");
    }
}

/// Asserts that a handler record has the root namespace span alone among
/// its SDK spans, without operation fields.
fn assert_handler_record(record: &Value) {
    let sdk: Vec<Value> = spans(record).into_iter().filter(is_sdk).collect();
    assert_eq!(sdk.len(), 1, "{record}");
    let root = &sdk[0];
    assert_eq!(name(root), "durable_execution", "{record}");
    for field in ["operationId", "operationName", "attempt", "tenantId"] {
        assert!(root.get(field).is_none(), "{field} in {record}");
    }
}

/// Runs `handler` with `factory` registered.
async fn run<F, Fut, O>(
    factory: impl PluginFactory + 'static,
    runner: LocalRunner,
    handler: F,
) -> TestResult<O>
where
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    runner
        .options(
            Options::builder()
                .plugin(factory)
                .build()
                .expect("valid options"),
        )
        .run(handler, ())
        .await
        .expect("runner")
}

#[tokio::test]
async fn example_2_span_reaches_step_and_condition_bodies() {
    let (capture, _guard) = capture(false);
    let result = run(TraceIdFactory, LocalRunner::new(), |(), ctx| async move {
        tracing::info!("handler record");
        ctx.step(|_| async {
            tracing::info!("step body");
            Ok(1_u32)
        })
        .name("s")
        .await?;
        ctx.wait_for_condition(
            |_ctx, state: u32| async move {
                tracing::info!("condition check");
                Ok(state + 1)
            },
            0,
        )
        .wait_strategy_fn(|_state, _attempt| WaitDecision::complete())
        .name("c")
        .await?;
        Ok(())
    })
    .await;
    assert!(result.output().is_some());
    assert_handler_record(&capture.first("handler record"));
    for (message, op) in [("step body", "s"), ("condition check", "c")] {
        let record = capture.first(message);
        let id = id_of(&result, op);
        let parent = parent_of_plugin_span(&record, &id);
        assert_correlation(&parent, "durable_operation", &id, op);
        assert_eq!(innermost_sdk(&record), parent, "{record}");
    }
}

#[tokio::test]
async fn example_2_span_reaches_recorded_child_and_with_retry_bodies() {
    let (capture, _guard) = capture(false);
    let result = run(TraceIdFactory, LocalRunner::new(), |(), ctx| async move {
        ctx.run_in_child_context(|_child| async move {
            tracing::info!("child body");
            Ok(1_u32)
        })
        .name("c")
        .await?;
        ctx.with_retry(|_child, _attempt| async move {
            tracing::info!("retry body");
            Ok(2_u32)
        })
        .name("r")
        .await?;
        Ok(())
    })
    .await;
    let record = capture.first("child body");
    let id = id_of(&result, "c");
    let parent = parent_of_plugin_span(&record, &id);
    assert_correlation(&parent, "durable_execution", &id, "c");
    assert_eq!(innermost_sdk(&record), parent, "{record}");

    // The block's attempt is a child namespace inside the block's wrap.
    let record = capture.first("retry body");
    let id = id_of(&result, "r");
    let parent = parent_of_plugin_span(&record, &id);
    assert_correlation(&parent, "durable_execution", &id, "r");
    let attempt = id_of(&result, "attempt-1");
    assert_correlation(
        &innermost_sdk(&record),
        "durable_execution",
        &attempt,
        "attempt-1",
    );
}

/// The handler of the virtual-context cases: a child `v` and a `with_retry`
/// block `r`, virtual when `virtual_context`.
async fn virtual_handler(ctx: DurableContext, virtual_context: bool) -> Result<(), BoxError> {
    let child = ctx
        .run_in_child_context(|_child| async move {
            tracing::info!("virtual child body");
            Ok(1_u32)
        })
        .name("v");
    let child = if virtual_context {
        child.virtual_context()
    } else {
        child
    };
    child.await?;
    let block = ctx
        .with_retry(|_child, _attempt| async move {
            tracing::info!("virtual retry body");
            Ok(2_u32)
        })
        .name("r");
    let block = if virtual_context {
        block.virtual_context()
    } else {
        block
    };
    block.await?;
    Ok(())
}

#[tokio::test]
async fn example_2_span_reaches_virtual_child_and_with_retry_bodies() {
    // The recorded run gives the IDs a virtual child keeps.
    let recorded = {
        let (_capture, _guard) = capture(false);
        run(TraceIdFactory, LocalRunner::new(), |(), ctx| {
            virtual_handler(ctx, false)
        })
        .await
    };
    let (child_id, block_id) = (id_of(&recorded, "v"), id_of(&recorded, "r"));
    let (capture, _guard) = capture(false);
    let result = run(TraceIdFactory, LocalRunner::new(), |(), ctx| {
        virtual_handler(ctx, true)
    })
    .await;
    for name in ["v", "r"] {
        assert!(result.operation(name).is_none(), "{name} wrote a record");
    }
    let record = capture.first("virtual child body");
    let parent = parent_of_plugin_span(&record, &child_id);
    assert_correlation(&parent, "durable_execution", &child_id, "v");
    assert_eq!(innermost_sdk(&record), parent, "{record}");

    let record = capture.first("virtual retry body");
    let parent = parent_of_plugin_span(&record, &block_id);
    assert_correlation(&parent, "durable_execution", &block_id, "r");
    let attempt = id_of(&result, "r-attempt-1");
    assert_correlation(
        &innermost_sdk(&record),
        "durable_execution",
        &attempt,
        "r-attempt-1",
    );
}

#[tokio::test]
async fn example_2_span_reaches_wait_for_callback_body_and_submitter() {
    let (capture, _guard) = capture(false);
    let runner =
        LocalRunner::new().callback_for("w", CallbackReply::success(&"ok").expect("encodes"));
    let result = run(TraceIdFactory, runner, |(), ctx| async move {
        Ok(ctx
            .wait_for_callback::<String, _, _>(|_step_ctx, _callback_id| async move {
                tracing::info!("submitter body");
                Ok(())
            })
            .name("w")
            .await?)
    })
    .await;
    assert_eq!(result.output(), Some(&"ok".to_owned()));
    let record = capture.first("submitter body");
    let body = id_of(&result, "w");
    let parent = parent_of_plugin_span(&record, &body);
    assert_correlation(&parent, "durable_execution", &body, "w");
    // The submitter step records no name of its own.
    let submitter = result
        .operations()
        .iter()
        .find(|op| op.kind() == durable::OperationKind::Step)
        .expect("the submitter step")
        .id()
        .to_owned();
    let parent = parent_of_plugin_span(&record, &submitter);
    assert_correlation(&parent, "durable_operation", &submitter, None);
    assert_eq!(innermost_sdk(&record), parent, "{record}");
}

/// Which batch a batch case runs.
#[derive(Debug, Clone, Copy)]
enum Batch {
    Map,
    Parallel,
    ParallelMixed,
}

impl Batch {
    /// The recorded name of its one item.
    fn item(self) -> &'static str {
        match self {
            Self::Map => "map-item-0",
            Self::Parallel | Self::ParallelMixed => "b",
        }
    }
}

/// Runs a batch `x` of one item that logs `"item body"`, in `mode`. With
/// `probe`, the batch takes a completion predicate that logs
/// `"container body"` and returns `Continue`: the SDK asks it from the
/// container's own body, the user function the shared runner polls, not
/// from an item task.
async fn batch_handler(
    ctx: DurableContext,
    batch: Batch,
    mode: NestingMode,
    probe: bool,
) -> Result<(), BoxError> {
    let completion = probe.then(|| {
        CompletionConfig::with_completion_predicate(|_stats| {
            tracing::info!("container body");
            CompletionDecision::Continue
        })
    });
    match batch {
        Batch::Map => {
            let mut builder = ctx
                .map(vec![1_u32], |_child, item, _index| async move {
                    tracing::info!("item body");
                    Ok(item)
                })
                .name("x")
                .nesting(mode);
            if let Some(completion) = completion {
                builder = builder.completion(completion);
            }
            builder.await?;
        }
        Batch::Parallel => {
            let branches: Vec<Branch<u32>> = vec![Branch::new("b", |_child| async move {
                tracing::info!("item body");
                Ok(1_u32)
            })];
            let mut builder = ctx.parallel(branches).name("x").nesting(mode);
            if let Some(completion) = completion {
                builder = builder.completion(completion);
            }
            builder.await?;
        }
        Batch::ParallelMixed => {
            let mut branches = BranchSet::new();
            let handle = branches.add("b", |_child| async move {
                tracing::info!("item body");
                Ok(1_u32)
            });
            let mut builder = ctx.parallel_mixed(branches).name("x").nesting(mode);
            if let Some(completion) = completion {
                builder = builder.completion(completion);
            }
            builder.await?;
            handle.into_result()?;
        }
    }
    Ok(())
}

/// The batch case of `batch`: the item record has Example 2's span of the
/// item, whose parent is the item's namespace span, in both modes; a flat
/// item writes no record and keeps the ID of the normal one.
async fn assert_batch_items(batch: Batch) {
    let item = batch.item();
    let normal = {
        let (capture, _guard) = capture(false);
        let result = run(TraceIdFactory, LocalRunner::new(), move |(), ctx| {
            batch_handler(ctx, batch, NestingMode::Normal, false)
        })
        .await;
        let id = id_of(&result, item);
        let record = capture.first("item body");
        let parent = parent_of_plugin_span(&record, &id);
        assert_correlation(&parent, "durable_execution", &id, item);
        assert_eq!(innermost_sdk(&record), parent, "{batch:?}: {record}");
        id
    };
    let (capture, _guard) = capture(false);
    let result = run(TraceIdFactory, LocalRunner::new(), move |(), ctx| {
        batch_handler(ctx, batch, NestingMode::Flat, false)
    })
    .await;
    assert!(
        result.operation(item).is_none(),
        "{batch:?}: a flat item wrote a record"
    );
    let record = capture.first("item body");
    let parent = parent_of_plugin_span(&record, &normal);
    assert_correlation(&parent, "durable_execution", &normal, item);
    assert_eq!(innermost_sdk(&record), parent, "{batch:?}: {record}");
}

#[tokio::test]
async fn example_2_span_reaches_map_items_in_both_modes() {
    assert_batch_items(Batch::Map).await;
}

#[tokio::test]
async fn example_2_span_reaches_parallel_branches_in_both_modes() {
    assert_batch_items(Batch::Parallel).await;
}

#[tokio::test]
async fn example_2_span_reaches_parallel_mixed_branches_in_both_modes() {
    assert_batch_items(Batch::ParallelMixed).await;
}

/// A batch container's user function runs inside the container's
/// namespace span: a record its body emits, from the completion predicate
/// it asks, has Example 2's span of the container, whose stored parent is
/// the container's namespace span, and no span of an item. In `Flat`
/// nesting the container keeps its own namespace and record.
#[tokio::test]
async fn container_user_functions_enter_the_namespace_span_before_plugin_wraps() {
    for batch in [Batch::Map, Batch::Parallel, Batch::ParallelMixed] {
        for mode in [NestingMode::Normal, NestingMode::Flat] {
            let (capture, _guard) = capture(false);
            let result = run(TraceIdFactory, LocalRunner::new(), move |(), ctx| {
                batch_handler(ctx, batch, mode, true)
            })
            .await;
            let id = id_of(&result, "x");
            let records = capture.all("container body");
            assert!(
                !records.is_empty(),
                "{batch:?} {mode:?}: no container record"
            );
            for record in records {
                let parent = parent_of_plugin_span(&record, &id);
                assert_correlation(&parent, "durable_execution", &id, "x");
                assert_eq!(
                    innermost_sdk(&record),
                    parent,
                    "{batch:?} {mode:?}: {record}"
                );
                let plugin_spans: Vec<_> = spans(&record)
                    .into_iter()
                    .filter(|s| name(s) == "plugin_user_function")
                    .collect();
                assert_eq!(
                    plugin_spans.len(),
                    1,
                    "{batch:?} {mode:?}: only the container's plugin span: {record}"
                );
            }
        }
    }
}

/// A plugin whose invocation wrap enters an `invocation_span` with the
/// field `invocation = "inv"` around `next()`.
#[derive(Clone)]
struct InvocationSpan;

impl Plugin for InvocationSpan {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        tracing::info_span!("invocation_span", invocation = "inv").in_scope(next);
    }
}

impl PluginFactory for InvocationSpan {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// Whether `record` descends from an [`InvocationSpan`] span.
fn under_invocation_span(record: &Value) -> bool {
    spans(record)
        .iter()
        .any(|s| name(s) == "invocation_span" && s["invocation"] == "inv")
}

#[tokio::test]
async fn wrap_invocation_span_reaches_handler_and_all_child_kinds() {
    let (capture, _guard) = capture(false);
    let result = run(InvocationSpan, LocalRunner::new(), |(), ctx| async move {
        tracing::info!("handler record");
        ctx.run_in_child_context(|_child| async move {
            tracing::info!("regular child record");
            Ok(1_u32)
        })
        .name("c")
        .await?;
        ctx.run_in_child_context(|_child| async move {
            tracing::info!("virtual child record");
            Ok(2_u32)
        })
        .name("v")
        .virtual_context()
        .await?;
        ctx.map(vec![3_u32], |_child, item, _index| async move {
            tracing::info!("flat item record");
            Ok(item)
        })
        .name("m")
        .nesting(NestingMode::Flat)
        .await?;
        Ok(())
    })
    .await;
    assert!(result.output().is_some());
    let handler = capture.first("handler record");
    assert!(under_invocation_span(&handler), "{handler}");
    assert_handler_record(&handler);
    let regular = capture.first("regular child record");
    assert!(under_invocation_span(&regular), "{regular}");
    let id = id_of(&result, "c");
    assert_correlation(&innermost_sdk(&regular), "durable_execution", &id, "c");
    for (message, op_name) in [
        ("virtual child record", "v"),
        ("flat item record", "map-item-0"),
    ] {
        let record = capture.first(message);
        assert!(under_invocation_span(&record), "{record}");
        let namespace = innermost_sdk(&record);
        assert_eq!(name(&namespace), "durable_execution", "{record}");
        assert_eq!(namespace["operationName"], op_name, "{record}");
        assert!(namespace["operationId"].is_string(), "{record}");
        assert!(
            result.operation(op_name).is_none(),
            "{op_name} wrote a record"
        );
    }
}

/// Logs `"handler guard dropped"` when dropped.
struct LogsOnDrop;

impl Drop for LogsOnDrop {
    fn drop(&mut self) {
        tracing::info!("handler guard dropped");
    }
}

/// A guard that the handler holds across a wait is dropped when the
/// suspended handler is stopped, under the root namespace span and the
/// invocation wrap's span.
#[tokio::test]
async fn wrap_invocation_span_reaches_handler_drop_record() {
    let (capture, _guard) = capture(false);
    let result = run(InvocationSpan, LocalRunner::new(), |(), ctx| async move {
        let _guard = LogsOnDrop;
        ctx.wait(Duration::from_secs(1)).name("w").await?;
        Ok(())
    })
    .await;
    assert!(
        result.invocation_count() >= 2,
        "the handler was stopped at the wait"
    );
    let records = capture.all("handler guard dropped");
    assert_eq!(records.len(), result.invocation_count(), "{records:#?}");
    for record in records {
        assert!(under_invocation_span(&record), "{record}");
        assert_handler_record(&record);
    }
}

/// A plugin whose start hook of the user function `c` logs one record
/// without `isReplay` and one with `isReplay = false`.
#[derive(Clone)]
struct StartLogs;

impl Plugin for StartLogs {
    fn on_user_function_start(&self, info: &OperationInfo) -> durable::plugin::HookWork {
        if info.name() == Some("c") {
            tracing::info!("hook start unmarked");
            tracing::info!(isReplay = false, "hook start marked");
        }
        Ok(None)
    }
}

impl PluginFactory for StartLogs {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// The start hook of a child that resumes replaying runs inside the
/// child's namespace span, while the parent namespace is live: its
/// unmarked record is written once across the invocations, its record
/// marked live once per invocation.
#[tokio::test]
async fn user_function_hook_records_follow_namespace_replay() {
    let (capture, _guard) = capture(true);
    let result = run(StartLogs, LocalRunner::new(), |(), ctx| async move {
        let child = ctx
            .run_in_child_context(|child| async move {
                child.step(|_| async { Ok(1_u32) }).name("s").await?;
                child.wait(Duration::from_secs(1)).name("w").await?;
                Ok(())
            })
            .name("c");
        // The child's ID is claimed: the root namespace is live from here.
        tracing::info!("root after child claim");
        child.await?;
        Ok(())
    })
    .await;
    let invocations = result.invocation_count();
    assert_eq!(invocations, 2, "the child suspends once on its wait");
    assert_eq!(capture.all("root after child claim").len(), invocations);
    assert_eq!(capture.all("hook start unmarked").len(), 1);
    assert_eq!(capture.all("hook start marked").len(), invocations);
}
