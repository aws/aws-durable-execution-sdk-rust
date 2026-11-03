//! Plugin factories and the invocation info they receive, through the public
//! API and `LocalRunner`.
//!
//! Local reported times are the run-start time plus virtual seconds, so the
//! tests assert differences and lower bounds, never the epoch.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::unwrap_used)] // reason: test assertions

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use aws_durable_execution_sdk as durable;
use durable::plugin::{
    HookFuture, InvocationEndInfo, InvocationInfo, OperationInfo, Plugin, PluginFactory,
};
use durable::serdes::{RetryableSerdesError, SerdesContext};
use durable::test_util::LocalRunner;
use durable::{
    BoxError, DurableContext, InvocationStatus, JsonSerdes, OperationKind, OperationStatus,
    Options, RetryDecision, Serdes, WireError,
};

/// Records every info it receives (cloned), and returns a plugin that
/// records `dropped <n>` when dropped. `n` counts the invocations from 1.
#[derive(Clone, Default)]
struct Recording {
    infos: Arc<Mutex<Vec<InvocationInfo>>>,
    events: Arc<Mutex<Vec<String>>>,
    count: Arc<AtomicUsize>,
}

struct Instance {
    n: usize,
    events: Arc<Mutex<Vec<String>>>,
}

impl Plugin for Instance {}

impl Drop for Instance {
    fn drop(&mut self) {
        self.events
            .lock()
            .unwrap()
            .push(format!("dropped {}", self.n));
    }
}

impl PluginFactory for Recording {
    fn create_plugin(&self, info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let n = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        self.infos.lock().unwrap().push(info.clone());
        self.events.lock().unwrap().push(format!("created {n}"));
        Ok(Some(Box::new(Instance {
            n,
            events: Arc::clone(&self.events),
        })))
    }
}

/// A factory that panics on every call.
struct Panics;

impl PluginFactory for Panics {
    #[expect(clippy::panic)] // reason: the factory panic under test
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        panic!("factory boom")
    }
}

/// The current time, truncated to whole milliseconds. The invocation
/// envelope carries timestamps in milliseconds, so a timestamp a plugin
/// reads is never finer than that.
fn now_millis() -> SystemTime {
    let since = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();
    SystemTime::UNIX_EPOCH + Duration::from_millis(u64::try_from(since.as_millis()).unwrap())
}

fn runner(factory: &Recording) -> LocalRunner {
    LocalRunner::new().options(
        Options::builder()
            .plugin(factory.clone())
            .build()
            .expect("valid options"),
    )
}

async fn two_waits(_event: (), ctx: DurableContext) -> Result<(), BoxError> {
    ctx.wait(Duration::from_secs(1)).await?;
    ctx.wait(Duration::from_secs(1)).await?;
    Ok(())
}

#[tokio::test]
async fn factory_runs_once_per_invocation() {
    let factory = Recording::default();
    runner(&factory).run(two_waits, ()).await.expect("runner");
    let infos = factory.infos.lock().unwrap();
    assert_eq!(infos.len(), 3);
    let first: Vec<bool> = infos
        .iter()
        .map(InvocationInfo::is_first_invocation)
        .collect();
    assert_eq!(first, [true, false, false]);
    let arn = infos.first().unwrap().execution_arn();
    assert!(infos.iter().all(|info| info.execution_arn() == arn));
}

#[tokio::test]
async fn plugin_instances_drop_when_invocation_returns() {
    let factory = Recording::default();
    runner(&factory).run(two_waits, ()).await.expect("runner");
    assert_eq!(
        *factory.events.lock().unwrap(),
        [
            "created 1",
            "dropped 1",
            "created 2",
            "dropped 2",
            "created 3",
            "dropped 3"
        ]
    );
}

#[tokio::test]
async fn invocation_info_lists_recorded_operations() {
    let before = now_millis();
    let factory = Recording::default();
    runner(&factory)
        .run(
            |_event: String, ctx: DurableContext| async move {
                ctx.run_in_child_context(|child| async move {
                    let value = child
                        .step(|step_ctx| async move {
                            if step_ctx.attempt() == 1 {
                                return Err::<String, BoxError>("attempt one failed".into());
                            }
                            Ok("x".to_owned())
                        })
                        .name("inner")
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .await?;
                    Ok(value)
                })
                .name("group")
                .await
                .map_err(BoxError::from)
            },
            "hi".to_owned(),
        )
        .await
        .expect("runner");

    let infos = factory.infos.lock().unwrap();
    let second = infos.get(1).expect("a second invocation");
    assert_eq!(second.execution_input(), "\"hi\"");
    let start = second
        .execution_start_timestamp()
        .expect("an execution start");
    assert!(start >= before, "{start:?} is before {before:?}");

    let ops = second.operations();
    let group = ops
        .values()
        .find(|op| op.name() == Some("group"))
        .expect("the child context");
    assert_eq!(group.status(), Some(OperationStatus::Started));
    assert_eq!(group.parent_id(), None);

    let inner = ops
        .values()
        .find(|op| op.name() == Some("inner"))
        .expect("the step");
    assert_eq!(inner.parent_id(), Some(group.id()));
    assert_eq!(inner.kind(), Some(OperationKind::Step));
    // The backend marks a due retry `Ready` before the next invocation.
    assert_eq!(inner.status(), Some(OperationStatus::Ready));
    assert_eq!(inner.attempt(), Some(1));
    let message = inner
        .error()
        .and_then(WireError::error_message)
        .expect("the attempt-1 error");
    assert!(message.contains("attempt one failed"), "{message}");
    let inner_start = inner.start_timestamp().expect("a step start");
    assert!(
        inner_start >= before,
        "{inner_start:?} is before {before:?}"
    );

    assert!(
        second
            .updated_operations()
            .iter()
            .any(|op| op.id() == inner.id()),
        "the ready retry is updated"
    );
}

#[tokio::test]
async fn wait_completed_between_invocations_is_updated() {
    let before = now_millis();
    let factory = Recording::default();
    runner(&factory)
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.wait(Duration::from_secs(1)).name("w").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let infos = factory.infos.lock().unwrap();
    assert!(infos.first().unwrap().updated_operations().is_empty());
    let updated = infos
        .get(1)
        .expect("a second invocation")
        .updated_operations();
    let w = updated
        .iter()
        .find(|op| op.name() == Some("w"))
        .expect("the wait is updated");
    assert_eq!(w.status(), Some(OperationStatus::Succeeded));
    let start = w.start_timestamp().expect("a start");
    assert!(start >= before, "{start:?} is before {before:?}");
    let end = w.end_timestamp().expect("an end");
    assert_eq!(end.duration_since(start).unwrap(), Duration::from_secs(1));
}

async fn two_steps(_event: (), ctx: DurableContext) -> Result<u32, BoxError> {
    let a = ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
    let b = ctx.step(|_| async { Ok(2_u32) }).name("b").await?;
    Ok(a + b)
}

#[tokio::test]
async fn panicking_factory_does_not_change_outcome() {
    let plain = LocalRunner::new().run(two_steps, ()).await.expect("runner");

    let factory = Recording::default();
    let with_plugins = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(Panics)
                .plugin(factory.clone())
                .build()
                .expect("valid options"),
        )
        .run(two_steps, ())
        .await
        .expect("runner");

    assert_eq!(with_plugins.output(), Some(&3));
    assert_eq!(with_plugins.output(), plain.output());
    let shape = |ops: &[durable::test_util::TestOperation]| {
        ops.iter()
            .map(|op| (op.name().map(str::to_owned), op.kind(), op.status()))
            .collect::<Vec<_>>()
    };
    assert_eq!(shape(with_plugins.operations()), shape(plain.operations()));
    assert_eq!(factory.count.load(Ordering::SeqCst), 1);
}

// The invocation start and end hooks.

/// What one invocation's plugin received: `start <request id>`, then
/// `end <status>`.
#[derive(Clone, Default)]
struct Lifecycle {
    starts: Arc<Mutex<Vec<InvocationInfo>>>,
    ends: Arc<Mutex<Vec<InvocationEndInfo>>>,
    order: Arc<Mutex<Vec<&'static str>>>,
}

impl Lifecycle {
    fn ends(&self) -> Vec<InvocationEndInfo> {
        self.ends.lock().unwrap().clone()
    }

    fn statuses(&self) -> Vec<InvocationStatus> {
        self.ends().iter().map(InvocationEndInfo::status).collect()
    }
}

impl Plugin for Lifecycle {
    fn on_invocation_start<'a>(&'a self, info: &'a InvocationInfo) -> HookFuture<'a> {
        self.starts.lock().unwrap().push(info.clone());
        self.order.lock().unwrap().push("start");
        Box::pin(async { Ok(()) })
    }

    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.ends.lock().unwrap().push(info.clone());
        self.order.lock().unwrap().push("end");
        Box::pin(async { Ok(()) })
    }
}

impl PluginFactory for Lifecycle {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

fn lifecycle_runner(plugin: &Lifecycle) -> LocalRunner {
    LocalRunner::new().options(
        Options::builder()
            .plugin(plugin.clone())
            .build()
            .expect("valid options"),
    )
}

#[tokio::test]
async fn end_hook_succeeded() {
    let plugin = Lifecycle::default();
    lifecycle_runner(&plugin)
        .run(
            |_e: (), ctx: DurableContext| async move {
                let value = ctx
                    .step(|_| async { Ok("ok".to_owned()) })
                    .name("s")
                    .await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    let ends = plugin.ends();
    let [end] = ends.as_slice() else {
        unreachable!("one end info, got {}", ends.len());
    };
    assert_eq!(end.status(), InvocationStatus::Succeeded);
    assert_eq!(end.execution_result(), Some("\"ok\""));
    assert!(end.execution_error().is_none());
    let operations: Vec<&OperationInfo> = end.operations().values().collect();
    assert!(
        operations
            .iter()
            .any(|op| op.name() == Some("s") && op.status() == Some(OperationStatus::Succeeded)),
        "{operations:?}"
    );
    assert!(
        operations
            .iter()
            .any(|op| op.kind() == Some(OperationKind::Execution)),
        "{operations:?}"
    );
}

#[tokio::test]
async fn end_hook_failed() {
    let plugin = Lifecycle::default();
    lifecycle_runner(&plugin)
        .run(
            |_e: (), _ctx: DurableContext| async move { Err::<(), BoxError>("boom".into()) },
            (),
        )
        .await
        .expect("runner");
    let ends = plugin.ends();
    let [end] = ends.as_slice() else {
        unreachable!("one end info, got {}", ends.len());
    };
    assert_eq!(end.status(), InvocationStatus::Failed);
    let message = end
        .execution_error()
        .and_then(WireError::error_message)
        .unwrap_or_default();
    assert!(message.contains("boom"), "{message}");
    assert_eq!(end.execution_result(), None);
}

#[tokio::test]
async fn end_hook_pending_then_succeeded() {
    let plugin = Lifecycle::default();
    lifecycle_runner(&plugin)
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.wait(Duration::from_secs(1)).await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        plugin.statuses(),
        [InvocationStatus::Pending, InvocationStatus::Succeeded]
    );
    let ends = plugin.ends();
    let first = ends.first().unwrap();
    assert_eq!(first.execution_result(), None);
    assert!(first.execution_error().is_none());
}

#[tokio::test]
async fn end_hook_retrying_on_checkpoint_fault() {
    let plugin = Lifecycle::default();
    lifecycle_runner(&plugin)
        .fail_checkpoints_after_retryable(1, 1)
        .run(two_steps, ())
        .await
        .expect("runner");
    let ends = plugin.ends();
    let first = ends.first().unwrap();
    assert_eq!(first.status(), InvocationStatus::Retrying);
    let error = first
        .execution_error()
        .expect("a Retrying end has an error");
    assert_eq!(
        error.error_type(),
        Some("CheckpointUnrecoverableInvocationError")
    );
    let message = error.error_message().unwrap_or_default();
    assert!(
        message.contains("injected retryable checkpoint failure"),
        "{message}"
    );
    assert_eq!(
        ends.last().map(InvocationEndInfo::status),
        Some(InvocationStatus::Succeeded)
    );
}

/// JSON, except that its first `serialize` call fails with a retryable
/// error.
#[derive(Clone, Default)]
struct FailFirstSerialize(Arc<AtomicBool>);

impl<T> Serdes<T> for FailFirstSerialize
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    async fn serialize(&self, value: T, context: SerdesContext) -> Result<String, BoxError> {
        if !self.0.swap(true, Ordering::SeqCst) {
            return Err(Box::new(RetryableSerdesError::new(
                "injected serdes failure",
            )));
        }
        JsonSerdes.serialize(value, context).await
    }

    async fn deserialize(&self, wire: String, context: SerdesContext) -> Result<T, BoxError> {
        JsonSerdes.deserialize(wire, context).await
    }
}

#[tokio::test]
async fn end_hook_retrying_on_serdes_fault() {
    let plugin = Lifecycle::default();
    let serdes = FailFirstSerialize::default();
    lifecycle_runner(&plugin)
        .run(
            move |_e: (), ctx: DurableContext| {
                let serdes = serdes.clone();
                async move {
                    let value = ctx.step(|_| async { Ok(7_u32) }).serdes(serdes).await?;
                    Ok::<_, BoxError>(value)
                }
            },
            (),
        )
        .await
        .expect("runner");
    let ends = plugin.ends();
    let first = ends.first().unwrap();
    assert_eq!(first.status(), InvocationStatus::Retrying);
    let error = first
        .execution_error()
        .expect("a Retrying end has an error");
    assert_eq!(error.error_type(), Some("SerdesFailedError"));
    let message = error.error_message().unwrap_or_default();
    assert!(message.contains("serdes serialize failed"), "{message}");
    assert_eq!(
        ends.last().map(InvocationEndInfo::status),
        Some(InvocationStatus::Succeeded)
    );
}

#[tokio::test]
async fn end_hook_fires_once_per_invocation() {
    let plugin = Lifecycle::default();
    lifecycle_runner(&plugin)
        .run(two_waits, ())
        .await
        .expect("runner");
    assert_eq!(
        *plugin.order.lock().unwrap(),
        ["start", "end", "start", "end", "start", "end"]
    );
    let starts = plugin.starts.lock().unwrap().clone();
    let ends = plugin.ends();
    assert_eq!(starts.len(), 3);
    for (start, end) in starts.iter().zip(&ends) {
        assert_eq!(start.request_id(), end.request_id());
        assert_eq!(start.execution_arn(), end.execution_arn());
        assert_eq!(start.execution_input(), end.execution_input());
    }
}

/// A plugin whose start hook sleeps 10 ms, then sets its flag.
#[derive(Clone, Default)]
struct SlowStart(Arc<AtomicBool>);

impl Plugin for SlowStart {
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            self.0.store(true, Ordering::SeqCst);
            Ok(())
        })
    }
}

impl PluginFactory for SlowStart {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

#[tokio::test]
async fn start_hook_completes_before_handler() {
    let plugin = SlowStart::default();
    let flag = Arc::clone(&plugin.0);
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(plugin)
                .build()
                .expect("valid options"),
        )
        .run(
            move |_e: (), ctx: DurableContext| {
                let flag = Arc::clone(&flag);
                async move {
                    let seen = ctx
                        .step(move |_| async move { Ok(flag.load(Ordering::SeqCst)) })
                        .await?;
                    Ok::<_, BoxError>(seen)
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&true));
}

/// An event that serializes as the string `"x"` and deserializes only from
/// a number, so the handler's input never decodes.
struct NotANumber;

impl serde::Serialize for NotANumber {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str("x")
    }
}

impl<'de> serde::Deserialize<'de> for NotANumber {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <u32 as serde::Deserialize>::deserialize(deserializer).map(|_| Self)
    }
}

/// A factory that counts its calls and returns [`Lifecycle`].
#[derive(Clone, Default)]
struct CountingEnds {
    calls: Arc<AtomicUsize>,
    plugin: Lifecycle,
}

impl PluginFactory for CountingEnds {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(Some(Box::new(self.plugin.clone())))
    }
}

#[tokio::test]
async fn no_hook_before_plugins_exist() {
    let factory = CountingEnds::default();
    let result = LocalRunner::new()
        .max_invocations(2)
        .options(
            Options::builder()
                .plugin(factory.clone())
                .build()
                .expect("valid options"),
        )
        .run(
            |_e: NotANumber, _ctx: DurableContext| async move { Ok::<_, BoxError>(()) },
            NotANumber,
        )
        .await;
    let reported = match &result {
        Ok(result) => result
            .invocations()
            .iter()
            .filter_map(|invocation| invocation.error())
            .filter_map(WireError::error_message)
            .any(|message| message.contains("input")),
        Err(error) => error.to_string().contains("input"),
    };
    assert!(reported, "the run reports the input error");
    assert_eq!(factory.calls.load(Ordering::SeqCst), 0);
    assert!(factory.plugin.order.lock().unwrap().is_empty());
}

/// Its start hook panics when called. Its end hook returns a future that
/// panics when polled.
#[derive(Clone)]
struct Faulty;

impl Plugin for Faulty {
    fn name(&self) -> &'static str {
        "Faulty"
    }

    #[expect(clippy::panic)] // reason: the hook panic under test
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        panic!("start boom")
    }

    #[expect(clippy::panic)] // reason: the hook panic under test
    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        Box::pin(async { panic!("end boom") })
    }
}

impl PluginFactory for Faulty {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// Its start hook returns `Err("x")`.
#[derive(Clone)]
struct Failing;

impl Plugin for Failing {
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        Box::pin(async { Err("x".into()) })
    }
}

impl PluginFactory for Failing {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// Captures the events of this thread as JSON lines.
fn capture() -> (Arc<Mutex<Vec<u8>>>, tracing::subscriber::DefaultGuard) {
    use tracing_subscriber::layer::SubscriberExt as _;
    #[derive(Clone)]
    struct Writer(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for Writer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let writer = Writer(Arc::clone(&buffer));
    let layer = tracing_subscriber::fmt::layer()
        .json()
        .flatten_event(true)
        .with_writer(move || writer.clone());
    let guard = tracing::subscriber::set_default(tracing_subscriber::registry().with(layer));
    (buffer, guard)
}

/// The plugin warnings `buffer` captured, as `(plugin, hook, message)`.
fn plugin_warnings(buffer: &Arc<Mutex<Vec<u8>>>) -> Vec<(String, String, String)> {
    let text = |v: &serde_json::Value, key: &str| {
        v.get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    String::from_utf8_lossy(&buffer.lock().unwrap())
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| {
            text(v, "target") == "aws_durable_execution_sdk::plugin" && text(v, "level") == "WARN"
        })
        .map(|v| (text(&v, "plugin"), text(&v, "hook"), text(&v, "message")))
        .collect()
}

#[tokio::test]
#[expect(clippy::panic)] // reason: reports that no attempt captured the warnings
async fn panicking_invocation_hooks_are_isolated() {
    let plain = Lifecycle::default();
    let alone = lifecycle_runner(&plain)
        .run(two_steps, ())
        .await
        .expect("runner");
    for _attempt in 0..5 {
        let (buffer, _guard) = capture();
        let healthy = Lifecycle::default();
        let result = LocalRunner::new()
            .options(
                Options::builder()
                    .plugin(Faulty)
                    .plugin(healthy.clone())
                    .plugin(Failing)
                    .build()
                    .expect("valid options"),
            )
            .run(two_steps, ())
            .await
            .expect("runner");
        assert!(result.is_success());
        assert_eq!(result.output(), alone.output());
        let order = healthy.order.lock().unwrap().clone();
        assert_eq!(order.len(), 2 * result.invocation_count());
        assert!(
            order.chunks(2).all(|pair| pair == ["start", "end"]),
            "{order:?}"
        );
        let warnings = plugin_warnings(&buffer);
        let panics: Vec<_> = warnings
            .iter()
            .filter(|(_, _, message)| message == "plugin hook panicked; the panic is ignored")
            .collect();
        let failures: Vec<_> = warnings
            .iter()
            .filter(|(_, _, message)| message == "plugin hook failed; the error is ignored")
            .collect();
        let ([start, end], [failure]) = (panics.as_slice(), failures.as_slice()) else {
            continue;
        };
        assert!(start.0.ends_with("Faulty (plugins[0])"), "{start:?}");
        assert_eq!(start.1, "on_invocation_start");
        assert!(end.0.ends_with("Faulty (plugins[0])"), "{end:?}");
        assert_eq!(end.1, "on_invocation_end");
        assert!(failure.0.ends_with("(plugins[2])"), "{failure:?}");
        assert_eq!(failure.1, "on_invocation_start");
        return;
    }
    panic!("no attempt captured the three warnings");
}

/// Plugin 0's end hook waits for a message that plugin 1's end hook sends.
#[derive(Clone, Default)]
struct Handoff {
    sender: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    receiver: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
    finished: Arc<AtomicUsize>,
}

struct Receives(Handoff);
struct Sends(Handoff);

impl Plugin for Receives {
    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        let receiver = self.0.receiver.lock().unwrap().take();
        Box::pin(async move {
            receiver.expect("one end hook").await?;
            self.0.finished.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

impl Plugin for Sends {
    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        let sender = self.0.sender.lock().unwrap().take();
        Box::pin(async move {
            sender
                .expect("one end hook")
                .send(())
                .map_err(|()| "the receiver is gone")?;
            self.0.finished.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}

struct ReceivesFactory(Handoff);
struct SendsFactory(Handoff);

impl PluginFactory for ReceivesFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(Receives(self.0.clone()))))
    }
}

impl PluginFactory for SendsFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(Sends(self.0.clone()))))
    }
}

#[tokio::test]
async fn end_hooks_run_concurrently() {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let handoff = Handoff {
        sender: Arc::new(Mutex::new(Some(sender))),
        receiver: Arc::new(Mutex::new(Some(receiver))),
        finished: Arc::new(AtomicUsize::new(0)),
    };
    let runner = LocalRunner::new().options(
        Options::builder()
            .plugin(ReceivesFactory(handoff.clone()))
            .plugin(SendsFactory(handoff.clone()))
            .build()
            .expect("valid options"),
    );
    let run = runner.run(two_steps, ());
    tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("the run completes")
        .expect("runner");
    assert_eq!(handoff.finished.load(Ordering::SeqCst), 2);
}

// The end hooks when a buffered outcome write fails.

/// An error whose destructor panics.
#[derive(Debug)]
struct DropPanics;

impl std::fmt::Display for DropPanics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("drop panics")
    }
}

impl std::error::Error for DropPanics {}

impl Drop for DropPanics {
    #[expect(clippy::panic)] // reason: the destructor panic under test
    fn drop(&mut self) {
        panic!("error drop");
    }
}

/// A plugin whose end hook returns [`DropPanics`] when the invocation ends
/// `Retrying`. The invocation already fails then, so the destructor's
/// panic is dropped and the service invokes the function again.
#[derive(Clone, Default)]
struct FailsWhenRetrying;

impl Plugin for FailsWhenRetrying {
    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        let retrying = info.status() == InvocationStatus::Retrying;
        Box::pin(async move {
            if retrying {
                return Err(Box::new(DropPanics) as BoxError);
            }
            Ok(())
        })
    }
}

impl PluginFactory for FailsWhenRetrying {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// A runner with checkpoint batching, `plugin` and [`FailsWhenRetrying`].
fn batching_runner(plugin: &Lifecycle) -> LocalRunner {
    LocalRunner::new().options(
        Options::builder()
            .checkpoint_batching(true)
            .plugin(plugin.clone())
            .plugin(FailsWhenRetrying)
            .build()
            .expect("valid options"),
    )
}

/// A handler that spawns a step, waits until its body ran and its outcome
/// write joined the batch, and drops it unawaited. The outcome write is
/// then still buffered when the handler returns, so the end-of-invocation
/// flush writes it.
async fn orphaned_step(_event: (), ctx: DurableContext) -> Result<String, BoxError> {
    let ran = Arc::new(AtomicBool::new(false));
    let ran_step = Arc::clone(&ran);
    let orphan = ctx
        .step(move |_| {
            let ran = Arc::clone(&ran_step);
            async move {
                ran.store(true, Ordering::SeqCst);
                Ok("orphaned-outcome".to_owned())
            }
        })
        .name("orphan")
        .spawn();
    while !ran.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    tokio::time::sleep(Duration::from_millis(30)).await;
    drop(orphan);
    Ok("handler-done".to_owned())
}

/// A non-retryable failure of the end-of-invocation flush fails the
/// execution. The end hook runs once, with `Failed` and the flush
/// failure's error.
#[tokio::test]
async fn end_hook_failed_after_nonretryable_flush_failure() {
    let plugin = Lifecycle::default();
    let result = batching_runner(&plugin)
        .fail_checkpoints_after(1, 1)
        .run(orphaned_step, ())
        .await
        .expect("runner");
    assert!(result.is_failure());
    assert_eq!(result.invocation_count(), 1);
    let ends = plugin.ends();
    let [end] = ends.as_slice() else {
        unreachable!("one end info, got {}", ends.len());
    };
    assert_eq!(end.status(), InvocationStatus::Failed);
    assert_eq!(end.execution_result(), None);
    assert_eq!(
        end.execution_error().and_then(WireError::error_type),
        Some("CheckpointUnrecoverableExecutionError")
    );
}

/// A retryable failure of the orphaned outcome write fails the
/// invocation. Its end hook runs once, with `Retrying` and the checkpoint
/// fault's type. A destructor panic of an end hook's error is dropped,
/// because the invocation already fails. The next invocation succeeds.
#[tokio::test]
async fn end_hook_retrying_after_retryable_orphaned_write_failure() {
    let plugin = Lifecycle::default();
    let result = batching_runner(&plugin)
        .fail_checkpoints_after_retryable(1, 1)
        .run(orphaned_step, ())
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 2);
    assert_eq!(
        plugin.statuses(),
        [InvocationStatus::Retrying, InvocationStatus::Succeeded]
    );
    let ends = plugin.ends();
    let first = ends.first().unwrap();
    assert_eq!(first.execution_result(), None);
    assert_eq!(
        first.execution_error().and_then(WireError::error_type),
        Some("CheckpointUnrecoverableInvocationError")
    );
}

/// A custom step and callback subtype reach the operations map of the last
/// invocation's end info, with the value the runner records.
#[tokio::test]
async fn custom_step_and_callback_subtypes_reach_plugin_operations() {
    let plugin = Lifecycle::default();
    let result = lifecycle_runner(&plugin)
        .callback_for(
            "cb",
            durable::test_util::CallbackReply::success(&true).expect("encodes"),
        )
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.step(|_| async { Ok(1_u32) })
                    .name("pay")
                    .sub_type(durable::OperationSubType::new("Payment")?)
                    .await?;
                let cb = ctx
                    .create_callback::<bool>()
                    .name("cb")
                    .sub_type(durable::OperationSubType::new("Approval")?)
                    .await?;
                Ok::<_, BoxError>(cb.result().await?)
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());

    let ends = plugin.ends();
    let operations = ends.last().expect("an end info").operations();
    for (name, sub_type) in [("pay", "Payment"), ("cb", "Approval")] {
        let recorded = result.operation(name).expect("the runner records it");
        let info = operations
            .get(recorded.id())
            .expect("the operation is in the operations map");
        assert_eq!(info.sub_type(), Some(sub_type));
        assert_eq!(info.sub_type(), recorded.sub_type());
    }
}

/// A custom child-context and `with_retry` subtype reach the operations
/// map of the last invocation's end info, with the value the runner
/// records. The `with_retry` subtype labels the block, not its attempts.
#[tokio::test]
async fn custom_child_and_with_retry_subtypes_reach_plugin_operations() {
    use std::sync::atomic::AtomicU32;

    let plugin = Lifecycle::default();
    let work_runs = Arc::new(AtomicU32::new(0));
    let result = lifecycle_runner(&plugin)
        .run(
            move |_e: (), ctx: DurableContext| {
                let work_runs = Arc::clone(&work_runs);
                async move {
                    let b = ctx
                        .run_in_child_context(|_child| async { Ok(2_u32) })
                        .name("case-b")
                        .sub_type(durable::OperationSubType::new("Checkout")?)
                        .await?;
                    let d = ctx
                        .with_retry(move |child| {
                            let work_runs = Arc::clone(&work_runs);
                            async move {
                                let v = child
                                    .step(move |_| async move {
                                        if work_runs.fetch_add(1, Ordering::SeqCst) == 0 {
                                            Err::<u32, BoxError>("transient".into())
                                        } else {
                                            Ok(4)
                                        }
                                    })
                                    .name("work")
                                    .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                                    .await?;
                                Ok(v)
                            }
                        })
                        .name("block")
                        .sub_type(durable::OperationSubType::new("Checkout")?)
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .await?;
                    Ok::<_, BoxError>(b + d)
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());

    let ends = plugin.ends();
    let operations = ends.last().expect("an end info").operations();
    for (name, sub_type) in [
        ("case-b", "Checkout"),
        ("block", "Checkout"),
        ("attempt-1", "RunInChildContext"),
        ("attempt-2", "RunInChildContext"),
    ] {
        let recorded = result.operation(name).expect("the runner records it");
        let info = operations
            .get(recorded.id())
            .expect("the operation is in the operations map");
        assert_eq!(info.sub_type(), Some(sub_type), "{name}");
        assert_eq!(info.sub_type(), recorded.sub_type(), "{name}");
    }
}
