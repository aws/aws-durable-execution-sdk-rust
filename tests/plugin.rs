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
    AttemptEndInfo, AttemptInfo, AttemptOutcome, HookFuture, HookWork, InvocationEndInfo,
    InvocationInfo, OperationInfo, Plugin, PluginFactory,
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

// The operation and attempt hooks of steps and `wait_for_condition`.

/// `(inv, hook, is_replay, status, attempt, outcome)` of one [`Rec`].
type Shape = (
    usize,
    &'static str,
    bool,
    Option<OperationStatus>,
    Option<u32>,
    Option<AttemptOutcome>,
);

/// One operation or attempt hook call that [`Ops`] recorded.
#[derive(Debug, Clone)]
struct Rec {
    /// The invocation, counted from 1.
    inv: usize,
    hook: &'static str,
    /// For an attempt hook, the info of its operation.
    info: OperationInfo,
    /// For an attempt hook, its attempt number.
    attempt: Option<u32>,
    /// For an attempt-end hook, its outcome.
    outcome: Option<AttemptOutcome>,
    /// For an attempt-end hook, its error message.
    attempt_error: Option<String>,
}

impl Rec {
    /// `(inv, hook, is_replay, status, attempt, outcome)`.
    fn shape(&self) -> Shape {
        (
            self.inv,
            self.hook,
            self.info.is_replay(),
            self.info.status(),
            self.attempt,
            self.outcome,
        )
    }

    fn error_message(&self) -> Option<&str> {
        self.info.error().and_then(WireError::error_message)
    }
}

/// Records every operation and attempt hook of a `Step`-kind operation,
/// for every invocation. Its plugins are named `Ops`.
#[derive(Clone, Default)]
struct Ops {
    recs: Arc<Mutex<Vec<Rec>>>,
    count: Arc<AtomicUsize>,
    /// Work `on_operation_end` returns, when set.
    end_work: Option<Arc<AtomicBool>>,
}

impl Ops {
    fn recs(&self) -> Vec<Rec> {
        self.recs.lock().unwrap().clone()
    }

    fn shapes(&self) -> Vec<Shape> {
        self.recs().iter().map(Rec::shape).collect()
    }

    fn find(&self, inv: usize, hook: &str, attempt: Option<u32>) -> Rec {
        self.recs()
            .into_iter()
            .find(|r| r.inv == inv && r.hook == hook && (attempt.is_none() || r.attempt == attempt))
            .unwrap_or_else(|| unreachable!("no {hook} in invocation {inv}"))
    }
}

struct OpsPlugin {
    inv: usize,
    ops: Ops,
}

impl OpsPlugin {
    fn push(
        &self,
        hook: &'static str,
        info: &OperationInfo,
        attempt: Option<u32>,
        outcome: Option<AttemptOutcome>,
        attempt_error: Option<&WireError>,
    ) {
        if info.kind() != Some(OperationKind::Step) {
            return;
        }
        self.ops.recs.lock().unwrap().push(Rec {
            inv: self.inv,
            hook,
            info: info.clone(),
            attempt,
            outcome,
            attempt_error: attempt_error
                .and_then(WireError::error_message)
                .map(str::to_owned),
        });
    }
}

impl Plugin for OpsPlugin {
    fn name(&self) -> &'static str {
        "Ops"
    }

    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.push("start", info, None, None, None);
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.push("end", info, None, None, None);
        let Some(flag) = self.ops.end_work.clone() else {
            return Ok(None);
        };
        Ok(Some(Box::pin(async move {
            tokio::task::yield_now().await;
            flag.store(true, Ordering::SeqCst);
            Ok(())
        })))
    }

    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        self.push(
            "attempt_start",
            info.operation(),
            Some(info.attempt()),
            None,
            None,
        );
        Ok(None)
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        self.push(
            "attempt_end",
            info.operation(),
            Some(info.attempt()),
            Some(info.outcome()),
            info.error(),
        );
        Ok(None)
    }
}

impl PluginFactory for Ops {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let inv = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Some(Box::new(OpsPlugin {
            inv,
            ops: self.clone(),
        })))
    }
}

fn ops_runner(ops: &Ops) -> LocalRunner {
    LocalRunner::new().options(
        Options::builder()
            .plugin(ops.clone())
            .build()
            .expect("valid options"),
    )
}

/// Retries attempts below `stop_at` after 1 s, and stops at `stop_at`.
fn retry_below(
    stop_at: u32,
) -> impl Fn(&durable::StepError, u32) -> RetryDecision + Send + Sync + 'static {
    move |_err, attempt| {
        if attempt < stop_at {
            RetryDecision::Retry {
                delay: Duration::from_secs(1),
            }
        } else {
            RetryDecision::Stop
        }
    }
}

async fn step_a(_e: (), ctx: DurableContext) -> Result<String, BoxError> {
    Ok(ctx
        .step(|_| async { Ok("task-a".to_owned()) })
        .name("a")
        .await?)
}

/// A step that fails its first attempt and succeeds its second.
async fn flaky_step(_e: (), ctx: DurableContext) -> Result<String, BoxError> {
    Ok(ctx
        .step(|step_ctx| async move {
            if step_ctx.attempt() == 1 {
                return Err::<String, BoxError>("attempt 1 failed".into());
            }
            Ok("ok".to_owned())
        })
        .name("flaky")
        .retry_strategy(retry_below(3))
        .await?)
}

const STARTED: Option<OperationStatus> = Some(OperationStatus::Started);
const SUCCEEDED: Option<OperationStatus> = Some(OperationStatus::Succeeded);
const FAILED: Option<OperationStatus> = Some(OperationStatus::Failed);
const PENDING: Option<OperationStatus> = Some(OperationStatus::Pending);
const OK: Option<AttemptOutcome> = Some(AttemptOutcome::Succeeded);
const KO: Option<AttemptOutcome> = Some(AttemptOutcome::Failed);

#[tokio::test]
async fn step_first_attempt_success() {
    let ops = Ops::default();
    let result = ops_runner(&ops).run(step_a, ()).await.expect("runner");
    assert_eq!(result.output().map(String::as_str), Some("task-a"));
    assert_eq!(
        ops.shapes(),
        [
            (1, "start", false, STARTED, None, None),
            (1, "attempt_start", false, STARTED, Some(1), None),
            (1, "attempt_end", false, SUCCEEDED, Some(1), OK),
            (1, "end", false, SUCCEEDED, None, None),
        ]
    );
    let end = ops.find(1, "end", None);
    assert_eq!(end.info.result(), Some("\"task-a\""));
    assert!(end.info.error().is_none());
    assert!(ops.find(1, "attempt_end", Some(1)).attempt_error.is_none());
}

#[tokio::test]
async fn step_retry_then_success() {
    let ops = Ops::default();
    let result = ops_runner(&ops).run(flaky_step, ()).await.expect("runner");
    assert_eq!(result.output().map(String::as_str), Some("ok"));
    assert_eq!(
        ops.shapes(),
        [
            (1, "start", false, STARTED, None, None),
            (1, "attempt_start", false, STARTED, Some(1), None),
            (1, "attempt_end", false, PENDING, Some(1), KO),
            (2, "start", true, STARTED, None, None),
            (2, "attempt_start", false, STARTED, Some(2), None),
            (2, "attempt_end", false, SUCCEEDED, Some(2), OK),
            (2, "end", false, SUCCEEDED, None, None),
        ]
    );
    let first = ops.find(1, "attempt_end", Some(1));
    assert_eq!(first.attempt_error.as_deref(), Some("attempt 1 failed"));
    assert_eq!(first.error_message(), Some("attempt 1 failed"));
}

#[tokio::test]
async fn step_retry_exhaustion() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<u32, BoxError>(
                    ctx.step(|_| async { Err::<u32, BoxError>("boom".into()) })
                        .name("a")
                        .retry_strategy(retry_below(2))
                        .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(!result.is_success());
    assert_eq!(
        ops.shapes(),
        [
            (1, "start", false, STARTED, None, None),
            (1, "attempt_start", false, STARTED, Some(1), None),
            (1, "attempt_end", false, PENDING, Some(1), KO),
            (2, "start", true, STARTED, None, None),
            (2, "attempt_start", false, STARTED, Some(2), None),
            (2, "attempt_end", false, FAILED, Some(2), KO),
            (2, "end", false, FAILED, None, None),
        ]
    );
    let last = ops.find(2, "attempt_end", Some(2));
    assert_eq!(last.attempt_error.as_deref(), Some("boom"));
    let end = ops.find(2, "end", None);
    assert_eq!(end.error_message(), Some("boom"));
    assert_eq!(end.info.result(), None);
}

#[tokio::test]
async fn step_terminal_replay_fires_nothing() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .run(
            |e: (), ctx: DurableContext| async move {
                let value = step_a(e, ctx.clone()).await?;
                ctx.wait(Duration::from_secs(1)).await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success());
    assert_eq!(result.invocation_count(), 2);
    assert!(ops.recs().iter().all(|r| r.inv == 1), "{:?}", ops.shapes());
    assert_eq!(ops.recs().len(), 4);
}

#[tokio::test]
async fn step_reentry_after_interruption() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .fail_checkpoints_after_retryable(1, 1)
        .run(step_a, ())
        .await
        .expect("runner");
    assert!(result.is_success());
    let second: Vec<_> = ops.shapes().into_iter().filter(|s| s.0 == 2).collect();
    assert_eq!(
        second.get(..2),
        Some(
            &[
                (2, "start", true, STARTED, None, None),
                (2, "attempt_start", false, STARTED, Some(1), None),
            ][..]
        ),
        "{:?}",
        ops.shapes()
    );
}

#[tokio::test]
async fn step_at_most_once_interruption_fires_nothing() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .fail_checkpoints_after_retryable(1, 1)
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<_, BoxError>(
                    ctx.step(|_| async { Ok("task-a".to_owned()) })
                        .name("a")
                        .semantics(durable::StepSemantics::AtMostOncePerRetry)
                        .retry_strategy(retry_below(3))
                        .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 3, "{:?}", ops.shapes());
    let shapes = ops.shapes();
    assert!(shapes.iter().all(|s| s.0 != 2), "{shapes:?}");
    let third: Vec<_> = shapes.into_iter().filter(|s| s.0 == 3).collect();
    assert_eq!(
        third.get(..2),
        Some(
            &[
                (3, "start", true, STARTED, None, None),
                (3, "attempt_start", false, STARTED, Some(2), None),
            ][..]
        )
    );
}

#[tokio::test]
async fn step_ids_match_history() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .run(
            |e: (), ctx: DurableContext| async move {
                Ok::<_, BoxError>(
                    ctx.run_in_child_context(move |child| async move { step_a(e, child).await })
                        .name("child")
                        .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    let step = result.operation("a").expect("the step is recorded");
    let child = result.operation("child").expect("the child is recorded");
    let recs = ops.recs();
    assert_eq!(recs.len(), 4);
    for rec in recs {
        assert_eq!(rec.info.id(), step.id(), "{}", rec.hook);
        assert_eq!(rec.info.parent_id(), Some(child.id()), "{}", rec.hook);
    }
}

/// A condition whose state counts the checks, and which completes at 2.
async fn counting_condition(_e: (), ctx: DurableContext) -> Result<u32, BoxError> {
    Ok(ctx
        .wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0_u32)
        .wait_strategy_fn(|state: u32, _attempt| {
            if state >= 2 {
                durable::builders::wait_for_condition::WaitDecision::complete()
            } else {
                durable::builders::wait_for_condition::WaitDecision::continue_with(
                    Duration::from_secs(1),
                )
            }
        })
        .name("cond")
        .await?)
}

#[tokio::test]
async fn wfc_continue_then_complete() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .run(counting_condition, ())
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&2));
    assert_eq!(
        ops.shapes(),
        [
            (1, "start", false, STARTED, None, None),
            (1, "attempt_start", false, STARTED, Some(1), None),
            (1, "attempt_end", false, PENDING, Some(1), OK),
            (2, "start", true, STARTED, None, None),
            (2, "attempt_start", false, STARTED, Some(2), None),
            (2, "attempt_end", false, SUCCEEDED, Some(2), OK),
            (2, "end", false, SUCCEEDED, None, None),
        ]
    );
    assert_eq!(ops.find(1, "attempt_end", Some(1)).info.result(), Some("1"));
    assert_eq!(ops.find(2, "attempt_end", Some(2)).info.result(), Some("2"));
    assert_eq!(ops.find(2, "end", None).info.result(), Some("2"));
    for n in [1, 2] {
        let start = ops.find(n as usize, "attempt_start", Some(n));
        assert!(start.info.start_timestamp().is_some(), "attempt {n}");
    }
    let cond = result.operation("cond").expect("the condition is recorded");
    let first_end = ops.find(1, "attempt_end", Some(1)).info;
    assert_eq!(first_end.start_timestamp(), cond.start_time());
    assert_ne!(first_end.start_timestamp(), first_end.end_timestamp());
}

#[tokio::test]
async fn wfc_check_error() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<u32, BoxError>(
                    ctx.wait_for_condition(
                        |_ctx, _state: u32| async move { Err::<u32, BoxError>("bad".into()) },
                        0_u32,
                    )
                    .name("cond")
                    .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(!result.is_success());
    assert_eq!(
        ops.shapes(),
        [
            (1, "start", false, STARTED, None, None),
            (1, "attempt_start", false, STARTED, Some(1), None),
            (1, "attempt_end", false, FAILED, Some(1), KO),
            (1, "end", false, FAILED, None, None),
        ]
    );
    assert_eq!(
        ops.find(1, "attempt_end", Some(1)).attempt_error.as_deref(),
        Some("bad")
    );
    assert_eq!(ops.find(1, "end", None).error_message(), Some("bad"));
}

#[tokio::test]
async fn wfc_exhaustion() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<u32, BoxError>(
                    ctx.wait_for_condition(|_ctx, state: u32| async move { Ok(state) }, 0_u32)
                        .wait_strategy(
                            durable::builders::wait_for_condition::WaitStrategy::builder(
                                |_state: &u32| false,
                            )
                            .max_attempts(1)
                            .build(),
                        )
                        .name("cond")
                        .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(!result.is_success());
    let ends: Vec<_> = ops
        .recs()
        .into_iter()
        .filter(|r| r.hook == "attempt_end" || r.hook == "end")
        .collect();
    assert_eq!(
        ends.iter().map(Rec::shape).collect::<Vec<_>>(),
        [
            (1, "attempt_end", false, FAILED, Some(1), KO),
            (1, "end", false, FAILED, None, None),
        ]
    );
    for rec in &ends {
        assert_eq!(
            rec.info.error().and_then(WireError::error_type),
            Some("WaitForCondition.MaxChecksExceeded"),
            "{}",
            rec.hook
        );
    }
}

#[tokio::test]
async fn wfc_reentry_after_interruption() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .fail_checkpoints_after_retryable(1, 1)
        .run(counting_condition, ())
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&2));
    let second: Vec<_> = ops.shapes().into_iter().filter(|s| s.0 == 2).collect();
    assert_eq!(
        second.get(..2),
        Some(
            &[
                (2, "start", true, STARTED, None, None),
                (2, "attempt_start", false, STARTED, Some(1), None),
            ][..]
        ),
        "{:?}",
        ops.shapes()
    );
}

#[tokio::test]
async fn attempt_end_carries_end_timestamp() {
    let ops = Ops::default();
    let result = ops_runner(&ops).run(flaky_step, ()).await.expect("runner");
    let step = result.operation("flaky").expect("the step is recorded");
    let first = ops.find(1, "attempt_end", Some(1)).info;
    let day = SystemTime::UNIX_EPOCH + Duration::from_hours(24);
    assert!(first.end_timestamp().expect("an end timestamp") > day);
    assert_eq!(
        ops.find(2, "attempt_end", Some(2)).info.end_timestamp(),
        step.end_time()
    );
    assert_eq!(
        ops.find(2, "end", None).info.end_timestamp(),
        step.end_time()
    );
    assert!(step.end_time().is_some());
}

/// Panics in every operation and attempt hook. Named `Faulty`.
#[derive(Clone)]
struct OpFaulty;

#[expect(clippy::panic)] // reason: the hook panics under test
impl Plugin for OpFaulty {
    fn name(&self) -> &'static str {
        "Faulty"
    }

    fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
        panic!("start boom")
    }

    fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
        panic!("end boom")
    }

    fn on_attempt_start(&self, _info: &AttemptInfo) -> HookWork {
        panic!("attempt start boom")
    }

    fn on_attempt_end(&self, _info: &AttemptEndInfo) -> HookWork {
        panic!("attempt end boom")
    }
}

impl PluginFactory for OpFaulty {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// Returns `Err("x")` from every operation and attempt hook.
#[derive(Clone)]
struct OpFailing;

impl Plugin for OpFailing {
    fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
        Err("x".into())
    }

    fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
        Err("x".into())
    }

    fn on_attempt_start(&self, _info: &AttemptInfo) -> HookWork {
        Err("x".into())
    }

    fn on_attempt_end(&self, _info: &AttemptEndInfo) -> HookWork {
        Err("x".into())
    }
}

impl PluginFactory for OpFailing {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

#[tokio::test]
#[expect(clippy::panic)] // reason: reports that no attempt captured the warnings
async fn panicking_operation_hook_is_isolated() {
    const HOOKS: [&str; 4] = [
        "on_operation_start",
        "on_attempt_start",
        "on_attempt_end",
        "on_operation_end",
    ];
    let alone = LocalRunner::new().run(step_a, ()).await.expect("runner");
    for _attempt in 0..5 {
        let (buffer, _guard) = capture();
        let healthy = Ops::default();
        let result = LocalRunner::new()
            .options(
                Options::builder()
                    .plugin(OpFaulty)
                    .plugin(healthy.clone())
                    .plugin(OpFailing)
                    .build()
                    .expect("valid options"),
            )
            .run(step_a, ())
            .await
            .expect("runner");
        assert!(result.is_success());
        assert_eq!(result.output(), alone.output());
        assert_eq!(
            healthy.shapes(),
            [
                (1, "start", false, STARTED, None, None),
                (1, "attempt_start", false, STARTED, Some(1), None),
                (1, "attempt_end", false, SUCCEEDED, Some(1), OK),
                (1, "end", false, SUCCEEDED, None, None),
            ]
        );
        let warnings = plugin_warnings(&buffer);
        let hooks_of = |message: &str| {
            warnings
                .iter()
                .filter(|(_, _, m)| m == message)
                .cloned()
                .collect::<Vec<_>>()
        };
        let panics = hooks_of("plugin hook panicked; the panic is ignored");
        let failures = hooks_of("plugin hook failed; the error is ignored");
        if panics.len() != HOOKS.len() || failures.len() != HOOKS.len() {
            continue;
        }
        for ((plugin, hook, _), expected) in panics.iter().zip(HOOKS) {
            assert!(plugin.ends_with("Faulty (plugins[0])"), "{plugin}");
            assert_eq!(hook, expected);
        }
        for ((plugin, hook, _), expected) in failures.iter().zip(HOOKS) {
            assert!(plugin.ends_with("(plugins[2])"), "{plugin}");
            assert_eq!(hook, expected);
        }
        return;
    }
    panic!("no attempt captured the eight warnings");
}

#[tokio::test]
async fn step_custom_sub_type_reaches_plugin() {
    let ops = Ops::default();
    let result = ops_runner(&ops)
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<_, BoxError>(
                    ctx.step(|_| async { Ok("ok".to_owned()) })
                        .name("pay")
                        .sub_type(durable::OperationSubType::new("Payment")?)
                        .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    let pay = result.operation("pay").expect("the step is recorded");
    let recs = ops.recs();
    assert_eq!(recs.len(), 4);
    for rec in recs {
        assert_eq!(rec.info.sub_type(), Some("Payment"), "{}", rec.hook);
        assert_eq!(rec.info.sub_type(), pay.sub_type(), "{}", rec.hook);
    }
}

#[tokio::test]
async fn operation_hook_work_runs() {
    let flag = Arc::new(AtomicBool::new(false));
    let ops = Ops {
        end_work: Some(Arc::clone(&flag)),
        ..Ops::default()
    };
    let alone = LocalRunner::new().run(step_a, ()).await.expect("runner");
    let result = ops_runner(&ops).run(step_a, ()).await.expect("runner");
    assert!(flag.load(Ordering::SeqCst));
    assert_eq!(result.output(), alone.output());
}

// The operation hooks of waits and chained invokes.

/// `(inv, hook, kind, name, is_replay, status)` of one operation hook call.
type OpShape = (
    usize,
    &'static str,
    Option<OperationKind>,
    Option<String>,
    bool,
    Option<OperationStatus>,
);

/// Records every operation hook of a wait or a chained invoke, for every
/// invocation.
#[derive(Clone, Default)]
struct Timers {
    recs: Arc<Mutex<Vec<(OpShape, OperationInfo)>>>,
    count: Arc<AtomicUsize>,
}

impl Timers {
    fn shapes(&self) -> Vec<OpShape> {
        self.recs
            .lock()
            .unwrap()
            .iter()
            .map(|(shape, _)| shape.clone())
            .collect()
    }

    /// The `(inv, hook, is_replay, status)` of the operation named `name`.
    fn of(&self, name: &str) -> Vec<(usize, &'static str, bool, Option<OperationStatus>)> {
        self.shapes()
            .into_iter()
            .filter(|s| s.3.as_deref() == Some(name))
            .map(|(inv, hook, _, _, replay, status)| (inv, hook, replay, status))
            .collect()
    }

    fn runner(&self) -> LocalRunner {
        LocalRunner::new().options(
            Options::builder()
                .plugin(self.clone())
                .build()
                .expect("valid options"),
        )
    }
}

struct TimersPlugin {
    inv: usize,
    timers: Timers,
}

impl TimersPlugin {
    fn push(&self, hook: &'static str, info: &OperationInfo) {
        if !matches!(
            info.kind(),
            Some(OperationKind::Wait | OperationKind::ChainedInvoke)
        ) {
            return;
        }
        let shape = (
            self.inv,
            hook,
            info.kind(),
            info.name().map(str::to_owned),
            info.is_replay(),
            info.status(),
        );
        self.timers.recs.lock().unwrap().push((shape, info.clone()));
    }
}

impl Plugin for TimersPlugin {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.push("start", info);
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.push("end", info);
        Ok(None)
    }
}

impl PluginFactory for Timers {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let inv = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Some(Box::new(TimersPlugin {
            inv,
            timers: self.clone(),
        })))
    }
}

#[tokio::test]
async fn wait_hooks_across_three_invocations() {
    let timers = Timers::default();
    let result = timers
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.wait(Duration::from_secs(1)).name("w").await?;
                ctx.wait(Duration::from_secs(1)).name("x").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success());
    assert_eq!(result.invocation_count(), 3);
    let wait = Some(OperationKind::Wait);
    let w = Some("w".to_owned());
    let x = Some("x".to_owned());
    assert_eq!(
        timers.shapes(),
        [
            (1, "start", wait, w.clone(), false, STARTED),
            (2, "end", wait, w.clone(), false, SUCCEEDED),
            (2, "start", wait, x.clone(), false, STARTED),
            (3, "end", wait, w, true, SUCCEEDED),
            (3, "end", wait, x, false, SUCCEEDED),
        ]
    );
}

#[tokio::test]
async fn pending_wait_restarts_with_replay() {
    let timers = Timers::default();
    let result = timers
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let branches: Vec<durable::Branch<String>> = vec![
                    durable::Branch::new("0", |child: DurableContext| async move {
                        child.wait(Duration::from_secs(2)).name("short").await?;
                        Ok("short-done".to_owned())
                    }),
                    durable::Branch::new("1", |child: DurableContext| async move {
                        child.wait(Duration::from_secs(8)).name("long").await?;
                        Ok("long-done".to_owned())
                    }),
                ];
                Ok::<_, BoxError>(ctx.parallel(branches).name("waits").await?)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output().map(Vec::as_slice),
        Some(["short-done".to_owned(), "long-done".to_owned()].as_slice())
    );
    let short = timers.of("short");
    let long = timers.of("long");
    assert_eq!(
        short.iter().filter(|r| r.1 == "start").collect::<Vec<_>>(),
        [&(1, "start", false, STARTED)],
        "{short:?}"
    );
    let short_ends: Vec<_> = short.iter().filter(|r| r.1 == "end").collect();
    assert_eq!(short_ends.len(), 1, "{short:?}");
    let short_end_inv = short_ends.first().unwrap().0;
    assert_eq!(
        short_ends.first().map(|r| (r.2, r.3)),
        Some((false, SUCCEEDED))
    );
    assert_eq!(
        long,
        [
            (1, "start", false, STARTED),
            (short_end_inv, "start", true, STARTED),
            (short_end_inv + 1, "end", false, SUCCEEDED),
        ]
    );
}

#[tokio::test]
async fn invoke_hooks() {
    let timers = Timers::default();
    let result = timers
        .runner()
        .register_function("double", |x: u32| async move { Ok(x * 2) })
        .run(
            |_e: (), ctx: DurableContext| async move {
                let value = ctx.invoke::<u32, _>("double", 21_u32).name("call").await?;
                ctx.wait(Duration::from_secs(1)).await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&42));
    assert_eq!(result.invocation_count(), 3);
    assert_eq!(
        timers.of("call"),
        [
            (1, "start", false, STARTED),
            (2, "end", false, SUCCEEDED),
            (3, "end", true, SUCCEEDED),
        ]
    );
    let kinds: Vec<_> = timers
        .shapes()
        .into_iter()
        .filter(|s| s.3.as_deref() == Some("call"))
        .map(|s| s.2)
        .collect();
    assert!(
        kinds
            .iter()
            .all(|k| *k == Some(OperationKind::ChainedInvoke))
    );
}

#[tokio::test]
async fn invoke_failure_hooks() {
    let timers = Timers::default();
    let result = timers
        .runner()
        .register_function("fails", |(): ()| async move {
            Err::<(), BoxError>("declined".into())
        })
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.invoke::<(), _>("fails", ()).name("call").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");
    assert!(!result.is_success());
    let recs = timers.recs.lock().unwrap().clone();
    let ends: Vec<_> = recs.iter().filter(|(shape, _)| shape.1 == "end").collect();
    assert_eq!(ends.len(), 1, "{:?}", timers.shapes());
    let (shape, info) = ends.first().unwrap();
    assert_eq!(shape.5, FAILED);
    assert!(info.error().is_some(), "{info:?}");
}

// The operation hooks of callbacks and `wait_for_callback`.

/// `(inv, hook, kind, name, is_replay, status)` of one operation or attempt
/// hook call, with its sub-type and result.
#[derive(Clone, Debug, PartialEq)]
struct CbRec {
    shape: OpShape,
    sub_type: Option<String>,
    result: Option<String>,
    error_type: Option<String>,
}

/// Records every operation and attempt hook of a callback, a context and
/// a step, for every invocation.
#[derive(Clone, Default)]
struct Callbacks {
    recs: Arc<Mutex<Vec<CbRec>>>,
    count: Arc<AtomicUsize>,
}

impl Callbacks {
    fn recs(&self) -> Vec<CbRec> {
        self.recs.lock().unwrap().clone()
    }

    /// The `(inv, hook, kind, name, is_replay, status)` of every record.
    fn shapes(&self) -> Vec<OpShape> {
        self.recs().into_iter().map(|r| r.shape).collect()
    }

    /// The records of the operation named `name`.
    fn of(&self, name: &str) -> Vec<CbRec> {
        self.recs()
            .into_iter()
            .filter(|r| r.shape.3.as_deref() == Some(name))
            .collect()
    }

    fn runner(&self) -> LocalRunner {
        LocalRunner::new().options(
            Options::builder()
                .plugin(self.clone())
                .build()
                .expect("valid options"),
        )
    }
}

struct CallbacksPlugin {
    inv: usize,
    callbacks: Callbacks,
}

impl CallbacksPlugin {
    fn push(&self, hook: &'static str, info: &OperationInfo) {
        if !matches!(
            info.kind(),
            Some(OperationKind::Callback | OperationKind::Context | OperationKind::Step)
        ) {
            return;
        }
        let rec = CbRec {
            shape: (
                self.inv,
                hook,
                info.kind(),
                info.name().map(str::to_owned),
                info.is_replay(),
                info.status(),
            ),
            sub_type: info.sub_type().map(str::to_owned),
            result: info.result().map(str::to_owned),
            error_type: info
                .error()
                .and_then(WireError::error_type)
                .map(str::to_owned),
        };
        self.callbacks.recs.lock().unwrap().push(rec);
    }
}

impl Plugin for CallbacksPlugin {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.push("start", info);
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.push("end", info);
        Ok(None)
    }

    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        self.push("attempt_start", info.operation());
        Ok(None)
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        self.push("attempt_end", info.operation());
        Ok(None)
    }
}

impl PluginFactory for Callbacks {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let inv = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Some(Box::new(CallbacksPlugin {
            inv,
            callbacks: self.clone(),
        })))
    }
}

const CALLBACK: Option<OperationKind> = Some(OperationKind::Callback);
const CONTEXT: Option<OperationKind> = Some(OperationKind::Context);
const STEP: Option<OperationKind> = Some(OperationKind::Step);

#[tokio::test]
async fn callback_hooks() {
    let callbacks = Callbacks::default();
    let result = callbacks
        .runner()
        .callback(durable::test_util::CallbackReply::success(&"ok").expect("encodes"))
        .run(
            |_e: (), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<String>().name("cb").await?;
                let value = cb.result().await?;
                ctx.wait(Duration::from_secs(1)).await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output().map(String::as_str), Some("ok"));
    assert_eq!(result.invocation_count(), 3);
    let cb = Some("cb".to_owned());
    let recs = callbacks.of("cb");
    assert_eq!(
        recs.iter().map(|r| r.shape.clone()).collect::<Vec<_>>(),
        [
            (1, "start", CALLBACK, cb.clone(), false, STARTED),
            (2, "end", CALLBACK, cb.clone(), false, SUCCEEDED),
            (3, "end", CALLBACK, cb, true, SUCCEEDED),
        ]
    );
    assert_eq!(
        recs.get(1).and_then(|r| r.result.as_deref()),
        Some("\"ok\""),
        "the end hook carries the raw reply"
    );
}

#[tokio::test]
async fn callback_timeout_hooks() {
    let callbacks = Callbacks::default();
    let result = callbacks
        .runner()
        .callback(durable::test_util::CallbackReply::timeout())
        .run(
            |_e: (), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<String>().name("cb").await?;
                Ok::<_, BoxError>(cb.result().await?)
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_failure());
    let ends: Vec<_> = callbacks
        .of("cb")
        .into_iter()
        .filter(|r| r.shape.1 == "end")
        .map(|r| r.shape.5)
        .collect();
    assert_eq!(ends, [Some(OperationStatus::TimedOut)]);
}

/// Runs a `wait_for_callback` named `name` (or unnamed), with the reply
/// `reply` queued for it, then a 1 s wait.
async fn run_wait_for_callback(
    callbacks: &Callbacks,
    name: Option<&'static str>,
    reply: durable::test_util::CallbackReply,
) -> durable::test_util::TestResult<String> {
    let runner = match name {
        Some(name) => callbacks.runner().callback_for(name, reply),
        None => callbacks.runner().callback(reply),
    };
    runner
        .run(
            move |_e: (), ctx: DurableContext| async move {
                let mut wfcb = ctx.wait_for_callback::<String, _, _>(|_step, _id| async { Ok(()) });
                if let Some(name) = name {
                    wfcb = wfcb.name(name);
                }
                let value = wfcb.await?;
                ctx.wait(Duration::from_secs(1)).await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner")
}

#[tokio::test]
async fn wait_for_callback_hooks() {
    let callbacks = Callbacks::default();
    let result = run_wait_for_callback(
        &callbacks,
        Some("approve"),
        durable::test_util::CallbackReply::success(&"yes").expect("encodes"),
    )
    .await;
    assert_eq!(result.output().map(String::as_str), Some("yes"));
    assert_eq!(result.invocation_count(), 3);
    let approve = Some("approve".to_owned());
    let callback = Some("approve-callback".to_owned());
    let submitter = Some("approve-submitter".to_owned());
    assert_eq!(
        callbacks.shapes(),
        [
            (1, "start", CONTEXT, approve.clone(), false, STARTED),
            (1, "start", CALLBACK, callback.clone(), false, STARTED),
            (1, "start", STEP, submitter.clone(), false, STARTED),
            (1, "attempt_start", STEP, submitter.clone(), false, STARTED),
            (1, "attempt_end", STEP, submitter.clone(), false, SUCCEEDED),
            (1, "end", STEP, submitter, false, SUCCEEDED),
            (2, "start", CONTEXT, approve.clone(), true, STARTED),
            (2, "end", CALLBACK, callback, false, SUCCEEDED),
            (2, "end", CONTEXT, approve, false, SUCCEEDED),
        ]
    );
    let end = callbacks
        .of("approve")
        .into_iter()
        .find(|r| r.shape.1 == "end")
        .expect("the context ended");
    assert_eq!(end.result.as_deref(), Some("\"yes\""));
    assert_eq!(end.sub_type.as_deref(), Some("WaitForCallback"));
}

#[tokio::test]
async fn wait_for_callback_failure_hooks() {
    let callbacks = Callbacks::default();
    let result = run_wait_for_callback(
        &callbacks,
        Some("approve"),
        durable::test_util::CallbackReply::failure(durable::test_util::CallbackFailure::new(
            "Declined", "no",
        )),
    )
    .await;
    assert!(result.is_failure());
    let ends: Vec<_> = callbacks
        .recs()
        .into_iter()
        .filter(|r| r.shape.0 == 2 && r.shape.1 == "end")
        .map(|r| (r.shape.3, r.shape.5, r.error_type))
        .collect();
    assert_eq!(
        ends,
        [
            (
                Some("approve-callback".to_owned()),
                FAILED,
                Some("Declined".to_owned())
            ),
            // The outer record carries the fixed external-failure type; the
            // external caller's type stays on the callback's own record.
            (
                Some("approve".to_owned()),
                FAILED,
                Some("Callback.ExternalFailure".to_owned())
            ),
        ]
    );
}

#[tokio::test]
async fn unnamed_wait_for_callback_has_unnamed_inner_ops() {
    let callbacks = Callbacks::default();
    let result = run_wait_for_callback(
        &callbacks,
        None,
        durable::test_util::CallbackReply::success(&"yes").expect("encodes"),
    )
    .await;
    assert_eq!(result.output().map(String::as_str), Some("yes"));
    let inner: Vec<_> = callbacks
        .recs()
        .into_iter()
        .filter(|r| {
            matches!(
                r.shape.2,
                Some(OperationKind::Callback | OperationKind::Step)
            )
        })
        .collect();
    assert_ne!(inner.len(), 0);
    assert!(inner.iter().all(|r| r.shape.3.is_none()), "{inner:?}");
}

#[tokio::test]
async fn callback_custom_sub_type_reaches_plugin() {
    let callbacks = Callbacks::default();
    let result = callbacks
        .runner()
        .callback(durable::test_util::CallbackReply::success(&"ok").expect("encodes"))
        .callback_for(
            "approve",
            durable::test_util::CallbackReply::success(&"yes").expect("encodes"),
        )
        .run(
            |_e: (), ctx: DurableContext| async move {
                let cb = ctx
                    .create_callback::<String>()
                    .name("cb")
                    .sub_type(durable::OperationSubType::new("Approval")?)
                    .await?;
                let first = cb.result().await?;
                let second = ctx
                    .wait_for_callback::<String, _, _>(|_step, _id| async { Ok(()) })
                    .name("approve")
                    .await?;
                Ok::<_, BoxError>(format!("{first}/{second}"))
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output().map(String::as_str), Some("ok/yes"));
    let recorded = result
        .operation("cb")
        .and_then(durable::test_util::TestOperation::sub_type)
        .map(str::to_owned);
    assert_eq!(recorded.as_deref(), Some("Approval"));
    let cb = callbacks.of("cb");
    assert_ne!(cb.len(), 0);
    assert!(cb.iter().all(|r| r.sub_type == recorded), "{cb:?}");
    let inner = callbacks.of("approve-callback");
    assert_ne!(inner.len(), 0);
    assert!(
        inner
            .iter()
            .all(|r| r.sub_type.as_deref() == Some("Callback")),
        "{inner:?}"
    );
}

// The operation hooks of child contexts, `map`, `parallel`, their items, and
// the combinators.

/// One hook call that [`Tree`] recorded: an operation or attempt hook with
/// its operation info, or an invocation end with its status.
#[derive(Debug, Clone)]
struct TreeRec {
    /// The invocation, counted from 1.
    inv: usize,
    hook: &'static str,
    info: Option<OperationInfo>,
    invocation_status: Option<InvocationStatus>,
}

/// `(inv, hook, name, is_replay, status)` of one operation hook record.
type TreeShape = (usize, &'static str, String, bool, Option<OperationStatus>);

impl TreeRec {
    fn info(&self) -> &OperationInfo {
        self.info.as_ref().expect("an operation hook record")
    }

    fn name(&self) -> Option<&str> {
        self.info.as_ref().and_then(OperationInfo::name)
    }

    fn shape(&self) -> TreeShape {
        let info = self.info();
        (
            self.inv,
            self.hook,
            info.name().unwrap_or_default().to_owned(),
            info.is_replay(),
            info.status(),
        )
    }

    fn error_message(&self) -> Option<&str> {
        self.info().error().and_then(WireError::error_message)
    }
}

/// Records every operation and attempt hook and every invocation end, for
/// every invocation. With `contexts_only`, it records only operations of
/// kind `Context`.
#[derive(Clone, Default)]
struct Tree {
    recs: Arc<Mutex<Vec<TreeRec>>>,
    count: Arc<AtomicUsize>,
    contexts_only: bool,
}

impl Tree {
    fn contexts() -> Self {
        Self {
            contexts_only: true,
            ..Self::default()
        }
    }

    fn recs(&self) -> Vec<TreeRec> {
        self.recs.lock().unwrap().clone()
    }

    /// The operation and attempt hook records of the operation `name`.
    fn of(&self, name: &str) -> Vec<TreeRec> {
        self.recs()
            .into_iter()
            .filter(|r| r.name() == Some(name))
            .collect()
    }

    /// The operation hook records (start and end), in order.
    fn operation_hooks(&self) -> Vec<TreeRec> {
        self.recs()
            .into_iter()
            .filter(|r| r.hook == "start" || r.hook == "end")
            .collect()
    }

    fn shapes(&self) -> Vec<TreeShape> {
        self.operation_hooks().iter().map(TreeRec::shape).collect()
    }

    /// The operation hook shapes of the operation `name`.
    fn shapes_of(&self, name: &str) -> Vec<TreeShape> {
        self.operation_hooks()
            .iter()
            .filter(|r| r.name() == Some(name))
            .map(TreeRec::shape)
            .collect()
    }

    /// The one record of `hook` for `name` in invocation `inv`.
    fn find(&self, inv: usize, hook: &str, name: &str) -> TreeRec {
        let found: Vec<TreeRec> = self
            .of(name)
            .into_iter()
            .filter(|r| r.inv == inv && r.hook == hook)
            .collect();
        assert_eq!(found.len(), 1, "{hook} of {name} in {inv}: {found:?}");
        found.into_iter().next().expect("checked above")
    }

    fn runner(&self) -> LocalRunner {
        LocalRunner::new().options(
            Options::builder()
                .plugin(self.clone())
                .build()
                .expect("valid options"),
        )
    }
}

struct TreePlugin {
    inv: usize,
    tree: Tree,
}

impl TreePlugin {
    fn push(&self, hook: &'static str, info: &OperationInfo) {
        if self.tree.contexts_only && info.kind() != Some(OperationKind::Context) {
            return;
        }
        self.tree.recs.lock().unwrap().push(TreeRec {
            inv: self.inv,
            hook,
            info: Some(info.clone()),
            invocation_status: None,
        });
    }
}

impl Plugin for TreePlugin {
    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.tree.recs.lock().unwrap().push(TreeRec {
            inv: self.inv,
            hook: "invocation_end",
            info: None,
            invocation_status: Some(info.status()),
        });
        Box::pin(std::future::ready(Ok(())))
    }

    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.push("start", info);
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.push("end", info);
        Ok(None)
    }

    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        self.push("attempt_start", info.operation());
        Ok(None)
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        self.push("attempt_end", info.operation());
        Ok(None)
    }
}

impl PluginFactory for Tree {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let inv = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Some(Box::new(TreePlugin {
            inv,
            tree: self.clone(),
        })))
    }
}

/// The wire ID of a positional ID: its SHA-256 digest in lowercase hex.
fn wire(positional: &str) -> String {
    use sha2::Digest as _;
    use std::fmt::Write as _;
    sha2::Sha256::digest(positional.as_bytes())
        .iter()
        .fold(String::new(), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

/// Asserts that no operation hook record has `is_virtual` true.
fn assert_none_virtual(tree: &Tree) {
    let recs = tree.recs();
    assert!(
        recs.iter()
            .filter_map(|r| r.info.as_ref())
            .all(|info| !info.is_virtual()),
        "{recs:?}"
    );
}

/// A child context `parent` holding the step `child`, which returns `"a"`.
async fn parent_with_child_step(_e: (), ctx: DurableContext) -> Result<String, BoxError> {
    Ok(ctx
        .run_in_child_context(|child| async move {
            Ok(child
                .step(|_| async { Ok("a".to_owned()) })
                .name("child")
                .await?)
        })
        .name("parent")
        .await?)
}

/// Fails the first run of the block and succeeds every later one.
fn fails_first_run(
    runs: Arc<AtomicUsize>,
) -> impl Fn(DurableContext) -> std::future::Ready<Result<u32, BoxError>> + Send + Sync + 'static {
    move |_child: DurableContext| {
        std::future::ready(if runs.fetch_add(1, Ordering::SeqCst) == 0 {
            Err("transient".into())
        } else {
            Ok(4)
        })
    }
}

/// Retries the first attempt after 1 s, and stops at the second.
fn retry_once(_err: &durable::StepError, attempt: u32) -> RetryDecision {
    if attempt >= 2 {
        RetryDecision::Stop
    } else {
        RetryDecision::Retry {
            delay: Duration::from_secs(1),
        }
    }
}

#[tokio::test]
async fn nested_hooks_in_lifecycle_order() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(parent_with_child_step, ())
        .await
        .expect("runner");
    assert!(result.is_success());
    let order: Vec<(String, &str)> = tree
        .operation_hooks()
        .iter()
        .map(|r| (r.name().unwrap_or_default().to_owned(), r.hook))
        .collect();
    assert_eq!(
        order,
        [
            ("parent".to_owned(), "start"),
            ("child".to_owned(), "start"),
            ("child".to_owned(), "end"),
            ("parent".to_owned(), "end"),
        ]
    );
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn nested_step_reports_child_as_parent() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(parent_with_child_step, ())
        .await
        .expect("runner");
    assert!(result.is_success());
    let parent = tree.find(1, "end", "parent");
    let child = tree.find(1, "end", "child");
    assert_eq!(child.info().parent_id(), Some(parent.info().id()));
    assert_eq!(parent.info().parent_id(), None);
    assert_eq!(
        parent.info().id(),
        result.operation("parent").expect("recorded").id()
    );
    assert_eq!(
        child.info().id(),
        result.operation("child").expect("recorded").id()
    );
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn child_failure_end_carries_error() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<u32, BoxError>(
                    ctx.run_in_child_context(|_child| async {
                        Err::<u32, BoxError>("boom".into())
                    })
                    .name("parent")
                    .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_failure());
    let end = tree.find(1, "end", "parent");
    assert_eq!(end.info().status(), FAILED);
    assert!(!end.info().is_replay());
    let message = end.error_message().expect("an error message");
    assert!(message.contains("boom"), "{message}");
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn terminal_child_replay_fires_nothing() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(
            |e: (), ctx: DurableContext| async move {
                let value = parent_with_child_step(e, ctx.clone()).await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success());
    assert_eq!(result.invocation_count(), 2);
    let in_second: Vec<TreeRec> = tree
        .recs()
        .into_iter()
        .filter(|r| r.inv == 2 && matches!(r.name(), Some("parent" | "child")))
        .collect();
    assert!(in_second.is_empty(), "{in_second:?}");
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn resumed_child_starts_with_replay() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.run_in_child_context(|child| async move {
                    child.wait(Duration::from_secs(1)).name("pause").await?;
                    Ok(1_u32)
                })
                .name("parent")
                .await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success());
    assert_eq!(
        tree.shapes_of("parent"),
        [
            (1, "start", "parent".to_owned(), false, STARTED),
            (2, "start", "parent".to_owned(), true, STARTED),
            (2, "end", "parent".to_owned(), false, SUCCEEDED),
        ]
    );
    assert_none_virtual(&tree);
}

/// A child whose 300 KB result `ReplayChildren` keeps out of its record
/// fires its operation hooks and S, W and E in the first invocation. The
/// second invocation reruns the body to rebuild the value, and fires no
/// hook of either kind (rule 7.8).
#[tokio::test]
async fn replay_children_reconstruction_fires_no_hook() {
    let tree = Tree::default();
    let uf = Uf::default();
    let runs = Arc::new(AtomicUsize::new(0));
    let handler_runs = Arc::clone(&runs);
    let result = LocalRunner::new()
        .options(
            uf.options()
                .plugin(tree.clone())
                .build()
                .expect("valid options"),
        )
        .run(
            move |_e: (), ctx: DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move {
                    let big = ctx
                        .run_in_child_context(move |_child| async move {
                            runs.fetch_add(1, Ordering::SeqCst);
                            Ok("x".repeat(300 * 1024))
                        })
                        .name("parent")
                        .await?;
                    ctx.wait(Duration::from_secs(1)).name("pause").await?;
                    Ok::<_, BoxError>(big.len())
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&(300 * 1024)));
    assert_eq!(result.invocation_count(), 2);
    assert!(replay_children(&result, "parent"));
    assert_eq!(
        runs.load(Ordering::SeqCst),
        2,
        "the second run rebuilds the value"
    );
    let end = tree.find(1, "end", "parent");
    assert_eq!(end.info().status(), SUCCEEDED);
    assert!(!end.info().is_replay());
    assert_eq!(end.info().result(), None);
    assert!(
        tree.of("parent").iter().all(|r| r.inv == 1),
        "{:?}",
        tree.of("parent")
    );
    assert_none_virtual(&tree);
    uf.assert_full(1, "parent");
    assert_eq!(uf.lines(2), Vec::<String>::new());
}

#[tokio::test]
async fn parallel_branch_hooks() {
    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let branches: Vec<durable::Branch<String>> = vec![
                    durable::Branch::new("first", |_child| async { Ok("task-1".to_owned()) }),
                    durable::Branch::new("second", |_child| async { Ok("task-2".to_owned()) }),
                ];
                Ok::<_, BoxError>(
                    ctx.parallel(branches)
                        .name("parallel")
                        .max_concurrency(1)
                        .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success());
    let s = |hook, name: &str, status| (1, hook, name.to_owned(), false, status);
    assert_eq!(
        tree.shapes(),
        [
            s("start", "parallel", STARTED),
            s("start", "first", STARTED),
            s("end", "first", SUCCEEDED),
            s("start", "second", STARTED),
            s("end", "second", SUCCEEDED),
            s("end", "parallel", SUCCEEDED),
        ]
    );
    let parallel = tree.find(1, "start", "parallel");
    assert_eq!(parallel.info().sub_type(), Some("Parallel"));
    let parallel_end = tree.find(1, "end", "parallel");
    let record = result.operation("parallel").expect("recorded").result();
    assert!(record.is_some());
    assert_eq!(parallel_end.info().result(), record);
    for (name, value) in [("first", "\"task-1\""), ("second", "\"task-2\"")] {
        for rec in tree.of(name) {
            assert_eq!(rec.info().parent_id(), Some(parallel.info().id()), "{name}");
            assert_eq!(rec.info().sub_type(), Some("ParallelBranch"), "{name}");
        }
        assert_eq!(tree.find(1, "end", name).info().result(), Some(value));
    }
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn map_item_hooks() {
    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<_, BoxError>(
                    ctx.map(
                        vec![1_u32, 2],
                        |_child, item, _idx| async move { Ok(item * 2) },
                    )
                    .name("m")
                    .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output().map(Vec::as_slice),
        Some([2_u32, 4].as_slice())
    );
    let map = tree.find(1, "start", "m");
    assert_eq!(map.info().sub_type(), Some("Map"));
    let items: Vec<TreeRec> = tree
        .operation_hooks()
        .into_iter()
        .filter(|r| r.name() != Some("m"))
        .collect();
    assert_eq!(items.iter().filter(|r| r.hook == "start").count(), 2);
    assert_eq!(items.iter().filter(|r| r.hook == "end").count(), 2);
    for rec in &items {
        assert_eq!(rec.info().sub_type(), Some("MapIteration"));
        assert_eq!(rec.info().parent_id(), Some(map.info().id()));
    }
    let ids: Vec<&str> = items
        .iter()
        .filter(|r| r.hook == "start")
        .map(|r| r.info().id())
        .collect();
    assert_eq!(ids, [wire("1-1"), wire("1-2")]);
    let last = tree.operation_hooks().pop().expect("a record");
    assert_eq!(last.shape(), (1, "end", "m".to_owned(), false, SUCCEEDED));
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn early_completed_batch_skips_aborted_items() {
    let tree = Tree::default();
    let notify = Arc::new(tokio::sync::Notify::new());
    let result = tree
        .runner()
        .run(
            move |_e: (), ctx: DurableContext| {
                let notify = Arc::clone(&notify);
                async move {
                    let settles = Arc::clone(&notify);
                    let stuck = Arc::clone(&notify);
                    let branches: Vec<durable::Branch<u32>> = vec![
                        durable::Branch::new("fast", move |child: DurableContext| async move {
                            Ok(child
                                .step(move |_| async move {
                                    settles.notified().await;
                                    Ok(1_u32)
                                })
                                .name("settles")
                                .await?)
                        }),
                        durable::Branch::new("slow", move |child: DurableContext| async move {
                            Ok(child
                                .step(move |_| async move {
                                    stuck.notify_one();
                                    std::future::pending::<()>().await;
                                    Ok(2_u32)
                                })
                                .name("stuck")
                                .await?)
                        }),
                    ];
                    let batch = ctx
                        .parallel(branches)
                        .name("p")
                        .completion(
                            durable::builders::map_parallel::CompletionConfig::with_min_successful(
                                1,
                            ),
                        )
                        .await_batch()
                        .await?;
                    Ok::<_, BoxError>(batch.succeeded().count())
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&1));
    let slow: Vec<&str> = tree.of("slow").iter().map(|r| r.hook).collect();
    assert_eq!(slow, ["start"]);
    let stuck: Vec<&str> = tree.of("stuck").iter().map(|r| r.hook).collect();
    assert_eq!(stuck, ["start", "attempt_start"]);
    let p_end = tree.find(1, "end", "p");
    assert_eq!(p_end.info().status(), SUCCEEDED);
    assert_none_virtual(&tree);
}

/// Runs `parallel` named `p` with `max_concurrency(1)` and
/// `with_min_successful(1)` under `nesting`. Branch 0 returns at once, so
/// the batch completes before branch 1 passes the dispatch gate.
async fn run_undispatched_branch(nesting: durable::builders::map_parallel::NestingMode) -> Tree {
    use durable::builders::map_parallel::CompletionConfig;
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(
            move |_e: (), ctx: DurableContext| async move {
                let branches: Vec<durable::Branch<u32>> = vec![
                    durable::Branch::new("first", |child: DurableContext| async move {
                        Ok(child.step(|_| async { Ok(1_u32) }).name("done").await?)
                    }),
                    durable::Branch::new("later", |child: DurableContext| async move {
                        Ok(child.step(|_| async { Ok(2_u32) }).name("never").await?)
                    }),
                ];
                let batch = ctx
                    .parallel(branches)
                    .name("p")
                    .max_concurrency(1)
                    .nesting(nesting)
                    .completion(CompletionConfig::with_min_successful(1))
                    .await_batch()
                    .await?;
                Ok::<_, BoxError>(batch.succeeded().count())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&1));
    tree
}

#[tokio::test]
async fn undispatched_item_fires_no_hooks() {
    use durable::builders::map_parallel::NestingMode;
    for nesting in [NestingMode::Normal, NestingMode::Flat] {
        let tree = run_undispatched_branch(nesting).await;
        assert_eq!(tree.shapes_of("first").len(), 2, "{nesting:?}");
        assert!(
            tree.of("later").is_empty(),
            "{nesting:?}: {:?}",
            tree.of("later")
        );
        assert!(
            tree.of("never").is_empty(),
            "{nesting:?}: {:?}",
            tree.of("never")
        );
        let later_id = wire("1-2");
        assert!(
            tree.recs()
                .iter()
                .filter_map(|r| r.info.as_ref())
                .all(|info| info.id() != later_id),
            "{nesting:?}"
        );
        let p_end = tree.find(1, "end", "p");
        assert_eq!(p_end.info().status(), SUCCEEDED, "{nesting:?}");
    }
}

#[tokio::test]
async fn combinator_hooks() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
                let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
                Ok::<_, BoxError>(ctx.try_join_all([a, b]).name("gather").await?)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output().map(Vec::as_slice),
        Some([1_u32, 2].as_slice())
    );
    let hooks = tree.operation_hooks();
    let position = |hook: &str, name: &str| {
        hooks
            .iter()
            .position(|r| r.hook == hook && r.name() == Some(name))
            .expect("the hook ran")
    };
    let gather_start = position("start", "gather");
    assert_eq!(
        hooks.get(gather_start).map(TreeRec::shape),
        Some((1, "start", "gather".to_owned(), false, STARTED))
    );
    // The inputs' queue-time start hooks fire before the combinator's.
    assert!(position("start", "a") < gather_start);
    assert!(position("start", "b") < gather_start);
    let last = hooks.last().expect("a record");
    assert_eq!(
        last.shape(),
        (1, "end", "gather".to_owned(), false, SUCCEEDED)
    );
    assert_eq!(last.info().result(), Some("[1,2]"));
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn resumed_batch_and_item_start_with_replay() {
    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<_, BoxError>(
                    ctx.map(vec![1_u32], |child, item, _idx| async move {
                        child.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok(item)
                    })
                    .name("m")
                    .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output().map(Vec::as_slice), Some([1_u32].as_slice()));
    let item = "map-item-0".to_owned();
    assert_eq!(
        tree.shapes(),
        [
            (1, "start", "m".to_owned(), false, STARTED),
            (1, "start", item.clone(), false, STARTED),
            (2, "start", "m".to_owned(), true, STARTED),
            (2, "start", item.clone(), true, STARTED),
            (2, "end", item, false, SUCCEEDED),
            (2, "end", "m".to_owned(), false, SUCCEEDED),
        ]
    );
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn resumed_combinator_starts_with_replay() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let pause = ctx.wait(Duration::from_secs(1)).name("pause").future();
                ctx.try_join_all([pause]).name("gather").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success());
    assert_eq!(
        tree.shapes_of("gather"),
        [
            (1, "start", "gather".to_owned(), false, STARTED),
            (2, "start", "gather".to_owned(), true, STARTED),
            (2, "end", "gather".to_owned(), false, SUCCEEDED),
        ]
    );
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn map_item_failure_end_carries_error() {
    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let outcome = ctx
                    .map(vec![1_u32], |_child, _item, _idx| async {
                        Err::<u32, BoxError>("bad".into())
                    })
                    .name("m")
                    .await;
                Ok::<_, BoxError>(outcome.is_err())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&true));
    let end = tree.find(1, "end", "map-item-0");
    assert_eq!(end.info().status(), FAILED);
    assert!(!end.info().is_replay());
    let message = end.error_message().expect("an error message");
    assert!(message.contains("bad"), "{message}");
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn combinator_failure_end_carries_error() {
    let tree = Tree::default();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let failing = ctx
                    .step(|_| async { Err::<u32, BoxError>("boom".into()) })
                    .name("a")
                    .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                    .future();
                let outcome = ctx.try_join_all([failing]).name("gather").await;
                Ok::<_, BoxError>(outcome.is_err())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&true));
    let end = tree.find(1, "end", "gather");
    assert_eq!(end.info().status(), FAILED);
    assert!(!end.info().is_replay());
    let message = end.error_message().expect("an error message");
    assert!(message.contains("boom"), "{message}");
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn panic_fires_no_operation_end_hooks() {
    let tree = Tree::contexts();
    let runner = tree.runner();
    #[expect(clippy::disallowed_methods)]
    // reason: the test observes the panic through a JoinHandle
    let handle = tokio::spawn(async move {
        runner
            .run(
                |_e: (), ctx: DurableContext| async move {
                    let branches: Vec<durable::Branch<u32>> = vec![
                        durable::Branch::new("ok", |child: DurableContext| async move {
                            Ok(child.step(|_| async { Ok(1_u32) }).name("one").await?)
                        }),
                        durable::Branch::new("bad", |child: DurableContext| async move {
                            #[expect(clippy::panic)] // reason: the step panic under test
                            let value = child
                                .step(|_| async { panic!("kaboom") })
                                .name("panics")
                                .await?;
                            Ok(value)
                        }),
                    ];
                    Ok::<_, BoxError>(ctx.parallel(branches).name("p").max_concurrency(1).await?)
                },
                (),
            )
            .await
    });
    let err = handle.await.expect_err("the run panics");
    assert!(err.is_panic());
    let payload = err.into_panic();
    let message = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    assert!(message.contains("kaboom"), "{message}");
    let recs = tree.recs();
    let summary: Vec<(String, &str, Option<InvocationStatus>)> = recs
        .iter()
        .map(|r| {
            (
                r.name().unwrap_or_default().to_owned(),
                r.hook,
                r.invocation_status,
            )
        })
        .collect();
    assert_eq!(
        summary,
        [
            ("p".to_owned(), "start", None),
            ("ok".to_owned(), "start", None),
            ("ok".to_owned(), "end", None),
            ("bad".to_owned(), "start", None),
            (
                String::new(),
                "invocation_end",
                Some(InvocationStatus::Retrying)
            ),
        ]
    );
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn child_custom_sub_type_reaches_plugin() {
    let tree = Tree::default();
    let runs = Arc::new(AtomicUsize::new(0));
    let result = tree
        .runner()
        .run(
            move |_e: (), ctx: DurableContext| {
                let block = fails_first_run(Arc::clone(&runs));
                async move {
                    ctx.run_in_child_context(|child| async move {
                        Ok(child
                            .step(|_| async { Ok("a".to_owned()) })
                            .name("child")
                            .await?)
                    })
                    .name("parent")
                    .sub_type(durable::OperationSubType::new("Checkout")?)
                    .await?;
                    ctx.with_retry(block)
                        .name("block")
                        .sub_type(durable::OperationSubType::new("Checkout")?)
                        .retry_strategy(retry_once)
                        .await?;
                    Ok::<_, BoxError>(())
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    for (name, sub_type) in [
        ("parent", "Checkout"),
        ("block", "Checkout"),
        ("attempt-1", "RunInChildContext"),
        ("attempt-2", "RunInChildContext"),
        ("child", "Step"),
    ] {
        let recorded = result.operation(name).expect("the runner records it");
        let recs = tree.of(name);
        assert_ne!(recs.len(), 0, "{name}");
        for rec in recs {
            assert_eq!(rec.info().id(), recorded.id(), "{name}");
            assert_eq!(rec.info().sub_type(), Some(sub_type), "{name}");
            assert_eq!(rec.info().sub_type(), recorded.sub_type(), "{name}");
        }
    }
    assert_none_virtual(&tree);
}

#[tokio::test]
async fn with_retry_attempt_hooks() {
    let tree = Tree::contexts();
    let runs = Arc::new(AtomicUsize::new(0));
    let result = tree
        .runner()
        .run(
            move |_e: (), ctx: DurableContext| {
                let block = fails_first_run(Arc::clone(&runs));
                async move {
                    Ok::<_, BoxError>(
                        ctx.with_retry(block)
                            .name("block")
                            .retry_strategy(retry_once)
                            .await?,
                    )
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&4));
    let s = |inv, hook, name: &str, replay, status| (inv, hook, name.to_owned(), replay, status);
    assert_eq!(
        tree.shapes(),
        [
            s(1, "start", "block", false, STARTED),
            s(1, "start", "attempt-1", false, STARTED),
            s(1, "end", "attempt-1", false, FAILED),
            s(2, "start", "block", true, STARTED),
            s(2, "start", "attempt-2", false, STARTED),
            s(2, "end", "attempt-2", false, SUCCEEDED),
            s(2, "end", "block", false, SUCCEEDED),
        ]
    );
    assert_none_virtual(&tree);
}

/// One expected hook of a flat item: the invocation, the hook and
/// `is_replay`.
type FlatHook = (usize, &'static str, bool);

/// The hooks of a flat item that waits 1 s and then returns: it starts
/// live in invocation 1 and suspends; in invocation 2 it starts replaying
/// its recorded wait, and ends after it, when no further record is ahead.
const WAITING_FLAT_ITEM: [FlatHook; 3] =
    [(1, "start", false), (2, "start", true), (2, "end", false)];

/// The hooks of a flat item that returns at once, in a batch that runs in
/// two invocations: a flat item writes no record, so it runs again, live,
/// in invocation 2.
const RETURNING_FLAT_ITEM: [FlatHook; 4] = [
    (1, "start", false),
    (1, "end", false),
    (2, "start", false),
    (2, "end", false),
];

/// Asserts the virtual hooks of the flat items `names` of the batch `batch`
/// with item sub-type `sub_type`, positional ID `batch_positional`: item
/// `i` records exactly the hooks `expected[i]`, with the batch as its
/// parent, at the ID `<batch>-<i+1>`, and writes no record.
fn assert_flat_items<O>(
    tree: &Tree,
    result: &durable::test_util::TestResult<O>,
    batch: &str,
    batch_positional: &str,
    names: [&str; 2],
    sub_type: &str,
    expected: [&[FlatHook]; 2],
) {
    let batch_id = tree.find(1, "start", batch).info().id().to_owned();
    assert_eq!(batch_id, wire(batch_positional));
    for (i, (name, expected)) in names.into_iter().zip(expected).enumerate() {
        let recs: Vec<TreeRec> = tree
            .of(name)
            .into_iter()
            .filter(|r| r.info().parent_id() == Some(batch_id.as_str()))
            .collect();
        let hooks: Vec<FlatHook> = recs
            .iter()
            .map(|r| (r.inv, r.hook, r.info().is_replay()))
            .collect();
        assert_eq!(hooks, expected, "{name}");
        let id = wire(&format!("{batch_positional}-{}", i + 1));
        for rec in &recs {
            let info = rec.info();
            assert_eq!(info.kind(), Some(OperationKind::Context), "{name}");
            assert_eq!(info.sub_type(), Some(sub_type), "{name}");
            assert!(info.is_virtual(), "{name}");
            assert_eq!(info.id(), id, "{name}");
            if rec.hook == "end" {
                assert_eq!(info.status(), SUCCEEDED, "{name}");
                assert_eq!(info.result(), None, "{name}");
            } else {
                // A flat item writes no START, so its start hook carries
                // no status, live and on replay.
                assert_eq!(info.status(), None, "{name}");
            }
        }
        assert!(
            result.operations().iter().all(|op| op.id() != id),
            "{name} has a record"
        );
    }
    assert!(
        tree.of(batch).iter().all(|r| !r.info().is_virtual()),
        "{batch}"
    );
}

/// Asserts that the flat item `name` of the batch at `batch_positional`,
/// abandoned at early completion, records a start and no end.
fn assert_abandoned_flat_item(tree: &Tree, batch_positional: &str, name: &str) {
    let batch_id = wire(batch_positional);
    let hooks: Vec<(&str, bool)> = tree
        .of(name)
        .iter()
        .filter(|r| r.info().parent_id() == Some(batch_id.as_str()))
        .map(|r| (r.hook, r.info().is_virtual()))
        .collect();
    assert_eq!(hooks, [("start", true)], "{name}");
}

#[tokio::test]
async fn flat_map_item_fires_virtual_hooks() {
    use durable::builders::map_parallel::{CompletionConfig, NestingMode};

    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let values = ctx
                    .map(vec![0_u32, 1], |child, item, idx| async move {
                        if idx == 0 {
                            child.wait(Duration::from_secs(1)).name("pause").await?;
                        }
                        Ok(item)
                    })
                    .name("m")
                    .nesting(NestingMode::Flat)
                    .await?;
                let early = ctx
                    .map(vec![0_u32, 1], |_child, item, idx| async move {
                        if idx == 1 {
                            std::future::pending::<()>().await;
                        }
                        Ok(item)
                    })
                    .name("early")
                    .nesting(NestingMode::Flat)
                    .completion(CompletionConfig::with_min_successful(1))
                    .await_batch()
                    .await?;
                Ok::<_, BoxError>((values, early.succeeded().count()))
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&(vec![0_u32, 1], 1)));
    assert_eq!(result.invocation_count(), 2);
    assert_flat_items(
        &tree,
        &result,
        "m",
        "1",
        ["map-item-0", "map-item-1"],
        "MapIteration",
        [&WAITING_FLAT_ITEM, &RETURNING_FLAT_ITEM],
    );
    assert_abandoned_flat_item(&tree, "2", "map-item-1");
    // Item 0's first operation, its wait, has a record in invocation 2, so
    // its context starts replaying there. Item 1 holds no operation.
    let m_id = wire("1");
    let starts = |name: &str| -> Vec<(usize, bool)> {
        tree.of(name)
            .iter()
            .filter(|r| r.hook == "start" && r.info().parent_id() == Some(m_id.as_str()))
            .map(|r| (r.inv, r.info().is_replay()))
            .collect()
    };
    assert_eq!(starts("map-item-0"), [(1, false), (2, true)]);
    assert_eq!(starts("map-item-1"), [(1, false), (2, false)]);
}

#[tokio::test]
async fn flat_parallel_branch_fires_virtual_hooks() {
    use durable::builders::map_parallel::{CompletionConfig, NestingMode};

    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let branches: Vec<durable::Branch<String>> = vec![
                    durable::Branch::new("first", |child: DurableContext| async move {
                        child.wait(Duration::from_secs(1)).name("pause").await?;
                        Ok("a".to_owned())
                    }),
                    durable::Branch::new("second", |_child| async { Ok("b".to_owned()) }),
                ];
                let values = ctx
                    .parallel(branches)
                    .name("p")
                    .nesting(NestingMode::Flat)
                    .await?;
                let racing: Vec<durable::Branch<String>> = vec![
                    durable::Branch::new("first", |_child| async { Ok("a".to_owned()) }),
                    durable::Branch::new("second", |_child| async {
                        std::future::pending::<()>().await;
                        Ok("b".to_owned())
                    }),
                ];
                let early = ctx
                    .parallel(racing)
                    .name("early")
                    .nesting(NestingMode::Flat)
                    .completion(CompletionConfig::with_min_successful(1))
                    .await_batch()
                    .await?;
                Ok::<_, BoxError>((values, early.succeeded().count()))
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output(),
        Some(&(vec!["a".to_owned(), "b".to_owned()], 1))
    );
    assert_eq!(result.invocation_count(), 2);
    assert_flat_items(
        &tree,
        &result,
        "p",
        "1",
        ["first", "second"],
        "ParallelBranch",
        [&WAITING_FLAT_ITEM, &RETURNING_FLAT_ITEM],
    );
    assert_abandoned_flat_item(&tree, "2", "second");
}

/// JSON, except that every `serialize` call fails permanently.
struct FailSerialize;

impl<T> Serdes<T> for FailSerialize
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    fn serialize(
        &self,
        _value: T,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(Err("cannot encode".into()))
    }

    async fn deserialize(&self, wire: String, context: SerdesContext) -> Result<T, BoxError> {
        JsonSerdes.deserialize(wire, context).await
    }
}

/// A result that fails to serialize writes `FAIL`, so the end hook of the
/// child context and of a map item reports `Failed` with that error.
#[tokio::test]
async fn serialization_failure_end_carries_error() {
    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let child = ctx
                    .run_in_child_context(|_child| async { Ok(1_u32) })
                    .name("parent")
                    .serdes(FailSerialize)
                    .await;
                let map = ctx
                    .map(vec![1_u32], |_child, item, _idx| async move { Ok(item) })
                    .name("m")
                    .serdes(FailSerialize)
                    .await;
                Ok::<_, BoxError>((child.is_err(), map.is_err()))
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&(true, true)));
    for name in ["parent", "map-item-0"] {
        let end = tree.find(1, "end", name);
        assert_eq!(end.info().status(), FAILED, "{name}");
        let message = end.error_message().expect("an error message");
        assert!(message.contains("cannot encode"), "{name}: {message}");
    }
    assert_none_virtual(&tree);
}

/// A rejected `SUCCEED` write of a child context fails the execution, and
/// no end hook fires for the child.
#[tokio::test]
async fn failed_child_write_fires_no_end_hook() {
    let tree = Tree::contexts();
    let result = tree
        .runner()
        .fail_checkpoints_after(1, 1)
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok::<u32, BoxError>(
                    ctx.run_in_child_context(|_child| async { Ok(1_u32) })
                        .name("parent")
                        .await?,
                )
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_failure());
    let hooks: Vec<(usize, &str)> = tree.of("parent").iter().map(|r| (r.inv, r.hook)).collect();
    assert_eq!(hooks, [(1, "start")]);
}

/// A terminal map item whose result `ReplayChildren` kept out of its record
/// runs again on replay to rebuild the result. Neither the map nor the item
/// fires a hook then.
#[tokio::test]
async fn replayed_item_reconstruction_fires_nothing() {
    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let values = ctx
                    .map(vec![300 * 1024_usize], |_child, len, _idx| async move {
                        Ok("x".repeat(len))
                    })
                    .name("m")
                    .await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                Ok::<_, BoxError>(values.iter().map(String::len).sum::<usize>())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&(300 * 1024)));
    assert_eq!(result.invocation_count(), 2);
    let item_end = tree.find(1, "end", "map-item-0");
    assert_eq!(item_end.info().result(), None);
    let in_second: Vec<TreeRec> = tree
        .recs()
        .into_iter()
        .filter(|r| r.inv == 2 && r.info.is_some())
        .collect();
    assert!(in_second.is_empty(), "{in_second:?}");
}

/// A flat item that fails reports a virtual end with status `Failed` and
/// the error the batch records for it.
#[tokio::test]
async fn flat_item_failure_end_carries_error() {
    use durable::builders::map_parallel::NestingMode;

    let tree = Tree::contexts();
    let result = tree
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let outcome = ctx
                    .map(vec![1_u32], |_child, _item, _idx| async {
                        Err::<u32, BoxError>("bad".into())
                    })
                    .name("m")
                    .nesting(NestingMode::Flat)
                    .await;
                Ok::<_, BoxError>(outcome.is_err())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&true));
    let hooks: Vec<FlatHook> = tree
        .of("map-item-0")
        .iter()
        .map(|r| (r.inv, r.hook, r.info().is_replay()))
        .collect();
    assert_eq!(hooks, [(1, "start", false), (1, "end", false)]);
    let end = tree.find(1, "end", "map-item-0");
    assert!(end.info().is_virtual());
    assert_eq!(end.info().status(), FAILED);
    assert_eq!(end.info().result(), None);
    let message = end.error_message().expect("an error message");
    assert!(message.contains("bad"), "{message}");
}

/// Each combinator whose record is not terminal starts with `is_replay`
/// `true` on the next invocation.
#[tokio::test]
async fn every_resumed_combinator_starts_with_replay() {
    for kind in ["join_all", "select_ok", "race"] {
        let tree = Tree::default();
        let result = tree
            .runner()
            .run(
                move |_e: (), ctx: DurableContext| async move {
                    let pause = ctx.wait(Duration::from_secs(1)).name("pause").future();
                    match kind {
                        "join_all" => {
                            ctx.join_all([pause]).name("gather").await?;
                        }
                        "select_ok" => ctx.select_ok([pause]).name("gather").await?,
                        _ => ctx.race([pause]).name("gather").await?,
                    }
                    Ok::<_, BoxError>(())
                },
                (),
            )
            .await
            .expect("runner");
        assert!(result.is_success(), "{kind}");
        assert_eq!(
            tree.shapes_of("gather"),
            [
                (1, "start", "gather".to_owned(), false, STARTED),
                (2, "start", "gather".to_owned(), true, STARTED),
                (2, "end", "gather".to_owned(), false, SUCCEEDED),
            ],
            "{kind}"
        );
    }
}

/// Records the operation hooks of the step `held` and the invocation end.
/// Each operation hook returns a future that sleeps 1 s before it records
/// its completion.
#[derive(Clone, Default)]
struct DelayedHooks {
    events: Arc<Mutex<Vec<String>>>,
}

impl DelayedHooks {
    fn events(&self) -> Vec<String> {
        self.events.lock().unwrap().clone()
    }

    fn delayed(&self, hook: &str, info: &OperationInfo) -> Option<HookFuture<'static>> {
        if info.name() != Some("held") {
            return None;
        }
        let events = Arc::clone(&self.events);
        let done = format!("{hook} done");
        Some(Box::pin(async move {
            tokio::time::sleep(Duration::from_secs(1)).await;
            events.lock().unwrap().push(done);
            Ok(())
        }))
    }
}

impl Plugin for DelayedHooks {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        if info.name() == Some("held") {
            self.events.lock().unwrap().push("start".to_owned());
        }
        Ok(self.delayed("start", info))
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if info.name() == Some("held") {
            self.events
                .lock()
                .unwrap()
                .push(format!("end {:?}", info.status()));
        }
        Ok(self.delayed("end", info))
    }

    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.events
            .lock()
            .unwrap()
            .push("invocation end".to_owned());
        Box::pin(async { Ok(()) })
    }
}

impl PluginFactory for DelayedHooks {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

/// A step whose handle the handler dropped fires both operation hooks
/// when it settles, and the futures those hooks return finish in the
/// invocation-end drain, before `on_invocation_end`.
#[tokio::test(start_paused = true)]
async fn detached_operation_hooks_finish_and_drain() {
    let plugin = DelayedHooks::default();
    let handler_plugin = plugin.clone();
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(plugin.clone())
                .build()
                .expect("valid options"),
        )
        .run(
            move |(), ctx: DurableContext| {
                let plugin = handler_plugin.clone();
                async move {
                    let started = Arc::new(tokio::sync::Notify::new());
                    let release = Arc::new(tokio::sync::Notify::new());
                    let (body_started, body_release) = (Arc::clone(&started), Arc::clone(&release));
                    let held = ctx
                        .step(move |_| async move {
                            body_started.notify_one();
                            body_release.notified().await;
                            Ok(())
                        })
                        .name("held")
                        .spawn();
                    started.notified().await;
                    drop(held);
                    release.notify_one();
                    for _ in 0..1_000 {
                        if plugin.events().iter().any(|e| e.starts_with("end ")) {
                            break;
                        }
                        tokio::task::yield_now().await;
                    }
                    Ok::<_, BoxError>(())
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        result
            .operation("held")
            .map(durable::test_util::TestOperation::status),
        Some(OperationStatus::Succeeded)
    );
    let events = plugin.events();
    assert!(events.contains(&"start".to_owned()), "{events:?}");
    assert!(
        events.contains(&format!("end {:?}", Some(OperationStatus::Succeeded))),
        "{events:?}"
    );
    let at = |event: &str| {
        events
            .iter()
            .position(|e| e == event)
            .unwrap_or_else(|| unreachable!("no {event} in {events:?}"))
    };
    assert!(at("start done") < at("invocation end"), "{events:?}");
    assert!(at("end done") < at("invocation end"), "{events:?}");
}

// The user-function hooks and wraps (#126).

thread_local! {
    /// Set while a wrap of this thread's plugin is inside its `next()`.
    static FLAG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The frames `p0` entered on this thread: `I` for the invocation, then
    /// the name of each user function, outermost first.
    static FRAMES: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The frames current on this thread, as [`FRAMES`] holds them.
fn frames() -> Vec<String> {
    FRAMES.with(|f| f.borrow().clone())
}

/// The name of an operation, or its ID when it has none.
fn label(info: &OperationInfo) -> String {
    info.name()
        .map_or_else(|| info.id().to_owned(), str::to_owned)
}

/// One call that [`Uf`] recorded.
#[derive(Debug, Clone)]
struct UfRec {
    /// The invocation, counted from 1.
    inv: usize,
    plugin: &'static str,
    /// `start`, `enter`, `exit` or `end` for the user-function hooks and
    /// wraps; `op-start` or `op-end` for the operation hooks.
    hook: &'static str,
    op: String,
    info: OperationInfo,
    /// The end hook's error, for an `end` that received one.
    wire: Option<WireError>,
}

impl UfRec {
    /// The record as the issue's fixture writes it.
    fn line(&self) -> String {
        let (plugin, op) = (self.plugin, &self.op);
        match self.hook {
            "start" => format!("{plugin}.start({op})"),
            "enter" => format!("{plugin}.wrap({op}) enter"),
            "exit" => format!("{plugin}.wrap({op}) exit"),
            "end" if self.wire.is_none() => format!("{plugin}.end({op}, Ok)"),
            "end" => format!("{plugin}.end({op}, Err)"),
            other => format!("{plugin}.{other}({op})"),
        }
    }

    fn is_user_function(&self) -> bool {
        matches!(self.hook, "start" | "enter" | "exit" | "end")
    }
}

/// The recording plugins `p0` and `p1`. Each logs its user-function hooks
/// and wraps (S, W and E) and its operation hooks. `p0` also pushes `I` in
/// `wrap_invocation` and the user function's name in `wrap_user_function`
/// onto [`FRAMES`]. With `lock`, every hook and wrap takes the lock with
/// `try_lock` and releases it before it returns or calls `next()`, and
/// counts a contended lock in `contended`.
#[derive(Clone, Default)]
struct Uf {
    recs: Arc<Mutex<Vec<UfRec>>>,
    lock: Option<Arc<Mutex<()>>>,
    contended: Arc<AtomicUsize>,
}

struct UfFactory {
    uf: Uf,
    plugin: &'static str,
    count: Arc<AtomicUsize>,
}

struct UfPlugin {
    uf: Uf,
    plugin: &'static str,
    inv: usize,
}

impl Uf {
    fn factories(&self) -> [UfFactory; 2] {
        ["p0", "p1"].map(|plugin| UfFactory {
            uf: self.clone(),
            plugin,
            count: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn options(&self) -> durable::OptionsBuilder {
        let [p0, p1] = self.factories();
        Options::builder().plugin(p0).plugin(p1)
    }

    fn runner(&self) -> LocalRunner {
        LocalRunner::new().options(self.options().build().expect("valid options"))
    }

    fn recs(&self) -> Vec<UfRec> {
        self.recs.lock().unwrap().clone()
    }

    /// The user-function records, in order.
    fn uf_recs(&self) -> Vec<UfRec> {
        self.recs()
            .into_iter()
            .filter(UfRec::is_user_function)
            .collect()
    }

    /// The user-function lines of invocation `inv`.
    fn lines(&self, inv: usize) -> Vec<String> {
        self.uf_recs()
            .iter()
            .filter(|r| r.inv == inv)
            .map(UfRec::line)
            .collect()
    }

    /// The user-function records of `op`.
    fn of(&self, op: &str) -> Vec<UfRec> {
        self.uf_recs().into_iter().filter(|r| r.op == op).collect()
    }

    /// Asserts that each plugin logged S, W (enter and exit) and E for
    /// `op` in invocation `inv`.
    fn assert_full(&self, inv: usize, op: &str) {
        let lines: Vec<String> = self
            .of(op)
            .iter()
            .filter(|r| r.inv == inv)
            .map(UfRec::line)
            .collect();
        for plugin in ["p0", "p1"] {
            for hook in ["start(OP)", "wrap(OP) enter", "wrap(OP) exit", "end(OP, "] {
                let want = format!("{plugin}.{}", hook.replace("OP", op));
                assert!(
                    lines.iter().any(|l| l.starts_with(&want)),
                    "{want} in invocation {inv}: {lines:?}"
                );
            }
        }
    }

    fn hold(&self) {
        if let Some(lock) = &self.lock {
            match lock.try_lock() {
                Ok(guard) => drop(guard),
                Err(_) => {
                    self.contended.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
    }
}

impl PluginFactory for UfFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let inv = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Some(Box::new(UfPlugin {
            uf: self.uf.clone(),
            plugin: self.plugin,
            inv,
        })))
    }
}

impl UfPlugin {
    fn push(&self, hook: &'static str, info: &OperationInfo, wire: Option<&WireError>) {
        self.uf.recs.lock().unwrap().push(UfRec {
            inv: self.inv,
            plugin: self.plugin,
            hook,
            op: label(info),
            info: info.clone(),
            wire: wire.cloned(),
        });
    }

    fn is_p0(&self) -> bool {
        self.plugin == "p0"
    }
}

impl Plugin for UfPlugin {
    fn name(&self) -> &'static str {
        self.plugin
    }

    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        if self.is_p0() {
            FRAMES.with(|f| f.borrow_mut().push("I".to_owned()));
            next();
            FRAMES.with(|f| f.borrow_mut().pop());
        } else {
            next();
        }
    }

    fn wrap_user_function(&self, info: &OperationInfo, next: &mut dyn FnMut()) {
        self.uf.hold();
        self.push("enter", info, None);
        if self.is_p0() {
            FRAMES.with(|f| f.borrow_mut().push(label(info)));
            next();
            FRAMES.with(|f| f.borrow_mut().pop());
        } else {
            next();
        }
        self.uf.hold();
        self.push("exit", info, None);
    }

    fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
        self.uf.hold();
        self.push("start", info, None);
        Ok(None)
    }

    fn on_user_function_end(
        &self,
        info: &OperationInfo,
        outcome: Result<(), &WireError>,
    ) -> HookWork {
        self.uf.hold();
        self.push("end", info, outcome.err());
        Ok(None)
    }

    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.push("op-start", info, None);
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.push("op-end", info, None);
        Ok(None)
    }
}

/// Whether the context operation `name` closed with `ReplayChildren`.
fn replay_children<O>(result: &durable::test_util::TestResult<O>, name: &str) -> bool {
    use durable::test_util::TestOperationDetails;
    result.operation(name).is_some_and(
        |op| matches!(op.details(), TestOperationDetails::Context(d) if d.replay_children()),
    )
}

/// Asserts that a run failed with the context usage error that names
/// plugin code.
fn assert_plugin_code_usage_error<O>(result: &durable::test_util::TestResult<O>) {
    assert!(result.is_failure(), "the execution fails");
    let message = result.error_message().unwrap_or_default();
    assert!(message.contains("inside plugin code"), "{message}");
}

/// The text of a panic payload.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

/// One step's start and end hooks return futures that wait on test-held
/// gates. The gates open once the step's SUCCEED was written; the
/// invocation-end drain finishes both futures before `on_invocation_end`.
#[derive(Clone, Default)]
struct Gated {
    start_gate: Arc<tokio::sync::Notify>,
    end_gate: Arc<tokio::sync::Notify>,
    started: Arc<AtomicBool>,
    ended: Arc<AtomicBool>,
    log: Arc<Mutex<Vec<String>>>,
}

impl Gated {
    fn gated(gate: &Arc<tokio::sync::Notify>, flag: &Arc<AtomicBool>) -> HookFuture<'static> {
        let (gate, flag) = (Arc::clone(gate), Arc::clone(flag));
        Box::pin(async move {
            gate.notified().await;
            flag.store(true, Ordering::SeqCst);
            Ok(())
        })
    }

    fn flags(&self) -> (bool, bool) {
        (
            self.started.load(Ordering::SeqCst),
            self.ended.load(Ordering::SeqCst),
        )
    }
}

impl Plugin for Gated {
    fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
        if info.name() != Some("s") {
            return Ok(None);
        }
        Ok(Some(Self::gated(&self.start_gate, &self.started)))
    }

    fn on_user_function_end(
        &self,
        info: &OperationInfo,
        _outcome: Result<(), &WireError>,
    ) -> HookWork {
        if info.name() != Some("s") {
            return Ok(None);
        }
        Ok(Some(Self::gated(&self.end_gate, &self.ended)))
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if info.name() == Some("s") && info.status() == SUCCEEDED {
            self.log
                .lock()
                .unwrap()
                .push(format!("succeed {:?}", self.flags()));
            self.start_gate.notify_one();
            self.end_gate.notify_one();
        }
        Ok(None)
    }

    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.log
            .lock()
            .unwrap()
            .push(format!("invocation end {:?}", self.flags()));
        Box::pin(std::future::ready(Ok(())))
    }
}

impl PluginFactory for Gated {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

#[tokio::test]
async fn hook_futures_drained_at_invocation_end() {
    let gated = Gated::default();
    let runner = LocalRunner::new().options(
        Options::builder()
            .plugin(gated.clone())
            .build()
            .expect("valid options"),
    );
    let run = runner.run(
        |_e: (), ctx: DurableContext| async move {
            Ok::<_, BoxError>(ctx.step(|_| async { Ok(1_u32) }).name("s").await?)
        },
        (),
    );
    let result = tokio::time::timeout(Duration::from_secs(5), run)
        .await
        .expect("the run finishes within 5 s")
        .expect("runner");
    assert_eq!(result.output(), Some(&1));
    assert_eq!(
        *gated.log.lock().unwrap(),
        ["succeed (false, false)", "invocation end (true, true)"]
    );
}

/// Sets [`FLAG`] around `next()` in `wrap_invocation`, and records the
/// value `on_invocation_end` reads in its synchronous call.
#[derive(Clone, Default)]
struct FlagProbe(Arc<Mutex<Vec<bool>>>);

impl Plugin for FlagProbe {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        let outer = FLAG.with(|flag| flag.replace(true));
        next();
        FLAG.with(|flag| flag.set(outer));
    }

    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.0.lock().unwrap().push(FLAG.with(std::cell::Cell::get));
        Box::pin(std::future::ready(Ok(())))
    }
}

impl PluginFactory for FlagProbe {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.clone())))
    }
}

#[tokio::test]
async fn invocation_wrap_active_during_end_hook() {
    let probe = FlagProbe::default();
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(probe.clone())
                .build()
                .expect("valid options"),
        )
        .run(two_steps, ())
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(*probe.0.lock().unwrap(), [true]);
    assert!(
        !FLAG.with(std::cell::Cell::get),
        "the test is outside every wrap"
    );
}

/// A plugin whose wrap (`wrap_invocation` or `wrap_user_function`, by
/// `user_function`) calls `wait` on the context the handler put in `slot`.
struct ClaimsInWrap {
    slot: Arc<Mutex<Option<DurableContext>>>,
    user_function: bool,
}

impl ClaimsInWrap {
    fn claim(&self) {
        let ctx = self.slot.lock().unwrap().clone();
        if let Some(ctx) = ctx {
            drop(ctx.wait(Duration::from_secs(1)));
        }
    }
}

impl Plugin for ClaimsInWrap {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        if !self.user_function {
            self.claim();
        }
        next();
    }

    fn wrap_user_function(&self, _info: &OperationInfo, next: &mut dyn FnMut()) {
        if self.user_function {
            self.claim();
        }
        next();
    }
}

struct ClaimsInWrapFactory {
    slot: Arc<Mutex<Option<DurableContext>>>,
    user_function: bool,
}

impl PluginFactory for ClaimsInWrapFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(ClaimsInWrap {
            slot: Arc::clone(&self.slot),
            user_function: self.user_function,
        })))
    }
}

/// A `wrap_user_function` and a `wrap_invocation` that call `wait` while
/// the handler runs each fail the execution with the context usage error
/// that names plugin code.
#[tokio::test]
async fn claim_while_handler_runs_fails_the_execution() {
    for user_function in [true, false] {
        let slot = Arc::new(Mutex::new(None));
        let handler_slot = Arc::clone(&slot);
        let result = LocalRunner::new()
            .options(
                Options::builder()
                    .plugin(ClaimsInWrapFactory {
                        slot: Arc::clone(&slot),
                        user_function,
                    })
                    .build()
                    .expect("valid options"),
            )
            .run(
                move |_e: (), ctx: DurableContext| {
                    let slot = Arc::clone(&handler_slot);
                    async move {
                        *slot.lock().unwrap() = Some(ctx.clone());
                        let value = ctx.step(|_| async { Ok(1_u32) }).name("s").await?;
                        Ok::<_, BoxError>(value)
                    }
                },
                (),
            )
            .await
            .expect("runner");
        // The context must not outlive the run.
        slot.lock().unwrap().take();
        assert_plugin_code_usage_error(&result);
    }
}

/// Logs `inv exit` after each `wrap_invocation`, the [`FLAG`] value
/// `on_invocation_end` reads, and `drop` when the plugin is dropped.
struct DriveProbe(Arc<Mutex<Vec<String>>>);

impl Plugin for DriveProbe {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        let outer = FLAG.with(|flag| flag.replace(true));
        next();
        FLAG.with(|flag| flag.set(outer));
        self.0.lock().unwrap().push("inv exit".to_owned());
    }

    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        let flag = FLAG.with(std::cell::Cell::get);
        self.0.lock().unwrap().push(format!("end FLAG={flag}"));
        Box::pin(std::future::ready(Ok(())))
    }
}

impl Drop for DriveProbe {
    fn drop(&mut self) {
        self.0.lock().unwrap().push("drop".to_owned());
    }
}

struct DriveProbeFactory(Arc<Mutex<Vec<String>>>);

impl PluginFactory for DriveProbeFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(DriveProbe(Arc::clone(&self.0)))))
    }
}

/// A step body's panic leaves the drive, runs `on_invocation_end` inside
/// `wrap_invocation`, and resumes only after the wrap returned and the
/// plugin was dropped.
#[tokio::test]
async fn drive_panic_resumes_after_plugins_dropped() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let runner = LocalRunner::new().options(
        Options::builder()
            .plugin(DriveProbeFactory(Arc::clone(&log)))
            .build()
            .expect("valid options"),
    );
    let task_log = Arc::clone(&log);
    #[expect(clippy::disallowed_methods)]
    // reason: the test observes the panic through a JoinHandle
    let handle = tokio::spawn(async move {
        let result = runner
            .run(
                |_e: (), ctx: DurableContext| async move {
                    #[expect(clippy::panic)] // reason: the step panic under test
                    let value: u32 = ctx.step(|_| async { panic!("kaboom") }).name("s").await?;
                    Ok::<_, BoxError>(value)
                },
                (),
            )
            .await;
        task_log.lock().unwrap().push("after drive".to_owned());
        result.map(|_| ())
    });
    let err = handle.await.expect_err("the run panics");
    assert!(err.is_panic());
    assert!(panic_text(&*err.into_panic()).contains("kaboom"));
    let log = log.lock().unwrap().clone();
    let ends: Vec<&String> = log.iter().filter(|l| l.starts_with("end ")).collect();
    assert!(!ends.is_empty(), "{log:?}");
    assert!(ends.iter().all(|l| *l == "end FLAG=true"), "{log:?}");
    assert!(!log.iter().any(|l| l == "after drive"), "{log:?}");
    assert!(
        log.ends_with(&["inv exit".to_owned(), "drop".to_owned()]),
        "{log:?}"
    );
}

/// Step `a`, child `c`, `wait_for_condition` `w` done at its first check
/// and `wait_for_callback` `cb` run their user functions live; once each
/// is terminal, its replay fires no user-function hook (rule 7.8).
#[tokio::test]
async fn terminal_replay_fires_no_user_function_hook() {
    use durable::builders::wait_for_condition::WaitDecision;

    let uf = Uf::default();
    let result = uf
        .runner()
        .callback_for(
            "cb",
            durable::test_util::CallbackReply::success(&"yes").expect("encodes"),
        )
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
                ctx.run_in_child_context(|_child| async { Ok(2_u32) })
                    .name("c")
                    .await?;
                ctx.wait_for_condition(|_ctx, state: u32| async move { Ok(state + 1) }, 0_u32)
                    .wait_strategy_fn(|state: u32, _attempt| {
                        if state >= 1 {
                            WaitDecision::complete()
                        } else {
                            WaitDecision::continue_with(Duration::from_secs(1))
                        }
                    })
                    .name("w")
                    .await?;
                let approval: String = ctx
                    .wait_for_callback::<String, _, _>(|_step, _id| async { Ok(()) })
                    .name("cb")
                    .await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                Ok::<_, BoxError>(approval)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output().map(String::as_str), Some("yes"));
    // The callback suspends the first invocation, its completion resumes
    // the `wait_for_callback` body in the second, and `pause` makes a third
    // that replays every record.
    let last = result.invocation_count();
    assert_eq!(last, 3);
    for op in ["a", "c", "w", "cb-submitter"] {
        uf.assert_full(1, op);
        let later: Vec<UfRec> = uf.of(op).into_iter().filter(|r| r.inv > 1).collect();
        assert!(later.is_empty(), "{op}: {later:?}");
    }
    uf.assert_full(2, "cb");
    assert_eq!(uf.lines(last), Vec::<String>::new());
}

/// Runs `with_retry` `r`, retried once after 1 s, whose block calls
/// `outcome` with the attempt number (from 1) and records the frames
/// current in it.
async fn run_with_retry(
    uf: &Uf,
    outcome: fn(usize) -> Result<u32, BoxError>,
) -> (durable::test_util::TestResult<u32>, Vec<Vec<String>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let (handler_seen, handler_calls) = (Arc::clone(&seen), Arc::clone(&calls));
    let result = uf
        .runner()
        .run(
            move |_e: (), ctx: DurableContext| {
                let (seen, calls) = (Arc::clone(&handler_seen), Arc::clone(&handler_calls));
                async move {
                    Ok::<_, BoxError>(
                        ctx.with_retry(move |_child| {
                            let (seen, calls) = (Arc::clone(&seen), Arc::clone(&calls));
                            async move {
                                seen.lock().unwrap().push(frames());
                                outcome(calls.fetch_add(1, Ordering::SeqCst) + 1)
                            }
                        })
                        .name("r")
                        .retry_strategy(retry_once)
                        .await?,
                    )
                }
            },
            (),
        )
        .await
        .expect("runner");
    let seen = seen.lock().unwrap().clone();
    (result, seen)
}

/// The user-function hooks and wraps of `with_retry` `r` run for the outer
/// block, once per invocation that runs it; its `attempt-1` and
/// `attempt-2` children keep their operation hooks and are no user
/// function (rule 7.7), and their bodies run inside `r`'s frame.
#[tokio::test]
async fn with_retry_wraps_outer_block_not_attempts() {
    let uf = Uf::default();
    let (result, seen) = run_with_retry(&uf, |attempt| {
        if attempt == 1 {
            Err("transient".into())
        } else {
            Ok(4)
        }
    })
    .await;
    assert_eq!(result.output(), Some(&4));
    assert_eq!(result.invocation_count(), 2);
    let starts: Vec<(usize, &str, bool)> = uf
        .of("r")
        .iter()
        .filter(|r| r.hook == "start")
        .map(|r| (r.inv, r.plugin, r.info.is_replay()))
        .collect();
    assert_eq!(
        starts,
        [
            (1, "p0", false),
            (1, "p1", false),
            (2, "p0", true),
            (2, "p1", true)
        ]
    );
    let ends: Vec<(usize, &str, bool)> = uf
        .of("r")
        .iter()
        .filter(|r| r.hook == "end")
        .map(|r| (r.inv, r.plugin, r.wire.is_none()))
        .collect();
    assert_eq!(ends, [(2, "p1", true), (2, "p0", true)]);
    for (inv, attempt) in [(1, "attempt-1"), (2, "attempt-2")] {
        assert!(uf.of(attempt).is_empty(), "{:?}", uf.of(attempt));
        let op_start = uf
            .recs()
            .into_iter()
            .filter(|r| r.hook == "op-start" && r.op == attempt && r.inv == inv)
            .count();
        assert_eq!(op_start, 2, "both plugins' operation start for {attempt}");
    }
    assert_eq!(seen, [["I", "r"], ["I", "r"]]);
}

/// Both attempts of `with_retry` `r` fail, and the strategy stops: the end
/// hooks receive the exhaustion error's wire, which `r`'s FAIL records.
#[tokio::test]
async fn with_retry_exhaustion_end_hook() {
    let uf = Uf::default();
    let (result, _) = run_with_retry(&uf, |attempt| Err(format!("nope {attempt}").into())).await;
    assert!(result.is_failure());
    let recorded = result
        .operation("r")
        .and_then(durable::test_util::TestOperation::error)
        .expect("r failed")
        .clone();
    let ends: Vec<UfRec> = uf.of("r").into_iter().filter(|r| r.hook == "end").collect();
    assert_eq!(ends.len(), 2, "{ends:?}");
    for end in ends {
        assert_eq!(end.inv, 2);
        let wire = end.wire.expect("the end hook received the error");
        assert_eq!(wire.error_type(), recorded.error_type());
        assert_eq!(wire.error_message(), recorded.error_message());
        assert_eq!(wire.error_data(), recorded.error_data());
        assert!(
            wire.error_message().is_some_and(|m| m.contains("nope 2")),
            "{wire:?}"
        );
    }
}

/// A `parallel` `p` with one branch `b` running a step `s`: each hook and
/// wrap takes a `std::sync::Mutex` only while it runs, so none is ever
/// contended. Every user-function hook runs in the wraps of the frames
/// around it, `p` outermost, and each end hook runs right after its
/// plugin's wrap returned (rules 3.2 to 3.4).
#[tokio::test]
async fn step_in_parallel_branch_reenters() {
    let uf = Uf {
        lock: Some(Arc::new(Mutex::new(()))),
        ..Uf::default()
    };
    let runner = uf.runner();
    let run = runner.run(
        |_e: (), ctx: DurableContext| async move {
            let branches = vec![durable::Branch::new(
                "b",
                |child: DurableContext| async move {
                    Ok(child.step(|_| async { Ok(1_u32) }).name("s").await?)
                },
            )];
            Ok::<_, BoxError>(ctx.parallel(branches).name("p").await?)
        },
        (),
    );
    let result = tokio::time::timeout(Duration::from_secs(1), run)
        .await
        .expect("no deadlock within 1 s")
        .expect("runner");
    assert_eq!(result.output(), Some(&vec![1]));
    assert_eq!(uf.contended.load(Ordering::SeqCst), 0);
    // The stack of wraps entered when each record was made.
    let mut stack: Vec<String> = Vec::new();
    let mut at: Vec<(String, Vec<String>)> = Vec::new();
    let recs = uf.uf_recs();
    for (i, rec) in recs.iter().enumerate() {
        let me = format!("{}:{}", rec.plugin, rec.op);
        match rec.hook {
            "enter" => stack.push(me),
            "exit" => assert_eq!(stack.pop(), Some(me), "{recs:?}"),
            "start" => {
                at.push((rec.line(), stack.clone()));
                let next = recs.get(i + 1).map(UfRec::line);
                let want = format!("{}.wrap({}) enter", rec.plugin, rec.op);
                assert_eq!(next.as_deref(), Some(want.as_str()), "{recs:?}");
            }
            _ => {
                at.push((rec.line(), stack.clone()));
                let previous = i.checked_sub(1).and_then(|i| recs.get(i)).map(UfRec::line);
                let want = format!("{}.wrap({}) exit", rec.plugin, rec.op);
                assert_eq!(previous.as_deref(), Some(want.as_str()), "{recs:?}");
            }
        }
    }
    assert!(stack.is_empty(), "{recs:?}");
    let around = |line: &str| -> Vec<String> {
        at.iter().find(|(l, _)| l == line).map_or_else(
            || unreachable!("no {line} in {recs:?}"),
            |(_, stack)| stack.clone(),
        )
    };
    let s = |names: &[&str]| -> Vec<String> { names.iter().map(|n| (*n).to_owned()).collect() };
    let pp = ["p0:p", "p1:p"];
    let ppbb = ["p0:p", "p1:p", "p0:b", "p1:b"];
    assert_eq!(around("p0.start(p)"), s(&[]));
    assert_eq!(around("p1.start(p)"), s(&["p0:p"]));
    assert_eq!(around("p0.start(b)"), s(&pp));
    assert_eq!(around("p1.start(b)"), s(&["p0:p", "p1:p", "p0:b"]));
    assert_eq!(around("p0.start(s)"), s(&ppbb));
    assert_eq!(
        around("p1.start(s)"),
        s(&["p0:p", "p1:p", "p0:b", "p1:b", "p0:s"])
    );
    assert_eq!(
        around("p1.end(s, Ok)"),
        s(&["p0:p", "p1:p", "p0:b", "p1:b", "p0:s"])
    );
    assert_eq!(around("p0.end(s, Ok)"), s(&ppbb));
    assert_eq!(around("p1.end(b, Ok)"), s(&["p0:p", "p1:p", "p0:b"]));
    assert_eq!(around("p0.end(b, Ok)"), s(&pp));
    assert_eq!(around("p1.end(p, Ok)"), s(&["p0:p"]));
    assert_eq!(around("p0.end(p, Ok)"), s(&[]));
}

/// A `map` over `[1]` whose item waits 1 s: the item's start hook has
/// `is_replay` false in the first invocation and true in the second, in
/// both modes; a flat item's info is virtual.
#[tokio::test]
async fn map_item_info_replay_flag_both_modes() {
    use durable::builders::map_parallel::NestingMode;

    for nesting in [NestingMode::Normal, NestingMode::Flat] {
        let uf = Uf::default();
        let result = uf
            .runner()
            .run(
                move |_e: (), ctx: DurableContext| async move {
                    Ok::<_, BoxError>(
                        ctx.map(vec![1_u32], |child, item, _idx| async move {
                            child.wait(Duration::from_secs(1)).name("w").await?;
                            Ok(item)
                        })
                        .name("m")
                        .item_namer(|_item, idx| format!("item-{idx}"))
                        .nesting(nesting)
                        .await?,
                    )
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(result.output(), Some(&vec![1]), "{nesting:?}");
        assert_eq!(result.invocation_count(), 2, "{nesting:?}");
        let flat = matches!(nesting, NestingMode::Flat);
        let starts: Vec<(usize, &str, bool, bool)> = uf
            .of("item-0")
            .iter()
            .filter(|r| r.hook == "start")
            .map(|r| (r.inv, r.plugin, r.info.is_replay(), r.info.is_virtual()))
            .collect();
        assert_eq!(
            starts,
            [
                (1, "p0", false, flat),
                (1, "p1", false, flat),
                (2, "p0", true, flat),
                (2, "p1", true, flat),
            ],
            "{nesting:?}"
        );
    }
}

/// Counts the runs of each named body.
#[derive(Clone, Default)]
struct Runs(Arc<Mutex<std::collections::BTreeMap<String, usize>>>);

impl Runs {
    fn hit(&self, name: &str) {
        *self.0.lock().unwrap().entry(name.to_owned()).or_default() += 1;
    }

    fn of(&self, name: &str) -> usize {
        self.0
            .lock()
            .unwrap()
            .get(name)
            .copied()
            .unwrap_or_default()
    }
}

/// A child input named `name` that returns `len` copies of `fill`, or
/// fails when `len` is 0, counting its body's runs.
fn sized_child(
    ctx: &DurableContext,
    runs: &Runs,
    name: &'static str,
    fill: char,
    len: usize,
) -> durable::DurableFuture<String> {
    let runs = runs.clone();
    ctx.run_in_child_context(move |_child| async move {
        runs.hit(name);
        if len == 0 {
            return Err::<String, BoxError>(format!("{name} fails").into());
        }
        Ok(fill.to_string().repeat(len))
    })
    .name(name)
    .future()
}

/// The handler of [`batch_and_combinator_replay_routing`]. It returns the
/// total length of every result.
async fn routed(ctx: DurableContext, runs: Runs) -> Result<usize, BoxError> {
    const KIB: usize = 1024;
    let mut total = 0;
    // Small batches and a small join.
    total += ctx
        .map(vec![1_u32, 2], |_child, item, _idx| async move { Ok(item) })
        .name("m1")
        .item_namer(|_item, idx| format!("m1-{idx}"))
        .await?
        .len();
    let branches: Vec<durable::Branch<u32>> = vec![
        durable::Branch::new("p1-a", |_child| async { Ok(1_u32) }),
        durable::Branch::new("p1-b", |_child| async { Ok(2_u32) }),
    ];
    total += ctx.parallel(branches).name("p1").await?.len();
    let mut set = durable::BranchSet::new();
    let text = set.add("pm1-a", |_child| async { Ok("text".to_owned()) });
    let number = set.add("pm1-b", |_child| async { Ok(3_u32) });
    ctx.parallel_mixed(set).name("pm1").await?;
    total += text.into_result()?.len() + usize::try_from(number.into_result()?)?;
    let a = ctx.step(|_| async { Ok(1_u32) }).name("j1-a").future();
    let b = ctx.step(|_| async { Ok(2_u32) }).name("j1-b").future();
    total += ctx.try_join_all([a, b]).name("j1").await?.len();
    // Batches and joins whose results exceed 256 KiB.
    let map_runs = runs.clone();
    total += ctx
        .map(vec![0_u32, 1], move |_child, _item, idx| {
            let runs = map_runs.clone();
            async move {
                runs.hit(&format!("m2-{idx}"));
                Ok("m".repeat(150 * KIB))
            }
        })
        .name("m2")
        .item_namer(|_item, idx| format!("m2-{idx}"))
        .await?
        .iter()
        .map(String::len)
        .sum::<usize>();
    let inputs = [
        sized_child(&ctx, &runs, "j2-a", 'a', 150 * KIB),
        sized_child(&ctx, &runs, "j2-b", 'b', 150 * KIB),
    ];
    total += ctx
        .try_join_all(inputs)
        .name("j2")
        .await?
        .iter()
        .map(String::len)
        .sum::<usize>();
    let inputs = [
        sized_child(&ctx, &runs, "j3-a", 'a', 150 * KIB),
        sized_child(&ctx, &runs, "j3-b", 'b', 150 * KIB),
    ];
    for outcome in ctx.join_all(inputs).name("j3").await? {
        total += outcome?.len();
    }
    let inputs = [
        sized_child(&ctx, &runs, "so-a", 'a', 0),
        sized_child(&ctx, &runs, "so-b", 'b', 300 * KIB),
    ];
    total += ctx.select_ok(inputs).name("so").await?.len();
    let inputs = [
        sized_child(&ctx, &runs, "r-a", 'a', 300 * KIB),
        sized_child(&ctx, &runs, "r-b", 'b', 300 * KIB),
    ];
    let winner = ctx.race(inputs).name("r").await?;
    runs.hit(if winner.starts_with('a') {
        "r won by a"
    } else {
        "r won by b"
    });
    total += winner.len();
    ctx.wait(Duration::from_secs(1)).name("pause").await?;
    Ok(total)
}

/// Every container, item, join and input runs its user function in the
/// first invocation. The second replays them all and fires no
/// user-function hook (rule 7.8): the large `map` rebuilds its result from
/// its items' records, `try_join_all` and `join_all` from every input's
/// record, and `select_ok` and `race` rerun only the winner, whose own
/// record also carries `ReplayChildren`. No terminal record is written
/// again.
#[tokio::test]
async fn batch_and_combinator_replay_routing() {
    let uf = Uf::default();
    let runs = Runs::default();
    let handler_runs = runs.clone();
    let result = uf
        .runner()
        .run(
            move |_e: (), ctx: DurableContext| routed(ctx, handler_runs.clone()),
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 2);
    for op in [
        "m1", "m1-0", "m1-1", "p1", "p1-a", "p1-b", "pm1", "pm1-a", "pm1-b", "j1", "j1-a", "j1-b",
        "m2", "m2-0", "m2-1", "j2", "j2-a", "j2-b", "j3", "j3-a", "j3-b", "so", "so-a", "so-b",
        "r", "r-a", "r-b",
    ] {
        uf.assert_full(1, op);
    }
    for op in ["m2", "j2", "j3", "so", "r"] {
        assert!(replay_children(&result, op), "{op} carries ReplayChildren");
    }
    assert_eq!(uf.lines(2), Vec::<String>::new());
    // Only the winners' bodies run again.
    for op in ["m2-0", "m2-1", "j2-a", "j2-b", "j3-a", "j3-b", "so-a"] {
        assert_eq!(runs.of(op), 1, "{op}");
    }
    assert_eq!(runs.of("so-b"), 2);
    let (winner, loser) = if runs.of("r won by a") > 0 {
        ("r-a", "r-b")
    } else {
        ("r-b", "r-a")
    };
    assert_eq!(runs.of(winner), 2, "{winner}");
    assert_eq!(runs.of(loser), 1, "{loser}");
    // No operation of the second invocation writes a terminal record,
    // except the wait it resumes.
    let ends: Vec<String> = uf
        .recs()
        .into_iter()
        .filter(|r| r.inv == 2 && r.hook == "op-end" && r.op != "pause")
        .map(|r| r.op)
        .collect();
    assert_eq!(ends, Vec::<String>::new());
}

/// A `map` over `[]`, a `parallel` and a `parallel_mixed` with no branch
/// each run their container body: S, W and E with `Ok`, and no item hook.
#[tokio::test]
async fn empty_containers_fire_container_hooks() {
    let uf = Uf::default();
    let result = uf
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let items = ctx
                    .map(Vec::<u32>::new(), |_child, item: u32, _idx| async move {
                        Ok(item)
                    })
                    .name("em")
                    .await?;
                let branches = ctx
                    .parallel(Vec::<durable::Branch<u32>>::new())
                    .name("ep")
                    .await?;
                ctx.parallel_mixed(durable::BranchSet::new())
                    .name("epm")
                    .await?;
                Ok::<_, BoxError>(items.len() + branches.len())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&0));
    for op in ["em", "ep", "epm"] {
        assert_eq!(
            uf.of(op).iter().map(UfRec::line).collect::<Vec<_>>(),
            [
                format!("p0.start({op})"),
                format!("p0.wrap({op}) enter"),
                format!("p1.start({op})"),
                format!("p1.wrap({op}) enter"),
                format!("p1.wrap({op}) exit"),
                format!("p1.end({op}, Ok)"),
                format!("p0.wrap({op}) exit"),
                format!("p0.end({op}, Ok)"),
            ]
        );
    }
    let ops: Vec<String> = uf.uf_recs().into_iter().map(|r| r.op).collect();
    assert!(
        ops.iter()
            .all(|op| matches!(op.as_str(), "em" | "ep" | "epm")),
        "{ops:?}"
    );
}

/// The `(name, status)` of each updated operation of one change call.
type Updated = Vec<(Option<String>, Option<OperationStatus>)>;

/// One call the change recorder saw, with the number of the invocation
/// whose plugin received it.
#[derive(Clone, Debug, PartialEq)]
enum ChangeEvent {
    /// `on_operation_change`: the `(name, status)` of each updated
    /// operation, and whether every updated ID is a key of `operations()`.
    Change {
        n: usize,
        updated: Updated,
        in_map: bool,
    },
    /// `on_operation_start` for the operation named `name`.
    Start { n: usize, name: Option<String> },
}

/// A factory that numbers its plugins by invocation, from 1, and records
/// each change and operation-start call of every plugin in one list.
#[derive(Clone, Default)]
struct ChangeRecorder {
    events: Arc<Mutex<Vec<ChangeEvent>>>,
    count: Arc<AtomicUsize>,
}

struct ChangeInstance {
    n: usize,
    events: Arc<Mutex<Vec<ChangeEvent>>>,
}

impl Plugin for ChangeInstance {
    fn on_operation_change(&self, info: &durable::plugin::OperationChangeInfo) -> HookWork {
        let updated = info
            .updated_operations()
            .iter()
            .map(|op| (op.name().map(str::to_owned), op.status()))
            .collect();
        let in_map = info
            .updated_operations()
            .iter()
            .all(|op| info.operations().contains_key(op.id()));
        self.events.lock().unwrap().push(ChangeEvent::Change {
            n: self.n,
            updated,
            in_map,
        });
        Ok(None)
    }

    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.events.lock().unwrap().push(ChangeEvent::Start {
            n: self.n,
            name: info.name().map(str::to_owned),
        });
        Ok(None)
    }
}

impl PluginFactory for ChangeRecorder {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let n = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Some(Box::new(ChangeInstance {
            n,
            events: Arc::clone(&self.events),
        })))
    }
}

impl ChangeRecorder {
    fn runner(&self) -> LocalRunner {
        LocalRunner::new().options(
            Options::builder()
                .plugin(self.clone())
                .build()
                .expect("valid options"),
        )
    }

    fn events(&self) -> Vec<ChangeEvent> {
        self.events.lock().unwrap().clone()
    }

    /// The change calls, as `(invocation, updated)`.
    fn changes(&self) -> Vec<(usize, Updated)> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                ChangeEvent::Change { n, updated, .. } => Some((n, updated)),
                ChangeEvent::Start { .. } => None,
            })
            .collect()
    }

    /// The number of change calls that list `name` with `status`.
    fn count_listing(&self, name: &str, status: OperationStatus) -> usize {
        self.changes()
            .iter()
            .filter(|(_, updated)| lists(updated, name, status))
            .count()
    }

    /// The index of the first change call that lists `name` with
    /// `Started`, and the index of the start hook of `name`.
    fn started_change_and_start(&self, name: &str) -> (usize, usize) {
        let events = self.events();
        let change = events
            .iter()
            .position(|event| {
                matches!(event, ChangeEvent::Change { updated, .. }
                    if lists(updated, name, OperationStatus::Started))
            })
            .expect("a change lists the operation STARTED");
        let start = events
            .iter()
            .position(|event| {
                matches!(event, ChangeEvent::Start { name: Some(started), .. } if started == name)
            })
            .expect("the operation's start hook ran");
        (change, start)
    }
}

/// Whether `updated` lists the operation `name` with `status`.
fn lists(
    updated: &[(Option<String>, Option<OperationStatus>)],
    name: &str,
    status: OperationStatus,
) -> bool {
    updated
        .iter()
        .any(|(n, s)| n.as_deref() == Some(name) && *s == Some(status))
}

/// A completed step is listed `Succeeded` by exactly one change call, and
/// every updated ID is in the full map.
#[tokio::test]
async fn step_completion_reports_one_succeeded_change() {
    let plugin = ChangeRecorder::default();
    let result = plugin
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok(ctx
                    .step(|_| async { Ok("hi".to_owned()) })
                    .name("a")
                    .await?)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output().map(String::as_str), Some("hi"));
    assert_eq!(plugin.count_listing("a", OperationStatus::Succeeded), 1);
    assert_eq!(plugin.count_listing("a", OperationStatus::Failed), 0);
    assert!(
        plugin
            .events()
            .iter()
            .all(|event| !matches!(event, ChangeEvent::Change { in_map: false, .. })),
        "{:?}",
        plugin.events()
    );
}

/// A default step calls `on_operation_start` before the change that
/// lists it `Started`.
#[tokio::test]
async fn operation_start_precedes_start_change() {
    let plugin = ChangeRecorder::default();
    plugin
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok(ctx
                    .step(|_| async { Ok("hi".to_owned()) })
                    .name("a")
                    .await?)
            },
            (),
        )
        .await
        .expect("runner");
    let (change, start) = plugin.started_change_and_start("a");
    assert!(start < change, "{:?}", plugin.events());
}

/// A wait, an at-most-once step and `create_callback` get the change that
/// lists them `Started` before `on_operation_start`.
#[tokio::test]
async fn response_first_start_change_precedes_operation_start() {
    let wait = ChangeRecorder::default();
    wait.runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.wait(Duration::from_secs(1)).name("w").await?;
                Ok(())
            },
            (),
        )
        .await
        .expect("runner");
    let (change, start) = wait.started_change_and_start("w");
    assert!(change < start, "{:?}", wait.events());

    let step = ChangeRecorder::default();
    step.runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok(ctx
                    .step(|_| async { Ok(1_u32) })
                    .name("m")
                    .semantics(durable::StepSemantics::AtMostOncePerRetry)
                    .await?)
            },
            (),
        )
        .await
        .expect("runner");
    let (change, start) = step.started_change_and_start("m");
    assert!(change < start, "{:?}", step.events());

    let callback = ChangeRecorder::default();
    callback
        .runner()
        .callback_for(
            "cb",
            durable::test_util::CallbackReply::success(&"ok").expect("encodes"),
        )
        .run(
            |_e: (), ctx: DurableContext| async move {
                let cb = ctx.create_callback::<String>().name("cb").await?;
                Ok(cb.result().await?)
            },
            (),
        )
        .await
        .expect("runner");
    let (change, start) = callback.started_change_and_start("cb");
    assert!(change < start, "{:?}", callback.events());
}

/// A child context, a map and its item, and a combinator call
/// `on_operation_start` before the change that lists them `Started`.
#[tokio::test]
async fn child_backed_operation_start_precedes_start_change() {
    let child = ChangeRecorder::default();
    child
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok(ctx
                    .run_in_child_context(|child| async move {
                        Ok(child.step(|_| async { Ok(1_u32) }).name("s").await?)
                    })
                    .name("c")
                    .await?)
            },
            (),
        )
        .await
        .expect("runner");
    let (change, start) = child.started_change_and_start("c");
    assert!(start < change, "{:?}", child.events());

    let map = ChangeRecorder::default();
    map.runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok(ctx
                    .map(vec![1_u32], |child, item, _idx| async move {
                        Ok(child
                            .step(move |_| async move { Ok(item) })
                            .name("s")
                            .await?)
                    })
                    .name("m")
                    .await?)
            },
            (),
        )
        .await
        .expect("runner");
    for name in ["m", "map-item-0"] {
        let (change, start) = map.started_change_and_start(name);
        assert!(start < change, "{name}: {:?}", map.events());
    }

    let all = ChangeRecorder::default();
    all.runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
                let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
                Ok(ctx.try_join_all([a, b]).name("all").await?)
            },
            (),
        )
        .await
        .expect("runner");
    let (change, start) = all.started_change_and_start("all");
    assert!(start < change, "{:?}", all.events());
}

/// A retried step is listed `Pending` in the invocation that scheduled
/// the retry, and `Succeeded` in the invocation that completed it.
#[tokio::test]
async fn retry_reports_pending_change() {
    let plugin = ChangeRecorder::default();
    let result = plugin
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok(ctx
                    .step(|step| async move {
                        if step.attempt() == 1 {
                            return Err::<u32, BoxError>("first attempt fails".into());
                        }
                        Ok(2_u32)
                    })
                    .name("r")
                    .retry_strategy(|_err, attempt| {
                        if attempt >= 2 {
                            RetryDecision::Stop
                        } else {
                            RetryDecision::Retry {
                                delay: Duration::from_secs(1),
                            }
                        }
                    })
                    .await?)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&2));
    let changes = plugin.changes();
    assert!(
        changes
            .iter()
            .any(|(n, updated)| *n == 1 && lists(updated, "r", OperationStatus::Pending)),
        "{changes:?}"
    );
    assert!(
        changes
            .iter()
            .any(|(n, updated)| *n == 2 && lists(updated, "r", OperationStatus::Succeeded)),
        "{changes:?}"
    );
}

/// A wait the service completed between invocations is part of the next
/// invocation's starting state, which is not a change.
#[tokio::test]
async fn bootstrap_state_is_not_a_change() {
    let plugin = ChangeRecorder::default();
    let result = plugin
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                ctx.wait(Duration::from_secs(1)).name("w").await?;
                Ok(())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.invocation_count(), 2);
    let changes = plugin.changes();
    assert!(
        !changes
            .iter()
            .any(|(n, updated)| *n == 2
                && updated.iter().any(|(name, _)| name.as_deref() == Some("w"))),
        "{changes:?}"
    );
}

/// Each map item's step is listed `Succeeded` by exactly one change call.
#[tokio::test]
async fn map_items_report_changes() {
    let plugin = ChangeRecorder::default();
    plugin
        .runner()
        .run(
            |_e: (), ctx: DurableContext| async move {
                Ok(ctx
                    .map(vec![1_u32, 2, 3], |child, item, _idx| async move {
                        Ok(child
                            .step(move |_| async move { Ok(item) })
                            .name(format!("item-{item}"))
                            .await?)
                    })
                    .name("items")
                    .await?)
            },
            (),
        )
        .await
        .expect("runner");
    for name in ["item-1", "item-2", "item-3"] {
        assert_eq!(
            plugin.count_listing(name, OperationStatus::Succeeded),
            1,
            "{name}: {:?}",
            plugin.changes()
        );
    }
}

/// A handler of a step, a wait and a map, for `no_plugin_no_change_work`.
async fn change_work_handler(_e: (), ctx: DurableContext) -> Result<Vec<u32>, BoxError> {
    let base = ctx.step(|_| async { Ok(10_u32) }).name("base").await?;
    ctx.wait(Duration::from_secs(1)).name("w").await?;
    Ok(ctx
        .map(vec![1_u32, 2], move |child, item, _idx| async move {
            Ok(child
                .step(move |_| async move { Ok(base + item) })
                .name(format!("item-{item}"))
                .await?)
        })
        .name("items")
        .await?)
}

/// The `(name, status)` of every operation of a run, in record order.
fn named_statuses<O>(
    result: &durable::test_util::TestResult<O>,
) -> Vec<(Option<String>, OperationStatus)> {
    result
        .operations()
        .iter()
        .map(|op| (op.name().map(str::to_owned), op.status()))
        .collect()
}

/// Without a plugin, the change work does not change a run: the handler
/// returns the same value, over the same invocations and operations, as
/// with a recording plugin, which does receive change calls.
#[tokio::test]
async fn no_plugin_no_change_work() {
    let plain = LocalRunner::new()
        .run(change_work_handler, ())
        .await
        .expect("runner");
    let plugin = ChangeRecorder::default();
    let recorded = plugin
        .runner()
        .run(change_work_handler, ())
        .await
        .expect("runner");
    assert_eq!(plain.output(), Some(&vec![11, 12]));
    assert_eq!(plain.output(), recorded.output());
    assert_eq!(plain.invocation_count(), recorded.invocation_count());
    assert_eq!(named_statuses(&plain), named_statuses(&recorded));
    assert!(
        !plugin.changes().is_empty(),
        "the plugin run reported changes"
    );
}

// `plugin_child_operations_depth`: which finished children later
// invocations see, and how a preserved record replays.

/// `(inv, hook, kind, name, is_replay, status)` of one hook call that
/// [`DepthRec`] recorded. `hook` is `start`, `end` or `uf_start`.
type DepthHook = (
    usize,
    &'static str,
    Option<OperationKind>,
    String,
    bool,
    Option<OperationStatus>,
);

/// `(name, kind, status)` of one operation an invocation received.
type DepthOp = (String, Option<OperationKind>, Option<OperationStatus>);

/// `(hook, name, kind, is_replay, status)` of one hook record of a
/// combinator test.
type GatherHook = (
    &'static str,
    String,
    Option<OperationKind>,
    bool,
    Option<OperationStatus>,
);

/// Records `InvocationInfo::operations()` of each invocation as
/// `(name, kind, status)`, and every operation hook and user-function start
/// hook.
#[derive(Clone, Default)]
struct DepthRec {
    operations: Arc<Mutex<Vec<Vec<DepthOp>>>>,
    hooks: Arc<Mutex<Vec<DepthHook>>>,
    count: Arc<AtomicUsize>,
}

struct DepthRecPlugin {
    inv: usize,
    rec: DepthRec,
}

impl DepthRec {
    fn runner(&self, depth: u32) -> LocalRunner {
        LocalRunner::new().options(
            Options::builder()
                .plugin(self.clone())
                .plugin_child_operations_depth(depth)
                .build()
                .expect("valid options"),
        )
    }

    /// The `(name, kind)` of every operation but the execution that
    /// invocation `inv` received, sorted.
    fn operations_of(&self, inv: usize) -> Vec<(String, Option<OperationKind>)> {
        let all = self.operations.lock().unwrap();
        let mut ops: Vec<_> = all
            .get(inv - 1)
            .expect("the invocation ran")
            .iter()
            .filter(|(_, kind, _)| *kind != Some(OperationKind::Execution))
            .map(|(name, kind, _)| (name.clone(), *kind))
            .collect();
        ops.sort_by(|x, y| {
            x.0.cmp(&y.0)
                .then_with(|| format!("{:?}", x.1).cmp(&format!("{:?}", y.1)))
        });
        ops
    }

    fn status_of(&self, inv: usize, name: &str) -> Option<OperationStatus> {
        self.operations
            .lock()
            .unwrap()
            .get(inv - 1)
            .expect("the invocation ran")
            .iter()
            .find(|(n, _, _)| n == name)
            .and_then(|(_, _, status)| *status)
    }

    fn hooks_of(&self, inv: usize) -> Vec<DepthHook> {
        self.hooks
            .lock()
            .unwrap()
            .iter()
            .filter(|h| h.0 == inv)
            .cloned()
            .collect()
    }
}

impl PluginFactory for DepthRec {
    fn create_plugin(&self, info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        let inv = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        let ops = info
            .operations()
            .values()
            .map(|op| {
                (
                    op.name().unwrap_or_default().to_owned(),
                    op.kind(),
                    op.status(),
                )
            })
            .collect();
        self.operations.lock().unwrap().push(ops);
        Ok(Some(Box::new(DepthRecPlugin {
            inv,
            rec: self.clone(),
        })))
    }
}

impl DepthRecPlugin {
    fn push(&self, hook: &'static str, info: &OperationInfo) {
        self.rec.hooks.lock().unwrap().push((
            self.inv,
            hook,
            info.kind(),
            info.name().unwrap_or_default().to_owned(),
            info.is_replay(),
            info.status(),
        ));
    }
}

impl Plugin for DepthRecPlugin {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.push("start", info);
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.push("end", info);
        Ok(None)
    }

    fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
        self.push("uf_start", info);
        Ok(None)
    }
}

fn op(name: &str, kind: OperationKind) -> (String, Option<OperationKind>) {
    (name.to_owned(), Some(kind))
}

/// `map` named `m` over `[1, 2]`, each item a step `s` returning its input
/// times 2, then `wait(1 s)` named `w`. `runs` counts the step bodies.
async fn depth_map_handler(
    ctx: DurableContext,
    nesting: durable::builders::map_parallel::NestingMode,
    runs: Arc<AtomicUsize>,
) -> Result<Vec<u32>, BoxError> {
    let out = ctx
        .map(vec![1_u32, 2], move |child, item, _idx| {
            let runs = Arc::clone(&runs);
            async move {
                Ok(child
                    .step(move |_| async move {
                        runs.fetch_add(1, Ordering::SeqCst);
                        Ok(item * 2)
                    })
                    .name("s")
                    .await?)
            }
        })
        .name("m")
        .nesting(nesting)
        .await?;
    ctx.wait(Duration::from_secs(1)).name("w").await?;
    Ok(out)
}

/// Runs [`depth_map_handler`] at `depth` and returns the recorder, the
/// step-body count and the invocation count.
async fn run_depth_map(
    depth: u32,
    nesting: durable::builders::map_parallel::NestingMode,
) -> (DepthRec, usize, usize) {
    let rec = DepthRec::default();
    let runs = Arc::new(AtomicUsize::new(0));
    let handler_runs = Arc::clone(&runs);
    let result = rec
        .runner(depth)
        .run(
            move |_event: (), ctx: DurableContext| {
                let runs = Arc::clone(&handler_runs);
                async move { depth_map_handler(ctx, nesting, runs).await }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&vec![2, 4]));
    (rec, runs.load(Ordering::SeqCst), result.invocations().len())
}

const NORMAL: durable::builders::map_parallel::NestingMode =
    durable::builders::map_parallel::NestingMode::Normal;

#[tokio::test]
async fn depth_zero_prunes_children() {
    let (rec, _, _) = run_depth_map(0, NORMAL).await;
    assert_eq!(
        rec.operations_of(2),
        [
            op("m", OperationKind::Context),
            op("w", OperationKind::Wait)
        ]
    );
}

#[tokio::test]
async fn depth_one_keeps_items() {
    let (rec, _, _) = run_depth_map(1, NORMAL).await;
    assert_eq!(
        rec.operations_of(2),
        [
            op("m", OperationKind::Context),
            op("map-item-0", OperationKind::Context),
            op("map-item-1", OperationKind::Context),
            op("w", OperationKind::Wait),
        ]
    );
}

fn depth_two_tree() -> Vec<(String, Option<OperationKind>)> {
    vec![
        op("m", OperationKind::Context),
        op("map-item-0", OperationKind::Context),
        op("map-item-1", OperationKind::Context),
        op("s", OperationKind::Step),
        op("s", OperationKind::Step),
        op("w", OperationKind::Wait),
    ]
}

#[tokio::test]
async fn depth_two_keeps_item_steps() {
    let (rec, _, _) = run_depth_map(2, NORMAL).await;
    assert_eq!(rec.operations_of(2), depth_two_tree());
}

#[tokio::test]
async fn full_depth_keeps_tree() {
    let (rec, _, _) = run_depth_map(u32::MAX, NORMAL).await;
    assert_eq!(rec.operations_of(2), depth_two_tree());
}

#[tokio::test]
async fn flat_items_add_no_level() {
    let (rec, _, _) = run_depth_map(1, durable::builders::map_parallel::NestingMode::Flat).await;
    assert_eq!(
        rec.operations_of(2),
        [
            op("m", OperationKind::Context),
            op("s", OperationKind::Step),
            op("s", OperationKind::Step),
            op("w", OperationKind::Wait),
        ]
    );
}

#[tokio::test]
async fn preserved_failed_child_keeps_children() {
    let rec = DepthRec::default();
    let result = rec
        .runner(1)
        .run(
            |_event: (), ctx: DurableContext| async move {
                let caught = ctx
                    .run_in_child_context(|child| async move {
                        child
                            .step(|_| async { Err::<u32, BoxError>("declined".into()) })
                            .name("s")
                            .retry_strategy(|_, _| RetryDecision::Stop)
                            .await?;
                        Ok(0_u32)
                    })
                    .name("c")
                    .await
                    .is_err();
                ctx.wait(Duration::from_secs(1)).name("w").await?;
                Ok::<_, BoxError>(caught)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&true));
    let ops = rec.operations_of(2);
    assert!(ops.contains(&op("c", OperationKind::Context)), "{ops:?}");
    assert!(ops.contains(&op("s", OperationKind::Step)), "{ops:?}");
    assert_eq!(rec.status_of(2, "c"), Some(OperationStatus::Failed));
}

#[tokio::test]
async fn preserved_container_replays_items() {
    let (rec, runs, invocations) = run_depth_map(1, NORMAL).await;
    assert_eq!(invocations, 2);
    assert_eq!(runs, 2, "no step body ran in invocation 2");
    let replayed: Vec<DepthHook> = rec
        .hooks_of(2)
        .into_iter()
        .filter(|h| (h.1 == "start" || h.1 == "end") && h.3 != "m" && h.3 != "w")
        .collect();
    assert_eq!(replayed, [], "no item or step hook in invocation 2");
}

#[tokio::test]
async fn preserved_replay_runs_no_step_body() {
    let (rec, runs, invocations) = run_depth_map(2, NORMAL).await;
    assert_eq!(rec.operations_of(2), depth_two_tree());
    assert_eq!(invocations, 2);
    assert_eq!(runs, 2, "each step body ran once");
}

#[tokio::test]
#[expect(clippy::expect_used, reason = "the test value always encodes")]
async fn preserved_wait_for_callback_has_flag() {
    let rec = DepthRec::default();
    let result = rec
        .runner(1)
        .callback_for(
            "approve",
            durable::test_util::CallbackReply::success(&"yes").expect("encodes"),
        )
        .run(
            |_event: (), ctx: DurableContext| async move {
                let value: String = ctx
                    .wait_for_callback::<String, _, _>(|_step, _id| async { Ok(()) })
                    .name("approve")
                    .await?;
                ctx.wait(Duration::from_secs(1)).name("w").await?;
                ctx.wait(Duration::from_secs(1)).name("x").await?;
                Ok::<_, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output().map(String::as_str), Some("yes"));
    let approve = result.operation("approve").expect("the wait_for_callback");
    let durable::test_util::TestOperationDetails::Context(details) = approve.details() else {
        unreachable!("a context record")
    };
    assert!(details.replay_children());
    assert_eq!(approve.result(), Some("\"yes\""));

    // The invocation that wrote the SUCCEED fired the live end hook.
    let hooks = rec.hooks.lock().unwrap().clone();
    let wrote = hooks
        .iter()
        .find(|h| h.1 == "end" && h.3 == "approve" && !h.4)
        .map(|h| h.0)
        .expect("the live end hook of approve");
    let next = rec.hooks_of(wrote + 1);
    let callback_ends: Vec<_> = next
        .iter()
        .filter(|h| h.3 == "approve-callback")
        .cloned()
        .collect();
    let [(_, hook, _, _, is_replay, _)] = callback_ends.as_slice() else {
        unreachable!("one hook of the callback: {next:?}")
    };
    assert_eq!(*hook, "end");
    assert!(*is_replay, "the callback's end hook replays");
    assert!(
        next.iter().all(|h| h.3 != "approve-submitter"),
        "the submitter fires no hook: {next:?}"
    );
    assert!(
        next.iter()
            .all(|h| !(h.1 == "uf_start" && h.3 == "approve")),
        "no user-function start names approve: {next:?}"
    );
}

/// Which combinator `gather` is.
#[derive(Clone, Copy)]
enum Gather {
    TryJoinAll,
    JoinAll,
    Race,
}

/// A combinator `gather` over `wait(1 s)` named `a` and `b`, then
/// `wait(1 s)` named `w`. Returns the recorder and the `gather` record's
/// `(replay_children, result)`.
async fn run_gather(depth: u32, gather: Gather) -> (DepthRec, bool, Option<String>) {
    let rec = DepthRec::default();
    let result = rec
        .runner(depth)
        .run(
            move |_event: (), ctx: DurableContext| async move {
                let a = ctx.wait(Duration::from_secs(1)).name("a").future();
                let b = ctx.wait(Duration::from_secs(1)).name("b").future();
                match gather {
                    Gather::TryJoinAll => {
                        ctx.try_join_all([a, b]).name("gather").await?;
                    }
                    Gather::JoinAll => {
                        ctx.join_all([a, b]).name("gather").await?;
                    }
                    Gather::Race => {
                        ctx.race([a, b]).name("gather").await?;
                    }
                }
                ctx.wait(Duration::from_secs(1)).name("w").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&()));
    assert_eq!(result.invocations().len(), 3);
    let gather = result.operation("gather").expect("the combinator");
    let durable::test_util::TestOperationDetails::Context(details) = gather.details() else {
        unreachable!("a context record")
    };
    (
        rec,
        details.replay_children(),
        gather.result().map(str::to_owned),
    )
}

/// Invocation 3's hook records of `a`, `b` and `gather`, as
/// `(hook, name, kind, is_replay, status)`.
fn gather_hooks(rec: &DepthRec) -> Vec<GatherHook> {
    rec.hooks_of(3)
        .into_iter()
        .filter(|h| ["a", "b", "gather"].contains(&h.3.as_str()))
        .map(|(_, hook, kind, name, replay, status)| (hook, name, kind, replay, status))
        .collect()
}

fn redriven_inputs() -> Vec<GatherHook> {
    ["a", "b"]
        .map(|name| {
            (
                "end",
                name.to_owned(),
                Some(OperationKind::Wait),
                true,
                Some(OperationStatus::Succeeded),
            )
        })
        .to_vec()
}

#[tokio::test]
async fn preserved_try_join_all_redrives_inputs() {
    let (rec, flag, result) = run_gather(1, Gather::TryJoinAll).await;
    assert!(flag);
    assert_eq!(result.as_deref(), Some("[null,null]"));
    assert_eq!(gather_hooks(&rec), redriven_inputs());
}

#[tokio::test]
async fn preserved_join_all_redrives_inputs() {
    let (_, _, plain) = run_gather(0, Gather::JoinAll).await;
    let (rec, flag, result) = run_gather(1, Gather::JoinAll).await;
    assert!(flag);
    assert_eq!(result, plain);
    assert_eq!(gather_hooks(&rec), redriven_inputs());
}

#[tokio::test]
async fn depth_zero_try_join_all_drops_inputs() {
    let (rec, flag, result) = run_gather(0, Gather::TryJoinAll).await;
    assert!(!flag);
    assert_eq!(result.as_deref(), Some("[null,null]"));
    assert_eq!(gather_hooks(&rec), []);
}

#[tokio::test]
async fn preserved_race_drops_inputs() {
    let (rec, flag, result) = run_gather(1, Gather::Race).await;
    assert!(!flag);
    assert_eq!(result.as_deref(), Some("null"));
    assert_eq!(gather_hooks(&rec), []);
}
