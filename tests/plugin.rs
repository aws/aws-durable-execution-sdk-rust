//! Plugin factories and the invocation info they receive, through the public
//! API and `LocalRunner`.
//!
//! Local reported times are the run-start time plus virtual seconds, so the
//! tests assert differences and lower bounds, never the epoch.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::unwrap_used)] // reason: test assertions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use aws_durable_execution_sdk as durable;
use durable::plugin::{InvocationInfo, Plugin, PluginFactory};
use durable::test_util::LocalRunner;
use durable::{BoxError, DurableContext, OperationKind, OperationStatus, Options, RetryDecision};

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
        .and_then(durable::WireError::error_message)
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
