//! Plugin frames reach the handler, `.spawn()`ed operations, combinator
//! inputs and map items through the SDK's spawn paths (#126).

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::cell::Cell;
use std::future::IntoFuture as _;

use aws_durable_execution_sdk as durable;
use durable::plugin::{InvocationInfo, Plugin, PluginFactory};
use durable::test_util::LocalRunner;
use durable::{BoxError, DurableContext, Options};

thread_local! {
    /// Set while a `wrap_invocation` of this thread's plugin is entered.
    static IN_INVOCATION: Cell<bool> = const { Cell::new(false) };
}

fn in_invocation() -> bool {
    IN_INVOCATION.with(Cell::get)
}

/// The current frames as the wraps entered on this thread show them: `I`
/// for the invocation, then the name of each user function, outermost
/// first. Every test plugin below that has wraps pushes onto [`IN_FN`].
fn frames() -> Vec<String> {
    IN_FN.with(|f| f.borrow().clone())
}

/// `[I, rest..]` as owned strings.
fn path(rest: &[&str]) -> Vec<String> {
    std::iter::once("I")
        .chain(rest.iter().copied())
        .map(str::to_owned)
        .collect()
}

fn enter_invocation(next: &mut dyn FnMut()) {
    IN_FN.with(|f| f.borrow_mut().push("I".to_owned()));
    next();
    IN_FN.with(|f| f.borrow_mut().pop());
}

fn enter_user_function(info: &durable::plugin::OperationInfo, next: &mut dyn FnMut()) {
    IN_FN.with(|f| {
        f.borrow_mut()
            .push(info.name().unwrap_or_default().to_owned());
    });
    next();
    IN_FN.with(|f| f.borrow_mut().pop());
}

/// One user-function wrap: the operation's name and whether its info is
/// virtual.
type WrapSeen = (String, bool);

/// A plugin whose wraps make the frames readable with [`frames`], and which
/// logs every user-function wrap it enters.
#[derive(Clone, Default)]
struct Wraps(std::sync::Arc<std::sync::Mutex<Vec<WrapSeen>>>);

impl Plugin for Wraps {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        enter_invocation(next);
    }

    fn wrap_user_function(&self, info: &durable::plugin::OperationInfo, next: &mut dyn FnMut()) {
        self.0.lock().expect("not poisoned").push((
            info.name().unwrap_or_default().to_owned(),
            info.is_virtual(),
        ));
        enter_user_function(info, next);
    }
}

/// Creates a clone of one plugin for every invocation.
struct Each<P>(P);

impl<P: Plugin + Clone + 'static> PluginFactory for Each<P> {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(self.0.clone())))
    }
}

/// Runs `handler` through a [`LocalRunner`] with `plugin` registered.
async fn run_with<P, F, Fut, O>(plugin: P, handler: F) -> durable::test_util::TestResult<O>
where
    P: Plugin + Clone + 'static,
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    LocalRunner::new()
        .options(
            Options::builder()
                .plugin(Each(plugin))
                .build()
                .expect("valid options"),
        )
        .run(handler, ())
        .await
        .expect("runner")
}

/// Polls `fut` once with a no-op waker.
fn poll_once<F: Future + Unpin>(fut: &mut F) -> bool {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    std::pin::Pin::new(fut).poll(&mut cx).is_ready()
}

/// What the handler gives the [`Poker`] plugin to poll.
enum Given {
    /// A step future the handler claimed with `.future()`.
    Claimed(durable::DurableFuture<u32>),
    /// The conversion of an unclaimed `try_join_all` builder to its future.
    Unclaimed(Box<dyn FnOnce() -> durable::DurableFuture<Vec<u32>> + Send>),
}

type GivenSlot = std::sync::Arc<std::sync::Mutex<Option<Given>>>;

/// Takes what the slot holds, polls it once, and puts a claimed future
/// back so the handler can await it.
fn poke(slot: &GivenSlot) {
    let given = slot.lock().expect("not poisoned").take();
    match given {
        Some(Given::Claimed(mut fut)) => {
            poll_once(&mut fut);
            *slot.lock().expect("not poisoned") = Some(Given::Claimed(fut));
        }
        Some(Given::Unclaimed(convert)) => {
            let mut fut = convert();
            poll_once(&mut fut);
        }
        None => {}
    }
}

/// Pokes the slot in the synchronous part of the user-function start hook
/// of step `trigger`.
#[derive(Clone, Default)]
struct Poker(GivenSlot);

impl Plugin for Poker {
    fn on_user_function_start(
        &self,
        info: &durable::plugin::OperationInfo,
    ) -> durable::plugin::HookWork {
        if info.name() == Some("trigger") {
            poke(&self.0);
        }
        Ok(None)
    }
}

/// Where [`guard_reach_matches_step_body_check`] polls what the handler gave.
#[derive(Debug, Clone, Copy)]
enum Reach {
    /// A claimed step future, in a synchronous start hook.
    ClaimedInHook,
    /// A claimed step future, in a step body.
    ClaimedInBody,
    /// An unclaimed `try_join_all` builder, converted and polled in a
    /// synchronous start hook.
    UnclaimedInHook,
}

/// #126 rule 8.3: the plugin guard reaches what the step-body check
/// reaches. Polling a future the handler already claimed, from a start
/// hook or a step body, claims nothing and records no error. Polling an
/// unclaimed combinator from a start hook claims its ID under the guard
/// and fails the execution with the context usage error naming plugin
/// code. Every poll runs synchronously on the SDK task that runs the hook
/// or body, never on a hook-work task, where the task-ownership check
/// would fail first.
#[tokio::test]
async fn guard_reach_matches_step_body_check() {
    for reach in [
        Reach::ClaimedInHook,
        Reach::ClaimedInBody,
        Reach::UnclaimedInHook,
    ] {
        let poker = Poker::default();
        let slot = std::sync::Arc::clone(&poker.0);
        let result = run_with(poker, move |_event: (), ctx: DurableContext| {
            let slot = std::sync::Arc::clone(&slot);
            async move {
                let given = match reach {
                    Reach::ClaimedInHook | Reach::ClaimedInBody => {
                        Given::Claimed(ctx.step(|_| async { Ok(7_u32) }).name("given").future())
                    }
                    Reach::UnclaimedInHook => {
                        let input = ctx.step(|_| async { Ok(7_u32) }).name("input").future();
                        let join = ctx.try_join_all([input]).name("j");
                        Given::Unclaimed(Box::new(move || join.future()))
                    }
                };
                *slot.lock().expect("not poisoned") = Some(given);
                let body_slot = std::sync::Arc::clone(&slot);
                ctx.step(move |_| async move {
                    if matches!(reach, Reach::ClaimedInBody) {
                        poke(&body_slot);
                    }
                    Ok(0_u32)
                })
                .name("trigger")
                .await?;
                let given = slot.lock().expect("not poisoned").take();
                match given {
                    Some(Given::Claimed(fut)) => Ok(fut.await?),
                    _ => Ok(0_u32),
                }
            }
        })
        .await;
        let message = result
            .error()
            .and_then(durable::WireError::error_message)
            .unwrap_or_default()
            .to_owned();
        match reach {
            Reach::ClaimedInHook | Reach::ClaimedInBody => {
                assert_eq!(result.output(), Some(&7), "{reach:?}: {message}");
            }
            Reach::UnclaimedInHook => {
                assert_eq!(result.output(), None, "{reach:?}");
                assert!(message.contains("plugin code"), "{reach:?}: {message}");
            }
        }
    }
}

/// Which callback of the step's SDK span [`PanicsInSpan`] panics in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanCallback {
    Enter,
    Exit,
}

/// A layer that panics once, in the first `on_enter` or `on_exit` of a
/// `durable_operation` span.
struct PanicsInSpan {
    at: SpanCallback,
    armed: std::sync::atomic::AtomicBool,
}

impl PanicsInSpan {
    fn hit<S>(&self, id: &tracing::span::Id, ctx: &tracing_subscriber::layer::Context<'_, S>)
    where
        S: tracing::Subscriber + for<'l> tracing_subscriber::registry::LookupSpan<'l>,
    {
        let operation = ctx
            .span(id)
            .is_some_and(|span| span.name() == "durable_operation");
        // The subscriber panic under test, once.
        let fire = operation && self.armed.swap(false, std::sync::atomic::Ordering::SeqCst);
        assert!(!fire, "span subscriber panics");
    }
}

impl<S> tracing_subscriber::Layer<S> for PanicsInSpan
where
    S: tracing::Subscriber + for<'l> tracing_subscriber::registry::LookupSpan<'l>,
{
    fn on_enter(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if self.at == SpanCallback::Enter {
            self.hit(id, &ctx);
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if self.at == SpanCallback::Exit {
            self.hit(id, &ctx);
        }
    }
}

/// #126 rule 6.2: the gateway enters a task's span inside its
/// `catch_unwind`, so a subscriber panic while the SDK span of a
/// `.spawn()`ed step is entered or exited is a task panic, and the
/// invocation fails with it. Current-thread, so the thread-local subscriber
/// sees the spawned step's task.
#[tokio::test]
async fn span_subscriber_panic_caught_by_gateway() {
    use tracing_subscriber::layer::SubscriberExt as _;

    for at in [SpanCallback::Enter, SpanCallback::Exit] {
        let _guard =
            tracing::subscriber::set_default(tracing_subscriber::registry().with(PanicsInSpan {
                at,
                armed: std::sync::atomic::AtomicBool::new(true),
            }));
        #[expect(clippy::disallowed_methods)]
        // reason: the test observes the panic the run resumes, which only a
        // task's JoinHandle reports.
        let joined = tokio::spawn(async {
            LocalRunner::new()
                .run(
                    |_event: (), ctx: DurableContext| async move {
                        let handle = ctx.step(|_| async { Ok(1_u32) }).name("s").spawn();
                        Ok::<_, BoxError>(handle.await?)
                    },
                    (),
                )
                .await
        })
        .await;
        let payload = joined.expect_err("the invocation fails").into_panic();
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert_eq!(message, "span subscriber panics", "{at:?}");
    }
}

/// #126 rules 5.2 and 6.2: a `.spawn()`ed step in child `c` captures the
/// frames current at `.spawn()`, and its task's gateway enters them at
/// every poll, on whichever worker polls it: the body reads `[I, c, step]`
/// before and after each of its awaits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawned_task_captures_creating_frames() {
    let result = run_with(
        Wraps::default(),
        |_event: (), ctx: DurableContext| async move {
            Ok(ctx
                .run_in_child_context(|child| async move {
                    let handle = child
                        .step(|_| async {
                            let mut seen = vec![frames()];
                            for _ in 0..8 {
                                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                                seen.push(frames());
                            }
                            Ok(seen)
                        })
                        .name("step")
                        .spawn();
                    Ok(handle.await?)
                })
                .name("c")
                .await?)
        },
    )
    .await;
    let seen = result.output().expect("the run succeeded");
    assert_eq!(seen.len(), 9, "{seen:?}");
    for frames in seen {
        assert_eq!(*frames, path(&["c", "step"]), "{seen:?}");
    }
}

/// #126 rule 5.2: a `.spawn()`ed step's handle that child `a` hands to a
/// `race` in child `b` keeps the frames the step captured at `.spawn()`:
/// the body reads `[I, a, step]`, never `b`'s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawned_handle_into_combinator_keeps_frames() {
    let result = run_with(
        Wraps::default(),
        |_event: (), ctx: DurableContext| async move {
            let (handle_tx, handle_rx) = tokio::sync::oneshot::channel();
            let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
            let a = ctx
                .run_in_child_context(move |child| async move {
                    let go = std::sync::Arc::new(tokio::sync::Notify::new());
                    let body_go = std::sync::Arc::clone(&go);
                    let handle = child
                        .step(move |_| async move {
                            body_go.notified().await;
                            Ok(frames())
                        })
                        .name("step")
                        .spawn();
                    if handle_tx.send((handle, go)).is_err() {
                        return Err::<u32, BoxError>("b stopped".into());
                    }
                    // Child `a` stays open until `b` raced the handle.
                    done_rx.await?;
                    Ok(0_u32)
                })
                .name("a")
                .spawn();
            let b = ctx
                .run_in_child_context(move |child| async move {
                    let (handle, go) = handle_rx.await?;
                    go.notify_one();
                    let won = child.race([handle]).name("r").await?;
                    let _done = done_tx.send(());
                    Ok::<_, BoxError>(won)
                })
                .name("b")
                .spawn();
            let (a, b) = tokio::join!(a, b);
            a?;
            Ok::<_, BoxError>(b?)
        },
    )
    .await;
    assert_eq!(result.output(), Some(&path(&["a", "step"])));
}

/// The frames each item or branch body of one container read.
type Container = Vec<Vec<String>>;

/// #126 rule 5.3: items of a `map` and branches of a `parallel` and a
/// `parallel_mixed` capture their task context at dispatch inside the
/// container's poll, so each body reads `[I, container, item]`, in both
/// nesting modes. A flat item's info is virtual; a normal item's is not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn map_items_and_branches_capture_container_frames() {
    for mode in [
        durable::builders::map_parallel::NestingMode::Normal,
        durable::builders::map_parallel::NestingMode::Flat,
    ] {
        let wraps = Wraps::default();
        let log = std::sync::Arc::clone(&wraps.0);
        let result = run_with(wraps, move |_event: (), ctx: DurableContext| async move {
            let map = ctx
                .map(vec![1_u32, 2], |_child, _item, _index| async move {
                    Ok(frames())
                })
                .name("m")
                .item_namer(|_item, index| format!("m{index}"))
                .nesting(mode)
                .await?;
            let parallel = ctx
                .parallel(vec![
                    durable::Branch::new("p0", |_child| async move { Ok(frames()) }),
                    durable::Branch::new("p1", |_child| async move { Ok(frames()) }),
                ])
                .name("p")
                .nesting(mode)
                .await?;
            let mut branches = durable::BranchSet::new();
            let x0 = branches.add("x0", |_child| async move { Ok(frames()) });
            let x1 = branches.add("x1", |_child| async move { Ok(frames()) });
            ctx.parallel_mixed(branches).name("x").nesting(mode).await?;
            let mixed = vec![x0.into_result()?, x1.into_result()?];
            Ok::<(Container, Container, Container), BoxError>((map, parallel, mixed))
        })
        .await;
        let (map, parallel, mixed) = result.output().expect("the run succeeded");
        assert_eq!(*map, [path(&["m", "m0"]), path(&["m", "m1"])], "{mode:?}");
        assert_eq!(
            *parallel,
            [path(&["p", "p0"]), path(&["p", "p1"])],
            "{mode:?}"
        );
        assert_eq!(*mixed, [path(&["x", "x0"]), path(&["x", "x1"])], "{mode:?}");
        let flat = matches!(mode, durable::builders::map_parallel::NestingMode::Flat);
        let seen = log.lock().expect("not poisoned").clone();
        for item in ["m0", "m1", "p0", "p1", "x0", "x1"] {
            let virtual_flags: Vec<bool> = seen
                .iter()
                .filter(|(name, _)| name == item)
                .map(|(_, is_virtual)| *is_virtual)
                .collect();
            assert!(!virtual_flags.is_empty(), "{mode:?} {item}: {seen:?}");
            assert!(
                virtual_flags.iter().all(|v| *v == flat),
                "{mode:?} {item}: {seen:?}"
            );
        }
        for container in ["m", "p", "x"] {
            assert!(
                seen.iter().any(|(name, v)| name == container && !v),
                "{mode:?} {container}: {seen:?}"
            );
        }
    }
}

/// #126 rule 5.3: the root task captures its task context at `spawn_root`
/// dispatch, so the handler runs in the invocation frame only.
#[tokio::test]
async fn root_task_captures_invocation_frame() {
    let result = run_with(
        Wraps::default(),
        |_event: (), _ctx: DurableContext| async move { Ok::<_, BoxError>(frames()) },
    )
    .await;
    assert_eq!(result.output(), Some(&path(&[])));
}

/// Where a hook ran, and the frames it read.
type HookSeen = (&'static str, Vec<String>);

/// Wraps as [`Wraps`] does, and records, for step `step`, the frames of
/// its user-function start hook's synchronous part and of its returned
/// future.
#[derive(Clone, Default)]
struct WorkProbe(std::sync::Arc<std::sync::Mutex<Vec<HookSeen>>>);

impl Plugin for WorkProbe {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        enter_invocation(next);
    }

    fn wrap_user_function(&self, info: &durable::plugin::OperationInfo, next: &mut dyn FnMut()) {
        enter_user_function(info, next);
    }

    fn on_user_function_start(
        &self,
        info: &durable::plugin::OperationInfo,
    ) -> durable::plugin::HookWork {
        if info.name() != Some("step") {
            return Ok(None);
        }
        self.0
            .lock()
            .expect("not poisoned")
            .push(("sync", frames()));
        let log = std::sync::Arc::clone(&self.0);
        Ok(Some(Box::pin(async move {
            log.lock().expect("not poisoned").push(("future", frames()));
            Ok(())
        })))
    }
}

/// #126 rules 5.5 and 14: a hook future runs on its own plugin-work task,
/// which carries no task context. In child `c`, the start hook's
/// synchronous part runs in the poller's frames `[I, c]`, and its future
/// reads no frame at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plugin_work_task_carries_no_frames() {
    let probe = WorkProbe::default();
    let log = std::sync::Arc::clone(&probe.0);
    let result = run_with(probe, |_event: (), ctx: DurableContext| async move {
        Ok(ctx
            .run_in_child_context(|child| async move {
                Ok(child.step(|_| async { Ok(1_u32) }).name("step").await?)
            })
            .name("c")
            .await?)
    })
    .await;
    assert_eq!(result.output(), Some(&1));
    let seen = log.lock().expect("not poisoned").clone();
    assert_eq!(
        seen,
        [("sync", path(&["c"])), ("future", Vec::new())],
        "{seen:?}"
    );
}

/// #126 rules 6.2 and #151: in child `c`, a `.spawn()`ed step whose handle
/// is dropped keeps running in the frames it captured. Its body waits 1 s,
/// then reads `[I, c, step]`, while `c` awaits a 2 s durable wait; the step
/// settles.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_handle_task_keeps_frames() {
    let read = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let handler_read = std::sync::Arc::clone(&read);
    let run = run_with(Wraps::default(), move |_event: (), ctx: DurableContext| {
        let read = std::sync::Arc::clone(&handler_read);
        async move {
            Ok(ctx
                .run_in_child_context(move |child| async move {
                    drop(
                        child
                            .step(move |_| async move {
                                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                                read.lock().expect("not poisoned").push(frames());
                                Ok(1_u32)
                            })
                            .name("step")
                            .spawn(),
                    );
                    child
                        .wait(std::time::Duration::from_secs(2))
                        .name("w")
                        .await?;
                    Ok(0_u32)
                })
                .name("c")
                .await?)
        }
    });
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), run)
        .await
        .expect("the run ends within 5 s");
    assert_eq!(result.output(), Some(&0));
    let step = result.operation("step").expect("the step recorded");
    assert_eq!(step.status(), durable::OperationStatus::Succeeded);
    let read = read.lock().expect("not poisoned").clone();
    assert_eq!(read, [path(&["c", "step"])]);
}

/// Emits a `probe` event when dropped.
struct EmitsOnDrop;

impl Drop for EmitsOnDrop {
    fn drop(&mut self) {
        tracing::info!(target: "probe", "drop");
    }
}

/// #126 rules 5.1 and 6.2, without plugins: a `.spawn()`ed step's task
/// enters the span current at `.spawn()` on every poll and on its drop. Its
/// body emits an event, then holds a guard that emits one when dropped and
/// stays pending; the enclosing scope closes, which aborts the task. Both
/// events are inside `outer`.
///
/// A child body's scope closing is what aborts the task, and a child body
/// runs in its own detached span, so `outer` is entered inside the child
/// body rather than around the handler.
/// Current-thread, so the thread-local subscriber sees every event.
#[tokio::test]
async fn span_entered_on_poll_and_on_drop() {
    use tracing::Instrument as _;
    use tracing_subscriber::layer::SubscriberExt as _;

    let names = ScopeNames::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(names.clone()));
    let runner = LocalRunner::new();
    let run = runner.run(
        |_event: (), ctx: DurableContext| async move {
            Ok::<_, BoxError>(
                ctx.run_in_child_context(|child| {
                    async move {
                        let ran = std::sync::Arc::new(tokio::sync::Notify::new());
                        let body_ran = std::sync::Arc::clone(&ran);
                        drop(
                            child
                                .step(move |_| async move {
                                    tracing::info!(target: "probe", "body");
                                    let _guard = EmitsOnDrop;
                                    body_ran.notify_one();
                                    std::future::pending::<()>().await;
                                    Ok(0_u32)
                                })
                                .name("s")
                                .spawn(),
                        );
                        ran.notified().await;
                        Ok(1_u32)
                    }
                    .instrument(tracing::info_span!("outer"))
                })
                .name("c")
                .await?,
            )
        },
        (),
    );
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), run)
        .await
        .expect("the run ends within 5 s")
        .expect("runner");
    assert_eq!(result.output(), Some(&1));
    let seen = names.0.lock().expect("not poisoned").clone();
    assert_eq!(seen.len(), 2, "a body and a drop event: {seen:?}");
    for scope in &seen {
        assert!(scope.contains(&"outer"), "{seen:?}");
    }
}

/// Reads the frames into its log when dropped.
struct ReadsOnDrop(std::sync::Arc<std::sync::Mutex<Vec<Vec<String>>>>);

impl Drop for ReadsOnDrop {
    fn drop(&mut self) {
        self.0.lock().expect("not poisoned").push(frames());
    }
}

/// #126 rules 3 and 4.5: when a scope closes, a pending body is dropped
/// inside the current frames and the wraps of the plugins that started it.
/// A `.spawn()`ed step `op` in child `c` holds a guard and stays pending on
/// `std::future::pending()`; `c` returns, its scope closes, and the guard
/// reads `[I, c, op]` in its `Drop`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_body_drops_inside_frames() {
    let read = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let handler_read = std::sync::Arc::clone(&read);
    let run = run_with(Wraps::default(), move |_event: (), ctx: DurableContext| {
        let read = std::sync::Arc::clone(&handler_read);
        async move {
            Ok(ctx
                .run_in_child_context(move |child| async move {
                    let ran = std::sync::Arc::new(tokio::sync::Notify::new());
                    let body_ran = std::sync::Arc::clone(&ran);
                    drop(
                        child
                            .step(move |_| async move {
                                let _guard = ReadsOnDrop(read);
                                body_ran.notify_one();
                                std::future::pending::<()>().await;
                                Ok(0_u32)
                            })
                            .name("op")
                            .spawn(),
                    );
                    ran.notified().await;
                    Ok(1_u32)
                })
                .name("c")
                .await?)
        }
    });
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), run)
        .await
        .expect("the run ends within 5 s");
    assert_eq!(result.output(), Some(&1));
    let read = read.lock().expect("not poisoned").clone();
    assert_eq!(read, [path(&["c", "op"])]);
}

/// Without plugins, no code runs in a frame: the handler, a `.spawn()`ed
/// step and a map item read no frame.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_plugins_no_code_is_wrapped() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let in_handler = frames();
                let spawned = ctx.step(|_| async { Ok(frames()) }).name("spawned").spawn();
                let items = ctx
                    .map(
                        vec![1_u32],
                        |_child, _item, _index| async move { Ok(frames()) },
                    )
                    .name("m")
                    .await?;
                let mut seen = vec![in_handler, spawned.await?];
                seen.extend(items);
                Ok::<_, BoxError>(seen)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&vec![Vec::<String>::new(); 3]));
}

/// Records, at the synchronous call of `on_invocation_end`, whether the
/// invocation wrap was entered.
struct EndProbe(std::sync::Arc<std::sync::Mutex<Vec<bool>>>);

impl Plugin for EndProbe {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        let outer = IN_INVOCATION.with(|flag| flag.replace(true));
        next();
        IN_INVOCATION.with(|flag| flag.set(outer));
    }

    fn on_invocation_end<'a>(
        &'a self,
        _info: &'a durable::plugin::InvocationEndInfo,
    ) -> durable::plugin::HookFuture<'a> {
        self.0.lock().expect("not poisoned").push(in_invocation());
        Box::pin(std::future::ready(Ok(())))
    }
}

struct EndFactory(std::sync::Arc<std::sync::Mutex<Vec<bool>>>);

impl PluginFactory for EndFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(EndProbe(std::sync::Arc::clone(&self.0)))))
    }
}

/// `wrap_invocation` is active while `on_invocation_end` runs, on every
/// invocation, the suspending one and the final one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_invocation_wrap_is_active_during_on_invocation_end() {
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(EndFactory(std::sync::Arc::clone(&seen)))
                .build()
                .expect("valid options"),
        )
        .run(
            |_event: (), ctx: DurableContext| async move {
                ctx.wait(std::time::Duration::from_secs(1))
                    .name("w")
                    .await?;
                Ok::<_, BoxError>(1_u32)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&1));
    let seen = seen.lock().expect("not poisoned").clone();
    assert_eq!(
        seen,
        [true, true],
        "one end hook per invocation, each inside the wrap"
    );
}

/// The names of the spans in scope of each `probe` event.
#[derive(Clone, Default)]
struct ScopeNames(std::sync::Arc<std::sync::Mutex<Vec<Vec<&'static str>>>>);

impl<S> tracing_subscriber::Layer<S> for ScopeNames
where
    S: tracing::Subscriber + for<'l> tracing_subscriber::registry::LookupSpan<'l>,
{
    fn on_event(&self, event: &tracing::Event<'_>, ctx: tracing_subscriber::layer::Context<'_, S>) {
        if event.metadata().target() != "probe" {
            return;
        }
        let names = ctx
            .event_scope(event)
            .map(|scope| scope.from_root().map(|span| span.name()).collect())
            .unwrap_or_default();
        self.0.lock().expect("not poisoned").push(names);
    }
}

/// A step created with `.future()` inside span `created` and joined inside
/// span `joined` runs its body inside `created`, the span current where
/// the user created the operation, not inside `joined`.
#[tokio::test]
async fn a_future_step_runs_inside_the_span_of_its_creation() {
    use tracing::Instrument as _;
    use tracing_subscriber::layer::SubscriberExt as _;

    let names = ScopeNames::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(names.clone()));
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: DurableContext| async move {
                let step = {
                    let _created = tracing::info_span!("created").entered();
                    ctx.step(|_| async {
                        tracing::info!(target: "probe", "in body");
                        Ok(1_u32)
                    })
                    .name("s")
                    .future()
                };
                let joined = ctx
                    .try_join_all([step])
                    .name("j")
                    .into_future()
                    .instrument(tracing::info_span!("joined"))
                    .await?;
                Ok::<_, BoxError>(joined)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&vec![1]));
    let seen = names.0.lock().expect("not poisoned").clone();
    assert!(!seen.is_empty(), "the body ran");
    for scope in &seen {
        assert!(scope.contains(&"created"), "{scope:?}");
        assert!(!scope.contains(&"joined"), "{scope:?}");
    }
}

/// Records the user-function hooks of parallel branches, and the active
/// user-function wraps when a step's start hook runs.
struct FnHooks(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

thread_local! {
    /// The user functions whose wrap is entered on this thread.
    static IN_FN: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl Plugin for FnHooks {
    fn wrap_user_function(&self, info: &durable::plugin::OperationInfo, next: &mut dyn FnMut()) {
        IN_FN.with(|f| {
            f.borrow_mut()
                .push(info.name().unwrap_or_default().to_owned());
        });
        next();
        IN_FN.with(|f| f.borrow_mut().pop());
    }

    fn on_user_function_start(
        &self,
        info: &durable::plugin::OperationInfo,
    ) -> durable::plugin::HookWork {
        let name = info.name().unwrap_or_default();
        let line = match info.sub_type() {
            Some("ParallelBranch") => {
                format!(
                    "fn-start {name} parent={}",
                    info.parent_id().unwrap_or_default()
                )
            }
            Some("Step") => format!(
                "step-start {name} in {:?}",
                IN_FN.with(|f| f.borrow().clone())
            ),
            _ => return Ok(None),
        };
        self.0.lock().expect("not poisoned").push(line);
        Ok(None)
    }

    fn on_user_function_end(
        &self,
        info: &durable::plugin::OperationInfo,
        outcome: Result<(), &durable::WireError>,
    ) -> durable::plugin::HookWork {
        if info.sub_type() == Some("ParallelBranch") {
            let status = if outcome.is_ok() {
                "SUCCEEDED"
            } else {
                "FAILED"
            };
            self.0.lock().expect("not poisoned").push(format!(
                "fn-end {} {status} parent={}",
                info.name().unwrap_or_default(),
                info.parent_id().unwrap_or_default()
            ));
        }
        Ok(None)
    }
}

struct FnFactory(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl PluginFactory for FnFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(FnHooks(std::sync::Arc::clone(&self.0)))))
    }
}

async fn run_fn_hooks<F, Fut>(
    handler: F,
) -> (durable::test_util::TestResult<Vec<String>>, Vec<String>)
where
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<Vec<String>, BoxError>> + Send + 'static,
{
    let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(FnFactory(std::sync::Arc::clone(&log)))
                .build()
                .expect("valid options"),
        )
        .run(handler, ())
        .await
        .expect("runner");
    let lines = log.lock().expect("not poisoned").clone();
    (result, lines)
}

/// Shaped like conformance 10-12: two parallel branches at concurrency 1
/// get one user-function start and end each, in order, each with the
/// parallel as its parent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parallel_branches_get_user_function_hooks_in_order() {
    let (result, lines) = run_fn_hooks(|_event: (), ctx: DurableContext| async move {
        let branches = vec![
            durable::Branch::new("0", |_child| async move { Ok("task-1".to_owned()) }),
            durable::Branch::new("1", |_child| async move { Ok("task-2".to_owned()) }),
        ];
        Ok(ctx
            .parallel(branches)
            .name("parallel")
            .max_concurrency(1)
            .await?)
    })
    .await;
    assert_eq!(
        result.output(),
        Some(&vec!["task-1".to_owned(), "task-2".to_owned()])
    );
    let parallel = result.operation("parallel").expect("the parallel ran");
    let id = parallel.id();
    assert_eq!(
        lines,
        [
            format!("fn-start 0 parent={id}"),
            format!("fn-end 0 SUCCEEDED parent={id}"),
            format!("fn-start 1 parent={id}"),
            format!("fn-end 1 SUCCEEDED parent={id}"),
        ]
    );
}

/// Re-entry: a step inside a parallel branch fires its user-function start
/// hook while the container's and the branch's wraps are active, because
/// the step's runner runs inside the branch body's poll, which runs inside
/// the container's frames.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_step_in_a_branch_starts_inside_the_branch_wrap() {
    let (_result, lines) = run_fn_hooks(|_event: (), ctx: DurableContext| async move {
        let branches = vec![durable::Branch::new(
            "0",
            |child: DurableContext| async move {
                Ok(child
                    .step(|_| async { Ok("inner".to_owned()) })
                    .name("inner")
                    .await?)
            },
        )];
        Ok(ctx.parallel(branches).name("parallel").await?)
    })
    .await;
    assert!(
        lines.contains(&"step-start inner in [\"parallel\", \"0\"]".to_owned()),
        "{lines:?}"
    );
}

/// Records every user-function hook and the attempt start, with the info
/// fields the tests below check, and a `probe` event in each start.
struct Rec(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl Rec {
    fn line(&self, line: String) {
        self.0.lock().expect("not poisoned").push(line);
    }
}

fn describe(info: &durable::plugin::OperationInfo) -> String {
    format!(
        "{} attempt={:?} status={:?} replay={} start={:?}",
        info.name().unwrap_or_default(),
        info.attempt(),
        info.status(),
        info.is_replay(),
        info.start_timestamp()
    )
}

impl Plugin for Rec {
    fn wrap_user_function(&self, info: &durable::plugin::OperationInfo, next: &mut dyn FnMut()) {
        self.line(format!("wrap {}", describe(info)));
        IN_FN.with(|f| {
            f.borrow_mut()
                .push(info.name().unwrap_or_default().to_owned());
        });
        next();
        IN_FN.with(|f| f.borrow_mut().pop());
    }

    fn on_attempt_start(&self, info: &durable::plugin::AttemptInfo) -> durable::plugin::HookWork {
        self.line(format!("attempt-start {}", describe(info.operation())));
        Ok(None)
    }

    fn on_operation_start(
        &self,
        info: &durable::plugin::OperationInfo,
    ) -> durable::plugin::HookWork {
        self.line(format!("op-start {}", describe(info)));
        Ok(None)
    }

    fn on_user_function_start(
        &self,
        info: &durable::plugin::OperationInfo,
    ) -> durable::plugin::HookWork {
        tracing::info!(target: "probe", "start");
        self.line(format!("fn-start {}", describe(info)));
        Ok(None)
    }

    fn on_user_function_end(
        &self,
        info: &durable::plugin::OperationInfo,
        outcome: Result<(), &durable::WireError>,
    ) -> durable::plugin::HookWork {
        if let Err(wire) = outcome {
            self.line(format!("fn-end-error {:?}", wire.error_message()));
        }
        self.line(format!("fn-end {}", describe(info)));
        Ok(None)
    }
}

struct RecFactory(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl PluginFactory for RecFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(Rec(std::sync::Arc::clone(&self.0)))))
    }
}

async fn run_rec<F, Fut, O>(handler: F) -> (durable::test_util::TestResult<O>, Vec<String>)
where
    F: Fn((), DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    O: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(RecFactory(std::sync::Arc::clone(&log)))
                .build()
                .expect("valid options"),
        )
        .run(handler, ())
        .await
        .expect("runner");
    let lines = log.lock().expect("not poisoned").clone();
    (result, lines)
}

/// The user-function hooks and the wrap of a step attempt see
/// the attempt's info: attempt 1, status `Started`, `is_replay` false, and
/// the start timestamp `on_attempt_start` saw.
#[tokio::test]
async fn a_step_attempts_user_function_hooks_see_the_started_attempt() {
    let (result, lines) = run_rec(|_event: (), ctx: DurableContext| async move {
        Ok(ctx.step(|_| async { Ok(1_u32) }).name("s").await?)
    })
    .await;
    assert_eq!(result.output(), Some(&1));
    let attempt = lines
        .iter()
        .find_map(|l| l.strip_prefix("attempt-start "))
        .expect("the attempt started")
        .to_owned();
    assert!(
        attempt.contains("attempt=Some(1) status=Some(Started) replay=false start=Some("),
        "{attempt}"
    );
    for hook in ["fn-start", "wrap", "fn-end"] {
        let line = format!("{hook} {attempt}");
        assert!(lines.contains(&line), "{line} in {lines:?}");
    }
}

/// A plugin's start hook runs outside the SDK's step span; the
/// body's event runs inside it.
#[tokio::test]
async fn a_start_hook_runs_outside_the_step_span_and_the_body_inside_it() {
    use tracing_subscriber::layer::SubscriberExt as _;

    let names = ScopeNames::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(names.clone()));
    let (result, _lines) = run_rec(|_event: (), ctx: DurableContext| async move {
        Ok(ctx
            .step(|_| async {
                tracing::info!(target: "probe", "body");
                Ok(1_u32)
            })
            .name("s")
            .await?)
    })
    .await;
    assert_eq!(result.output(), Some(&1));
    let seen = names.0.lock().expect("not poisoned").clone();
    // The start hook's event comes first, the body's second.
    assert_eq!(seen.len(), 2, "one start and one body event: {seen:?}");
    let start = seen.first().expect("a start event");
    let body = seen.get(1).expect("a body event");
    assert!(!start.contains(&"durable_operation"), "{start:?}");
    assert!(body.contains(&"durable_operation"), "{body:?}");
}

/// A `with_retry` that fails once and succeeds has
/// user-function hooks only for the outer block; its attempt-1 and
/// attempt-2 children have none, and their bodies run inside the outer
/// block's wrap.
#[tokio::test]
async fn with_retry_attempts_are_not_user_functions() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let (result, lines) = run_rec(move |_event: (), ctx: DurableContext| {
        let calls = std::sync::Arc::clone(&calls);
        async move {
            Ok(ctx
                .with_retry(move |_child, _attempt| {
                    let calls = std::sync::Arc::clone(&calls);
                    async move {
                        let wraps = IN_FN.with(|f| f.borrow().clone());
                        assert_eq!(wraps, ["block"], "the attempt runs inside the block's wrap");
                        if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                            return Err::<u32, BoxError>("once".into());
                        }
                        Ok(5)
                    }
                })
                .name("block")
                .retry_strategy(|_err, attempt| {
                    if attempt >= 2 {
                        durable::RetryDecision::Stop
                    } else {
                        durable::RetryDecision::Retry {
                            delay: std::time::Duration::from_secs(1),
                        }
                    }
                })
                .await?)
        }
    })
    .await;
    assert_eq!(result.output(), Some(&5));
    let hooks: Vec<&String> = lines
        .iter()
        .filter(|l| l.starts_with("fn-start") || l.starts_with("fn-end"))
        .collect();
    let names: Vec<String> = hooks
        .iter()
        .map(|l| l.split(" attempt=").next().unwrap_or_default().to_owned())
        .collect();
    // The retry delay suspends the first invocation with the block
    // pending, so the second invocation runs the block again: one start
    // per invocation, and one end, in the invocation that finished it.
    assert_eq!(
        names,
        ["fn-start block", "fn-start block", "fn-end block"],
        "{lines:?}"
    );
    assert_eq!(result.invocations().len(), 2);
}

/// A child's user-function start receives the info its
/// `on_operation_start` received, with `is_replay` the child context's
/// `is_replaying()` at the body's start: false for a new child, true for a
/// STARTED child whose body has an inner record (the wait `w`).
#[tokio::test]
async fn a_childs_user_function_start_replays_as_its_context_does() {
    let (result, lines) = run_rec(|_event: (), ctx: DurableContext| async move {
        Ok(ctx
            .run_in_child_context(|child| async move {
                child
                    .wait(std::time::Duration::from_secs(1))
                    .name("w")
                    .await?;
                Ok(2_u32)
            })
            .name("c")
            .await?)
    })
    .await;
    assert_eq!(result.output(), Some(&2));
    assert_eq!(result.invocations().len(), 2);
    let resumed: Vec<&str> = lines
        .iter()
        .filter_map(|l| l.strip_prefix("op-start "))
        .filter(|l| l.starts_with("c ") && l.contains("replay=true"))
        .collect();
    assert_eq!(resumed.len(), 1, "{lines:?}");
    let start = format!("fn-start {}", resumed.first().expect("one"));
    assert!(lines.contains(&start), "{start} in {lines:?}");
    let starts: Vec<bool> = lines
        .iter()
        .filter_map(|l| l.strip_prefix("fn-start c "))
        .map(|l| l.contains("replay=true"))
        .collect();
    assert_eq!(starts, [false, true], "{lines:?}");
}

/// An error whose `Display` counts its calls and, at the first, polls an
/// operation the child created and never awaited. The SDK displays the
/// error to derive the wire, before the end hooks, so a closed scope
/// keeps that operation pending and its body never runs.
struct Counting {
    calls: std::sync::Arc<std::sync::atomic::AtomicU32>,
    pending: std::sync::Mutex<Option<durable::DurableFuture<u32>>>,
}

impl std::fmt::Debug for Counting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Counting")
    }
}

impl std::fmt::Display for Counting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let pending = self.pending.lock().expect("not poisoned").take();
        if let Some(mut fut) = pending {
            let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
            let polled = std::pin::Pin::new(&mut fut).poll(&mut cx);
            assert!(polled.is_pending(), "the child's scope is closed");
        }
        f.write_str("counted")
    }
}

impl std::error::Error for Counting {}

/// A child body's error is displayed once; the recorded FAIL
/// error is the end hook's; an operation the child started is refused
/// before the end hook runs (its admission is closed, so it stays pending).
#[tokio::test]
async fn a_child_body_error_is_displayed_once_and_its_scope_closes_first() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (handler_calls, handler_ran) = (std::sync::Arc::clone(&calls), std::sync::Arc::clone(&ran));
    let (result, lines) = run_rec(move |_event: (), ctx: DurableContext| {
        let (calls, ran) = (
            std::sync::Arc::clone(&handler_calls),
            std::sync::Arc::clone(&handler_ran),
        );
        async move {
            let outcome = ctx
                .run_in_child_context(move |child| async move {
                    let step_ran = ran;
                    let late = child
                        .step(move |_| async move {
                            step_ran.store(true, std::sync::atomic::Ordering::SeqCst);
                            Ok(1_u32)
                        })
                        .name("late")
                        .future();
                    Err::<u32, BoxError>(Box::new(Counting {
                        calls,
                        pending: std::sync::Mutex::new(Some(late)),
                    }))
                })
                .name("c")
                .await;
            Ok(outcome.is_err())
        }
    })
    .await;
    assert_eq!(result.output(), Some(&true));
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "{lines:?}"
    );
    assert!(
        !ran.load(std::sync::atomic::Ordering::SeqCst),
        "the abandoned step never ran"
    );
    let recorded = result
        .operation("c")
        .and_then(|op| op.error())
        .and_then(durable::WireError::error_message)
        .map(str::to_owned);
    let ended = format!("fn-end-error {recorded:?}");
    assert!(lines.contains(&ended), "{ended} in {lines:?}");
    assert!(
        result.operation("late").is_none(),
        "the abandoned step recorded nothing"
    );
}

/// A child-like site whose body returns a [`Counting`] error.
#[derive(Clone, Copy, Debug)]
enum ScopeSite {
    MapItem,
    Branch,
    MixedBranch,
}

/// #126 rule 7.9: a map item, a `parallel` branch and a `parallel_mixed`
/// branch close their scope inside the body, as a child context does, so
/// the error's `Display`, which the SDK calls before the end hooks, sees a
/// closed scope: the operation the body started and did not await stays
/// pending, never runs and records nothing.
#[tokio::test]
async fn an_item_or_branch_body_closes_its_scope_before_the_end_hooks() {
    for site in [
        ScopeSite::MapItem,
        ScopeSite::Branch,
        ScopeSite::MixedBranch,
    ] {
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let ran = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (handler_calls, handler_ran) =
            (std::sync::Arc::clone(&calls), std::sync::Arc::clone(&ran));
        let (result, lines) = run_rec(move |_event: (), ctx: DurableContext| {
            let (calls, ran) = (
                std::sync::Arc::clone(&handler_calls),
                std::sync::Arc::clone(&handler_ran),
            );
            let body = move |child: DurableContext| async move {
                let step_ran = ran;
                let late = child
                    .step(move |_| async move {
                        step_ran.store(true, std::sync::atomic::Ordering::SeqCst);
                        Ok(1_u32)
                    })
                    .name("late")
                    .future();
                Err::<u32, BoxError>(Box::new(Counting {
                    calls,
                    pending: std::sync::Mutex::new(Some(late)),
                }))
            };
            async move {
                let failed = match site {
                    ScopeSite::MapItem => ctx
                        .map(vec![1_u32], move |child, _item, _idx| body.clone()(child))
                        .name("b")
                        .await
                        .is_err(),
                    ScopeSite::Branch => ctx
                        .parallel(vec![durable::Branch::new("b", body)])
                        .name("p")
                        .await
                        .is_err(),
                    ScopeSite::MixedBranch => {
                        let mut branches = durable::BranchSet::new();
                        let handle = branches.add("b", body);
                        let batch = ctx.parallel_mixed(branches).name("p").await;
                        batch.is_err() || handle.into_result().is_err()
                    }
                };
                Ok(failed)
            }
        })
        .await;
        assert_eq!(result.output(), Some(&true), "{site:?}: {lines:?}");
        assert!(
            calls.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "{site:?}: the error was displayed"
        );
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "{site:?}: the abandoned step never ran"
        );
        assert!(
            result.operation("late").is_none(),
            "{site:?}: the abandoned step recorded nothing"
        );
    }
}

/// A resumed child with no inner record. Its operation start
/// reports `is_replay` true; its user-function start, wrap and end report
/// the child context's `is_replaying()`, false.
#[tokio::test]
async fn a_resumed_child_without_an_inner_record_runs_its_user_function_live() {
    let (result, lines) = run_rec(|_event: (), ctx: DurableContext| async move {
        let resumed = ctx.is_replaying();
        let child = ctx
            .run_in_child_context(move |_child| async move {
                if !resumed {
                    // The first invocation suspends on the wait below with
                    // this body pending and no record inside the child.
                    std::future::pending::<()>().await;
                }
                Ok(5_u32)
            })
            .name("c")
            .into_future();
        let wait = ctx
            .wait(std::time::Duration::from_secs(1))
            .name("w")
            .into_future();
        let (child, wait) = tokio::join!(child, wait);
        wait?;
        Ok(child?)
    })
    .await;
    assert_eq!(result.output(), Some(&5));
    assert_eq!(result.invocations().len(), 2, "{lines:?}");
    let flags = |prefix: &str| -> Vec<bool> {
        lines
            .iter()
            .filter_map(|l| l.strip_prefix(prefix))
            .map(|l| l.contains("replay=true"))
            .collect()
    };
    assert_eq!(flags("op-start c "), [false, true], "{lines:?}");
    assert_eq!(flags("fn-start c "), [false, false], "{lines:?}");
    // A wrap runs at every poll of the body, so only the values count.
    let wraps = flags("wrap c ");
    assert!(!wraps.is_empty() && wraps.iter().all(|r| !r), "{lines:?}");
    assert_eq!(flags("fn-end c "), [false], "{lines:?}");
}

/// An error whose `Display` calls a durable operation on the context it
/// captured: the body's own child context.
struct CallsInDisplay(DurableContext);

impl std::fmt::Debug for CallsInDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CallsInDisplay")
    }
}

impl std::fmt::Display for CallsInDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        drop(
            self.0
                .wait(std::time::Duration::from_secs(1))
                .name("in-display"),
        );
        f.write_str("calls in display")
    }
}

impl std::error::Error for CallsInDisplay {}

/// Which user function returns the [`CallsInDisplay`] error.
#[derive(Clone, Copy, Debug)]
enum OutcomeSite {
    Child,
    MapItem,
    Branch,
}

/// #126 rule 8.2: a child body, a map item and a parallel branch whose error's
/// `Display` calls the captured context fail the execution with the
/// outcome context usage message.
#[tokio::test]
async fn an_error_display_calling_an_operation_fails_the_execution() {
    for site in [
        OutcomeSite::Child,
        OutcomeSite::MapItem,
        OutcomeSite::Branch,
    ] {
        let result = LocalRunner::new()
            .run(
                move |_event: (), ctx: DurableContext| async move {
                    let failed = match site {
                        OutcomeSite::Child => ctx
                            .run_in_child_context(|child| async move {
                                Err::<u32, BoxError>(Box::new(CallsInDisplay(child)))
                            })
                            .name("c")
                            .await
                            .is_err(),
                        OutcomeSite::MapItem => ctx
                            .map(vec![1_u32], |child, _item, _idx| async move {
                                Err::<u32, BoxError>(Box::new(CallsInDisplay(child)))
                            })
                            .name("m")
                            .await
                            .is_err(),
                        OutcomeSite::Branch => ctx
                            .parallel(vec![durable::Branch::new("b", |child| async move {
                                Err::<u32, BoxError>(Box::new(CallsInDisplay(child)))
                            })])
                            .name("p")
                            .await
                            .is_err(),
                    };
                    Ok::<_, BoxError>(failed)
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(result.output(), None, "{site:?}");
        let message = result
            .error()
            .and_then(durable::WireError::error_message)
            .unwrap_or_default();
        assert!(
            message.contains("called while the SDK records the outcome of operation"),
            "{site:?}: {message}"
        );
        assert!(
            result.operation("in-display").is_none(),
            "{site:?}: nothing recorded"
        );
    }
}

/// One observation of a `.spawn()`ed step: the hook or body, its tokio task,
/// and the user-function wraps entered on its thread.
type SpawnSeen = (String, Option<tokio::task::Id>, Vec<String>);

/// Records where a step's start hooks run: their task and the entered
/// user-function wraps, and emits a `probe` event for the span scope.
struct SpawnProbe(std::sync::Arc<std::sync::Mutex<Vec<SpawnSeen>>>);

fn spawn_seen(log: &std::sync::Mutex<Vec<SpawnSeen>>, what: &str) {
    log.lock().expect("not poisoned").push((
        what.to_owned(),
        tokio::task::try_id(),
        IN_FN.with(|f| f.borrow().clone()),
    ));
    tracing::info!(target: "probe", "{what}");
}

impl Plugin for SpawnProbe {
    fn wrap_user_function(&self, info: &durable::plugin::OperationInfo, next: &mut dyn FnMut()) {
        IN_FN.with(|f| {
            f.borrow_mut()
                .push(info.name().unwrap_or_default().to_owned());
        });
        next();
        IN_FN.with(|f| f.borrow_mut().pop());
    }

    fn on_operation_start(
        &self,
        info: &durable::plugin::OperationInfo,
    ) -> durable::plugin::HookWork {
        if info.name() == Some("s") {
            spawn_seen(&self.0, "op-start");
        }
        Ok(None)
    }

    fn on_attempt_start(&self, info: &durable::plugin::AttemptInfo) -> durable::plugin::HookWork {
        if info.operation().name() == Some("s") {
            spawn_seen(&self.0, "attempt-start");
        }
        Ok(None)
    }
}

struct SpawnProbeFactory(std::sync::Arc<std::sync::Mutex<Vec<SpawnSeen>>>);

impl PluginFactory for SpawnProbeFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(SpawnProbe(std::sync::Arc::clone(&self.0)))))
    }
}

/// Runs a child `outer` that, inside span `caller`, records itself and
/// `.spawn()`s step `s` with `semantics`, then awaits it. Returns the
/// observations and the span scope of each, in order.
async fn run_spawned_step(
    semantics: durable::StepSemantics,
) -> (Vec<SpawnSeen>, Vec<Vec<&'static str>>) {
    use tracing_subscriber::layer::SubscriberExt as _;

    let names = ScopeNames::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(names.clone()));
    let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let body_log = std::sync::Arc::clone(&log);
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(SpawnProbeFactory(std::sync::Arc::clone(&log)))
                .build()
                .expect("valid options"),
        )
        .run(
            move |_event: (), ctx: DurableContext| {
                let body_log = std::sync::Arc::clone(&body_log);
                async move {
                    Ok::<_, BoxError>(
                        ctx.run_in_child_context(move |child| async move {
                            let handle = {
                                let _caller = tracing::info_span!("caller").entered();
                                spawn_seen(&body_log, "caller");
                                let step_log = std::sync::Arc::clone(&body_log);
                                let handle = child
                                    .step(move |_| async move {
                                        spawn_seen(&step_log, "body");
                                        Ok(1_u32)
                                    })
                                    .name("s")
                                    .semantics(semantics)
                                    .spawn();
                                spawn_seen(&body_log, "spawned");
                                handle
                            };
                            Ok(handle.await?)
                        })
                        .name("outer")
                        .await?,
                    )
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&1));
    let seen = log.lock().expect("not poisoned").clone();
    let scopes = names.0.lock().expect("not poisoned").clone();
    (seen, scopes)
}

/// #126 rules 5.2 and 11 (#153): a `.spawn()`ed
/// `AtLeastOncePerRetry` step fires `on_operation_start` on the caller's
/// task before `.spawn()` returns, inside the caller's frames and span, and
/// `on_attempt_start` and the body on the step's own task, inside the frames
/// and span the step captured, which equal the caller's.
/// Current-thread, so the thread-local subscriber sees every event.
#[tokio::test]
async fn a_spawned_step_starts_on_the_caller_in_the_callers_frames_and_span() {
    let (seen, scopes) = run_spawned_step(durable::StepSemantics::AtLeastOncePerRetry).await;
    let order: Vec<&str> = seen.iter().map(|(what, _, _)| what.as_str()).collect();
    assert_eq!(
        order,
        ["caller", "op-start", "spawned", "attempt-start", "body"],
        "{seen:?}"
    );
    let caller = seen.first().expect("the caller").1;
    assert!(caller.is_some());
    for (what, task, frames) in &seen {
        let on_caller = matches!(what.as_str(), "caller" | "op-start" | "spawned");
        assert_eq!(*task == caller, on_caller, "{what} task: {seen:?}");
        let expected: &[&str] = if what == "body" {
            &["outer", "s"]
        } else {
            &["outer"]
        };
        assert_eq!(frames, expected, "{what} frames: {seen:?}");
    }
    assert_eq!(scopes.len(), seen.len(), "{scopes:?}");
    for ((what, _, _), scope) in seen.iter().zip(&scopes) {
        assert!(scope.contains(&"caller"), "{what} span: {scope:?}");
    }
}

/// #126 rule 11: a `.spawn()`ed `AtMostOncePerRetry` step fires both start
/// hooks on its own task after the START response, in the captured frames
/// and span.
/// Current-thread, so the thread-local subscriber sees every event.
#[tokio::test]
async fn a_spawned_at_most_once_step_starts_on_its_own_task() {
    let (seen, scopes) = run_spawned_step(durable::StepSemantics::AtMostOncePerRetry).await;
    let order: Vec<&str> = seen.iter().map(|(what, _, _)| what.as_str()).collect();
    assert_eq!(
        order,
        ["caller", "spawned", "op-start", "attempt-start", "body"],
        "{seen:?}"
    );
    let caller = seen.first().expect("the caller").1;
    for (what, task, frames) in &seen {
        let on_caller = matches!(what.as_str(), "caller" | "spawned");
        assert_eq!(*task == caller, on_caller, "{what} task: {seen:?}");
        let expected: &[&str] = if what == "body" {
            &["outer", "s"]
        } else {
            &["outer"]
        };
        assert_eq!(frames, expected, "{what} frames: {seen:?}");
    }
    assert_eq!(scopes.len(), seen.len(), "{scopes:?}");
    for ((what, _, _), scope) in seen.iter().zip(&scopes) {
        assert!(scope.contains(&"caller"), "{what} span: {scope:?}");
    }
}

/// One observation of [`future_input_keeps_creator_frames`]: the hook or
/// body, and the invocation and user-function wraps entered on its thread.
type FrameSeen = (String, Vec<String>);

/// Records the frames `on_operation_start` of operation `in` runs in, with
/// `I` for the invocation wrap and each user-function wrap's name, and
/// emits a `probe` event for the span scope.
struct FrameProbe(std::sync::Arc<std::sync::Mutex<Vec<FrameSeen>>>);

fn frame_seen(log: &std::sync::Mutex<Vec<FrameSeen>>, what: &str) {
    log.lock()
        .expect("not poisoned")
        .push((what.to_owned(), IN_FN.with(|f| f.borrow().clone())));
    tracing::info!(target: "probe", "{what}");
}

impl Plugin for FrameProbe {
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        IN_FN.with(|f| f.borrow_mut().push("I".to_owned()));
        next();
        IN_FN.with(|f| f.borrow_mut().pop());
    }

    fn wrap_user_function(&self, info: &durable::plugin::OperationInfo, next: &mut dyn FnMut()) {
        IN_FN.with(|f| {
            f.borrow_mut()
                .push(info.name().unwrap_or_default().to_owned());
        });
        next();
        IN_FN.with(|f| f.borrow_mut().pop());
    }

    fn on_operation_start(
        &self,
        info: &durable::plugin::OperationInfo,
    ) -> durable::plugin::HookWork {
        if info.name() == Some("in") {
            frame_seen(&self.0, "op-start");
        }
        Ok(None)
    }
}

struct FrameProbeFactory(std::sync::Arc<std::sync::Mutex<Vec<FrameSeen>>>);

impl PluginFactory for FrameProbeFactory {
    fn create_plugin(&self, _info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError> {
        Ok(Some(Box::new(FrameProbe(std::sync::Arc::clone(&self.0)))))
    }
}

/// Which lazy input [`future_input_keeps_creator_frames`] creates.
#[derive(Debug, Clone, Copy)]
enum LazyInput {
    /// A step, whose start step a combinator runs on its own task.
    Step,
    /// A `wait_for_condition`, which a combinator polls once on a task it
    /// spawns.
    WaitForCondition,
}

/// How child `b` of [`run_cross_child`] awaits the input child `a` made.
#[derive(Debug, Clone, Copy)]
enum AwaitIn {
    /// Inside a `try_join_all` `j`, which begins the input.
    Combinator,
    /// Directly, so the first poll begins it.
    Direct,
}

/// Runs a handler in which child `a`, inside span `A`, creates a lazy
/// input `in` of `kind` with `.future()`, and child `b`, inside span `B`,
/// awaits it as `how` says. Returns the observations of a [`FrameProbe`]
/// and the span scope of each. Child `a` stays open until `b` awaited the
/// input, so the input's context is still open while it runs.
async fn run_cross_child(
    kind: LazyInput,
    how: AwaitIn,
) -> (Vec<FrameSeen>, Vec<Vec<&'static str>>) {
    use tracing::Instrument as _;
    use tracing_subscriber::layer::SubscriberExt as _;

    let names = ScopeNames::default();
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::registry().with(names.clone()));
    let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let body_log = std::sync::Arc::clone(&log);
    let result = LocalRunner::new()
        .options(
            Options::builder()
                .plugin(FrameProbeFactory(std::sync::Arc::clone(&log)))
                .build()
                .expect("valid options"),
        )
        .run(
            move |_event: (), ctx: DurableContext| {
                let body_log = std::sync::Arc::clone(&body_log);
                async move {
                    let (input_tx, input_rx) = tokio::sync::oneshot::channel();
                    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
                    let a = ctx
                        .run_in_child_context(move |child| async move {
                            let input = {
                                let _a = tracing::info_span!("A").entered();
                                lazy_input(&child, kind, &body_log)
                            };
                            if input_tx.send(input).is_err() {
                                return Err::<u32, BoxError>("b stopped".into());
                            }
                            // Child `a` stays open until `b` awaited the input.
                            done_rx.await?;
                            Ok(0_u32)
                        })
                        .name("a")
                        .spawn();
                    let b = ctx
                        .run_in_child_context(move |child| async move {
                            let input = input_rx.await?;
                            let values = match how {
                                AwaitIn::Combinator => {
                                    child
                                        .try_join_all([input])
                                        .name("j")
                                        .into_future()
                                        .instrument(tracing::info_span!("B"))
                                        .await?
                                }
                                AwaitIn::Direct => {
                                    vec![input.instrument(tracing::info_span!("B")).await?]
                                }
                            };
                            let _done = done_tx.send(());
                            Ok::<_, BoxError>(values)
                        })
                        .name("b")
                        .spawn();
                    let (a, b) = tokio::join!(a, b);
                    a?;
                    Ok::<_, BoxError>(b?)
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&vec![1]), "{kind:?} {how:?}");
    let seen = log.lock().expect("not poisoned").clone();
    let scopes = names.0.lock().expect("not poisoned").clone();
    (seen, scopes)
}

/// Creates the lazy input `in` of `kind` on `child`: a step returning 1, or
/// a `wait_for_condition` from 0 whose single check returns 1. The body or
/// check records itself as `body`.
fn lazy_input(
    child: &DurableContext,
    kind: LazyInput,
    log: &std::sync::Arc<std::sync::Mutex<Vec<FrameSeen>>>,
) -> durable::DurableFuture<u32> {
    let log = std::sync::Arc::clone(log);
    match kind {
        LazyInput::Step => child
            .step(move |_| async move {
                frame_seen(&log, "body");
                Ok(1_u32)
            })
            .name("in")
            .future(),
        LazyInput::WaitForCondition => child
            .wait_for_condition(
                move |_ctx, state: u32| {
                    let log = std::sync::Arc::clone(&log);
                    async move {
                        frame_seen(&log, "body");
                        Ok(state + 1)
                    }
                },
                0_u32,
            )
            .name("in")
            .future(),
    }
}

/// #126 rule 5.2: a lazy combinator input runs its
/// start phase inside the task context it captured at `.future()`. The
/// input is created in child `a` inside span `A`, and a `try_join_all` in
/// child `b` inside span `B` polls it. `on_operation_start` sees `[I, a]`
/// and span `A`; the step body and the condition check see `[I, a, in]`,
/// both in span `A`; none sees `B`, and none has `b` innermost.
/// Current-thread, so the thread-local subscriber sees every event.
#[tokio::test]
async fn future_input_keeps_creator_frames() {
    for kind in [LazyInput::Step, LazyInput::WaitForCondition] {
        let (seen, scopes) = run_cross_child(kind, AwaitIn::Combinator).await;
        assert_frames(kind, &seen, &scopes, ("a", "A"), "B");
    }
}

/// #126 rules 4.6 and 5.2: only a gateway or a combinator
/// enters a future's captured context. A `.future()` created in child `a`
/// inside span `A` and awaited directly in child `b` inside span `B` runs
/// in its awaiter's context, both phases: `on_operation_start` sees
/// `[I, b]` and span `B`; the step body and the condition check see
/// `[I, b, in]`, both in span `B`; none sees `A`.
/// Current-thread, so the thread-local subscriber sees every event.
#[tokio::test]
async fn a_directly_awaited_future_runs_in_its_awaiters_context() {
    for kind in [LazyInput::Step, LazyInput::WaitForCondition] {
        let (seen, scopes) = run_cross_child(kind, AwaitIn::Direct).await;
        assert_frames(kind, &seen, &scopes, ("b", "B"), "A");
    }
}

/// The assertions of [`future_input_keeps_creator_frames`] and
/// [`a_directly_awaited_future_runs_in_its_awaiters_context`] for one row:
/// the hook and the body run in child `child` and span `span`, and never in
/// span `other`.
fn assert_frames(
    kind: LazyInput,
    seen: &[FrameSeen],
    scopes: &[Vec<&'static str>],
    (child, span): (&str, &str),
    other: &str,
) {
    let order: Vec<&str> = seen.iter().map(|(what, _)| what.as_str()).collect();
    assert_eq!(order, ["op-start", "body"], "{kind:?}: {seen:?}");
    for (what, frames) in seen {
        // The body or check runs in its own frame, above the frames of
        // the child that created it; the start hook runs below it.
        let expected: Vec<&str> = match what.as_str() {
            "body" => vec!["I", child, "in"],
            _ => vec!["I", child],
        };
        // The frames entered innermost, from the last invocation wrap:
        // the combinator runs a step input's start step in its own
        // poll, so the poller's wraps `[I, b]` are still on the stack
        // below the entered `[I, a]`, and a context plugin's innermost
        // wrap sets the current context (#126 rule 4.4).
        let innermost = frames
            .iter()
            .rposition(|frame| frame == "I")
            .and_then(|at| frames.get(at..))
            .unwrap_or(frames);
        assert_eq!(innermost, expected, "{kind:?} {what} frames: {seen:?}");
    }
    assert_eq!(scopes.len(), seen.len(), "{kind:?}: {scopes:?}");
    for ((what, _), scope) in seen.iter().zip(scopes) {
        assert!(scope.contains(&span), "{kind:?} {what} span: {scope:?}");
        assert!(!scope.contains(&other), "{kind:?} {what} span: {scope:?}");
    }
}
