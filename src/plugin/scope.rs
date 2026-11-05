//! Frames, the per-poll wrap chain, and the user-function runner.
//!
//! The SDK owns and polls every user function. A plugin supplies only a
//! synchronous wrap around each poll and each drop, and start and end
//! hooks the SDK calls and never awaits on the operation's path. So no plugin code ever holds,
//! moves or drives a body.
//!
//! 1. [`Frames`] is an immutable list of the sites whose wraps are
//!    current: the invocation, then each enclosing user function. Each
//!    node holds the plugins that wrap it and, for a user function, the
//!    plugins whose start hook succeeded.
//! 2. The current frames follow the thread, as `tracing::Span::current()`
//!    does: [`Frames::enter`] makes its frames current around its
//!    innermost closure. [`TaskContext::current`](crate::task::TaskContext::current)
//!    captures them, so a task runs inside the frames current when its
//!    operation was created, and an operation awaited directly runs inside
//!    the frames of the code that polls it.
//! 3. [`Frames::enter`] calls every plugin's wrap for every frame,
//!    outermost first, and runs the SDK's closure innermost. A user panic
//!    is caught there, never inside a plugin frame, and resumed after the
//!    outermost wrap returned.
//! 4. [`run_user_function`] interleaves the start and end hooks into its
//!    own frame's wrap chain, as JS's `reduceRight` chain nests each
//!    plugin inside the plugins before it. At every poll it builds its own
//!    node above the current frames. It is the only constructor of
//!    [`Ran`].

use std::cell::RefCell;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::{InvocationPlugins, OperationInfo, Payload, call_plugin, log_hook_panic};
use crate::context::{ActiveBody, current_body, run_body, run_body_sync};
use crate::{BoxError, WireError};

/// One site whose wraps are current.
#[derive(Debug, Clone)]
enum Site {
    /// The invocation, wrapped by every plugin. Its info is
    /// [`InvocationPlugins::info`].
    Invocation,
    /// One user function, wrapped by the plugins whose start hook
    /// returned `Ok`, by index, in registration order.
    UserFunction {
        info: Arc<OperationInfo>,
        started: Arc<[usize]>,
    },
}

/// One frame: its plugins, its site and the frames outside it. Each node
/// holds its plugins, so a node built above any parent, or none, keeps
/// the plugins its started list indexes.
struct Node {
    plugins: Arc<InvocationPlugins>,
    parent: Option<Arc<Node>>,
    site: Site,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("parent", &self.parent)
            .field("site", &self.site)
            .finish_non_exhaustive()
    }
}

/// The sites whose wraps are current, innermost node first. Empty when
/// the invocation has no plugins.
///
/// A value is immutable and cheap to clone.
#[derive(Debug, Clone, Default)]
pub(crate) struct Frames(Option<Arc<Node>>);

thread_local! {
    /// The frames whose innermost closure runs on this thread now.
    static CURRENT: RefCell<Frames> = const { RefCell::new(Frames(None)) };
}

/// Restores the current frames when dropped, so a panic restores them too.
struct Restore(Option<Frames>);

impl Drop for Restore {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            CURRENT.with(|current| *current.borrow_mut() = previous);
        }
    }
}

/// Makes `frames` current until the returned guard drops.
fn make_current(frames: &Frames) -> Restore {
    Restore(Some(
        CURRENT.with(|current| current.replace(frames.clone())),
    ))
}

impl Frames {
    /// No frames: no plugin wrap runs.
    pub(crate) fn none() -> Self {
        Self(None)
    }

    /// The frames current on this thread.
    pub(crate) fn current() -> Self {
        CURRENT.with(|current| current.borrow().clone())
    }

    /// The frames of the invocation: the invocation alone.
    pub(crate) fn invocation(plugins: Option<Arc<InvocationPlugins>>) -> Self {
        Self(plugins.map(|plugins| {
            Arc::new(Node {
                plugins,
                parent: None,
                site: Site::Invocation,
            })
        }))
    }

    /// These frames, plus the frame of the user function `info`, wrapped
    /// by every plugin.
    #[cfg(test)]
    pub(crate) fn with_user_function(&self, info: Arc<OperationInfo>) -> Self {
        Self(self.0.as_ref().map(|top| {
            Arc::new(Node {
                plugins: Arc::clone(&top.plugins),
                parent: Some(Arc::clone(top)),
                site: Site::UserFunction {
                    info,
                    started: (0..top.plugins.plugins.len()).collect(),
                },
            })
        }))
    }

    /// Runs `f` once, inside every plugin's wrap for every frame, with
    /// these frames current.
    ///
    /// 1. Frames nest outermost first. Within a frame, the first
    ///    registered plugin's wrap is outermost.
    /// 2. Each wrap is called inside `catch_unwind`.
    /// 3. A wrap that returns, or panics, without calling `next` gets its
    ///    `next` called by the SDK right after it returned. A second call
    ///    of `next` does nothing. So `f` runs exactly once.
    /// 4. `f` runs inside its own `catch_unwind`, with these frames
    ///    current. A panic of `f` never unwinds through a plugin frame: it
    ///    is resumed with its original payload after the outermost wrap
    ///    returned.
    /// 5. A wrap's panic is logged after the outermost wrap returned.
    /// 6. Each wrap runs with [`ActiveBody::Plugin`] active. Each `next`
    ///    restores the active body that was current before `enter`.
    pub(crate) fn enter<R>(&self, f: impl FnOnce() -> R) -> R {
        self.enter_from(false, f)
    }

    /// [`Frames::enter`], entering only the innermost frame when
    /// `only_top`: the runner's own frame, whose parent is the frames
    /// current at the call, and so entered already.
    fn enter_from<R>(&self, only_top: bool, f: impl FnOnce() -> R) -> R {
        let Some(top) = &self.0 else {
            let _restore = make_current(self);
            return f();
        };
        let outer = current_body();
        let mut f = Some(f);
        let mut out = None;
        let mut innermost = || {
            if let Some(f) = f.take() {
                let _restore = make_current(self);
                out = Some(catch_unwind(AssertUnwindSafe(f)));
            }
        };
        let panics = RefCell::new(Vec::new());
        let level = Level {
            panics: &panics,
            outer: &outer,
        };
        if only_top {
            enter_site(top, &level, &mut innermost);
        } else {
            enter_node(top, &level, &mut innermost);
        }
        log_panics(panics);
        #[expect(clippy::expect_used)]
        // reason: every level calls `next` itself when its wrap did not,
        // so the innermost closure ran exactly once.
        let out = out.expect("the innermost call always runs");
        match out {
            Ok(value) => value,
            Err(payload) => resume_unwind(payload),
        }
    }

    /// `fut`, polled and dropped inside these frames.
    pub(crate) fn instrument<F: Future>(&self, fut: F) -> impl Future<Output = F::Output> {
        Scoped {
            frames: self.clone(),
            inner: Some(Box::pin(fut)),
        }
    }
}

/// The wrap panics of one chain, logged after its outermost wrap.
type Panics<'a> = RefCell<Vec<(&'a str, &'static str, Payload)>>;

fn log_panics(panics: Panics<'_>) {
    for (identity, hook, payload) in panics.into_inner() {
        log_hook_panic(identity, hook, payload);
    }
}

/// What every level of one [`Frames::enter`] shares.
struct Level<'a, 'p> {
    panics: &'p Panics<'a>,
    /// The active body current before `enter`, restored by every `next`.
    outer: &'p ActiveBody,
}

/// Enters the frames of `node`, outermost first, and runs `f` inside them.
fn enter_node<'a>(node: &'a Arc<Node>, level: &Level<'a, '_>, f: &mut dyn FnMut()) {
    match &node.parent {
        Some(parent) => enter_node(parent, level, &mut || {
            enter_site(node, level, f);
        }),
        None => enter_site(node, level, f),
    }
}

/// Calls the wraps of the plugins of `node`'s site, the first outermost,
/// and runs `f` inside the last.
fn enter_site<'a>(node: &'a Node, level: &Level<'a, '_>, f: &mut dyn FnMut()) {
    match &node.site {
        Site::Invocation => {
            let mut all = 0..node.plugins.plugins.len();
            wrap_each(&mut all, node, Target::Invocation, level, f);
        }
        Site::UserFunction { info, started } => {
            let mut started = started.iter().copied();
            wrap_each(&mut started, node, Target::UserFunction(info), level, f);
        }
    }
}

/// What a wrap wraps.
#[derive(Clone, Copy)]
enum Target<'s> {
    Invocation,
    UserFunction(&'s OperationInfo),
}

/// Calls the wrap of each plugin in `slots` for `target`, the first
/// outermost, and runs `f` inside the last.
fn wrap_each<'a>(
    slots: &mut dyn Iterator<Item = usize>,
    node: &'a Node,
    target: Target<'_>,
    level: &Level<'a, '_>,
    f: &mut dyn FnMut(),
) {
    let Some(index) = slots.next() else {
        f();
        return;
    };
    let mut next = Once::new(|| wrap_each(slots, node, target, level, f));
    wrap_one(&node.plugins, index, target, level, &mut next);
}

/// A `next` that runs its closure at most once.
struct Once<F>(Option<F>);

impl<F: FnOnce()> Once<F> {
    fn new(f: F) -> Self {
        Self(Some(f))
    }

    fn call(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

/// Calls plugin `index`'s wrap for `site` around `next`, then calls `next`
/// itself if the wrap did not.
fn wrap_one<'a>(
    plugins: &'a InvocationPlugins,
    index: usize,
    target: Target<'_>,
    level: &Level<'a, '_>,
    next: &mut Once<impl FnOnce()>,
) {
    let slot = plugins.slot(index);
    let outer = level.outer;
    // The wrap runs as plugin code. The code inside it is not, so it gets
    // back the active body it had before the wraps.
    let mut inside = || run_body_sync(outer.clone(), || next.call());
    let (hook, wrapped) = match target {
        Target::Invocation => (
            "wrap_invocation",
            call_plugin(|| slot.plugin.wrap_invocation(plugins.info(), &mut inside)),
        ),
        Target::UserFunction(info) => (
            "wrap_user_function",
            call_plugin(|| slot.plugin.wrap_user_function(info, &mut inside)),
        ),
    };
    if let Err(payload) = wrapped {
        level
            .panics
            .borrow_mut()
            .push((slot.identity.as_str(), hook, payload));
    }
    // The plugin returned without calling `next`, or panicked before it:
    // run the rest of the chain outside this plugin's wrap.
    next.call();
}

/// A future polled, and dropped, inside fixed frames: [`Frames::instrument`].
struct Scoped<F> {
    frames: Frames,
    /// `None` once it finished or was dropped.
    inner: Option<Pin<Box<F>>>,
}

impl<F: Future> Future for Scoped<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        let slot = &mut this.inner;
        this.frames.enter(|| poll_slot(slot, cx))
    }
}

/// Polls the future in `slot`, and drops it in the poll in which it
/// finished, so its destructor sees the same frames as a pending one does
/// in `Drop`.
///
/// A destructor panic of a finished future is the poll's panic. The
/// output is then disposed of with its own panic contained, as the task
/// gateway does, because a second panic while unwinding would abort.
fn poll_slot<F: Future>(slot: &mut Option<Pin<Box<F>>>, cx: &mut Context<'_>) -> Poll<F::Output> {
    let Some(inner) = slot.as_mut() else {
        return Poll::Pending;
    };
    let Poll::Ready(output) = inner.as_mut().poll(cx) else {
        return Poll::Pending;
    };
    let finished = slot.take();
    if let Err(payload) = catch_unwind(AssertUnwindSafe(move || drop(finished))) {
        crate::task::drop_after_first_panic(output);
        resume_unwind(payload);
    }
    Poll::Ready(output)
}

impl<F> Drop for Scoped<F> {
    /// Drops the inner future inside the wraps. A destructor panic is
    /// resumed after the wraps returned, so it propagates as the drop
    /// would have without them.
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            self.frames.enter(move || drop(inner));
        }
    }
}

/// What one run of a user function returned.
///
/// Only [`run_user_function`] creates one. So a record writer that takes a
/// `Ran` cannot write a user function's outcome without its hooks, and
/// always records the one `WireError` the end hooks saw.
#[derive(Debug)]
pub(crate) struct Ran<T, E = BoxError>(Outcome<T, E>);

#[derive(Debug)]
enum Outcome<T, E> {
    Ok(T),
    Err { error: E, wire: WireError },
}

impl<T, E> Ran<T, E> {
    /// A `Ran` for a unit test of an outcome writer: `wire` derives the
    /// error's wire, as the runner does.
    #[cfg(test)]
    pub(crate) fn from_result(result: Result<T, E>, wire: impl FnOnce(&E) -> WireError) -> Self {
        match result {
            Ok(value) => Self(Outcome::Ok(value)),
            Err(error) => {
                let wire = wire(&error);
                Self(Outcome::Err { error, wire })
            }
        }
    }

    /// The outcome the end hooks receive.
    pub(crate) fn outcome(&self) -> Result<(), &WireError> {
        match &self.0 {
            Outcome::Ok(_) => Ok(()),
            Outcome::Err { wire, .. } => Err(wire),
        }
    }

    /// The value, or the error and the `WireError` derived for it.
    pub(crate) fn into_parts(self) -> Result<T, (E, WireError)> {
        match self.0 {
            Outcome::Ok(value) => Ok(value),
            Outcome::Err { error, wire } => Err((error, wire)),
        }
    }
}

/// The wire function of a body that never fails: a container's batch, or
/// a `join_all` join, whose end hooks always see `Ok`.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "a runner's wire function takes its error by reference"
)]
pub(crate) fn never_fails(never: &std::convert::Infallible) -> WireError {
    match *never {}
}

/// Runs one user function live: the only way to get a [`Ran`].
///
/// 1. The call site calls it only from its operation's live branch; the
///    operation's own prelude decides live, waiting, replay and
///    interruption.
/// 2. `plugins` come from the operation's own context, never from the
///    current frames, which give only the parent of the own node. Without
///    plugins no hook or wrap runs.
/// 3. Level `i` of its own frame's chain, at the first poll, calls plugin
///    `i`'s `on_user_function_start`. An `Err` or a panic marks plugin `i`
///    not started: the chain continues with level `i + 1`, without plugin
///    `i`'s wrap or end. Otherwise plugin `i`'s wrap runs around level
///    `i + 1`. So each start runs inside the wraps of the plugins before
///    it, as in JS's `reduceRight` chain. Later polls use the started
///    list.
/// 4. The innermost level builds the own node above the frames current at
///    this poll, also when none are, and polls `body` inside it, with
///    `active` as the active body. `body` is called in the first poll.
///    `span` covers only the body: its call and its polls. The `WireError`
///    is derived once, inside the own node and `active`, outside `span`,
///    right after the body returned.
/// 5. In the poll in which the body returned, level `i` calls plugin `i`'s
///    `on_user_function_end` after plugin `i`'s wrap returned, inside the
///    wraps of the plugins before it: ends run in reverse order.
/// 6. Each returned hook future runs on its own task; the runner awaits
///    nothing but the body and holds no permit.
pub(crate) fn run_user_function<T, E, B>(
    plugins: Option<Arc<InvocationPlugins>>,
    info: Arc<OperationInfo>,
    active: UserBody,
    span: tracing::Span,
    body: impl FnOnce() -> B,
    wire: impl FnOnce(&E) -> WireError,
) -> impl Future<Output = Ran<T, E>>
where
    B: Future<Output = Result<T, E>>,
{
    let outcome = active.id().to_owned();
    run_user_function_at(plugins, info, active.into(), outcome, span, body, wire)
}

/// The active body of a user function that [`run_user_function`] runs. It
/// holds the operation's positional ID, which the wire derivation reports.
pub(crate) enum UserBody {
    /// A child-like body: a child context, a map item, a parallel branch,
    /// a batch container or a `wait_for_callback` submitter context.
    Context(String),
    /// A step body or a `wait_for_condition` check.
    Step(String),
}

impl UserBody {
    /// The operation's positional ID.
    fn id(&self) -> &str {
        match self {
            Self::Context(id) | Self::Step(id) => id,
        }
    }
}

impl From<UserBody> for ActiveBody {
    fn from(body: UserBody) -> Self {
        match body {
            UserBody::Context(id) => Self::Context(id),
            UserBody::Step(id) => Self::Step(id),
        }
    }
}

/// [`run_user_function`] for a body whose active body does not hold the
/// operation's positional ID: a combinator's join, which runs under its
/// caller's active body. `outcome` is the positional ID the wire
/// derivation reports.
pub(crate) fn run_user_function_at<T, E, B>(
    plugins: Option<Arc<InvocationPlugins>>,
    info: Arc<OperationInfo>,
    active: ActiveBody,
    outcome: String,
    span: tracing::Span,
    body: impl FnOnce() -> B,
    wire: impl FnOnce(&E) -> WireError,
) -> impl Future<Output = Ran<T, E>>
where
    B: Future<Output = Result<T, E>>,
{
    // The wire is derived inside the body's scope, under
    // `ActiveBody::Outcome`. Its `Display` and `source` are user code, and
    // an operation they call claims an ID on a live run only, so the claim
    // fails the execution, even on the body's own context.
    let inner = run_body(active, async move {
        // Polled through `poll_slot`, not awaited, so a finished body
        // whose destructor panics cannot abort while its output is held.
        let body = tracing::Instrument::instrument(span.in_scope(body), span);
        let mut body = Some(Box::pin(body));
        match std::future::poll_fn(|cx| poll_slot(&mut body, cx)).await {
            Ok(value) => Ran(Outcome::Ok(value)),
            Err(error) => {
                let wire = run_body_sync(ActiveBody::Outcome(outcome), || wire(&error));
                Ran(Outcome::Err { error, wire })
            }
        }
    });
    Runner {
        info,
        plugins,
        started: None,
        own: None,
        inner: Some(Box::pin(inner)),
    }
}

/// The future [`run_user_function`] returns.
struct Runner<F> {
    info: Arc<OperationInfo>,
    /// The plugins of the operation's own context.
    plugins: Option<Arc<InvocationPlugins>>,
    /// `None` before the first poll; then the started plugins.
    started: Option<Arc<[usize]>>,
    /// The own node of the last poll, reused while its parent is the same.
    own: Option<Arc<Node>>,
    /// `None` once it finished or was dropped.
    inner: Option<Pin<Box<F>>>,
}

/// The own node above the frames current now. Reuses `own` when its parent
/// is the current frame and its started list the same `Arc`, so a poll
/// under the same frames allocates nothing.
fn own_node(
    own: &mut Option<Arc<Node>>,
    plugins: &Arc<InvocationPlugins>,
    info: &Arc<OperationInfo>,
    started: Arc<[usize]>,
) -> Frames {
    let parent = Frames::current().0;
    if let Some(node) = own.as_ref()
        && same_node(node.parent.as_ref(), parent.as_ref())
        && matches!(&node.site, Site::UserFunction { started: kept, .. } if Arc::ptr_eq(kept, &started))
    {
        return Frames(Some(Arc::clone(node)));
    }
    let node = Arc::new(Node {
        plugins: Arc::clone(plugins),
        parent,
        site: Site::UserFunction {
            info: Arc::clone(info),
            started,
        },
    });
    *own = Some(Arc::clone(&node));
    Frames(Some(node))
}

/// Whether two parents are the same node, or both absent.
fn same_node(a: Option<&Arc<Node>>, b: Option<&Arc<Node>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

impl<T, E, F: Future<Output = Ran<T, E>>> Future for Runner<F> {
    type Output = Ran<T, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Ran<T, E>> {
        let this = self.get_mut();
        let first = this.started.is_none();
        let Some(plugins) = this.plugins.clone() else {
            this.started = Some(Arc::from([]));
            return poll_slot(&mut this.inner, cx);
        };
        let candidates: Arc<[usize]> = match &this.started {
            Some(started) => Arc::clone(started),
            None => (0..plugins.plugins.len()).collect(),
        };
        let outer = current_body();
        let panics = RefCell::new(Vec::new());
        let chain = OwnChain {
            plugins: &plugins,
            info: &this.info,
            candidates: &candidates,
            first,
            started: RefCell::new(Vec::new()),
            published: RefCell::new(None),
            out: RefCell::new(None),
            level: Level {
                panics: &panics,
                outer: &outer,
            },
        };
        let (inner, own, info) = (&mut this.inner, &mut this.own, &this.info);
        let mut innermost = |started: Arc<[usize]>| {
            let node = own_node(own, &plugins, info, started);
            let _restore = make_current(&node);
            catch_unwind(AssertUnwindSafe(|| poll_slot(inner, cx)))
        };
        chain.level(0, &mut innermost);
        let OwnChain { published, out, .. } = chain;
        // The `Arc` the innermost level built is kept for every later poll.
        #[expect(clippy::expect_used)]
        // reason: the innermost level always runs (see below), and it
        // publishes the started list before it polls.
        let published = published
            .into_inner()
            .expect("the innermost call always runs");
        this.started = Some(published);
        log_panics(panics);
        #[expect(clippy::expect_used)]
        // reason: every level calls `next` itself when its wrap did not,
        // so the innermost closure ran exactly once.
        match out.into_inner().expect("the innermost call always runs") {
            Ok(polled) => polled,
            Err(payload) => resume_unwind(payload),
        }
    }
}

impl<F> Drop for Runner<F> {
    /// Drops a pending body inside its own frame's wraps, above the frames
    /// current at the drop.
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            match (self.started.clone(), self.plugins.as_ref()) {
                (Some(started), Some(plugins)) => {
                    own_node(&mut self.own, plugins, &self.info, started)
                        .enter_from(true, move || drop(inner));
                }
                _ => drop(inner),
            }
        }
    }
}

/// The result of polling the body: its poll, or its panic.
type Polled<T> = std::thread::Result<Poll<T>>;

/// One poll of a runner's own wrap chain.
struct OwnChain<'a, 'p, T, E> {
    plugins: &'a InvocationPlugins,
    info: &'p Arc<OperationInfo>,
    /// Every plugin at the first poll; the started ones later.
    candidates: &'p Arc<[usize]>,
    first: bool,
    /// The plugins whose start returned `Ok` at this first poll.
    started: RefCell<Vec<usize>>,
    /// The started list the innermost level used: built once at the first
    /// poll, the candidates' `Arc` later.
    published: RefCell<Option<Arc<[usize]>>>,
    out: RefCell<Option<Polled<Ran<T, E>>>>,
    level: Level<'a, 'p>,
}

impl<T, E> OwnChain<'_, '_, T, E> {
    /// Runs level `k`: plugin `candidates[k]`'s start, wrap and end around
    /// the levels inside it.
    fn level(&self, k: usize, innermost: &mut dyn FnMut(Arc<[usize]>) -> Polled<Ran<T, E>>) {
        let Some(&index) = self.candidates.get(k) else {
            let started: Arc<[usize]> = if self.first {
                self.started.borrow().as_slice().into()
            } else {
                Arc::clone(self.candidates)
            };
            *self.published.borrow_mut() = Some(Arc::clone(&started));
            let polled = innermost(started);
            *self.out.borrow_mut() = Some(polled);
            return;
        };
        if self.first {
            let ok = self
                .plugins
                .dispatch_to("on_user_function_start", [index], |plugin| {
                    plugin.on_user_function_start(self.info)
                });
            if ok.is_empty() {
                self.level(k + 1, innermost);
                return;
            }
            self.started.borrow_mut().push(index);
        }
        let mut next = Once::new(|| self.level(k + 1, innermost));
        let target = Target::UserFunction(self.info);
        wrap_one(self.plugins, index, target, &self.level, &mut next);
        if let Some(Ok(Poll::Ready(ran))) = &*self.out.borrow() {
            let outcome = ran.outcome();
            self.plugins
                .dispatch_to("on_user_function_end", [index], |plugin| {
                    plugin.on_user_function_end(self.info, outcome)
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;
    use crate::plugin::test_support::{capture, field, warnings};
    use crate::plugin::{
        HookWork, InvocationInfo, Plugin, PluginFactory, WorkState, create_plugins,
    };

    thread_local! {
        /// The wraps entered on this thread, innermost last. A wrap pushes
        /// its tag before `next` and pops it after.
        static ACTIVE: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    }

    fn active() -> Vec<String> {
        ACTIVE.with(|a| a.borrow().clone())
    }

    /// The label of `info` in the logs: its name, or its ID.
    fn label(info: &OperationInfo) -> String {
        info.name().unwrap_or(info.id()).to_owned()
    }

    /// The current frames, outermost first: `I` for the invocation node,
    /// the label of each user function's node.
    fn frame_ids() -> Vec<String> {
        let mut ids = Vec::new();
        let mut node = Frames::current().0;
        while let Some(current) = node {
            ids.push(match &current.site {
                Site::Invocation => "I".to_owned(),
                Site::UserFunction { info, .. } => label(info),
            });
            node = current.parent.clone();
        }
        ids.reverse();
        ids
    }

    /// How a test plugin behaves.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Mode {
        /// Enters, calls `next`, exits.
        Normal,
        /// The wrap returns without calling `next`.
        Skip,
        /// The wrap calls `next` twice.
        Twice,
        /// The wrap panics `"B"` before `next`.
        PanicBefore,
        /// The wrap calls `next`, then panics `"B"`.
        PanicAfter,
        /// The start hook returns `Err`.
        StartFails,
        /// The start hook panics.
        StartPanics,
        /// The start hook returns a future that never completes.
        StartPending,
        /// The start hook returns a future that returns `Err`.
        StartFutureFails,
        /// The start hook returns a future that panics.
        StartFuturePanics,
        /// The end hook returns `Err`.
        EndFails,
        /// The end hook panics.
        EndPanics,
        /// The end hook returns a future that returns `Err`.
        EndFutureFails,
        /// The end hook returns a future that panics.
        EndFuturePanics,
    }

    /// A future that panics with its message when polled.
    struct PanicsWhenPolled(&'static str);

    impl Future for PanicsWhenPolled {
        type Output = Result<(), BoxError>;

        #[expect(clippy::panic, reason = "the test needs a future that panics")]
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            std::panic::panic_any(self.0)
        }
    }

    /// A future that is pending at its first `pending` polls, then ready.
    /// `on_poll` runs at every poll.
    fn polls(pending: usize, mut on_poll: impl FnMut()) -> impl Future<Output = ()> {
        let mut left = pending;
        std::future::poll_fn(move |cx| {
            on_poll();
            if left == 0 {
                Poll::Ready(())
            } else {
                left -= 1;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
    }

    /// The recording plugin: it logs its start hook (S), its wraps'
    /// `enter` and `exit` (W) and its end hook (E), each with the wraps
    /// active then.
    struct Recorder {
        tag: &'static str,
        mode: Mode,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl Recorder {
        fn wrap(&self, site: &str, next: &mut dyn FnMut()) {
            let tag = format!("{}:{site}", self.tag);
            self.log.lock().unwrap().push(format!("enter {tag}"));
            assert!(self.mode != Mode::PanicBefore, "B");
            if self.mode != Mode::Skip {
                ACTIVE.with(|a| a.borrow_mut().push(tag.clone()));
                next();
                if self.mode == Mode::Twice {
                    next();
                }
                ACTIVE.with(|a| a.borrow_mut().pop());
            }
            self.log.lock().unwrap().push(format!("exit {tag}"));
            assert!(self.mode != Mode::PanicAfter, "B");
        }
    }

    impl Plugin for Recorder {
        fn name(&self) -> &'static str {
            self.tag
        }

        fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
            self.wrap("inv", next);
        }

        fn wrap_user_function(&self, info: &OperationInfo, next: &mut dyn FnMut()) {
            self.wrap(&label(info), next);
        }

        #[expect(clippy::panic, reason = "one mode tests a panicking start hook")]
        fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
            let line = format!("start {}:{} in {:?}", self.tag, label(info), active());
            self.log.lock().unwrap().push(line);
            match self.mode {
                Mode::StartFails => Err("start failed".into()),
                Mode::StartPanics => std::panic::panic_any("start panicked"),
                Mode::StartPending => Ok(Some(Box::pin(std::future::pending()))),
                Mode::StartFutureFails => Ok(Some(Box::pin(async { Err("future failed".into()) }))),
                Mode::StartFuturePanics => Ok(Some(Box::pin(PanicsWhenPolled("future panicked")))),
                _ => Ok(None),
            }
        }

        #[expect(clippy::panic, reason = "one mode tests a panicking end hook")]
        fn on_user_function_end(
            &self,
            info: &OperationInfo,
            outcome: Result<(), &WireError>,
        ) -> HookWork {
            let result = match outcome {
                Ok(()) => "ok".to_owned(),
                Err(e) => format!("err {}", e.error_message().unwrap_or_default()),
            };
            let line = format!(
                "end {}:{} {result} in {:?}",
                self.tag,
                label(info),
                active()
            );
            self.log.lock().unwrap().push(line);
            match self.mode {
                Mode::EndFails => Err("end failed".into()),
                Mode::EndPanics => std::panic::panic_any("end panicked"),
                Mode::EndFutureFails => Ok(Some(Box::pin(async { Err("future failed".into()) }))),
                Mode::EndFuturePanics => Ok(Some(Box::pin(PanicsWhenPolled("future panicked")))),
                _ => Ok(None),
            }
        }
    }

    struct Factory {
        tag: &'static str,
        mode: Mode,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl PluginFactory for Factory {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(Recorder {
                tag: self.tag,
                mode: self.mode,
                log: Arc::clone(&self.log),
            })))
        }
    }

    fn plugins(
        modes: &[(&'static str, Mode)],
        log: &Arc<Mutex<Vec<String>>>,
    ) -> Arc<InvocationPlugins> {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> = modes
            .iter()
            .map(|&(tag, mode)| {
                let factory: Arc<dyn PluginFactory> = Arc::new(Factory {
                    tag,
                    mode,
                    log: Arc::clone(log),
                });
                ("Factory", factory)
            })
            .collect();
        create_plugins(&factories, None, crate::plugin::tests_support_info)
            .expect("factories were given")
    }

    fn info(id: &str) -> Arc<OperationInfo> {
        Arc::new(OperationInfo::for_test(id))
    }

    /// The frames `[invocation, ids...]`.
    fn frames(plugins: &Arc<InvocationPlugins>, ids: &[&str]) -> Frames {
        let mut frames = Frames::invocation(Some(Arc::clone(plugins)));
        for id in ids {
            frames = frames.with_user_function(info(id));
        }
        frames
    }

    /// The invocation frames, which the root task enters.
    fn inv(plugins: &Arc<InvocationPlugins>) -> Frames {
        Frames::invocation(Some(Arc::clone(plugins)))
    }

    fn take(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        std::mem::take(&mut *log.lock().unwrap())
    }

    fn push(log: &Arc<Mutex<Vec<String>>>, line: String) {
        log.lock().unwrap().push(line);
    }

    /// The lines of `log` without the start and end lines.
    fn wrap_lines(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        take(log)
            .into_iter()
            .filter(|l| !l.starts_with("start") && !l.starts_with("end"))
            .collect()
    }

    /// A runner of the step `s` with no SDK span and a wire that keeps
    /// the error's message.
    fn step_runner<T, B>(
        plugins: &Arc<InvocationPlugins>,
        body: impl FnOnce() -> B,
    ) -> impl Future<Output = Ran<T>>
    where
        B: Future<Output = Result<T, BoxError>>,
    {
        run_user_function(
            Some(Arc::clone(plugins)),
            info("s"),
            UserBody::Step("s".to_owned()),
            tracing::Span::none(),
            body,
            |e: &BoxError| WireError::new(Some("Error".to_owned()), Some(e.to_string())),
        )
    }

    /// Waits until every hook future started so far on `plugins` ended.
    async fn settle_work(plugins: &InvocationPlugins) {
        loop {
            let handles = match &mut *plugins.lock_work() {
                WorkState::Open(list) => std::mem::take(list),
                WorkState::Closed => Vec::new(),
            };
            if handles.is_empty() {
                return;
            }
            for handle in handles {
                // A hook future catches its own panic, so the join only
                // reports an abort.
                let _ = handle.await;
            }
        }
    }

    /// Each poll of a step runner in child `c` enters the invocation's,
    /// `c`'s and the step's wraps, outermost frame first and, within a
    /// frame, the first plugin first; then the body; then exits them in
    /// reverse. Each wrap runs once per poll.
    #[test]
    fn wraps_nest_by_frame_then_registration() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal), ("p1", Mode::Normal)], &log);
        let body_log = Arc::clone(&log);
        let mut run = std::pin::pin!(step_runner(&plugins, move || async move {
            polls(1, || push(&body_log, format!("body in {:?}", frame_ids()))).await;
            Ok::<u32, BoxError>(1)
        }));
        let poll = [
            "enter p0:inv",
            "enter p1:inv",
            "enter p0:c",
            "enter p1:c",
            "enter p0:s",
            "enter p1:s",
            "body in [\"I\", \"c\", \"s\"]",
            "exit p1:s",
            "exit p0:s",
            "exit p1:c",
            "exit p0:c",
            "exit p1:inv",
            "exit p0:inv",
        ];
        let c = frames(&plugins, &["c"]);
        assert!(c.enter(|| futures_poll_once(run.as_mut())).is_pending());
        assert_eq!(wrap_lines(&log), poll);
        assert!(c.enter(|| futures_poll_once(run.as_mut())).is_ready());
        assert_eq!(wrap_lines(&log), poll);
    }

    /// In a step done in its first poll, the hooks interleave with the
    /// step's wraps: each start inside the wraps of the plugins before it,
    /// each end after its own wrap returned.
    #[test]
    fn hooks_interleave_with_wraps() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal), ("p1", Mode::Normal)], &log);
        let body_log = Arc::clone(&log);
        let mut run = std::pin::pin!(step_runner(&plugins, move || {
            push(&body_log, "closure called".to_owned());
            async move {
                push(&body_log, "body".to_owned());
                Ok::<u32, BoxError>(1)
            }
        }));
        let Poll::Ready(ran) = futures_poll_once(run.as_mut()) else {
            unreachable!("the body is done in its first poll");
        };
        assert_eq!(ran.into_parts().ok(), Some(1));
        assert_eq!(
            take(&log),
            [
                "start p0:s in []",
                "enter p0:s",
                "start p1:s in [\"p0:s\"]",
                "enter p1:s",
                "closure called",
                "body",
                "exit p1:s",
                "end p1:s ok in [\"p0:s\"]",
                "exit p0:s",
                "end p0:s ok in []",
            ]
        );
    }

    /// A runner first polled with no current frames fires its start and
    /// end hooks once, and its body runs inside its own wrap with only its
    /// own node current. A runner first polled inside `[I, a]` and then
    /// with no current frames keeps its own node too.
    #[test]
    fn runner_polled_under_no_frames() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal)], &log);
        let body = |log: &Arc<Mutex<Vec<String>>>| {
            let log = Arc::clone(log);
            move || async move {
                polls(1, || {
                    push(&log, format!("body {:?} in {:?}", frame_ids(), active()));
                })
                .await;
                Ok::<u32, BoxError>(3)
            }
        };
        let read = "body [\"s\"] in [\"p0:s\"]";
        let mut run = std::pin::pin!(step_runner(&plugins, body(&log)));
        assert!(Frames::current().0.is_none());
        assert!(futures_poll_once(run.as_mut()).is_pending());
        assert!(futures_poll_once(run.as_mut()).is_ready());
        assert_eq!(
            take(&log),
            [
                "start p0:s in []",
                "enter p0:s",
                read,
                "exit p0:s",
                "enter p0:s",
                read,
                "exit p0:s",
                "end p0:s ok in []",
            ]
        );

        let mut run = std::pin::pin!(step_runner(&plugins, body(&log)));
        let a = frames(&plugins, &["a"]);
        assert!(a.enter(|| futures_poll_once(run.as_mut())).is_pending());
        drop(take(&log));
        assert!(futures_poll_once(run.as_mut()).is_ready());
        assert_eq!(
            take(&log),
            ["enter p0:s", read, "exit p0:s", "end p0:s ok in []"]
        );
    }

    /// The synchronous code of a `map` item factory runs inside the item's
    /// SDK span.
    #[cfg(feature = "test-util")]
    #[tokio::test]
    async fn map_item_factory_runs_inside_span() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let (buffer, _guard) = capture();
        let options = crate::Options::builder()
            .plugin(Factory {
                tag: "p0",
                mode: Mode::Normal,
                log: Arc::clone(&log),
            })
            .plugin(Factory {
                tag: "p1",
                mode: Mode::Normal,
                log: Arc::clone(&log),
            })
            .build()
            .expect("valid options");
        let result = crate::test_util::LocalRunner::new()
            .options(options)
            .run(
                |_event: (), ctx: crate::DurableContext| async move {
                    let values = ctx
                        .map(vec![1_u32], |_child, item, _idx| {
                            tracing::info!(target: "probe", "factory");
                            async move { Ok(item) }
                        })
                        .name("m")
                        .await?;
                    Ok::<_, BoxError>(values)
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(result.output(), Some(&vec![1]));
        let item_id = result
            .operation("map-item-0")
            .expect("the item is recorded")
            .id()
            .to_owned();
        let events: Vec<serde_json::Value> = String::from_utf8_lossy(&buffer.lock().unwrap())
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|v| field(v, "target") == "probe")
            .collect();
        assert_eq!(events.len(), 1, "{events:?}");
        let span = events
            .first()
            .and_then(|e| e.get("span"))
            .expect("the event has a span");
        assert_eq!(field(span, "name"), "durable_execution", "{span}");
        assert_eq!(field(span, "operationName"), "map-item-0", "{span}");
        assert_eq!(field(span, "operationId"), item_id, "{span}");
    }

    /// An error whose `Display` counts its calls and records the span and
    /// the active body current at each.
    #[derive(Debug)]
    struct Dc {
        calls: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<String>>>,
    }

    impl std::fmt::Display for Dc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.calls.fetch_add(1, Ordering::Relaxed);
            push(
                &self.seen,
                format!("display in {:?} under {:?}", span_name(), current_body()),
            );
            f.write_str("dc")
        }
    }

    impl std::error::Error for Dc {}

    fn span_name() -> Option<&'static str> {
        tracing::Span::current()
            .metadata()
            .map(tracing::Metadata::name)
    }

    /// The body's factory and polls run inside the SDK span; the wire is
    /// derived once, outside it, under `ActiveBody::Outcome` with the
    /// body's ID, and the end hooks receive it.
    #[test]
    fn wire_derived_outside_span() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal)], &log);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));
        let (factory, polled) = (Arc::clone(&seen), Arc::clone(&seen));
        let error = Dc {
            calls: Arc::clone(&calls),
            seen: Arc::clone(&seen),
        };
        let ran = tracing::subscriber::with_default(tracing_subscriber::registry(), || {
            let outer = tracing::info_span!("outer");
            let _entered = outer.enter();
            let mut run = std::pin::pin!(run_user_function(
                Some(Arc::clone(&plugins)),
                info("m"),
                UserBody::Context("m".to_owned()),
                tracing::info_span!("body_span"),
                move || {
                    push(&factory, format!("factory in {:?}", span_name()));
                    async move {
                        push(&polled, format!("poll in {:?}", span_name()));
                        Err::<(), BoxError>(Box::new(error))
                    }
                },
                |e: &BoxError| WireError::new(Some("Error".to_owned()), Some(e.to_string())),
            ));
            let Poll::Ready(ran) = futures_poll_once(run.as_mut()) else {
                unreachable!("the body fails in its first poll");
            };
            ran
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            take(&seen),
            [
                "factory in Some(\"body_span\")",
                "poll in Some(\"body_span\")",
                "display in Some(\"outer\") under Outcome(\"m\")",
            ]
        );
        let (_, wire) = ran.into_parts().expect_err("the body failed");
        assert_eq!(wire.error_message(), Some("dc"));
        assert!(
            take(&log).contains(&"end p0:m err dc in []".to_owned()),
            "the end hook received the wire"
        );
    }

    /// The runner reads the current frames at every poll: polled in
    /// `[I, a]`, then in `[I, b]`, its body reads `[I, a, s]`, then
    /// `[I, b, s]`, and each poll enters its poller's wraps once.
    #[test]
    fn runner_polled_under_two_parent_chains() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal)], &log);
        let body_log = Arc::clone(&log);
        let mut run = std::pin::pin!(step_runner(&plugins, move || async move {
            polls(1, || push(&body_log, format!("body in {:?}", frame_ids()))).await;
            Ok::<u32, BoxError>(3)
        }));
        let first = frames(&plugins, &["a"]).enter(|| futures_poll_once(run.as_mut()));
        assert!(first.is_pending());
        let Poll::Ready(ran) = frames(&plugins, &["b"]).enter(|| futures_poll_once(run.as_mut()))
        else {
            unreachable!("the body returns at its second poll");
        };
        assert_eq!(ran.into_parts().ok(), Some(3));
        assert_eq!(
            wrap_lines(&log),
            [
                "enter p0:inv",
                "enter p0:a",
                "enter p0:s",
                "body in [\"I\", \"a\", \"s\"]",
                "exit p0:s",
                "exit p0:a",
                "exit p0:inv",
                "enter p0:inv",
                "enter p0:b",
                "enter p0:s",
                "body in [\"I\", \"b\", \"s\"]",
                "exit p0:s",
                "exit p0:b",
                "exit p0:inv",
            ]
        );
    }

    type Call = Arc<dyn Fn() + Send + Sync>;
    type Work = Arc<dyn Fn() -> HookWork + Send + Sync>;

    /// A plugin whose methods call closures. A method without a closure
    /// does what the trait's default does.
    #[derive(Clone, Default)]
    struct Calls {
        name: Option<Call>,
        wrap_invocation: Option<Call>,
        wrap_user_function: Option<Call>,
        user_function_start: Option<Work>,
        user_function_end: Option<Work>,
        operation_start: Option<Work>,
        invocation_start: Option<Call>,
    }

    impl Plugin for Calls {
        fn name(&self) -> &'static str {
            if let Some(name) = &self.name {
                name();
            }
            "calls"
        }

        fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
            if let Some(wrap) = &self.wrap_invocation {
                wrap();
            }
            next();
        }

        fn wrap_user_function(&self, _info: &OperationInfo, next: &mut dyn FnMut()) {
            if let Some(wrap) = &self.wrap_user_function {
                wrap();
            }
            next();
        }

        fn on_user_function_start(&self, _info: &OperationInfo) -> HookWork {
            self.user_function_start.as_ref().map_or(Ok(None), |f| f())
        }

        fn on_user_function_end(
            &self,
            _info: &OperationInfo,
            _outcome: Result<(), &WireError>,
        ) -> HookWork {
            self.user_function_end.as_ref().map_or(Ok(None), |f| f())
        }

        fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
            self.operation_start.as_ref().map_or(Ok(None), |f| f())
        }

        fn on_invocation_start<'a>(
            &'a self,
            _info: &'a InvocationInfo,
        ) -> crate::plugin::HookFuture<'a> {
            if let Some(start) = &self.invocation_start {
                start();
            }
            Box::pin(std::future::ready(Ok(())))
        }
    }

    impl PluginFactory for Calls {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    fn calls_plugins(calls: Calls) -> Arc<InvocationPlugins> {
        let factory: Arc<dyn PluginFactory> = Arc::new(calls);
        create_plugins(
            &[("Calls", factory)],
            None,
            crate::plugin::tests_support_info,
        )
        .expect("a factory was given")
    }

    /// A start hook runs outside the SDK span, in the poller's span; the
    /// step body runs inside the SDK span.
    #[test]
    fn start_hook_outside_sdk_span() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let start_seen = Arc::clone(&seen);
        let plugins = calls_plugins(Calls {
            user_function_start: Some(Arc::new(move || {
                push(&start_seen, format!("start in {:?}", span_name()));
                Ok(None)
            })),
            ..Calls::default()
        });
        let body_seen = Arc::clone(&seen);
        tracing::subscriber::with_default(tracing_subscriber::registry(), || {
            let outer = tracing::info_span!("outer");
            let _entered = outer.enter();
            let mut run = std::pin::pin!(run_user_function(
                Some(Arc::clone(&plugins)),
                info("s"),
                UserBody::Step("s".to_owned()),
                tracing::info_span!("durable_operation"),
                move || async move {
                    tracing::info!("body event");
                    push(&body_seen, format!("body in {:?}", span_name()));
                    Ok::<(), BoxError>(())
                },
                |_: &BoxError| WireError::new(None::<String>, None::<String>),
            ));
            assert!(futures_poll_once(run.as_mut()).is_ready());
        });
        assert_eq!(
            take(&seen),
            [
                "start in Some(\"outer\")",
                "body in Some(\"durable_operation\")",
            ]
        );
    }

    /// When the first plugin's wraps skip `next()`, the handler and the
    /// step body still run once, inside the second plugin's wraps only;
    /// the response is unchanged and nothing is logged.
    #[cfg(feature = "test-util")]
    #[tokio::test]
    async fn wrap_without_next_falls_through() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (buffer, _guard) = capture();
        let options = crate::Options::builder()
            .plugin(Factory {
                tag: "p0",
                mode: Mode::Skip,
                log: Arc::clone(&log),
            })
            .plugin(Factory {
                tag: "p1",
                mode: Mode::Normal,
                log: Arc::clone(&log),
            })
            .build()
            .expect("valid options");
        let handler_seen = Arc::clone(&seen);
        let result = crate::test_util::LocalRunner::new()
            .options(options)
            .run(
                move |_event: (), ctx: crate::DurableContext| {
                    let seen = Arc::clone(&handler_seen);
                    async move {
                        push(&seen, format!("handler in {:?}", active()));
                        let body_seen = Arc::clone(&seen);
                        ctx.step(move |_| async move {
                            push(&body_seen, format!("body in {:?}", active()));
                            Ok(())
                        })
                        .name("s")
                        .await?;
                        Ok::<_, BoxError>("done".to_owned())
                    }
                },
                (),
            )
            .await
            .expect("runner");
        assert_eq!(result.output().map(String::as_str), Some("done"));
        assert_eq!(
            take(&seen),
            ["handler in [\"p1:inv\"]", "body in [\"p1:inv\", \"p1:s\"]",]
        );
        assert!(warnings(&buffer).is_empty(), "{:?}", warnings(&buffer));
    }

    /// A wrap that calls `next()` twice runs the code inside once per
    /// call of the wraps: once for `Frames::enter`, and one body poll per
    /// runner poll.
    #[test]
    fn second_next_call_does_nothing() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Twice), ("p1", Mode::Normal)], &log);
        let runs = Cell::new(0);
        frames(&plugins, &[]).enter(|| runs.set(runs.get() + 1));
        assert_eq!(runs.get(), 1);
        assert_eq!(
            take(&log),
            ["enter p0:inv", "enter p1:inv", "exit p1:inv", "exit p0:inv"]
        );

        let body_polls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&body_polls);
        let mut run = std::pin::pin!(step_runner(&plugins, move || async move {
            polls(1, || {
                counter.fetch_add(1, Ordering::Relaxed);
            })
            .await;
            Ok::<u32, BoxError>(1)
        }));
        assert!(futures_poll_once(run.as_mut()).is_pending());
        assert_eq!(body_polls.load(Ordering::Relaxed), 1);
        assert!(futures_poll_once(run.as_mut()).is_ready());
        assert_eq!(body_polls.load(Ordering::Relaxed), 2);
    }

    /// The one warning of `buffer`: a wrap panic of `p0`.
    fn assert_one_panic_warning(buffer: &Arc<Mutex<Vec<u8>>>, hook: &str) {
        let warnings = warnings(buffer);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        let warning = warnings.first().expect("one warning");
        assert_eq!(
            field(warning, "message"),
            "plugin hook panicked; the panic is ignored"
        );
        assert_eq!(field(warning, "hook"), hook);
        assert!(
            field(warning, "plugin").ends_with("(plugins[0])"),
            "{warning}"
        );
    }

    /// A wrap of `wrap_invocation` or `wrap_user_function` that panics
    /// before `next()` is contained: the rest of the chain runs after it
    /// exited, the code inside once, in the second plugin's wrap. A wrap
    /// that panics after `next()` keeps the result. Each logs one warning.
    #[test]
    fn wrap_panics_are_contained() {
        for mode in [Mode::PanicBefore, Mode::PanicAfter] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let plugins = plugins(&[("p0", mode), ("p1", Mode::Normal)], &log);

            let (buffer, guard) = capture();
            let runs = Cell::new(0);
            let value = inv(&plugins).enter(|| {
                runs.set(runs.get() + 1);
                push(&log, format!("handler in {:?}", active()));
                "handler"
            });
            drop(guard);
            assert_eq!((value, runs.get()), ("handler", 1));
            let expected: &[&str] = if mode == Mode::PanicBefore {
                &[
                    "enter p0:inv",
                    "enter p1:inv",
                    "handler in [\"p1:inv\"]",
                    "exit p1:inv",
                ]
            } else {
                &[
                    "enter p0:inv",
                    "enter p1:inv",
                    "handler in [\"p0:inv\", \"p1:inv\"]",
                    "exit p1:inv",
                    "exit p0:inv",
                ]
            };
            assert_eq!(take(&log), expected, "{mode:?}");
            assert_one_panic_warning(&buffer, "wrap_invocation");

            let (buffer, guard) = capture();
            let body_log = Arc::clone(&log);
            let mut run = std::pin::pin!(step_runner(&plugins, move || async move {
                push(&body_log, format!("body in {:?}", active()));
                Ok::<u32, BoxError>(7)
            }));
            let Poll::Ready(ran) = futures_poll_once(run.as_mut()) else {
                unreachable!("the body is done in its first poll");
            };
            drop(guard);
            assert_eq!(ran.into_parts().ok(), Some(7));
            let expected: &[&str] = if mode == Mode::PanicBefore {
                &[
                    "start p0:s in []",
                    "enter p0:s",
                    "start p1:s in []",
                    "enter p1:s",
                    "body in [\"p1:s\"]",
                    "exit p1:s",
                    "end p1:s ok in []",
                    "end p0:s ok in []",
                ]
            } else {
                &[
                    "start p0:s in []",
                    "enter p0:s",
                    "start p1:s in [\"p0:s\"]",
                    "enter p1:s",
                    "body in [\"p0:s\", \"p1:s\"]",
                    "exit p1:s",
                    "end p1:s ok in [\"p0:s\"]",
                    "exit p0:s",
                    "end p0:s ok in []",
                ]
            };
            assert_eq!(take(&log), expected, "{mode:?}");
            assert_one_panic_warning(&buffer, "wrap_user_function");
        }
    }

    /// A body panic never reaches a plugin frame: every wrap returns
    /// normally, the caller gets the original payload afterwards, and no
    /// end hook runs.
    #[test]
    #[expect(clippy::panic, reason = "the test body panics on purpose")]
    fn body_panic_resumes_after_every_wrap_returned() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal), ("p1", Mode::Normal)], &log);
        let c = frames(&plugins, &["c"]);
        let caught = catch_unwind(AssertUnwindSafe(|| {
            c.enter(|| -> u32 { std::panic::panic_any("kaboom") })
        }));
        let payload = caught.expect_err("the body panicked");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"kaboom"));
        assert_eq!(
            take(&log),
            [
                "enter p0:inv",
                "enter p1:inv",
                "enter p0:c",
                "enter p1:c",
                "exit p1:c",
                "exit p0:c",
                "exit p1:inv",
                "exit p0:inv",
            ]
        );
        assert_eq!(active(), Vec::<String>::new());

        let mut run = std::pin::pin!(step_runner(&plugins, || PanicsWhenPolled("kaboom")));
        let caught = catch_unwind(AssertUnwindSafe(|| {
            c.enter(|| futures_poll_once(run.as_mut()))
        }));
        let payload = caught.expect_err("the body panicked");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"kaboom"));
        let lines = take(&log);
        assert!(!lines.iter().any(|l| l.starts_with("end")), "{lines:?}");
        assert_eq!(lines.last().map(String::as_str), Some("exit p0:inv"));
        assert!(lines.contains(&"exit p0:s".to_owned()), "{lines:?}");
    }

    /// The warnings of `buffer` for the hook `hook` of `p0`.
    fn assert_one_hook_warning(buffer: &Arc<Mutex<Vec<u8>>>, hook: &str, error: Option<&str>) {
        let warnings = warnings(buffer);
        assert_eq!(warnings.len(), 1, "{hook}: {warnings:?}");
        let warning = warnings.first().expect("one warning");
        assert_eq!(field(warning, "plugin"), "p0 (plugins[0])", "{warning}");
        assert_eq!(field(warning, "hook"), hook, "{warning}");
        match error {
            Some(error) => {
                assert_eq!(
                    field(warning, "message"),
                    "plugin hook failed; the error is ignored"
                );
                assert_eq!(field(warning, "error"), error, "{warning}");
            }
            None => assert_eq!(
                field(warning, "message"),
                "plugin hook panicked; the panic is ignored"
            ),
        }
    }

    /// A start or end hook that returns a future keeps its plugin started,
    /// whatever the future does later: the plugin's wrap and end hook run.
    /// The future's `Err` or panic is logged once.
    #[tokio::test]
    async fn failing_hook_future_keeps_the_plugin_started() {
        for (mode, hook, error) in [
            (
                Mode::StartFutureFails,
                "on_user_function_start",
                Some("future failed"),
            ),
            (Mode::StartFuturePanics, "on_user_function_start", None),
            (
                Mode::EndFutureFails,
                "on_user_function_end",
                Some("future failed"),
            ),
            (Mode::EndFuturePanics, "on_user_function_end", None),
        ] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let plugins = plugins(&[("p0", mode), ("p1", Mode::Normal)], &log);
            let (buffer, _guard) = capture();
            let ran: Ran<u32> = step_runner(&plugins, || async { Ok(7) }).await;
            settle_work(&plugins).await;
            assert_eq!(ran.into_parts().ok(), Some(7));
            let lines = take(&log);
            for line in ["enter p0:s", "exit p0:s", "end p0:s ok in []"] {
                assert!(lines.contains(&line.to_owned()), "{mode:?}: {lines:?}");
            }
            assert_one_hook_warning(&buffer, hook, error);
        }
    }

    /// A start hook that returns `Err` or panics skips that plugin's wrap
    /// and end hook; the next plugin starts, wraps the body and ends. An
    /// end hook that returns `Err` or panics leaves the result unchanged.
    /// Each logs one warning.
    #[tokio::test]
    async fn failed_user_function_hook_is_contained() {
        for (mode, hook, error) in [
            (
                Mode::StartFails,
                "on_user_function_start",
                Some("start failed"),
            ),
            (Mode::StartPanics, "on_user_function_start", None),
            (Mode::EndFails, "on_user_function_end", Some("end failed")),
            (Mode::EndPanics, "on_user_function_end", None),
        ] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let plugins = plugins(&[("p0", mode), ("p1", Mode::Normal)], &log);
            let (buffer, _guard) = capture();
            let runs = AtomicUsize::new(0);
            let body_log = Arc::clone(&log);
            let ran: Ran<u32> = step_runner(&plugins, || {
                runs.fetch_add(1, Ordering::Relaxed);
                async move {
                    push(&body_log, format!("body in {:?}", active()));
                    Ok(7)
                }
            })
            .await;
            assert_eq!(ran.into_parts().ok(), Some(7), "{mode:?}");
            assert_eq!(runs.load(Ordering::Relaxed), 1);
            let lines = take(&log);
            if hook == "on_user_function_start" {
                assert_eq!(
                    lines,
                    [
                        "start p0:s in []",
                        "start p1:s in []",
                        "enter p1:s",
                        "body in [\"p1:s\"]",
                        "exit p1:s",
                        "end p1:s ok in []",
                    ],
                    "{mode:?}"
                );
            } else {
                assert!(
                    lines.contains(&"end p0:s ok in []".to_owned()),
                    "{mode:?}: {lines:?}"
                );
            }
            assert_one_hook_warning(&buffer, hook, error);
        }
    }

    /// The end hooks run in the poll in which the body returned: none in
    /// the first poll of a body done at its second, both in the second.
    #[test]
    fn end_hook_runs_in_the_poll_of_the_return() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal), ("p1", Mode::Normal)], &log);
        let body_log = Arc::clone(&log);
        let mut run = std::pin::pin!(step_runner(&plugins, move || async move {
            push(&body_log, "body".to_owned());
            polls(1, || {}).await;
            Ok::<u32, BoxError>(1)
        }));
        assert!(futures_poll_once(run.as_mut()).is_pending());
        assert_eq!(
            take(&log),
            [
                "start p0:s in []",
                "enter p0:s",
                "start p1:s in [\"p0:s\"]",
                "enter p1:s",
                "body",
                "exit p1:s",
                "exit p0:s",
            ]
        );
        let Poll::Ready(ran) = futures_poll_once(run.as_mut()) else {
            unreachable!("the body returns at its second poll");
        };
        assert_eq!(ran.into_parts().ok(), Some(1));
        assert_eq!(
            take(&log),
            [
                "enter p0:s",
                "enter p1:s",
                "exit p1:s",
                "end p1:s ok in [\"p0:s\"]",
                "exit p0:s",
                "end p0:s ok in []",
            ]
        );
    }

    /// A start hook whose future never completes does not delay the user
    /// function: one runner poll returns `Ready`, and the future pends on
    /// its own task.
    #[tokio::test]
    async fn pending_start_future_does_not_delay_the_body() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::StartPending), ("p1", Mode::Normal)], &log);
        let mut run = std::pin::pin!(step_runner(&plugins, || async { Ok::<u32, BoxError>(1) }));
        let Poll::Ready(ran) = inv(&plugins).enter(|| futures_poll_once(run.as_mut())) else {
            unreachable!("the runner does not wait for the start future");
        };
        assert_eq!(ran.into_parts().ok(), Some(1));
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        let WorkState::Open(list) = &*plugins.lock_work() else {
            unreachable!("the drain has not closed");
        };
        assert_eq!(list.len(), 1, "the start future runs on its own task");
        assert!(!list.iter().any(tokio::task::JoinHandle::is_finished));
        for handle in list {
            handle.abort();
        }
    }

    /// An error whose `Display` calls a durable operation.
    struct CallsOnDisplay(Call);

    impl std::fmt::Debug for CallsOnDisplay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("CallsOnDisplay")
        }
    }

    impl std::fmt::Display for CallsOnDisplay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            (self.0)();
            f.write_str("calls on display")
        }
    }

    impl std::error::Error for CallsOnDisplay {}

    /// An error, or a panic payload, whose destructor calls a durable
    /// operation.
    struct CallsOnDrop(Call);

    impl std::fmt::Debug for CallsOnDrop {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("CallsOnDrop")
        }
    }

    impl std::fmt::Display for CallsOnDrop {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("calls on drop")
        }
    }

    impl std::error::Error for CallsOnDrop {}

    impl Drop for CallsOnDrop {
        fn drop(&mut self) {
            (self.0)();
        }
    }

    /// A factory that calls a durable operation.
    struct CallingFactory(Call);

    impl PluginFactory for CallingFactory {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            (self.0)();
            Ok(None)
        }
    }

    /// Runs every plugin entry point the SDK calls during an invocation's
    /// drive: the invocation start hooks, the invocation and user-function
    /// wraps, the user-function and operation hooks, and their futures.
    async fn exercise(plugins: &Arc<InvocationPlugins>) {
        plugins.invocation_start().await;
        let ran: Ran<()> = inv(plugins)
            .instrument(step_runner(plugins, || async { Ok(()) }))
            .await;
        assert!(ran.into_parts().is_ok());
        let op = OperationInfo::for_test("s");
        plugins.dispatch_work("on_operation_start", |p| p.on_operation_start(&op));
        settle_work(plugins).await;
    }

    fn assert_context_usage_error(ctx: &crate::DurableContext, case: &str) {
        let fatal = ctx.coordinator().fatal_error();
        let message = fatal.map(|f| f.error_message).unwrap_or_default();
        assert_eq!(
            message,
            "context usage error in \"wait\": the operation was called inside plugin \
             code. Plugin code cannot call durable operations.",
            "{case}"
        );
    }

    /// Plugin code that calls a durable operation runs under the plugin
    /// guard in every hook family, and fails the execution with a context
    /// usage error. A call from `on_invocation_end` records the error too,
    /// but after the invocation's outcome was decided, so the response is
    /// unchanged.
    #[tokio::test]
    #[expect(clippy::panic, reason = "one case panics with a payload")]
    #[expect(clippy::too_many_lines, reason = "one case per hook family")]
    async fn plugin_guard_covers_every_hook_family() {
        type Case = (&'static str, fn(Call) -> Calls);
        fn work(call: Call) -> Work {
            Arc::new(move || {
                call();
                Ok(None)
            })
        }
        fn future(call: Call) -> Work {
            Arc::new(move || {
                let call = Arc::clone(&call);
                Ok(Some(Box::pin(async move {
                    call();
                    Ok(())
                })))
            })
        }
        let cases: [Case; 13] = [
            ("wrap_invocation", |c| Calls {
                wrap_invocation: Some(c),
                ..Calls::default()
            }),
            ("wrap_user_function", |c| Calls {
                wrap_user_function: Some(c),
                ..Calls::default()
            }),
            ("on_user_function_start", |c| Calls {
                user_function_start: Some(work(c)),
                ..Calls::default()
            }),
            ("on_user_function_start future", |c| Calls {
                user_function_start: Some(future(c)),
                ..Calls::default()
            }),
            ("on_user_function_end", |c| Calls {
                user_function_end: Some(work(c)),
                ..Calls::default()
            }),
            ("on_user_function_end future", |c| Calls {
                user_function_end: Some(future(c)),
                ..Calls::default()
            }),
            ("on_operation_start", |c| Calls {
                operation_start: Some(work(c)),
                ..Calls::default()
            }),
            ("on_operation_start future", |c| Calls {
                operation_start: Some(future(c)),
                ..Calls::default()
            }),
            ("on_invocation_start", |c| Calls {
                invocation_start: Some(c),
                ..Calls::default()
            }),
            ("name", |c| Calls {
                name: Some(c),
                ..Calls::default()
            }),
            ("error Display", |c| Calls {
                operation_start: Some(Arc::new(move || {
                    Err(Box::new(CallsOnDisplay(Arc::clone(&c))))
                })),
                ..Calls::default()
            }),
            ("error destructor", |c| Calls {
                operation_start: Some(Arc::new(move || Err(Box::new(CallsOnDrop(Arc::clone(&c)))))),
                ..Calls::default()
            }),
            ("panic payload destructor", |c| Calls {
                operation_start: Some(Arc::new(move || {
                    std::panic::panic_any(CallsOnDrop(Arc::clone(&c)))
                })),
                ..Calls::default()
            }),
        ];
        let calls_wait = |ctx: &crate::DurableContext| -> Call {
            let ctx = ctx.clone();
            Arc::new(move || drop(ctx.wait(Duration::from_secs(1))))
        };
        for (case, calls) in cases {
            let ctx = crate::DurableContext::test_context();
            let plugins = calls_plugins(calls(calls_wait(&ctx)));
            exercise(&plugins).await;
            assert_context_usage_error(&ctx, case);
        }

        let ctx = crate::DurableContext::test_context();
        let factory: Arc<dyn PluginFactory> = Arc::new(CallingFactory(calls_wait(&ctx)));
        drop(create_plugins(
            &[("CallingFactory", factory)],
            None,
            crate::plugin::tests_support_info,
        ));
        assert_context_usage_error(&ctx, "factory");

        let slot: Arc<Mutex<Option<crate::DurableContext>>> = Arc::default();
        let end_slot = Arc::clone(&slot);
        let plugin = crate::plugin::test_support::Hooks::default().on_end(move |_| {
            if let Some(ctx) = end_slot.lock().unwrap().as_ref() {
                drop(ctx.wait(Duration::from_secs(1)));
            }
            Box::pin(std::future::ready(Ok(())))
        });
        let handler_slot = Arc::clone(&slot);
        let (service, event) =
            crate::plugin::test_support::service(vec![plugin], 0, move |_event, ctx| {
                *handler_slot.lock().unwrap() = Some(ctx);
                async { Ok(serde_json::json!("done")) }
            });
        let response = service(event).await.expect("the invocation responds");
        assert_eq!(
            (
                response.0.get("Status").and_then(serde_json::Value::as_str),
                response.0.get("Result").and_then(serde_json::Value::as_str),
            ),
            (Some("SUCCEEDED"), Some("\"done\"")),
            "{}",
            response.0
        );
        let ctx = slot.lock().unwrap().take().expect("the handler ran");
        assert_context_usage_error(&ctx, "on_invocation_end");
    }

    /// On a task the SDK did not bless, as hook work runs on, an operation
    /// claims its ID with no error, because the ownership check runs only
    /// when the operation is awaited. So the handler's next operation gets
    /// a different ID. The same call from a hook future, which `HookRun`
    /// polls under the plugin guard, records the context usage error.
    #[tokio::test]
    async fn unguarded_plugin_task_claims_an_id_without_an_error() {
        let ctx = crate::DurableContext::test_context();
        let task_ctx = ctx.clone();
        crate::task::spawn_plugin_work(async move {
            drop(task_ctx.step(|_| async { Ok(1_u32) }));
        })
        .expect("a runtime is running")
        .await
        .expect("the task finished");
        assert!(ctx.coordinator().fatal_error().is_none());
        let fresh = crate::DurableContext::test_context();
        assert_ne!(
            ctx.claim_operation_id("step").positional(),
            fresh.claim_operation_id("step").positional()
        );

        let ctx = crate::DurableContext::test_context();
        let hook_ctx = ctx.clone();
        let plugins = calls_plugins(Calls {
            operation_start: Some(Arc::new(move || {
                let ctx = hook_ctx.clone();
                Ok(Some(Box::pin(async move {
                    drop(ctx.step(|_| async { Ok(1_u32) }));
                    Ok(())
                })))
            })),
            ..Calls::default()
        });
        let op = OperationInfo::for_test("s");
        plugins.dispatch_work("on_operation_start", |p| p.on_operation_start(&op));
        settle_work(&plugins).await;
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert!(
            fatal.error_message.contains("inside plugin code"),
            "{}",
            fatal.error_message
        );
    }

    /// In one poll the body panics `"A"`, the first plugin's wrap panics
    /// `"B"` after `next()`, and the subscriber panics `"C"` on the
    /// warning about `"B"`: `"A"` resumes after every wrap exited, and
    /// `"B"` and `"C"` are discarded.
    #[test]
    fn body_panic_wins_over_wrap_and_subscriber_panics() {
        use lambda_runtime::tracing::subscriber::layer::SubscriberExt as _;
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::PanicAfter), ("p1", Mode::Normal)], &log);
        let mut run = std::pin::pin!(step_runner(&plugins, || PanicsWhenPolled("A")));
        let subscriber = lambda_runtime::tracing::subscriber::registry().with(PanickingLayer);
        let caught = tracing::subscriber::with_default(subscriber, || {
            catch_unwind(AssertUnwindSafe(|| futures_poll_once(run.as_mut())))
        });
        let payload = caught.expect_err("the body panicked");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"A"));
        assert_eq!(
            take(&log),
            [
                "start p0:s in []",
                "enter p0:s",
                "start p1:s in [\"p0:s\"]",
                "enter p1:s",
                "exit p1:s",
                "exit p0:s",
            ]
        );
    }

    /// A body future that returns `Ready` and panics `"drop"` in its
    /// destructor resumes that panic after the wraps returned, as a body
    /// panic, with no end hook; when the output's destructor panics too,
    /// that second panic is discarded and nothing aborts.
    #[test]
    fn ready_body_destructor_panic_resumes_after_wraps() {
        #[derive(Debug)]
        struct PanicsOnDrop(Option<&'static str>);
        impl Drop for PanicsOnDrop {
            #[expect(clippy::panic, reason = "the test needs destructors that panic")]
            fn drop(&mut self) {
                if let Some(message) = self.0 {
                    std::panic::panic_any(message);
                }
            }
        }
        struct ReadyBody {
            _guard: PanicsOnDrop,
            output: Option<&'static str>,
        }
        impl Future for ReadyBody {
            type Output = Result<PanicsOnDrop, BoxError>;
            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
                Poll::Ready(Ok(PanicsOnDrop(self.output)))
            }
        }
        for output in [None, Some("output")] {
            let log = Arc::new(Mutex::new(Vec::new()));
            let plugins = plugins(&[("p0", Mode::Normal)], &log);
            let mut run = std::pin::pin!(step_runner(&plugins, move || ReadyBody {
                _guard: PanicsOnDrop(Some("drop")),
                output,
            }));
            let payload = catch_unwind(AssertUnwindSafe(|| futures_poll_once(run.as_mut())))
                .expect_err("the destructor panic resumes");
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"drop"));
            assert_eq!(
                take(&log),
                ["start p0:s in []", "enter p0:s", "exit p0:s"],
                "{output:?}"
            );
        }
    }

    /// A wrap's `next` restores the active body current before the wraps:
    /// none when none was set, the outer value otherwise.
    #[tokio::test]
    async fn next_restores_the_active_body_from_before_the_wraps() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal), ("p1", Mode::Normal)], &log);
        let frames = frames(&plugins, &["c"]);
        let unset = frames.enter(current_body);
        assert!(matches!(unset, ActiveBody::Unset), "{unset:?}");
        let outer = run_body(ActiveBody::Step("9".to_owned()), async {
            frames.enter(current_body)
        })
        .await;
        assert!(
            matches!(outer, ActiveBody::Step(ref id) if id == "9"),
            "{outer:?}"
        );
    }

    /// A body dropped before it finished is dropped inside its wraps, so its
    /// destructor sees the plugins' context, and no end hook runs.
    #[tokio::test]
    async fn a_dropped_body_is_dropped_inside_its_wraps() {
        struct OnDrop(Arc<Mutex<Vec<String>>>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0
                    .lock()
                    .unwrap()
                    .push(format!("drop in {:?}", active()));
            }
        }
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal), ("p1", Mode::Normal)], &log);
        let guard_log = Arc::clone(&log);
        let mut run = Box::pin(run_user_function(
            Some(Arc::clone(&plugins)),
            info("c"),
            UserBody::Step("c".to_owned()),
            tracing::Span::none(),
            move || async move {
                let _guard = OnDrop(guard_log);
                std::future::pending::<()>().await;
                Ok::<u32, BoxError>(0)
            },
            |_| WireError::new(None::<String>, None::<String>),
        ));
        for _ in 0..8 {
            assert!(
                inv(&plugins)
                    .enter(|| futures_poll_once(run.as_mut()))
                    .is_pending()
            );
            tokio::task::yield_now().await;
        }
        inv(&plugins).enter(|| drop(run));
        let log = take(&log);
        assert!(
            log.contains(&"drop in [\"p0:inv\", \"p1:inv\", \"p0:c\", \"p1:c\"]".to_owned()),
            "{log:?}"
        );
        assert!(!log.iter().any(|l| l.starts_with("end")), "{log:?}");
    }

    fn futures_poll_once<F: Future>(fut: Pin<&mut F>) -> Poll<F::Output> {
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        fut.poll(&mut cx)
    }

    /// A task polled inside its creator's frames sees every wrap of those
    /// frames at every poll, whichever worker thread polls it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_task_runs_inside_its_creators_frames_at_every_poll() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal), ("p1", Mode::Normal)], &log);
        let frames = frames(&plugins, &["c"]);
        let coordinator = crate::coordinator::InvocationCoordinator::new();
        let ownership = Arc::new(crate::driver::TaskOwnership::new_current());
        let task = crate::task::spawn_tracked(
            crate::task::TaskContext::new(frames, tracing::Span::none()),
            &coordinator,
            &ownership,
            async {
                let mut seen = Vec::new();
                for _ in 0..16 {
                    seen.push((std::thread::current().id(), active()));
                    tokio::task::yield_now().await;
                }
                seen
            },
        )
        .expect("a runtime is running");
        let seen = task.await.into_value();
        assert!(
            seen.iter()
                .all(|(_, a)| a == &["p0:inv", "p1:inv", "p0:c", "p1:c"]),
            "{seen:?}"
        );
        assert!(active().is_empty(), "the spawning task has no frames");
    }

    /// A task aborted before it finished drops its future inside its
    /// creator's span, so a destructor's logs carry that span.
    #[tokio::test]
    async fn an_aborted_task_drops_its_future_inside_its_span() {
        use lambda_runtime::tracing::subscriber::registry;
        struct OnDrop(Arc<Mutex<Vec<Option<tracing::span::Id>>>>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                self.0.lock().unwrap().push(tracing::Span::current().id());
            }
        }
        let _default = tracing::subscriber::set_default(registry());
        let span = tracing::info_span!("creator");
        let seen = Arc::new(Mutex::new(Vec::new()));
        let guard = OnDrop(Arc::clone(&seen));
        let coordinator = crate::coordinator::InvocationCoordinator::new();
        let ownership = Arc::new(crate::driver::TaskOwnership::new_current());
        let task = crate::task::spawn_tracked(
            crate::task::TaskContext::new(Frames::none(), span.clone()),
            &coordinator,
            &ownership,
            async move {
                let _guard = guard;
                std::future::pending::<()>().await;
            },
        )
        .expect("a runtime is running");
        tokio::task::yield_now().await;
        // Dropping the handle keeps the task running (main `b34af60`), so
        // the test aborts it, as a closed scope does.
        task.abort_handle().expect("the task started").abort();
        drop(task);
        for _ in 0..8 {
            if !seen.lock().unwrap().is_empty() {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(span.id().is_some(), "the subscriber enables the span");
        assert_eq!(*seen.lock().unwrap(), [span.id()]);
    }

    /// (a) A `.spawn()`ed operation created inside a frame keeps that
    /// frame: its handle captured the current frames at creation, so a
    /// combinator that takes it spawns its input inside them, and the task
    /// gateway enters them around every poll.
    #[tokio::test]
    async fn a_spawned_handle_keeps_its_creators_task_context() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let plugins = plugins(&[("p0", Mode::Normal)], &log);
        let ctx = crate::DurableContext::test_context();
        // The step body never finishes, so the test context writes no
        // checkpoint. Only the handle's context is under test.
        let handle = frames(&plugins, &["c"]).enter(|| {
            ctx.step(|_| async { std::future::pending::<Result<u32, BoxError>>().await })
                .name("s")
                .spawn()
        });
        let seen = handle.task_context().frames().enter(active);
        assert_eq!(seen, ["p0:inv", "p0:c"]);
    }

    /// A `tracing` layer whose `on_event` panics. The SDK's warning about
    /// a panicking wrap then panics too.
    struct PanickingLayer;

    impl<S: tracing::Subscriber> lambda_runtime::tracing::subscriber::Layer<S> for PanickingLayer {
        #[expect(clippy::panic, reason = "the test needs a subscriber that panics")]
        fn on_event(
            &self,
            _event: &tracing::Event<'_>,
            _ctx: lambda_runtime::tracing::subscriber::layer::Context<'_, S>,
        ) {
            panic!("subscriber panicked");
        }
    }
}
