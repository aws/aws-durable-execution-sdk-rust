//! Instrumentation plugins: per-invocation hooks for tracing and telemetry.
//!
//! A plugin observes one invocation of a durable function. It is built by a
//! [`PluginFactory`], which the caller registers with
//! [`OptionsBuilder::plugin`](crate::OptionsBuilder::plugin). Each invocation
//! calls every registered factory once, in registration order, and each
//! [`Plugin`] the factories return serves that one invocation. A plugin
//! receives:
//!
//! - the invocation's [`InvocationInfo`]: the request ID, the execution ARN
//!   and input, every operation the service reports, the operations the
//!   service changed since the previous invocation, and the invocation's
//!   X-Ray trace header;
//! - lifecycle hooks for the invocation, each operation, each attempt, and
//!   each change the service reports;
//! - wrap hooks that make the plugin's context current around each poll of
//!   the invocation and of each user function, also on the SDK tasks they
//!   start, and start and end hooks around each user function;
//! - a propagation hook that supplies the X-Ray header of a chained invoke.
//!
//! Every [`Plugin`] method has a no-op default, so a plugin implements only
//! the hooks it needs.
//!
//! # Example
//!
//! A factory holds state shared across invocations, here a counter. Each
//! invocation's plugin counts the operations that end in it.
//!
//! ```
//! use std::sync::Arc;
//! use std::sync::atomic::{AtomicUsize, Ordering};
//!
//! use aws_durable_execution_sdk::plugin::{
//!     HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory,
//! };
//! use aws_durable_execution_sdk::{BoxError, Options};
//!
//! #[derive(Default)]
//! struct MyFactory {
//!     ended: Arc<AtomicUsize>,
//! }
//!
//! struct Counter {
//!     ended: Arc<AtomicUsize>,
//! }
//!
//! impl Plugin for Counter {
//!     fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
//!         self.ended.fetch_add(1, Ordering::Relaxed);
//!         Ok(None)
//!     }
//! }
//!
//! impl PluginFactory for MyFactory {
//!     fn create_plugin(
//!         &self,
//!         _info: &InvocationInfo,
//!     ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
//!         Ok(Some(Box::new(Counter {
//!             ended: Arc::clone(&self.ended),
//!         })))
//!     }
//! }
//!
//! let options = Options::builder()
//!     .plugin(MyFactory::default())
//!     .build()
//!     .expect("valid options");
//! # drop(options);
//! ```
//!
//! # Hook order
//!
//! - The [`PluginFactory::create_plugin`] of every factory runs first.
//! - [`Plugin::on_invocation_start`] runs before
//!   [`Plugin::wrap_invocation`].
//! - The handler and [`Plugin::on_invocation_end`] run inside
//!   [`Plugin::wrap_invocation`].
//! - [`Plugin::on_operation_start`], [`Plugin::on_attempt_start`],
//!   [`Plugin::on_attempt_end`] and [`Plugin::on_operation_end`] run in
//!   lifecycle order: a parent's start before a child's, and a child's end
//!   before the parent's.
//! - For one user function, [`Plugin::on_user_function_start`] runs after
//!   [`Plugin::on_attempt_start`] (or the child's
//!   [`Plugin::on_operation_start`]), each plugin's start inside the wraps
//!   of the plugins before it. Every poll of the body runs inside
//!   [`Plugin::wrap_user_function`], and each plugin's
//!   [`Plugin::on_user_function_end`] runs inside those wraps after its own
//!   wrap returned, before the outcome checkpoint and the end hooks that
//!   follow it.
//! - [`Plugin::on_operation_change`] calls, then the drain of [`HookWork`],
//!   run before [`Plugin::on_invocation_end`].
//! - [`Plugin::on_invocation_end`] runs last.
//!
//! The README section
//! [Instrumentation plugins](https://github.com/aws/aws-durable-execution-sdk-rust#instrumentation-plugins)
//! gives the firing table of every operation kind, the propagation rules
//! and the isolation rules.

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime};

use tokio::task::JoinHandle;

use crate::coordinator::InvocationCoordinator;
use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
use crate::{BoxError, InvocationStatus, OperationKind, OperationStatus, WireError};

#[cfg(test)]
pub(crate) mod recorder;
mod scope;

pub(crate) use scope::{
    Frames, Ran, UserBody, never_fails, run_user_function, run_user_function_at,
};

/// The `tracing` target of every warning the SDK logs for a plugin.
pub(crate) const TARGET: &str = "aws_durable_execution_sdk::plugin";

/// The future an async hook or a wrapper returns.
pub type HookFuture<'a> = Pin<Box<dyn Future<Output = Result<(), BoxError>> + Send + 'a>>;

/// What an operation hook returns: optional work.
///
/// The SDK runs the returned future on its own task, which the
/// invocation-end drain awaits until 500 ms before the invocation deadline.
/// That task holds neither the SDK's task cleanup nor the suspension
/// decision. `Ok(None)` returns no work.
pub type HookWork = Result<Option<HookFuture<'static>>, BoxError>;

/// Builds the plugin for each invocation.
///
/// Register a factory with
/// [`OptionsBuilder::plugin`](crate::OptionsBuilder::plugin). State shared
/// across invocations belongs on the factory; the plugin it returns serves
/// one invocation only.
pub trait PluginFactory: Send + Sync + 'static {
    /// Returns the plugin for one invocation.
    ///
    /// The SDK calls this once per invocation, before any hook, on the
    /// invocation's task. `Ok(None)` skips the plugin for this invocation.
    /// `Err` skips it too, and the SDK logs the error. A panic skips it as
    /// well, and the SDK logs it. The other factories keep their order.
    ///
    /// # Errors
    ///
    /// An error skips the plugin for this invocation and is logged; it does
    /// not change the execution.
    fn create_plugin(&self, info: &InvocationInfo) -> Result<Option<Box<dyn Plugin>>, BoxError>;

    /// Factories that return the same non-empty group cannot be registered
    /// together. `None` and `Some("")` mean no group.
    ///
    /// [`OptionsBuilder::build`](crate::OptionsBuilder::build) reads the
    /// group once per factory and rejects a conflict.
    fn exclusive_group(&self) -> Option<&str> {
        None
    }
}

/// The hooks of one invocation's plugin.
///
/// One instance serves exactly one invocation. The SDK releases the
/// instance when the invocation returns. A `DurableContext` clone that
/// outlives the invocation keeps it alive until that clone drops. Every
/// method has a no-op default, so a later SDK
/// version can add a hook without breaking implementors.
///
/// # Operation, attempt and change hooks
///
/// [`on_operation_start`](Self::on_operation_start),
/// [`on_operation_end`](Self::on_operation_end),
/// [`on_attempt_start`](Self::on_attempt_start),
/// [`on_attempt_end`](Self::on_attempt_end) and
/// [`on_operation_change`](Self::on_operation_change) run synchronously in
/// lifecycle order and must not block. Each may return a future. The SDK
/// runs that future on its own task. That task holds neither the SDK's
/// task cleanup nor the suspension decision. The invocation-end drain may
/// await it until the drain deadline, 500 ms before the invocation
/// deadline.
///
/// An operation whose `.spawn()` handle the handler dropped still runs, so
/// its hooks fire as for any other operation: its end hook fires when it
/// settles, and the futures its hooks return follow the same
/// invocation-end drain. An operation the SDK stops, because its scope
/// closed, a combinator stopped it, or the invocation ended, fires no end
/// hook.
///
/// No hook future may block a thread, in `poll` or in `Drop`. The futures
/// of the operation hooks are `'static`; a plugin that needs its own state there clones an owned
/// handle, such as an `Arc` field, into the future.
///
/// # Errors and panics
///
/// An `Err` or a panic from any hook, factory or hook future is logged and
/// never changes the execution. Catching a panic relies on unwinding, so a
/// binary built with `panic = "abort"` aborts instead.
///
/// The SDK drops each error it logs. A panic in that error's destructor
/// fails the invocation, as a panic in the SDK's own code does, and the
/// service invokes the function again. When the panic happens after the
/// handler returned, the SDK resumes it after the end hooks, instead of
/// returning the response. Such a panic during the drain gives the end
/// hooks `Retrying` with its message.
///
/// Every such warning is logged on the target
/// `aws_durable_execution_sdk::plugin`, with the field
/// `plugin = "<name()> (plugins[<i>])"` and the field
/// `hook = "<method name>"`. A factory's warning has the field
/// `factory = "<factory type> (plugins[<i>])"` instead. `<i>` is the
/// 0-based registration index of the factory.
///
/// # User functions
///
/// A **user function** is one run of code that the SDK owns and polls for
/// an operation, such as a step attempt or a child context body.
/// [`wrap_invocation`](Self::wrap_invocation),
/// [`wrap_user_function`](Self::wrap_user_function),
/// [`on_user_function_start`](Self::on_user_function_start) and
/// [`on_user_function_end`](Self::on_user_function_end) observe the
/// invocation and each user function. The documentation of
/// `wrap_user_function` lists the user functions and gives the contract of
/// the four methods. Its section on durable operations in plugin code
/// applies to every method of this trait.
pub trait Plugin: Send + Sync {
    /// The name the SDK uses for this plugin in its warnings. The default
    /// is the implementing type's path.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Runs when the invocation starts. The SDK awaits the returned future
    /// of every plugin, concurrently, before the handler runs.
    ///
    /// An `Err` or a panic is logged at WARN with the plugin's name and
    /// registration index, and changes nothing else.
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        Box::pin(std::future::ready(Ok(())))
    }

    /// Runs when the invocation ends. The SDK awaits the returned future of
    /// every plugin, concurrently.
    ///
    /// 1. It runs exactly once per invocation that created the plugin,
    ///    after every checkpoint of the invocation is written. The status
    ///    of [`InvocationEndInfo::status`] is:
    ///
    ///    | The invocation | `status` | `execution_result` | `execution_error` |
    ///    | --- | --- | --- | --- |
    ///    | completed the execution | `Succeeded` | the serialized result | `None` |
    ///    | failed the execution | `Failed` | `None` | the recorded error |
    ///    | suspended | `Pending` | `None` | `None` |
    ///    | suspended after a checkpoint response without a token | `Pending` | `None` | the fatal error that decided the drive, else the handler's error, or `None` |
    ///    | failed itself after a retryable checkpoint or serdes failure | `Retrying` | `None` | the fault's type (`CheckpointUnrecoverableInvocationError` or `SerdesFailedError`) and message |
    ///    | panicked | `Retrying` | `None` | no type, and the panic's message |
    ///
    /// 2. Before this hook, the SDK waits for the work the operation hooks
    ///    returned until 500 ms before the Lambda deadline. Work still
    ///    running then is aborted, and the SDK waits for each aborted task
    ///    to end. So no hook work runs after `on_invocation_end` starts.
    /// 3. When a panic ends the invocation, this hook runs with `Retrying`
    ///    after the SDK has stopped every task it started, and the panic
    ///    then continues. No operation end hook runs for an operation the
    ///    panic interrupted. With `panic = "abort"` the process exits at the
    ///    panic, and no hook runs.
    /// 4. The SDK cuts this hook off 50 ms before the Lambda deadline. When
    ///    the context has no deadline, as under `LocalRunner`, it runs to
    ///    completion.
    /// 5. An `Err` or a panic is logged at WARN with the plugin's name and
    ///    registration index, and changes nothing else.
    /// 6. An operation that claims an operation ID here records a context
    ///    usage error, as anywhere in plugin code. No operation of the
    ///    invocation follows this hook, so the claim shifts no ID the
    ///    execution uses, and the response is the one built before it.
    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        Box::pin(std::future::ready(Ok(())))
    }

    /// Runs when a live attempt of an operation starts.
    ///
    /// For a step and a `wait_for_condition`:
    ///
    /// - an operation replayed from a terminal record fires no hook;
    /// - every live attempt fires `on_operation_start`, then
    ///   `on_attempt_start`. `on_operation_start`'s `is_replay` is `true`
    ///   from the second attempt on, and when the attempt re-enters a
    ///   `STARTED` record. The attempt's own metadata,
    ///   [`AttemptInfo::operation`]`().is_replay()`, which the attempt end
    ///   and the user-function hooks and wrap also receive, is `false`, as
    ///   in JS;
    /// - an attempt that schedules a retry fires only `on_attempt_end`;
    /// - the attempt that ends the operation fires `on_attempt_end`, then
    ///   `on_operation_end`.
    ///
    /// For a wait, a chained invoke and a callback:
    ///
    /// - a new operation fires `on_operation_start` with `is_replay`
    ///   `false` once its `START` is written. A callback fires it once the
    ///   service has also assigned its callback ID;
    /// - a replayed record that has not finished fires `on_operation_start`
    ///   with `is_replay` `true`: a `STARTED` wait, a `STARTED`, `PENDING`
    ///   or `READY` invoke, or a `STARTED` or `PENDING` callback;
    /// - a `SUCCEEDED` wait, an invoke with status `SUCCEEDED`, `FAILED`,
    ///   `TIMED_OUT`, `STOPPED` or `CANCELLED`, or a callback with status
    ///   `SUCCEEDED`, `FAILED` or `TIMED_OUT`, fires `on_operation_end` on
    ///   every replay. `is_replay` is `false` only in the first invocation
    ///   after the service completed it, the one whose
    ///   [`InvocationInfo::updated_operations`] lists it. A replayed
    ///   callback fires it when it is created, before `result()` is
    ///   awaited;
    /// - a wait that becomes `SUCCEEDED`, an invoke that becomes one of
    ///   its terminal statuses, or a callback that becomes `SUCCEEDED`,
    ///   `FAILED` or `TIMED_OUT` while its `result()` waits in the
    ///   invocation, fires `on_operation_end` with `is_replay` `false`. A
    ///   callback fires it before `result()` returns;
    /// - a wait or a callback with any other status, and a callback record
    ///   without a callback ID, fire no hook.
    ///
    /// A `wait_for_callback` reports three operations. The operation itself
    /// is a child context with sub-type `WaitForCallback`: it fires
    /// `on_operation_start` before its body, with `is_replay` `true` when
    /// its record exists, and `on_operation_end` after its outcome is
    /// written. Inside it, a callback and a submitter step fire the hooks
    /// above. When the `wait_for_callback` is named `n`, they report the
    /// names `n-callback` and `n-submitter`; their records carry no name.
    ///
    /// A child context (and so a `with_retry` block with its own record and
    /// each `attempt-N` child context of a `with_retry` block in either
    /// mode), a `map` or `parallel` container, each map item and
    /// parallel branch, and the combinators `try_join_all`, `join_all`,
    /// `select_ok` and `race` report kind `Context`. The rules below apply
    /// to each of them that writes a record, which includes a map item or
    /// parallel branch with `NestingMode::Normal`. A map item or parallel
    /// branch with `NestingMode::Flat` writes no `START`, `SUCCEED` or
    /// `FAIL` record, so these rules do not apply to it; its own rules
    /// follow the list.
    ///
    /// - a child context, a `wait_for_callback`, a `with_retry` block with
    ///   its own record, the attempts of any `with_retry` block, a `map`,
    ///   `parallel` or `parallel_mixed` container
    ///   and a map item or `parallel` or `parallel_mixed` branch without a
    ///   record fire `on_operation_start` with status `Started` and
    ///   `is_replay` `false` once their `START` is
    ///   queued, as JS does. The SDK sends that `START` only after the
    ///   hook returns, and the body or the first item runs without waiting
    ///   for its response. The outcome write waits for it;
    /// - a combinator without a record first begins the start phase of
    ///   each of its `.future()` inputs, in input order. Then it fires
    ///   `on_operation_start` with status `Started` and `is_replay`
    ///   `false` once its `START` is queued, as a JS combinator does. The
    ///   SDK sends that `START` only after the hook returns, and the inputs
    ///   run without waiting for its response. So the hook of an input that
    ///   fires when its `START` is queued comes before the combinator's,
    ///   and the hook of an input that waits for its `START` response, a
    ///   wait or a step under `AtMostOncePerRetry`, comes after it;
    /// - a record that is not terminal (`STARTED`, `PENDING` or `READY`)
    ///   fires `on_operation_start` with `is_replay` `true`, before the
    ///   body or the first item runs. A combinator fires it after its
    ///   `.future()` inputs' start phases begin, so the replay hook of a
    ///   step, wait or child-context input that resumes comes first;
    /// - a body or batch that succeeds fires `on_operation_end` with status
    ///   `Succeeded`, `is_replay` `false` and the payload the `SUCCEED`
    ///   carries as `result()`, once the `SUCCEED` is written. A result
    ///   too large for the record is written with `ReplayChildren`, and
    ///   `result()` is the payload that update carries instead: `None` for
    ///   a child context and a map item or parallel branch, which write no
    ///   payload; the JSON summary of the batch for a `map` or `parallel`
    ///   container; `""` for `try_join_all` and `join_all`; and the JSON
    ///   object `{"winner":<index>}` for `select_ok` and `race`;
    /// - a body that fails fires `on_operation_end` with status `Failed`,
    ///   `is_replay` `false` and the error, once the `FAIL` is written;
    /// - a `SUCCEEDED` or `FAILED` record fires nothing, also when the SDK
    ///   runs the body again to rebuild a result that `ReplayChildren`
    ///   kept out of the record;
    /// - after a failed outcome write, or a serdes failure that ends the
    ///   invocation, no hook fires.
    ///
    /// A map item or parallel branch names the container as its parent.
    /// Its start hook fires when the item is dispatched, and its end hook
    /// on the item's task. When the batch completes early, an item that is
    /// still running is abandoned: it and the unfinished operations inside
    /// it fire no `on_operation_end`. An item that was never dispatched
    /// fires no hook.
    ///
    /// A map item or parallel branch with `NestingMode::Flat` writes no
    /// record. It fires `on_operation_start` and `on_operation_end` with
    /// kind `Context`, sub-type `MapIteration` or `ParallelBranch`, its
    /// stable ID, the container as its parent and `is_virtual()` `true`.
    /// Its start hook fires when the item is dispatched and carries no
    /// status: `status()` is `None`, because no `START` is written, also
    /// when the item ran in an earlier invocation. Its `is_replay` is the `is_replaying()` value of that item's own
    /// context when the hook fires, so two flat items of one batch can
    /// report different values in the same invocation. The end hook carries
    /// status `Succeeded` and no `result()`, or status `Failed` and the
    /// error the batch records for the item. An abandoned flat item fires
    /// no end hook.
    ///
    /// A panic in a body, an item or a combinator input ends the
    /// invocation. No operation hook fires after it: no `on_operation_end`
    /// for the operation that panicked, nor for any operation around it.
    /// The invocation still ends with one `on_invocation_end` with status
    /// `Retrying`.
    ///
    /// A step that is terminal on replay fires nothing.
    ///
    /// The hook runs synchronously, on the task that runs the operation.
    /// For `.spawn()` and `.spawn_batch()` of a step, a wait, a child
    /// context, a `map`, a `parallel`, a `parallel_mixed`, a
    /// `create_callback`, a `wait_for_callback` and a `with_retry`, the
    /// start step runs on the caller's task before the call returns (see
    /// [`StepBuilder::spawn`](crate::builders::StepBuilder::spawn)). So when the operation's
    /// JS order is "queue `START`, then hook" (a step under
    /// `AtLeastOncePerRetry` and every child-backed operation above), or
    /// when its record resumes work and fires the replay start hook, this
    /// hook runs on the caller's task before `.spawn()` returns. An
    /// operation that fires it after the `START` response fires it on the
    /// spawned task. `on_attempt_start` always runs on the operation's own
    /// task, before the body.
    ///
    /// It can return work: the SDK
    /// runs that future on its own task, and waits for it before
    /// `on_invocation_end` until 500 ms before the invocation deadline.
    ///
    /// # Errors
    ///
    /// An error or a panic is logged at WARN with the plugin's name and
    /// registration index, and does not change the execution.
    fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
        Ok(None)
    }

    /// Runs when an operation ends, after the checkpoint that records its
    /// outcome. The info carries the same end timestamp as the
    /// `on_attempt_end` before it.
    ///
    /// For a step and a `wait_for_condition`:
    ///
    /// - an operation replayed from a terminal record fires no hook;
    /// - every live attempt fires `on_operation_start`, then
    ///   `on_attempt_start`. `on_operation_start`'s `is_replay` is `true`
    ///   from the second attempt on, and when the attempt re-enters a
    ///   `STARTED` record. The attempt's own metadata,
    ///   [`AttemptInfo::operation`]`().is_replay()`, which the attempt end
    ///   and the user-function hooks and wrap also receive, is `false`, as
    ///   in JS;
    /// - an attempt that schedules a retry fires only `on_attempt_end`;
    /// - the attempt that ends the operation fires `on_attempt_end`, then
    ///   `on_operation_end`.
    ///
    /// For a wait, a chained invoke and a callback:
    ///
    /// - a new operation fires `on_operation_start` with `is_replay`
    ///   `false` once its `START` is written. A callback fires it once the
    ///   service has also assigned its callback ID;
    /// - a replayed record that has not finished fires `on_operation_start`
    ///   with `is_replay` `true`: a `STARTED` wait, a `STARTED`, `PENDING`
    ///   or `READY` invoke, or a `STARTED` or `PENDING` callback;
    /// - a `SUCCEEDED` wait, an invoke with status `SUCCEEDED`, `FAILED`,
    ///   `TIMED_OUT`, `STOPPED` or `CANCELLED`, or a callback with status
    ///   `SUCCEEDED`, `FAILED` or `TIMED_OUT`, fires `on_operation_end` on
    ///   every replay. `is_replay` is `false` only in the first invocation
    ///   after the service completed it, the one whose
    ///   [`InvocationInfo::updated_operations`] lists it. A replayed
    ///   callback fires it when it is created, before `result()` is
    ///   awaited;
    /// - a wait that becomes `SUCCEEDED`, an invoke that becomes one of
    ///   its terminal statuses, or a callback that becomes `SUCCEEDED`,
    ///   `FAILED` or `TIMED_OUT` while its `result()` waits in the
    ///   invocation, fires `on_operation_end` with `is_replay` `false`. A
    ///   callback fires it before `result()` returns;
    /// - a wait or a callback with any other status, and a callback record
    ///   without a callback ID, fire no hook.
    ///
    /// A `wait_for_callback` reports three operations. The operation itself
    /// is a child context with sub-type `WaitForCallback`: it fires
    /// `on_operation_start` before its body, with `is_replay` `true` when
    /// its record exists, and `on_operation_end` after its outcome is
    /// written. Inside it, a callback and a submitter step fire the hooks
    /// above. When the `wait_for_callback` is named `n`, they report the
    /// names `n-callback` and `n-submitter`; their records carry no name.
    ///
    /// A child context (and so a `with_retry` block with its own record and
    /// each `attempt-N` child context of a `with_retry` block in either
    /// mode), a `map` or `parallel` container, each map item and
    /// parallel branch, and the combinators `try_join_all`, `join_all`,
    /// `select_ok` and `race` report kind `Context`. The rules below apply
    /// to each of them that writes a record, which includes a map item or
    /// parallel branch with `NestingMode::Normal`. A map item or parallel
    /// branch with `NestingMode::Flat` writes no `START`, `SUCCEED` or
    /// `FAIL` record, so these rules do not apply to it; its own rules
    /// follow the list.
    ///
    /// - a child context, a `wait_for_callback`, a `with_retry` block with
    ///   its own record, the attempts of any `with_retry` block, a `map`,
    ///   `parallel` or `parallel_mixed` container
    ///   and a map item or `parallel` or `parallel_mixed` branch without a
    ///   record fire `on_operation_start` with status `Started` and
    ///   `is_replay` `false` once their `START` is queued, as JS does. The
    ///   SDK sends that `START` only after the hook returns, and the body or
    ///   the first item runs without waiting for its response;
    /// - a combinator without a record first begins the start phase of
    ///   each of its `.future()` inputs, in input order. Then it fires
    ///   `on_operation_start` with status `Started` and `is_replay`
    ///   `false` once its `START` is queued, as a JS combinator does. The
    ///   SDK sends that `START` only after the hook returns, and the inputs
    ///   run without waiting for its response. So the hook of an input that
    ///   fires when its `START` is queued comes before the combinator's,
    ///   and the hook of an input that waits for its `START` response, a
    ///   wait or a step under `AtMostOncePerRetry`, comes after it;
    /// - a record that is not terminal (`STARTED`, `PENDING` or `READY`)
    ///   fires `on_operation_start` with `is_replay` `true`, before the
    ///   body or the first item runs. A combinator fires it after its
    ///   `.future()` inputs' start phases begin, so the replay hook of a
    ///   step, wait or child-context input that resumes comes first;
    /// - a body or batch that succeeds fires `on_operation_end` with status
    ///   `Succeeded`, `is_replay` `false` and the payload the `SUCCEED`
    ///   carries as `result()`, once the `SUCCEED` is written. The
    ///   `SUCCEED` of an operation whose `START` was queued is written only
    ///   after that `START` is. A result
    ///   too large for the record is written with `ReplayChildren`, and
    ///   `result()` is the payload that update carries instead: `None` for
    ///   a child context and a map item or parallel branch, which write no
    ///   payload; the JSON summary of the batch for a `map` or `parallel`
    ///   container; `""` for `try_join_all` and `join_all`; and the JSON
    ///   object `{"winner":<index>}` for `select_ok` and `race`;
    /// - a body that fails fires `on_operation_end` with status `Failed`,
    ///   `is_replay` `false` and the error, once the `FAIL` is written;
    /// - a `SUCCEEDED` or `FAILED` record fires nothing, also when the SDK
    ///   runs the body again to rebuild a result that `ReplayChildren`
    ///   kept out of the record;
    /// - after a failed outcome write, or a serdes failure that ends the
    ///   invocation, no hook fires.
    ///
    /// A map item or parallel branch names the container as its parent.
    /// Its start hook fires when the item is dispatched, and its end hook
    /// on the item's task. When the batch completes early, an item that is
    /// still running is abandoned: it and the unfinished operations inside
    /// it fire no `on_operation_end`. An item that was never dispatched
    /// fires no hook.
    ///
    /// A map item or parallel branch with `NestingMode::Flat` writes no
    /// record. It fires `on_operation_start` and `on_operation_end` with
    /// kind `Context`, sub-type `MapIteration` or `ParallelBranch`, its
    /// stable ID, the container as its parent and `is_virtual()` `true`.
    /// Its start hook fires when the item is dispatched and carries no
    /// status: `status()` is `None`, because no `START` is written, also
    /// when the item ran in an earlier invocation. Its `is_replay` is the `is_replaying()` value of that item's own
    /// context when the hook fires, so two flat items of one batch can
    /// report different values in the same invocation. The end hook carries
    /// status `Succeeded` and no `result()`, or status `Failed` and the
    /// error the batch records for the item. An abandoned flat item fires
    /// no end hook.
    ///
    /// A panic in a body, an item or a combinator input ends the
    /// invocation. No operation hook fires after it: no `on_operation_end`
    /// for the operation that panicked, nor for any operation around it.
    /// The invocation still ends with one `on_invocation_end` with status
    /// `Retrying`.
    ///
    /// A step that is terminal on replay fires nothing.
    ///
    /// The hook runs synchronously, on the task that runs the operation.
    /// It can return work: the SDK
    /// runs that future on its own task, and waits for it before
    /// `on_invocation_end` until 500 ms before the invocation deadline.
    ///
    /// # Errors
    ///
    /// An error or a panic is logged at WARN with the plugin's name and
    /// registration index, and does not change the execution.
    fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
        Ok(None)
    }

    /// Runs when a step attempt or a condition check starts, right after
    /// `on_operation_start`. The operation's start timestamp is the time
    /// the attempt started.
    ///
    /// For a step and a `wait_for_condition`:
    ///
    /// - an operation replayed from a terminal record fires no hook;
    /// - every live attempt fires `on_operation_start`, then
    ///   `on_attempt_start`. `on_operation_start`'s `is_replay` is `true`
    ///   from the second attempt on, and when the attempt re-enters a
    ///   `STARTED` record. The attempt's own metadata,
    ///   [`AttemptInfo::operation`]`().is_replay()`, which the attempt end
    ///   and the user-function hooks and wrap also receive, is `false`, as
    ///   in JS;
    /// - an attempt that schedules a retry fires only `on_attempt_end`;
    /// - the attempt that ends the operation fires `on_attempt_end`, then
    ///   `on_operation_end`.
    ///
    /// The hook runs synchronously, on the task that runs the operation.
    /// It can return work: the SDK
    /// runs that future on its own task, and waits for it before
    /// `on_invocation_end` until 500 ms before the invocation deadline.
    ///
    /// # Errors
    ///
    /// An error or a panic is logged at WARN with the plugin's name and
    /// registration index, and does not change the execution.
    fn on_attempt_start(&self, _info: &AttemptInfo) -> HookWork {
        Ok(None)
    }

    /// Runs when a step attempt or a condition check ends, after the
    /// checkpoint that records its outcome.
    ///
    /// The operation's end timestamp is the recorded end time when the
    /// checkpoint response carries one. Otherwise it is the local time the
    /// SDK observed right after the checkpoint call returned. A condition
    /// check that continues ends its attempt `Succeeded`.
    ///
    /// For a step and a `wait_for_condition`:
    ///
    /// - an operation replayed from a terminal record fires no hook;
    /// - every live attempt fires `on_operation_start`, then
    ///   `on_attempt_start`. `on_operation_start`'s `is_replay` is `true`
    ///   from the second attempt on, and when the attempt re-enters a
    ///   `STARTED` record. The attempt's own metadata,
    ///   [`AttemptInfo::operation`]`().is_replay()`, which the attempt end
    ///   and the user-function hooks and wrap also receive, is `false`, as
    ///   in JS;
    /// - an attempt that schedules a retry fires only `on_attempt_end`;
    /// - the attempt that ends the operation fires `on_attempt_end`, then
    ///   `on_operation_end`.
    ///
    /// The hook runs synchronously, on the task that runs the operation.
    /// It can return work: the SDK
    /// runs that future on its own task, and waits for it before
    /// `on_invocation_end` until 500 ms before the invocation deadline.
    ///
    /// # Errors
    ///
    /// An error or a panic is logged at WARN with the plugin's name and
    /// registration index, and does not change the execution.
    fn on_attempt_end(&self, _info: &AttemptEndInfo) -> HookWork {
        Ok(None)
    }

    /// Runs after a checkpoint response that changed the status of at
    /// least one operation.
    ///
    /// The SDK calls the hook after it merged the response into its state,
    /// once per plugin, in registration order, and once per such response,
    /// in response order. A refresh checkpoint without updates is a
    /// checkpoint response too. A paginated response and the state pages
    /// the SDK reads after it count as one response. When reading those
    /// pages fails, the changes already merged are still reported.
    ///
    /// An operation counts as changed when the SDK held no record of it,
    /// or held one with another status. Other fields do not count. The
    /// `EXECUTION` operation is listed like any other. The state the
    /// invocation starts with is not a change.
    ///
    /// [`OperationChangeInfo::updated_operations`] lists the changed
    /// operations as of that response. [`OperationChangeInfo::operations`]
    /// is the full map when the hook is called, including the `EXECUTION`
    /// operation, and contains every updated ID. Every [`OperationInfo`]
    /// has `is_replay() == false`.
    ///
    /// Every call finishes before
    /// [`on_invocation_end`](Self::on_invocation_end), including the call
    /// for a write whose caller dropped it and for the end-of-invocation
    /// flush. The change that completes a waiting operation comes before
    /// that operation's [`on_operation_end`](Self::on_operation_end).
    ///
    /// Against [`on_operation_start`](Self::on_operation_start): a wait,
    /// `create_callback`, an invoke and a step under
    /// `StepSemantics::AtMostOncePerRetry` call `on_operation_start` after
    /// their `START` response, so the change that lists them `STARTED`
    /// comes first. A default step, `wait_for_condition` and every
    /// child-backed operation that writes a record, including a map, a
    /// parallel and a combinator, reserve `START` in submission order,
    /// call `on_operation_start` synchronously, and only then release
    /// `START` to the checkpoint writer, without awaiting its response. So
    /// their `STARTED` change comes after their start hook, on every
    /// runtime flavor.
    ///
    /// The hook runs synchronously. A plugin must not assume which task
    /// calls it: an operation's task, a task that sent a refresh, or the
    /// invocation wrapper, not necessarily the task that submitted the
    /// write. One task delivers at a time, and an operation whose change
    /// another task is delivering waits for that delivery before it calls
    /// its next hook, on every runtime flavor. So a hook that blocks delays
    /// those operations. It never runs while the SDK holds a checkpoint
    /// lock. It can return work: the SDK runs that future on its own task,
    /// and waits for it before `on_invocation_end` until 500 ms before the
    /// invocation deadline.
    ///
    /// # Errors
    ///
    /// An error or a panic is logged at WARN with the plugin's name and
    /// registration index, and does not change the execution.
    fn on_operation_change(&self, _info: &OperationChangeInfo) -> HookWork {
        Ok(None)
    }

    /// Makes this plugin's context current around each poll of the
    /// invocation.
    ///
    /// The SDK awaits [`on_invocation_start`](Self::on_invocation_start)
    /// first. Then every poll of the rest of the invocation runs inside the
    /// `wrap_invocation` of every plugin, the first registered plugin's
    /// outermost: the handler, every SDK task that runs an operation, the
    /// end-of-invocation flush, the drain of the hook work and
    /// [`on_invocation_end`](Self::on_invocation_end). A wrap that calls
    /// `next()` inside its context, for example with a `tracing` span's
    /// `in_scope`, makes the context current in all of that code. The
    /// invocation is no user function, so no start or end hook fires for it;
    /// the wraps of each user function run inside these wraps, as
    /// [`wrap_user_function`](Self::wrap_user_function) describes.
    ///
    /// `next()` follows the rules of `wrap_user_function`: it runs the rest
    /// of the chain once, the SDK calls it right after the wrap when the
    /// wrap returns or panics without calling it, and a second call does
    /// nothing. A wrap runs on whichever thread polls, as often as the SDK
    /// polls, so it must be cheap and must not block. This wrap's `next()`
    /// runs every hook and wrap of the invocation, this plugin's included,
    /// so do not hold across it a non-reentrant lock that any hook or wrap
    /// of this plugin can take.
    ///
    /// The SDK drops the rest of the invocation inside these wraps. When the
    /// handler panics, or another panic fails the invocation, the panic
    /// continues after `on_invocation_end` ran and after every plugin was
    /// dropped. Code of this wrap after `next()` in the last poll runs after
    /// the SDK decided the invocation's outcome: a durable operation it
    /// calls is recorded as a context usage error and changes nothing, as
    /// the section on durable operations of `wrap_user_function` says.
    ///
    /// A panic in this wrap is logged at WARN, with the field
    /// `hook = "wrap_invocation"`, and changes nothing else.
    fn wrap_invocation(&self, _info: &InvocationInfo, next: &mut dyn FnMut()) {
        next();
    }

    /// Makes this plugin's context current around each poll of a user
    /// function.
    ///
    /// A **user function** is one run of code that the SDK owns and polls
    /// for an operation:
    ///
    /// - a step attempt, including the submitter step of a
    ///   `wait_for_callback` named `<name>-submitter`;
    /// - a `wait_for_condition` check;
    /// - a `run_in_child_context` body, the block of a `with_retry` with its
    ///   own record, and the body of a `wait_for_callback`. A `with_retry`
    ///   block with `without_child_context` has no record and is not a user
    ///   function;
    /// - each item of a `map` and each branch of a `parallel` or
    ///   `parallel_mixed`, and the container of each of those three, whose
    ///   user function produces the batch from the items;
    /// - the join of each combinator: `try_join_all`, `join_all`,
    ///   `select_ok` and `race`.
    ///
    /// # Order
    ///
    /// For one user function and two plugins `p0` and `p1`, registered in
    /// that order, all of the following runs on the task that polls the
    /// user function.
    ///
    /// 1. [`on_operation_start`](Self::on_operation_start) and, for a step
    ///    attempt or a check, [`on_attempt_start`](Self::on_attempt_start)
    ///    run first, as for any operation.
    /// 2. At the first poll, `p0.on_user_function_start` runs. When it
    ///    started `p0`, `p0.wrap_user_function` runs, and inside its
    ///    `next()` the same happens for `p1`. Inside `p1`'s `next()` the SDK
    ///    polls the user function. A start hook whose synchronous call
    ///    returns `Err` or panics did not start its plugin: that plugin gets
    ///    no wrap and no end hook for this user function, and the SDK logs a
    ///    warning. A start hook that returns `Ok(Some(future))` started its
    ///    plugin, and a later failure of that future is only logged.
    /// 3. At every poll, the wraps of the started plugins run, then the SDK
    ///    enters the operation's `tracing` span, then it polls the user
    ///    function. The SDK calls the user's closure at the first poll,
    ///    inside the wraps and the span. Later polls run no start hook.
    /// 4. In the poll in which the user function returns, the SDK derives
    ///    the error it records, once, when the user function failed. The
    ///    wraps then unwind with no await between them: `p1`'s wrap returns,
    ///    `p1.on_user_function_end` runs inside `p0`'s wrap, `p0`'s wrap
    ///    returns, and `p0.on_user_function_end` runs. So the start hooks
    ///    run in registration order and the end hooks in reverse.
    /// 5. Then the SDK serializes the value, writes the outcome checkpoint,
    ///    and fires [`on_attempt_end`](Self::on_attempt_end) for a step
    ///    attempt or a check, and
    ///    [`on_operation_end`](Self::on_operation_end). So an end hook runs
    ///    before the outcome is recorded.
    ///
    /// # Nesting and `next()`
    ///
    /// The wraps nest by user function. The invocation's wraps
    /// ([`wrap_invocation`](Self::wrap_invocation)) are outermost, then the
    /// wraps of each enclosing user function, outermost first, then the
    /// wraps of the user function itself. Within one of them, the first
    /// registered plugin's wrap is outermost. Each poll runs each wrap once.
    ///
    /// `next()` runs the rest of the chain once. When a wrap returns or
    /// panics without calling `next()`, the SDK calls it right after that
    /// wrap exited, so the rest of the chain, the later plugins' start and
    /// end hooks included, runs outside that wrap. A second call to
    /// `next()` does nothing. So the user function always runs, and its
    /// value is the operation's result: a plugin cannot skip a user
    /// function or replace its result. `next` cannot be stored, sent to
    /// another thread or awaited; the examples at the end show each case
    /// failing to compile.
    ///
    /// The wraps also run around every poll of an SDK task that runs an
    /// operation started inside the user function: an operation started
    /// with `.spawn()`, a map item, a parallel branch, a combinator input.
    /// An operation started with `.spawn()`, and a `.future()` that a
    /// combinator starts, keep the wraps and the `tracing` span of the code
    /// that made them. An operation awaited directly runs in the wraps and
    /// the span of the code that awaits it.
    ///
    /// Inside a wrap, the current `tracing` span is the poller's. The SDK
    /// enters the operation's span inside the innermost `next()`, so the
    /// user function runs in it, and so does the synchronous code of a map
    /// item's closure. The start hooks, and the formatting of the error
    /// that the SDK records, run outside the operation's span.
    ///
    /// A wrap runs on whichever thread polls, as often as the SDK polls, and
    /// the synchronous part of each start and end hook runs on the
    /// operation's path. All of them must be cheap and must not block; put
    /// slow work in the future a hook returns.
    ///
    /// `next()` runs the later plugins' hooks and wraps, and the hooks and
    /// wraps of every user function polled inside this one, this plugin's
    /// own included. So do not hold across `next()` a non-reentrant lock,
    /// such as a `std::sync::Mutex` guard or a `RefCell` borrow, that any
    /// hook or wrap of this plugin can take: it deadlocks or panics. Copy
    /// what the wrap needs out of the lock, release it, then call `next()`,
    /// as the first example does.
    ///
    /// # End hooks and hook work
    ///
    /// [`on_user_function_end`](Self::on_user_function_end) receives
    /// `Ok(())` when the user function returned `Ok`, and otherwise the
    /// [`WireError`] the SDK records for its error. It never receives the
    /// value.
    ///
    /// The SDK calls each start and end hook synchronously, catching a
    /// panic, as it calls `on_operation_start`. A future a hook returns runs
    /// on its own task. The SDK never awaits it on the operation's path, and
    /// it does not keep the invocation from suspending. The SDK waits for it
    /// only in the drain before `on_invocation_end`, until 500 ms before the
    /// invocation deadline, and aborts it then. Hook futures complete in no
    /// defined order.
    ///
    /// # When no hook fires
    ///
    /// A user function that panics gets no end hook; its panic continues
    /// after every wrap returned. A user function that the SDK drops before
    /// it returns gets no end hook either: its scope closed, a combinator
    /// stopped it as a loser, its batch completed early, or the invocation
    /// ended. The SDK drops it inside its wraps, so a destructor in it runs
    /// in the plugins' context. A container whose item `START` write fails
    /// fails the invocation and never returns, so it gets no end hook.
    ///
    /// These run no user function and fire no start hook, no wrap and no
    /// end hook:
    ///
    /// - an operation replayed from a terminal record;
    /// - a child context run again to rebuild a result that its record did
    ///   not store (`ReplayChildren`);
    /// - a step attempt interrupted under
    ///   [`StepSemantics::AtMostOncePerRetry`](crate::StepSemantics::AtMostOncePerRetry),
    ///   which the SDK records as failed;
    /// - a step waiting for its retry delay;
    /// - the `attempt-N` child contexts of a `with_retry` block, in either
    ///   mode. A block with its own record is the user function, and its
    ///   attempts run inside its wraps. A block with `without_child_context`
    ///   has no user function, wrap or scope of its own, so its attempts run
    ///   inside the caller's wraps. The attempts still fire their operation
    ///   hooks;
    /// - the join of a combinator that fails because an input was not
    ///   claimed, after its own `START` and `on_operation_start`.
    ///
    /// # Durable operations in plugin code
    ///
    /// A plugin receives no [`DurableContext`](crate::DurableContext), so it
    /// can reach an operation only through a value the handler gave it. When
    /// plugin code that the SDK runs creates a durable operation, the
    /// operation's builder is built as usual and takes its operation ID,
    /// and the SDK records a context usage error that names plugin code, as
    /// it does for an operation created in a step body. Plugin code is
    /// every hook, wrap and factory, [`name`](Self::name), each poll and
    /// drop of a future a hook returns, the `Display` and destructor of an
    /// error a hook returns, the destructor of a plugin's panic payload, and
    /// a `tracing` subscriber that the SDK calls for a warning about a
    /// plugin. Like the step-body check, the check runs where an operation
    /// takes its ID. Most builders take it when the operation is created; a
    /// combinator takes it when it is first polled or spawned; a `with_retry`
    /// block with its own record takes it when the builder becomes its
    /// future; and a block with `without_child_context` takes none, while
    /// each of its attempts and waits takes the caller's next ID when it
    /// starts. The check does
    /// not follow a builder or future that other code created and that took
    /// its ID, a later poll of it, a handle returned by `.spawn()`, or code
    /// on a task or thread that the plugin starts itself.
    ///
    /// A context usage error recorded before the SDK decides the
    /// invocation's outcome fails the execution. One recorded later, by a
    /// hook's future during the drain, by `on_invocation_end`, or by
    /// [`wrap_invocation`](Self::wrap_invocation)'s code after `next()` in
    /// the last poll, is recorded but changes neither the response nor
    /// whether a later panic fails the invocation.
    ///
    /// [`PluginFactory::exclusive_group`] is not plugin code here:
    /// [`OptionsBuilder::build`](crate::OptionsBuilder::build) calls it
    /// once, on its caller's stack, before any invocation, outside the
    /// check.
    ///
    /// A plugin destructor that runs after the SDK returned the response,
    /// for example of a value the plugin kept beyond the invocation, is
    /// outside every guarantee here.
    ///
    /// # The operation metadata and its replay flag
    ///
    /// The start hook, the wrap and the end hook of one user function
    /// receive the same `info`.
    ///
    /// - For a step attempt or a check, it is the value
    ///   [`AttemptInfo::operation`] returns to `on_attempt_start`: its
    ///   status is `Started`, [`attempt()`](OperationInfo::attempt) is the
    ///   attempt number, and [`is_replay()`](OperationInfo::is_replay) is
    ///   `false`. `on_operation_start` can report `is_replay` `true` for the
    ///   same attempt, from the second attempt on or on a `STARTED` record.
    /// - For a child context body, a map item, a parallel branch or a
    ///   container, it is `on_operation_start`'s value with `is_replay` set
    ///   to whether the child context replays when the body starts. So a
    ///   resumed child with no record inside it reports `true` to
    ///   `on_operation_start` and `false` here. A `NestingMode::Flat` item
    ///   or branch has [`is_virtual()`](OperationInfo::is_virtual) `true`.
    /// - For a combinator's join, it is `on_operation_start`'s value
    ///   unchanged: `is_replay` is `false` when the combinator is new and
    ///   `true` when it resumes a `STARTED` record of an earlier invocation.
    ///
    /// # Tasks the handler starts
    ///
    /// The wraps run around the handler's task and the SDK tasks that run
    /// operations. A task the SDK starts for other work, such as the future
    /// a hook returns, and a task the handler starts itself, with
    /// `tokio::spawn` or a thread, carry no plugin context. To carry the
    /// context into the handler's own task, wrap the task's future in a
    /// carrier: `tracing`'s `.instrument(tracing::Span::current())`,
    /// OpenTelemetry's `.with_current_context()`, or `LocalKey::scope` for
    /// a task-local value.
    ///
    /// # Examples
    ///
    /// A plugin in the shape of OpenTelemetry stores a context for each user
    /// function in its start hook and attaches it around `next()`. The
    /// `Context` type below stands in for `opentelemetry::Context`, so the
    /// example has no dependency; with OpenTelemetry, `Context::attach` and
    /// its guard do the same. The wrap clones the context out of the lock
    /// and releases the lock before it calls `next()`.
    ///
    /// ```
    /// use std::cell::RefCell;
    /// use std::collections::HashMap;
    /// use std::sync::Mutex;
    ///
    /// use aws_durable_execution_sdk::WireError;
    /// use aws_durable_execution_sdk::plugin::{HookWork, OperationInfo, Plugin};
    ///
    /// /// Stands in for `opentelemetry::Context`.
    /// #[derive(Clone)]
    /// struct Context {
    ///     operation_id: String,
    /// }
    ///
    /// thread_local! {
    ///     static CURRENT: RefCell<Option<Context>> = const { RefCell::new(None) };
    /// }
    ///
    /// /// Stands in for `opentelemetry::ContextGuard`: restores the previous
    /// /// context when dropped.
    /// struct ContextGuard {
    ///     previous: Option<Context>,
    /// }
    ///
    /// impl Context {
    ///     /// Stands in for `opentelemetry::Context::attach`.
    ///     fn attach(self) -> ContextGuard {
    ///         let previous = CURRENT.with(|current| current.replace(Some(self)));
    ///         ContextGuard { previous }
    ///     }
    /// }
    ///
    /// impl Drop for ContextGuard {
    ///     fn drop(&mut self) {
    ///         CURRENT.with(|current| *current.borrow_mut() = self.previous.take());
    ///     }
    /// }
    ///
    /// #[derive(Default)]
    /// struct Otel {
    ///     contexts: Mutex<HashMap<String, Context>>,
    /// }
    ///
    /// impl Plugin for Otel {
    ///     fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
    ///         let context = Context {
    ///             operation_id: info.id().to_owned(),
    ///         };
    ///         self.contexts
    ///             .lock()
    ///             .map_err(|_| "context map poisoned")?
    ///             .insert(info.id().to_owned(), context);
    ///         Ok(None)
    ///     }
    ///
    ///     fn wrap_user_function(&self, info: &OperationInfo, next: &mut dyn FnMut()) {
    ///         // Clone under the lock; the guard drops at the end of the statement.
    ///         let context = self
    ///             .contexts
    ///             .lock()
    ///             .ok()
    ///             .and_then(|contexts| contexts.get(info.id()).cloned());
    ///         let _attached = context.map(Context::attach);
    ///         next();
    ///     }
    ///
    ///     fn on_user_function_end(
    ///         &self,
    ///         info: &OperationInfo,
    ///         _outcome: Result<(), &WireError>,
    ///     ) -> HookWork {
    ///         self.contexts
    ///             .lock()
    ///             .map_err(|_| "context map poisoned")?
    ///             .remove(info.id());
    ///         Ok(None)
    ///     }
    /// }
    ///
    /// // The guard makes the context current until it drops.
    /// {
    ///     let _attached = Context { operation_id: "op".to_owned() }.attach();
    ///     let current = CURRENT.with(|current| current.borrow().clone());
    ///     assert_eq!(current.map(|c| c.operation_id), Some("op".to_owned()));
    /// }
    /// assert!(CURRENT.with(|current| current.borrow().is_none()));
    /// # drop(Otel::default());
    /// ```
    ///
    /// A `tracing` plugin enters a span for the user function around
    /// `next()`:
    ///
    /// ```
    /// use aws_durable_execution_sdk::plugin::{OperationInfo, Plugin};
    ///
    /// struct Traced;
    ///
    /// impl Traced {
    ///     fn span_for(&self, info: &OperationInfo) -> tracing::Span {
    ///         tracing::info_span!(
    ///             "user_function",
    ///             operation_id = info.id(),
    ///             attempt = info.attempt(),
    ///         )
    ///     }
    /// }
    ///
    /// impl Plugin for Traced {
    ///     fn wrap_user_function(&self, info: &OperationInfo, next: &mut dyn FnMut()) {
    ///         self.span_for(info).in_scope(next);
    ///     }
    /// }
    /// # drop(Traced);
    /// ```
    ///
    /// A plugin that records the start and end of each parallel branch,
    /// with the parallel operation as the parent, as the conformance test
    /// for parallel branch hooks requires. Its hooks record in their
    /// synchronous call and return no work.
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    ///
    /// use aws_durable_execution_sdk as durable;
    /// use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
    /// use durable::{BoxError, Branch, DurableContext, Options, WireError};
    ///
    /// /// The sub-type of a parallel branch's child context.
    /// const PARALLEL_BRANCH: &str = "ParallelBranch";
    ///
    /// struct Recorder {
    ///     records: Arc<Mutex<Vec<String>>>,
    /// }
    ///
    /// impl Recorder {
    ///     fn record(&self, record: String) {
    ///         if let Ok(mut records) = self.records.lock() {
    ///             records.push(record);
    ///         }
    ///     }
    /// }
    ///
    /// impl Plugin for Recorder {
    ///     fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
    ///         if info.sub_type() == Some(PARALLEL_BRANCH) {
    ///             let parent = info.parent_id().unwrap_or("NONE");
    ///             self.record(format!("CONFPLUGIN fn-start op={} parent={parent}", info.id()));
    ///         }
    ///         Ok(None)
    ///     }
    ///
    ///     fn on_user_function_end(
    ///         &self,
    ///         info: &OperationInfo,
    ///         outcome: Result<(), &WireError>,
    ///     ) -> HookWork {
    ///         if info.sub_type() == Some(PARALLEL_BRANCH) {
    ///             let parent = info.parent_id().unwrap_or("NONE");
    ///             let outcome = if outcome.is_ok() { "SUCCEEDED" } else { "FAILED" };
    ///             self.record(format!(
    ///                 "CONFPLUGIN fn-end op={} parent={parent} outcome={outcome}",
    ///                 info.id(),
    ///             ));
    ///         }
    ///         Ok(None)
    ///     }
    /// }
    ///
    /// #[derive(Default)]
    /// struct Factory {
    ///     records: Arc<Mutex<Vec<String>>>,
    /// }
    ///
    /// impl PluginFactory for Factory {
    ///     fn create_plugin(
    ///         &self,
    ///         _info: &InvocationInfo,
    ///     ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
    ///         Ok(Some(Box::new(Recorder {
    ///             records: Arc::clone(&self.records),
    ///         })))
    ///     }
    /// }
    ///
    /// /// Two branches that run one at a time. The records are, in order:
    /// /// fn-start and fn-end of branch 0, then of branch 1.
    /// async fn handler(
    ///     _event: serde_json::Value,
    ///     ctx: DurableContext,
    /// ) -> Result<Vec<String>, BoxError> {
    ///     let branches: Vec<Branch<String>> = vec![
    ///         Branch::new("0", |_: DurableContext| async { Ok("task-1".to_owned()) }),
    ///         Branch::new("1", |_: DurableContext| async { Ok("task-2".to_owned()) }),
    ///     ];
    ///     Ok(ctx
    ///         .parallel(branches)
    ///         .name("parallel")
    ///         .max_concurrency(1)
    ///         .await?)
    /// }
    ///
    /// let options = Options::builder()
    ///     .plugin(Factory::default())
    ///     .build()
    ///     .expect("valid options");
    /// let service = durable::wrap(handler, options);
    /// # drop(service);
    /// ```
    ///
    /// A wrap cannot keep `next` after it returns:
    ///
    /// ```compile_fail,E0521
    /// # // next_cannot_be_stored
    /// use std::cell::RefCell;
    ///
    /// use aws_durable_execution_sdk::plugin::{OperationInfo, Plugin};
    ///
    /// thread_local! {
    ///     static KEPT: RefCell<Option<&'static mut dyn FnMut()>> = RefCell::new(None);
    /// }
    ///
    /// struct Keeps;
    ///
    /// impl Plugin for Keeps {
    ///     fn wrap_user_function(&self, _info: &OperationInfo, next: &mut dyn FnMut()) {
    ///         KEPT.with(|kept| *kept.borrow_mut() = Some(next));
    ///     }
    /// }
    /// ```
    ///
    /// A wrap cannot run `next` on another thread:
    ///
    /// ```compile_fail,E0277
    /// # // next_cannot_be_sent
    /// use aws_durable_execution_sdk::plugin::{OperationInfo, Plugin};
    ///
    /// struct Moves;
    ///
    /// impl Plugin for Moves {
    ///     fn wrap_user_function(&self, _info: &OperationInfo, next: &mut dyn FnMut()) {
    ///         drop(std::thread::spawn(move || next()));
    ///     }
    /// }
    /// ```
    ///
    /// A wrap cannot await `next`, so it cannot hold the user function
    /// across an await:
    ///
    /// ```compile_fail,E0277
    /// # // next_cannot_be_awaited
    /// use aws_durable_execution_sdk::plugin::{OperationInfo, Plugin};
    ///
    /// struct Awaits;
    ///
    /// impl Plugin for Awaits {
    ///     fn wrap_user_function(&self, _info: &OperationInfo, next: &mut dyn FnMut()) {
    ///         drop(async move { next.await });
    ///     }
    /// }
    /// ```
    fn wrap_user_function(&self, _info: &OperationInfo, next: &mut dyn FnMut()) {
        next();
    }

    /// Runs at a user function's first poll, before the SDK calls its
    /// closure.
    ///
    /// It runs after `on_operation_start` and, for a step attempt or a
    /// check, `on_attempt_start`. It runs inside the wraps of the enclosing
    /// user functions and of the plugins registered before this one, and
    /// before this plugin's [`wrap_user_function`](Self::wrap_user_function).
    /// It runs outside the operation's `tracing` span. `info` is described
    /// in the section on operation metadata of `wrap_user_function`, which
    /// also gives the order of every hook and wrap and the cases that fire
    /// no hook.
    ///
    /// The SDK calls it synchronously, as it calls `on_operation_start`, so
    /// it must be cheap and must not block. A returned future runs on its
    /// own task. The SDK never awaits it on the operation's path, and it
    /// does not keep the invocation from suspending; the drain before
    /// `on_invocation_end` waits for it until 500 ms before the invocation
    /// deadline, then aborts it.
    ///
    /// # Errors
    ///
    /// An `Err` returned by this call, or a panic in it, is logged at WARN
    /// and means this plugin did not start: it gets no wrap and no end hook
    /// for this user function. The other plugins and the user function are
    /// unchanged. A returned future that fails later is logged and changes
    /// nothing.
    fn on_user_function_start(&self, _info: &OperationInfo) -> HookWork {
        Ok(None)
    }

    /// Runs in the poll in which a user function returned, after this
    /// plugin's wrap returned and before the SDK serializes the value or
    /// writes the outcome.
    ///
    /// Only a plugin whose start hook started it gets this hook. It runs
    /// inside the wraps of the enclosing user functions and of the plugins
    /// registered before this one, so the end hooks run in reverse
    /// registration order. A user function that panics, or that the SDK
    /// drops before it returns, gets no end hook; the section on when no
    /// hook fires of [`wrap_user_function`](Self::wrap_user_function) lists
    /// every case.
    ///
    /// `outcome` is `Ok(())` when the user function returned `Ok`, and
    /// otherwise the [`WireError`] that the SDK records for its error. The
    /// hook never sees the value. By user function, an error is:
    ///
    /// | User function | `outcome` on error |
    /// | --- | --- |
    /// | step attempt, `wait_for_callback` submitter step | the step's error, with the type `Error` unless the error names one, as a `TypedError` does |
    /// | `wait_for_condition` check | the check's error, with the type `Error` unless the error names one, as a `TypedError` does |
    /// | `run_in_child_context` body, map item, parallel or `parallel_mixed` branch | the body's error, as the child context records it |
    /// | `with_retry` block with its own record | the error that ends the retries; a block with `without_child_context` is not a user function |
    /// | `wait_for_callback` body | the error the operation records, such as `Callback.ExternalFailure` |
    /// | `map`, `parallel` or `parallel_mixed` container | never: the batch holds the item failures |
    /// | `try_join_all` | `CombinatorError.JoinFailed.<i>` for input `i` |
    /// | `join_all` | never: each input's error is in the result |
    /// | `select_ok` | `CombinatorError.AllFailed`, or `CombinatorError.EmptyInput` for no input |
    /// | `race` | `CombinatorError.FirstSettledFailed`, or `CombinatorError.EmptyInput` for no input |
    ///
    /// A failure after the user function returned `Ok`, such as a value
    /// that does not serialize or a rejected checkpoint, keeps its own
    /// error in the record, after an `Ok` end hook.
    ///
    /// The SDK calls this hook synchronously, as it calls
    /// `on_operation_start`, so it must be cheap and must not block. A
    /// returned future runs on its own task, as for
    /// [`on_user_function_start`](Self::on_user_function_start).
    ///
    /// # Errors
    ///
    /// An error or a panic is logged at WARN and changes nothing else.
    fn on_user_function_end(
        &self,
        _info: &OperationInfo,
        _outcome: Result<(), &WireError>,
    ) -> HookWork {
        Ok(None)
    }

    /// Returns the X-Ray trace header the SDK puts on a chained invoke's
    /// `START` checkpoint, or `None` for no header.
    ///
    /// The SDK calls it once for each new chained-invoke `START`, after it
    /// serialized the invoke's input and before it writes the `START`. It
    /// never calls it when the invoke's `START` is already recorded. A
    /// `START` whose write failed is not recorded, so the next invocation
    /// calls it again.
    ///
    /// The SDK asks every plugin, in registration order. The first
    /// non-blank value is sent unchanged as the `X-Amzn-Trace-Id` of the
    /// chained invoke. A value that is empty or only whitespace counts as no
    /// value. When a later plugin returns a different non-blank value, the
    /// SDK keeps the first value and logs a warning that names both plugins.
    /// A panic is logged and counts as no value.
    ///
    /// The hook is synchronous and runs on the operation's task, so it must
    /// not block.
    ///
    /// # Errors
    ///
    /// An error is logged and counts as no value: the next plugin's value
    /// is used.
    fn propagation_trace_id(&self, _input: &PropagationInput) -> Result<Option<String>, BoxError> {
        Ok(None)
    }
}

/// How a step attempt or a condition check ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AttemptOutcome {
    /// The attempt succeeded.
    Succeeded,
    /// The attempt failed.
    Failed,
}

impl AttemptOutcome {
    /// Returns the token JS plugins use for this outcome: `SUCCEEDED` or
    /// `FAILED`.
    #[must_use]
    pub fn wire_str(self) -> &'static str {
        match self {
            Self::Succeeded => "SUCCEEDED",
            Self::Failed => "FAILED",
        }
    }
}

impl fmt::Display for AttemptOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Succeeded => "Succeeded",
            Self::Failed => "Failed",
        })
    }
}

/// What a plugin knows about one invocation when it starts.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InvocationInfo {
    request_id: String,
    execution_arn: String,
    execution_input: String,
    operations: BTreeMap<String, OperationInfo>,
    updated_operations: Vec<OperationInfo>,
    execution_start_timestamp: Option<SystemTime>,
    is_first_invocation: bool,
    x_ray_trace_id: Option<String>,
}

impl InvocationInfo {
    /// Builds the info of one invocation.
    ///
    /// `updated_ids` are the wire IDs the service reports as changed. The
    /// info keeps, in order, each ID that is a key of `operations` and was
    /// not kept before; an unknown or repeated ID is skipped.
    #[expect(clippy::too_many_arguments, reason = "one argument per field")]
    pub(crate) fn new(
        request_id: String,
        execution_arn: String,
        execution_input: String,
        operations: BTreeMap<String, OperationInfo>,
        updated_ids: Vec<String>,
        execution_start_timestamp: Option<SystemTime>,
        is_first_invocation: bool,
        x_ray_trace_id: Option<String>,
    ) -> Self {
        let mut kept: Vec<String> = Vec::new();
        for id in updated_ids {
            if operations.contains_key(&id) && !kept.contains(&id) {
                kept.push(id);
            }
        }
        let updated_operations = kept
            .iter()
            .filter_map(|id| operations.get(id).cloned())
            .collect();
        Self {
            request_id,
            execution_arn,
            execution_input,
            operations,
            updated_operations,
            execution_start_timestamp,
            is_first_invocation,
            x_ray_trace_id,
        }
    }

    /// The Lambda request ID of this invocation.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// The ARN of the durable execution.
    pub fn execution_arn(&self) -> &str {
        &self.execution_arn
    }

    /// The execution input as raw JSON text, or `{}` when the service sent
    /// none.
    pub fn execution_input(&self) -> &str {
        &self.execution_input
    }

    /// Every operation the service reports for this execution, including
    /// the `EXECUTION` operation, keyed by wire ID.
    pub fn operations(&self) -> &BTreeMap<String, OperationInfo> {
        &self.operations
    }

    /// The operations the service changed between the previous invocation
    /// and this one: a wait that ended, a retry that became ready, a
    /// callback reply, an invoke completion. Empty on the first invocation.
    pub fn updated_operations(&self) -> &[OperationInfo] {
        &self.updated_operations
    }

    /// When the execution started, from the `EXECUTION` operation.
    pub fn execution_start_timestamp(&self) -> Option<SystemTime> {
        self.execution_start_timestamp
    }

    /// Whether this is the first invocation of the execution: the service
    /// reported no operation besides the `EXECUTION` operation.
    pub fn is_first_invocation(&self) -> bool {
        self.is_first_invocation
    }

    /// The invocation's X-Ray trace header, from the Lambda context's
    /// `xray_trace_id`. It is never read from the process environment.
    pub fn x_ray_trace_id(&self) -> Option<&str> {
        self.x_ray_trace_id.as_deref()
    }
}

/// What a plugin knows about one invocation when it ends.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InvocationEndInfo {
    request_id: String,
    execution_arn: String,
    execution_input: String,
    operations: BTreeMap<String, OperationInfo>,
    execution_start_timestamp: Option<SystemTime>,
    status: InvocationStatus,
    execution_result: Option<String>,
    execution_error: Option<WireError>,
}

impl InvocationEndInfo {
    /// The Lambda request ID of this invocation.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// The ARN of the durable execution.
    pub fn execution_arn(&self) -> &str {
        &self.execution_arn
    }

    /// The execution input as raw JSON text, or `{}` when the service sent
    /// none.
    pub fn execution_input(&self) -> &str {
        &self.execution_input
    }

    /// Every operation of the execution when the invocation ended,
    /// including the `EXECUTION` operation, keyed by wire ID.
    pub fn operations(&self) -> &BTreeMap<String, OperationInfo> {
        &self.operations
    }

    /// When the execution started, from the `EXECUTION` operation.
    pub fn execution_start_timestamp(&self) -> Option<SystemTime> {
        self.execution_start_timestamp
    }

    /// How the invocation ended.
    pub fn status(&self) -> InvocationStatus {
        self.status
    }

    /// The serialized handler result, when the execution succeeded in this
    /// invocation.
    pub fn execution_result(&self) -> Option<&str> {
        self.execution_result.as_deref()
    }

    /// The recorded handler error, when the execution failed in this
    /// invocation.
    pub fn execution_error(&self) -> Option<&WireError> {
        self.execution_error.as_ref()
    }
}

/// What a plugin knows about one operation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OperationInfo {
    id: String,
    name: Option<String>,
    kind: Option<OperationKind>,
    type_str: Option<String>,
    sub_type: Option<String>,
    parent_id: Option<String>,
    status: Option<OperationStatus>,
    status_str: Option<String>,
    start_timestamp: Option<SystemTime>,
    end_timestamp: Option<SystemTime>,
    result: Option<String>,
    error: Option<WireError>,
    attempt: Option<u32>,
    is_replay: bool,
    is_virtual: bool,
}

#[cfg(test)]
std::thread_local! {
    /// How many times [`OperationInfo::from_record`] ran on this thread. A
    /// `#[tokio::test]` runs every SDK task on the test's own thread.
    static INFOS_FROM_RECORDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The number of [`OperationInfo::from_record`] calls on this thread.
#[cfg(test)]
pub(crate) fn infos_from_records() -> usize {
    INFOS_FROM_RECORDS.with(std::cell::Cell::get)
}

impl OperationInfo {
    /// Converts the checkpoint record stored under the wire ID `id`.
    ///
    /// The result comes from the record's result, else from its invoke
    /// result. The error comes from the record's error fields when any is
    /// present, else from its invoke error fields when any is present, else
    /// it is `None`. An attempt of 0 means the service sent none, so it
    /// gives `None`.
    pub(crate) fn from_record(id: &str, record: &CheckpointRecord) -> Self {
        #[cfg(test)]
        INFOS_FROM_RECORDS.with(|count| count.set(count.get() + 1));
        let (status, status_str) = status_fields(&record.status);
        let error = wire_error_from_fields(
            record.error_type.as_deref(),
            record.error_message.as_deref(),
            record.error_data.as_deref(),
            record.stack_trace.as_deref(),
        )
        .or_else(|| {
            wire_error_from_fields(
                record.invoke_error_type.as_deref(),
                record.invoke_error_message.as_deref(),
                record.invoke_error_data.as_deref(),
                record.invoke_stack_trace.as_deref(),
            )
        });
        Self {
            id: id.to_owned(),
            name: record.op_name.clone(),
            kind: record.op_type.as_deref().and_then(kind_from_wire),
            type_str: record.op_type.clone(),
            sub_type: record.sub_type.clone(),
            parent_id: record.parent_id.clone(),
            status,
            status_str: Some(status_str),
            start_timestamp: record.start_timestamp,
            end_timestamp: record.end_timestamp,
            result: record
                .result
                .clone()
                .or_else(|| record.invoke_result.clone()),
            error,
            attempt: (record.attempt > 0).then_some(record.attempt),
            is_replay: false,
            is_virtual: false,
        }
    }

    /// Builds the info of an `EXECUTION` operation from its wire fields.
    ///
    /// Returns `None` unless `type_str` names the `EXECUTION` type. Every
    /// field not listed is `None` or `false`.
    pub(crate) fn execution(
        id: &str,
        type_str: &str,
        status: &CheckpointStatus,
        parent_id: Option<String>,
        start_timestamp: Option<SystemTime>,
        end_timestamp: Option<SystemTime>,
    ) -> Option<Self> {
        let kind = kind_from_wire(type_str);
        if kind != Some(OperationKind::Execution) {
            return None;
        }
        let (status, status_str) = status_fields(status);
        Some(Self {
            id: id.to_owned(),
            name: None,
            kind,
            type_str: Some(type_str.to_owned()),
            sub_type: None,
            parent_id,
            status,
            status_str: Some(status_str),
            start_timestamp,
            end_timestamp,
            result: None,
            error: None,
            attempt: None,
            is_replay: false,
            is_virtual: false,
        })
    }

    /// The operation's wire ID, as the history records it in `Id`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The name the handler gave the operation.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The operation's kind, or `None` when this SDK version does not know
    /// the service's type.
    pub fn kind(&self) -> Option<OperationKind> {
        self.kind
    }

    /// The service's token, also when `kind()` or `status()` is `None`
    /// because this SDK version does not know the value.
    pub fn type_str(&self) -> Option<&str> {
        self.type_str.as_deref()
    }

    /// The operation's sub-type, such as `Step`, `Map` or `Parallel`.
    pub fn sub_type(&self) -> Option<&str> {
        self.sub_type.as_deref()
    }

    /// The wire ID of the parent operation, as the history records it in
    /// `ParentId`. The SDK infers no parent.
    pub fn parent_id(&self) -> Option<&str> {
        self.parent_id.as_deref()
    }

    /// The operation's status, or `None` when this SDK version does not
    /// know the service's status.
    pub fn status(&self) -> Option<OperationStatus> {
        self.status
    }

    /// The service's token, also when `kind()` or `status()` is `None`
    /// because this SDK version does not know the value.
    pub fn status_str(&self) -> Option<&str> {
        self.status_str.as_deref()
    }

    /// When the operation started.
    pub fn start_timestamp(&self) -> Option<SystemTime> {
        self.start_timestamp
    }

    /// When the operation ended.
    pub fn end_timestamp(&self) -> Option<SystemTime> {
        self.end_timestamp
    }

    /// The raw serialized result payload, as recorded.
    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }

    /// The recorded error, or `None` when the record carries no error
    /// field. Use [`status`](Self::status) to detect a failure.
    pub fn error(&self) -> Option<&WireError> {
        self.error.as_ref()
    }

    /// The step's attempt number, 1-based, or `None` when the service sent
    /// none.
    pub fn attempt(&self) -> Option<u32> {
        self.attempt
    }

    /// Whether the hook reports a replayed operation. `false` for every
    /// entry of an operations map.
    pub fn is_replay(&self) -> bool {
        self.is_replay
    }

    /// Whether the operation is a context that writes no record: a
    /// `NestingMode::Flat` item or branch, or a virtual child context
    /// ([`ChildBuilder::virtual_context`](crate::builders::ChildBuilder::virtual_context)
    /// or [`WithRetryBuilder::virtual_context`](crate::builders::WithRetryBuilder::virtual_context)).
    /// `false` for every other operation and for every entry of an
    /// operations map.
    ///
    /// A virtual child context fires its hooks in every invocation, because
    /// its body runs in every invocation. Its start hook reports
    /// [`is_replay`](Self::is_replay) `true` when an earlier invocation
    /// already ran its body, and its end hook reports `true` when an
    /// earlier invocation already got past the child.
    pub fn is_virtual(&self) -> bool {
        self.is_virtual
    }
}

/// What a plugin knows about one step attempt or condition check when it
/// starts.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AttemptInfo {
    /// Shared with the attempt's runner, so the user-function hooks and
    /// wraps receive this very value.
    operation: Arc<OperationInfo>,
    attempt: u32,
}

impl AttemptInfo {
    /// The operation the attempt belongs to.
    pub fn operation(&self) -> &OperationInfo {
        &self.operation
    }

    /// The attempt number, 1-based.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}

/// What a plugin knows about one step attempt or condition check when it
/// ends.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AttemptEndInfo {
    operation: OperationInfo,
    attempt: u32,
    outcome: AttemptOutcome,
    error: Option<WireError>,
}

impl AttemptEndInfo {
    /// The operation the attempt belongs to.
    pub fn operation(&self) -> &OperationInfo {
        &self.operation
    }

    /// The attempt number, 1-based.
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// How the attempt ended.
    pub fn outcome(&self) -> AttemptOutcome {
        self.outcome
    }

    /// The attempt's error, when it failed.
    pub fn error(&self) -> Option<&WireError> {
        self.error.as_ref()
    }
}

// The builders the operation hook sites use.

impl OperationInfo {
    /// The info of the operation `wire_id`, created in `ctx`.
    ///
    /// It starts from the operation's record when the log holds one. Each
    /// identity field the record leaves absent is then set from the
    /// arguments: the kind, the sub-type, the name, and the parent, which
    /// is `ctx.parent_wire_id()`. A field the record populates keeps the
    /// record's value. Without a record the info holds only the ID and
    /// those fields. The service's type and status tokens come only from a
    /// record, so this sets neither.
    pub(crate) fn for_operation(
        ctx: &crate::DurableContext,
        wire_id: &str,
        kind: OperationKind,
        sub_type: &str,
        name: Option<&str>,
    ) -> Self {
        let mut info = ctx
            .with_checkpoint_record_by_wire(wire_id, |record| Self::from_record(wire_id, record))
            .unwrap_or_else(|| Self {
                id: wire_id.to_owned(),
                name: None,
                kind: None,
                type_str: None,
                sub_type: None,
                parent_id: None,
                status: None,
                status_str: None,
                start_timestamp: None,
                end_timestamp: None,
                result: None,
                error: None,
                attempt: None,
                is_replay: false,
                is_virtual: false,
            });
        info.kind = info.kind.or(Some(kind));
        if info.sub_type.is_none() {
            info.sub_type = Some(sub_type.to_owned());
        }
        if info.name.is_none() {
            info.name = name.map(str::to_owned);
        }
        if info.parent_id.is_none() {
            info.parent_id = ctx.parent_wire_id().map(str::to_owned);
        }
        info
    }

    /// Replaces the status. The service's status token is left as the
    /// record set it.
    pub(crate) fn with_status(mut self, status: OperationStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Replaces the result.
    pub(crate) fn with_result(mut self, result: Option<String>) -> Self {
        self.result = result;
        self
    }

    /// Replaces the error.
    pub(crate) fn with_error(mut self, error: Option<WireError>) -> Self {
        self.error = error;
        self
    }

    /// Replaces the name.
    pub(crate) fn with_name(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }

    /// Replaces the replay flag.
    pub(crate) fn with_replay(mut self, is_replay: bool) -> Self {
        self.is_replay = is_replay;
        self
    }

    /// Replaces the virtual flag. A flat map item or parallel branch, which
    /// writes no record, reports with `true`.
    pub(crate) fn with_virtual(mut self, is_virtual: bool) -> Self {
        self.is_virtual = is_virtual;
        self
    }

    /// Replaces the attempt number.
    pub(crate) fn with_attempt(mut self, attempt: u32) -> Self {
        self.attempt = Some(attempt);
        self
    }

    /// Replaces the start timestamp.
    pub(crate) fn with_start_timestamp(mut self, at: SystemTime) -> Self {
        self.start_timestamp = Some(at);
        self
    }

    /// Replaces the end timestamp.
    pub(crate) fn with_end_timestamp(mut self, at: SystemTime) -> Self {
        self.end_timestamp = Some(at);
        self
    }
}

impl AttemptInfo {
    pub(crate) fn new(operation: Arc<OperationInfo>, attempt: u32) -> Self {
        Self { operation, attempt }
    }
}

impl AttemptEndInfo {
    pub(crate) fn new(
        operation: OperationInfo,
        attempt: u32,
        outcome: AttemptOutcome,
        error: Option<WireError>,
    ) -> Self {
        Self {
            operation,
            attempt,
            outcome,
            error,
        }
    }
}

/// Gives `info` the end timestamp an attempt-end hook reports.
///
/// 1. The record's end timestamp, which the checkpoint response sets when
///    it returns the operation, is kept.
/// 2. Without one, as on a `RETRY` record, the end timestamp is the local
///    time now. Every hook site calls this right after its checkpoint call
///    returned, so it is the time the SDK observed then.
///
/// The clock read feeds only the plugin info, so it does not affect
/// replay.
pub(crate) fn with_observed_end(info: OperationInfo) -> OperationInfo {
    if info.end_timestamp.is_some() {
        return info;
    }
    info.with_end_timestamp(SystemTime::now())
}

/// Fires the hooks of a live attempt that starts: `on_operation_start`
/// with `is_replay`, then `on_attempt_start` with the attempt's start time.
///
/// `operation` is the info [`OperationInfo::for_operation`] built; both
/// hooks see it with the status `Started`. The start hook's copy keeps the
/// record's `attempt` and carries `is_replay`, as on main and in JS; the
/// attempt's copy has `attempt` set and `is_replay` false. It
/// returns the attempt's info exactly as `on_attempt_start` saw it, so the
/// user-function hooks and wraps of the attempt see the same value, as JS
/// passes `attemptInfo` to both.
pub(crate) fn attempt_started(
    plugins: &InvocationPlugins,
    operation: OperationInfo,
    attempt: u32,
    is_replay: bool,
) -> Arc<OperationInfo> {
    let started = operation.with_status(OperationStatus::Started);
    operation_started(plugins, &started.clone().with_replay(is_replay));
    attempt_start_only(plugins, started, attempt)
}

/// Fires only `on_attempt_start` of a live attempt that starts, with the
/// attempt's start time. The operation's start step already fired
/// `on_operation_start` (see [`crate::start`]), as JS fires
/// `onOperationAttemptStart` after `onOperationStart`.
///
/// `operation` is the info [`OperationInfo::for_operation`] built; the hook
/// sees it with the status `Started` and `attempt` set. It returns the
/// attempt's info exactly as `on_attempt_start` saw it, as
/// [`attempt_started`] does.
pub(crate) fn attempt_start_only(
    plugins: &InvocationPlugins,
    operation: OperationInfo,
    attempt: u32,
) -> Arc<OperationInfo> {
    let started = operation
        .with_status(OperationStatus::Started)
        .with_attempt(attempt)
        .with_start_timestamp(SystemTime::now());
    let info = AttemptInfo::new(Arc::new(started), attempt);
    plugins.dispatch_work("on_attempt_start", |plugin| plugin.on_attempt_start(&info));
    Arc::clone(&info.operation)
}

/// Fires the hooks of an attempt that ended, right after the checkpoint
/// call that recorded its outcome returned: `on_attempt_end`, then, when
/// the attempt ended the operation, `on_operation_end` with the same
/// operation info.
///
/// `operation` carries the status, result and error the SDK wrote. Its end
/// timestamp is set by [`with_observed_end`].
pub(crate) fn attempt_ended(
    plugins: &InvocationPlugins,
    operation: OperationInfo,
    end: AttemptEnd,
) {
    let operation = with_observed_end(operation);
    let ends_operation = end.ends_operation;
    let info = AttemptEndInfo::new(operation, end.attempt, end.outcome, end.error);
    plugins.dispatch_work("on_attempt_end", |plugin| plugin.on_attempt_end(&info));
    if ends_operation {
        let operation = info.operation;
        plugins.dispatch_work("on_operation_end", |plugin| {
            plugin.on_operation_end(&operation)
        });
    }
}

/// Fires `on_operation_start` with `info`.
pub(crate) fn operation_started(plugins: &InvocationPlugins, info: &OperationInfo) {
    plugins.dispatch_work("on_operation_start", |plugin| {
        plugin.on_operation_start(info)
    });
}

/// Fires `on_operation_end` with `info`.
pub(crate) fn operation_ended(plugins: &InvocationPlugins, info: &OperationInfo) {
    plugins.dispatch_work("on_operation_end", |plugin| plugin.on_operation_end(info));
}

/// How an attempt ended, for [`attempt_ended`].
pub(crate) struct AttemptEnd {
    /// The 1-based attempt number.
    pub(crate) attempt: u32,
    pub(crate) outcome: AttemptOutcome,
    /// The attempt's error, when it failed.
    pub(crate) error: Option<WireError>,
    /// Whether the attempt ended the operation: the SDK wrote `SUCCEED` or
    /// `FAIL`, not `RETRY`.
    pub(crate) ends_operation: bool,
}

/// What a plugin knows when a checkpoint response reports operations the
/// service changed.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OperationChangeInfo {
    execution_arn: String,
    updated_operations: Vec<OperationInfo>,
    operations: BTreeMap<String, OperationInfo>,
}

impl OperationChangeInfo {
    /// The ARN of the durable execution.
    pub fn execution_arn(&self) -> &str {
        &self.execution_arn
    }

    /// The operations the response reports as changed.
    pub fn updated_operations(&self) -> &[OperationInfo] {
        &self.updated_operations
    }

    /// Every operation of the execution after the response, including the
    /// `EXECUTION` operation, keyed by wire ID.
    pub fn operations(&self) -> &BTreeMap<String, OperationInfo> {
        &self.operations
    }

    /// The change info of one checkpoint response.
    pub(crate) fn new(
        execution_arn: String,
        updated_operations: Vec<OperationInfo>,
        operations: BTreeMap<String, OperationInfo>,
    ) -> Self {
        Self {
            execution_arn,
            updated_operations,
            operations,
        }
    }
}

/// What a plugin knows when it supplies the trace header of a chained
/// invoke.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PropagationInput {
    execution_arn: String,
    operation_id: String,
    parent_operation_id: Option<String>,
    target_function_name: String,
}

impl PropagationInput {
    /// The input of the propagation hook for the chained invoke
    /// `operation_id`, whose parent is `parent_operation_id` and which calls
    /// `target_function_name`.
    pub(crate) fn new(
        execution_arn: String,
        operation_id: String,
        parent_operation_id: Option<String>,
        target_function_name: String,
    ) -> Self {
        Self {
            execution_arn,
            operation_id,
            parent_operation_id,
            target_function_name,
        }
    }

    /// The ARN of the durable execution.
    pub fn execution_arn(&self) -> &str {
        &self.execution_arn
    }

    /// The wire ID of the invoke operation.
    pub fn operation_id(&self) -> &str {
        &self.operation_id
    }

    /// The wire ID of the invoke's parent operation.
    pub fn parent_operation_id(&self) -> Option<&str> {
        self.parent_operation_id.as_deref()
    }

    /// The function the invoke calls.
    pub fn target_function_name(&self) -> &str {
        &self.target_function_name
    }
}

/// Maps a stored type token, in either spelling, to its kind.
fn kind_from_wire(op_type: &str) -> Option<OperationKind> {
    let canonical = crate::context::canonical_op_type(op_type);
    OperationKind::from_wire(&aws_sdk_lambda::types::OperationType::from(
        canonical.as_str(),
    ))
}

/// The typed status and the wire token of a stored status. An unknown
/// status has no typed value and keeps its raw token.
fn status_fields(status: &CheckpointStatus) -> (Option<OperationStatus>, String) {
    let typed = match status {
        CheckpointStatus::Started => Some(OperationStatus::Started),
        CheckpointStatus::Pending => Some(OperationStatus::Pending),
        CheckpointStatus::Ready => Some(OperationStatus::Ready),
        CheckpointStatus::Succeeded => Some(OperationStatus::Succeeded),
        CheckpointStatus::Failed => Some(OperationStatus::Failed),
        CheckpointStatus::Cancelled => Some(OperationStatus::Cancelled),
        CheckpointStatus::TimedOut => Some(OperationStatus::TimedOut),
        CheckpointStatus::Stopped => Some(OperationStatus::Stopped),
        CheckpointStatus::Unknown(_) => None,
    };
    (typed, status.wire_str().to_owned())
}

/// A wire error from the four error fields, or `None` when every field is
/// absent.
fn wire_error_from_fields(
    error_type: Option<&str>,
    error_message: Option<&str>,
    error_data: Option<&str>,
    stack_trace: Option<&[String]>,
) -> Option<WireError> {
    if error_type.is_none()
        && error_message.is_none()
        && error_data.is_none()
        && stack_trace.is_none()
    {
        return None;
    }
    Some(
        WireError::new(error_type, error_message)
            .with_error_data(error_data)
            .with_stack_trace(stack_trace.map(<[String]>::to_vec).unwrap_or_default()),
    )
}

/// Projects every record of `log` into an operation info, with one read of
/// the log.
///
/// The log holds no `EXECUTION` record: the bootstrap parsers and the
/// checkpoint-response merge both skip it. The `EXECUTION` slot of an
/// operations map comes from the plugins' own slot instead.
pub(crate) fn operation_infos(log: &CheckpointLog) -> BTreeMap<String, OperationInfo> {
    log.project_records(OperationInfo::from_record)
        .into_iter()
        .map(|info| (info.id.clone(), info))
        .collect()
}

/// The operations map a plugin sees: [`operation_infos`] of `log`, with the
/// `EXECUTION` operation inserted under its wire ID when it is known.
pub(crate) fn operations_map(
    log: &CheckpointLog,
    execution: Option<&OperationInfo>,
) -> BTreeMap<String, OperationInfo> {
    let mut map = operation_infos(log);
    if let Some(execution) = execution {
        map.insert(execution.id.clone(), execution.clone());
    }
    map
}

/// A caught panic payload.
type Payload = Box<dyn std::any::Any + Send + 'static>;

/// One plugin of an invocation, with the name the SDK uses for it in
/// warnings.
pub(crate) struct PluginSlot {
    pub(crate) plugin: Arc<dyn Plugin>,
    /// `<Plugin::name()> (plugins[<i>])`, or `plugins[<i>]` when `name()`
    /// panicked.
    pub(crate) identity: String,
}

/// The time the invocation-end drain leaves before the Lambda deadline:
/// hook work still running then is aborted. Named after JS
/// `PLUGIN_DRAIN_RESERVE_MS`.
pub(crate) const PLUGIN_DRAIN_RESERVE: Duration = Duration::from_millis(500);

/// The time the end hooks leave before the Lambda deadline: the response
/// returns without them then. Named after JS `PLUGIN_END_RESERVE_MS`.
pub(crate) const PLUGIN_END_RESERVE: Duration = Duration::from_millis(50);

/// The hook work of one invocation.
///
/// `start_work` spawns while it holds the lock, and the drain closes the
/// state and takes the list in one critical section. So a `start_work`
/// call either lands in a list the drain takes, or sees `Closed`.
enum WorkState {
    /// The drain has not closed: the handles of the work started so far.
    Open(Vec<JoinHandle<()>>),
    /// The drain closed: work started from now on is dropped unspawned.
    Closed,
}

/// Where a panic goes that the destructor of an undelivered plugin value
/// raised: a hook error, or a factory error.
///
/// Such a panic fails the invocation, so it goes to the invocation
/// coordinator. The plugins are created and their start hooks run before
/// the coordinator exists, so the panics of that time are kept until the
/// invocation attaches its coordinator.
///
/// The drive closes the coordinator's panic ownership before the drain
/// and the end hooks run, so the coordinator returns a panic recorded
/// after that. Such a panic is kept as a late panic, and the invocation
/// end decides what it does (see [`LatePanics`]). The coordinator also
/// returns a second panic while ownership is open. Then the drive already
/// ended with the first panic or with a fatal error, and the invocation
/// end discards every late panic.
///
/// The plugins shut down when they are dropped, or when the invocation
/// future is dropped before it returned. Nobody then delivers a panic, so
/// every kept or late panic is dropped through
/// [`crate::task::discard_panic`], and so is every panic recorded after
/// that: an aborted hook task can still be polled on another thread.
enum Undelivered {
    /// No coordinator yet: the panics so far.
    Kept(Vec<Payload>),
    /// The invocation's coordinator, and the panics it returned.
    Attached {
        coordinator: Arc<InvocationCoordinator>,
        late: Vec<Payload>,
    },
    /// The plugins shut down: every panic is dropped at once.
    Discarding,
}

impl Undelivered {
    /// Records `payload` on the coordinator, or keeps it until one is
    /// attached, or keeps it as a late panic when the coordinator returns
    /// it, or drops it once the plugins shut down.
    fn record(this: &Mutex<Self>, payload: Payload) {
        let coordinator = {
            let mut state = this
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match &mut *state {
                Self::Kept(kept) => {
                    kept.push(payload);
                    return;
                }
                Self::Attached { coordinator, .. } => Arc::clone(coordinator),
                Self::Discarding => {
                    drop(state);
                    discard_plugin_panic(payload);
                    return;
                }
            }
        };
        let Some(rejected) = coordinator.record_panic(payload) else {
            return;
        };
        Self::keep_late(this, rejected);
    }

    /// Keeps `payload`, which the coordinator returned, as a late panic.
    /// The plugins may have shut down since the coordinator was read; the
    /// payload is then dropped.
    ///
    /// The payload is dropped after the lock is released. A payload can
    /// hold the last context that keeps the plugins alive. Dropping it
    /// then drops the plugins, and their shutdown locks this state again.
    /// So a drop under the lock would deadlock.
    fn keep_late(this: &Mutex<Self>, payload: Payload) {
        let rejected = {
            let mut state = this
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Self::Attached { late, .. } = &mut *state {
                late.push(payload);
                return;
            }
            // `Kept` never follows `Attached`, so this is `Discarding`.
            payload
        };
        discard_plugin_panic(rejected);
    }

    /// Enters `Discarding`, and drops every panic kept so far.
    fn discard_all(this: &Mutex<Self>) {
        let state = std::mem::replace(
            &mut *this
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Self::Discarding,
        );
        if let Self::Kept(left) | Self::Attached { late: left, .. } = state {
            for payload in left {
                discard_plugin_panic(payload);
            }
        }
    }
}

/// What a late panic does: a panic of an undelivered plugin value's
/// destructor that the coordinator returned (see [`Undelivered`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LatePanics {
    /// The invocation was to return a response: the first late panic
    /// fails the invocation instead, as it would have during the drive.
    Fail,
    /// The invocation already fails: with a panic, with a fatal error, or
    /// with a Lambda error. The drive keeps a fatal error over a panic and
    /// the first panic over a later one, so every late panic is dropped.
    Discard,
}

/// The plugins of one invocation and what they share.
///
/// The root context and every context derived from it hold an `Arc` of
/// the value, and so does the invocation future until it returns. Its
/// `Drop` drops each plugin inside its own catch, so a plugin destructor
/// that panics never changes the invocation's outcome.
pub(crate) struct InvocationPlugins {
    info: InvocationInfo,
    plugins: Vec<PluginSlot>,
    execution: Mutex<Option<OperationInfo>>,
    /// The panics of undelivered plugin values' destructors, which fail
    /// the invocation.
    undelivered: Arc<Mutex<Undelivered>>,
    /// The work the operation hooks returned.
    work: Mutex<WorkState>,
    /// The change sets of the checkpoint responses not yet delivered to
    /// `on_operation_change`, in response order.
    change_queue: Mutex<std::collections::VecDeque<Vec<OperationInfo>>>,
    /// Whether a task is delivering the change sets of `change_queue`.
    draining: std::sync::atomic::AtomicBool,
    /// The number of change sets ever pushed to `change_queue`. It grows
    /// under the queue lock.
    changes_pushed: std::sync::atomic::AtomicU64,
    /// How many change sets were delivered, and the tasks that wait for a
    /// delivery (see [`Self::poll_changes_delivered`]).
    delivery: Mutex<ChangeDelivery>,
    /// Runs right after a drain cleared the drain flag, before the drain
    /// checks the queue again. A test pushes a change set there.
    #[cfg(test)]
    after_release: Mutex<Option<AfterRelease>>,
}

/// The test hook of [`InvocationPlugins::set_after_release`].
#[cfg(test)]
type AfterRelease = Box<dyn FnMut(&InvocationPlugins) + Send>;

/// The delivery progress of the change sets of one invocation.
#[derive(Default)]
pub(crate) struct ChangeDelivery {
    /// The number of change sets whose `on_operation_change` calls
    /// returned.
    delivered: u64,
    /// The waker of each pending waiter, keyed by the waiter's
    /// registration key. A waiter has at most one entry: a later poll
    /// replaces its waker instead of adding one, so the map holds at most
    /// one entry per live waiter. Each delivery takes and wakes them all.
    waiters: BTreeMap<u64, std::task::Waker>,
    /// The key the next registering waiter receives.
    next_key: u64,
}

/// The state of one wait for change-set deliveries (see
/// [`InvocationPlugins::poll_changes_delivered`]). Its owner calls
/// [`InvocationPlugins::cancel_wait`] when it stops waiting before the
/// wait is ready, so no registration outlives it.
#[derive(Debug, Default)]
pub(crate) struct DeliveryWait {
    /// The number of change sets to wait for. The first poll sets it.
    target: Option<u64>,
    /// The key of this wait's waker registration, once it has one.
    key: Option<u64>,
}

std::thread_local! {
    /// The address of the [`InvocationPlugins`] whose change sets this
    /// thread is delivering, or 0. A hook that waits for deliveries on the
    /// thread that runs it would wait for itself.
    static DRAINING_HERE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Marks the current thread as the drain of one [`InvocationPlugins`]
/// while it lives, and restores the previous mark when dropped.
struct DrainingHere(usize);

impl DrainingHere {
    fn enter(plugins: &InvocationPlugins) -> Self {
        let address = std::ptr::from_ref(plugins) as usize;
        Self(DRAINING_HERE.with(|here| here.replace(address)))
    }
}

impl Drop for DrainingHere {
    fn drop(&mut self) {
        DRAINING_HERE.with(|here| here.set(self.0));
    }
}

impl InvocationPlugins {
    /// The plugin at `index`.
    ///
    /// Every index the SDK passes comes from `0..self.plugins.len()`, or
    /// from a list built from that range, and the plugin vector never
    /// changes after [`create_plugins`] built it. So `index` is in range.
    #[expect(
        clippy::expect_used,
        reason = "every index comes from the range of the immutable plugin vector"
    )]
    pub(crate) fn slot(&self, index: usize) -> &PluginSlot {
        self.plugins
            .get(index)
            .expect("a plugin index comes from the plugin vector's range")
    }

    /// The info every plugin of this invocation was created with.
    pub(crate) fn info(&self) -> &InvocationInfo {
        &self.info
    }

    /// Whether the service changed the operation `id` (its wire ID) since
    /// the previous invocation: [`InvocationInfo::updated_operations`]
    /// holds an operation with that ID.
    pub(crate) fn is_updated(&self, id: &str) -> bool {
        self.info
            .updated_operations()
            .iter()
            .any(|operation| operation.id() == id)
    }

    /// Replaces the `EXECUTION` slot with `info`, the `EXECUTION` operation
    /// a checkpoint response returned. Returns whether that is a change:
    /// the slot held no operation, or held one with another status.
    pub(crate) fn replace_execution(&self, info: OperationInfo) -> bool {
        let mut slot = self
            .execution
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let changed = slot
            .as_ref()
            .is_none_or(|held| held.status_str != info.status_str);
        *slot = Some(info);
        changed
    }

    /// Queues the change set of one checkpoint response for
    /// `on_operation_change`. The SDK never queues an empty change set.
    pub(crate) fn push_operation_change(&self, updated: Vec<OperationInfo>) {
        let mut queue = self
            .change_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        queue.push_back(updated);
        self.changes_pushed
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// Delivers every queued change set to `on_operation_change`, in queue
    /// order, unless another task is delivering them already.
    ///
    /// One task delivers at a time, which keeps the calls in response
    /// order. A task that finds the drain flag set returns at once: the
    /// task that holds the flag delivers its entry, because it checks the
    /// queue again after it clears the flag. The queue lock is released
    /// before each delivery, so a hook never runs under it.
    ///
    /// Each entry builds one operations map, the operations of `log` with
    /// the `EXECUTION` slot, and every plugin receives the same info.
    pub(crate) fn drain_operation_changes(&self, log: &CheckpointLog) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        loop {
            if self
                .draining
                .compare_exchange(false, true, AcqRel, Acquire)
                .is_err()
            {
                return;
            }
            let here = DrainingHere::enter(self);
            while let Some(entry) = self.pop_operation_change() {
                let operations = {
                    let execution = self
                        .execution
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    operations_map(log, execution.as_ref())
                };
                let info = OperationChangeInfo::new(
                    self.info().execution_arn().to_owned(),
                    entry,
                    operations,
                );
                self.dispatch_work("on_operation_change", |p| p.on_operation_change(&info));
                drop(info);
                self.finish_delivery();
            }
            drop(here);
            self.draining
                .store(false, std::sync::atomic::Ordering::Release);
            #[cfg(test)]
            self.run_after_release();
            if self.change_queue_is_empty() {
                return;
            }
        }
    }

    /// Counts one delivered change set, then wakes every task that waits
    /// for a delivery.
    fn finish_delivery(&self) {
        let waiters = {
            let mut delivery = self
                .delivery
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            delivery.delivered += 1;
            std::mem::take(&mut delivery.waiters)
        };
        for waiter in waiters.into_values() {
            waiter.wake();
        }
    }

    /// Delivers the queued change sets, and is ready once every change set
    /// pushed before the first poll was delivered.
    ///
    /// A drain that finds another task delivering returns at once (see
    /// [`Self::drain_operation_changes`]). A caller whose own change set
    /// must reach `on_operation_change` before it continues, such as an
    /// operation that fires `on_operation_start` or `on_operation_end`
    /// next, waits here instead, until the delivering task got through
    /// that change set. `target` holds the number of change sets to wait
    /// for; the first poll sets it.
    ///
    /// On the thread that is delivering change sets, the poll is ready at
    /// once: a hook that waited there would wait for itself.
    pub(crate) fn poll_changes_delivered(
        &self,
        log: &CheckpointLog,
        wait: &mut DeliveryWait,
        cx: &mut Context<'_>,
    ) -> Poll<()> {
        use std::sync::atomic::Ordering::Acquire;
        let target = *wait
            .target
            .get_or_insert_with(|| self.changes_pushed.load(Acquire));
        self.drain_operation_changes(log);
        let mut delivery = self
            .delivery
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if delivery.delivered >= target
            || DRAINING_HERE.with(std::cell::Cell::get) == std::ptr::from_ref(self) as usize
        {
            // A delivery takes every registration, and the owner's
            // `cancel_wait` removes any other one when it drops the wait.
            return Poll::Ready(());
        }
        // The drain above lost to a task that was delivering. That task,
        // or the one that takes the flag after it, delivers every change
        // set pushed before this poll, because a drain checks the queue
        // again after it clears the flag. Each delivery wakes this task.
        // The registration happens under the lock a delivery counts
        // under, so no delivery is missed. A repeated poll refreshes this
        // wait's one entry rather than adding another.
        let key = *wait.key.get_or_insert_with(|| {
            let key = delivery.next_key;
            delivery.next_key += 1;
            key
        });
        match delivery.waiters.entry(key) {
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if !entry.get().will_wake(cx.waker()) {
                    entry.insert(cx.waker().clone());
                }
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(cx.waker().clone());
            }
        }
        Poll::Pending
    }

    /// Removes the waker registration of `wait`, whose owner stops
    /// waiting before the wait was ready.
    pub(crate) fn cancel_wait(&self, wait: &mut DeliveryWait) {
        if let Some(key) = wait.key.take() {
            self.delivery
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .waiters
                .remove(&key);
        }
    }

    /// [`Self::poll_changes_delivered`] as a future. Dropping it before it
    /// is ready removes its registration.
    pub(crate) async fn changes_delivered(&self, log: &CheckpointLog) {
        /// Cancels the wait when the future is dropped.
        struct Guard<'a> {
            plugins: &'a InvocationPlugins,
            wait: DeliveryWait,
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.plugins.cancel_wait(&mut self.wait);
            }
        }
        let mut guard = Guard {
            plugins: self,
            wait: DeliveryWait::default(),
        };
        std::future::poll_fn(|cx| {
            guard
                .plugins
                .poll_changes_delivered(log, &mut guard.wait, cx)
        })
        .await;
    }

    /// The number of waker registrations of pending waiters.
    #[cfg(test)]
    pub(crate) fn delivery_waiter_count(&self) -> usize {
        self.delivery
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .waiters
            .len()
    }

    /// Runs and clears the test hook of `after_release`.
    #[cfg(test)]
    fn run_after_release(&self) {
        let hook = self
            .after_release
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(mut hook) = hook {
            hook(self);
        }
    }

    /// Sets a hook that runs once, right after a drain cleared the drain
    /// flag and before it checks the queue again.
    #[cfg(test)]
    pub(crate) fn set_after_release(&self, hook: impl FnMut(&Self) + Send + 'static) {
        *self
            .after_release
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Box::new(hook));
    }

    /// Takes the oldest queued change set.
    fn pop_operation_change(&self) -> Option<Vec<OperationInfo>> {
        self.change_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
    }

    /// Whether no change set is queued.
    fn change_queue_is_empty(&self) -> bool {
        self.change_queue
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }

    /// Whether a task is delivering change sets now.
    #[cfg(test)]
    pub(crate) fn is_draining(&self) -> bool {
        self.draining.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Returns the kept destructor panics, and leaves none behind.
    #[cfg(test)]
    pub(crate) fn take_error_drop_panics(&self) -> Vec<Payload> {
        match &mut *self
            .undelivered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Undelivered::Kept(kept) => std::mem::take(kept),
            Undelivered::Attached { .. } | Undelivered::Discarding => Vec::new(),
        }
    }

    /// Hands the invocation's coordinator to the plugins. The destructor
    /// panics kept so far are recorded on it, which ends the drive, and
    /// every later one is recorded on it at once. A kept panic the
    /// coordinator rejects is plugin code, so it is dropped through
    /// [`discard_plugin_panic`].
    pub(crate) fn attach(&self, coordinator: &Arc<InvocationCoordinator>) {
        let kept = std::mem::replace(
            &mut *self
                .undelivered
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            Undelivered::Attached {
                coordinator: Arc::clone(coordinator),
                late: Vec::new(),
            },
        );
        if let Undelivered::Kept(kept) = kept {
            for payload in kept {
                if let Some(rejected) = coordinator.record_panic(payload) {
                    discard_plugin_panic(rejected);
                }
            }
        }
    }

    /// Shuts the plugins down: they are dropped, or the invocation future
    /// was dropped before it returned, so nobody runs the invocation end.
    ///
    /// It closes the work state and aborts every hook task. It cannot wait
    /// for them, so an aborted task drops its work on its runtime, inside
    /// the catch of its run. Every panic kept so far, and every panic
    /// recorded from now on, is dropped through
    /// [`crate::task::discard_panic`]. After the invocation end, the drain
    /// has joined every task and the late panics were taken, so this
    /// aborts nothing and drops nothing then.
    pub(crate) fn shut_down(&self) {
        if let WorkState::Open(list) = std::mem::replace(&mut *self.lock_work(), WorkState::Closed)
        {
            for handle in list {
                handle.abort();
            }
        }
        Undelivered::discard_all(&self.undelivered);
    }

    /// Runs `on_invocation_start` on every plugin concurrently, and waits
    /// for all of them.
    pub(crate) async fn invocation_start(&self) {
        let info = self.info();
        JoinHooks::new(self.runs("on_invocation_start", |plugin| {
            plugin.on_invocation_start(info)
        }))
        .await;
    }

    /// Drains the hook work, then runs `on_invocation_end` on every plugin
    /// concurrently, with the end info built from `log` and `end`.
    ///
    /// `remaining` is the time left until the Lambda deadline, or `None`
    /// without a usable deadline. `late` says what a late panic does (see
    /// [`Undelivered`]).
    ///
    /// 1. The drain waits for the hook work until
    ///    [`PLUGIN_DRAIN_RESERVE`] before the deadline. Work that settled
    ///    work started is waited for in the next round.
    /// 2. When that time passes, the drain closes, aborts the work still
    ///    running, and waits for every task to end. So no hook work runs
    ///    once the end hooks start.
    /// 3. The end hooks run until [`PLUGIN_END_RESERVE`] before the
    ///    deadline. When that time passes, the SDK logs a warning and
    ///    returns without them.
    ///
    /// Without a deadline both waits last until everything settles.
    ///
    /// Every hook task has ended when the drain returns. So with
    /// [`LatePanics::Fail`], a late panic recorded so far makes the end
    /// `Retrying`, with the panic's message, as a panic during the drive
    /// does. The first late panic, from the drain or from the end hooks,
    /// is returned, and the caller resumes it. With
    /// [`LatePanics::Discard`], every late panic is dropped and this
    /// returns `None`.
    pub(crate) async fn invocation_end(
        &self,
        log: &CheckpointLog,
        end: crate::InvocationEnd,
        remaining: Option<Duration>,
        late: LatePanics,
    ) -> Option<Payload> {
        let now = tokio::time::Instant::now();
        let by = |reserve: Duration| remaining.map(|left| now + left.saturating_sub(reserve));
        self.drain(by(PLUGIN_DRAIN_RESERVE)).await;
        let drained = self.take_late_panic(late);
        let end = match &drained {
            Some(payload) => crate::InvocationEnd::panicked(&**payload),
            None => end,
        };
        let info = self.end_info(log, end);
        let hooks = JoinHooks::new(self.runs("on_invocation_end", |plugin| {
            plugin.on_invocation_end(&info)
        }));
        if !wait_until(by(PLUGIN_END_RESERVE), hooks).await {
            warn_contained(|| {
                tracing::warn!(
                    target: TARGET,
                    "plugin on_invocation_end hooks did not finish before the invocation deadline; \
                     the response returns without them"
                );
            });
        }
        let ended = self.take_late_panic(late);
        match (drained, ended) {
            (Some(first), later) => {
                if let Some(later) = later {
                    discard_plugin_panic(later);
                }
                Some(first)
            }
            (None, ended) => ended,
        }
    }

    /// Takes the late panics. With [`LatePanics::Fail`] it returns the
    /// first one and drops the others; with [`LatePanics::Discard`] it
    /// drops all of them.
    fn take_late_panic(&self, policy: LatePanics) -> Option<Payload> {
        let late = match &mut *self
            .undelivered
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            Undelivered::Attached { late, .. } => std::mem::take(late),
            Undelivered::Kept(_) | Undelivered::Discarding => Vec::new(),
        };
        let mut late = late.into_iter();
        let first = match policy {
            LatePanics::Fail => late.next(),
            LatePanics::Discard => None,
        };
        for payload in late {
            discard_plugin_panic(payload);
        }
        first
    }

    /// Steps 1 and 2 of [`Self::invocation_end`]. Closes the work state.
    async fn drain(&self, drain_by: Option<tokio::time::Instant>) {
        loop {
            let mut joining = AbortOnDrop({
                let mut state = self.lock_work();
                match std::mem::replace(&mut *state, WorkState::Closed) {
                    WorkState::Open(list) if !list.is_empty() => {
                        *state = WorkState::Open(Vec::new());
                        list
                    }
                    _ => return,
                }
            });
            let joined = wait_until(
                drain_by,
                std::future::poll_fn(|cx| {
                    joining
                        .0
                        .retain_mut(|handle| Pin::new(handle).poll(cx).is_pending());
                    if joining.0.is_empty() {
                        Poll::Ready(())
                    } else {
                        Poll::Pending
                    }
                }),
            )
            .await;
            if !joined {
                if let WorkState::Open(rest) =
                    std::mem::replace(&mut *self.lock_work(), WorkState::Closed)
                {
                    joining.0.extend(rest);
                }
                let abandoned = joining
                    .0
                    .iter()
                    .filter(|handle| !handle.is_finished())
                    .count();
                for handle in &joining.0 {
                    handle.abort();
                }
                for handle in &mut joining.0 {
                    // The work catches its own panics, so a join error
                    // only reports the abort.
                    let _ = handle.await;
                }
                if abandoned > 0 {
                    warn_contained(|| {
                        tracing::warn!(
                            target: TARGET,
                            abandoned,
                            "plugin hook work did not finish before the drain deadline; it is aborted"
                        );
                    });
                }
                return;
            }
        }
    }

    /// Calls the synchronous operation hook `hook` on every plugin, in
    /// registration order, through `call`.
    ///
    /// Each call runs inside its own catch, and the remaining plugins run
    /// whatever one call did:
    ///
    /// 1. `Ok(None)` does nothing.
    /// 2. `Ok(Some(work))` hands `work` to [`Self::start_work`].
    /// 3. `Err(error)` is logged at WARN by [`report_hook_error`].
    /// 4. A panic is logged at WARN, and its payload is dropped through
    ///    [`crate::task::discard_panic`]: a plugin panic does not fail the
    ///    invocation.
    pub(crate) fn dispatch_work(&self, hook: &'static str, call: impl Fn(&dyn Plugin) -> HookWork) {
        self.dispatch_to(hook, 0..self.plugins.len(), call);
    }

    /// [`Self::dispatch_work`] for the plugins at `indices`, in that
    /// order. It returns the indices whose call returned `Ok`, in order:
    /// the user-function runner pairs a start with a wrap and an end only
    /// for those.
    pub(crate) fn dispatch_to(
        &self,
        hook: &'static str,
        indices: impl IntoIterator<Item = usize>,
        call: impl Fn(&dyn Plugin) -> HookWork,
    ) -> Vec<usize> {
        let mut ok = Vec::new();
        for index in indices {
            let slot = self.slot(index);
            match call_plugin(|| call(slot.plugin.as_ref())) {
                Ok(Ok(None)) => ok.push(index),
                Ok(Ok(Some(work))) => {
                    ok.push(index);
                    self.start_work(index, hook, work);
                }
                Ok(Err(error)) => {
                    report_hook_error(&slot.identity, hook, &self.undelivered, error);
                }
                Err(payload) => log_hook_panic(&slot.identity, hook, payload),
            }
        }
        ok
    }

    /// Asks every plugin, in registration order, for the X-Ray trace header
    /// of the chained invoke that `input` describes, and returns the header
    /// to send, or `None` for no header.
    ///
    /// Each call runs inside its own catch, and the remaining plugins are
    /// asked whatever one call did:
    ///
    /// 1. A panic is logged at WARN, and its payload is dropped through
    ///    [`crate::task::discard_panic`]: a plugin panic does not fail the
    ///    invocation.
    /// 2. `Err(error)` is logged at WARN by [`report_propagation_error`].
    /// 3. `Ok(None)` and a value that is empty or only whitespace count as
    ///    no value.
    /// 4. The first other value is kept unchanged. A later value equal to
    ///    it does nothing. A later different value is logged at WARN with
    ///    both plugins' identities and the number of conflicts so far in
    ///    this call, and the first value stays.
    pub(crate) fn propagation_trace_id(&self, input: &PropagationInput) -> Option<String> {
        let mut kept: Option<(String, &str)> = None;
        let mut conflict_count: u64 = 0;
        for slot in &self.plugins {
            let value = match call_plugin(|| slot.plugin.propagation_trace_id(input)) {
                Ok(Ok(Some(value))) => value,
                Ok(Ok(None)) => continue,
                Ok(Err(error)) => {
                    report_propagation_error(&slot.identity, &self.undelivered, error);
                    continue;
                }
                Err(payload) => {
                    warn_contained(|| {
                        tracing::warn!(
                            target: TARGET,
                            plugin = %slot.identity,
                            hook = "propagation_trace_id",
                            "plugin propagation_trace_id panicked; the value is ignored"
                        );
                    });
                    discard_plugin_panic(payload);
                    continue;
                }
            };
            if value.trim().is_empty() {
                continue;
            }
            match &kept {
                None => kept = Some((value, slot.identity.as_str())),
                Some((first, _)) if *first == value => {}
                Some((_, first_identity)) => {
                    conflict_count = conflict_count.saturating_add(1);
                    warn_contained(|| {
                        tracing::warn!(
                            target: TARGET,
                            field = "x_amzn_trace_id",
                            first_plugin = %first_identity,
                            later_plugin = %slot.identity,
                            conflict_count,
                            "conflicting propagation trace ids; keeping the first value"
                        );
                    });
                }
            }
        }
        kept.map(|(value, _)| value)
    }

    /// Runs `work`, which the operation hook `hook` of the plugin at index
    /// `slot` returned, on its own task.
    ///
    /// The task is not counted by the SDK's task cleanup and holds no
    /// suspension permit. Before the drain closes, the task is spawned
    /// under the lock and its handle is kept for the drain. After it
    /// closed, `work` is dropped unspawned and a warning is logged.
    pub(crate) fn start_work(&self, slot: usize, hook: &'static str, work: HookFuture<'static>) {
        #[expect(clippy::expect_used)]
        // reason: every caller passes the index of a plugin of this
        // invocation, which it read from `self.plugins`.
        let identity = &self
            .plugins
            .get(slot)
            .expect("the slot names a plugin of this invocation")
            .identity;
        let mut state = self.lock_work();
        let WorkState::Open(list) = &mut *state else {
            drop(state);
            drop_hook_future(identity, hook, work);
            warn_contained(|| {
                tracing::warn!(
                    target: TARGET,
                    plugin = %identity,
                    hook,
                    "plugin hook work started after the drain closed; it is dropped"
                );
            });
            return;
        };
        list.retain(|handle| !handle.is_finished());
        let run = HookRun {
            identity: identity.clone().into(),
            hook,
            future: Some(work),
            undelivered: Arc::clone(&self.undelivered),
        };
        if let Ok(handle) = crate::task::spawn_plugin_work(run) {
            list.push(handle);
        } else {
            drop(state);
            warn_contained(|| {
                tracing::warn!(
                    target: TARGET,
                    plugin = %identity,
                    hook,
                    "plugin hook work could not be spawned; it is dropped"
                );
            });
        }
    }

    /// Builds the end info of this invocation: the start info's request
    /// ID, execution ARN, input and start time, the operations of `log`
    /// with the `EXECUTION` operation, and `end`. It projects the log
    /// once, whatever the number of plugins.
    fn end_info(&self, log: &CheckpointLog, end: crate::InvocationEnd) -> InvocationEndInfo {
        let execution = self
            .execution
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        InvocationEndInfo {
            request_id: self.info.request_id.clone(),
            execution_arn: self.info.execution_arn.clone(),
            execution_input: self.info.execution_input.clone(),
            operations: operations_map(log, execution.as_ref()),
            execution_start_timestamp: self.info.execution_start_timestamp,
            status: end.status,
            execution_result: end.result,
            execution_error: end.error,
        }
    }

    /// One hook run per plugin, each with the future `call` returned.
    fn runs<'a>(
        &'a self,
        hook: &'static str,
        call: impl Fn(&'a dyn Plugin) -> HookFuture<'a>,
    ) -> Vec<HookRun<'a>> {
        self.plugins
            .iter()
            .map(|slot| {
                let future = match call_plugin(|| call(slot.plugin.as_ref())) {
                    Ok(future) => Some(future),
                    Err(payload) => {
                        log_hook_panic(&slot.identity, hook, payload);
                        None
                    }
                };
                HookRun {
                    identity: slot.identity.as_str().into(),
                    hook,
                    future,
                    undelivered: Arc::clone(&self.undelivered),
                }
            })
            .collect()
    }

    fn lock_work(&self) -> std::sync::MutexGuard<'_, WorkState> {
        self.work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// The hook tasks the drain took out of the work state. Dropping it
/// aborts every task still in it.
///
/// The drain future can be dropped while it waits, when the invocation
/// future is dropped. The handles are then no longer in the work state, so
/// [`InvocationPlugins::shut_down`] cannot abort them. A dropped
/// `JoinHandle` detaches its task, and the task would keep running the
/// hook work after the invocation. So the drain keeps the handles in this
/// guard while it waits.
struct AbortOnDrop(Vec<JoinHandle<()>>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        for handle in &self.0 {
            handle.abort();
        }
    }
}

/// Awaits `future` until `deadline`, or until it finishes without a
/// deadline. Returns whether it finished.
///
/// `tokio::time::timeout_at` polls `future` once before it checks the
/// deadline, so a future that finishes on its first poll finishes even
/// when the deadline already passed.
async fn wait_until(
    deadline: Option<tokio::time::Instant>,
    future: impl Future<Output = ()>,
) -> bool {
    if let Some(deadline) = deadline {
        tokio::time::timeout_at(deadline, future).await.is_ok()
    } else {
        future.await;
        true
    }
}

/// Runs plugin code synchronously: with [`ActiveBody::Plugin`] active and
/// inside a catch. Every synchronous call into a plugin goes through it:
/// the factory and `name`, every hook, every wrap, every poll and drop of
/// a hook future, and the drop of a plugin or of an error it returned. So
/// a durable operation called there records a context usage error, which
/// fails the execution when it is recorded before the driver's
/// arbitration (a later record changes neither the response nor the
/// late-panic policy), and a panic comes back as a value.
///
/// [`ActiveBody::Plugin`]: crate::context::ActiveBody::Plugin
pub(crate) fn call_plugin<R>(f: impl FnOnce() -> R) -> Result<R, Payload> {
    crate::context::run_body_sync(crate::context::ActiveBody::Plugin, || {
        catch_unwind(AssertUnwindSafe(f))
    })
}

/// Emits one SDK diagnostic about a plugin. It runs the user's `tracing`
/// subscriber, so a panic there is discarded: a log call never unwinds out
/// of plugin handling. Callers dispose of payloads after it returns.
///
/// It runs under [`ActiveBody::Plugin`](crate::context::ActiveBody::Plugin),
/// with the disposal of a subscriber panic: a durable operation the
/// subscriber calls while it reports a plugin records a context usage
/// error, which fails the execution when it is recorded before the
/// driver's arbitration; a later record changes neither the response nor
/// the late-panic policy.
pub(crate) fn warn_contained(emit: impl FnOnce()) {
    crate::context::run_body_sync(crate::context::ActiveBody::Plugin, || {
        if let Err(payload) = catch_unwind(AssertUnwindSafe(emit)) {
            crate::task::discard_panic(payload);
        }
    });
}

/// Drops a panic payload a plugin raised, or a plugin error's destructor
/// raised. Its destructor is plugin code, so it runs under
/// [`ActiveBody::Plugin`](crate::context::ActiveBody::Plugin): a durable
/// operation it calls records a context usage error, which fails the
/// execution when it is recorded before the driver's arbitration; a later
/// record changes neither the response nor the late-panic policy.
pub(crate) fn discard_plugin_panic(payload: Payload) {
    crate::context::run_body_sync(crate::context::ActiveBody::Plugin, || {
        crate::task::discard_panic(payload);
    });
}

/// Logs a caught hook panic, and drops its payload.
pub(crate) fn log_hook_panic(identity: &str, hook: &'static str, payload: Payload) {
    warn_contained(|| {
        tracing::warn!(
            target: TARGET,
            plugin = %identity,
            hook,
            "plugin hook panicked; the panic is ignored"
        );
    });
    discard_plugin_panic(payload);
}

/// Drops a hook future inside a catch. Its destructor is plugin code, so a
/// panic there is logged as the hook's panic.
fn drop_hook_future(identity: &str, hook: &'static str, future: HookFuture<'_>) {
    if let Err(payload) = call_plugin(move || drop(future)) {
        log_hook_panic(identity, hook, payload);
    }
}

/// One hook future of one plugin, polled inside a catch.
///
/// It finishes when the future returns, returns `Err` or panics. An `Err`
/// and a panic are logged. The future is dropped once it finished, or
/// with the run, each time inside a catch.
struct HookRun<'a> {
    identity: std::borrow::Cow<'a, str>,
    hook: &'static str,
    /// `None` once the hook finished.
    future: Option<HookFuture<'a>>,
    undelivered: Arc<Mutex<Undelivered>>,
}

/// Logs the error the hook `hook` of the plugin `identity` returned, and
/// drops it.
///
/// The error's `Display` is plugin code. So it runs inside a catch that
/// only borrows the error, and a panic there is logged and dropped. The
/// SDK delivers the error to nobody, so it then drops the error inside a
/// second catch. A panic from its destructor fails the invocation, so it is
/// recorded on the coordinator through `undelivered`.
fn report_hook_error(
    identity: &str,
    hook: &'static str,
    undelivered: &Mutex<Undelivered>,
    error: BoxError,
) {
    match call_plugin(|| error.to_string()) {
        Ok(message) => warn_contained(|| {
            tracing::warn!(
                target: TARGET,
                plugin = %identity,
                hook,
                error = %message,
                "plugin hook failed; the error is ignored"
            );
        }),
        Err(payload) => {
            warn_contained(|| {
                tracing::warn!(
                    target: TARGET,
                    plugin = %identity,
                    hook,
                    "plugin hook failed, and its error panicked while it was formatted; \
                     the error is ignored"
                );
            });
            discard_plugin_panic(payload);
        }
    }
    if let Err(payload) = call_plugin(move || drop(error)) {
        Undelivered::record(undelivered, payload);
    }
}

/// Logs the error the `propagation_trace_id` hook of the plugin `identity`
/// returned, and drops it.
///
/// The error's `Display` is plugin code. So it runs inside a catch that
/// only borrows the error, and a panic there is logged and dropped. The
/// SDK delivers the error to nobody, so it then drops the error inside a
/// second catch. A panic from its destructor fails the invocation, so it is
/// recorded on the coordinator through `undelivered`.
fn report_propagation_error(identity: &str, undelivered: &Mutex<Undelivered>, error: BoxError) {
    match call_plugin(|| error.to_string()) {
        Ok(message) => warn_contained(|| {
            tracing::warn!(
                target: TARGET,
                plugin = %identity,
                hook = "propagation_trace_id",
                error = %message,
                "plugin propagation_trace_id failed; the value is ignored"
            );
        }),
        Err(payload) => {
            warn_contained(|| {
                tracing::warn!(
                    target: TARGET,
                    plugin = %identity,
                    hook = "propagation_trace_id",
                    "plugin propagation_trace_id failed, and its error panicked while it was \
                     formatted; the value is ignored"
                );
            });
            discard_plugin_panic(payload);
        }
    }
    if let Err(payload) = call_plugin(move || drop(error)) {
        Undelivered::record(undelivered, payload);
    }
}

impl Future for HookRun<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let Some(future) = this.future.as_mut() else {
            return Poll::Ready(());
        };
        let polled = call_plugin(|| future.as_mut().poll(cx));
        if matches!(polled, Ok(Poll::Pending)) {
            return Poll::Pending;
        }
        if let Some(future) = this.future.take() {
            drop_hook_future(&this.identity, this.hook, future);
        }
        match polled {
            Ok(Poll::Ready(Err(error))) => {
                report_hook_error(&this.identity, this.hook, &this.undelivered, error);
            }
            Err(payload) => log_hook_panic(&this.identity, this.hook, payload),
            Ok(Poll::Ready(Ok(())) | Poll::Pending) => {}
        }
        Poll::Ready(())
    }
}

impl Drop for HookRun<'_> {
    /// A run dropped before its hook finished: the drain aborted it, or
    /// the end hooks were cut off at the deadline.
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            drop_hook_future(&self.identity, self.hook, future);
        }
    }
}

/// Polls one hook run per plugin concurrently. Ready when every run
/// finished.
struct JoinHooks<'a> {
    runs: Vec<HookRun<'a>>,
}

impl<'a> JoinHooks<'a> {
    fn new(runs: Vec<HookRun<'a>>) -> Self {
        Self { runs }
    }
}

impl Future for JoinHooks<'_> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let mut finished = true;
        for run in &mut self.get_mut().runs {
            finished &= Pin::new(run).poll(cx).is_ready();
        }
        if finished {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

impl Drop for InvocationPlugins {
    /// Shuts the plugins down (see [`Self::shut_down`]), then drops every
    /// plugin, in registration order, each inside its own catch.
    ///
    /// A plugin's destructor is plugin code, and a plugin panic never
    /// changes the execution. So a destructor panic is logged and its
    /// payload is dropped through [`crate::task::discard_panic`]. Each
    /// plugin has its own catch, because a second panic while the first
    /// one unwinds a shared drop would abort the process.
    fn drop(&mut self) {
        self.shut_down();
        for PluginSlot { plugin, identity } in std::mem::take(&mut self.plugins) {
            if let Err(payload) = call_plugin(move || drop(plugin)) {
                warn_contained(|| {
                    tracing::warn!(
                        target: TARGET,
                        plugin = %identity,
                        hook = "drop",
                        "plugin panicked while it was dropped; the panic is ignored"
                    );
                });
                discard_plugin_panic(payload);
            }
        }
    }
}

/// Shuts the plugins down when it is dropped (see
/// [`InvocationPlugins::shut_down`]).
///
/// The invocation future holds one. A context clone can outlive the
/// invocation and keep the plugins alive, so their own `Drop` may run
/// late. This guard runs when the invocation future returns or is dropped
/// before it returned, so the hook tasks are aborted then. It holds a weak
/// reference, so it never keeps the plugins alive itself.
pub(crate) struct ShutDownOnDrop(Weak<InvocationPlugins>);

impl ShutDownOnDrop {
    pub(crate) fn new(plugins: &Arc<InvocationPlugins>) -> Self {
        Self(Arc::downgrade(plugins))
    }
}

impl Drop for ShutDownOnDrop {
    fn drop(&mut self) {
        if let Some(plugins) = self.0.upgrade() {
            plugins.shut_down();
        }
    }
}

/// The registered factories, in registration order, each with its type
/// name.
pub(crate) type Factories = [(&'static str, Arc<dyn PluginFactory>)];

/// Creates the plugins of one invocation.
///
/// With no factories it returns `None` and does not call `build_info`.
/// Otherwise it calls `build_info` once, then every factory once, in
/// order. A factory that returns `Ok(None)`, returns `Err` or panics is
/// skipped, and the others keep their order. It returns `Some` even when
/// every factory was skipped.
pub(crate) fn create_plugins(
    factories: &Factories,
    execution: Option<OperationInfo>,
    build_info: impl FnOnce() -> InvocationInfo,
) -> Option<Arc<InvocationPlugins>> {
    if factories.is_empty() {
        return None;
    }
    let info = build_info();
    let mut error_drop_panics = Vec::new();
    let plugins = create_slots(factories, &info, &mut error_drop_panics);
    Some(Arc::new(InvocationPlugins {
        info,
        plugins,
        execution: Mutex::new(execution),
        undelivered: Arc::new(Mutex::new(Undelivered::Kept(error_drop_panics))),
        work: Mutex::new(WorkState::Open(Vec::new())),
        change_queue: Mutex::new(std::collections::VecDeque::new()),
        draining: std::sync::atomic::AtomicBool::new(false),
        changes_pushed: std::sync::atomic::AtomicU64::new(0),
        delivery: Mutex::new(ChangeDelivery::default()),
        #[cfg(test)]
        after_release: Mutex::new(None),
    }))
}

/// Calls every factory once, in order, and keeps the plugins they return.
///
/// Each factory call runs inside `catch_unwind`.
///
/// - `Ok(Some(plugin))` is kept. `plugin.name()` runs inside its own catch,
///   and a panic there is logged as the hook `name` of `plugins[<i>]`,
///   which is then the plugin's identity.
/// - `Ok(None)` is skipped silently.
/// - `Err` is skipped and logged by [`log_factory_error`].
/// - A panic is skipped and logged.
///
/// A plugin panic does not fail the invocation, so every caught plugin
/// payload is dropped through [`crate::task::discard_panic`]. A panic of a
/// factory error's destructor fails the invocation, so its payload is
/// pushed onto `error_drop_panics` instead.
fn create_slots(
    factories: &Factories,
    info: &InvocationInfo,
    error_drop_panics: &mut Vec<Payload>,
) -> Vec<PluginSlot> {
    let mut slots = Vec::new();
    for (index, (type_name, factory)) in factories.iter().enumerate() {
        let factory_identity = format!("{type_name} (plugins[{index}])");
        match call_plugin(|| factory.create_plugin(info)) {
            Ok(Ok(Some(plugin))) => {
                let plugin: Arc<dyn Plugin> = Arc::from(plugin);
                let identity = match call_plugin(|| plugin.name()) {
                    Ok(name) => format!("{name} (plugins[{index}])"),
                    Err(payload) => {
                        let identity = format!("plugins[{index}]");
                        log_hook_panic(&identity, "name", payload);
                        identity
                    }
                };
                slots.push(PluginSlot { plugin, identity });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => log_factory_error(&factory_identity, error, error_drop_panics),
            Err(payload) => {
                warn_contained(|| {
                    tracing::warn!(
                        target: TARGET,
                        factory = %factory_identity,
                        "plugin factory panicked; the plugin is skipped for this invocation"
                    );
                });
                discard_plugin_panic(payload);
            }
        }
    }
    slots
}

/// Logs a factory's error and drops it.
///
/// 1. The error's `Display` is user code. So it runs inside a catch that
///    only borrows the error. A panic there is a plugin panic: it is
///    logged as a warning, and its payload goes to
///    [`crate::task::discard_panic`]. The error is not dropped while that
///    panic unwinds, because the catch does not own it.
/// 2. The SDK delivers the error to nobody, so it then drops the error
///    inside a second catch. A panic from its destructor is the panic of
///    an undelivered value, which fails the invocation. So its payload is
///    pushed onto `error_drop_panics`, which the invocation records on its
///    coordinator.
fn log_factory_error(
    factory_identity: &str,
    error: BoxError,
    error_drop_panics: &mut Vec<Payload>,
) {
    match call_plugin(|| error.to_string()) {
        Ok(message) => warn_contained(|| {
            tracing::warn!(
                target: TARGET,
                factory = %factory_identity,
                error = %message,
                "plugin factory failed; the plugin is skipped for this invocation"
            );
        }),
        Err(payload) => {
            warn_contained(|| {
                tracing::warn!(
                    target: TARGET,
                    factory = %factory_identity,
                    "plugin factory failed, and its error panicked while it was formatted; \
                     the plugin is skipped for this invocation"
                );
            });
            discard_plugin_panic(payload);
        }
    }
    if let Err(payload) = call_plugin(move || drop(error)) {
        error_drop_panics.push(payload);
    }
}

/// Plugins built from closures, and a `wrap` service that runs them, for
/// the hook tests of this module and of `lib.rs`.
/// The info of a first invocation with no operations, for tests.
#[cfg(test)]
pub(crate) fn tests_support_info() -> InvocationInfo {
    InvocationInfo::new(
        "r".to_owned(),
        "arn".to_owned(),
        "{}".to_owned(),
        BTreeMap::new(),
        Vec::new(),
        None,
        true,
        None,
    )
}

#[cfg(test)]
impl OperationInfo {
    /// An info that holds only the ID `id`, for tests.
    pub(crate) fn for_test(id: &str) -> Self {
        Self {
            id: id.to_owned(),
            name: None,
            kind: None,
            type_str: None,
            sub_type: None,
            parent_id: None,
            status: None,
            status_str: None,
            start_timestamp: None,
            end_timestamp: None,
            result: None,
            error: None,
            attempt: None,
            is_replay: false,
            is_virtual: false,
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::future::Future;
    use std::sync::{Arc, Mutex};

    use super::{HookFuture, InvocationEndInfo, InvocationInfo, Plugin, PluginFactory, TARGET};
    use crate::{BoxError, DurableContext, InvocationPayload, Options};

    type StartFn = dyn Fn(&InvocationInfo) -> HookFuture<'static> + Send + Sync;
    type EndFn = dyn Fn(&InvocationEndInfo) -> HookFuture<'static> + Send + Sync;

    /// A plugin whose hooks are closures. A hook without a closure does
    /// nothing. Every info it receives is recorded.
    #[derive(Clone, Default)]
    pub(crate) struct Hooks {
        pub(crate) name: Option<&'static str>,
        pub(crate) start: Option<Arc<StartFn>>,
        pub(crate) end: Option<Arc<EndFn>>,
        pub(crate) starts: Arc<Mutex<Vec<InvocationInfo>>>,
        pub(crate) ends: Arc<Mutex<Vec<InvocationEndInfo>>>,
    }

    impl Hooks {
        pub(crate) fn on_start(
            mut self,
            hook: impl Fn(&InvocationInfo) -> HookFuture<'static> + Send + Sync + 'static,
        ) -> Self {
            self.start = Some(Arc::new(hook));
            self
        }

        pub(crate) fn on_end(
            mut self,
            hook: impl Fn(&InvocationEndInfo) -> HookFuture<'static> + Send + Sync + 'static,
        ) -> Self {
            self.end = Some(Arc::new(hook));
            self
        }

        pub(crate) fn named(mut self, name: &'static str) -> Self {
            self.name = Some(name);
            self
        }

        pub(crate) fn ends(&self) -> Vec<InvocationEndInfo> {
            self.ends.lock().unwrap().clone()
        }
    }

    impl Plugin for Hooks {
        fn name(&self) -> &'static str {
            self.name.unwrap_or("Hooks")
        }

        fn on_invocation_start<'a>(&'a self, info: &'a InvocationInfo) -> HookFuture<'a> {
            self.starts.lock().unwrap().push(info.clone());
            match &self.start {
                Some(hook) => hook(info),
                None => Box::pin(std::future::ready(Ok(()))),
            }
        }

        fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
            self.ends.lock().unwrap().push(info.clone());
            match &self.end {
                Some(hook) => hook(info),
                None => Box::pin(std::future::ready(Ok(()))),
            }
        }
    }

    impl PluginFactory for Hooks {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    type ChangeFn = dyn Fn(&super::OperationChangeInfo) + Send + Sync;

    /// A plugin that records every [`super::OperationChangeInfo`] its
    /// `on_operation_change` receives, then runs its closure, if any. With
    /// `panics`, the hook panics instead and records nothing.
    ///
    /// Its timeline records, in call order, `change:<name>:<status>` for
    /// each updated operation of a change call (its wire status token), `start:<name>` and
    /// `end:<name>` for the operation hooks, and `invocation_end:<status>`.
    /// An operation without a name is named by its wire ID.
    #[derive(Clone, Default)]
    pub(crate) struct Changes {
        pub(crate) calls: Arc<Mutex<Vec<super::OperationChangeInfo>>>,
        pub(crate) timeline: Arc<Mutex<Vec<String>>>,
        pub(crate) on_change: Option<Arc<ChangeFn>>,
        pub(crate) panics: bool,
    }

    impl Changes {
        /// The infos recorded so far.
        pub(crate) fn calls(&self) -> Vec<super::OperationChangeInfo> {
            self.calls.lock().unwrap().clone()
        }

        /// The timeline so far.
        pub(crate) fn timeline(&self) -> Vec<String> {
            self.timeline.lock().unwrap().clone()
        }

        /// The plugins of an invocation whose one plugin is this recorder.
        pub(crate) fn plugins(&self) -> Arc<super::InvocationPlugins> {
            plugins_of(self.clone())
        }

        fn label(info: &super::OperationInfo) -> String {
            info.name().unwrap_or(info.id()).to_owned()
        }

        /// The updated wire IDs of each recorded call.
        pub(crate) fn updated_ids(&self) -> Vec<Vec<String>> {
            self.calls()
                .iter()
                .map(|info| {
                    info.updated_operations()
                        .iter()
                        .map(|op| op.id().to_owned())
                        .collect()
                })
                .collect()
        }
    }

    impl Plugin for Changes {
        fn name(&self) -> &'static str {
            "Changes"
        }

        fn on_operation_change(&self, info: &super::OperationChangeInfo) -> super::HookWork {
            assert!(!self.panics, "change hook panicked");
            self.calls.lock().unwrap().push(info.clone());
            {
                let mut timeline = self.timeline.lock().unwrap();
                for op in info.updated_operations() {
                    timeline.push(format!(
                        "change:{}:{}",
                        Self::label(op),
                        op.status_str().unwrap_or_default()
                    ));
                }
            }
            if let Some(hook) = &self.on_change {
                hook(info);
            }
            Ok(None)
        }

        fn on_operation_start(&self, info: &super::OperationInfo) -> super::HookWork {
            let label = format!("start:{}", Self::label(info));
            self.timeline.lock().unwrap().push(label);
            Ok(None)
        }

        fn on_operation_end(&self, info: &super::OperationInfo) -> super::HookWork {
            let label = format!("end:{}", Self::label(info));
            self.timeline.lock().unwrap().push(label);
            Ok(None)
        }

        fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
            let label = format!("invocation_end:{:?}", info.status());
            self.timeline.lock().unwrap().push(label);
            Box::pin(std::future::ready(Ok(())))
        }
    }

    impl PluginFactory for Changes {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    /// The plugins of an invocation with one plugin per factory, in order.
    pub(crate) fn plugins_of_all(
        factories: Vec<Arc<dyn PluginFactory>>,
    ) -> Arc<super::InvocationPlugins> {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> = factories
            .into_iter()
            .map(|factory| ("test::Plugin", factory))
            .collect();
        super::create_plugins(&factories, None, empty_info).expect("a factory is registered")
    }

    /// One operation or attempt hook call that [`OpHooks`] recorded.
    #[derive(Clone, Debug)]
    pub(crate) struct OpRecord {
        pub(crate) hook: &'static str,
        /// The operation's name.
        pub(crate) name: Option<String>,
        /// The operation's parent ID.
        pub(crate) parent_id: Option<String>,
        pub(crate) replay: bool,
        /// The info's `is_virtual()`.
        pub(crate) is_virtual: bool,
        pub(crate) status: Option<crate::OperationStatus>,
        /// For an attempt hook, its attempt number.
        pub(crate) attempt: Option<u32>,
        /// The operation info's `attempt()`.
        pub(crate) info_attempt: Option<u32>,
        /// For an attempt-end hook, its outcome.
        pub(crate) outcome: Option<super::AttemptOutcome>,
        pub(crate) result: Option<String>,
        /// For an attempt-end hook, its error message.
        pub(crate) error: Option<String>,
        /// What the probe returned when the hook ran.
        pub(crate) probe: String,
    }

    impl OpRecord {
        /// `(hook, replay, attempt, outcome)`.
        pub(crate) fn shape(
            &self,
        ) -> (
            &'static str,
            bool,
            Option<u32>,
            Option<super::AttemptOutcome>,
        ) {
            (self.hook, self.replay, self.attempt, self.outcome)
        }
    }

    type Probe = dyn Fn() -> String + Send + Sync;

    /// A plugin that records every operation and attempt hook call, with
    /// what its probe returns at the call.
    #[derive(Clone, Default)]
    pub(crate) struct OpHooks {
        pub(crate) records: Arc<Mutex<Vec<OpRecord>>>,
        pub(crate) probe: Option<Arc<Probe>>,
    }

    impl OpHooks {
        pub(crate) fn probing(probe: impl Fn() -> String + Send + Sync + 'static) -> Self {
            Self {
                probe: Some(Arc::new(probe)),
                ..Self::default()
            }
        }

        pub(crate) fn records(&self) -> Vec<OpRecord> {
            self.records.lock().unwrap().clone()
        }

        /// The records of the operation named `name`.
        pub(crate) fn of(&self, name: &str) -> Vec<OpRecord> {
            self.records()
                .into_iter()
                .filter(|r| r.name.as_deref() == Some(name))
                .collect()
        }

        fn push(
            &self,
            hook: &'static str,
            info: &super::OperationInfo,
            attempt: Option<u32>,
            end: Option<&super::AttemptEndInfo>,
        ) {
            let probe = self.probe.as_ref().map(|probe| probe()).unwrap_or_default();
            self.records.lock().unwrap().push(OpRecord {
                hook,
                name: info.name().map(str::to_owned),
                parent_id: info.parent_id().map(str::to_owned),
                replay: info.is_replay(),
                is_virtual: info.is_virtual(),
                status: info.status(),
                info_attempt: info.attempt(),
                attempt,
                outcome: end.map(super::AttemptEndInfo::outcome),
                result: info.result().map(str::to_owned),
                error: end
                    .and_then(super::AttemptEndInfo::error)
                    .and_then(crate::WireError::error_message)
                    .map(str::to_owned),
                probe,
            });
        }

        /// The plugins of an invocation whose one plugin is this recorder.
        pub(crate) fn plugins(&self) -> Arc<super::InvocationPlugins> {
            let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> =
                vec![("test::OpHooks", Arc::new(self.clone()))];
            super::create_plugins(&factories, None, empty_info).expect("one factory is registered")
        }
    }

    impl Plugin for OpHooks {
        fn on_operation_start(&self, info: &super::OperationInfo) -> super::HookWork {
            self.push("start", info, None, None);
            Ok(None)
        }

        fn on_operation_end(&self, info: &super::OperationInfo) -> super::HookWork {
            self.push("end", info, None, None);
            Ok(None)
        }

        fn on_attempt_start(&self, info: &super::AttemptInfo) -> super::HookWork {
            self.push(
                "attempt_start",
                info.operation(),
                Some(info.attempt()),
                None,
            );
            Ok(None)
        }

        fn on_attempt_end(&self, info: &super::AttemptEndInfo) -> super::HookWork {
            self.push(
                "attempt_end",
                info.operation(),
                Some(info.attempt()),
                Some(info),
            );
            Ok(None)
        }
    }

    impl PluginFactory for OpHooks {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    type TraceFn =
        dyn Fn(&super::PropagationInput) -> Result<Option<String>, BoxError> + Send + Sync;

    /// A plugin whose `propagation_trace_id` answers with a closure. Every
    /// input it receives is recorded.
    #[derive(Clone)]
    pub(crate) struct TraceHook {
        answer: Arc<TraceFn>,
        inputs: Arc<Mutex<Vec<super::PropagationInput>>>,
    }

    impl TraceHook {
        pub(crate) fn new(
            answer: impl Fn(&super::PropagationInput) -> Result<Option<String>, BoxError>
            + Send
            + Sync
            + 'static,
        ) -> Self {
            Self {
                answer: Arc::new(answer),
                inputs: Arc::default(),
            }
        }

        /// A plugin that answers `Ok(value)`.
        pub(crate) fn value(value: Option<&'static str>) -> Self {
            Self::new(move |_| Ok(value.map(str::to_owned)))
        }

        /// The inputs the hook received, in call order.
        pub(crate) fn inputs(&self) -> Vec<super::PropagationInput> {
            self.inputs.lock().unwrap().clone()
        }
    }

    impl Plugin for TraceHook {
        fn name(&self) -> &'static str {
            "TraceHook"
        }

        fn propagation_trace_id(
            &self,
            input: &super::PropagationInput,
        ) -> Result<Option<String>, BoxError> {
            self.inputs.lock().unwrap().push(input.clone());
            (self.answer)(input)
        }
    }

    impl PluginFactory for TraceHook {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    /// The plugins of an invocation whose plugins are `hooks`, in order.
    pub(crate) fn trace_plugins(hooks: &[TraceHook]) -> Arc<super::InvocationPlugins> {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> = hooks
            .iter()
            .map(|hook| {
                (
                    "test::TraceHook",
                    Arc::new(hook.clone()) as Arc<dyn PluginFactory>,
                )
            })
            .collect();
        super::create_plugins(&factories, None, empty_info).expect("a factory is registered")
    }

    /// The info of a first invocation with no operations.
    fn empty_info() -> InvocationInfo {
        InvocationInfo::new(
            "r".to_owned(),
            "arn".to_owned(),
            "{}".to_owned(),
            std::collections::BTreeMap::new(),
            Vec::new(),
            None,
            true,
            None,
        )
    }

    /// The plugins of an invocation whose one plugin `factory` builds.
    pub(crate) fn plugins_of(
        factory: impl PluginFactory + 'static,
    ) -> Arc<super::InvocationPlugins> {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> =
            vec![("test::Plugin", Arc::new(factory))];
        super::create_plugins(&factories, None, empty_info).expect("one factory is registered")
    }

    /// The plugins of an invocation with one [`Hooks`] plugin, as
    /// `create_plugins` returns them.
    pub(crate) fn invocation_plugins() -> Arc<super::InvocationPlugins> {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> =
            vec![("test::Hooks", Arc::new(Hooks::default()))];
        super::create_plugins(&factories, None, empty_info).expect("one factory is registered")
    }

    /// The deadline `millis` from now, in epoch milliseconds, as a Lambda
    /// context carries it.
    pub(crate) fn deadline_in(millis: u64) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        u64::try_from(now.as_millis()).unwrap() + millis
    }

    /// A `wrap` service with the in-memory transport and `plugins`, and the
    /// event of a first invocation whose Lambda deadline is `deadline`
    /// (epoch milliseconds, 0 for none).
    pub(crate) fn service<F, Fut>(
        plugins: Vec<Hooks>,
        deadline: u64,
        handler: F,
    ) -> (
        impl Fn(lambda_runtime::LambdaEvent<InvocationPayload>) -> crate::BoxedInvocationFuture,
        lambda_runtime::LambdaEvent<InvocationPayload>,
    )
    where
        F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, BoxError>> + Send,
    {
        service_with(plugins, deadline, Vec::new(), handler)
    }

    /// [`service`], with `recorded` added to the invocation's operations
    /// after the `EXECUTION` operation.
    pub(crate) fn service_with<F, Fut>(
        plugins: Vec<Hooks>,
        deadline: u64,
        recorded: Vec<serde_json::Value>,
        handler: F,
    ) -> (
        impl Fn(lambda_runtime::LambdaEvent<InvocationPayload>) -> crate::BoxedInvocationFuture,
        lambda_runtime::LambdaEvent<InvocationPayload>,
    )
    where
        F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, BoxError>> + Send,
    {
        service_configured(
            move |mut builder| {
                for plugin in plugins {
                    builder = builder.plugin(plugin);
                }
                builder
            },
            deadline,
            recorded,
            handler,
        )
    }

    /// [`service_with`], whose plugins `configure` registers on the
    /// builder.
    pub(crate) fn service_configured<F, Fut>(
        configure: impl FnOnce(crate::options::OptionsBuilder) -> crate::options::OptionsBuilder,
        deadline: u64,
        recorded: Vec<serde_json::Value>,
        handler: F,
    ) -> (
        impl Fn(lambda_runtime::LambdaEvent<InvocationPayload>) -> crate::BoxedInvocationFuture,
        lambda_runtime::LambdaEvent<InvocationPayload>,
    )
    where
        F: Fn(serde_json::Value, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, BoxError>> + Send,
    {
        let builder = configure(Options::builder().execution_client(Arc::new(
            crate::client::InMemoryExecutionClient::new(Vec::new()),
        )));
        let mut operations = vec![serde_json::json!({
            "Id": "exec-1",
            "Type": "EXECUTION",
            "Status": "STARTED",
            "ExecutionDetails": { "InputPayload": "{}" }
        })];
        operations.extend(recorded);
        let payload = serde_json::json!({
            "DurableExecutionArn": "arn:test",
            "CheckpointToken": "tok",
            "InitialExecutionState": { "Operations": operations }
        });
        let mut lambda_ctx = lambda_runtime::Context::default();
        lambda_ctx.deadline = deadline;
        let event = lambda_runtime::LambdaEvent::new(InvocationPayload(payload), lambda_ctx);
        (crate::wrap(handler, builder.build().unwrap()), event)
    }

    /// Captures the events of this thread as JSON lines.
    pub(crate) fn capture() -> (Arc<Mutex<Vec<u8>>>, tracing::subscriber::DefaultGuard) {
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
        let guard =
            test_tracing::install_test_subscriber(tracing_subscriber::registry().with(layer));
        (buffer, guard)
    }

    /// The warnings on the plugin target that `buffer` captured.
    pub(crate) fn warnings(buffer: &Arc<Mutex<Vec<u8>>>) -> Vec<serde_json::Value> {
        String::from_utf8_lossy(&buffer.lock().unwrap())
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|v| {
                v.get("target").and_then(serde_json::Value::as_str) == Some(TARGET)
                    && v.get("level").and_then(serde_json::Value::as_str) == Some("WARN")
            })
            .collect()
    }

    /// The string field `name` of `event`, or `""`.
    pub(crate) fn field<'a>(event: &'a serde_json::Value, name: &str) -> &'a str {
        event
            .get(name)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
    }

    /// The warnings whose message is `message`.
    pub(crate) fn with_message<'a>(
        warnings: &'a [serde_json::Value],
        message: &str,
    ) -> Vec<&'a serde_json::Value> {
        warnings
            .iter()
            .filter(|w| field(w, "message") == message)
            .collect()
    }

    /// A waker that counts its wakes and signals each one on a channel, so
    /// a test sees a wake without polling again.
    pub(crate) struct SignalWaker {
        woken: std::sync::atomic::AtomicUsize,
        signal: Mutex<std::sync::mpsc::Sender<()>>,
    }

    impl SignalWaker {
        /// A waker and the receiver of its wake signals.
        pub(crate) fn new() -> (Arc<Self>, std::sync::mpsc::Receiver<()>) {
            let (signal, signals) = std::sync::mpsc::channel();
            let waker = Arc::new(Self {
                woken: std::sync::atomic::AtomicUsize::new(0),
                signal: Mutex::new(signal),
            });
            (waker, signals)
        }

        /// How often the waker was woken.
        pub(crate) fn woken(&self) -> usize {
            self.woken.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl std::task::Wake for SignalWaker {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.woken.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _receiver_gone = self.signal.lock().unwrap().send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    /// A record of `op_type` with `status` and every other field empty.
    fn record(op_type: &str, status: CheckpointStatus) -> CheckpointRecord {
        let mut record = CheckpointRecord::with_status("id", status);
        record.op_type = Some(op_type.to_owned());
        record
    }

    /// The info of a started step record with no timestamps.
    fn bare_info() -> OperationInfo {
        OperationInfo::from_record("a", &record("STEP", CheckpointStatus::Started))
    }

    /// A plugin whose `name()` panics is kept under the identity
    /// `plugins[<i>]`, the panic is logged as one contained WARN, and the
    /// plugin's hooks still run. The scenario retries, as the other capture
    /// tests do, because a concurrent subscriber change can drop a warning.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    fn name_panic_is_logged() {
        struct BadName(Arc<AtomicUsize>);
        impl Plugin for BadName {
            #[expect(clippy::panic, reason = "the test plugin's name panics")]
            fn name(&self) -> &'static str {
                panic!("name panicked")
            }
            fn on_user_function_start(&self, _info: &OperationInfo) -> HookWork {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(None)
            }
        }
        struct Factory(Arc<AtomicUsize>);
        impl PluginFactory for Factory {
            fn create_plugin(
                &self,
                _info: &InvocationInfo,
            ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
                Ok(Some(Box::new(BadName(Arc::clone(&self.0)))))
            }
        }
        for _attempt in 0..5 {
            let (buffer, guard) = capture();
            let calls = Arc::new(AtomicUsize::new(0));
            let factory: Arc<dyn PluginFactory> = Arc::new(Factory(Arc::clone(&calls)));
            let plugins = create_plugins(&[("Factory", factory)], None, tests_support_info)
                .expect("factories were given");
            assert_eq!(
                plugins.plugins.first().map(|s| s.identity.as_str()),
                Some("plugins[0]")
            );
            let started = plugins.dispatch_to("on_user_function_start", [0], |plugin| {
                plugin.on_user_function_start(&OperationInfo::for_test("s"))
            });
            assert_eq!(started, [0], "the start hook still runs and succeeds");
            assert_eq!(calls.load(Ordering::SeqCst), 1);
            drop(guard);
            let warnings = warnings(&buffer);
            let [warning] = warnings.as_slice() else {
                continue;
            };
            assert_eq!(field(warning, "plugin"), "plugins[0]");
            assert_eq!(field(warning, "hook"), "name");
            assert_eq!(
                field(warning, "message"),
                "plugin hook panicked; the panic is ignored"
            );
            return;
        }
        panic!("no attempt captured exactly one name() warning");
    }

    /// A hook error whose `Display` calls a durable operation on a captured
    /// context records a context usage error: the `Display` is plugin code.
    #[test]
    fn a_hook_errors_display_calling_an_operation_fails_the_execution() {
        #[derive(Debug)]
        struct Calls(DurableContext);
        impl fmt::Display for Calls {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                drop(self.0.claim_operation_id("wait"));
                f.write_str("calls")
            }
        }
        impl std::error::Error for Calls {}
        struct Failing(DurableContext);
        impl Plugin for Failing {
            fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
                Err(Box::new(Calls(self.0.clone())))
            }
        }
        struct Factory(DurableContext);
        impl PluginFactory for Factory {
            fn create_plugin(
                &self,
                _info: &InvocationInfo,
            ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
                Ok(Some(Box::new(Failing(self.0.clone()))))
            }
        }
        let ctx = DurableContext::test_context();
        let factory: Arc<dyn PluginFactory> = Arc::new(Factory(ctx.clone()));
        let plugins = create_plugins(&[("Factory", factory)], None, tests_support_info)
            .expect("factories were given");
        plugins.dispatch_work("on_operation_start", |plugin| {
            plugin.on_operation_start(&OperationInfo::for_test("s"))
        });
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert!(
            fatal.error_message.contains("inside plugin code"),
            "{}",
            fatal.error_message
        );
    }

    /// A hook's panic payload whose destructor calls a durable operation on
    /// a captured context records a context usage error: the payload is
    /// plugin code, so the SDK disposes of it as a plugin body.
    #[test]
    fn a_panic_payloads_destructor_calling_an_operation_fails_the_execution() {
        struct Calls(DurableContext);
        impl Drop for Calls {
            fn drop(&mut self) {
                drop(self.0.claim_operation_id("wait"));
            }
        }
        struct Panicking(DurableContext);
        impl Plugin for Panicking {
            #[expect(clippy::panic, reason = "the test hook panics with a payload")]
            fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
                std::panic::panic_any(Calls(self.0.clone()))
            }
        }
        struct Factory(DurableContext);
        impl PluginFactory for Factory {
            fn create_plugin(
                &self,
                _info: &InvocationInfo,
            ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
                Ok(Some(Box::new(Panicking(self.0.clone()))))
            }
        }
        let ctx = DurableContext::test_context();
        let factory: Arc<dyn PluginFactory> = Arc::new(Factory(ctx.clone()));
        let plugins = create_plugins(&[("Factory", factory)], None, tests_support_info)
            .expect("factories were given");
        plugins.dispatch_work("on_operation_start", |plugin| {
            plugin.on_operation_start(&OperationInfo::for_test("s"))
        });
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert!(
            fatal.error_message.contains("inside plugin code"),
            "{}",
            fatal.error_message
        );
    }

    /// Two factory errors whose destructors panic before the coordinator
    /// is attached: the coordinator keeps the first payload and rejects
    /// the second. The rejected payload is plugin code, so its destructor
    /// runs under `ActiveBody::Plugin`, and the durable operation it calls
    /// records the context usage error that names plugin code.
    #[test]
    fn a_rejected_kept_payloads_destructor_calling_an_operation_fails_the_execution() {
        struct Calls(DurableContext);
        impl Drop for Calls {
            fn drop(&mut self) {
                drop(self.0.claim_operation_id("wait"));
            }
        }
        #[derive(Debug)]
        struct DropPanics(Option<DurableContext>);
        impl fmt::Display for DropPanics {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("factory failed")
            }
        }
        impl std::error::Error for DropPanics {}
        impl Drop for DropPanics {
            #[expect(clippy::panic, reason = "the destructor panic under test")]
            fn drop(&mut self) {
                match self.0.take() {
                    Some(ctx) => std::panic::panic_any(Calls(ctx)),
                    None => panic!("first error drop"),
                }
            }
        }
        struct Factory(Option<DurableContext>);
        impl PluginFactory for Factory {
            fn create_plugin(
                &self,
                _info: &InvocationInfo,
            ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
                Err(Box::new(DropPanics(self.0.clone())))
            }
        }
        let ctx = DurableContext::test_context();
        let first: Arc<dyn PluginFactory> = Arc::new(Factory(None));
        let second: Arc<dyn PluginFactory> = Arc::new(Factory(Some(ctx.clone())));
        let plugins = create_plugins(
            &[("First", first), ("Second", second)],
            None,
            tests_support_info,
        )
        .expect("factories were given");
        assert!(ctx.coordinator().fatal_error().is_none());
        plugins.attach(ctx.coordinator());
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert!(
            fatal.error_message.contains("inside plugin code"),
            "{}",
            fatal.error_message
        );
    }

    /// A `tracing` subscriber that calls `ctx.step(..)` on each event of
    /// the plugin target runs under `ActiveBody::Plugin` while the SDK logs
    /// a failed start hook, so the call records the context usage error
    /// and the execution fails with it. The scenario retries when the
    /// warning was dropped by a concurrent subscriber change.
    #[tokio::test]
    #[expect(clippy::panic)] // reason: reports that no attempt reached the subscriber
    async fn subscriber_in_plugin_warning_calls_durable_op() {
        type Slot = Arc<Mutex<Option<DurableContext>>>;
        struct Calling(Slot);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Calling {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if event.metadata().target() == TARGET
                    && let Some(ctx) = self.0.lock().unwrap().clone()
                {
                    drop(ctx.step(|_| async { Ok::<u32, BoxError>(1) }));
                }
            }
        }
        struct FailingStart;
        impl Plugin for FailingStart {
            fn on_user_function_start(&self, _info: &OperationInfo) -> HookWork {
                Err("start failed".into())
            }
        }
        impl PluginFactory for FailingStart {
            fn create_plugin(
                &self,
                _info: &InvocationInfo,
            ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
                Ok(Some(Box::new(FailingStart)))
            }
        }
        use tracing_subscriber::layer::SubscriberExt as _;
        for _attempt in 0..5 {
            let slot: Slot = Arc::default();
            let _guard = test_tracing::install_test_subscriber(
                tracing_subscriber::registry().with(Calling(Arc::clone(&slot))),
            );
            let (service, event) = test_support::service_configured(
                |builder| builder.plugin(FailingStart),
                0,
                Vec::new(),
                move |_e, ctx| {
                    *slot.lock().unwrap() = Some(ctx.clone());
                    async move {
                        let value = ctx.step(|_| async { Ok(7_u32) }).name("s").await?;
                        Ok(serde_json::json!(value))
                    }
                },
            );
            let json = service(event).await.unwrap().0;
            if json.get("Status") == Some(&serde_json::json!("SUCCEEDED")) {
                continue;
            }
            assert_eq!(json.get("Status"), Some(&serde_json::json!("FAILED")));
            assert!(json.to_string().contains("inside plugin code"), "{json}");
            return;
        }
        panic!("no attempt delivered the warning to the subscriber");
    }

    #[test]
    fn for_operation_without_record_holds_the_identity() {
        let ctx = DurableContext::test_context();
        let info = OperationInfo::for_operation(&ctx, "w", OperationKind::Step, "Pay", Some("a"))
            .with_status(OperationStatus::Started);
        assert_eq!(info.id(), "w");
        assert_eq!(info.kind(), Some(OperationKind::Step));
        assert_eq!(info.type_str(), None);
        assert_eq!(info.sub_type(), Some("Pay"));
        assert_eq!(info.name(), Some("a"));
        assert_eq!(info.parent_id(), None);
        assert_eq!(info.status(), Some(OperationStatus::Started));
        assert_eq!(info.status_str(), None);
        assert_eq!(info.start_timestamp(), None);
        assert_eq!(info.attempt(), None);
    }

    #[test]
    fn with_virtual_sets_the_flag() {
        let ctx = DurableContext::test_context();
        let info = OperationInfo::for_operation(&ctx, "w", OperationKind::Context, "Map", None);
        assert!(!info.is_virtual());
        let info = info.with_virtual(true);
        assert!(info.is_virtual());
        assert!(!info.with_virtual(false).is_virtual());
    }

    /// Without a record, the parent comes from the context the operation
    /// was created in.
    #[test]
    fn for_operation_without_record_takes_the_context_parent() {
        let ctx = DurableContext::test_context().new_child("1", None);
        let parent = ctx
            .parent_wire_id()
            .expect("a child has a parent")
            .to_owned();
        let info = OperationInfo::for_operation(&ctx, "w", OperationKind::Step, "Step", None);
        assert_eq!(info.parent_id(), Some(parent.as_str()));
        assert_eq!(info.name(), None);
    }

    /// A field the record populates keeps the record's value, and only the
    /// absent identity fields come from the arguments.
    #[test]
    fn for_operation_keeps_populated_record_fields() {
        let mut rec = record("STEP", CheckpointStatus::Started);
        rec.op_name = Some("recorded".to_owned());
        rec.sub_type = Some("Recorded".to_owned());
        rec.parent_id = Some("p".to_owned());
        let ctx = DurableContext::new_root(
            String::from("arn"),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(vec![("w".to_owned(), rec)])),
        );
        let info = OperationInfo::for_operation(
            &ctx.new_child("1", None),
            "w",
            OperationKind::Wait,
            "Pay",
            Some("a"),
        )
        .with_status(OperationStatus::Succeeded);
        assert_eq!(info.kind(), Some(OperationKind::Step));
        assert_eq!(info.type_str(), Some("STEP"));
        assert_eq!(info.sub_type(), Some("Recorded"));
        assert_eq!(info.name(), Some("recorded"));
        assert_eq!(info.parent_id(), Some("p"));
        assert_eq!(info.status(), Some(OperationStatus::Succeeded));
        assert_eq!(info.status_str(), Some("STARTED"));
    }

    /// A record without a type, sub-type or name gets them from the
    /// arguments.
    #[test]
    fn for_operation_fills_absent_record_fields() {
        let rec = CheckpointRecord::with_status("w", CheckpointStatus::Started);
        let ctx = DurableContext::new_root(
            String::from("arn"),
            lambda_runtime::Context::default(),
            Arc::new(CheckpointLog::from_records(vec![("w".to_owned(), rec)])),
        );
        let info = OperationInfo::for_operation(&ctx, "w", OperationKind::Step, "Pay", Some("a"));
        assert_eq!(info.kind(), Some(OperationKind::Step));
        assert_eq!(info.type_str(), None);
        assert_eq!(info.sub_type(), Some("Pay"));
        assert_eq!(info.name(), Some("a"));
        assert_eq!(info.parent_id(), None);
        assert_eq!(info.status_str(), Some("STARTED"));
    }

    #[test]
    fn observed_end_keeps_the_recorded_timestamp() {
        let recorded = UNIX_EPOCH + Duration::from_secs(5);
        let info = with_observed_end(bare_info().with_end_timestamp(recorded));
        assert_eq!(info.end_timestamp(), Some(recorded));
    }

    #[test]
    fn observed_end_uses_local_time_without_record() {
        let info = bare_info();
        assert_eq!(info.end_timestamp(), None);
        let before = SystemTime::now();
        let info = with_observed_end(info);
        let after = SystemTime::now();
        let end = info.end_timestamp().expect("an end timestamp");
        assert!(before <= end && end <= after, "{end:?}");
    }

    #[test]
    fn operation_info_from_record_maps_every_field() {
        let mut rec = record("STEP", CheckpointStatus::Succeeded);
        rec.sub_type = Some("Step".to_owned());
        rec.op_name = Some("n".to_owned());
        rec.parent_id = Some("p".to_owned());
        rec.result = Some("\"x\"".to_owned());
        rec.attempt = 2;
        rec.start_timestamp = Some(UNIX_EPOCH + Duration::from_secs(1));
        rec.end_timestamp = Some(UNIX_EPOCH + Duration::from_secs(2));
        let info = OperationInfo::from_record("op-1", &rec);
        assert_eq!(info.id(), "op-1");
        assert_eq!(info.name(), Some("n"));
        assert_eq!(info.kind(), Some(OperationKind::Step));
        assert_eq!(info.type_str(), Some("STEP"));
        assert_eq!(info.sub_type(), Some("Step"));
        assert_eq!(info.parent_id(), Some("p"));
        assert_eq!(info.status(), Some(OperationStatus::Succeeded));
        assert_eq!(info.status_str(), Some("SUCCEEDED"));
        assert_eq!(
            info.start_timestamp(),
            Some(UNIX_EPOCH + Duration::from_secs(1))
        );
        assert_eq!(
            info.end_timestamp(),
            Some(UNIX_EPOCH + Duration::from_secs(2))
        );
        assert_eq!(info.result(), Some("\"x\""));
        assert!(info.error().is_none());
        assert_eq!(info.attempt(), Some(2));
        assert!(!info.is_replay());
        assert!(!info.is_virtual());
    }

    #[test]
    fn operation_info_reads_both_type_spellings() {
        for spelling in ["ChainedInvoke", "CHAINED_INVOKE"] {
            let info =
                OperationInfo::from_record("a", &record(spelling, CheckpointStatus::Started));
            assert_eq!(
                info.kind(),
                Some(OperationKind::ChainedInvoke),
                "{spelling}"
            );
        }
    }

    #[test]
    fn operation_info_unknown_type_and_status_are_none() {
        let info = OperationInfo::from_record(
            "a",
            &record("BOGUS", CheckpointStatus::Unknown("NEW".to_owned())),
        );
        assert_eq!(info.kind(), None);
        assert_eq!(info.status(), None);
        assert_eq!(info.type_str(), Some("BOGUS"));
        assert_eq!(info.status_str(), Some("NEW"));
    }

    /// A record without `op_type` has no kind and no type token.
    #[test]
    fn operation_info_without_type_has_no_kind() {
        let rec = CheckpointRecord::with_status("id", CheckpointStatus::Started);
        let info = OperationInfo::from_record("a", &rec);
        assert_eq!(info.kind(), None);
        assert_eq!(info.type_str(), None);
    }

    /// Every known stored status maps to its typed status by name.
    #[test]
    fn operation_info_maps_every_known_status() {
        for (stored, typed) in [
            (CheckpointStatus::Started, OperationStatus::Started),
            (CheckpointStatus::Pending, OperationStatus::Pending),
            (CheckpointStatus::Ready, OperationStatus::Ready),
            (CheckpointStatus::Succeeded, OperationStatus::Succeeded),
            (CheckpointStatus::Failed, OperationStatus::Failed),
            (CheckpointStatus::Cancelled, OperationStatus::Cancelled),
            (CheckpointStatus::TimedOut, OperationStatus::TimedOut),
            (CheckpointStatus::Stopped, OperationStatus::Stopped),
        ] {
            let info = OperationInfo::from_record("a", &record("STEP", stored));
            assert_eq!(info.status(), Some(typed));
            assert_eq!(info.status_str(), Some(typed.wire_str()));
        }
    }

    #[test]
    fn operation_info_error_absent_when_no_error_field() {
        let info = OperationInfo::from_record("a", &record("STEP", CheckpointStatus::Failed));
        assert!(info.error().is_none());
        assert_eq!(info.status(), Some(OperationStatus::Failed));
    }

    #[test]
    fn operation_info_error_kept_with_message_only() {
        let mut rec = record("STEP", CheckpointStatus::Failed);
        rec.error_message = Some(String::new());
        let info = OperationInfo::from_record("a", &rec);
        let error = info.error().expect("an error with an empty message");
        assert_eq!(error.error_message(), Some(""));
        assert_eq!(error.error_type(), None);
    }

    /// Each of the four error fields alone yields an error, and the
    /// error carries every field present.
    #[test]
    fn operation_info_error_kept_with_any_field() {
        let mut data_only = record("STEP", CheckpointStatus::Failed);
        data_only.error_data = Some("d".to_owned());
        let info = OperationInfo::from_record("a", &data_only);
        assert_eq!(info.error().and_then(WireError::error_data), Some("d"));

        let mut trace_only = record("STEP", CheckpointStatus::Failed);
        trace_only.stack_trace = Some(vec!["frame".to_owned()]);
        let info = OperationInfo::from_record("a", &trace_only);
        assert_eq!(
            info.error().map(WireError::stack_trace),
            Some(&["frame".to_owned()][..])
        );

        let mut type_only = record("STEP", CheckpointStatus::Failed);
        type_only.error_type = Some("T".to_owned());
        let info = OperationInfo::from_record("a", &type_only);
        assert_eq!(info.error().and_then(WireError::error_type), Some("T"));
    }

    #[test]
    fn operation_info_reads_invoke_payloads() {
        let mut rec = record("CHAINED_INVOKE", CheckpointStatus::Succeeded);
        rec.invoke_result = Some("\"r\"".to_owned());
        rec.invoke_error_type = Some("E".to_owned());
        let info = OperationInfo::from_record("a", &rec);
        assert_eq!(info.result(), Some("\"r\""));
        assert_eq!(info.error().and_then(WireError::error_type), Some("E"));
    }

    /// The record's own result and error take precedence over the invoke
    /// fields, and each invoke error field alone yields an error.
    #[test]
    fn operation_info_prefers_record_payloads_over_invoke() {
        let mut rec = record("STEP", CheckpointStatus::Failed);
        rec.result = Some("own".to_owned());
        rec.invoke_result = Some("invoke".to_owned());
        rec.error_type = Some("Own".to_owned());
        rec.invoke_error_type = Some("Invoke".to_owned());
        let info = OperationInfo::from_record("a", &rec);
        assert_eq!(info.result(), Some("own"));
        assert_eq!(info.error().and_then(WireError::error_type), Some("Own"));

        let mut message = record("CHAINED_INVOKE", CheckpointStatus::Failed);
        message.invoke_error_message = Some("m".to_owned());
        message.invoke_error_data = Some("d".to_owned());
        message.invoke_stack_trace = Some(vec!["f".to_owned()]);
        let error = OperationInfo::from_record("a", &message)
            .error()
            .cloned()
            .expect("invoke error");
        assert_eq!(error.error_message(), Some("m"));
        assert_eq!(error.error_data(), Some("d"));
        assert_eq!(error.stack_trace(), ["f".to_owned()]);
    }

    #[test]
    fn operation_info_attempt_zero_is_none() {
        let info = OperationInfo::from_record("a", &record("STEP", CheckpointStatus::Started));
        assert_eq!(info.attempt(), None);
    }

    /// The `EXECUTION` constructor rejects every other type and keeps the
    /// wire fields it is given.
    #[test]
    fn execution_info_requires_execution_type() {
        assert!(
            OperationInfo::execution("s", "STEP", &CheckpointStatus::Started, None, None, None)
                .is_none()
        );
        let start = UNIX_EPOCH + Duration::from_secs(3);
        let info = OperationInfo::execution(
            "e",
            "EXECUTION",
            &CheckpointStatus::Started,
            Some("p".to_owned()),
            Some(start),
            None,
        )
        .expect("an EXECUTION operation");
        assert_eq!(info.id(), "e");
        assert_eq!(info.kind(), Some(OperationKind::Execution));
        assert_eq!(info.type_str(), Some("EXECUTION"));
        assert_eq!(info.status(), Some(OperationStatus::Started));
        assert_eq!(info.status_str(), Some("STARTED"));
        assert_eq!(info.parent_id(), Some("p"));
        assert_eq!(info.start_timestamp(), Some(start));
        assert_eq!(info.end_timestamp(), None);
        assert_eq!(info.name(), None);
        assert_eq!(info.result(), None);
        assert_eq!(info.attempt(), None);
    }

    fn info_with(ops: &[&str], updated: &[&str]) -> InvocationInfo {
        let operations = ops
            .iter()
            .map(|id| {
                (
                    (*id).to_owned(),
                    OperationInfo::from_record(id, &record("STEP", CheckpointStatus::Started)),
                )
            })
            .collect();
        InvocationInfo::new(
            "r".to_owned(),
            "arn".to_owned(),
            "{}".to_owned(),
            operations,
            updated.iter().map(|id| (*id).to_owned()).collect(),
            None,
            false,
            None,
        )
    }

    #[test]
    fn invocation_info_filters_updated_ids() {
        let info = info_with(&["a", "b"], &["b", "x", "b", "a"]);
        let ids: Vec<&str> = info
            .updated_operations()
            .iter()
            .map(OperationInfo::id)
            .collect();
        assert_eq!(ids, ["b", "a"]);
    }

    /// Every accessor of `InvocationInfo` returns the value it was built
    /// with.
    #[test]
    fn invocation_info_accessors_return_fields() {
        let start = UNIX_EPOCH + Duration::from_secs(5);
        let info = InvocationInfo::new(
            "req".to_owned(),
            "arn:x".to_owned(),
            "\"in\"".to_owned(),
            BTreeMap::new(),
            Vec::new(),
            Some(start),
            true,
            Some("Root=1".to_owned()),
        );
        assert_eq!(info.request_id(), "req");
        assert_eq!(info.execution_arn(), "arn:x");
        assert_eq!(info.execution_input(), "\"in\"");
        assert!(info.operations().is_empty());
        assert!(info.updated_operations().is_empty());
        assert_eq!(info.execution_start_timestamp(), Some(start));
        assert!(info.is_first_invocation());
        assert_eq!(info.x_ray_trace_id(), Some("Root=1"));
    }

    /// A factory for the `create_plugins` tests.
    enum Behavior {
        /// Returns a plugin that logs `dropped <label>` when dropped.
        Some,
        /// Returns `Ok(None)`.
        None,
        /// Panics.
        Panic,
        /// Returns `Err("x")`.
        Err,
        /// Returns a plugin whose `name()` panics.
        PanickingName,
        /// Returns `Err` whose `Display` panics.
        PanickingError,
        /// Returns `Err` whose `Display` writes `drop err` and whose
        /// destructor panics with `error drop`.
        ErrorPanickingDrop,
        /// Returns `Err` whose `Display` and destructor both panic.
        ErrorPanickingDisplayAndDrop,
        /// Returns a plugin whose destructor panics with `plugin drop`.
        PanickingDrop,
        /// Returns `Err` whose destructor panics with a payload whose own
        /// destructor panics.
        ErrorPanickingNestedDrop,
    }

    struct TestFactory {
        label: &'static str,
        behavior: Behavior,
        log: Arc<Mutex<Vec<String>>>,
    }

    struct Dropping {
        label: &'static str,
        log: Arc<Mutex<Vec<String>>>,
    }

    impl Plugin for Dropping {}

    impl Drop for Dropping {
        fn drop(&mut self) {
            self.log
                .lock()
                .unwrap()
                .push(format!("dropped {}", self.label));
        }
    }

    struct PanickingName;

    impl Plugin for PanickingName {
        #[expect(clippy::panic)] // reason: the name panic under test
        fn name(&self) -> &'static str {
            panic!("name boom")
        }
    }

    #[derive(Debug)]
    struct PanickingDisplay;

    impl fmt::Display for PanickingDisplay {
        #[expect(clippy::panic)] // reason: the Display panic under test
        fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
            panic!("display boom")
        }
    }

    impl std::error::Error for PanickingDisplay {}

    /// An error whose destructor panics with `error drop`, and whose
    /// `Display` panics too when `display_panics` is set.
    #[derive(Debug)]
    struct PanickingDropError {
        display_panics: bool,
    }

    impl fmt::Display for PanickingDropError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            assert!(!self.display_panics, "display boom");
            f.write_str("drop err")
        }
    }

    impl std::error::Error for PanickingDropError {}

    impl Drop for PanickingDropError {
        #[expect(clippy::panic)] // reason: the destructor panic under test
        fn drop(&mut self) {
            panic!("error drop");
        }
    }

    /// A panic payload whose destructor panics.
    struct PanickingPayload;

    impl Drop for PanickingPayload {
        #[expect(clippy::panic)] // reason: the nested destructor panic under test
        fn drop(&mut self) {
            panic!("payload drop");
        }
    }

    /// An error whose destructor panics with a [`PanickingPayload`].
    #[derive(Debug)]
    struct NestedDropError;

    impl fmt::Display for NestedDropError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("nested")
        }
    }

    impl std::error::Error for NestedDropError {}

    impl Drop for NestedDropError {
        #[expect(clippy::panic)] // reason: the destructor panic under test
        fn drop(&mut self) {
            std::panic::panic_any(PanickingPayload);
        }
    }

    /// A plugin whose destructor panics with `plugin drop`.
    struct PanickingDropPlugin;

    impl Plugin for PanickingDropPlugin {}

    impl Drop for PanickingDropPlugin {
        #[expect(clippy::panic)] // reason: the destructor panic under test
        fn drop(&mut self) {
            panic!("plugin drop");
        }
    }

    impl PluginFactory for TestFactory {
        #[expect(clippy::panic)] // reason: the factory panic under test
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            self.log.lock().unwrap().push(self.label.to_owned());
            match self.behavior {
                Behavior::Some => Ok(Some(Box::new(Dropping {
                    label: self.label,
                    log: Arc::clone(&self.log),
                }))),
                Behavior::None => Ok(None),
                Behavior::Panic => panic!("factory boom"),
                Behavior::Err => Err("x".into()),
                Behavior::PanickingName => Ok(Some(Box::new(PanickingName))),
                Behavior::PanickingError => Err(Box::new(PanickingDisplay)),
                Behavior::ErrorPanickingDrop => Err(Box::new(PanickingDropError {
                    display_panics: false,
                })),
                Behavior::ErrorPanickingDisplayAndDrop => Err(Box::new(PanickingDropError {
                    display_panics: true,
                })),
                Behavior::PanickingDrop => Ok(Some(Box::new(PanickingDropPlugin))),
                Behavior::ErrorPanickingNestedDrop => Err(Box::new(NestedDropError)),
            }
        }
    }

    fn factories(
        specs: Vec<(&'static str, Behavior)>,
        log: &Arc<Mutex<Vec<String>>>,
    ) -> Vec<(&'static str, Arc<dyn PluginFactory>)> {
        specs
            .into_iter()
            .map(|(label, behavior)| {
                (
                    "test::TestFactory",
                    Arc::new(TestFactory {
                        label,
                        behavior,
                        log: Arc::clone(log),
                    }) as Arc<dyn PluginFactory>,
                )
            })
            .collect()
    }

    fn empty_info() -> InvocationInfo {
        info_with(&[], &[])
    }

    #[test]
    #[expect(clippy::panic)] // reason: build_info must not run
    fn create_plugins_without_factories_builds_no_info() {
        let result = create_plugins(&[], None, || panic!("build_info must not run"));
        assert!(result.is_none());
    }

    use super::test_support::{capture, field, warnings};

    /// The skipped factories are logged with their identities.
    ///
    /// The scenario retries a few times, as the lifecycle capture tests in
    /// `context.rs` do: another test that installs or drops a subscriber
    /// at the same time rebuilds `tracing`'s callsite-interest cache, and
    /// a rebuild that races this test's warning can drop it before it
    /// reaches this test's thread-local subscriber. The plugin outcome is
    /// asserted on every attempt.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured both warnings
    fn create_plugins_skips_none_and_panicking_factories() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let log = Arc::new(Mutex::new(Vec::new()));
            let factories = factories(
                vec![
                    ("A", Behavior::Some),
                    ("none", Behavior::None),
                    ("panic", Behavior::Panic),
                    ("B", Behavior::Some),
                    ("err", Behavior::Err),
                ],
                &log,
            );
            let plugins = create_plugins(&factories, None, empty_info);
            assert!(plugins.is_some());
            drop(plugins);
            assert_eq!(
                *log.lock().unwrap(),
                ["A", "none", "panic", "B", "err", "dropped A", "dropped B"]
            );
            let warnings = warnings(&buffer);
            if warnings.len() != 2 {
                continue;
            }
            let panic = warnings.first().unwrap();
            assert!(field(panic, "factory").ends_with("(plugins[2])"), "{panic}");
            assert!(field(panic, "message").contains("panicked"), "{panic}");
            let err = warnings.get(1).unwrap();
            assert!(field(err, "factory").ends_with("(plugins[4])"), "{err}");
            assert_eq!(field(err, "error"), "x");
            return;
        }
        panic!("no attempt captured the two factory warnings");
    }

    #[test]
    fn create_plugins_returns_some_when_every_factory_skips() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factories = factories(vec![("none", Behavior::None)], &log);
        assert!(create_plugins(&factories, None, empty_info).is_some());
    }

    /// An error whose `Display` panics is still skipped, and the panic is
    /// logged as a warning instead of escaping. The error's destructor
    /// does not panic, so no panic is kept for the invocation. Retried as
    /// in `create_plugins_skips_none_and_panicking_factories`.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    fn create_plugins_contains_a_panicking_error_display() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let log = Arc::new(Mutex::new(Vec::new()));
            let factories = factories(
                vec![("bad", Behavior::PanickingError), ("A", Behavior::Some)],
                &log,
            );
            let mut error_drop_panics = Vec::new();
            let slots = create_slots(&factories, &empty_info(), &mut error_drop_panics);
            assert_eq!(slots.len(), 1);
            assert!(error_drop_panics.is_empty());
            let warnings = warnings(&buffer);
            let [warning] = warnings.as_slice() else {
                continue;
            };
            assert!(
                field(warning, "factory").ends_with("(plugins[0])"),
                "{warning}"
            );
            assert!(
                field(warning, "message").contains("panicked while it was formatted"),
                "{warning}"
            );
            return;
        }
        panic!("no attempt captured the warning of the panicking error");
    }

    /// A factory error whose destructor panics is logged with its message,
    /// and the destructor's payload is kept for the invocation
    /// coordinator, because it fails the invocation. Retried as in
    /// `create_plugins_skips_none_and_panicking_factories`.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    fn factory_error_destructor_panic_is_kept_for_the_coordinator() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let log = Arc::new(Mutex::new(Vec::new()));
            let factories = factories(
                vec![("bad", Behavior::ErrorPanickingDrop), ("A", Behavior::Some)],
                &log,
            );
            let plugins = create_plugins(&factories, None, empty_info).unwrap();
            let panics = plugins.take_error_drop_panics();
            let [payload] = panics.as_slice() else {
                panic!("one destructor panic is kept, got {}", panics.len());
            };
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
            assert!(plugins.take_error_drop_panics().is_empty());
            drop(plugins);
            assert_eq!(*log.lock().unwrap(), ["bad", "A", "dropped A"]);
            let warnings = warnings(&buffer);
            let [warning] = warnings.as_slice() else {
                continue;
            };
            assert!(
                field(warning, "factory").ends_with("(plugins[0])"),
                "{warning}"
            );
            assert_eq!(field(warning, "error"), "drop err");
            return;
        }
        panic!("no attempt captured the warning of the failed factory");
    }

    /// A factory error whose `Display` and destructor both panic does not
    /// abort: the `Display` panic is logged and discarded, and the
    /// destructor's payload is kept for the invocation coordinator.
    #[test]
    #[expect(clippy::panic)] // reason: reports a wrong number of kept panics
    fn factory_error_display_and_destructor_panics_do_not_abort() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factories = factories(vec![("bad", Behavior::ErrorPanickingDisplayAndDrop)], &log);
        let plugins = create_plugins(&factories, None, empty_info).unwrap();
        let panics = plugins.take_error_drop_panics();
        let [payload] = panics.as_slice() else {
            panic!("one destructor panic is kept, got {}", panics.len());
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
    }

    /// A destructor panic that nobody took is dropped with the plugins
    /// through `discard_panic`, so a payload whose own destructor panics
    /// does not escape.
    #[test]
    fn untaken_error_drop_panic_does_not_escape() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factories = factories(vec![("bad", Behavior::ErrorPanickingNestedDrop)], &log);
        let plugins = create_plugins(&factories, None, empty_info).unwrap();
        let result = catch_unwind(AssertUnwindSafe(move || drop(plugins)));
        assert!(result.is_ok(), "no panic escapes the drop");
    }

    /// Two plugins whose destructors panic are each caught, so the second
    /// panic does not abort the process while the first unwinds.
    #[test]
    fn two_plugin_drop_panics_do_not_abort() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factories = factories(
            vec![
                ("bad", Behavior::PanickingDrop),
                ("bad2", Behavior::PanickingDrop),
                ("B", Behavior::Some),
            ],
            &log,
        );
        let plugins = create_plugins(&factories, None, empty_info);
        let result = catch_unwind(AssertUnwindSafe(move || drop(plugins)));
        assert!(result.is_ok(), "no panic escapes the drop");
        assert_eq!(*log.lock().unwrap(), ["bad", "bad2", "B", "dropped B"]);
    }

    /// A plugin whose destructor panics is logged and discarded, and the
    /// plugins after it are still dropped. Retried as in
    /// `create_plugins_skips_none_and_panicking_factories`.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    fn plugin_drop_panic_is_logged_and_discarded() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let log = Arc::new(Mutex::new(Vec::new()));
            let factories = factories(
                vec![("bad", Behavior::PanickingDrop), ("B", Behavior::Some)],
                &log,
            );
            let plugins = create_plugins(&factories, None, empty_info);
            drop(plugins);
            assert_eq!(*log.lock().unwrap(), ["bad", "B", "dropped B"]);
            let warnings = warnings(&buffer);
            let [warning] = warnings.as_slice() else {
                continue;
            };
            assert!(
                field(warning, "plugin").ends_with("PanickingDropPlugin (plugins[0])"),
                "{warning}"
            );
            assert_eq!(field(warning, "hook"), "drop");
            return;
        }
        panic!("no attempt captured the warning of the panicking plugin destructor");
    }

    /// A plugin whose destructor panics while another panic unwinds does
    /// not abort, and the original panic keeps unwinding.
    #[test]
    fn plugin_drop_panic_during_unwind_does_not_abort() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factories = factories(
            vec![("bad", Behavior::PanickingDrop), ("B", Behavior::Some)],
            &log,
        );
        let plugins = create_plugins(&factories, None, empty_info);
        let result = catch_unwind(AssertUnwindSafe(move || {
            let _plugins = plugins;
            std::panic::resume_unwind(Box::new("outer"));
        }));
        let payload = result.expect_err("the outer panic unwinds");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"outer"));
        assert_eq!(*log.lock().unwrap(), ["bad", "B", "dropped B"]);
    }

    struct Probe;

    impl Plugin for Probe {}

    #[test]
    fn plugin_name_defaults_to_type_path() {
        assert!(Probe.name().ends_with("Probe"), "{}", Probe.name());
    }

    struct ProbeFactory(bool);

    impl PluginFactory for ProbeFactory {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(self.0.then(|| Box::new(Probe) as Box<dyn Plugin>))
        }
    }

    #[test]
    fn identity_uses_registration_index() {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> = vec![
            ("skip", Arc::new(ProbeFactory(false))),
            ("probe", Arc::new(ProbeFactory(true))),
        ];
        let slots = create_slots(&factories, &empty_info(), &mut Vec::new());
        assert_eq!(slots.len(), 1);
        let slot = slots.first().unwrap();
        assert!(
            slot.identity.ends_with("Probe (plugins[1])"),
            "{}",
            slot.identity
        );
        assert!(slot.plugin.name().ends_with("Probe"));
    }

    /// A plugin whose `name()` panics is kept, with the identity
    /// `plugins[<i>]`.
    #[test]
    fn identity_falls_back_when_name_panics() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factories = factories(vec![("n", Behavior::PanickingName)], &log);
        let slots = create_slots(&factories, &empty_info(), &mut Vec::new());
        assert_eq!(slots.len(), 1);
        assert_eq!(slots.first().unwrap().identity, "plugins[0]");
    }

    fn step_records(count: usize) -> Vec<(String, CheckpointRecord)> {
        (0..count)
            .map(|i| {
                let id = format!("op-{i}");
                (id, record("STEP", CheckpointStatus::Succeeded))
            })
            .collect()
    }

    #[test]
    fn operation_infos_projects_large_log_once() {
        let log = CheckpointLog::from_records(step_records(10_000));
        let before = crate::engine::projections();
        let map = operation_infos(&log);
        assert_eq!(map.len(), 10_000);
        assert_eq!(crate::engine::projections() - before, 1);
    }

    /// A change set holding one started step info with wire ID `id`.
    fn change_entry(id: &str) -> Vec<OperationInfo> {
        vec![OperationInfo::from_record(
            id,
            &record("STEP", CheckpointStatus::Started),
        )]
    }

    /// The drain delivers every queued change set once, in push order,
    /// and leaves the queue empty.
    #[test]
    fn drain_delivers_in_order_and_empties_queue() {
        let recorder = test_support::Changes::default();
        let plugins = test_support::plugins_of(recorder.clone());
        let log = CheckpointLog::empty();
        for id in ["a", "b", "c"] {
            plugins.push_operation_change(change_entry(id));
        }
        plugins.drain_operation_changes(&log);
        assert_eq!(recorder.updated_ids(), [["a"], ["b"], ["c"]]);
        plugins.drain_operation_changes(&log);
        assert_eq!(recorder.calls().len(), 3, "a second drain makes no call");
    }

    /// A drain started while another one runs returns at once, and the
    /// running drain delivers the entry that was pushed meanwhile.
    #[test]
    fn drain_is_skipped_while_another_drain_runs() {
        let target: Arc<std::sync::OnceLock<Weak<InvocationPlugins>>> = Arc::default();
        let inner_calls = Arc::new(Mutex::new(None::<usize>));
        let mut recorder = test_support::Changes::default();
        let first = Arc::new(AtomicBool::new(true));
        {
            let target = Arc::clone(&target);
            let calls = Arc::clone(&recorder.calls);
            let inner_calls = Arc::clone(&inner_calls);
            recorder.on_change = Some(Arc::new(move |_info| {
                if !first.swap(false, Ordering::SeqCst) {
                    return;
                }
                let plugins = target.get().unwrap().upgrade().unwrap();
                plugins.push_operation_change(change_entry("d"));
                plugins.drain_operation_changes(&CheckpointLog::empty());
                *inner_calls.lock().unwrap() = Some(calls.lock().unwrap().len());
            }));
        }
        let plugins = test_support::plugins_of(recorder.clone());
        target.set(Arc::downgrade(&plugins)).unwrap();
        for id in ["a", "b", "c"] {
            plugins.push_operation_change(change_entry(id));
        }
        plugins.drain_operation_changes(&CheckpointLog::empty());
        assert_eq!(
            *inner_calls.lock().unwrap(),
            Some(1),
            "the inner drain made no call"
        );
        assert_eq!(recorder.updated_ids(), [["a"], ["b"], ["c"], ["d"]]);
        assert!(!plugins.is_draining());
    }

    /// A change set pushed after a drain cleared its flag, by a task that
    /// does not drain, is delivered by that drain's check of the queue,
    /// exactly once and after the earlier one.
    #[test]
    fn drain_rechecks_queue_after_clearing_flag() {
        let recorder = test_support::Changes::default();
        let plugins = test_support::plugins_of(recorder.clone());
        let log = CheckpointLog::empty();
        plugins.set_after_release(|plugins| {
            assert!(
                !plugins.is_draining(),
                "the hook runs after the flag is clear"
            );
            plugins.push_operation_change(change_entry("b"));
        });
        plugins.push_operation_change(change_entry("a"));
        plugins.drain_operation_changes(&log);
        assert_eq!(recorder.updated_ids(), [["a"], ["b"]]);
        assert!(!plugins.is_draining());
        plugins.drain_operation_changes(&log);
        assert_eq!(recorder.calls().len(), 2, "the entry was delivered once");
    }

    /// A no-op waker, for polling a future once by hand.
    fn noop_context() -> Context<'static> {
        Context::from_waker(std::task::Waker::noop())
    }

    /// A drain of change set `a` held open by its change hook on another
    /// thread, with change set `b` queued behind it.
    struct HeldDrain {
        plugins: Arc<InvocationPlugins>,
        log: Arc<CheckpointLog>,
        recorder: test_support::Changes,
        release: std::sync::mpsc::Sender<()>,
        holder: std::thread::JoinHandle<()>,
    }

    impl HeldDrain {
        fn start() -> Self {
            let (entered_tx, entered_rx) = std::sync::mpsc::channel::<()>();
            let (release, release_rx) = std::sync::mpsc::channel::<()>();
            let release_rx = Mutex::new(release_rx);
            let first = AtomicBool::new(true);
            let recorder = test_support::Changes {
                on_change: Some(Arc::new(move |_info| {
                    if first.swap(false, Ordering::SeqCst) {
                        entered_tx.send(()).unwrap();
                        release_rx.lock().unwrap().recv().unwrap();
                    }
                })),
                ..test_support::Changes::default()
            };
            let plugins = test_support::plugins_of(recorder.clone());
            let log = Arc::new(CheckpointLog::empty());
            plugins.push_operation_change(change_entry("a"));
            let holder = {
                let plugins = Arc::clone(&plugins);
                let log = Arc::clone(&log);
                std::thread::spawn(move || plugins.drain_operation_changes(&log))
            };
            entered_rx.recv().unwrap();
            plugins.push_operation_change(change_entry("b"));
            Self {
                plugins,
                log,
                recorder,
                release,
                holder,
            }
        }

        /// Lets the holder deliver both change sets, and waits until it
        /// returned.
        fn finish(self) {
            self.release.send(()).unwrap();
            self.holder.join().unwrap();
        }
    }

    /// The holder's delivery wakes a pending waiter's registered waker by
    /// itself: the test sees the wake before it polls the waiter again.
    /// Once both change sets are delivered, the waiter is ready and its
    /// registration is gone.
    #[test]
    fn delivery_wakes_registered_waiter() {
        let held = HeldDrain::start();
        let (signal, signals) = test_support::SignalWaker::new();
        let waker = std::task::Waker::from(Arc::clone(&signal));
        let mut cx = Context::from_waker(&waker);
        let mut waiter = Box::pin(held.plugins.changes_delivered(&held.log));
        assert!(waiter.as_mut().poll(&mut cx).is_pending());
        assert_eq!(held.plugins.delivery_waiter_count(), 1);
        assert_eq!(signal.woken(), 0);
        assert_eq!(held.recorder.updated_ids(), [["a"]]);
        held.release.send(()).unwrap();
        signals
            .recv_timeout(Duration::from_secs(30))
            .expect("a delivery woke the registered waker");
        held.holder.join().unwrap();
        assert!(waiter.as_mut().poll(&mut cx).is_ready());
        assert_eq!(held.plugins.delivery_waiter_count(), 0);
        assert_eq!(held.recorder.updated_ids(), [["a"], ["b"]]);
    }

    /// Repeated polls of one pending waiter keep one registration, and a
    /// poll with another waker replaces the registered one: the delivery
    /// wakes only the latest waker.
    #[test]
    fn repeated_polls_keep_one_registration() {
        let held = HeldDrain::start();
        let (first, _first_signals) = test_support::SignalWaker::new();
        let (latest, latest_signals) = test_support::SignalWaker::new();
        let first_waker = std::task::Waker::from(Arc::clone(&first));
        let latest_waker = std::task::Waker::from(Arc::clone(&latest));
        let mut waiter = Box::pin(held.plugins.changes_delivered(&held.log));
        for _ in 0..100 {
            assert!(
                waiter
                    .as_mut()
                    .poll(&mut Context::from_waker(&first_waker))
                    .is_pending()
            );
        }
        assert_eq!(held.plugins.delivery_waiter_count(), 1);
        assert!(
            waiter
                .as_mut()
                .poll(&mut Context::from_waker(&latest_waker))
                .is_pending()
        );
        assert_eq!(held.plugins.delivery_waiter_count(), 1);
        held.release.send(()).unwrap();
        latest_signals
            .recv_timeout(Duration::from_secs(30))
            .expect("the delivery woke the latest waker");
        held.holder.join().unwrap();
        assert_eq!(first.woken(), 0, "the replaced waker is not woken");
        assert!(
            waiter
                .as_mut()
                .poll(&mut Context::from_waker(&latest_waker))
                .is_ready()
        );
        assert_eq!(held.plugins.delivery_waiter_count(), 0);
    }

    /// A waiter dropped while it waits removes its registration, so a
    /// cancelled wait leaves nothing behind for a delivery to wake.
    #[test]
    fn dropped_waiter_removes_registration() {
        let held = HeldDrain::start();
        let mut waiter = Box::pin(held.plugins.changes_delivered(&held.log));
        assert!(waiter.as_mut().poll(&mut noop_context()).is_pending());
        assert_eq!(held.plugins.delivery_waiter_count(), 1);
        drop(waiter);
        assert_eq!(held.plugins.delivery_waiter_count(), 0);
        let plugins = Arc::clone(&held.plugins);
        let recorder = held.recorder.clone();
        held.finish();
        assert_eq!(recorder.updated_ids(), [["a"], ["b"]]);
        assert_eq!(plugins.delivery_waiter_count(), 0);
    }

    /// A change hook that waits for deliveries on the thread that is
    /// delivering is ready at once, instead of waiting for itself.
    #[test]
    fn waiter_on_delivering_thread_is_ready() {
        let target: Arc<std::sync::OnceLock<Weak<InvocationPlugins>>> = Arc::default();
        let inner = Arc::new(Mutex::new(None::<bool>));
        let mut recorder = test_support::Changes::default();
        {
            let target = Arc::clone(&target);
            let inner = Arc::clone(&inner);
            let first = AtomicBool::new(true);
            recorder.on_change = Some(Arc::new(move |_info| {
                if !first.swap(false, Ordering::SeqCst) {
                    return;
                }
                let plugins = target.get().unwrap().upgrade().unwrap();
                plugins.push_operation_change(change_entry("b"));
                let log = CheckpointLog::empty();
                let mut waiter = Box::pin(plugins.changes_delivered(&log));
                let ready = waiter.as_mut().poll(&mut noop_context()).is_ready();
                inner.lock().unwrap().get_or_insert(ready);
            }));
        }
        let plugins = test_support::plugins_of(recorder.clone());
        target.set(Arc::downgrade(&plugins)).unwrap();
        plugins.push_operation_change(change_entry("a"));
        let log = CheckpointLog::empty();
        let mut waiter = Box::pin(plugins.changes_delivered(&log));
        assert!(waiter.as_mut().poll(&mut noop_context()).is_ready());
        assert_eq!(*inner.lock().unwrap(), Some(true));
        assert_eq!(recorder.updated_ids(), [["a"], ["b"]]);
    }

    /// A panicking change hook does not keep the next plugin from any
    /// call, and the drain flag is cleared afterwards.
    #[test]
    fn panicking_change_hook_is_isolated() {
        let recorder = test_support::Changes::default();
        let panics = test_support::Changes {
            panics: true,
            ..test_support::Changes::default()
        };
        let plugins = test_support::plugins_of_all(vec![
            Arc::new(panics) as Arc<dyn PluginFactory>,
            Arc::new(recorder.clone()),
        ]);
        let log = CheckpointLog::empty();
        plugins.push_operation_change(change_entry("a"));
        plugins.push_operation_change(change_entry("b"));
        plugins.drain_operation_changes(&log);
        assert_eq!(recorder.updated_ids(), [["a"], ["b"]]);
        assert!(!plugins.is_draining());
    }

    /// Each drained entry projects the log once, and its operations map
    /// holds the log's records and the `EXECUTION` slot.
    #[test]
    fn drained_change_holds_the_full_map() {
        let recorder = test_support::Changes::default();
        let plugins = test_support::plugins_of(recorder.clone());
        let execution = OperationInfo::execution(
            "e",
            "EXECUTION",
            &CheckpointStatus::Started,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(plugins.replace_execution(execution));
        let log = CheckpointLog::from_records(vec![(
            "s".to_owned(),
            record("STEP", CheckpointStatus::Started),
        )]);
        plugins.push_operation_change(change_entry("s"));
        let before = crate::engine::projections();
        plugins.drain_operation_changes(&log);
        assert_eq!(crate::engine::projections() - before, 1);
        let calls = recorder.calls();
        let info = calls.first().unwrap();
        assert_eq!(info.operations().keys().collect::<Vec<_>>(), ["e", "s"]);
        assert_eq!(info.execution_arn(), plugins.info().execution_arn());
    }

    /// Replacing the `EXECUTION` slot is a change when the slot was empty
    /// or held another status, and not when the status is the same.
    #[test]
    fn replace_execution_reports_status_changes() {
        let plugins = test_support::plugins_of(test_support::Changes::default());
        let exec = |status: CheckpointStatus| {
            OperationInfo::execution("e", "EXECUTION", &status, None, None, None).unwrap()
        };
        assert!(plugins.replace_execution(exec(CheckpointStatus::Started)));
        assert!(!plugins.replace_execution(exec(CheckpointStatus::Started)));
        assert!(plugins.replace_execution(exec(CheckpointStatus::Succeeded)));
    }

    /// The operations map inserts the `EXECUTION` slot under its wire ID,
    /// and holds only the log's operations without one.
    #[test]
    fn operations_map_inserts_execution_slot() {
        let log = CheckpointLog::from_records(vec![(
            "s".to_owned(),
            record("STEP", CheckpointStatus::Started),
        )]);
        assert_eq!(operations_map(&log, None).len(), 1);
        let execution = OperationInfo::execution(
            "e",
            "EXECUTION",
            &CheckpointStatus::Started,
            None,
            None,
            None,
        );
        let map = operations_map(&log, execution.as_ref());
        assert_eq!(map.keys().collect::<Vec<_>>(), ["e", "s"]);
    }

    #[test]
    fn attempt_outcome_wire_str_and_display() {
        assert_eq!(AttemptOutcome::Succeeded.wire_str(), "SUCCEEDED");
        assert_eq!(AttemptOutcome::Failed.wire_str(), "FAILED");
        assert_eq!(AttemptOutcome::Succeeded.to_string(), "Succeeded");
        assert_eq!(AttemptOutcome::Failed.to_string(), "Failed");
    }

    /// Keeps a clone of the `AttemptInfo` its `on_attempt_start` receives.
    #[derive(Clone, Default)]
    struct AttemptCapture(Arc<Mutex<Option<AttemptInfo>>>);

    impl Plugin for AttemptCapture {
        fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
            *self.0.lock().unwrap() = Some(info.clone());
            Ok(None)
        }
    }

    impl PluginFactory for AttemptCapture {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    /// r7 item 6: the runner receives the very `OperationInfo` that
    /// `on_attempt_start` saw, not a copy.
    #[test]
    fn attempt_started_returns_the_attempt_infos_own_operation() {
        let capture = AttemptCapture::default();
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> =
            vec![("test::AttemptCapture", Arc::new(capture.clone()))];
        let plugins = create_plugins(&factories, None, empty_info).expect("registered");
        let operation = OperationInfo::from_record("a", &record("STEP", CheckpointStatus::Started));
        let returned = attempt_started(&plugins, operation, 1, false);
        let seen = capture.0.lock().unwrap().take().expect("the hook ran");
        assert!(std::ptr::eq(seen.operation(), Arc::as_ptr(&returned)));
    }

    /// A STARTED step record resumed as attempt 3: `on_operation_start`
    /// gets the record's attempt with `is_replay` true, as main and JS do;
    /// `on_attempt_start` gets attempt 3, `Started` and `is_replay` false,
    /// in the info `attempt_started` returns for the runner (rule 7.3).
    #[test]
    fn attempt_started_keeps_the_records_attempt_in_the_start_hook() {
        let hooks = test_support::OpHooks::default();
        let plugins = hooks.plugins();
        let mut started = record("STEP", CheckpointStatus::Started);
        started.attempt = 2;
        let operation = OperationInfo::from_record("a", &started);
        assert_eq!(operation.attempt(), Some(2));
        let returned = attempt_started(&plugins, operation, 3, true);
        let seen: Vec<_> = hooks
            .records()
            .iter()
            .map(|r| (r.hook, r.info_attempt, r.replay, r.status))
            .collect();
        assert_eq!(
            seen,
            [
                ("start", Some(2), true, Some(OperationStatus::Started)),
                (
                    "attempt_start",
                    Some(3),
                    false,
                    Some(OperationStatus::Started)
                ),
            ]
        );
        assert_eq!(returned.attempt(), Some(3));
        assert!(!returned.is_replay());
        assert_eq!(returned.status(), Some(OperationStatus::Started));
        assert!(returned.start_timestamp().is_some());
    }

    /// Every hook that is not a wrap hook has a no-op default: the async
    /// hooks resolve with `Ok(())`, the operation hooks return no work, and
    /// the propagation hook returns no header. A factory has no group by
    /// default.
    #[tokio::test]
    async fn default_hooks_do_nothing() {
        let operation = OperationInfo::from_record("a", &record("STEP", CheckpointStatus::Started));
        let attempt = AttemptInfo {
            operation: Arc::new(operation.clone()),
            attempt: 1,
        };
        let attempt_end = AttemptEndInfo {
            operation: operation.clone(),
            attempt: 1,
            outcome: AttemptOutcome::Succeeded,
            error: None,
        };
        let change = OperationChangeInfo {
            execution_arn: "arn".to_owned(),
            updated_operations: Vec::new(),
            operations: BTreeMap::new(),
        };
        let end = InvocationEndInfo {
            request_id: "r".to_owned(),
            execution_arn: "arn".to_owned(),
            execution_input: "{}".to_owned(),
            operations: BTreeMap::new(),
            execution_start_timestamp: None,
            status: InvocationStatus::Succeeded,
            execution_result: None,
            execution_error: None,
        };
        let propagation = PropagationInput {
            execution_arn: "arn".to_owned(),
            operation_id: "a".to_owned(),
            parent_operation_id: None,
            target_function_name: "f".to_owned(),
        };
        let info = empty_info();
        assert!(Probe.on_invocation_start(&info).await.is_ok());
        assert!(Probe.on_invocation_end(&end).await.is_ok());
        assert!(Probe.on_operation_start(&operation).unwrap().is_none());
        assert!(Probe.on_operation_end(&operation).unwrap().is_none());
        assert!(Probe.on_attempt_start(&attempt).unwrap().is_none());
        assert!(Probe.on_attempt_end(&attempt_end).unwrap().is_none());
        assert!(Probe.on_operation_change(&change).unwrap().is_none());
        assert_eq!(Probe.propagation_trace_id(&propagation).unwrap(), None);
        assert_eq!(ProbeFactory(true).exclusive_group(), None);
    }

    // Hook work. The operation hooks that return work are called by later
    // issues, so each handler starts its work directly through
    // `ctx.plugins()`, as a hook call site does.

    use super::test_support::{Hooks, deadline_in, service, with_message};
    use crate::DurableContext;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Starts `work` as the `on_operation_end` work of plugin 0.
    fn start(
        ctx: &DurableContext,
        work: impl Future<Output = Result<(), BoxError>> + Send + 'static,
    ) {
        ctx.plugins()
            .expect("the invocation has plugins")
            .start_work(0, "on_operation_end", Box::pin(work));
    }

    /// Runs its closure when dropped.
    struct OnDrop(Option<Box<dyn FnOnce() + Send>>);

    impl OnDrop {
        fn new(f: impl FnOnce() + Send + 'static) -> Self {
            Self(Some(Box::new(f)))
        }
    }

    impl Drop for OnDrop {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }

    /// The time since `start`, in whole milliseconds.
    fn millis_since(start: tokio::time::Instant) -> u128 {
        start.elapsed().as_millis()
    }

    /// How many times a deadline test runs its scenario at most.
    const DEADLINE_ATTEMPTS: usize = 5;

    /// The real time a run may take for its result to be checked exactly.
    const EXACT_SLACK_MS: u128 = 50;

    /// The slack of a deadline test's run that started at `real`.
    ///
    /// Tokio's time is paused, so a run's tokio time moves only through
    /// its timers. The Lambda deadline is an epoch time, so the real time
    /// that passes before the SDK reads it, and the millisecond truncation
    /// of the deadline and of the readings, shorten the budget. So an event
    /// that the SDK places `x` ms before the deadline is seen at most
    /// `slack` ms before its exact time, and never after it.
    fn slack(real: std::time::Instant) -> u128 {
        real.elapsed().as_millis() + 2
    }

    /// Whether a deadline test should run its scenario again: the run took
    /// too much real time for an exact check, and attempts remain.
    fn retry(slack: u128, attempt: usize) -> bool {
        slack >= EXACT_SLACK_MS && attempt + 1 < DEADLINE_ATTEMPTS
    }

    /// Asserts that `what` happened at `actual` ms: at most `exact` ms,
    /// and at least `exact` ms less `slack`. With the slack under
    /// [`EXACT_SLACK_MS`], a reserve 50 ms off fails the check.
    fn assert_at(what: &str, actual: u128, exact: u128, slack: u128) {
        assert!(
            actual <= exact && actual + slack >= exact,
            "{what} at {actual} ms; expected {exact} ms, less at most {slack} ms"
        );
    }

    type Shared<T> = Arc<Mutex<T>>;

    fn shared<T: Default>() -> Shared<T> {
        Arc::new(Mutex::new(T::default()))
    }

    /// A plugin whose end hook records the milliseconds since `start` at
    /// its start, then awaits `then`.
    fn end_at<F>(start: tokio::time::Instant, at: &Shared<Option<u128>>, then: F) -> Hooks
    where
        F: Fn() -> HookFuture<'static> + Send + Sync + 'static,
    {
        let at = Arc::clone(at);
        Hooks::default().on_end(move |_| {
            *at.lock().unwrap() = Some(millis_since(start));
            then()
        })
    }

    fn ready() -> HookFuture<'static> {
        Box::pin(std::future::ready(Ok(())))
    }

    #[tokio::test(start_paused = true)]
    async fn hook_work_runs_during_the_invocation() {
        let flag = Arc::new(AtomicBool::new(false));
        let seen = shared::<Option<bool>>();
        let plugin = {
            let flag = Arc::clone(&flag);
            let seen = Arc::clone(&seen);
            Hooks::default().on_end(move |_| {
                *seen.lock().unwrap() = Some(flag.load(Ordering::SeqCst));
                ready()
            })
        };
        let (service, event) = service(vec![plugin], 0, move |_e, ctx| {
            let flag = Arc::clone(&flag);
            async move {
                ctx.step(|_| async { Ok(1_u32) }).await?;
                start(&ctx, async move {
                    flag.store(true, Ordering::SeqCst);
                    Ok(())
                });
                Ok(serde_json::Value::Null)
            }
        });
        service(event).await.unwrap();
        assert_eq!(*seen.lock().unwrap(), Some(true));
    }

    #[tokio::test(start_paused = true)]
    async fn hook_work_does_not_delay_task_cleanup() {
        for attempt in 0..DEADLINE_ATTEMPTS {
            if hook_work_does_not_delay_task_cleanup_once(attempt).await {
                return;
            }
        }
    }

    /// One run of `hook_work_does_not_delay_task_cleanup`. Returns whether
    /// the run was checked.
    async fn hook_work_does_not_delay_task_cleanup_once(attempt: usize) -> bool {
        let real = std::time::Instant::now();
        let begin = tokio::time::Instant::now();
        let observed = shared::<Option<u128>>();
        let ended = shared::<Option<u128>>();
        let end_started = shared::<Option<u128>>();
        let plugin = end_at(begin, &end_started, ready);
        let (service, event) = service(vec![plugin], deadline_in(2_000), {
            let observed = Arc::clone(&observed);
            let ended = Arc::clone(&ended);
            move |_e, ctx| {
                let observed = Arc::clone(&observed);
                let ended = Arc::clone(&ended);
                async move {
                    ctx.step(|_| async { Ok(1_u32) }).await?;
                    let probe = ctx.clone();
                    start(&ctx, async move {
                        let _guard = OnDrop::new(move || {
                            *ended.lock().unwrap() = Some(millis_since(begin));
                        });
                        // The drive returns while the work runs: the task
                        // count reaches zero at once.
                        while probe.coordinator().tasks().live() > 0 {
                            tokio::task::yield_now().await;
                        }
                        *observed.lock().unwrap() = Some(millis_since(begin));
                        drop(probe);
                        tokio::time::sleep(Duration::from_secs(20)).await;
                        Ok(())
                    });
                    Ok(serde_json::Value::Null)
                }
            }
        });
        service(event).await.unwrap();
        let slack = slack(real);
        if retry(slack, attempt) {
            return false;
        }
        assert_eq!(*observed.lock().unwrap(), Some(0));
        let aborted = ended
            .lock()
            .unwrap()
            .expect("the work ended when the service returned");
        assert_at("the drain aborted the work", aborted, 1_500, slack);
        let end = end_started.lock().unwrap().expect("the end hook ran");
        assert!(
            end >= aborted,
            "the end hook started at {end} ms, before the abort"
        );
        true
    }

    #[tokio::test(start_paused = true)]
    async fn drain_waits_for_work_started_while_draining() {
        let second_done = Arc::new(AtomicBool::new(false));
        let seen = shared::<Option<bool>>();
        let plugin = {
            let second_done = Arc::clone(&second_done);
            let seen = Arc::clone(&seen);
            Hooks::default().on_end(move |_| {
                *seen.lock().unwrap() = Some(second_done.load(Ordering::SeqCst));
                ready()
            })
        };
        let (service, event) = service(vec![plugin], 0, move |_e, ctx| {
            let second_done = Arc::clone(&second_done);
            async move {
                let starter = ctx.clone();
                start(&ctx, async move {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    start(&starter, async move {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        second_done.store(true, Ordering::SeqCst);
                        Ok(())
                    });
                    Ok(())
                });
                Ok(serde_json::Value::Null)
            }
        });
        service(event).await.unwrap();
        assert_eq!(*seen.lock().unwrap(), Some(true));
    }

    /// The drain aborts work that was started while it waited and that it
    /// had not taken yet: the drain waits for the first list, which holds
    /// 20 s of work, while the second work is started.
    #[tokio::test(start_paused = true)]
    async fn drain_aborts_work_started_while_waiting() {
        for attempt in 0..DEADLINE_ATTEMPTS {
            if drain_aborts_work_started_while_waiting_once(attempt).await {
                return;
            }
        }
    }

    /// One run of `drain_aborts_work_started_while_waiting`. Returns
    /// whether the run was checked.
    async fn drain_aborts_work_started_while_waiting_once(attempt: usize) -> bool {
        let real = std::time::Instant::now();
        let begin = tokio::time::Instant::now();
        let ended = shared::<Option<u128>>();
        let end_started = shared::<Option<u128>>();
        let plugin = end_at(begin, &end_started, ready);
        let (service, event) = service(vec![plugin], deadline_in(2_000), {
            let ended = Arc::clone(&ended);
            move |_e, ctx| {
                let ended = Arc::clone(&ended);
                async move {
                    let starter = ctx.clone();
                    start(&ctx, async move {
                        tokio::time::sleep(Duration::from_secs(20)).await;
                        Ok(())
                    });
                    start(&ctx, async move {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                        start(&starter, async move {
                            let _guard = OnDrop::new(move || {
                                *ended.lock().unwrap() = Some(millis_since(begin));
                            });
                            tokio::time::sleep(Duration::from_secs(20)).await;
                            Ok(())
                        });
                        Ok(())
                    });
                    Ok(serde_json::Value::Null)
                }
            }
        });
        service(event).await.unwrap();
        let slack = slack(real);
        if retry(slack, attempt) {
            return false;
        }
        let aborted = ended.lock().unwrap().expect("the second work ended");
        assert_at("the drain aborted the second work", aborted, 1_500, slack);
        let end = end_started.lock().unwrap().expect("the end hook ran");
        assert!(end >= aborted, "the end hook started at {end} ms");
        true
    }

    /// A handler that starts 20 s of work whose drop is recorded in
    /// `ended`, and returns.
    fn twenty_seconds_of_work(
        begin: tokio::time::Instant,
        ended: &Shared<Option<u128>>,
    ) -> impl Fn(
        serde_json::Value,
        DurableContext,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, BoxError>> + Send>>
    + Send
    + Sync
    + 'static {
        let ended = Arc::clone(ended);
        move |_e, ctx| {
            let ended = Arc::clone(&ended);
            Box::pin(async move {
                start(&ctx, async move {
                    let _guard = OnDrop::new(move || {
                        *ended.lock().unwrap() = Some(millis_since(begin));
                    });
                    tokio::time::sleep(Duration::from_secs(20)).await;
                    Ok(())
                });
                Ok(serde_json::Value::Null)
            })
        }
    }

    #[tokio::test(start_paused = true)]
    #[expect(clippy::panic)] // reason: reports that no attempt captured both warnings
    async fn drain_stops_500ms_and_end_50ms_before_deadline() {
        for attempt in 0..DEADLINE_ATTEMPTS {
            let (buffer, _guard) = capture();
            let real = std::time::Instant::now();
            let begin = tokio::time::Instant::now();
            let ended = shared::<Option<u128>>();
            let end_dropped = Arc::new(AtomicBool::new(false));
            let plugin = {
                let end_dropped = Arc::clone(&end_dropped);
                Hooks::default().on_end(move |_| {
                    let guard = OnDrop::new({
                        let end_dropped = Arc::clone(&end_dropped);
                        move || end_dropped.store(true, Ordering::SeqCst)
                    });
                    Box::pin(async move {
                        let _guard = guard;
                        std::future::pending::<()>().await;
                        Ok(())
                    })
                })
            };
            let (service, event) = service(
                vec![plugin],
                deadline_in(2_000),
                twenty_seconds_of_work(begin, &ended),
            );
            service(event).await.unwrap();
            let returned = millis_since(begin);
            let slack = slack(real);
            if retry(slack, attempt) {
                continue;
            }
            assert_at("the service returned", returned, 1_950, slack);
            let aborted = ended.lock().unwrap().expect("the work was aborted");
            assert_at("the drain aborted the work", aborted, 1_500, slack);
            assert!(
                end_dropped.load(Ordering::SeqCst),
                "the cut-off end hook was dropped"
            );
            let warnings = warnings(&buffer);
            let drained = with_message(
                &warnings,
                "plugin hook work did not finish before the drain deadline; it is aborted",
            );
            let cut = with_message(
                &warnings,
                "plugin on_invocation_end hooks did not finish before the invocation deadline; \
                 the response returns without them",
            );
            let ([drained], [_]) = (drained.as_slice(), cut.as_slice()) else {
                continue;
            };
            assert_eq!(drained.get("abandoned"), Some(&serde_json::json!(1)));
            return;
        }
        panic!("no attempt captured both warnings");
    }

    #[tokio::test(start_paused = true)]
    async fn end_hook_keeps_its_time() {
        for attempt in 0..DEADLINE_ATTEMPTS {
            if end_hook_keeps_its_time_once(attempt).await {
                return;
            }
        }
    }

    /// One run of `end_hook_keeps_its_time`. Returns whether the run was
    /// checked.
    async fn end_hook_keeps_its_time_once(attempt: usize) -> bool {
        let real = std::time::Instant::now();
        let begin = tokio::time::Instant::now();
        let ended = shared::<Option<u128>>();
        let finished = shared::<Option<u128>>();
        let plugin = {
            let finished = Arc::clone(&finished);
            Hooks::default().on_end(move |_| {
                let finished = Arc::clone(&finished);
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    *finished.lock().unwrap() = Some(millis_since(begin));
                    Ok(())
                })
            })
        };
        let (service, event) = service(
            vec![plugin],
            deadline_in(2_000),
            twenty_seconds_of_work(begin, &ended),
        );
        service(event).await.unwrap();
        let slack = slack(real);
        if retry(slack, attempt) {
            return false;
        }
        let at = finished.lock().unwrap().expect("the end hook finished");
        assert_at("the end hook finished", at, 1_800, slack);
        true
    }

    #[tokio::test(start_paused = true)]
    async fn drain_unbounded_without_deadline() {
        let begin = tokio::time::Instant::now();
        let ended = shared::<Option<u128>>();
        let end_started = shared::<Option<u128>>();
        let plugin = end_at(begin, &end_started, ready);
        let (service, event) = service(vec![plugin], 0, twenty_seconds_of_work(begin, &ended));
        service(event).await.unwrap();
        assert_eq!(*ended.lock().unwrap(), Some(20_000));
        assert_eq!(*end_started.lock().unwrap(), Some(20_000));
    }

    #[tokio::test(start_paused = true)]
    async fn drain_skipped_inside_the_reserve() {
        let begin = tokio::time::Instant::now();
        let ended = shared::<Option<u128>>();
        let end_started = shared::<Option<u128>>();
        let plugin = end_at(begin, &end_started, ready);
        let (service, event) = service(
            vec![plugin],
            deadline_in(400),
            twenty_seconds_of_work(begin, &ended),
        );
        service(event).await.unwrap();
        assert_eq!(*ended.lock().unwrap(), Some(0));
        assert_eq!(*end_started.lock().unwrap(), Some(0));
    }

    /// An error whose `Display` panics with `display boom`.
    #[derive(Debug)]
    struct PanickingDisplayError;

    impl fmt::Display for PanickingDisplayError {
        #[expect(clippy::panic)] // reason: the formatting panic under test
        fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
            panic!("display boom")
        }
    }

    impl std::error::Error for PanickingDisplayError {}

    #[tokio::test(start_paused = true)]
    #[expect(clippy::panic)] // reason: the hook panic under test, and the report of no capture
    async fn hook_work_errors_and_panics_are_logged() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let (service, event) = service(
                vec![Hooks::default().named("Worker")],
                0,
                |_e, ctx| async move {
                    start(&ctx, async { Err("x".into()) });
                    start(&ctx, async { panic!("work boom") });
                    start(&ctx, async {
                        Err(Box::new(PanickingDisplayError) as BoxError)
                    });
                    Ok(serde_json::json!("out"))
                },
            );
            let json = service(event).await.unwrap().0;
            assert_eq!(json.get("Status"), Some(&serde_json::json!("SUCCEEDED")));
            assert_eq!(json.get("Result"), Some(&serde_json::json!("\"out\"")));
            let warnings = warnings(&buffer);
            let failed = with_message(&warnings, "plugin hook failed; the error is ignored");
            let panicked = with_message(&warnings, "plugin hook panicked; the panic is ignored");
            let unformatted = with_message(
                &warnings,
                "plugin hook failed, and its error panicked while it was formatted; \
                 the error is ignored",
            );
            let ([failed], [panicked], [unformatted]) = (
                failed.as_slice(),
                panicked.as_slice(),
                unformatted.as_slice(),
            ) else {
                continue;
            };
            for warning in [failed, panicked, unformatted] {
                assert_eq!(field(warning, "plugin"), "Worker (plugins[0])", "{warning}");
                assert_eq!(field(warning, "hook"), "on_operation_end", "{warning}");
            }
            assert_eq!(field(failed, "error"), "x");
            return;
        }
        panic!("no attempt captured the three warnings");
    }

    /// The SDK drops a hook future inside a catch: a panic from the
    /// destructor of a finished future is logged as the hook's panic.
    #[tokio::test(start_paused = true)]
    #[expect(clippy::panic)] // reason: the destructor panic under test, and the report of no capture
    async fn hook_future_drop_panic_is_logged() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let plugin = Hooks::default().on_end(|_| {
                let guard = OnDrop::new(|| panic!("future drop"));
                Box::pin(async move {
                    let _guard = guard;
                    Ok(())
                })
            });
            let (service, event) = service(vec![plugin], 0, |_e, _ctx| async move {
                Ok(serde_json::Value::Null)
            });
            service(event).await.unwrap();
            let warnings = warnings(&buffer);
            let [warning] =
                with_message(&warnings, "plugin hook panicked; the panic is ignored")[..]
            else {
                continue;
            };
            assert_eq!(field(warning, "hook"), "on_invocation_end");
            return;
        }
        panic!("no attempt captured the warning");
    }

    // User-function hooks: the plugin guard, warnings and dispatch.

    /// A user-function plugin whose start hook and wrap are closures.
    #[derive(Clone, Default)]
    struct UfPlugin {
        start: Option<Arc<dyn Fn() -> HookWork + Send + Sync>>,
        /// Runs in `wrap_user_function` after `next()`.
        after_next: Option<Arc<dyn Fn() + Send + Sync>>,
    }

    impl UfPlugin {
        fn starting(start: impl Fn() -> HookWork + Send + Sync + 'static) -> Self {
            Self {
                start: Some(Arc::new(start)),
                ..Self::default()
            }
        }

        fn after_next(after: impl Fn() + Send + Sync + 'static) -> Self {
            Self {
                after_next: Some(Arc::new(after)),
                ..Self::default()
            }
        }
    }

    impl Plugin for UfPlugin {
        fn name(&self) -> &'static str {
            "UfPlugin"
        }

        fn on_user_function_start(&self, _info: &OperationInfo) -> HookWork {
            self.start.as_ref().map_or(Ok(None), |start| start())
        }

        fn wrap_user_function(&self, _info: &OperationInfo, next: &mut dyn FnMut()) {
            next();
            if let Some(after) = &self.after_next {
                after();
            }
        }
    }

    impl PluginFactory for UfPlugin {
        fn create_plugin(
            &self,
            _info: &InvocationInfo,
        ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    /// Polls once a runner of the step `s` over `plugins` whose body
    /// returns `Ok(7)`, and returns the body's value.
    fn run_step_once(plugins: &Arc<InvocationPlugins>) -> Option<u32> {
        let mut run = std::pin::pin!(run_user_function(
            Some(Arc::clone(plugins)),
            Arc::new(OperationInfo::for_test("s")),
            UserBody::Step("s".to_owned()),
            tracing::Span::none(),
            || async { Ok::<u32, BoxError>(7) },
            |e: &BoxError| WireError::new(Some("Error".to_owned()), Some(e.to_string())),
        ));
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        match run.as_mut().poll(&mut cx) {
            Poll::Ready(ran) => ran.into_parts().ok(),
            Poll::Pending => None,
        }
    }

    /// A pending hook future that its `HookRun` holds, dropped directly,
    /// with no drain and no deadline, is dropped under the plugin guard:
    /// the `wait` its destructor calls records the context usage error.
    #[tokio::test]
    async fn hook_future_drop_calls_wait_records_context_usage() {
        let ctx = DurableContext::test_context();
        let plugins = test_support::plugins_of(Hooks::default());
        let dropped = Arc::new(AtomicBool::new(false));
        let guard = OnDrop::new({
            let ctx = ctx.clone();
            let dropped = Arc::clone(&dropped);
            move || {
                drop(ctx.wait(Duration::from_secs(1)));
                dropped.store(true, Ordering::SeqCst);
            }
        });
        let mut run = HookRun {
            identity: "Hooks (plugins[0])".into(),
            hook: "on_user_function_start",
            future: Some(Box::pin(async move {
                let _guard = guard;
                std::future::pending::<()>().await;
                Ok(())
            })),
            undelivered: Arc::clone(&plugins.undelivered),
        };
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        assert!(Pin::new(&mut run).poll(&mut cx).is_pending());
        assert!(ctx.coordinator().fatal_error().is_none());
        drop(run);
        assert!(dropped.load(Ordering::SeqCst), "the destructor ran");
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert!(
            fatal.error_message.contains("inside plugin code"),
            "{}",
            fatal.error_message
        );
    }

    /// A start hook future that waits on a gate the test holds does not
    /// delay the suspension: the handler's `wait(1 s)` suspends the
    /// invocation with the future pending, and the invocation-end drain
    /// waits for it. Once the gate opens, the drain ends, and the response
    /// is `PENDING`.
    #[tokio::test]
    async fn pending_hook_work_does_not_delay_suspension() {
        #[derive(Clone)]
        struct Gated {
            gate: Arc<tokio::sync::Notify>,
            polled: Arc<AtomicBool>,
            done: Arc<AtomicBool>,
        }
        impl Plugin for Gated {
            fn on_user_function_start(&self, _info: &OperationInfo) -> HookWork {
                let (gate, polled, done) = (
                    Arc::clone(&self.gate),
                    Arc::clone(&self.polled),
                    Arc::clone(&self.done),
                );
                Ok(Some(Box::pin(async move {
                    polled.store(true, Ordering::SeqCst);
                    gate.notified().await;
                    done.store(true, Ordering::SeqCst);
                    Ok(())
                })))
            }
        }
        impl PluginFactory for Gated {
            fn create_plugin(
                &self,
                _info: &InvocationInfo,
            ) -> Result<Option<Box<dyn Plugin>>, BoxError> {
                Ok(Some(Box::new(self.clone())))
            }
        }
        let gated = Gated {
            gate: Arc::new(tokio::sync::Notify::new()),
            polled: Arc::default(),
            done: Arc::default(),
        };
        let suspended = Arc::new(AtomicBool::new(false));
        let (service, event) = test_support::service_configured(
            {
                let gated = gated.clone();
                move |builder| builder.plugin(gated)
            },
            0,
            Vec::new(),
            {
                let suspended = Arc::clone(&suspended);
                move |_e, ctx| {
                    let suspended = Arc::clone(&suspended);
                    async move {
                        ctx.step(|_| async { Ok(1_u32) }).name("s").await?;
                        // Dropped when the suspension drops the handler.
                        let _guard = OnDrop::new(move || suspended.store(true, Ordering::SeqCst));
                        ctx.wait(Duration::from_secs(1)).name("w").await?;
                        Ok(serde_json::json!("done"))
                    }
                }
            },
        );
        let mut invocation = std::pin::pin!(service(event));
        let early = tokio::time::timeout(Duration::from_millis(200), invocation.as_mut()).await;
        assert!(early.is_err(), "the drain waits for the hook future");
        assert!(suspended.load(Ordering::SeqCst), "the invocation suspended");
        assert!(gated.polled.load(Ordering::SeqCst), "the hook future ran");
        assert!(
            !gated.done.load(Ordering::SeqCst),
            "the hook future is pending"
        );
        gated.gate.notify_one();
        let json = tokio::time::timeout(Duration::from_secs(5), invocation)
            .await
            .expect("the drain ends once the gate opens")
            .unwrap()
            .0;
        assert_eq!(
            json.get("Status"),
            Some(&serde_json::json!("PENDING")),
            "{json}"
        );
        assert!(gated.done.load(Ordering::SeqCst));
    }

    /// An error whose `Display` counts its calls, and panics when `panics`
    /// is set.
    #[derive(Debug)]
    struct CountingDisplay {
        calls: Arc<AtomicUsize>,
        panics: bool,
    }

    impl fmt::Display for CountingDisplay {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panics, "display boom");
            f.write_str("counted")
        }
    }

    impl std::error::Error for CountingDisplay {}

    /// A start hook error is formatted once, for its one WARN with the
    /// `error` field. When its `Display` panics, the formatting-panic WARN
    /// is logged instead. Either way the body's result returns.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warnings
    fn error_display_counted_once_and_formatting_panic_logged() {
        'attempt: for _attempt in 0..5 {
            for panics in [false, true] {
                let calls = Arc::new(AtomicUsize::new(0));
                let plugins = test_support::plugins_of(UfPlugin::starting({
                    let calls = Arc::clone(&calls);
                    move || {
                        Err(Box::new(CountingDisplay {
                            calls: Arc::clone(&calls),
                            panics,
                        }) as BoxError)
                    }
                }));
                let (buffer, guard) = capture();
                let result = run_step_once(&plugins);
                drop(guard);
                assert_eq!(result, Some(7), "panics: {panics}");
                assert_eq!(calls.load(Ordering::SeqCst), 1, "panics: {panics}");
                let warnings = warnings(&buffer);
                let [warning] = warnings.as_slice() else {
                    continue 'attempt;
                };
                assert_eq!(field(warning, "hook"), "on_user_function_start");
                assert_eq!(field(warning, "plugin"), "UfPlugin (plugins[0])");
                if panics {
                    assert_eq!(
                        field(warning, "message"),
                        "plugin hook failed, and its error panicked while it was formatted; \
                         the error is ignored"
                    );
                    assert!(warning.get("error").is_none(), "{warning}");
                } else {
                    assert_eq!(
                        field(warning, "message"),
                        "plugin hook failed; the error is ignored"
                    );
                    assert_eq!(field(warning, "error"), "counted");
                }
            }
            return;
        }
        panic!("no attempt captured exactly one warning per run");
    }

    /// A start hook error whose destructor panics: the error is logged in
    /// one WARN, and the destructor's panic fails the invocation.
    #[tokio::test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    async fn error_destructor_panic_fails_the_invocation() {
        for _attempt in 0..5 {
            let (buffer, guard) = capture();
            let (service, event) = test_support::service_configured(
                |builder| {
                    builder.plugin(UfPlugin::starting(|| {
                        Err(Box::new(DropPanicError) as BoxError)
                    }))
                },
                0,
                Vec::new(),
                |_e, ctx| async move {
                    let value = ctx.step(|_| async { Ok(7_u32) }).name("s").await?;
                    Ok(serde_json::json!(value))
                },
            );
            let payload = panic_of(service(event)).await;
            drop(guard);
            assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
            let warnings = warnings(&buffer);
            let [warning] = warnings.as_slice() else {
                continue;
            };
            assert_eq!(
                field(warning, "message"),
                "plugin hook failed; the error is ignored"
            );
            assert_eq!(field(warning, "hook"), "on_user_function_start");
            assert_eq!(field(warning, "error"), "drop err");
            return;
        }
        panic!("no attempt captured exactly one warning");
    }

    /// A wrap that panics after `next()` with a payload whose destructor
    /// panics: the wrap panic is logged, the payload's destructor panic is
    /// discarded, and the body's result returns.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    fn panic_payload_destructor_panic_is_discarded() {
        for _attempt in 0..5 {
            #[expect(clippy::panic, reason = "the wrap panics with a payload")]
            fn panic_with_payload() {
                std::panic::panic_any(PanickingPayload)
            }
            let plugins = test_support::plugins_of(UfPlugin::after_next(panic_with_payload));
            let (buffer, guard) = capture();
            let result = run_step_once(&plugins);
            drop(guard);
            assert_eq!(result, Some(7));
            let warnings = warnings(&buffer);
            let [warning] = warnings.as_slice() else {
                continue;
            };
            assert_eq!(
                field(warning, "message"),
                "plugin hook panicked; the panic is ignored"
            );
            assert_eq!(field(warning, "hook"), "wrap_user_function");
            return;
        }
        panic!("no attempt captured exactly one warning");
    }

    /// A subscriber that panics on every event, while the SDK logs a wrap
    /// panic: the subscriber's panic is discarded, and the body's result
    /// returns.
    #[test]
    fn subscriber_panic_while_logging_is_discarded() {
        struct Panicking(Arc<AtomicUsize>);
        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Panicking {
            #[expect(clippy::panic, reason = "the subscriber panics on every event")]
            fn on_event(
                &self,
                _event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                self.0.fetch_add(1, Ordering::SeqCst);
                panic!("subscriber boom");
            }
        }
        #[expect(clippy::panic, reason = "the wrap panics after next()")]
        fn wrap_boom() {
            panic!("wrap boom")
        }
        use tracing_subscriber::layer::SubscriberExt as _;
        let events = Arc::new(AtomicUsize::new(0));
        let plugins = test_support::plugins_of(UfPlugin::after_next(wrap_boom));
        let subscriber = tracing_subscriber::registry().with(Panicking(Arc::clone(&events)));
        let result = {
            let _guard = test_tracing::install_test_subscriber(subscriber);
            run_step_once(&plugins)
        };
        assert_eq!(result, Some(7));
        assert!(
            events.load(Ordering::SeqCst) >= 1,
            "the wrap panic was logged"
        );
    }

    /// Work started after the drain closed is dropped unpolled, with a
    /// warning.
    #[tokio::test(start_paused = true)]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    async fn work_started_after_the_drain_closed_is_dropped() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let escaped = shared::<Option<DurableContext>>();
            let (service, event) = service(vec![Hooks::default()], 0, {
                let escaped = Arc::clone(&escaped);
                move |_e, ctx| {
                    let escaped = Arc::clone(&escaped);
                    async move {
                        *escaped.lock().unwrap() = Some(ctx);
                        Ok(serde_json::Value::Null)
                    }
                }
            });
            service(event).await.unwrap();
            let ctx = escaped.lock().unwrap().take().unwrap();
            let polled = Arc::new(AtomicBool::new(false));
            let dropped = Arc::new(AtomicBool::new(false));
            let guard = OnDrop::new({
                let dropped = Arc::clone(&dropped);
                move || dropped.store(true, Ordering::SeqCst)
            });
            start(&ctx, {
                let polled = Arc::clone(&polled);
                async move {
                    let _guard = guard;
                    polled.store(true, Ordering::SeqCst);
                    Ok(())
                }
            });
            tokio::task::yield_now().await;
            assert!(!polled.load(Ordering::SeqCst));
            assert!(dropped.load(Ordering::SeqCst));
            let warnings = warnings(&buffer);
            let [warning] = with_message(
                &warnings,
                "plugin hook work started after the drain closed; it is dropped",
            )[..] else {
                continue;
            };
            assert_eq!(field(warning, "hook"), "on_operation_end");
            assert!(field(warning, "plugin").ends_with("(plugins[0])"));
            return;
        }
        panic!("no attempt captured the warning");
    }

    /// Without a Tokio runtime the work cannot be spawned: it is dropped,
    /// with a warning.
    #[test]
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warning
    fn work_without_a_runtime_is_dropped() {
        for _attempt in 0..5 {
            let (buffer, _guard) = capture();
            let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> =
                vec![("Hooks", Arc::new(Hooks::default()))];
            let plugins = create_plugins(&factories, None, empty_info).unwrap();
            let dropped = Arc::new(AtomicBool::new(false));
            let guard = OnDrop::new({
                let dropped = Arc::clone(&dropped);
                move || dropped.store(true, Ordering::SeqCst)
            });
            plugins.start_work(
                0,
                "on_operation_end",
                Box::pin(async move {
                    let _guard = guard;
                    Ok(())
                }),
            );
            assert!(dropped.load(Ordering::SeqCst));
            let warnings = warnings(&buffer);
            let [warning] = with_message(
                &warnings,
                "plugin hook work could not be spawned; it is dropped",
            )[..] else {
                continue;
            };
            assert_eq!(field(warning, "hook"), "on_operation_end");
            return;
        }
        panic!("no attempt captured the warning");
    }

    /// `start_work` removes the handles of finished work before it adds
    /// one.
    #[tokio::test(start_paused = true)]
    async fn start_work_removes_finished_handles() {
        let factories: Vec<(&'static str, Arc<dyn PluginFactory>)> =
            vec![("Hooks", Arc::new(Hooks::default()))];
        let plugins = create_plugins(&factories, None, empty_info).unwrap();
        plugins.start_work(0, "on_operation_end", Box::pin(async { Ok(()) }));
        tokio::time::sleep(Duration::from_millis(1)).await;
        plugins.start_work(0, "on_operation_end", Box::pin(std::future::pending()));
        let WorkState::Open(list) = &*plugins.lock_work() else {
            unreachable!("the drain has not closed");
        };
        assert_eq!(list.len(), 1);
    }

    /// An error whose destructor panics with `error drop`.
    #[derive(Debug)]
    struct DropPanicError;

    impl fmt::Display for DropPanicError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("drop err")
        }
    }

    impl std::error::Error for DropPanicError {}

    impl Drop for DropPanicError {
        #[expect(clippy::panic)] // reason: the destructor panic under test
        fn drop(&mut self) {
            panic!("error drop");
        }
    }

    /// An error whose destructor panics with its text.
    #[derive(Debug)]
    struct DropPanicDisplayError(&'static str);

    impl fmt::Display for DropPanicDisplayError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for DropPanicDisplayError {}

    impl Drop for DropPanicDisplayError {
        #[expect(clippy::panic)] // reason: the destructor panic under test
        fn drop(&mut self) {
            std::panic::panic_any(self.0);
        }
    }

    /// Runs `service` on its own task and returns its panic payload.
    #[expect(clippy::disallowed_methods)]
    // reason: the test calls the service on its own task so the panic is
    // observable as that task's `JoinError`.
    async fn panic_of<Fut>(future: Fut) -> Payload
    where
        Fut: Future + Send + 'static,
        Fut::Output: Send,
    {
        match tokio::spawn(future).await {
            Ok(_) => unreachable!("the service panicked"),
            Err(join_err) => join_err.into_panic(),
        }
    }

    /// A start hook error whose destructor panics fails the invocation:
    /// the panic is kept until the coordinator exists, and resumes.
    #[tokio::test(start_paused = true)]
    async fn start_hook_error_destructor_panic_fails_the_invocation() {
        let plugin = Hooks::default()
            .on_start(|_| Box::pin(async { Err(Box::new(DropPanicError) as BoxError) }));
        let (service, event) = service(vec![plugin.clone()], 0, |_e, _ctx| async move {
            Ok(serde_json::Value::Null)
        });
        let payload = panic_of(service(event)).await;
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
        let ends = plugin.ends();
        let [end] = ends.as_slice() else {
            unreachable!("one end info, got {}", ends.len());
        };
        assert_eq!(end.status(), InvocationStatus::Retrying);
    }

    /// A hook work error whose destructor panics while the handler runs
    /// fails the invocation: the panic is recorded on the coordinator.
    #[tokio::test(start_paused = true)]
    async fn hook_work_error_destructor_panic_fails_the_invocation() {
        let (service, event) = service(vec![Hooks::default()], 0, |_e, ctx| async move {
            start(&ctx, async { Err(Box::new(DropPanicError) as BoxError) });
            std::future::pending::<()>().await;
            Ok(serde_json::Value::Null)
        });
        let payload = panic_of(service(event)).await;
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
    }

    /// A hook work error whose destructor panics while the drain waits
    /// fails the invocation, as it does while the handler runs. The drive
    /// already returned, so the end hooks see `Retrying` with the panic's
    /// message, and the panic resumes after them.
    #[tokio::test(start_paused = true)]
    async fn hook_work_error_destructor_panic_during_the_drain_fails_the_invocation() {
        let plugin = Hooks::default();
        let (service, event) = service(vec![plugin.clone()], 0, |_e, ctx| async move {
            start(&ctx, async {
                tokio::time::sleep(Duration::from_secs(1)).await;
                Err(Box::new(DropPanicError) as BoxError)
            });
            Ok(serde_json::Value::Null)
        });
        let payload = panic_of(service(event)).await;
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
        let ends = plugin.ends();
        let [end] = ends.as_slice() else {
            unreachable!("one end info, got {}", ends.len());
        };
        assert_eq!(end.status(), InvocationStatus::Retrying);
        assert_eq!(end.execution_result(), None);
        let error = end.execution_error().expect("the panic's error");
        assert_eq!(error.error_type(), None);
        assert_eq!(error.error_message(), Some("error drop"));
    }

    /// An end hook error whose destructor panics fails the invocation. The
    /// end hooks already saw the response's status, so the panic resumes
    /// after them, instead of the response.
    #[tokio::test(start_paused = true)]
    async fn end_hook_error_destructor_panic_fails_the_invocation() {
        let failing = Hooks::default()
            .on_end(|_| Box::pin(async { Err(Box::new(DropPanicError) as BoxError) }));
        let healthy = Hooks::default();
        let (service, event) = service(vec![failing, healthy.clone()], 0, |_e, _ctx| async move {
            Ok(serde_json::json!("ok"))
        });
        let payload = panic_of(service(event)).await;
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
        let ends = healthy.ends();
        let [end] = ends.as_slice() else {
            unreachable!("one end info, got {}", ends.len());
        };
        assert_eq!(end.status(), InvocationStatus::Succeeded);
    }

    /// When a late panic comes from the drain, the end hooks' late panic
    /// is dropped, and the drain's resumes.
    #[tokio::test(start_paused = true)]
    async fn the_first_late_panic_resumes() {
        let failing = Hooks::default().on_end(|_| {
            Box::pin(async { Err(Box::new(DropPanicDisplayError("end drop")) as BoxError) })
        });
        let (service, event) = service(vec![failing], 0, |_e, ctx| async move {
            start(&ctx, async {
                tokio::time::sleep(Duration::from_secs(1)).await;
                Err(Box::new(DropPanicDisplayError("work drop")) as BoxError)
            });
            Ok(serde_json::Value::Null)
        });
        let payload = panic_of(service(event)).await;
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"work drop"));
    }

    /// A late panic, from the drain or from an end hook, never replaces a
    /// panic that ended the drive: the end hooks see the drive's panic,
    /// and it resumes.
    #[tokio::test(start_paused = true)]
    #[expect(clippy::panic)] // reason: the handler panic under test
    async fn late_panic_after_a_drive_panic_is_dropped() {
        let failing = Hooks::default()
            .on_end(|_| Box::pin(async { Err(Box::new(DropPanicError) as BoxError) }));
        let (service, event) = service(vec![failing.clone()], 0, |_e, ctx| async move {
            start(&ctx, async {
                tokio::time::sleep(Duration::from_secs(1)).await;
                Err(Box::new(DropPanicDisplayError("work drop")) as BoxError)
            });
            if ctx.is_replaying() {
                return Ok(serde_json::Value::Null);
            }
            panic!("boom");
        });
        let payload = panic_of(service(event)).await;
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom"));
        let ends = failing.ends();
        let [end] = ends.as_slice() else {
            unreachable!("one end info, got {}", ends.len());
        };
        assert_eq!(
            end.execution_error().and_then(WireError::error_message),
            Some("boom")
        );
    }

    /// A late panic never replaces a fatal error, as the drive keeps a
    /// fatal error over a panic: the `FAILED` envelope returns.
    #[tokio::test(start_paused = true)]
    async fn late_panic_after_a_fatal_error_is_dropped() {
        let failing = Hooks::default()
            .on_end(|_| Box::pin(async { Err(Box::new(DropPanicError) as BoxError) }));
        let (service, event) = service(vec![failing.clone()], 0, |_e, ctx| async move {
            ctx.coordinator()
                .record_fatal("Error".to_owned(), "fatal".to_owned());
            std::future::pending::<()>().await;
            Ok(serde_json::Value::Null)
        });
        let response = service(event).await.expect("the FAILED envelope");
        assert_eq!(
            response.0.get("Status"),
            Some(&serde_json::json!("FAILED")),
            "{:?}",
            response.0
        );
        let ends = failing.ends();
        let [end] = ends.as_slice() else {
            unreachable!("one end info, got {}", ends.len());
        };
        assert_eq!(end.status(), InvocationStatus::Failed);
    }

    /// Counts a poll after the end hook started as a violation.
    struct Checked<F> {
        inner: Pin<Box<F>>,
        ended: Arc<AtomicBool>,
        violations: Arc<AtomicUsize>,
        _live: OnDrop,
    }
    impl<F: Future<Output = Result<(), BoxError>>> Future for Checked<F> {
        type Output = Result<(), BoxError>;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if self.ended.load(Ordering::SeqCst) {
                self.violations.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.as_mut().poll(cx)
        }
    }
    #[derive(Clone)]
    struct Round {
        ended: Arc<AtomicBool>,
        violations: Arc<AtomicUsize>,
        live: Arc<AtomicUsize>,
    }
    impl Round {
        /// Work that sleeps 1 ms, then, while `depth` is above 0, starts
        /// more work.
        fn work(&self, ctx: DurableContext, depth: u32) -> HookFuture<'static> {
            self.live.fetch_add(1, Ordering::SeqCst);
            let live = Arc::clone(&self.live);
            let round = self.clone();
            Box::pin(Checked {
                inner: Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    if depth > 0 {
                        let next = round.work(ctx.clone(), depth - 1);
                        ctx.plugins()
                            .unwrap()
                            .start_work(0, "on_operation_end", next);
                    }
                    Ok(())
                }),
                ended: Arc::clone(&self.ended),
                violations: Arc::clone(&self.violations),
                _live: OnDrop::new(move || {
                    live.fetch_sub(1, Ordering::SeqCst);
                }),
            })
        }
    }

    /// A `start_work` call racing the drain either lands in a list the
    /// drain takes, or sees the drain closed. So no work runs once the end
    /// hook started.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[expect(clippy::disallowed_methods)]
    // reason: the starters run on their own tasks, outside the SDK, to
    // race the drain from other threads.
    async fn late_start_racing_the_drain_never_runs_after_end_starts() {
        for round_index in 0..20 {
            let round = Round {
                ended: Arc::new(AtomicBool::new(false)),
                violations: Arc::new(AtomicUsize::new(0)),
                live: Arc::new(AtomicUsize::new(0)),
            };
            let plugin = {
                let ended = Arc::clone(&round.ended);
                Hooks::default().on_end(move |_| {
                    ended.store(true, Ordering::SeqCst);
                    ready()
                })
            };
            let starters = shared::<Vec<JoinHandle<()>>>();
            // The starters keep the work list from emptying, so only the
            // drain deadline, 200 ms in, closes the drain.
            let (service, event) = service(vec![plugin], deadline_in(700), {
                let round = round.clone();
                let starters = Arc::clone(&starters);
                move |_e, ctx| {
                    let round = round.clone();
                    let starters = Arc::clone(&starters);
                    async move {
                        for _ in 0..4 {
                            ctx.plugins().unwrap().start_work(
                                0,
                                "on_operation_end",
                                round.work(ctx.clone(), 5),
                            );
                        }
                        for _ in 0..3 {
                            let ctx = ctx.clone();
                            let round = round.clone();
                            starters.lock().unwrap().push(tokio::spawn(async move {
                                let mut after_end: Option<std::time::Instant> = None;
                                loop {
                                    let work = round.work(ctx.clone(), 1);
                                    ctx.plugins()
                                        .unwrap()
                                        .start_work(0, "on_operation_end", work);
                                    if round.ended.load(Ordering::SeqCst) {
                                        let since =
                                            *after_end.get_or_insert_with(std::time::Instant::now);
                                        if since.elapsed() > Duration::from_millis(20) {
                                            break;
                                        }
                                    }
                                    tokio::task::yield_now().await;
                                }
                            }));
                        }
                        Ok(serde_json::Value::Null)
                    }
                }
            });
            service(event).await.unwrap();
            let handles = std::mem::take(&mut *starters.lock().unwrap());
            for handle in handles {
                handle.await.unwrap();
            }
            assert_eq!(
                round.violations.load(Ordering::SeqCst),
                0,
                "round {round_index}"
            );
            assert_eq!(round.live.load(Ordering::SeqCst), 0, "round {round_index}");
        }
    }

    // Shutdown: the plugins are dropped, or the invocation future is
    // dropped before it returned.

    /// Sets its flag when dropped.
    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Plugins with one plugin, outside an invocation.
    fn one_plugin() -> Arc<InvocationPlugins> {
        let log = Arc::new(Mutex::new(Vec::new()));
        create_plugins(
            &factories(vec![("A", Behavior::Some)], &log),
            None,
            empty_info,
        )
        .unwrap()
    }

    /// Work that sets `flag` when it is dropped, then panics with a
    /// payload whose destructor panics too. It never finishes.
    #[expect(clippy::panic)] // reason: the destructor panic under test
    fn pending_work_dropping(
        flag: Arc<AtomicBool>,
    ) -> impl Future<Output = Result<(), BoxError>> + Send + 'static {
        let guard = OnDrop::new({
            move || {
                flag.store(true, Ordering::SeqCst);
                std::panic::panic_any(PanickingPayload);
            }
        });
        async move {
            let _guard = guard;
            std::future::pending::<Result<(), BoxError>>().await
        }
    }

    /// Lets the runtime run the tasks an abort cancelled.
    async fn settle() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    /// Dropping the plugins aborts their hook work. A detached task would
    /// keep running the work after its invocation.
    #[tokio::test(start_paused = true)]
    async fn dropping_the_plugins_aborts_hook_work() {
        let dropped = Arc::new(AtomicBool::new(false));
        let plugins = one_plugin();
        plugins.start_work(
            0,
            "on_operation_end",
            Box::pin(pending_work_dropping(Arc::clone(&dropped))),
        );
        settle().await;
        assert!(!dropped.load(Ordering::SeqCst), "the work runs");
        drop(plugins);
        settle().await;
        assert!(dropped.load(Ordering::SeqCst), "the work was aborted");
    }

    /// Dropping the invocation future before it returned aborts the hook
    /// work, even while a context clone that escaped the handler keeps the
    /// plugins alive.
    #[tokio::test(start_paused = true)]
    async fn cancelled_invocation_aborts_hook_work() {
        let dropped = Arc::new(AtomicBool::new(false));
        let escaped = shared::<Option<DurableContext>>();
        let (service, event) = service(vec![Hooks::default()], 0, {
            let dropped = Arc::clone(&dropped);
            let escaped = Arc::clone(&escaped);
            move |_e, ctx| {
                let dropped = Arc::clone(&dropped);
                let escaped = Arc::clone(&escaped);
                async move {
                    start(&ctx, pending_work_dropping(dropped));
                    *escaped.lock().unwrap() = Some(ctx);
                    std::future::pending::<()>().await;
                    Ok(serde_json::Value::Null)
                }
            }
        });
        let cancelled = tokio::time::timeout(Duration::from_secs(1), service(event)).await;
        assert!(cancelled.is_err(), "the invocation was cancelled");
        settle().await;
        let ctx = escaped.lock().unwrap().take().expect("the handler ran");
        assert!(
            ctx.plugins().is_some(),
            "the escaped context keeps the plugins"
        );
        assert!(dropped.load(Ordering::SeqCst), "the work was aborted");
    }

    /// Dropping the invocation future while the drain waits for the hook
    /// work aborts that work. The drain has taken the handles out of the
    /// work state then, so the shutdown alone would find none to abort.
    #[tokio::test(start_paused = true)]
    async fn cancelled_drain_aborts_hook_work() {
        let dropped = Arc::new(AtomicBool::new(false));
        let escaped = shared::<Option<DurableContext>>();
        let (service, event) = service(vec![Hooks::default()], 0, {
            let dropped = Arc::clone(&dropped);
            let escaped = Arc::clone(&escaped);
            move |_e, ctx| {
                let dropped = Arc::clone(&dropped);
                let escaped = Arc::clone(&escaped);
                async move {
                    start(&ctx, pending_work_dropping(dropped));
                    *escaped.lock().unwrap() = Some(ctx);
                    Ok(serde_json::Value::Null)
                }
            }
        });
        let cancelled = tokio::time::timeout(Duration::from_secs(1), service(event)).await;
        assert!(cancelled.is_err(), "the drain waited until the cancel");
        settle().await;
        let ctx = escaped.lock().unwrap().take().expect("the handler ran");
        assert!(
            ctx.plugins().is_some(),
            "the escaped context keeps the plugins"
        );
        assert!(dropped.load(Ordering::SeqCst), "the work was aborted");
    }

    /// After the plugins shut down, a recorded panic is dropped at once,
    /// through `discard_panic`: an aborted hook task can still report an
    /// error destructor panic on another thread.
    #[test]
    fn panic_recorded_after_shut_down_is_dropped() {
        let plugins = one_plugin();
        plugins.shut_down();
        let dropped = Arc::new(AtomicBool::new(false));
        Undelivered::record(
            &plugins.undelivered,
            Box::new(DropFlag(Arc::clone(&dropped))),
        );
        assert!(
            dropped.load(Ordering::SeqCst),
            "the payload was dropped at once"
        );
        let nested = catch_unwind(AssertUnwindSafe(|| {
            Undelivered::record(&plugins.undelivered, Box::new(PanickingPayload));
        }));
        assert!(
            nested.is_ok(),
            "a nested payload's destructor panic escaped"
        );
        assert!(plugins.take_error_drop_panics().is_empty());
    }

    /// A panic the coordinator returned after the plugins shut down is
    /// dropped through `discard_panic`, not kept.
    #[test]
    fn late_panic_after_shut_down_is_dropped() {
        let state = Mutex::new(Undelivered::Discarding);
        let dropped = Arc::new(AtomicBool::new(false));
        Undelivered::keep_late(&state, Box::new(DropFlag(Arc::clone(&dropped))));
        assert!(
            dropped.load(Ordering::SeqCst),
            "the payload was dropped at once"
        );
        let nested = catch_unwind(AssertUnwindSafe(|| {
            Undelivered::keep_late(&state, Box::new(PanickingPayload));
        }));
        assert!(
            nested.is_ok(),
            "a nested payload's destructor panic escaped"
        );
    }

    /// A payload that holds the plugins.
    struct HoldsPlugins(
        #[expect(dead_code, reason = "held only to be dropped")] Arc<InvocationPlugins>,
    );

    /// A late panic is dropped after the state's lock is released. The
    /// payload here holds the last reference to the plugins, so dropping
    /// it drops the plugins, and their shutdown locks the state again.
    /// Under the lock, that would deadlock. The test runs on its own
    /// thread, so a deadlock fails it instead of hanging the suite.
    #[test]
    fn late_panic_holding_the_plugins_is_dropped_after_the_lock() {
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let plugins = one_plugin();
            let undelivered = Arc::clone(&plugins.undelivered);
            plugins.shut_down();
            Undelivered::keep_late(&undelivered, Box::new(HoldsPlugins(plugins)));
            let _ = done.send(());
        });
        assert!(
            finished.recv_timeout(Duration::from_secs(30)).is_ok(),
            "dropping the payload deadlocked on the state's lock"
        );
    }

    /// A panic kept before the shutdown is dropped by it, through
    /// `discard_panic`.
    #[test]
    fn shut_down_drops_kept_panics() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let factories = factories(vec![("bad", Behavior::ErrorPanickingNestedDrop)], &log);
        let plugins = create_plugins(&factories, None, empty_info).unwrap();
        let shut = catch_unwind(AssertUnwindSafe(|| plugins.shut_down()));
        assert!(shut.is_ok(), "a nested payload's destructor panic escaped");
        assert!(plugins.take_error_drop_panics().is_empty());
    }

    // Propagation hook.

    use super::test_support::{TraceHook, trace_plugins};

    const CONFLICT: &str = "conflicting propagation trace ids; keeping the first value";
    const TRACE_PANICKED: &str = "plugin propagation_trace_id panicked; the value is ignored";
    const TRACE_FAILED: &str = "plugin propagation_trace_id failed; the value is ignored";

    /// The input every propagation test passes.
    fn trace_input() -> PropagationInput {
        PropagationInput::new(
            "arn:exec".to_owned(),
            "op-1".to_owned(),
            Some("parent-1".to_owned()),
            "target-fn".to_owned(),
        )
    }

    /// Asks `hooks` for a trace header, and returns the header and the
    /// plugin-target warnings.
    ///
    /// The scenario retries until it captures `expected` warnings, as the
    /// other capture tests of this module do: a concurrent test can
    /// rebuild `tracing`'s callsite-interest cache and drop a warning
    /// before it reaches this thread's subscriber. The returned header is
    /// the same on every attempt.
    #[expect(clippy::panic)] // reason: reports that no attempt captured the warnings
    fn trace_id_of(
        hooks: &[TraceHook],
        expected: usize,
    ) -> (Option<String>, Vec<serde_json::Value>) {
        for _attempt in 0..5 {
            let (buffer, guard) = capture();
            let value = trace_plugins(hooks).propagation_trace_id(&trace_input());
            drop(guard);
            let warnings = warnings(&buffer);
            if warnings.len() == expected {
                return (value, warnings);
            }
            assert!(
                warnings.len() < expected,
                "more warnings than expected: {warnings:?}"
            );
        }
        panic!("no attempt captured {expected} warnings");
    }

    #[test]
    fn trace_id_none_without_answers() {
        let (value, warnings) = trace_id_of(&[TraceHook::value(None), TraceHook::value(None)], 0);
        assert_eq!(value, None);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A plugin that answers no value does not end the loop: every later
    /// plugin is asked, and a conflict names the plugins that gave the two
    /// values, not the first plugin asked.
    #[test]
    fn trace_id_none_continues_to_later_plugins() {
        let hooks = [
            TraceHook::value(None),
            TraceHook::value(Some("Root=1-a")),
            TraceHook::value(None),
            TraceHook::value(Some("Root=1-b")),
        ];
        let (value, warnings) = trace_id_of(&hooks, 1);
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        // `trace_id_of` may retry, and each attempt asks every plugin once,
        // so every hook was called the same, non-zero number of times.
        let calls: Vec<usize> = hooks.iter().map(|hook| hook.inputs().len()).collect();
        let first = calls.first().copied().unwrap_or(0);
        assert!(first > 0, "{calls:?}");
        assert!(calls.iter().all(|&count| count == first), "{calls:?}");
        let [conflict] = with_message(&warnings, CONFLICT)[..] else {
            unreachable!("one conflict: {warnings:?}");
        };
        assert_eq!(field(conflict, "first_plugin"), "TraceHook (plugins[1])");
        assert_eq!(field(conflict, "later_plugin"), "TraceHook (plugins[3])");
        assert_eq!(conflict.get("conflict_count"), Some(&serde_json::json!(1)));
    }

    #[test]
    fn trace_id_first_non_blank_wins() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::value(Some("Root=1-a")),
                TraceHook::value(Some("Root=1-b")),
            ],
            1,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        let [conflict] = with_message(&warnings, CONFLICT)[..] else {
            unreachable!("one conflict: {warnings:?}");
        };
        assert_eq!(field(conflict, "field"), "x_amzn_trace_id");
        assert!(
            field(conflict, "first_plugin").ends_with("(plugins[0])"),
            "{conflict}"
        );
        assert!(
            field(conflict, "later_plugin").ends_with("(plugins[1])"),
            "{conflict}"
        );
        assert_eq!(conflict.get("conflict_count"), Some(&serde_json::json!(1)));
    }

    /// Each conflict of one call counts from 1, and every later value is
    /// compared with the first value.
    #[test]
    fn trace_id_conflicts_are_counted() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::value(Some("Root=1-a")),
                TraceHook::value(Some("Root=1-b")),
                TraceHook::value(Some("Root=1-c")),
            ],
            2,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        let conflicts = with_message(&warnings, CONFLICT);
        let counts: Vec<_> = conflicts
            .iter()
            .map(|w| {
                (
                    w.get("conflict_count").cloned(),
                    field(w, "later_plugin").to_owned(),
                )
            })
            .collect();
        assert_eq!(
            counts,
            [
                (
                    Some(serde_json::json!(1)),
                    "TraceHook (plugins[1])".to_owned()
                ),
                (
                    Some(serde_json::json!(2)),
                    "TraceHook (plugins[2])".to_owned()
                ),
            ]
        );
        for conflict in conflicts {
            assert_eq!(field(conflict, "first_plugin"), "TraceHook (plugins[0])");
        }
    }

    #[test]
    fn trace_id_blank_is_skipped() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::value(Some("  ")),
                TraceHook::value(Some("Root=1-a")),
            ],
            0,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A blank value after a kept value is no value, not a conflict.
    #[test]
    fn trace_id_blank_after_value_is_no_conflict() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::value(Some("Root=1-a")),
                TraceHook::value(Some("")),
            ],
            0,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn trace_id_same_value_is_no_conflict() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::value(Some("Root=1-a")),
                TraceHook::value(Some("Root=1-a")),
            ],
            0,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn trace_id_kept_unchanged() {
        let (value, warnings) = trace_id_of(&[TraceHook::value(Some(" Root=1-a "))], 0);
        assert_eq!(value.as_deref(), Some(" Root=1-a "));
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    #[expect(clippy::panic)] // reason: the plugin panic under test
    fn trace_id_panic_is_ignored() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::new(|_| panic!("trace boom")),
                TraceHook::value(Some("Root=1-a")),
            ],
            1,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        let [panicked] = with_message(&warnings, TRACE_PANICKED)[..] else {
            unreachable!("one panic record: {warnings:?}");
        };
        assert!(
            field(panicked, "plugin").ends_with("(plugins[0])"),
            "{panicked}"
        );
        assert_eq!(field(panicked, "hook"), "propagation_trace_id");
    }

    #[test]
    fn propagation_err_is_logged_and_skipped() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::new(|_| Err("x".into())),
                TraceHook::value(Some("Root=1-a")),
            ],
            1,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        let [failed] = with_message(&warnings, TRACE_FAILED)[..] else {
            unreachable!("one failure record: {warnings:?}");
        };
        assert!(
            field(failed, "plugin").ends_with("(plugins[0])"),
            "{failed}"
        );
        assert_eq!(field(failed, "error"), "x");
        assert_eq!(field(failed, "hook"), "propagation_trace_id");
    }

    /// An error whose `Display` panics is logged without its text, and the
    /// next plugin's value is used.
    #[test]
    fn propagation_err_display_panic_is_logged() {
        let (value, warnings) = trace_id_of(
            &[
                TraceHook::new(|_| Err(Box::new(PanickingDisplayError))),
                TraceHook::value(Some("Root=1-a")),
            ],
            1,
        );
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        let [unformatted] = with_message(
            &warnings,
            "plugin propagation_trace_id failed, and its error panicked while it was \
             formatted; the value is ignored",
        )[..] else {
            unreachable!("one record: {warnings:?}");
        };
        assert_eq!(field(unformatted, "plugin"), "TraceHook (plugins[0])");
        assert_eq!(field(unformatted, "hook"), "propagation_trace_id");
    }

    /// `propagation_trace_id` is plugin code: a durable operation it calls
    /// records the context usage error that names plugin code.
    #[test]
    fn trace_id_hook_calling_an_operation_fails_the_execution() {
        let ctx = DurableContext::test_context();
        let captured = ctx.clone();
        let plugins = trace_plugins(&[TraceHook::new(move |_| {
            drop(captured.claim_operation_id("wait"));
            Ok(None)
        })]);
        assert_eq!(plugins.propagation_trace_id(&trace_input()), None);
        let fatal = ctx.coordinator().fatal_error().expect("fatal recorded");
        assert!(
            fatal.error_message.contains("inside plugin code"),
            "{}",
            fatal.error_message
        );
    }

    /// The SDK drops an error it delivers to nobody inside a catch, and
    /// records a panic of its destructor, which fails the invocation.
    #[test]
    fn propagation_err_destructor_panic_is_recorded() {
        let plugins = trace_plugins(&[
            TraceHook::new(|_| Err(Box::new(DropPanicError))),
            TraceHook::value(Some("Root=1-a")),
        ]);
        let value = plugins.propagation_trace_id(&trace_input());
        assert_eq!(value.as_deref(), Some("Root=1-a"));
        let kept = plugins.take_error_drop_panics();
        let [payload] = kept.as_slice() else {
            unreachable!("one kept panic, got {}", kept.len());
        };
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"error drop"));
    }

    #[test]
    fn trace_id_input_fields() {
        let hook = TraceHook::value(None);
        let value = trace_plugins(std::slice::from_ref(&hook)).propagation_trace_id(&trace_input());
        assert_eq!(value, None);
        let inputs = hook.inputs();
        let [input] = inputs.as_slice() else {
            unreachable!("one call, got {}", inputs.len());
        };
        assert_eq!(input.execution_arn(), "arn:exec");
        assert_eq!(input.operation_id(), "op-1");
        assert_eq!(input.parent_operation_id(), Some("parent-1"));
        assert_eq!(input.target_function_name(), "target-fn");
    }
}
