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
//! - wrap hooks that run around the invocation, a step attempt or
//!   condition check, a child context body, and each SDK operation task;
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

use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll};
use std::time::SystemTime;

use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
use crate::{BoxError, InvocationStatus, OperationKind, OperationStatus, WireError};

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
/// One instance serves exactly one invocation and is dropped when the
/// invocation returns. Every method has a no-op default, so a later SDK
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
/// No hook future may block a thread, in `poll` or in `Drop`. The futures
/// of [`wrap_task`](Self::wrap_task) and of the operation hooks are
/// `'static`; a plugin that needs its own state there clones an owned
/// handle, such as an `Arc` field, into the future.
///
/// # Errors and panics
///
/// An `Err` or a panic from any hook, factory or hook future is logged and
/// never changes the execution. Catching a panic relies on unwinding, so a
/// binary built with `panic = "abort"` aborts instead.
///
/// Every such warning is logged on the target
/// `aws_durable_execution_sdk::plugin`, with the field
/// `plugin = "<name()> (plugins[<i>])"` and the field
/// `hook = "<method name>"`. A factory's warning has the field
/// `factory = "<factory type> (plugins[<i>])"` instead. `<i>` is the
/// 0-based registration index of the factory.
///
/// # Wrap hooks
///
/// [`wrap_invocation`](Self::wrap_invocation),
/// [`wrap_attempt`](Self::wrap_attempt),
/// [`wrap_child_context`](Self::wrap_child_context) and
/// [`wrap_task`](Self::wrap_task) receive a [`Next`] that runs the inner
/// work: the inner wrappers and the body. If a wrapper returns, returns
/// `Err`, panics or drops `next`, the body runs once and its value is the
/// operation's result. A wrapper that stays pending delays the body (before
/// `next`) or the outcome checkpoint (after `next`) until it finishes, as in
/// JS; the SDK sets no time limit.
///
/// 1. The first registered plugin's wrapper is outermost. The SDK calls a
///    wrapper's method when its link is first polled, so an outer wrapper's
///    code before `next` runs before an inner wrapper is called.
/// 2. `next.await` runs the inner wrappers and the body. It returns
///    `Ok(())` when the body succeeded, or `Err` with the [`WireError`] the
///    SDK records for the body's error. The operation's result is always
///    the body's. A wrapper cannot inspect the typed successful value.
///    [`on_operation_end`](Self::on_operation_end) can inspect the
///    serialized checkpoint payload through [`OperationInfo::result`] when
///    the terminal update carries one; it is `None` when that update
///    carries no payload. An object-safe `Plugin` cannot expose a method
///    generic over each operation's output type; keeping typed inline
///    observation would need another erased-value hook. The SDK provides
///    the smaller API.
/// 3. The SDK calls the user closure inside the body future. So the
///    closure's synchronous part runs at the first poll of `next`, inside
///    every wrapper that awaits it.
/// 4. A wrapper that ends without leaving the body to finish inside it is
///    contained: it returns `Err`, panics (in the method or in its future),
///    drops `next`, or returns `Ok` without awaiting `next`. The SDK then
///    polls the rest of the chain itself: the inner wrappers keep running,
///    and the body runs to its end. `Err` and panics are logged. Returning
///    `Ok` without awaiting `next` is legal and logs nothing.
/// 5. A wrapper that stays pending is not contained. Pending before it
///    polls `next`, it delays the inner wrappers and the body. Pending
///    after `next` completed, it delays the outcome checkpoint. The delay
///    lasts as long as the wrapper stays pending, as in JS. The SDK sets no
///    time limit. [`WireError`] does not implement `std::error::Error`, so
///    `next.await?` does not compile in a hook future: match on
///    `next.await`, or ignore it with `let _ = next.await;`.
/// 6. The operation continues after every wrapper has finished. A
///    wrapper's work after `next` runs before the SDK serializes the result
///    or writes the outcome checkpoint.
/// 7. A body panic is never visible to a wrapper. `next` stays pending, the
///    SDK drops every wrapper future, and then resumes the original
///    payload.
/// 8. Dropping the operation (suspension, early completion) drops every
///    wrapper and the body. No wrapper code runs after that, and no
///    reference cycle keeps the body alive.
/// 9. [`wrap_task`](Self::wrap_task) runs on the task that starts a tracked
///    SDK operation task (`.spawn()`, a map item, a parallel branch, a
///    combinator input), before the new task runs. Its future runs on the
///    new task around all of the task's work, outside that task's own
///    wrappers. It is never called for the SDK's detached work (the
///    checkpoint flusher, detached writes) or for hook work. The SDK
///    carries plugin context only onto the tracked operation tasks it
///    starts. A task the user starts is outside SDK propagation and must
///    use the relevant explicit carrier: `tracing` instrumentation,
///    capturing the OpenTelemetry `Context::current()` and calling
///    `.with_context(captured)` on the future, or the plugin's own
///    mechanism.
pub trait Plugin: Send + Sync {
    /// The name the SDK uses for this plugin in its warnings. The default
    /// is the implementing type's path.
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }

    /// Runs when the invocation starts, before the handler. The SDK awaits
    /// the returned future.
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        Box::pin(std::future::ready(Ok(())))
    }

    /// Runs when the invocation ends, after its last checkpoint. The SDK
    /// awaits the returned future.
    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        Box::pin(std::future::ready(Ok(())))
    }

    /// Runs when an operation starts or replays.
    ///
    /// # Errors
    ///
    /// An error is logged and does not change the execution.
    fn on_operation_start(&self, _info: &OperationInfo) -> HookWork {
        Ok(None)
    }

    /// Runs when an operation ends or replays its recorded outcome.
    ///
    /// # Errors
    ///
    /// An error is logged and does not change the execution.
    fn on_operation_end(&self, _info: &OperationInfo) -> HookWork {
        Ok(None)
    }

    /// Runs when a step attempt or a condition check starts.
    ///
    /// # Errors
    ///
    /// An error is logged and does not change the execution.
    fn on_attempt_start(&self, _info: &AttemptInfo) -> HookWork {
        Ok(None)
    }

    /// Runs when a step attempt or a condition check ends.
    ///
    /// # Errors
    ///
    /// An error is logged and does not change the execution.
    fn on_attempt_end(&self, _info: &AttemptEndInfo) -> HookWork {
        Ok(None)
    }

    /// Runs when a checkpoint response reports operations the service
    /// changed.
    ///
    /// # Errors
    ///
    /// An error is logged and does not change the execution.
    fn on_operation_change(&self, _info: &OperationChangeInfo) -> HookWork {
        Ok(None)
    }

    /// Wraps the whole invocation: the handler and everything it awaits.
    /// See the wrap hooks section of [`Plugin`].
    fn wrap_invocation<'a>(&'a self, _info: &'a InvocationInfo, next: Next<'a>) -> HookFuture<'a> {
        Box::pin(async move {
            let _ = next.await;
            Ok(())
        })
    }

    /// Wraps one step attempt or condition check. See the wrap hooks
    /// section of [`Plugin`].
    fn wrap_attempt<'a>(&'a self, _info: &'a AttemptInfo, next: Next<'a>) -> HookFuture<'a> {
        Box::pin(async move {
            let _ = next.await;
            Ok(())
        })
    }

    /// Wraps one child context body. See the wrap hooks section of
    /// [`Plugin`].
    fn wrap_child_context<'a>(
        &'a self,
        _info: &'a OperationInfo,
        next: Next<'a>,
    ) -> HookFuture<'a> {
        Box::pin(async move {
            let _ = next.await;
            Ok(())
        })
    }

    /// Wraps all the work of one tracked SDK operation task. See the wrap
    /// hooks section of [`Plugin`], item 9.
    fn wrap_task(&self, next: Next<'static>) -> HookFuture<'static> {
        Box::pin(async move {
            let _ = next.await;
            Ok(())
        })
    }

    /// Returns the X-Ray trace header the SDK puts on a chained invoke's
    /// `START` checkpoint, or `None` for no header.
    ///
    /// # Errors
    ///
    /// An error is logged, and the invoke starts without a header from this
    /// plugin.
    fn propagation_trace_id(&self, _input: &PropagationInput) -> Result<Option<String>, BoxError> {
        Ok(None)
    }
}

/// The wrap chain behind a [`Next`]. The wrap engine of the issue "Call the
/// plugin wrap hooks around the invocation, step attempts, child bodies and
/// SDK tasks" fills it in.
struct Chain<'a> {
    _lifetime: std::marker::PhantomData<&'a ()>,
}

/// The rest of a wrap chain: the inner wrappers and the body.
///
/// Await it in a wrapper to run the inner work. It resolves with `Ok(())`
/// when the body succeeded, or with the [`WireError`] the SDK records for
/// the body's error. See the wrap hooks section of [`Plugin`].
#[must_use = "futures do nothing unless awaited"]
#[non_exhaustive]
pub struct Next<'a> {
    #[expect(
        dead_code,
        reason = "read by the wrap engine of the issue that calls the wrap hooks"
    )]
    chain: Weak<Chain<'a>>,
    #[expect(
        dead_code,
        reason = "read by the wrap engine of the issue that calls the wrap hooks"
    )]
    link: usize,
}

impl Future for Next<'_> {
    type Output = Result<(), WireError>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // No `Next` exists before the wrap engine builds one, so this body
        // is never reached. The wrap engine replaces it.
        Poll::Pending
    }
}

impl fmt::Debug for Next<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Next")
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

impl OperationInfo {
    /// Converts the checkpoint record stored under the wire ID `id`.
    ///
    /// The result comes from the record's result, else from its invoke
    /// result. The error comes from the record's error fields when any is
    /// present, else from its invoke error fields when any is present, else
    /// it is `None`. An attempt of 0 means the service sent none, so it
    /// gives `None`.
    pub(crate) fn from_record(id: &str, record: &CheckpointRecord) -> Self {
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
    /// `NestingMode::Flat` item or branch, or a virtual child context.
    /// `false` for every other operation and for every entry of an
    /// operations map.
    pub fn is_virtual(&self) -> bool {
        self.is_virtual
    }
}

/// What a plugin knows about one step attempt or condition check when it
/// starts.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AttemptInfo {
    operation: OperationInfo,
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
/// the log, and skips any `EXECUTION` record.
///
/// The bootstrap parsers drop the `EXECUTION` record, but a checkpoint
/// response can insert one into the log. So this function skips it itself;
/// the `EXECUTION` slot of an operations map comes from the bootstrap
/// state instead.
pub(crate) fn operation_infos(log: &CheckpointLog) -> BTreeMap<String, OperationInfo> {
    log.project_records(OperationInfo::from_record)
        .into_iter()
        .filter(|info| info.kind != Some(OperationKind::Execution))
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

/// The plugins of one invocation and what they share.
///
/// The invocation future holds the value in a local until it returns, so
/// every plugin instance is dropped when its invocation returns. Its
/// `Drop` drops each plugin inside its own catch, so a plugin destructor
/// that panics never changes the invocation's outcome.
pub(crate) struct InvocationPlugins {
    #[expect(dead_code, reason = "read by the invocation hook issue")]
    info: InvocationInfo,
    plugins: Vec<PluginSlot>,
    #[expect(dead_code, reason = "read by the invocation hook issue")]
    execution: Mutex<Option<OperationInfo>>,
    /// The panics of factory error destructors, which fail the invocation.
    /// The plugins are created before the invocation coordinator exists, so
    /// the invocation records them on the coordinator once it does.
    error_drop_panics: Mutex<Vec<Payload>>,
}

impl InvocationPlugins {
    /// Returns the panics that factory error destructors raised, and
    /// leaves none behind.
    pub(crate) fn take_error_drop_panics(&self) -> Vec<Payload> {
        std::mem::take(
            &mut *self
                .error_drop_panics
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

impl Drop for InvocationPlugins {
    /// Drops every plugin, in registration order, each inside its own
    /// catch.
    ///
    /// A plugin's destructor is plugin code, and a plugin panic never
    /// changes the execution. So a destructor panic is logged and its
    /// payload is dropped through [`crate::task::discard_panic`]. Each
    /// plugin has its own catch, because a second panic while the first
    /// one unwinds a shared drop would abort the process.
    fn drop(&mut self) {
        for PluginSlot { plugin, identity } in std::mem::take(&mut self.plugins) {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(move || drop(plugin))) {
                tracing::warn!(
                    target: TARGET,
                    plugin = %identity,
                    hook = "drop",
                    "plugin panicked while it was dropped; the panic is ignored"
                );
                crate::task::discard_panic(payload);
            }
        }
        // A panic nobody took is dropped without escaping.
        for payload in self.take_error_drop_panics() {
            crate::task::discard_panic(payload);
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
        error_drop_panics: Mutex::new(error_drop_panics),
    }))
}

/// Calls every factory once, in order, and keeps the plugins they return.
///
/// Each factory call runs inside `catch_unwind`.
///
/// - `Ok(Some(plugin))` is kept. `plugin.name()` runs inside its own catch,
///   and a panic there gives the identity `plugins[<i>]`.
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
        match catch_unwind(AssertUnwindSafe(|| factory.create_plugin(info))) {
            Ok(Ok(Some(plugin))) => {
                let plugin: Arc<dyn Plugin> = Arc::from(plugin);
                let identity = match catch_unwind(AssertUnwindSafe(|| plugin.name())) {
                    Ok(name) => format!("{name} (plugins[{index}])"),
                    Err(payload) => {
                        crate::task::discard_panic(payload);
                        format!("plugins[{index}]")
                    }
                };
                slots.push(PluginSlot { plugin, identity });
            }
            Ok(Ok(None)) => {}
            Ok(Err(error)) => log_factory_error(&factory_identity, error, error_drop_panics),
            Err(payload) => {
                tracing::warn!(
                    target: TARGET,
                    factory = %factory_identity,
                    "plugin factory panicked; the plugin is skipped for this invocation"
                );
                crate::task::discard_panic(payload);
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
    match catch_unwind(AssertUnwindSafe(|| error.to_string())) {
        Ok(message) => tracing::warn!(
            target: TARGET,
            factory = %factory_identity,
            error = %message,
            "plugin factory failed; the plugin is skipped for this invocation"
        ),
        Err(payload) => {
            tracing::warn!(
                target: TARGET,
                factory = %factory_identity,
                "plugin factory failed, and its error panicked while it was formatted; \
                 the plugin is skipped for this invocation"
            );
            crate::task::discard_panic(payload);
        }
    }
    if let Err(payload) = catch_unwind(AssertUnwindSafe(move || drop(error))) {
        error_drop_panics.push(payload);
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

    /// Captures the warnings on the plugin target as JSON lines.
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

    fn warnings(buffer: &Arc<Mutex<Vec<u8>>>) -> Vec<serde_json::Value> {
        String::from_utf8_lossy(&buffer.lock().unwrap())
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|v| {
                v.get("target").and_then(serde_json::Value::as_str) == Some(TARGET)
                    && v.get("level").and_then(serde_json::Value::as_str) == Some("WARN")
            })
            .collect()
    }

    fn field<'a>(event: &'a serde_json::Value, name: &str) -> &'a str {
        event
            .get(name)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
    }

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

    #[test]
    fn operation_infos_skips_execution_record() {
        let log = CheckpointLog::from_records(vec![
            (
                "exec".to_owned(),
                record("EXECUTION", CheckpointStatus::Started),
            ),
            ("s".to_owned(), record("STEP", CheckpointStatus::Started)),
        ]);
        let map = operation_infos(&log);
        assert_eq!(map.keys().collect::<Vec<_>>(), ["s"]);
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

    /// Every hook that is not a wrap hook has a no-op default: the async
    /// hooks resolve with `Ok(())`, the operation hooks return no work, and
    /// the propagation hook returns no header. A factory has no group by
    /// default.
    #[tokio::test]
    async fn default_hooks_do_nothing() {
        let operation = OperationInfo::from_record("a", &record("STEP", CheckpointStatus::Started));
        let attempt = AttemptInfo {
            operation: operation.clone(),
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

    /// The declared `Next` has no chain to run yet, so it stays pending,
    /// and a default wrap hook that awaits it stays pending too.
    #[test]
    fn next_without_chain_stays_pending() {
        let waker = std::task::Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut next = Next {
            chain: Weak::new(),
            link: 0,
        };
        assert!(Pin::new(&mut next).poll(&mut cx).is_pending());
        let mut wrapped = Probe.wrap_task(Next {
            chain: Weak::new(),
            link: 0,
        });
        assert!(wrapped.as_mut().poll(&mut cx).is_pending());
    }

    /// `Next` prints as `Next`.
    #[test]
    fn next_debug_writes_next() {
        let next = Next {
            chain: Weak::new(),
            link: 0,
        };
        assert_eq!(format!("{next:?}"), "Next");
    }
}
