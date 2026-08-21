//! In-process local testing runner (`test-util` feature).
//!
//! [`LocalRunner`] drives a durable handler to completion entirely in
//! memory: each simulated invocation goes through the **same service
//! function production uses** — the one [`wrap`](crate::wrap) builds —
//! fed a synthesized invocation envelope and backed by an internal
//! in-memory execution client instead of the Lambda checkpoint API.
//! Envelope parsing, bootstrap pagination, the suspension driver, wire
//! error mapping, and the response envelope are therefore the production
//! code paths, not a parallel reimplementation. When the handler
//! suspends, the runner advances the simulated backend (timers fire,
//! retry delays elapse, callbacks are delivered) and re-invokes, exactly
//! as the real service would, until the execution reaches a terminal
//! outcome.
//!
//! The runner also reproduces the **production task topology**: the
//! service future is awaited inline (under the caller's `block_on`, as
//! `lambda_runtime` does), never on a spawned task, so
//! `tokio::task::try_id()` is `None` at context-creation time and the
//! task-ownership guard behaves exactly as it does in deployment.
//!
//! By default the in-memory backend serves execution state in **multiple
//! pages** (2+ whenever the recorded history is large enough), matching
//! the paginating service; [`LocalRunner::single_page`] opts into the
//! single-page special case.
//!
//! The runner records every checkpointed operation so a test can assert on
//! the execution history via [`TestResult::operations`] and
//! [`TestOperation`].
//!
//! [`CloudRunner`] is the real-AWS counterpart: it invokes a deployed durable
//! function, polls the execution to a terminal state, and folds the recorded
//! service history into the SAME [`TestResult`] / [`TestOperation`] types, so a
//! test written against [`LocalRunner`] can be re-pointed at real AWS with only
//! the runner swapped.
//!
//! # Backend fidelity
//!
//! The in-memory client reproduces the behaviours the SDK depends on,
//! traced to this crate's own engine:
//!
//! - Checkpoint responses return whole-record operation updates that the
//!   SDK merges into its in-memory log (`client::merge_operations_into_log`
//!   → `CheckpointLog::insert` whole-record overwrite). A `Start` update
//!   carries no payload, so the merged record clears any carried result —
//!   the exact condition the `wait_for_condition` read-before-`Start` fix
//!   guards against.
//! - Timers (`wait`) and retry delays are modelled as state transitions
//!   between invocations rather than wall-clock sleeps, so there is no
//!   real time in the runner or its tests.
//!
//! # Runner divergences
//!
//! [`LocalRunner`] and [`CloudRunner`] return the same types, and a test can
//! be re-pointed between them by swapping the runner. The two agree on
//! operation type, status, result, error, [`TestOperation::attempt`] (both
//! report completed retries), and success/failure disposition. A few
//! behaviours still differ because one runner observes an in-memory model and
//! the other observes the recorded execution history; assert around these
//! when a test targets both runners:
//!
//! - **Open-callback status.** A created but not-yet-resolved callback reads
//!   `"Pending"` under [`LocalRunner`] and `"Started"` under [`CloudRunner`]
//!   (the recorded history has no distinct callback-pending status). Once the
//!   callback resolves, both report `"Succeeded"` / `"Failed"` / `"TimedOut"`.
//! - **Truncated results.** When a result exceeds the recorded-history inline
//!   size limit, [`CloudRunner`] reports [`TestOperation::result`] as `None`
//!   (the payload is withheld); [`LocalRunner`] always holds the full
//!   in-memory result.
//! - **Timed-out / stopped executions.** [`CloudRunner`] maps a terminal
//!   `TimedOut` or `Stopped` execution to [`TestResult::is_failure`] with the
//!   status name (or the recorded error) as the error type; [`LocalRunner`]
//!   has no execution-level timed-out or stopped disposition (it reports
//!   success, failure, or suspended).
//! - **Transient operation statuses.** [`LocalRunner`] may surface the
//!   intermediate `"Ready"` status it uses to model an elapsed retry delay;
//!   [`CloudRunner`]'s status vocabulary comes from the event stream and does
//!   not include `"Ready"`.
//!
//! # Examples
//!
//! ```
//! use aws_durable_execution_sdk_rust as durable;
//! use durable::test_util::LocalRunner;
//!
//! # #[tokio::main]
//! # async fn main() {
//! let result = LocalRunner::new()
//!     .run(
//!         |event: i32, ctx: durable::DurableContext| async move {
//!             let doubled = ctx.step(move |_| async move { Ok(event * 2) }).await?;
//!             Ok::<_, durable::BoxError>(doubled)
//!         },
//!         21_i32,
//!     )
//!     .await;
//!
//! assert!(result.is_success());
//! assert_eq!(result.output(), Some(&42));
//! # }
//! ```

use std::sync::{Arc, Mutex};

use serde::Serialize;
use serde::de::DeserializeOwned;

use aws_sdk_lambda::primitives::{Blob, DateTime};
use aws_sdk_lambda::types::{
    CallbackDetails, ChainedInvokeDetails, ContextDetails, ErrorObject, Event, EventType,
    ExecutionStatus, InvocationType, Operation, OperationAction, OperationStatus, OperationType,
    OperationUpdate, StepDetails,
};

use crate::BoxError;
use crate::client::{CheckpointOutput, ClientError, ExecutionClient, GetStateOutput};
use crate::context::DurableContext;

/// Default cap on the number of invocations the runner will drive before
/// declaring the execution stuck. Generous enough for deep timer/retry
/// chains, low enough to fail a non-terminating handler (e.g. a
/// `wait_for_condition` that never advances) instead of looping forever.
const DEFAULT_MAX_INVOCATIONS: usize = 100;

/// Default page size for simulated execution-state pagination. `1` maximizes
/// fidelity pressure: any history of two or more operations is served as 2+
/// pages (inline first page plus a `get_state` fetch), so every non-trivial
/// test exercises the bootstrap and checkpoint pagination paths by default —
/// the dimension where a missing-pagination defect was previously untestable
/// by construction.
const DEFAULT_STATE_PAGE_SIZE: usize = 1;

// ────────────────────────────────────────────────────────────────────────────
// TestOperation
// ────────────────────────────────────────────────────────────────────────────

/// A single checkpointed operation recorded during a [`LocalRunner`] run.
///
/// Returned in order by [`TestResult::operations`]. All accessors return
/// borrowed views; construct these only through the runner.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::LocalRunner;
///
/// # #[tokio::main]
/// # async fn main() {
/// let result = LocalRunner::new()
///     .run(
///         |_e: (), ctx: durable::DurableContext| async move {
///             ctx.step(|_| async { Ok(1_i32) }).name("only").await?;
///             Ok::<_, durable::BoxError>(())
///         },
///         (),
///     )
///     .await;
///
/// let ops = result.operations();
/// assert_eq!(ops.len(), 1);
/// assert_eq!(ops[0].op_type(), "Step");
/// assert!(ops[0].succeeded());
/// assert_eq!(ops[0].name(), Some("only"));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TestOperation {
    id: String,
    op_type: String,
    sub_type: Option<String>,
    status: String,
    result: Option<String>,
    error_type: Option<String>,
    error_message: Option<String>,
    attempt: u32,
    name: Option<String>,
}

impl TestOperation {
    /// The operation's wire ID (the SHA-256 hex of its positional path).
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The operation type, e.g. `"Step"`, `"Wait"`, `"Context"`,
    /// `"Callback"`.
    #[must_use]
    pub fn op_type(&self) -> &str {
        &self.op_type
    }

    /// The operation sub-type, e.g. `"Step"`, `"Wait"`, `"Map"`,
    /// `"WaitForCondition"`, or `None` if the operation carried none.
    #[must_use]
    pub fn sub_type(&self) -> Option<&str> {
        self.sub_type.as_deref()
    }

    /// The terminal-or-current status, e.g. `"Succeeded"`, `"Failed"`,
    /// `"Pending"`, `"TimedOut"`.
    #[must_use]
    pub fn status(&self) -> &str {
        &self.status
    }

    /// Returns `true` if the operation completed successfully.
    #[must_use]
    pub fn succeeded(&self) -> bool {
        self.status == "Succeeded"
    }

    /// Returns `true` if the operation ended in a failed status.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.status == "Failed"
    }

    /// The serialized result payload, if the operation stored one.
    #[must_use]
    pub fn result(&self) -> Option<&str> {
        self.result.as_deref()
    }

    /// The recorded error type, if the operation failed.
    #[must_use]
    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    /// The recorded error message, if the operation failed.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// The number of completed retries recorded for this operation
    /// (`0` when never retried). [`LocalRunner`] and [`CloudRunner`] report
    /// this value identically.
    #[must_use]
    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    /// The operation's caller-assigned name, if any.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// TestResult
// ────────────────────────────────────────────────────────────────────────────

/// Terminal disposition of a [`LocalRunner`] run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    Succeeded,
    Failed,
    Suspended,
}

/// The outcome of a [`LocalRunner::run`], including the deserialized handler
/// output (on success) and the recorded operation history.
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::LocalRunner;
///
/// # #[tokio::main]
/// # async fn main() {
/// let result = LocalRunner::new()
///     .run(
///         |_e: (), _ctx: durable::DurableContext| async move {
///             Ok::<_, durable::BoxError>("hi".to_owned())
///         },
///         (),
///     )
///     .await;
///
/// assert!(result.is_success());
/// assert_eq!(result.output().map(String::as_str), Some("hi"));
/// assert_eq!(result.invocation_count(), 1);
/// # }
/// ```
#[derive(Debug)]
pub struct TestResult<O> {
    disposition: Disposition,
    output: Option<O>,
    error_type: Option<String>,
    error_message: Option<String>,
    operations: Vec<TestOperation>,
    invocations: usize,
}

impl<O> TestResult<O> {
    /// Returns `true` if the handler completed successfully.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.disposition == Disposition::Succeeded
    }

    /// Returns `true` if the handler returned an error.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        self.disposition == Disposition::Failed
    }

    /// Returns `true` if the handler suspended and never reached a terminal
    /// outcome within the invocation budget (e.g. it awaits a callback the
    /// test never delivered, or it never terminates).
    #[must_use]
    pub fn is_suspended(&self) -> bool {
        self.disposition == Disposition::Suspended
    }

    /// The deserialized handler output on success, or `None` otherwise.
    #[must_use]
    pub fn output(&self) -> Option<&O> {
        self.output.as_ref()
    }

    /// Consumes the result and returns the owned handler output on success.
    #[must_use]
    pub fn into_output(self) -> Option<O> {
        self.output
    }

    /// The wire error type on failure (e.g. `"StepError"`), or `None`.
    #[must_use]
    pub fn error_type(&self) -> Option<&str> {
        self.error_type.as_deref()
    }

    /// The wire error message on failure, or `None`.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    /// The recorded operation history, in first-checkpoint order.
    #[must_use]
    pub fn operations(&self) -> &[TestOperation] {
        &self.operations
    }

    /// The number of invocations that drove this execution.
    ///
    /// For [`LocalRunner`] this is the count of simulated invocations; for
    /// [`CloudRunner`] it is the number of recorded `InvocationCompleted`
    /// events (falling back to the poll count when none are recorded).
    #[must_use]
    pub fn invocation_count(&self) -> usize {
        self.invocations
    }
}

// ────────────────────────────────────────────────────────────────────────────
// LocalRunner
// ────────────────────────────────────────────────────────────────────────────

/// A queued external outcome for a pending callback.
#[derive(Debug, Clone)]
enum CallbackOutcome {
    Success(String),
    Timeout,
}

/// Drives a durable handler to completion in memory, simulating the backend.
///
/// Construct with [`LocalRunner::new`], optionally queue callback outcomes
/// with [`callback_success`](Self::callback_success) /
/// [`callback_timeout`](Self::callback_timeout), then drive a handler with
/// [`run`](Self::run).
///
/// # Examples
///
/// ```
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::LocalRunner;
///
/// # #[tokio::main]
/// # async fn main() {
/// // Deliver a value to the first callback the handler awaits.
/// let runner = LocalRunner::new().callback_success(&true);
/// let result = runner
///     .run(
///         |_e: (), ctx: durable::DurableContext| async move {
///             let cb = ctx.create_callback::<bool>().await?;
///             let approved = cb.result().await?;
///             Ok::<_, durable::BoxError>(approved)
///         },
///         (),
///     )
///     .await;
///
/// assert_eq!(result.output(), Some(&true));
/// # }
/// ```
#[derive(Debug)]
pub struct LocalRunner {
    max_invocations: usize,
    callback_outcomes: Vec<CallbackOutcome>,
    /// Initial-state page size. When set, the synthesized invocation
    /// envelope carries at most this many history operations inline plus a
    /// pagination marker, so the context fetches the remainder via
    /// `get_state` — the paginating service is the DEFAULT
    /// ([`DEFAULT_STATE_PAGE_SIZE`]); [`LocalRunner::single_page`] opts
    /// into the single-page special case (`None`).
    initial_page_size: Option<usize>,
    /// Checkpoint-response page size. When set, the backend's checkpoint
    /// response includes `next_marker` once the total stored operations
    /// exceed this threshold, simulating a paginated checkpoint response
    /// (the default; see [`DEFAULT_STATE_PAGE_SIZE`]). This exercises the
    /// checkpoint pagination path in `DurableContext::checkpoint_updates`.
    checkpoint_page_size: Option<usize>,
    /// Checkpoint coalescing window, mirroring
    /// [`Options`](crate::Options)'s `checkpoint_delay`. `None` (the
    /// default) writes every checkpoint immediately.
    checkpoint_delay: Option<std::time::Duration>,
    /// Whether checkpoint batching is enabled, mirroring
    /// [`Options`](crate::Options)'s `checkpoint_batching`.
    checkpoint_batching: bool,
    /// Number of upcoming checkpoint calls the backend rejects with a
    /// non-retryable error (fault injection; see
    /// [`fail_next_checkpoints`](Self::fail_next_checkpoints)).
    checkpoint_failures: usize,
    /// Number of checkpoint calls to let through before the injected
    /// failures start (see
    /// [`fail_checkpoints_after`](Self::fail_checkpoints_after)).
    checkpoint_failure_skip: usize,
    /// Whether the injected checkpoint failures are retryable (invocation
    /// fault; the runner re-invokes) rather than non-retryable (terminal
    /// `FAIL` then execution failure).
    checkpoint_failures_retryable: bool,
}

impl Default for LocalRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalRunner {
    /// Creates a runner with default settings and no queued callback
    /// outcomes.
    ///
    /// By default the simulated backend **paginates execution state**
    /// (both the initial invocation envelope and checkpoint responses
    /// split into 2+ pages once history is large enough), matching the
    /// real service. Use [`single_page`](Self::single_page) for the
    /// explicit single-page special case.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_invocations: DEFAULT_MAX_INVOCATIONS,
            callback_outcomes: Vec::new(),
            initial_page_size: Some(DEFAULT_STATE_PAGE_SIZE),
            checkpoint_page_size: Some(DEFAULT_STATE_PAGE_SIZE),
            checkpoint_delay: None,
            checkpoint_batching: false,
            checkpoint_failures: 0,
            checkpoint_failure_skip: 0,
            checkpoint_failures_retryable: false,
        }
    }

    /// Disables state pagination: the invocation envelope carries the full
    /// history inline and checkpoint responses never set a pagination
    /// marker.
    ///
    /// This is the **special case** — the real service paginates, and the
    /// runner does too by default. Reach for this only when a test
    /// specifically targets single-page behavior.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().single_page();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn single_page(mut self) -> Self {
        self.initial_page_size = None;
        self.checkpoint_page_size = None;
        self
    }

    /// Sets the maximum number of invocations the runner will drive before
    /// returning a suspended result.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().max_invocations(10);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn max_invocations(mut self, max: usize) -> Self {
        self.max_invocations = max.max(1);
        self
    }

    /// Sets the initial-state page size for pagination testing.
    ///
    /// When set, the runner truncates the checkpoint log passed to each
    /// invocation to at most this many operations, simulating the service
    /// embedding only the first page in `InitialExecutionState`. The
    /// context then calls `get_state` to fetch the rest.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().initial_page_size(1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn initial_page_size(mut self, size: usize) -> Self {
        self.initial_page_size = Some(size.max(1));
        self
    }

    /// Sets the checkpoint-response page size for pagination testing.
    ///
    /// When set, the backend's checkpoint response includes a pagination
    /// marker once total stored operations exceed this threshold, forcing
    /// the context to paginate via `get_state`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().checkpoint_page_size(1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn checkpoint_page_size(mut self, size: usize) -> Self {
        self.checkpoint_page_size = Some(size.max(1));
        self
    }

    /// Enables checkpoint coalescing with the given delay window, exactly
    /// as [`Options`](crate::Options)'s
    /// [`checkpoint_delay`](crate::OptionsBuilder::checkpoint_delay) does
    /// in production: checkpoints from concurrently running operations
    /// coalesce into fewer writes, and the buffer flushes unconditionally
    /// at suspension, execution completion, and callback creation.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    /// use std::time::Duration;
    ///
    /// let runner = LocalRunner::new().checkpoint_delay(Duration::from_millis(20));
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn checkpoint_delay(mut self, delay: std::time::Duration) -> Self {
        self.checkpoint_delay = Some(delay);
        self
    }

    /// Makes the simulated backend reject the next `count` checkpoint
    /// calls with a non-retryable error, persisting nothing for them —
    /// exactly like a permanent service-side rejection.
    ///
    /// Under the #43 model a non-retryable checkpoint failure never
    /// reaches the handler: the SDK persists a small terminal `FAIL` for
    /// the operation (when user code already ran) and fails the
    /// execution. Use this to test that model, and to assert no
    /// record-transition lifecycle event is emitted for the rejected
    /// write (see [`crate::observability`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().fail_next_checkpoints(1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn fail_next_checkpoints(mut self, count: usize) -> Self {
        self.checkpoint_failures = count;
        self
    }

    /// Like [`fail_next_checkpoints`](Self::fail_next_checkpoints), but
    /// lets the first `skip` checkpoint calls through before rejecting
    /// the following `count` non-retryably.
    ///
    /// Use it to reject a specific write: a step's live path writes START
    /// first and its outcome second, so `fail_checkpoints_after(1, 1)`
    /// rejects exactly the outcome write while the START (and everything
    /// after the failure window, such as the SDK's terminal `FAIL`
    /// record) persists.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().fail_checkpoints_after(1, 1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn fail_checkpoints_after(mut self, skip: usize, count: usize) -> Self {
        self.checkpoint_failure_skip = skip;
        self.checkpoint_failures = count;
        self.checkpoint_failures_retryable = false;
        self
    }

    /// Like [`fail_checkpoints_after`](Self::fail_checkpoints_after), but
    /// the injected failures are RETRYABLE — simulating a transient
    /// failure that exhausted the transport's own retries.
    ///
    /// Under the #43 model a retryable checkpoint failure fails the
    /// invocation with no further writes; the runner then re-invokes,
    /// exactly as the durable service does, and replay resumes from the
    /// recorded state.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().fail_checkpoints_after_retryable(1, 1);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn fail_checkpoints_after_retryable(mut self, skip: usize, count: usize) -> Self {
        self.checkpoint_failure_skip = skip;
        self.checkpoint_failures = count;
        self.checkpoint_failures_retryable = true;
        self
    }

    /// Enables checkpoint batching, exactly as [`Options`](crate::Options)'s
    /// [`checkpoint_batching`](crate::OptionsBuilder::checkpoint_batching)
    /// does in production: checkpoint writes go through a single ordered
    /// writer, checkpoints arriving while a write is in flight are sent
    /// together in the next call (split to respect per-request size
    /// limits), and the buffer flushes unconditionally at suspension,
    /// execution completion, and callback creation.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().checkpoint_batching();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn checkpoint_batching(mut self) -> Self {
        self.checkpoint_batching = true;
        self
    }

    /// Queues a successful outcome for the next callback the handler awaits.
    ///
    /// The value is serialized as the callback payload and delivered when
    /// the callback becomes pending, in the order queued.
    ///
    /// # Panics
    ///
    /// Never panics: a value that fails to serialize is queued as an empty
    /// payload, which surfaces as a deserialization error to the handler
    /// rather than aborting the test.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().callback_success(&"approved");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn callback_success<T: Serialize>(mut self, value: &T) -> Self {
        let payload = serde_json::to_string(value).unwrap_or_default();
        self.callback_outcomes
            .push(CallbackOutcome::Success(payload));
        self
    }

    /// Queues a timeout outcome for the next callback the handler awaits.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::LocalRunner;
    ///
    /// let runner = LocalRunner::new().callback_timeout();
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn callback_timeout(mut self) -> Self {
        self.callback_outcomes.push(CallbackOutcome::Timeout);
        self
    }

    /// Drives `handler` to a terminal outcome, feeding it `event`.
    ///
    /// Each simulated invocation goes through the **production service
    /// function** ([`wrap`](crate::wrap)'s body) fed a synthesized
    /// invocation envelope, so envelope parsing, bootstrap pagination, the
    /// suspension driver, and wire error mapping are the exact code
    /// production runs. The service future is awaited inline — the same
    /// task topology `lambda_runtime` produces under `block_on` — so
    /// task-ownership behavior matches deployment.
    ///
    /// The event is serialized once and embedded in each invocation's
    /// envelope, mirroring the way the service re-delivers the input
    /// payload on every re-invocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust as durable;
    /// use durable::test_util::LocalRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let result = LocalRunner::new()
    ///     .run(
    ///         |n: u32, ctx: durable::DurableContext| async move {
    ///             let v = ctx.step(move |_| async move { Ok(n + 1) }).await?;
    ///             Ok::<_, durable::BoxError>(v)
    ///         },
    ///         41_u32,
    ///     )
    ///     .await;
    /// assert_eq!(result.output(), Some(&42));
    /// # }
    /// ```
    #[allow(clippy::too_many_lines)] // reason: the invoke → drive → advance loop reads better as one flow
    pub async fn run<E, O, F, Fut>(&self, handler: F, event: E) -> TestResult<O>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        let backend = Arc::new(Backend::new(
            self.callback_outcomes.clone(),
            self.checkpoint_page_size,
        ));
        backend.checkpoint_failures_remaining.store(
            self.checkpoint_failures,
            std::sync::atomic::Ordering::SeqCst,
        );
        backend.checkpoint_failures_skip.store(
            self.checkpoint_failure_skip,
            std::sync::atomic::Ordering::SeqCst,
        );
        backend.checkpoint_failures_retryable.store(
            self.checkpoint_failures_retryable,
            std::sync::atomic::Ordering::SeqCst,
        );
        self.run_on_backend(backend, handler, event).await
    }

    /// The invoke → parse → advance loop over an externally supplied
    /// backend. Internal seam: unit tests inject a backend they retain a
    /// handle to, so they can assert on transport-level facts (e.g. how
    /// many `get_state` fetches pagination forced).
    #[allow(clippy::too_many_lines)] // reason: the invoke → drive → advance loop reads better as one flow
    async fn run_on_backend<E, O, F, Fut>(
        &self,
        backend: Arc<Backend>,
        handler: F,
        event: E,
    ) -> TestResult<O>
    where
        E: Serialize + DeserializeOwned + Send + 'static,
        O: Serialize + DeserializeOwned + Send + 'static,
        F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O, BoxError>> + Send,
    {
        let client: Arc<dyn ExecutionClient> = Arc::clone(&backend) as Arc<dyn ExecutionClient>;

        // The handler runs behind the SAME service function production
        // registers with the Lambda runtime — only the transport (the
        // execution client) is faked. The buffer window is derived from the
        // two knobs exactly as `wrap` derives it from `Options`.
        let checkpoint_buffer_window = match (self.checkpoint_delay, self.checkpoint_batching) {
            (Some(delay), _) => Some(delay),
            (None, true) => Some(std::time::Duration::ZERO),
            (None, false) => None,
        };
        let service = crate::wrap_with_execution_client(handler, client, checkpoint_buffer_window);

        // Serialize the event once; the envelope re-delivers it per
        // invocation.
        let event_json = match serde_json::to_string(&event) {
            Ok(json) => json,
            Err(e) => {
                return TestResult {
                    disposition: Disposition::Failed,
                    output: None,
                    error_type: Some("SerializationFailed".to_owned()),
                    error_message: Some(format!("serialize event: {e}")),
                    operations: Vec::new(),
                    invocations: 0,
                };
            }
        };

        let mut invocations = 0_usize;
        // The most recent invocation fault (Lambda runtime error), kept so
        // an execution that never terminates because every invocation
        // faults reports the fault rather than a generic budget message.
        let mut last_invocation_fault: Option<String> = None;

        loop {
            invocations += 1;
            if invocations > self.max_invocations {
                return TestResult {
                    disposition: Disposition::Suspended,
                    output: None,
                    error_type: last_invocation_fault
                        .is_some()
                        .then(|| "RuntimeError".to_owned()),
                    error_message: Some(last_invocation_fault.map_or_else(
                        || {
                            format!(
                                "execution did not terminate within {} invocations",
                                self.max_invocations
                            )
                        },
                        |fault| {
                            format!(
                                "execution did not terminate within {} invocations; \
                                 last invocation fault: {fault}",
                                self.max_invocations
                            )
                        },
                    )),
                    operations: backend.snapshot_operations(),
                    invocations: invocations - 1,
                };
            }

            let payload = build_envelope(&backend, &event_json, self.initial_page_size);
            let lambda_event =
                lambda_runtime::LambdaEvent::new(payload, lambda_runtime::Context::default());

            // Await the service future INLINE — never on a spawned task.
            // This reproduces the production topology (`lambda_runtime`
            // awaits the handler under `block_on`), so
            // `tokio::task::try_id()` is `None` at context-creation time
            // and the task-ownership guard behaves as it does deployed.
            let response = match service(lambda_event).await {
                Ok(envelope) => envelope,
                Err(e) => {
                    // The invocation itself failed (a Lambda runtime
                    // error) — e.g. a retryable checkpoint failure that
                    // exhausted transport retries (issue #43), or a
                    // bootstrap failure. The durable service re-invokes
                    // on an invocation fault, so the runner does too;
                    // `max_invocations` bounds a persistent fault.
                    last_invocation_fault = Some(e.to_string());
                    continue;
                }
            };

            match response.get("Status").and_then(serde_json::Value::as_str) {
                Some("SUCCEEDED") => {
                    let serialized = response
                        .get("Result")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("null");
                    let output = serde_json::from_str::<O>(serialized).ok();
                    let (error_type, error_message, disposition) = if output.is_some() {
                        (None, None, Disposition::Succeeded)
                    } else {
                        (
                            Some("SerializationFailed".to_owned()),
                            Some("could not deserialize handler output".to_owned()),
                            Disposition::Failed,
                        )
                    };
                    return TestResult {
                        disposition,
                        output,
                        error_type,
                        error_message,
                        operations: backend.snapshot_operations(),
                        invocations,
                    };
                }
                Some("FAILED") => {
                    let error = response.get("Error");
                    let error_type = error
                        .and_then(|e| e.get("ErrorType"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("HandlerError")
                        .to_owned();
                    let error_message = error
                        .and_then(|e| e.get("ErrorMessage"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    return TestResult {
                        disposition: Disposition::Failed,
                        output: None,
                        error_type: Some(error_type),
                        error_message: Some(error_message),
                        operations: backend.snapshot_operations(),
                        invocations,
                    };
                }
                Some("PENDING") => {
                    // Advance the simulated backend (timers, retries,
                    // callbacks). If nothing can advance, the execution is
                    // genuinely stuck.
                    if !backend.advance() {
                        return TestResult {
                            disposition: Disposition::Suspended,
                            output: None,
                            error_type: None,
                            error_message: Some(
                                "execution suspended with no pending timer, retry, or callback \
                                 to advance (missing callback outcome?)"
                                    .to_owned(),
                            ),
                            operations: backend.snapshot_operations(),
                            invocations,
                        };
                    }
                }
                other => {
                    return TestResult {
                        disposition: Disposition::Failed,
                        output: None,
                        error_type: Some("RuntimeError".to_owned()),
                        error_message: Some(format!(
                            "unexpected response envelope status: {other:?}"
                        )),
                        operations: backend.snapshot_operations(),
                        invocations,
                    };
                }
            }
        }
    }
}

/// The execution ARN the local runner stamps on every synthesized envelope.
const LOCAL_EXECUTION_ARN: &str = "arn:aws:lambda:us-west-2:000000000000:function:local-test";

/// Synthesizes the durable invocation envelope for one simulated
/// invocation, in the exact wire shape the service delivers and
/// [`wrap`](crate::wrap) parses: `DurableExecutionArn`, `CheckpointToken`,
/// and `InitialExecutionState.Operations` with the customer input embedded
/// in the leading `Execution` operation's `ExecutionDetails.InputPayload`.
///
/// When `initial_page_size` is set and the recorded history exceeds it,
/// only the first page of history rides inline and
/// `InitialExecutionState.NextMarker` signals the truncation — the
/// production bootstrap path then fetches the remainder via `get_state`,
/// exactly as it does against the paginating service.
fn build_envelope(
    backend: &Backend,
    event_json: &str,
    initial_page_size: Option<usize>,
) -> serde_json::Value {
    let history = backend.envelope_history();
    let (page, next_marker) = match initial_page_size {
        Some(size) if history.len() > size => (
            history.get(..size).unwrap_or(&history).to_vec(),
            Some(format!("initial-marker-{}", history.len())),
        ),
        _ => (history, None),
    };

    let mut operations = Vec::with_capacity(page.len() + 1);
    operations.push(serde_json::json!({
        "Id": "execution",
        "Type": "Execution",
        "Status": "Started",
        "ExecutionDetails": { "InputPayload": event_json }
    }));
    operations.extend(page);

    let initial_state = match next_marker {
        Some(marker) => serde_json::json!({
            "Operations": operations,
            "NextMarker": marker,
        }),
        None => serde_json::json!({ "Operations": operations }),
    };

    serde_json::json!({
        "DurableExecutionArn": LOCAL_EXECUTION_ARN,
        "CheckpointToken": backend.current_token(),
        "InitialExecutionState": initial_state,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// CloudRunner (real-AWS runner returning the same TestResult/TestOperation)
// ────────────────────────────────────────────────────────────────────────────

/// Default seconds between execution-status polls.
const DEFAULT_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// Default cap on status polls before the runner reports the execution as
/// still running (suspended). 150 × 2 s ≈ 5 minutes.
const DEFAULT_MAX_POLL_ATTEMPTS: usize = 150;

/// `error_type` used on a [`TestResult`] when a `CloudRunner` infrastructure
/// step fails (invoke error, missing execution ARN, poll-budget exhaustion,
/// or history retrieval). Distinct from a durable execution `FAILED`, whose
/// `error_type` carries the service-recorded error.
const CLOUD_RUNNER_ERROR: &str = "CloudRunnerError";

/// Drives a durable function against **real AWS** and returns the same
/// [`TestResult`] / [`TestOperation`] types [`LocalRunner`] returns.
///
/// `CloudRunner` invokes a deployed durable function (`InvocationType=Event`),
/// polls [`GetDurableExecution`] until the execution reaches a terminal state,
/// then folds the recorded [`GetDurableExecutionHistory`] event stream into
/// per-operation [`TestOperation`]s. Because both runners return the same
/// types, a test can be re-pointed from in-memory to real AWS by swapping the
/// runner.
///
/// The AWS client is built internally from the ambient credentials and region
/// (the same way the production entry point builds it), so no AWS type appears
/// in any public signature. Configure the region explicitly with
/// [`region`](Self::region) when the environment does not supply one.
///
/// Infrastructure failures (invoke error, no execution ARN, poll-budget
/// exhaustion, deserialization) surface through the returned [`TestResult`]:
/// [`is_failure`](TestResult::is_failure) with `error_type() ==
/// "CloudRunnerError"`, or [`is_suspended`](TestResult::is_suspended) when the
/// execution is still running at the poll budget.
///
/// [`GetDurableExecution`]: https://docs.aws.amazon.com/lambda/latest/api/API_GetDurableExecution.html
/// [`GetDurableExecutionHistory`]: https://docs.aws.amazon.com/lambda/latest/api/API_GetDurableExecutionHistory.html
///
/// # Examples
///
/// ```no_run
/// use aws_durable_execution_sdk_rust as durable;
/// use durable::test_util::CloudRunner;
/// use std::time::Duration;
///
/// # #[tokio::main]
/// # async fn main() {
/// let result = CloudRunner::new("my-durable-function")
///     .region("us-west-2")
///     .poll_interval(Duration::from_secs(3))
///     .run::<_, i32>(21_i32)
///     .await;
///
/// if result.is_success() {
///     assert_eq!(result.output(), Some(&42));
///     for op in result.operations() {
///         println!("{} {} {}", op.id(), op.op_type(), op.status());
///     }
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct CloudRunner {
    function_name: String,
    region: Option<String>,
    qualifier: String,
    poll_interval: std::time::Duration,
    max_poll_attempts: usize,
}

impl CloudRunner {
    /// Creates a runner targeting the named deployed durable function.
    ///
    /// `function_name` may be a function name or a fully-qualified ARN. The
    /// qualifier defaults to `$LATEST` (durable functions require a qualified
    /// invocation); override it with [`qualifier`](Self::qualifier).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("order-processor");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn new(function_name: impl Into<String>) -> Self {
        Self {
            function_name: function_name.into(),
            region: None,
            qualifier: String::from("$LATEST"),
            poll_interval: DEFAULT_POLL_INTERVAL,
            max_poll_attempts: DEFAULT_MAX_POLL_ATTEMPTS,
        }
    }

    /// Sets the AWS region used to build the Lambda client.
    ///
    /// When unset, the region is resolved from the ambient environment
    /// (`AWS_REGION`, profile, or instance metadata).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("f").region("us-west-2");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Sets the invocation qualifier (function version or alias).
    ///
    /// Defaults to `$LATEST`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("f").qualifier("PROD");
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn qualifier(mut self, qualifier: impl Into<String>) -> Self {
        self.qualifier = qualifier.into();
        self
    }

    /// Sets the delay between execution-status polls.
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    /// use std::time::Duration;
    ///
    /// let runner = CloudRunner::new("f").poll_interval(Duration::from_secs(5));
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn poll_interval(mut self, interval: std::time::Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Sets the maximum number of status polls before the runner reports the
    /// execution as still running (a suspended [`TestResult`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// let runner = CloudRunner::new("f").max_poll_attempts(30);
    /// # drop(runner);
    /// ```
    #[must_use]
    pub fn max_poll_attempts(mut self, max: usize) -> Self {
        self.max_poll_attempts = max.max(1);
        self
    }

    /// Invokes the durable function with `event`, polls to a terminal state,
    /// and returns the folded [`TestResult`].
    ///
    /// The event is serialized to JSON and delivered as the invocation
    /// payload. The handler output (on success) is deserialized into `O`.
    ///
    /// This method never panics: every AWS or serialization failure is
    /// reported through the returned [`TestResult`] (see [`CloudRunner`]).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use aws_durable_execution_sdk_rust::test_util::CloudRunner;
    ///
    /// # #[tokio::main]
    /// # async fn main() {
    /// let result = CloudRunner::new("echo-fn")
    ///     .run::<_, String>("hello".to_owned())
    ///     .await;
    /// assert!(result.is_success() || result.is_failure() || result.is_suspended());
    /// # }
    /// ```
    #[allow(clippy::too_many_lines)] // reason: the invoke → poll → history → fold flow reads better as one sequence
    pub async fn run<E, O>(&self, event: E) -> TestResult<O>
    where
        E: Serialize,
        O: DeserializeOwned,
    {
        let payload = match serde_json::to_vec(&event) {
            Ok(bytes) => bytes,
            Err(e) => return cloud_failure(format!("serialize event: {e}")),
        };

        let client = self.build_client().await;

        // 1. Invoke (async) — start the durable execution and capture its ARN.
        let invoke_result = client
            .invoke()
            .function_name(&self.function_name)
            .invocation_type(InvocationType::Event)
            .qualifier(&self.qualifier)
            .payload(Blob::new(payload))
            .send()
            .await;
        let invoke_output = match invoke_result {
            Ok(out) => out,
            Err(e) => return cloud_failure(format!("invoke {}: {e}", self.function_name)),
        };
        let Some(execution_arn) = invoke_output.durable_execution_arn().map(ToOwned::to_owned)
        else {
            return cloud_failure(
                "invoke response carried no DurableExecutionArn (is the target a durable \
                 function invoked with a qualifier?)"
                    .to_owned(),
            );
        };

        // 2. Poll GetDurableExecution until terminal or the budget is spent.
        let mut polls = 0_usize;
        let terminal = loop {
            if polls >= self.max_poll_attempts {
                let operations = self.fetch_operations(&client, &execution_arn).await;
                return TestResult {
                    disposition: Disposition::Suspended,
                    output: None,
                    error_type: None,
                    error_message: Some(format!(
                        "execution {execution_arn} still running after {polls} polls"
                    )),
                    operations,
                    invocations: polls,
                };
            }
            polls += 1;
            tokio::time::sleep(self.poll_interval).await;
            match client
                .get_durable_execution()
                .durable_execution_arn(&execution_arn)
                .include_execution_data(true)
                .send()
                .await
            {
                Ok(out) if matches!(out.status(), ExecutionStatus::Running) => {}
                Ok(out) => break out,
                Err(e) => return cloud_failure(format!("get_durable_execution: {e}")),
            }
        };

        // 3. Retrieve the recorded history and fold it into operations.
        let (operations, invocation_events) =
            match self.fetch_history(&client, &execution_arn).await {
                Ok(events) => (fold_history(&events), invocations_from_history(&events)),
                Err(msg) => return cloud_failure(msg),
            };
        let invocations = if invocation_events == 0 {
            polls
        } else {
            invocation_events
        };

        // 4. Map the terminal status into a TestResult disposition.
        match terminal.status() {
            ExecutionStatus::Succeeded => match deserialize_output::<O>(terminal.result()) {
                Ok(output) => TestResult {
                    disposition: Disposition::Succeeded,
                    output: Some(output),
                    error_type: None,
                    error_message: None,
                    operations,
                    invocations,
                },
                Err(msg) => TestResult {
                    disposition: Disposition::Failed,
                    output: None,
                    error_type: Some("SerializationFailed".to_owned()),
                    error_message: Some(msg),
                    operations,
                    invocations,
                },
            },
            other => {
                let status_name = execution_status_str(other);
                let (error_type, error_message) = terminal.error().map_or_else(
                    || {
                        (
                            status_name.to_owned(),
                            format!("execution ended with status {status_name}"),
                        )
                    },
                    |err| {
                        (
                            err.error_type().unwrap_or(status_name).to_owned(),
                            err.error_message().unwrap_or_default().to_owned(),
                        )
                    },
                );
                TestResult {
                    disposition: Disposition::Failed,
                    output: None,
                    error_type: Some(error_type),
                    error_message: Some(error_message),
                    operations,
                    invocations,
                }
            }
        }
    }

    /// Builds the Lambda client from the ambient config, honoring an explicit
    /// region override.
    async fn build_client(&self) -> aws_sdk_lambda::Client {
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(region) = &self.region {
            loader = loader.region(aws_sdk_lambda::config::Region::new(region.clone()));
        }
        let config = loader.load().await;
        aws_sdk_lambda::Client::new(&config)
    }

    /// Best-effort history fetch used on the suspended (budget-exhausted) path.
    async fn fetch_operations(
        &self,
        client: &aws_sdk_lambda::Client,
        execution_arn: &str,
    ) -> Vec<TestOperation> {
        (self.fetch_history(client, execution_arn).await)
            .map(|events| fold_history(&events))
            .unwrap_or_default()
    }

    /// Retrieves the full paginated execution history.
    async fn fetch_history(
        &self,
        client: &aws_sdk_lambda::Client,
        execution_arn: &str,
    ) -> Result<Vec<Event>, String> {
        let mut events = Vec::new();
        let mut marker: Option<String> = None;
        loop {
            let mut builder = client
                .get_durable_execution_history()
                .durable_execution_arn(execution_arn)
                .include_execution_data(true);
            if let Some(m) = &marker {
                builder = builder.marker(m);
            }
            match builder.send().await {
                Ok(out) => {
                    events.extend(out.events().iter().cloned());
                    match out.next_marker() {
                        Some(m) if !m.is_empty() => marker = Some(m.to_owned()),
                        _ => break,
                    }
                }
                Err(e) => return Err(format!("get_durable_execution_history: {e}")),
            }
        }
        Ok(events)
    }
}

/// Builds a failed [`TestResult`] for a `CloudRunner` infrastructure error.
fn cloud_failure<O>(message: String) -> TestResult<O> {
    TestResult {
        disposition: Disposition::Failed,
        output: None,
        error_type: Some(CLOUD_RUNNER_ERROR.to_owned()),
        error_message: Some(message),
        operations: Vec::new(),
        invocations: 0,
    }
}

/// Deserializes the handler output from the `GetDurableExecution` result
/// string, tolerating a double-encoded (JSON-string-of-JSON) payload.
fn deserialize_output<O: DeserializeOwned>(result: Option<&str>) -> Result<O, String> {
    let raw = result.unwrap_or("null");
    match serde_json::from_str::<O>(raw) {
        Ok(value) => Ok(value),
        Err(first) => match serde_json::from_str::<String>(raw) {
            Ok(inner) => serde_json::from_str::<O>(&inner)
                .map_err(|e| format!("deserialize handler output: {e}")),
            Err(_) => Err(format!("deserialize handler output: {first}")),
        },
    }
}

/// Folds a chronological execution-event stream into per-operation
/// [`TestOperation`]s in first-appearance order.
///
/// Only operation-scoped events (`Step*`, `Wait*`, `Context*`, `Callback*`,
/// `ChainedInvoke*`) with an operation id are folded; execution-level and
/// `InvocationCompleted` events are ignored here (the execution outcome comes
/// from `GetDurableExecution`, the invocation count from
/// [`invocations_from_history`]). Later transitions overwrite earlier ones, so
/// the recorded `status` is the operation's terminal-or-latest state.
fn fold_history(events: &[Event]) -> Vec<TestOperation> {
    use std::collections::HashMap;

    let mut order: Vec<String> = Vec::new();
    let mut acc: HashMap<String, TestOperation> = HashMap::new();

    for event in events {
        let Some(op_type) = event_family_op_type(event.event_type()) else {
            continue;
        };
        let Some(id) = event.id() else { continue };

        let op = acc.entry(id.to_owned()).or_insert_with(|| {
            order.push(id.to_owned());
            TestOperation {
                id: id.to_owned(),
                op_type: op_type.to_owned(),
                sub_type: None,
                status: String::from("Started"),
                result: None,
                error_type: None,
                error_message: None,
                attempt: 0,
                name: None,
            }
        });

        if op.sub_type.is_none() {
            op.sub_type = event.sub_type().map(ToOwned::to_owned);
        }
        if op.name.is_none() {
            op.name = event.name().map(ToOwned::to_owned);
        }
        apply_event_transition(op, event);
    }

    order.iter().filter_map(|id| acc.get(id).cloned()).collect()
}

/// Applies a single event's transition (status, result, error, attempt) to the
/// accumulating operation record.
fn apply_event_transition(op: &mut TestOperation, event: &Event) {
    match event.event_type() {
        Some(
            EventType::StepStarted
            | EventType::WaitStarted
            | EventType::ContextStarted
            | EventType::CallbackStarted
            | EventType::ChainedInvokeStarted,
        ) => op.status = String::from("Started"),
        Some(EventType::StepSucceeded) => {
            op.status = String::from("Succeeded");
            if let Some(details) = event.step_succeeded_details() {
                op.result = event_result_payload(details.result());
                set_attempt(op, details.retry_details());
            }
        }
        Some(EventType::StepFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.step_failed_details() {
                set_error(op, details.error());
                set_attempt(op, details.retry_details());
            }
        }
        Some(EventType::WaitSucceeded) => op.status = String::from("Succeeded"),
        Some(EventType::WaitCancelled) => op.status = String::from("Cancelled"),
        Some(EventType::ContextSucceeded) => {
            op.status = String::from("Succeeded");
            op.result = event
                .context_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::ContextFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.context_failed_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::CallbackSucceeded) => {
            op.status = String::from("Succeeded");
            op.result = event
                .callback_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::CallbackFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.callback_failed_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::CallbackTimedOut) => {
            op.status = String::from("TimedOut");
            if let Some(details) = event.callback_timed_out_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::ChainedInvokeSucceeded) => {
            op.status = String::from("Succeeded");
            op.result = event
                .chained_invoke_succeeded_details()
                .and_then(|d| event_result_payload(d.result()));
        }
        Some(EventType::ChainedInvokeFailed) => {
            op.status = String::from("Failed");
            if let Some(details) = event.chained_invoke_failed_details() {
                set_error(op, details.error());
            }
        }
        Some(EventType::ChainedInvokeTimedOut) => op.status = String::from("TimedOut"),
        Some(EventType::ChainedInvokeStopped) => op.status = String::from("Stopped"),
        _ => {}
    }
}

/// Extracts the JSON payload string from an optional `EventResult`.
fn event_result_payload(result: Option<&aws_sdk_lambda::types::EventResult>) -> Option<String> {
    result
        .and_then(aws_sdk_lambda::types::EventResult::payload)
        .map(ToOwned::to_owned)
}

/// Records the error type/message from an optional `EventError` onto the op.
fn set_error(op: &mut TestOperation, error: Option<&aws_sdk_lambda::types::EventError>) {
    if let Some(obj) = error.and_then(aws_sdk_lambda::types::EventError::payload) {
        op.error_type = obj.error_type().map(ToOwned::to_owned);
        op.error_message = obj.error_message().map(ToOwned::to_owned);
    }
}

/// Records the number of completed retries from an optional `RetryDetails`
/// onto the op.
///
/// `RetryDetails.current_attempt` is 1-based (a first-try success reports `1`),
/// while [`TestOperation::attempt`] reports completed retries (a first-try
/// success reports `0`). The two are reconciled here so both runners agree.
/// `saturating_sub` guards the conversion: a non-positive `current_attempt`
/// maps to `0` without underflow.
fn set_attempt(op: &mut TestOperation, retry: Option<&aws_sdk_lambda::types::RetryDetails>) {
    if let Some(details) = retry {
        let one_based = u32::try_from(details.current_attempt().max(0)).unwrap_or(0);
        op.attempt = one_based.saturating_sub(1);
    }
}

/// Maps an event type to the operation-type string [`LocalRunner`] uses, or
/// `None` for execution-level / invocation-level events that are not folded
/// into operations.
fn event_family_op_type(event_type: Option<&EventType>) -> Option<&'static str> {
    match event_type? {
        EventType::StepStarted | EventType::StepSucceeded | EventType::StepFailed => Some("Step"),
        EventType::WaitStarted | EventType::WaitSucceeded | EventType::WaitCancelled => {
            Some("Wait")
        }
        EventType::ContextStarted | EventType::ContextSucceeded | EventType::ContextFailed => {
            Some("Context")
        }
        EventType::CallbackStarted
        | EventType::CallbackSucceeded
        | EventType::CallbackFailed
        | EventType::CallbackTimedOut => Some("Callback"),
        EventType::ChainedInvokeStarted
        | EventType::ChainedInvokeSucceeded
        | EventType::ChainedInvokeFailed
        | EventType::ChainedInvokeTimedOut
        | EventType::ChainedInvokeStopped => Some("ChainedInvoke"),
        _ => None,
    }
}

/// Counts `InvocationCompleted` events — the number of times the service
/// invoked the function to drive the execution.
fn invocations_from_history(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e.event_type(), Some(EventType::InvocationCompleted)))
        .count()
}

/// Human-readable execution status for a failed/terminal disposition.
fn execution_status_str(status: &ExecutionStatus) -> &'static str {
    match *status {
        ExecutionStatus::Succeeded => "Succeeded",
        ExecutionStatus::Failed => "Failed",
        ExecutionStatus::TimedOut => "TimedOut",
        ExecutionStatus::Stopped => "Stopped",
        ExecutionStatus::Running => "Running",
        _ => "Unknown",
    }
}

// ────────────────────────────────────────────────────────────────────────────
// In-memory backend (internal — implements the pub(crate) ExecutionClient)
// ────────────────────────────────────────────────────────────────────────────

/// A stored operation record inside the in-memory backend.
#[derive(Debug, Clone)]
struct StoredOp {
    id: String,
    op_type: OperationType,
    sub_type: Option<String>,
    status: OperationStatus,
    name: Option<String>,
    result: Option<String>,
    error_type: Option<String>,
    error_message: Option<String>,
    attempt: i32,
    callback_id: Option<String>,
    wait_seconds: Option<i32>,
    retry_delay: Option<i32>,
    callback_delivered: bool,
}

/// Mutable state of the in-memory backend, guarded by a single mutex.
#[derive(Debug)]
struct BackendState {
    /// Operations in first-checkpoint order.
    ops: Vec<StoredOp>,
    token_counter: u64,
    callback_counter: u64,
    /// Queued external callback outcomes, delivered FIFO.
    callback_outcomes: Vec<CallbackOutcome>,
}

/// The in-memory execution client. Shared (via `Arc`) between the runner and
/// the durable context. Debug-printable and `Send + Sync` as the
/// `ExecutionClient` trait requires.
#[derive(Debug)]
struct Backend {
    state: Mutex<BackendState>,
    /// When set, the checkpoint response will include `next_marker` when
    /// the number of updated operations exceeds this value, simulating a
    /// paginated checkpoint response.
    checkpoint_page_size: Option<usize>,
    /// Counts `get_state` calls — lets tests assert that pagination
    /// actually forced a state fetch (or, under `single_page`, that none
    /// occurred).
    get_state_calls: std::sync::atomic::AtomicUsize,
    /// Counts `checkpoint` calls — lets tests assert that checkpoint
    /// coalescing (`checkpoint_delay`) actually reduced the number of
    /// writes.
    checkpoint_calls: std::sync::atomic::AtomicUsize,
    /// Number of upcoming `checkpoint` calls to reject, before touching
    /// any state (fault injection; see
    /// [`LocalRunner::fail_next_checkpoints`] and
    /// [`LocalRunner::fail_checkpoints_after`]).
    checkpoint_failures_remaining: std::sync::atomic::AtomicUsize,
    /// Number of `checkpoint` calls to let through before the injected
    /// failures begin.
    checkpoint_failures_skip: std::sync::atomic::AtomicUsize,
    /// Whether injected checkpoint failures are retryable (see
    /// [`LocalRunner::fail_checkpoints_after_retryable`]).
    checkpoint_failures_retryable: std::sync::atomic::AtomicBool,
    /// Per-call checkpoint plan (in-crate test seam): while non-empty,
    /// each `checkpoint` call pops and obeys the front entry instead of
    /// the counter-based injection (see
    /// [`Backend::plan_checkpoint_calls`]).
    #[cfg(test)]
    checkpoint_plan: Mutex<std::collections::VecDeque<PlannedCheckpoint>>,
}

/// One planned behavior for an upcoming [`Backend`] `checkpoint` call
/// (in-crate test seam; see [`Backend::plan_checkpoint_calls`]).
#[cfg(test)]
#[derive(Debug)]
enum PlannedCheckpoint {
    /// The call proceeds normally against the simulated store.
    Pass,
    /// The call is rejected with a non-retryable error, persisting
    /// nothing. When `gate` is set, the rejection is held in flight until
    /// the test adds a permit to the semaphore, so the test can order
    /// events (e.g. dropping a contributor) against the in-flight write
    /// deterministically.
    FailNonRetryable {
        gate: Option<Arc<tokio::sync::Semaphore>>,
    },
}

impl Backend {
    fn new(callback_outcomes: Vec<CallbackOutcome>, checkpoint_page_size: Option<usize>) -> Self {
        Self {
            state: Mutex::new(BackendState {
                ops: Vec::new(),
                token_counter: 0,
                callback_counter: 0,
                callback_outcomes,
            }),
            checkpoint_page_size,
            get_state_calls: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_calls: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_remaining: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_skip: std::sync::atomic::AtomicUsize::new(0),
            checkpoint_failures_retryable: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            checkpoint_plan: Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// Queues a per-call plan for upcoming `checkpoint` calls (in-crate
    /// test seam). While entries remain, each call pops and obeys the
    /// front entry — pass, fail retryably, or fail non-retryably (with an
    /// optional in-flight gate) — instead of the counter-based injection.
    /// Once the plan is exhausted, calls fall back to the counters.
    #[cfg(test)]
    fn plan_checkpoint_calls(&self, plan: Vec<PlannedCheckpoint>) {
        self.checkpoint_plan
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend(plan);
    }

    /// Pops the next planned checkpoint behavior, if a plan is queued.
    #[cfg(test)]
    fn next_planned_checkpoint(&self) -> Option<PlannedCheckpoint> {
        self.checkpoint_plan
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
    }

    /// The number of `get_state` calls the SDK has made against this
    /// backend.
    #[cfg(test)]
    fn get_state_call_count(&self) -> usize {
        self.get_state_calls
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// The number of `checkpoint` calls the SDK has made against this
    /// backend.
    #[cfg(test)]
    fn checkpoint_call_count(&self) -> usize {
        self.checkpoint_calls
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BackendState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The current checkpoint token string.
    fn current_token(&self) -> String {
        let state = self.lock();
        format!("local-token-{}", state.token_counter)
    }

    /// Builds `Operation` records for the current state (used to seed the
    /// per-invocation checkpoint log).
    fn build_operations(&self) -> Vec<Operation> {
        let state = self.lock();
        state.ops.iter().filter_map(build_operation).collect()
    }

    /// Renders the recorded history as envelope-wire JSON operation
    /// objects — the exact shape `parse_inline_operations` reads from
    /// `InitialExecutionState.Operations`.
    fn envelope_history(&self) -> Vec<serde_json::Value> {
        let state = self.lock();
        state.ops.iter().map(stored_op_envelope_json).collect()
    }

    /// Snapshots the recorded operations as public [`TestOperation`]s.
    fn snapshot_operations(&self) -> Vec<TestOperation> {
        let state = self.lock();
        state
            .ops
            .iter()
            .map(|op| TestOperation {
                id: op.id.clone(),
                op_type: operation_type_str(&op.op_type).to_owned(),
                sub_type: op.sub_type.clone(),
                status: operation_status_str(&op.status).to_owned(),
                result: op.result.clone(),
                error_type: op.error_type.clone(),
                error_message: op.error_message.clone(),
                attempt: u32::try_from(op.attempt.max(0)).unwrap_or(0),
                name: op.name.clone(),
            })
            .collect()
    }

    /// Advances the simulated backend between invocations: fire elapsed
    /// timers, elapse retry delays, and deliver or time out pending
    /// callbacks. Returns `true` if any operation was advanced.
    ///
    /// Backend-behaviour anchors (this crate's own replay logic):
    /// - `wait`: a `Started` wait becomes `Succeeded` once its timer
    ///   elapses (`src/wait.rs` replay: `Succeeded` returns immediately,
    ///   `Started` suspends).
    /// - `step`/`wait_for_condition` retry: a `Pending` retry becomes
    ///   `Ready` once its delay elapses, so the next invocation re-executes
    ///   the body (`src/step.rs` and `src/wait_for_condition.rs` replay:
    ///   `Pending` suspends, `Ready` falls through to live execution).
    /// - callback: a `Pending` callback resolves to `Succeeded` (with a
    ///   payload) or `TimedOut` (`src/callback.rs` replay).
    fn advance(&self) -> bool {
        let mut state = self.lock();
        let mut progressed = false;

        // Timers and retry delays.
        for op in &mut state.ops {
            match op.op_type {
                OperationType::Wait if op.status == OperationStatus::Started => {
                    op.status = OperationStatus::Succeeded;
                    progressed = true;
                }
                OperationType::Step
                    if op.status == OperationStatus::Pending && op.retry_delay.is_some() =>
                {
                    // Retry timer elapsed: make the op re-executable.
                    op.status = OperationStatus::Ready;
                    op.retry_delay = None;
                    progressed = true;
                }
                _ => {}
            }
        }

        // Callback delivery (FIFO across queued outcomes).
        // Split the borrow so we can pop the outcome queue while iterating.
        let BackendState {
            ops,
            callback_outcomes,
            ..
        } = &mut *state;
        for op in ops.iter_mut() {
            let is_pending_callback = op.op_type == OperationType::Callback
                && !op.callback_delivered
                && matches!(
                    op.status,
                    OperationStatus::Pending | OperationStatus::Started
                );
            if !is_pending_callback {
                continue;
            }
            if callback_outcomes.is_empty() {
                continue;
            }
            let outcome = callback_outcomes.remove(0);
            match outcome {
                CallbackOutcome::Success(payload) => {
                    op.status = OperationStatus::Succeeded;
                    op.result = Some(payload);
                }
                CallbackOutcome::Timeout => {
                    op.status = OperationStatus::TimedOut;
                }
            }
            op.callback_delivered = true;
            progressed = true;
        }

        progressed
    }
}

impl BackendState {
    /// Finds a stored op by wire ID.
    fn find_mut(&mut self, id: &str) -> Option<&mut StoredOp> {
        self.ops.iter_mut().find(|op| op.id == id)
    }

    /// Applies a single `OperationUpdate` to the store, returning the
    /// resulting stored op (cloned) for the checkpoint response.
    fn apply_update(&mut self, update: &OperationUpdate) -> Option<StoredOp> {
        let id = update.id().to_owned();
        let op_type = update.r#type().clone();
        let sub_type = update.sub_type().map(ToOwned::to_owned);
        let action = update.action().clone();
        let name = update.name().map(ToOwned::to_owned);
        let payload = update.payload().map(ToOwned::to_owned);
        let (err_type, err_msg) = update.error().map_or((None, None), |e| {
            (
                e.error_type().map(ToOwned::to_owned),
                e.error_message().map(ToOwned::to_owned),
            )
        });
        let wait_seconds = update
            .wait_options()
            .and_then(aws_sdk_lambda::types::WaitOptions::wait_seconds);
        let retry_delay = update
            .step_options()
            .and_then(aws_sdk_lambda::types::StepOptions::next_attempt_delay_seconds);

        // Ensure the op exists.
        if self.find_mut(&id).is_none() {
            self.ops.push(StoredOp {
                id: id.clone(),
                op_type: op_type.clone(),
                sub_type: sub_type.clone(),
                status: OperationStatus::Started,
                name: name.clone(),
                result: None,
                error_type: None,
                error_message: None,
                attempt: 0,
                callback_id: None,
                wait_seconds: None,
                retry_delay: None,
                callback_delivered: false,
            });
        }

        // Decide callback-id assignment before taking the mutable op borrow
        // (assigning self.callback_counter later would conflict with it).
        let needs_callback_id = matches!(action, OperationAction::Start)
            && op_type == OperationType::Callback
            && self
                .ops
                .iter()
                .find(|o| o.id == id)
                .is_some_and(|o| o.callback_id.is_none());
        let new_callback_id = if needs_callback_id {
            self.callback_counter += 1;
            Some(format!("local-callback-{}", self.callback_counter))
        } else {
            None
        };

        let op = self.find_mut(&id)?;

        // Keep metadata current.
        if op.sub_type.is_none() {
            op.sub_type = sub_type;
        }
        if name.is_some() {
            op.name = name;
        }

        match action {
            OperationAction::Start => {
                // A Start carries no payload — clear any prior result so the
                // whole-record merge in the SDK clobbers the carried value
                // (the wait_for_condition read-before-Start invariant).
                op.result = None;
                op.error_type = None;
                op.error_message = None;
                if op.op_type == OperationType::Wait {
                    op.status = OperationStatus::Started;
                    op.wait_seconds = wait_seconds;
                } else if op.op_type == OperationType::Callback {
                    // Callbacks are pending until an external outcome; the
                    // backend assigns the callback id here.
                    op.status = OperationStatus::Pending;
                    if let Some(cb) = new_callback_id {
                        op.callback_id = Some(cb);
                    }
                } else {
                    op.status = OperationStatus::Started;
                }
            }
            OperationAction::Succeed => {
                op.status = OperationStatus::Succeeded;
                op.result = payload;
            }
            OperationAction::Fail => {
                op.status = OperationStatus::Failed;
                op.error_type = err_type;
                op.error_message = err_msg;
            }
            OperationAction::Retry => {
                op.status = OperationStatus::Pending;
                op.attempt += 1;
                op.retry_delay = retry_delay.or(Some(1));
                op.error_type = err_type;
                op.error_message = err_msg;
                // wait_for_condition carries per-attempt state in the payload.
                if payload.is_some() {
                    op.result = payload;
                }
            }
            _ => {}
        }

        Some(op.clone())
    }
}

impl ExecutionClient for Backend {
    fn checkpoint(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
        updates: Vec<OperationUpdate>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>>
    {
        self.checkpoint_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // In-crate test seam: a per-call plan (see
        // [`Backend::plan_checkpoint_calls`]) takes precedence over the
        // counter-based injection below while entries remain.
        #[cfg(test)]
        if let Some(planned) = self.next_planned_checkpoint() {
            match planned {
                PlannedCheckpoint::Pass => {}
                PlannedCheckpoint::FailNonRetryable { gate } => {
                    return Box::pin(async move {
                        if let Some(gate) = gate {
                            // Hold the failing write in flight until the
                            // test releases it — lets a test drop a
                            // contributor deterministically BEFORE the
                            // failure publishes.
                            let permit = gate.acquire().await;
                            drop(permit);
                        }
                        Err(ClientError::new_non_retryable(
                            "planned non-retryable checkpoint failure (test plan)",
                        ))
                    });
                }
            }
        }
        // Injected fault (`LocalRunner::fail_next_checkpoints` /
        // `fail_checkpoints_after[_retryable]`): after letting `skip`
        // calls through, reject BEFORE touching any state, so the
        // rejected write persists nothing — exactly like a service-side
        // rejection.
        let skipped = self
            .checkpoint_failures_skip
            .fetch_update(
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::SeqCst,
                |remaining| remaining.checked_sub(1),
            )
            .is_ok();
        if !skipped
            && self
                .checkpoint_failures_remaining
                .fetch_update(
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                    |remaining| remaining.checked_sub(1),
                )
                .is_ok()
        {
            let retryable = self
                .checkpoint_failures_retryable
                .load(std::sync::atomic::Ordering::SeqCst);
            return Box::pin(async move {
                if retryable {
                    Err(ClientError::from_retryable(
                        "injected retryable checkpoint failure \
                         (LocalRunner::fail_checkpoints_after_retryable)"
                            .to_owned(),
                    ))
                } else {
                    Err(ClientError::new_non_retryable(
                        "injected checkpoint failure (LocalRunner::fail_next_checkpoints)",
                    ))
                }
            });
        }
        let updated_ops: Vec<Operation> = {
            let mut state = self.lock();
            state.token_counter += 1;
            let mut out = Vec::with_capacity(updates.len());
            for update in &updates {
                if let Some(stored) = state.apply_update(update)
                    && let Some(op) = build_operation(&stored)
                {
                    out.push(op);
                }
            }
            out
        };
        let token = self.current_token();

        // If a checkpoint page size is configured and the total stored
        // operations exceed it, simulate a genuinely paginated response:
        // return only the FIRST PAGE of the full execution state (mirroring
        // the service's NewExecutionState) and set next_marker. Operations
        // beyond the page — including the ones this call just updated when
        // they fall outside it — are only observable through the follow-up
        // get_state fetch, so a caller that ignores the marker genuinely
        // misses state.
        let (response_ops, next_marker) = if let Some(page_size) = self.checkpoint_page_size {
            let all_ops = self.build_operations();
            if all_ops.len() > page_size {
                let first_page: Vec<Operation> =
                    all_ops.get(..page_size).unwrap_or(&all_ops).to_vec();
                let marker = format!("page-marker-{}", all_ops.len());
                (first_page, Some(marker))
            } else {
                (updated_ops, None)
            }
        } else {
            (updated_ops, None)
        };

        Box::pin(async move {
            Ok(CheckpointOutput {
                checkpoint_token: token,
                updated_operations: response_ops,
                next_marker,
            })
        })
    }

    fn get_state(
        &self,
        _execution_arn: &str,
        _checkpoint_token: &str,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>
    {
        self.get_state_calls
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let operations = self.build_operations();
        Box::pin(async move { Ok(GetStateOutput { operations }) })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// StoredOp → aws_sdk_lambda::types::Operation
// ────────────────────────────────────────────────────────────────────────────

/// Builds an SDK `Operation` from a stored op, placing result/error in the
/// details bucket the SDK reads for that operation type.
fn build_operation(stored: &StoredOp) -> Option<Operation> {
    let error_obj = if stored.error_type.is_some() || stored.error_message.is_some() {
        let mut eb = ErrorObject::builder();
        if let Some(t) = &stored.error_type {
            eb = eb.error_type(t);
        }
        if let Some(m) = &stored.error_message {
            eb = eb.error_message(m);
        }
        Some(eb.build())
    } else {
        None
    };

    let mut builder = Operation::builder()
        .id(&stored.id)
        .r#type(stored.op_type.clone())
        .status(stored.status.clone())
        .start_timestamp(DateTime::from_secs(0));

    if let Some(ref st) = stored.sub_type {
        builder = builder.sub_type(st);
    }
    if let Some(ref n) = stored.name {
        builder = builder.name(n);
    }

    match stored.op_type {
        OperationType::Step => {
            let mut sd = StepDetails::builder().attempt(stored.attempt);
            if let Some(r) = &stored.result {
                sd = sd.result(r);
            }
            if let Some(e) = error_obj.clone() {
                sd = sd.error(e);
            }
            builder = builder.step_details(sd.build());
        }
        OperationType::Context => {
            let mut cd = ContextDetails::builder();
            if let Some(r) = &stored.result {
                cd = cd.result(r);
            }
            if let Some(e) = error_obj.clone() {
                cd = cd.error(e);
            }
            builder = builder.context_details(cd.build());
        }
        OperationType::Callback => {
            let mut cb = CallbackDetails::builder();
            if let Some(id) = &stored.callback_id {
                cb = cb.callback_id(id);
            }
            if let Some(r) = &stored.result {
                cb = cb.result(r);
            }
            if let Some(e) = error_obj.clone() {
                cb = cb.error(e);
            }
            builder = builder.callback_details(cb.build());
        }
        OperationType::ChainedInvoke => {
            let mut ci = ChainedInvokeDetails::builder();
            if let Some(r) = &stored.result {
                ci = ci.result(r);
            }
            if let Some(e) = error_obj.clone() {
                ci = ci.error(e);
            }
            builder = builder.chained_invoke_details(ci.build());
        }
        _ => {}
    }

    builder.build().ok()
}

/// Renders a stored op as an envelope-wire JSON operation object, placing
/// result/error in the details bucket `parse_single_operation` reads for
/// that operation type. Mirrors [`build_operation`] field-for-field so the
/// inline envelope page and the `get_state` fetch describe identical
/// records.
fn stored_op_envelope_json(stored: &StoredOp) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("Id".to_owned(), serde_json::json!(stored.id));
    obj.insert(
        "Type".to_owned(),
        serde_json::json!(operation_type_str(&stored.op_type)),
    );
    obj.insert(
        "Status".to_owned(),
        serde_json::json!(operation_status_str(&stored.status)),
    );
    if let Some(st) = &stored.sub_type {
        obj.insert("SubType".to_owned(), serde_json::json!(st));
    }
    if let Some(n) = &stored.name {
        obj.insert("Name".to_owned(), serde_json::json!(n));
    }

    let error_obj = (stored.error_type.is_some() || stored.error_message.is_some()).then(|| {
        let mut err = serde_json::Map::new();
        if let Some(t) = &stored.error_type {
            err.insert("ErrorType".to_owned(), serde_json::json!(t));
        }
        if let Some(m) = &stored.error_message {
            err.insert("ErrorMessage".to_owned(), serde_json::json!(m));
        }
        serde_json::Value::Object(err)
    });

    let mut details = serde_json::Map::new();
    if let Some(r) = &stored.result {
        details.insert("Result".to_owned(), serde_json::json!(r));
    }
    if let Some(e) = error_obj {
        details.insert("Error".to_owned(), e);
    }

    match stored.op_type {
        OperationType::Step => {
            details.insert("Attempt".to_owned(), serde_json::json!(stored.attempt));
            obj.insert("StepDetails".to_owned(), serde_json::Value::Object(details));
        }
        OperationType::Context => {
            obj.insert(
                "ContextDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        OperationType::Callback => {
            if let Some(id) = &stored.callback_id {
                details.insert("CallbackId".to_owned(), serde_json::json!(id));
            }
            obj.insert(
                "CallbackDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        OperationType::ChainedInvoke => {
            obj.insert(
                "ChainedInvokeDetails".to_owned(),
                serde_json::Value::Object(details),
            );
        }
        _ => {}
    }

    serde_json::Value::Object(obj)
}

/// Human-readable operation type for [`TestOperation::op_type`].
fn operation_type_str(t: &OperationType) -> &'static str {
    match *t {
        OperationType::Step => "Step",
        OperationType::Wait => "Wait",
        OperationType::Context => "Context",
        OperationType::Callback => "Callback",
        OperationType::ChainedInvoke => "ChainedInvoke",
        OperationType::Execution => "Execution",
        _ => "Unknown",
    }
}

/// Human-readable operation status for [`TestOperation::status`].
fn operation_status_str(s: &OperationStatus) -> &'static str {
    match *s {
        OperationStatus::Started => "Started",
        OperationStatus::Pending => "Pending",
        OperationStatus::Ready => "Ready",
        OperationStatus::Succeeded => "Succeeded",
        OperationStatus::Failed => "Failed",
        OperationStatus::Cancelled => "Cancelled",
        OperationStatus::TimedOut => "TimedOut",
        OperationStatus::Stopped => "Stopped",
        _ => "Unknown",
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)] // reason: test assertions
#[allow(clippy::expect_used)] // reason: test assertions
#[allow(clippy::indexing_slicing)] // reason: test assertions index known-length op vectors
mod tests {
    use super::*;
    use crate::builders::map_parallel::CompletionConfig;
    use crate::{RetryDecision, StepError};
    use serde::{Deserialize, Serialize};
    use std::sync::atomic::{AtomicU32, Ordering};

    // 1. step success ────────────────────────────────────────────────────

    #[tokio::test]
    async fn step_success_returns_output_and_records_op() {
        let result = LocalRunner::new()
            .run(
                |n: i32, ctx: DurableContext| async move {
                    let v = ctx
                        .step(move |_| async move { Ok(n + 1) })
                        .name("inc")
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                41_i32,
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&42));
        assert_eq!(result.invocation_count(), 1);
        assert_eq!(result.operations().len(), 1);
        assert!(result.operations()[0].succeeded());
        assert_eq!(result.operations()[0].name(), Some("inc"));
    }

    // 1. step failure ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn step_failure_propagates_and_records_failed_op() {
        // A strategy that never retries so the failure is permanent.
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let strategy = |_e: &StepError, _a: u32| RetryDecision::Stop;
                    let v: i32 = ctx
                        .step(|_| async { Err("boom".into()) })
                        .retry_strategy(strategy)
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        // The execution record re-records the step failure's own recorded
        // identity: a plain boxed error carries no user type, so the step
        // record's generic "Error" passes through the boundary unchanged
        // (the "StepError" registry name is only the fallback for an
        // operation error with no attached wire identity).
        assert_eq!(result.error_type(), Some("Error"));
        assert!(result.operations().iter().any(TestOperation::failed));
    }

    // 1. step retry via RetryStrategyConfig ───────────────────────────────

    #[tokio::test]
    async fn step_retry_config_drives_retries_to_success() {
        // A config-based strategy (no hand-written closure): 3 total
        // attempts, 1s deterministic delays. The step fails on attempts 1
        // and 2 and succeeds on attempt 3, exercising the config through
        // real suspend/replay cycles.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let attempts_seen = Arc::clone(&attempts_seen);
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(3)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .max_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .build();
                        let v: i32 = ctx
                            .step(move |sc| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    if sc.attempt() < 3 {
                                        Err(format!("attempt {} failed", sc.attempt()).into())
                                    } else {
                                        Ok(7)
                                    }
                                }
                            })
                            .name("flaky-config")
                            .retry_strategy_config(config)
                            .await?;
                        Ok::<_, BoxError>(v)
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&7));
        // Body executed exactly 3 times (attempts 1, 2, 3).
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert!(result.invocation_count() >= 3);
    }

    #[tokio::test]
    async fn step_retry_config_max_attempts_exhausts_and_fails() {
        // max_attempts(2): the body runs exactly twice, then the error
        // propagates instead of retrying a third time.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let attempts_seen = Arc::clone(&attempts_seen);
                        let config = crate::builders::RetryStrategyConfig::builder()
                            .max_attempts(2)
                            .initial_delay(std::time::Duration::from_secs(1))
                            .jitter(crate::builders::JitterStrategy::None)
                            .build();
                        let v: i32 = ctx
                            .step(move |_| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    Err("always fails".into())
                                }
                            })
                            .name("always-fails")
                            .retry_strategy_config(config)
                            .await?;
                        Ok::<_, BoxError>(v)
                    }
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(result.operations().iter().any(TestOperation::failed));
    }

    // 1. step retry (retry strategy honored) ──────────────────────────────

    #[tokio::test]
    async fn step_retry_succeeds_on_third_attempt() {
        // Fails on attempts 1 and 2, succeeds on attempt 3. Each retry
        // suspends and the runner elapses the retry delay before re-invoking.
        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_seen = Arc::clone(&attempts);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let attempts_seen = Arc::clone(&attempts_seen);
                    async move {
                        let attempts_seen = Arc::clone(&attempts_seen);
                        let strategy = |_e: &StepError, attempt: u32| {
                            if attempt >= 3 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::Retry {
                                    delay: std::time::Duration::from_secs(1),
                                }
                            }
                        };
                        let v: i32 = ctx
                            .step(move |sc| {
                                let attempts_seen = Arc::clone(&attempts_seen);
                                async move {
                                    attempts_seen.fetch_add(1, Ordering::SeqCst);
                                    if sc.attempt() < 3 {
                                        Err(format!("attempt {} failed", sc.attempt()).into())
                                    } else {
                                        Ok(99)
                                    }
                                }
                            })
                            .name("flaky")
                            .retry_strategy(strategy)
                            .await?;
                        Ok::<_, BoxError>(v)
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&99));
        // Body executed exactly 3 times (attempts 1, 2, 3).
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        // The step took multiple invocations (one per retry suspension).
        assert!(result.invocation_count() >= 3);
        let step = result
            .operations()
            .iter()
            .find(|o| o.name() == Some("flaky"))
            .expect("flaky step recorded");
        assert!(step.succeeded());
        assert_eq!(step.attempt(), 2, "backend recorded 2 completed retries");
    }

    // 2. wait then resume across invocations ──────────────────────────────

    #[tokio::test]
    async fn wait_resumes_across_invocations() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let before = ctx
                        .step(|_| async { Ok("before".to_owned()) })
                        .name("before")
                        .await?;
                    ctx.wait(std::time::Duration::from_mins(5))
                        .name("nap")
                        .await?;
                    let after = ctx
                        .step(|_| async { Ok("after".to_owned()) })
                        .name("after")
                        .await?;
                    Ok::<_, BoxError>(format!("{before}-{after}"))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output().map(String::as_str), Some("before-after"));
        // The wait forced at least a second invocation.
        assert!(result.invocation_count() >= 2);
        let wait_op = result
            .operations()
            .iter()
            .find(|o| o.op_type() == "Wait")
            .expect("wait op recorded");
        assert!(wait_op.succeeded());
    }

    // 3. wait_for_condition state carry (exercises the merge-back clobber) ─

    #[derive(Clone, Serialize, Deserialize)]
    struct Counter {
        count: u32,
    }

    #[tokio::test]
    async fn wait_for_condition_carries_state_across_attempts() {
        // The check increments the carried count each attempt; the strategy
        // completes once count reaches 3. If the carried state were clobbered
        // to initial_state each attempt (the pre-554ac90 bug), count would
        // never advance and the runner would exhaust its invocation budget.
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let strategy = |state: Counter, _attempt: u32| {
                        if state.count >= 3 {
                            crate::builders::wait_for_condition::WaitDecision::complete()
                        } else {
                            crate::builders::wait_for_condition::WaitDecision::continue_with(
                                std::time::Duration::from_secs(1),
                            )
                        }
                    };
                    let final_state = ctx
                        .wait_for_condition(
                            |_sc, state: Counter| async move {
                                Ok(Counter {
                                    count: state.count + 1,
                                })
                            },
                            Counter { count: 0 },
                        )
                        .name("poll")
                        .wait_strategy_fn(strategy)
                        .await?;
                    Ok::<_, BoxError>(final_state.count)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&3),
            "carried state must advance across attempts (merge-back clobber must not reset it)"
        );
    }

    // 4. child contexts ───────────────────────────────────────────────────

    #[tokio::test]
    async fn child_context_returns_nested_result() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let v = ctx
                        .run_in_child_context(|child| async move {
                            let a = child.step(|_| async { Ok(10) }).await?;
                            let b = child.step(|_| async { Ok(20) }).await?;
                            Ok(a + b)
                        })
                        .name("branch")
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&30));
        // Parent context op + two child steps recorded.
        assert!(result.operations().iter().any(|o| o.op_type() == "Context"));
        let step_count = result
            .operations()
            .iter()
            .filter(|o| o.op_type() == "Step")
            .count();
        assert_eq!(step_count, 2);
    }

    // 5. parallel and map basics ──────────────────────────────────────────

    #[tokio::test]
    async fn parallel_runs_branches_and_collects_results() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let branches = vec![
                        crate::Branch::new("a", |c: DurableContext| async move {
                            let v = c.step(|_| async { Ok(1) }).await?;
                            Ok(v)
                        }),
                        crate::Branch::new("b", |c: DurableContext| async move {
                            let v = c.step(|_| async { Ok(2) }).await?;
                            Ok(v)
                        }),
                    ];
                    let results: Vec<i32> = ctx.parallel(branches).name("fan").await?;
                    Ok::<_, BoxError>(results.into_iter().sum::<i32>())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&3));
    }

    #[tokio::test]
    async fn map_applies_over_items() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let items = vec![1_i32, 2, 3];
                    let out: Vec<i32> = ctx
                        .map(items, |child, item, _idx| async move {
                            let scaled = child.step(move |_| async move { Ok(item * 10) }).await?;
                            Ok(scaled)
                        })
                        .name("scale")
                        .max_concurrency(2)
                        .completion(CompletionConfig::default())
                        .await?;
                    Ok::<_, BoxError>(out.into_iter().sum::<i32>())
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&60));
    }

    /// Replay of a predicate-completed batch is stable: the execution
    /// suspends mid-batch (a timer inside an item), resumes on a later
    /// invocation, replays the already-settled item from its checkpoint
    /// record without re-running its body, and the custom predicate then
    /// ends the batch early with the `PREDICATE_MATCHED` reason.
    #[tokio::test]
    async fn map_completion_predicate_survives_replay() {
        let item0_runs = Arc::new(AtomicU32::new(0));
        let item0_runs_handle = Arc::clone(&item0_runs);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let item0_runs = Arc::clone(&item0_runs_handle);
                    async move {
                        let batch = ctx
                            .map(vec![10_u32, 20, 30], move |child, item, idx| {
                                let item0_runs = Arc::clone(&item0_runs);
                                async move {
                                    if idx == 1 {
                                        // Suspends the batch mid-run, forcing a
                                        // second invocation that replays item 0.
                                        child
                                            .wait(std::time::Duration::from_secs(1))
                                            .name("stall")
                                            .await?;
                                    }
                                    let value = child
                                        .step(move |_| async move {
                                            if idx == 0 {
                                                item0_runs.fetch_add(1, Ordering::SeqCst);
                                            }
                                            Ok(item)
                                        })
                                        .name("work")
                                        .await?;
                                    Ok(value)
                                }
                            })
                            .name("predicated")
                            .max_concurrency(1)
                            .completion(
                                CompletionConfig::builder()
                                    .completion_predicate(|stats| stats.settled() >= 2)
                                    .build()?,
                            )
                            .await_batch()
                            .await?;
                        Ok::<_, BoxError>(format!(
                            "{}:{}",
                            batch.reason.as_str(),
                            batch.items.len()
                        ))
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // The predicate fired after two settles; item 2 never started.
        assert_eq!(result.output(), Some(&"PREDICATE_MATCHED:2".to_owned()));
        // The timer forced at least one suspension, so the batch spanned
        // several invocations and item 0 was replayed at least once.
        assert!(
            result.invocation_count() >= 2,
            "expected a replay, got {} invocation(s)",
            result.invocation_count()
        );
        // Replay returned item 0's recorded result instead of re-running
        // its step body.
        assert_eq!(
            item0_runs.load(Ordering::SeqCst),
            1,
            "item 0's step body must run exactly once across replays"
        );
    }

    /// Completion-trigger evaluation is deterministic under replay for a
    /// CONCURRENT batch: recorded-terminal children feed the statistics
    /// inline, in input order, before any resumed (live) branch joins.
    ///
    /// Three items at `max_concurrency(2)`: items 0 and 2 succeed on the
    /// first invocation, item 1 parks on a timer and fails after resume.
    /// The order-sensitive predicate `failed() > succeeded()` must never
    /// fire: on the resumed invocation the two recorded successes are
    /// applied (in input order) before item 1's live failure joins, so the
    /// predicate always sees 2 succeeded before it sees 1 failed —
    /// regardless of how the scheduler orders the join events. Without
    /// canonical ordering, resumed item 1's failure could join before
    /// item 0's recorded success replays, the predicate would see
    /// 1 failed / 0 succeeded and stop the batch, and item 2 — whose
    /// success is already in the checkpoint log — would be dropped from
    /// the result: a different operation history from identical recorded
    /// state.
    #[tokio::test]
    async fn map_completion_predicate_replay_order_is_canonical() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let batch = ctx
                        .map(vec![10_u32, 20, 30], |child, item, idx| async move {
                            if idx == 1 {
                                // Parks the branch, forcing a second
                                // invocation where items 0 and 2 are
                                // recorded-terminal while item 1 resumes
                                // live — and then fails.
                                child
                                    .wait(std::time::Duration::from_secs(1))
                                    .name("stall")
                                    .await?;
                                return Err("item 1 fails after resume".into());
                            }
                            Ok(item)
                        })
                        .name("order-sensitive")
                        .max_concurrency(2)
                        .completion(
                            CompletionConfig::builder()
                                // Keep the fixed failure trigger out of the
                                // way; only the predicate could stop early.
                                .tolerated_failure_count(10)
                                .completion_predicate(|stats| stats.failed() > stats.succeeded())
                                .build()?,
                        )
                        .await_batch()
                        .await?;
                    Ok::<_, BoxError>(format!(
                        "{}:{}:{}:{}",
                        batch.reason.as_str(),
                        batch.items.len(),
                        batch.success_count(),
                        batch.failure_count()
                    ))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // All three items are in the result, the lone failure never
        // outnumbered the recorded successes, and the batch ran to
        // completion.
        assert_eq!(result.output(), Some(&"ALL_COMPLETED:3:2:1".to_owned()));
        // The timer forced a suspension, so the second invocation really did
        // mix recorded-terminal children with a resumed live child.
        assert!(
            result.invocation_count() >= 2,
            "expected a replay, got {} invocation(s)",
            result.invocation_count()
        );
    }

    /// Regression for reversed LIVE settlement order (finding: replay
    /// ordering nondeterminism). Three items at `max_concurrency(3)`:
    /// item 1 succeeds BEFORE the deliberately delayed item 0 fails, and
    /// item 2 parks on a timer. The order-sensitive predicate
    /// `failed() > succeeded()` must make the same decision however the
    /// scheduler interleaves the live settlements: outcomes commit to the
    /// statistics strictly in input order, so the predicate's first
    /// evaluation is on the prefix `[item 0: failed]` — one failure, zero
    /// successes — and it fires, on the FIRST invocation, before any
    /// suspension. Under settlement-order evaluation the fresh run would
    /// instead see `[1 succeeded]` then `[1 succeeded, 1 failed]`, never
    /// fire, suspend on item 2 — and then the resumed invocation, replaying
    /// recorded outcomes in input order, would fire where the original run
    /// did not: two runs of the same execution disagreeing (the exact
    /// reviewed defect).
    #[tokio::test]
    async fn map_predicate_deterministic_under_reversed_live_settlement() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate_handle = Arc::clone(&gate);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let gate = Arc::clone(&gate_handle);
                    async move {
                        let batch = ctx
                            .map(vec![10_u32, 20, 30], move |child, item, idx| {
                                let gate = Arc::clone(&gate);
                                async move {
                                    match idx {
                                        0 => {
                                            // Settle strictly after item 1:
                                            // wait for its signal, then let
                                            // its join drain first.
                                            gate.notified().await;
                                            for _ in 0..16 {
                                                tokio::task::yield_now().await;
                                            }
                                            Err("item 0 fails last".into())
                                        }
                                        1 => {
                                            gate.notify_one();
                                            Ok(item)
                                        }
                                        _ => {
                                            // Parks; excluded once the
                                            // trigger fires.
                                            child
                                                .wait(std::time::Duration::from_secs(1))
                                                .name("stall")
                                                .await?;
                                            Ok(item)
                                        }
                                    }
                                }
                            })
                            .name("reversed-order")
                            .max_concurrency(3)
                            .completion(
                                CompletionConfig::builder()
                                    // Keep the fixed failure trigger out of
                                    // the way; only the predicate can stop
                                    // the batch.
                                    .tolerated_failure_count(10)
                                    .completion_predicate(|stats| {
                                        stats.failed() > stats.succeeded()
                                    })
                                    .build()?,
                            )
                            .await_batch()
                            .await?;
                        Ok::<_, BoxError>(format!(
                            "{}:{}:{}:{}",
                            batch.reason.as_str(),
                            batch.items.len(),
                            batch.success_count(),
                            batch.failure_count()
                        ))
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // The predicate fired at item 0's committed failure; items 0 and 1
        // both settled and are in the result, parked item 2 is excluded.
        assert_eq!(result.output(), Some(&"PREDICATE_MATCHED:2:1:1".to_owned()));
        // The trigger fired on the first invocation — the batch never
        // suspended, so the fresh run and any replay cannot disagree.
        assert_eq!(
            result.invocation_count(),
            1,
            "the predicate must fire before the batch suspends"
        );
    }

    /// Reversed live settlement order PLUS a genuine suspension: item 1
    /// succeeds before the delayed item 0 fails (both on the first
    /// invocation), item 2 parks and only succeeds after resume. The
    /// order-sensitive predicate stays false on every committed prefix —
    /// `[0: failed]`, `[0: failed, 1: succeeded]`, then after resume
    /// `[.., 2: succeeded]` — and those are the SAME prefixes the resumed
    /// invocation derives from the recorded outcomes, so the batch runs to
    /// completion with every item's outcome in the result, on both sides
    /// of the suspension boundary.
    #[tokio::test]
    async fn map_predicate_reversed_settlement_with_suspension_replays_stably() {
        let gate = Arc::new(tokio::sync::Notify::new());
        let gate_handle = Arc::clone(&gate);

        let result = LocalRunner::new()
            .run(
                move |(), ctx: DurableContext| {
                    let gate = Arc::clone(&gate_handle);
                    async move {
                        let batch = ctx
                            .map(vec![10_u32, 20, 30], move |child, item, idx| {
                                let gate = Arc::clone(&gate);
                                async move {
                                    match idx {
                                        0 => {
                                            gate.notified().await;
                                            for _ in 0..16 {
                                                tokio::task::yield_now().await;
                                            }
                                            Err("item 0 fails last".into())
                                        }
                                        1 => {
                                            gate.notify_one();
                                            Ok(item)
                                        }
                                        _ => {
                                            // Parks the batch mid-run,
                                            // forcing a resumed invocation
                                            // that replays items 0 and 1
                                            // from their records.
                                            child
                                                .wait(std::time::Duration::from_secs(1))
                                                .name("stall")
                                                .await?;
                                            Ok(item)
                                        }
                                    }
                                }
                            })
                            .name("reversed-then-suspend")
                            .max_concurrency(3)
                            .completion(
                                CompletionConfig::builder()
                                    .tolerated_failure_count(10)
                                    // Order-sensitive but never true here:
                                    // one failure can never outnumber the
                                    // successes by two.
                                    .completion_predicate(|stats| {
                                        stats.failed() > stats.succeeded() + 1
                                    })
                                    .build()?,
                            )
                            .await_batch()
                            .await?;
                        Ok::<_, BoxError>(format!(
                            "{}:{}:{}:{}",
                            batch.reason.as_str(),
                            batch.items.len(),
                            batch.success_count(),
                            batch.failure_count()
                        ))
                    }
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        // No trigger fired on either side of the suspension: all three
        // outcomes — including recorded items replayed after the resume —
        // are in the result.
        assert_eq!(result.output(), Some(&"ALL_COMPLETED:3:2:1".to_owned()));
        // The timer forced a suspension, so the batch really did span a
        // replay of the reversed-order recorded outcomes.
        assert!(
            result.invocation_count() >= 2,
            "expected a replay, got {} invocation(s)",
            result.invocation_count()
        );
    }

    // An arbitrary user error returned as BoxError from a child closure
    // surfaces with its message intact through the operation's error type,
    // with no error-conversion ceremony at the boundary.
    #[tokio::test]
    async fn child_closure_boxerror_message_propagates() {
        #[derive(Debug)]
        struct CustomBoundaryError;
        impl std::fmt::Display for CustomBoundaryError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "custom-boundary-failure")
            }
        }
        impl std::error::Error for CustomBoundaryError {}

        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let v: i32 = ctx
                        .run_in_child_context(|_child| async move {
                            // Plain `?`-free early return of an arbitrary error;
                            // no map_err, no ChildFnError.
                            Err(CustomBoundaryError)?;
                            Ok(0)
                        })
                        .name("boundary")
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(
            !result.is_success(),
            "child failure must surface as an error"
        );
        let msg = result.error_message().unwrap_or_default();
        assert!(
            msg.contains("custom-boundary-failure"),
            "user error message must survive the boundary: {msg}"
        );
    }

    // 6. callback success and callback timeout ────────────────────────────

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Approval {
        approved: bool,
    }

    #[tokio::test]
    async fn callback_success_delivers_payload() {
        let result = LocalRunner::new()
            .callback_success(&Approval { approved: true })
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<Approval>().name("approval").await?;
                    let approval = cb.result().await?;
                    Ok::<_, BoxError>(approval.approved)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&true));
        assert!(result.invocation_count() >= 2);
    }

    #[tokio::test]
    async fn callback_timeout_surfaces_error() {
        let result = LocalRunner::new()
            .callback_timeout()
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<Approval>().name("approval").await?;
                    let approval = cb.result().await?;
                    Ok::<_, BoxError>(approval.approved)
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("CallbackError"));
        let cb_op = result
            .operations()
            .iter()
            .find(|o| o.op_type() == "Callback")
            .expect("callback op recorded");
        assert_eq!(cb_op.status(), "TimedOut");
    }

    /// A callback result created via the public `ctx.create_callback` API
    /// composes with a real step future inside `try_join_all`. This exercises
    /// the full builder→callback→combinator path through replay, ensuring
    /// regressions in the public surface are caught.
    #[tokio::test]
    async fn callback_result_composes_with_step_in_combinator() {
        let result = LocalRunner::new()
            .callback_success(&"from-callback".to_owned())
            .run(
                |(), ctx: DurableContext| async move {
                    let cb = ctx.create_callback::<String>().name("approval").await?;

                    let step_future = ctx
                        .step(|_| async { Ok("from-step".to_owned()) })
                        .name("work")
                        .spawn();

                    let results = ctx.try_join_all([cb.result(), step_future]).await?;

                    Ok::<_, BoxError>(results)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler should succeed: {:?}",
            result.error_message()
        );
        let output = result.output().expect("should have output");
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], "from-callback");
        assert_eq!(output[1], "from-step");
        // At least 2 invocations: first suspends on the callback, second
        // replays with the settled outcome and completes.
        assert!(result.invocation_count() >= 2);
    }

    // 7. replay determinism ───────────────────────────────────────────────

    #[tokio::test]
    async fn replay_is_deterministic() {
        // The same handler + event yields identical operation IDs and
        // results across two independent runs.
        let make_run = || async {
            LocalRunner::new()
                .run(
                    |(), ctx: DurableContext| async move {
                        let a = ctx.step(|_| async { Ok(1) }).name("a").await?;
                        ctx.wait(std::time::Duration::from_mins(1))
                            .name("w")
                            .await?;
                        let b = ctx.step(|_| async { Ok(2) }).name("b").await?;
                        Ok::<_, BoxError>(a + b)
                    },
                    (),
                )
                .await
        };

        let first = make_run().await;
        let second = make_run().await;

        assert!(first.is_success());
        assert!(second.is_success());
        assert_eq!(first.output(), second.output());

        let ids_first: Vec<&str> = first.operations().iter().map(TestOperation::id).collect();
        let ids_second: Vec<&str> = second.operations().iter().map(TestOperation::id).collect();
        assert_eq!(ids_first, ids_second, "operation IDs must be deterministic");

        let results_first: Vec<Option<&str>> = first
            .operations()
            .iter()
            .map(TestOperation::result)
            .collect();
        let results_second: Vec<Option<&str>> = second
            .operations()
            .iter()
            .map(TestOperation::result)
            .collect();
        assert_eq!(results_first, results_second);
    }

    // ── CloudRunner pure-mapping tests (no AWS) ──────────────────────────

    use aws_sdk_lambda::types::{
        CallbackTimedOutDetails, EventError, EventResult, RetryDetails, StepFailedDetails,
        StepSucceededDetails,
    };

    #[test]
    fn fold_history_folds_step_lifecycle() {
        let events = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("op-1")
                .name("charge")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("op-1")
                .event_id(2)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("\"ok\"").build())
                        .retry_details(RetryDetails::builder().current_attempt(0).build())
                        .build(),
                )
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type(), "Step");
        assert_eq!(ops[0].status(), "Succeeded");
        assert_eq!(ops[0].result(), Some("\"ok\""));
        assert_eq!(ops[0].name(), Some("charge"));
        assert_eq!(ops[0].id(), "op-1");
        assert!(ops[0].succeeded());
    }

    #[test]
    fn fold_history_records_retry_attempt() {
        let events = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepFailed)
                .id("s")
                .event_id(2)
                .step_failed_details(
                    StepFailedDetails::builder()
                        .error(
                            EventError::builder()
                                .payload(
                                    ErrorObject::builder()
                                        .error_type("StepError")
                                        .error_message("boom")
                                        .build(),
                                )
                                .build(),
                        )
                        .retry_details(RetryDetails::builder().current_attempt(1).build())
                        .build(),
                )
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(3)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("99").build())
                        .retry_details(RetryDetails::builder().current_attempt(2).build())
                        .build(),
                )
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].status(), "Succeeded");
        assert_eq!(
            ops[0].attempt(),
            1,
            "current_attempt 2 normalizes to 1 completed retry"
        );
        assert_eq!(ops[0].result(), Some("99"));
    }

    #[test]
    fn fold_history_normalizes_attempt_to_completed_retries() {
        // A first-try success reports current_attempt=1, which must normalize
        // to 0 completed retries so LocalRunner and CloudRunner agree.
        let first_try = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(2)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("\"ok\"").build())
                        .retry_details(RetryDetails::builder().current_attempt(1).build())
                        .build(),
                )
                .build(),
        ];
        assert_eq!(
            fold_history(&first_try)[0].attempt(),
            0,
            "first-try success (current_attempt=1) is 0 completed retries"
        );

        // An unexpected current_attempt=0 must not underflow or panic.
        let zero = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("s")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("s")
                .event_id(2)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("\"ok\"").build())
                        .retry_details(RetryDetails::builder().current_attempt(0).build())
                        .build(),
                )
                .build(),
        ];
        assert_eq!(
            fold_history(&zero)[0].attempt(),
            0,
            "non-positive current_attempt saturates to 0"
        );
    }

    #[test]
    fn fold_history_maps_callback_timeout() {
        let events = vec![
            Event::builder()
                .event_type(EventType::CallbackStarted)
                .id("cb")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::CallbackTimedOut)
                .id("cb")
                .event_id(2)
                .callback_timed_out_details(
                    CallbackTimedOutDetails::builder()
                        .error(
                            EventError::builder()
                                .payload(
                                    ErrorObject::builder()
                                        .error_type("CallbackError")
                                        .error_message("timed out")
                                        .build(),
                                )
                                .build(),
                        )
                        .build(),
                )
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type(), "Callback");
        assert_eq!(ops[0].status(), "TimedOut");
        assert_eq!(ops[0].error_type(), Some("CallbackError"));
    }

    #[test]
    fn fold_history_skips_execution_and_invocation_events() {
        let events = vec![
            Event::builder()
                .event_type(EventType::ExecutionStarted)
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::InvocationCompleted)
                .event_id(2)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("op")
                .event_id(3)
                .step_succeeded_details(
                    StepSucceededDetails::builder()
                        .result(EventResult::builder().payload("1").build())
                        .build(),
                )
                .build(),
            Event::builder()
                .event_type(EventType::InvocationCompleted)
                .event_id(4)
                .build(),
            Event::builder()
                .event_type(EventType::ExecutionSucceeded)
                .event_id(5)
                .build(),
        ];
        let ops = fold_history(&events);
        assert_eq!(ops.len(), 1, "only the operation-scoped event is folded");
        assert_eq!(ops[0].op_type(), "Step");
        assert_eq!(invocations_from_history(&events), 2);
    }

    #[test]
    fn fold_history_preserves_first_appearance_order() {
        let events = vec![
            Event::builder()
                .event_type(EventType::StepStarted)
                .id("first")
                .event_id(1)
                .build(),
            Event::builder()
                .event_type(EventType::WaitStarted)
                .id("second")
                .event_id(2)
                .build(),
            Event::builder()
                .event_type(EventType::WaitSucceeded)
                .id("second")
                .event_id(3)
                .build(),
            Event::builder()
                .event_type(EventType::StepSucceeded)
                .id("first")
                .event_id(4)
                .step_succeeded_details(StepSucceededDetails::builder().build())
                .build(),
        ];
        let ops = fold_history(&events);
        let ids: Vec<&str> = ops.iter().map(TestOperation::id).collect();
        assert_eq!(ids, vec!["first", "second"]);
        assert_eq!(ops[1].op_type(), "Wait");
        assert_eq!(ops[1].status(), "Succeeded");
    }

    #[test]
    fn deserialize_output_handles_plain_missing_and_double_encoded() {
        assert_eq!(deserialize_output::<i32>(Some("42")).unwrap(), 42);
        assert_eq!(deserialize_output::<()>(None).unwrap(), ());
        // Double-encoded: a JSON string whose contents are the JSON value.
        assert_eq!(deserialize_output::<i32>(Some("\"7\"")).unwrap(), 7);
        assert!(deserialize_output::<i32>(Some("not json")).is_err());
    }

    #[test]
    fn execution_status_and_family_maps() {
        assert_eq!(execution_status_str(&ExecutionStatus::TimedOut), "TimedOut");
        assert_eq!(execution_status_str(&ExecutionStatus::Stopped), "Stopped");
        assert_eq!(
            event_family_op_type(Some(&EventType::ChainedInvokeSucceeded)),
            Some("ChainedInvoke")
        );
        assert_eq!(
            event_family_op_type(Some(&EventType::ExecutionStarted)),
            None
        );
        assert_eq!(event_family_op_type(None), None);
    }

    #[test]
    fn cloud_failure_is_reported_as_failure() {
        let result: TestResult<i32> = cloud_failure("boom".to_owned());
        assert!(result.is_failure());
        assert_eq!(result.error_type(), Some("CloudRunnerError"));
        assert_eq!(result.output(), None);
    }

    // ── Pagination tests ────────────────────────────────────────────────

    /// A two-page initial state replays all operations: the runner calls
    /// `get_state` when the history exceeds `initial_page_size`, and the
    /// handler sees results from prior invocations without re-executing.
    #[tokio::test]
    async fn two_page_initial_state_replays_all_operations() {
        // Use a static counter to detect re-execution of the first step.
        static FIRST_STEP_EXECUTIONS: AtomicU32 = AtomicU32::new(0);
        FIRST_STEP_EXECUTIONS.store(0, Ordering::SeqCst);

        let result = LocalRunner::new()
            .initial_page_size(1) // Only 1 op per "page" in initial state
            .run(
                |_event: (), ctx: DurableContext| async move {
                    // First step — should only execute on the first invocation.
                    let a = ctx
                        .step(|_| async {
                            FIRST_STEP_EXECUTIONS.fetch_add(1, Ordering::SeqCst);
                            Ok(10_i32)
                        })
                        .name("first")
                        .await?;

                    // Wait triggers a re-invocation boundary.
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("pause")
                        .await?;

                    // Second step on re-invocation — initial state now has
                    // 2+ ops, exceeding page_size=1. The runner must fetch
                    // the full state via get_state so `a` replays correctly.
                    let b = ctx
                        .step(move |_| async move { Ok(a + 32) })
                        .name("second")
                        .await?;

                    Ok::<_, BoxError>(b)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&42));
        // The first step must execute exactly once (not re-executed on replay).
        assert_eq!(
            FIRST_STEP_EXECUTIONS.load(Ordering::SeqCst),
            1,
            "first step must execute exactly once; pagination must not cause re-execution"
        );
        // Should take 2 invocations: first runs both steps+wait, second replays
        // the wait (now elapsed) and runs the second step.
        assert!(
            result.invocation_count() >= 2,
            "expected at least 2 invocations"
        );
    }

    /// A two-page checkpoint response is fully consumed: when the backend
    /// returns `next_marker` in a checkpoint response, `checkpoint_updates`
    /// calls `get_state` to merge remaining operations, ensuring subsequent
    /// reads see all state.
    #[tokio::test]
    async fn two_page_checkpoint_response_is_fully_consumed() {
        // When checkpoint_page_size is 1, the second checkpoint response
        // (which sees 2+ stored ops) includes a marker, triggering
        // pagination in checkpoint_updates.
        let result = LocalRunner::new()
            .checkpoint_page_size(1) // Trigger pagination after 1 stored op
            .run(
                |_event: (), ctx: DurableContext| async move {
                    // First step: creates 1 stored op. Checkpoint response
                    // has no marker (1 op <= page_size=1 threshold).
                    let a = ctx.step(|_| async { Ok(10_i32) }).name("step-a").await?;

                    // Second step: creates 2nd stored op. The checkpoint
                    // response now returns next_marker (2 ops > 1), so
                    // checkpoint_updates must call get_state to merge all
                    // operations.
                    let b = ctx
                        .step(move |_| async move { Ok(a + 5) })
                        .name("step-b")
                        .await?;

                    // Third step proves subsequent operations see the full
                    // state, including operations merged via get_state.
                    let c = ctx
                        .step(move |_| async move { Ok(b + 27) })
                        .name("step-c")
                        .await?;

                    Ok::<_, BoxError>(c)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&42));
        assert_eq!(result.operations().len(), 3);
    }

    /// A truncated checkpoint page genuinely requires the second page: the
    /// backend returns only the FIRST stored operation in the callback
    /// Start checkpoint response (plus a marker), so the backend-assigned
    /// `callback_id` is observable only through the follow-up `get_state`
    /// merge in `checkpoint_updates`. If the marker were ignored, the
    /// handler would see an empty callback id and fail.
    #[tokio::test]
    async fn truncated_checkpoint_page_requires_second_page() {
        let result = LocalRunner::new()
            .checkpoint_page_size(1)
            .callback_success(&"approved".to_owned())
            .run(
                |_event: (), ctx: DurableContext| async move {
                    // First stored operation: fills page 1.
                    let base = ctx.step(|_| async { Ok(1_i32) }).name("filler").await?;

                    // Second stored operation: the callback. Its Start
                    // checkpoint response is truncated to page 1 (the
                    // filler step), so the callback_id below arrives only
                    // via the paginated get_state merge.
                    let cb = ctx.create_callback::<String>().name("approval").await?;
                    if cb.id().is_empty() {
                        return Err::<String, BoxError>(
                            "callback id missing: second checkpoint page was not consumed".into(),
                        );
                    }

                    let approval = cb.result().await?;
                    Ok(format!("{base}:{approval}"))
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&"1:approved".to_owned()));
    }

    /// Combining both pagination modes: a handler with paginated initial
    /// state AND paginated checkpoint responses still runs correctly.
    #[tokio::test]
    async fn combined_initial_and_checkpoint_pagination() {
        let result = LocalRunner::new()
            .initial_page_size(1)
            .checkpoint_page_size(2)
            .run(
                |_event: (), ctx: DurableContext| async move {
                    let a = ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("timer")
                        .await?;
                    let b = ctx
                        .step(move |_| async move { Ok(a + 1) })
                        .name("b")
                        .await?;
                    let c = ctx
                        .step(move |_| async move { Ok(b + 1) })
                        .name("c")
                        .await?;
                    Ok::<_, BoxError>(c)
                },
                (),
            )
            .await;

        assert!(
            result.is_success(),
            "handler failed: {:?}",
            result.error_message()
        );
        assert_eq!(result.output(), Some(&3));
    }

    /// Tests that the Backend with `checkpoint_page_size` returns a
    /// `next_marker` when operations exceed the page size, simulating
    /// a two-page checkpoint response. The caller (`DurableContext`)
    /// should then paginate via `get_state`.
    #[tokio::test]
    async fn backend_paginated_checkpoint_returns_marker() {
        let backend = Arc::new(Backend::new(Vec::new(), Some(1)));

        // Checkpoint two operations so we exceed page_size=1.
        let updates = vec![
            OperationUpdate::builder()
                .id("op-1")
                .r#type(OperationType::Step)
                .action(OperationAction::Start)
                .build()
                .unwrap(),
            OperationUpdate::builder()
                .id("op-2")
                .r#type(OperationType::Step)
                .action(OperationAction::Start)
                .build()
                .unwrap(),
        ];

        let result = backend.checkpoint("arn:test", "token-0", updates).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(
            output.next_marker.is_some(),
            "expected a next_marker when operations exceed page_size"
        );

        // Verify that get_state returns all operations (used for pagination).
        let full_state = backend.get_state("arn:test", "token-0").await;
        assert!(full_state.is_ok());
        let full_state = full_state.unwrap();
        assert_eq!(
            full_state.operations.len(),
            2,
            "get_state must return all operations for pagination"
        );
    }

    /// Tests that the shared production bootstrap helper
    /// (`resolve_bootstrap_log`) follows the pagination marker: given a
    /// truncated first page plus a marker, it fetches the complete state
    /// via `get_state`; without a marker it uses the first page as-is and
    /// never calls `get_state`.
    #[tokio::test]
    async fn two_page_bootstrap_replays_all_operations() {
        use crate::client::{
            InMemoryExecutionClient, operations_to_checkpoint_log, resolve_bootstrap_log,
        };

        // Full backend state: two steps. Page 1 delivers only step-1.
        let all_ops = vec![
            Operation::builder()
                .id("step-1")
                .r#type(OperationType::Step)
                .status(OperationStatus::Succeeded)
                .start_timestamp(DateTime::from_secs(0))
                .step_details(StepDetails::builder().result("\"page1-result\"").build())
                .build()
                .unwrap(),
            Operation::builder()
                .id("step-2")
                .r#type(OperationType::Step)
                .status(OperationStatus::Succeeded)
                .start_timestamp(DateTime::from_secs(0))
                .step_details(StepDetails::builder().result("\"page2-result\"").build())
                .build()
                .unwrap(),
        ];

        let first_page: Vec<Operation> = all_ops.get(..1).unwrap_or(&all_ops).to_vec();

        let client = Arc::new(InMemoryExecutionClient::new(all_ops));

        // Marker present: the helper must fetch the complete state.
        let log = resolve_bootstrap_log(
            client.as_ref(),
            "arn:test",
            "token",
            operations_to_checkpoint_log(&first_page),
            Some("marker-1"),
        )
        .await
        .unwrap();

        assert!(
            log.get("step-1").is_some(),
            "step-1 from page 1 must be in the log"
        );
        assert!(
            log.get("step-2").is_some(),
            "step-2 from page 2 must be in the log"
        );

        let r1 = log.get("step-1").unwrap();
        assert_eq!(r1.result.as_deref(), Some("\"page1-result\""));

        let r2 = log.get("step-2").unwrap();
        assert_eq!(r2.result.as_deref(), Some("\"page2-result\""));

        let calls_after_marker = *client
            .get_state_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            calls_after_marker, 1,
            "marker must trigger exactly one get_state fetch"
        );

        // No marker: the first page is complete — no get_state call, and
        // the log holds only page-1 operations.
        let log = resolve_bootstrap_log(
            client.as_ref(),
            "arn:test",
            "token",
            operations_to_checkpoint_log(&first_page),
            None,
        )
        .await
        .unwrap();

        assert!(log.get("step-1").is_some());
        assert!(
            log.get("step-2").is_none(),
            "without a marker the helper must not fetch further pages"
        );
        let calls_after_no_marker = *client
            .get_state_call_count
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            calls_after_no_marker, 1,
            "no-marker path must not call get_state again"
        );
    }

    // ── Harness fidelity: production task topology ──────────────────────

    /// The runner awaits the handler INLINE under the caller's `block_on`
    /// — the production `lambda_runtime` topology — so the context is
    /// created where `tokio::task::try_id()` is `None`, not on a spawned
    /// task. This is the dimension that previously hid the ownership
    /// guard's production inertness.
    #[tokio::test]
    async fn runner_topology_matches_production_inline_block_on() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    // Same task visibility production gives the handler:
                    // awaited inline under block_on → no task ID.
                    let inline = tokio::task::try_id().is_none();
                    let v = ctx.step(|_| async { Ok(1_i32) }).await?;
                    Ok::<_, BoxError>((inline, v))
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&(true, 1)),
            "handler must run inline (try_id() == None), matching the deployed topology"
        );
    }

    /// Under the production topology, a durable operation invoked from a
    /// bare user `tokio::spawn` (unblessed) is rejected by the ownership
    /// guard — the runner must reproduce that, not mask it.
    #[tokio::test]
    async fn runner_rejects_durable_ops_from_unblessed_spawned_task() {
        let result = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let foreign = ctx.clone();
                    let joined: Result<i32, String> = tokio::spawn(async move {
                        foreign
                            .step(|_| async { Ok(7_i32) })
                            .await
                            // `{:#}` flattens the frame and its chain.
                            .map_err(|e| format!("{e:#}"))
                    })
                    .await
                    .map_err(|e| e.to_string())?;
                    let v = joined.map_err(BoxError::from)?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(
            result.is_failure(),
            "unblessed tokio::spawn must fail the ownership check as it does in production"
        );
        let msg = result.error_message().unwrap_or_default();
        assert!(
            msg.contains("Use .spawn()"),
            "ownership rejection should carry the production guidance: {msg}"
        );
    }

    // ── Harness fidelity: state pagination is the DEFAULT ───────────────

    /// By default the backend serves execution state in 2+ pages: a
    /// re-invocation with two or more recorded operations gets a truncated
    /// inline envelope page plus `NextMarker`, and checkpoint responses
    /// paginate — both force real `get_state` fetches through the
    /// production pagination paths.
    #[tokio::test]
    async fn default_state_pagination_forces_get_state_fetches() {
        let runner = LocalRunner::new();
        let backend = Arc::new(Backend::new(Vec::new(), runner.checkpoint_page_size));

        let result = runner
            .run_on_backend(
                Arc::clone(&backend),
                |(), ctx: DurableContext| async move {
                    let a = ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
                    // Suspend so the next invocation bootstraps from a
                    // multi-operation (hence multi-page) history.
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("cooldown")
                        .await?;
                    let b = ctx
                        .step(move |_| async move { Ok(a + 1) })
                        .name("b")
                        .await?;
                    Ok::<_, BoxError>(b)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&2));
        assert!(
            result.invocation_count() >= 2,
            "wait must force a re-invocation"
        );
        assert!(
            backend.get_state_call_count() >= 2,
            "multi-page default must force get_state fetches for both the bootstrap \
             (envelope NextMarker) and checkpoint pagination paths; saw {}",
            backend.get_state_call_count()
        );
    }

    /// `single_page()` is the explicit special case: the full history rides
    /// inline in every envelope and no checkpoint response paginates, so
    /// the SDK never needs a `get_state` fetch.
    #[tokio::test]
    async fn single_page_never_fetches_state() {
        let runner = LocalRunner::new().single_page();
        let backend = Arc::new(Backend::new(Vec::new(), runner.checkpoint_page_size));

        let result = runner
            .run_on_backend(
                Arc::clone(&backend),
                |(), ctx: DurableContext| async move {
                    let a = ctx.step(|_| async { Ok(1_i32) }).name("a").await?;
                    ctx.wait(std::time::Duration::from_secs(1))
                        .name("cooldown")
                        .await?;
                    let b = ctx
                        .step(move |_| async move { Ok(a + 1) })
                        .name("b")
                        .await?;
                    Ok::<_, BoxError>(b)
                },
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            backend.get_state_call_count(),
            0,
            "single-page mode must never require a get_state fetch"
        );
    }

    // ── Checkpoint coalescing (`checkpoint_delay`) ──────────────────────

    /// A handler with two concurrently-spawned steps, a suspension, and a
    /// post-resume step. `executions` counts step-body runs so replay
    /// fidelity is observable from outside.
    fn coalescing_probe_handler(
        executions: Arc<std::sync::atomic::AtomicUsize>,
    ) -> impl Fn(
        (),
        DurableContext,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<i32, BoxError>> + Send>>
    + Send
    + Sync
    + 'static {
        move |(), ctx: DurableContext| {
            let executions = Arc::clone(&executions);
            Box::pin(async move {
                let e1 = Arc::clone(&executions);
                let e2 = Arc::clone(&executions);
                let e3 = Arc::clone(&executions);

                // Two concurrent steps: their START checkpoints (and their
                // SUCCEED checkpoints) land within one coalescing window.
                let a = ctx
                    .step(move |_| {
                        let e = e1;
                        async move {
                            e.fetch_add(1, Ordering::SeqCst);
                            Ok(1_i32)
                        }
                    })
                    .name("a")
                    .spawn();
                let b = ctx
                    .step(move |_| {
                        let e = e2;
                        async move {
                            e.fetch_add(1, Ordering::SeqCst);
                            Ok(2_i32)
                        }
                    })
                    .name("b")
                    .spawn();
                let (ra, rb) = tokio::join!(a, b);
                let sum = ra? + rb?;

                // Suspension boundary: the coalesced SUCCEED checkpoints
                // above must have landed by now, or replay after resume
                // would re-execute the bodies.
                ctx.wait(std::time::Duration::from_secs(1))
                    .name("pause")
                    .await?;

                let c = ctx
                    .step(move |_| {
                        let e = e3;
                        async move {
                            e.fetch_add(1, Ordering::SeqCst);
                            Ok(sum + 39)
                        }
                    })
                    .name("c")
                    .await?;
                Ok::<_, BoxError>(c)
            })
        }
    }

    /// `checkpoint_delay` coalesces concurrent checkpoints into fewer API
    /// calls, and everything that must land before the suspension still
    /// does: the spawned step bodies run exactly once across the
    /// suspend/resume boundary, proving their coalesced checkpoints
    /// flushed before the invocation reported PENDING.
    #[tokio::test(start_paused = true)]
    async fn checkpoint_delay_coalesces_and_flushes_before_suspension() {
        // Baseline: identical handler, no coalescing.
        let baseline_execs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let baseline_runner = LocalRunner::new().single_page();
        let baseline_backend = Arc::new(Backend::new(Vec::new(), None));
        let baseline = baseline_runner
            .run_on_backend(
                Arc::clone(&baseline_backend),
                coalescing_probe_handler(Arc::clone(&baseline_execs)),
                (),
            )
            .await;
        assert!(baseline.is_success(), "{:?}", baseline.error_message());
        assert_eq!(baseline.output(), Some(&42));

        // Coalesced: same handler with a checkpoint delay window.
        let coalesced_execs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let coalesced_runner = LocalRunner::new()
            .single_page()
            .checkpoint_delay(std::time::Duration::from_millis(50));
        let coalesced_backend = Arc::new(Backend::new(Vec::new(), None));
        let coalesced = coalesced_runner
            .run_on_backend(
                Arc::clone(&coalesced_backend),
                coalescing_probe_handler(Arc::clone(&coalesced_execs)),
                (),
            )
            .await;
        assert!(coalesced.is_success(), "{:?}", coalesced.error_message());
        assert_eq!(
            coalesced.output(),
            Some(&42),
            "coalescing must not change the handler's result"
        );

        // Coalescing must have merged the concurrent steps' checkpoints
        // into fewer API calls than the immediate-write baseline.
        let baseline_calls = baseline_backend.checkpoint_call_count();
        let coalesced_calls = coalesced_backend.checkpoint_call_count();
        assert!(
            coalesced_calls < baseline_calls,
            "expected fewer checkpoint calls with coalescing: \
             coalesced={coalesced_calls}, baseline={baseline_calls}"
        );

        // Flush-before-suspension: each step body ran exactly once. A
        // checkpoint held past the PENDING boundary would force the
        // resumed invocation to re-execute the body.
        assert_eq!(
            coalesced_execs.load(Ordering::SeqCst),
            3,
            "each step body must run exactly once across suspend/resume"
        );
        assert_eq!(
            baseline_execs.load(Ordering::SeqCst),
            3,
            "baseline sanity: each step body runs exactly once"
        );
    }

    /// `checkpoint_delay` with a whole-handler completion: the terminal
    /// SUCCEEDED envelope is only reported after the end-of-invocation
    /// flush drains the buffer, so a sequential handler under a large
    /// delay window still completes with all operations recorded.
    #[tokio::test(start_paused = true)]
    async fn checkpoint_delay_completes_sequential_handlers() {
        let result = LocalRunner::new()
            .single_page()
            .checkpoint_delay(std::time::Duration::from_millis(20))
            .run(
                |n: i32, ctx: DurableContext| async move {
                    let v = ctx
                        .step(move |_| async move { Ok(n + 1) })
                        .name("inc")
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                41_i32,
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(result.output(), Some(&42));
        // The step's START and SUCCEED records both landed.
        assert!(
            result
                .operations()
                .iter()
                .any(|op| op.status == "Succeeded" && op.name.as_deref() == Some("inc")),
            "the step's coalesced checkpoints must be recorded: {:?}",
            result.operations()
        );
    }

    /// `checkpoint_batching` preserves handler semantics end-to-end: the
    /// same probe handler (concurrent steps, a suspension, a post-resume
    /// step) completes with the same output, and every step body runs
    /// exactly once across the suspend/resume boundary — proving the
    /// batched checkpoints flushed (and any in-flight batched write was
    /// awaited) before the invocation reported PENDING.
    #[tokio::test(start_paused = true)]
    async fn checkpoint_batching_preserves_handler_semantics() {
        let execs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let backend = Arc::new(Backend::new(Vec::new(), None));
        let result = LocalRunner::new()
            .single_page()
            .checkpoint_batching()
            .run_on_backend(
                Arc::clone(&backend),
                coalescing_probe_handler(Arc::clone(&execs)),
                (),
            )
            .await;

        assert!(result.is_success(), "{:?}", result.error_message());
        assert_eq!(
            result.output(),
            Some(&42),
            "batching must not change the handler's result"
        );
        assert_eq!(
            execs.load(Ordering::SeqCst),
            3,
            "each step body must run exactly once across suspend/resume — \
             a checkpoint held past the PENDING boundary would re-execute it"
        );
    }

    /// Review regression (issue #43): a failure RETAINED by a detached
    /// batch flush — its only contributor was dropped before the rejection
    /// published — must not be discarded, and the orphaned operation's
    /// unwritten outcome must be terminalized. The retained rejection here
    /// is NON-retryable (deterministic on every future invocation): were
    /// it dropped, the orphaned operation would stay `Started` and
    /// re-execute on every lap.
    ///
    /// With the coalescer's failure latch, the very next buffered write —
    /// the later live step's START — never reaches the backend: its
    /// flusher observes the latch under the writer lock, republishes the
    /// retained non-retryable error, and the live contributor's
    /// unrecoverable routing fails the execution. The end-of-invocation
    /// flush point then classifies the retained failures and persists the
    /// orphan's terminal FAIL. (The `Fault`-path retained-failure drain in
    /// the wrapper remains as defense in depth: with the latch, a
    /// non-retryable retained failure surfaces through the first
    /// subsequent contributor as here, before any retryable fault can.)
    #[tokio::test]
    async fn retained_nonretryable_failure_terminalizes_orphan_and_fails_execution() {
        let runner = LocalRunner::new().checkpoint_batching();
        let backend = Arc::new(Backend::new(Vec::new(), runner.checkpoint_page_size));

        // Call plan:
        //   1. orphan START — passes.
        //   2. orphan SUCCEED (detached batch flush) — held at the gate
        //      until the handler releases it, then rejected non-retryably.
        //      The handler drops the contributor while the write is held,
        //      so the rejection publishes to nobody and is only RETAINED
        //      (and latched).
        // The live step's START never reaches the backend (failure
        // latch), so the next real call is the orphan's terminal FAIL —
        // it falls past the exhausted plan and persists.
        let gate = Arc::new(tokio::sync::Semaphore::new(0));
        backend.plan_checkpoint_calls(vec![
            PlannedCheckpoint::Pass,
            PlannedCheckpoint::FailNonRetryable {
                gate: Some(Arc::clone(&gate)),
            },
        ]);

        let body_runs = Arc::new(AtomicU32::new(0));
        let body_runs_h = Arc::clone(&body_runs);
        let backend_h = Arc::clone(&backend);
        let gate_h = Arc::clone(&gate);

        let result = runner
            .run_on_backend(
                Arc::clone(&backend),
                move |(), ctx: DurableContext| {
                    let body_runs = Arc::clone(&body_runs_h);
                    let backend = Arc::clone(&backend_h);
                    let gate = Arc::clone(&gate_h);
                    async move {
                        let body_runs_step = Arc::clone(&body_runs);
                        let orphan = ctx
                            .step(move |_| {
                                let body_runs = Arc::clone(&body_runs_step);
                                async move {
                                    body_runs.fetch_add(1, Ordering::SeqCst);
                                    Ok("orphaned-outcome".to_owned())
                                }
                            })
                            .name("orphan")
                            .spawn();

                        // Wait until the orphan's SUCCEED write is IN
                        // FLIGHT (call 2 reached the backend, held at the
                        // gate), then drop the contributor and release the
                        // gate: the rejection now publishes to nobody and
                        // is only retained by the coalescer.
                        while backend.checkpoint_call_count() < 2 {
                            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
                        }
                        drop(orphan);
                        gate.add_permits(1);

                        // A later LIVE operation: its START write hits the
                        // failure latch — no backend call — and republishes
                        // the retained non-retryable rejection, whose
                        // unrecoverable routing fails the execution. Before
                        // the retention + latch, the retained failure above
                        // was silently discarded.
                        let live: String = ctx
                            .step(|_| async { Ok("live".to_owned()) })
                            .name("live")
                            .await?;
                        Ok::<_, BoxError>(live)
                    }
                },
                (),
            )
            .await;

        assert_eq!(
            body_runs.load(Ordering::SeqCst),
            1,
            "the orphaned body must run exactly once — a discarded retained \
             rejection would leave it `Started` and re-execute it"
        );
        assert_eq!(
            result.invocation_count(),
            1,
            "the retained non-retryable rejection is deterministic: the \
             execution must die in this invocation, not re-invoke into the \
             same rejection"
        );
        assert!(
            result.is_failure(),
            "the execution must fail — the orphaned operation's record \
             would otherwise claim less than what executed; got {:?} / {:?}",
            result.error_type(),
            result.error_message()
        );
        assert_eq!(
            result.error_type(),
            Some(crate::error::CHECKPOINT_FAILED_ERROR_TYPE)
        );

        // The terminal FAIL persisted for the orphaned operation.
        let ops = result.operations();
        let orphan_op = ops
            .iter()
            .find(|op| op.name.as_deref() == Some("orphan"))
            .expect("the orphaned step's operation record exists");
        assert_eq!(
            orphan_op.status, "Failed",
            "a terminal FAIL must be recorded for the operation whose \
             retained outcome rejection was classified at the flush point"
        );
    }

    // ── Harness fidelity: production wire-error mapping ─────────────────

    /// The runner reports the SAME wire error message production reports.
    /// Every failure flattens through the one flattening site, so a child
    /// failure's wire message is the full chain — one frame per layer —
    /// with the raw child message as its final frame. (The old special
    /// case that reported only the raw message was the Display/wire
    /// asymmetry the error-model redesign removed.)
    #[tokio::test]
    async fn child_failure_wire_error_matches_production_raw_message() {
        let result: TestResult<i32> = LocalRunner::new()
            .run(
                |(), ctx: DurableContext| async move {
                    let v = ctx
                        .run_in_child_context(|_child| async move {
                            Err::<i32, BoxError>("boom-child".into())
                        })
                        .await?;
                    Ok::<_, BoxError>(v)
                },
                (),
            )
            .await;

        assert!(result.is_failure());
        // The execution record re-records the child-context record's own
        // recorded identity ("ChildFnError", the boundary's fallback for a
        // plain boxed error) rather than degrading it to the outer kind's
        // registry name — recorded identity passes through boundaries.
        assert_eq!(result.error_type(), Some("ChildFnError"));
        let msg = result.error_message().unwrap_or_default();
        assert!(
            msg.contains("boom-child"),
            "raw child message must survive as the chain's final frame: {msg}"
        );
        assert_eq!(
            msg, "operation error: child_context: child failed: boom-child",
            "wire message is the flattened chain, one frame per layer"
        );
    }
}
