//! Block-level retry: run a closure against a child context and apply a
//! retry strategy to the closure's OVERALL outcome, so a multi-operation
//! block retries as a unit.
//!
//! Structure on the wire — everything is expressed through the existing,
//! backend-proven child-context and wait protocols; no new operation types
//! are introduced:
//!
//! - The `with_retry` operation itself is a child context
//!   (`OperationType::Context`, sub-type `RunInChildContext`) whose body is
//!   the retry loop.
//! - Each ATTEMPT is a nested child context under it (named `attempt-N`),
//!   which is what gives every attempt a fresh operation namespace: attempt
//!   1's operations live under `<id>-1-*`, attempt 2's under `<id>-3-*`,
//!   and so on. A failed attempt's recorded operations can therefore never
//!   be confused with (or replayed into) the next attempt's.
//! - The delay between attempts is a durable wait (named `retry-delay-N`),
//!   so the backend owns the retry timer and the execution suspends between
//!   attempts exactly as a retrying step does.
//!
//! Retry-state checkpointing: the loop carries no in-process state that
//! matters across a suspension. The current attempt number is re-derived on
//! every (re)entry purely from recorded checkpoint results — a finished
//! attempt replays its frozen outcome (a failed attempt replays its `Fail`
//! record without re-running its body), the strategy decision is a
//! deterministic function of that recorded error and the attempt number,
//! and the inter-attempt wait is itself a checkpointed operation. The retry
//! state therefore survives suspension by construction.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::context::DurableContext;
use crate::error::{ChildFnError, StepError, StepErrorKind};
use crate::{BoxError, RetryDecision, RetryStrategy};

/// The shared, re-invokable closure a `with_retry` block runs per attempt.
///
/// Unlike `run_in_child_context` (whose closure is `FnOnce`), a retry block
/// must be able to call the closure once per attempt, so it is stored as a
/// shared `Fn`.
pub(crate) type WithRetryClosure<O> = Arc<
    dyn Fn(DurableContext) -> Pin<Box<dyn Future<Output = Result<O, BoxError>> + Send>>
        + Send
        + Sync,
>;

/// Runs the retry loop inside the outer `with_retry` child context.
///
/// `outer` is the child context whose namespace the loop owns; the loop
/// mints one nested child context per attempt plus one wait per retry
/// delay from it. Returns the first successful attempt's value, or a
/// `ChildFnError` describing exhaustion (carrying the last attempt's
/// error), which the enclosing child-context protocol checkpoints as the
/// operation's permanent failure.
pub(crate) async fn retry_loop<O>(
    outer: DurableContext,
    closure: WithRetryClosure<O>,
    strategy: Arc<RetryStrategy>,
) -> Result<O, ChildFnError>
where
    O: Serialize + DeserializeOwned + Send + 'static,
{
    let mut attempt: u32 = 1;
    loop {
        // One nested child context per attempt: a fresh operation
        // namespace, so nothing recorded by a previous (failed) attempt
        // replays into this one. On replay, a finished attempt returns its
        // frozen outcome without re-running the closure.
        let f = Arc::clone(&closure);
        let result: Result<O, crate::error::OperationError> = outer
            .run_in_child_context(move |attempt_ctx| f(attempt_ctx))
            .name(format!("attempt-{attempt}"))
            .await;

        let err = match result {
            Ok(value) => return Ok(value),
            Err(err) => err,
        };

        // Present the failure to the strategy in the same shape a step's
        // strategy sees. Live and replayed failures carry the same message
        // (the replay path reads the recorded error), so a strategy that
        // inspects the error decides identically across invocations.
        let step_err = StepError::from_kind(StepErrorKind::ExecutionFailed {
            message: err.to_string(),
        });
        match strategy(&step_err, attempt) {
            RetryDecision::Retry { delay } => {
                // Durable wait between attempts: checkpoint-suspend, the
                // backend owns the timer. Enforce the same 1-second minimum
                // a step retry does (`ctx.wait` already rounds fractional
                // delays up to whole seconds).
                outer
                    .wait(delay.max(Duration::from_secs(1)))
                    .name(format!("retry-delay-{attempt}"))
                    .await
                    .map_err(|wait_err| {
                        ChildFnError::new(format!("with_retry delay wait failed: {wait_err}"))
                    })?;
                attempt = attempt.saturating_add(1);
            }
            RetryDecision::Stop => {
                return Err(ChildFnError::new(format!(
                    "retries exhausted after {attempt} attempts: {err}"
                )));
            }
        }
    }
}
