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
use std::sync::Arc;
use std::time::Duration;

use crate::context::DurableContext;
use crate::error::{ChildFnError, StepError, StepErrorKind};
use crate::serdes::Serdes;
use crate::{BoxError, RetryDecision, RetryStrategy};

/// Runs the retry loop inside the outer `with_retry` child context.
///
/// `outer` is the child context whose namespace the loop owns; the loop
/// mints one nested child context per attempt plus one wait per retry
/// delay from it. Returns the first successful attempt's value, or a
/// `ChildFnError` describing exhaustion (carrying the last attempt's
/// error), which the enclosing child-context protocol checkpoints as the
/// operation's permanent failure.
/// The closure is generic (`Arc<F>` rather than a boxed `dyn Fn`): each
/// attempt produces a concrete future that flows into
/// `run_in_child_context` without its own box — the single erasure point
/// stays at the enclosing builder's [`DurableFuture`](crate::DurableFuture).
/// `Arc` because the block runs the closure once per attempt.
/// The builder's configured serdes is shared into every attempt's nested
/// child context (through the forwarding `impl Serdes<T> for Arc<S>`), so
/// each attempt round-trips `O` through the SAME wire format as the block
/// itself — which is also what lets a custom serdes carry an `O` without
/// `Serialize`/`DeserializeOwned` implementations.
pub(crate) async fn retry_loop<O, F, Fut, S>(
    outer: DurableContext,
    closure: Arc<F>,
    strategy: Arc<RetryStrategy>,
    serdes: Arc<S>,
) -> Result<O, ChildFnError>
where
    O: Send + 'static,
    F: Fn(DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
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
            .serdes(Arc::clone(&serdes))
            .await;

        let err = match result {
            Ok(value) => return Ok(value),
            Err(err) => err,
        };

        // Present the failure to the strategy in the same shape a step's
        // strategy sees: the attempt's error is the step error's source.
        // Live and replayed failures carry the same recorded failure (the
        // replay path reads the recorded error), so a strategy that
        // inspects the error decides identically across invocations.
        let step_err = StepError::new(StepErrorKind::ExecutionFailed, Some(Box::new(err)));
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
                        ChildFnError::new(crate::error::ContextualError::source_from(
                            "with_retry delay wait failed",
                            Box::new(wait_err) as crate::error::Source,
                        ))
                    })?;
                attempt = attempt.saturating_add(1);
            }
            RetryDecision::Stop => {
                // Exhaustion carries the last attempt's error as its
                // source rather than a flattened string.
                let last = step_err
                    .into_source()
                    .unwrap_or_else(|| "with_retry attempt failed".into());
                return Err(ChildFnError::new(
                    crate::error::ContextualError::source_from(
                        format!("retries exhausted after {attempt} attempts"),
                        last,
                    ),
                ));
            }
        }
    }
}
