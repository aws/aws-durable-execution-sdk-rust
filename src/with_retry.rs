//! Block-level retry: run a closure against a child context and apply a
//! retry strategy to the closure's OVERALL outcome, so a multi-operation
//! block retries as a unit.
//!
//! Structure on the wire: everything is expressed through the existing,
//! backend-proven child-context and wait protocols; no new operation types
//! are introduced:
//!
//! - The `with_retry` operation itself is a child context
//!   (`OperationType::Context`) whose body is the retry loop. Its sub-type
//!   is the one the caller sets with `WithRetryBuilder::sub_type`, and
//!   `RunInChildContext` when the caller sets none. With
//!   `WithRetryBuilder::without_child_context`, the block writes no record
//!   of its own: the loop runs in the caller's context, and a block named
//!   `b` names its attempts `b-attempt-N`. With
//!   `WithRetryBuilder::virtual_context`, the block is a virtual child
//!   context: the loop runs in its own namespace but the block writes no
//!   record, and a block named `b` names its attempts `b-attempt-N`.
//! - Each ATTEMPT is a nested child context under it (named `attempt-N`,
//!   sub-type `RunInChildContext` whatever the outer sub-type is),
//!   which is what gives every attempt a fresh operation namespace: attempt
//!   1's operations live under `<id>-1-*`, attempt 2's under `<id>-3-*`,
//!   and so on. A failed attempt's recorded operations can therefore never
//!   be confused with (or replayed into) the next attempt's.
//! - The delay after failed attempt N is a durable wait, named
//!   `{name}-backoff-N` when the block has a non-empty name and unnamed
//!   otherwise, as in the JS SDK. So the backend owns the retry timer, and
//!   the block waits between attempts exactly as a retrying step does: the
//!   invocation suspends during the wait only when no other SDK work can
//!   make progress.
//!
//! Retry-state checkpointing: the loop carries no in-process state that
//! matters across a suspension. The current attempt number is re-derived on
//! every (re)entry purely from recorded checkpoint results: a finished
//! attempt replays its frozen outcome (a failed attempt replays its `Fail`
//! record without re-running its body), the strategy decision is a
//! deterministic function of that recorded error and the attempt number,
//! and the inter-attempt wait is itself a checkpointed operation. The retry
//! state therefore survives suspension by construction.

use std::future::{Future, IntoFuture};
use std::sync::Arc;

use crate::context::DurableContext;
use crate::error::{ChildFnError, OperationError, StepError, StepErrorKind};
use crate::future::DurableFuture;
use crate::serdes::Serdes;
use crate::{BoxError, RetryDecision, RetryStrategy};

/// Whether a `with_retry` block writes a child context record of its own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum RetryWrap {
    /// The block is a child context, and its attempts are `attempt-N`
    /// children inside it.
    #[default]
    Child,
    /// The block writes no record: its attempts and waits are created in
    /// the caller's context, as JS `wrapWithRunInChildContext: false`.
    None,
    /// The block is a virtual child context: it claims an ID and its
    /// attempts and waits are created in its own namespace, but it writes
    /// no record, so they name the caller's parent as their parent.
    Virtual,
}

/// Why the retry loop ended without a value.
#[derive(Debug)]
pub(crate) enum LoopFailure {
    /// The strategy stopped: the last attempt's error, unchanged.
    Exhausted(OperationError),
    /// The wait between two attempts failed: the wait's error.
    Wait(OperationError),
}

impl LoopFailure {
    /// The error a block with its own child context records: the last
    /// attempt's error as the child failure's direct source, or the failed
    /// wait under `with_retry delay wait failed`.
    pub(crate) fn into_child_error(self) -> ChildFnError {
        match self {
            Self::Exhausted(last) => ChildFnError::new(last),
            Self::Wait(wait_err) => ChildFnError::new(crate::error::ContextualError::source_from(
                "with_retry delay wait failed",
                Box::new(wait_err) as crate::error::Source,
            )),
        }
    }

    /// The error an unwrapped block returns: the attempt's or the wait's
    /// error, unchanged.
    pub(crate) fn into_operation_error(self) -> OperationError {
        match self {
            Self::Exhausted(err) | Self::Wait(err) => err,
        }
    }
}

/// The name of attempt `attempt`: `{block}-attempt-N` for an unwrapped or
/// virtual block with a non-empty name, `attempt-N` otherwise.
pub(crate) fn attempt_name(block: Option<&str>, wrap: RetryWrap, attempt: u32) -> String {
    match (wrap, block) {
        (RetryWrap::None | RetryWrap::Virtual, Some(block)) if !block.is_empty() => {
            format!("{block}-attempt-{attempt}")
        }
        _ => format!("attempt-{attempt}"),
    }
}

/// Creates the future of attempt `attempt` as a nested child context of
/// `outer`, which claims its operation ID now. With `started`, the
/// attempt's start step runs now, on the caller's task.
///
/// The closure is shared as `Arc<F>`, because the block runs it once per
/// attempt, and receives the attempt number with the attempt's context.
/// Each attempt round-trips `O` through the builder's serdes, shared
/// through the forwarding `impl Serdes<T> for Arc<S>`.
pub(crate) fn start_attempt<O, F, Fut, S>(
    outer: &DurableContext,
    name: String,
    attempt: u32,
    closure: &Arc<F>,
    serdes: &Arc<S>,
    started: bool,
) -> DurableFuture<O>
where
    O: Send + 'static,
    F: Fn(DurableContext, u32) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    let f = Arc::clone(closure);
    let n = attempt;
    let builder = outer
        .run_in_child_context(move |attempt_ctx| f(attempt_ctx, n))
        .without_user_function()
        .name(name)
        .serdes(Arc::clone(serdes));
    if started {
        builder.started_future()
    } else {
        builder.into_future()
    }
}

/// Runs the retry loop in `outer`.
///
/// `outer` is the block's own child context, or the caller's context for
/// an unwrapped block. The loop creates one nested child context per
/// attempt plus one wait per retry delay in it. `first` is attempt 1's
/// future when the caller already created it; otherwise the loop creates
/// it. Returns the first successful attempt's value, or the
/// [`LoopFailure`] that ended the loop.
///
/// The closure is generic (`Arc<F>` rather than a boxed `dyn Fn`): each
/// attempt produces a concrete future without its own box, so the single
/// erasure point stays at the enclosing builder's
/// [`DurableFuture`](crate::DurableFuture). `name` is the block's name; it
/// names the waits between attempts, and an unwrapped block's attempts.
pub(crate) async fn retry_loop<O, F, Fut, S>(
    outer: DurableContext,
    name: Option<String>,
    wrap: RetryWrap,
    first: Option<DurableFuture<O>>,
    closure: Arc<F>,
    strategy: Arc<RetryStrategy>,
    serdes: Arc<S>,
) -> Result<O, LoopFailure>
where
    O: Send + 'static,
    F: Fn(DurableContext, u32) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send + 'static,
    S: Serdes<O>,
{
    let mut attempt: u32 = 1;
    let mut first = first;
    loop {
        // One nested child context per attempt: a fresh operation
        // namespace, so nothing recorded by a previous (failed) attempt
        // replays into this one. On replay, a finished attempt returns its
        // frozen outcome without calling the closure.
        let future = match first.take() {
            Some(future) => future,
            None => start_attempt(
                &outer,
                attempt_name(name.as_deref(), wrap, attempt),
                attempt,
                &closure,
                &serdes,
                false,
            ),
        };
        let err = match future.await {
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
                // Durable wait between attempts: checkpointed, and the
                // backend owns the timer. `ctx.wait` rounds the delay up
                // to whole seconds with a one-second minimum. JS names
                // this wait `{name}-backoff-{attempt}` only when the
                // block's name is non-empty, so an unnamed block's waits
                // are unnamed.
                let wait = outer.wait(delay);
                let wait = match name.as_deref() {
                    Some(name) if !name.is_empty() => {
                        wait.name(format!("{name}-backoff-{attempt}"))
                    }
                    _ => wait,
                };
                wait.await.map_err(LoopFailure::Wait)?;
                attempt = attempt.saturating_add(1);
            }
            RetryDecision::Stop => return Err(LoopFailure::Exhausted(last_error(step_err))),
        }
    }
}

/// Takes the attempt's error back out of the step error the strategy saw.
#[expect(clippy::expect_used)]
// reason: `retry_loop` builds the step error with `Some(Box<OperationError>)`
// as its source, so both the source and the downcast are always present.
fn last_error(step_err: StepError) -> OperationError {
    *step_err
        .into_source()
        .expect("the step error carries the attempt's error")
        .downcast::<OperationError>()
        .expect("the attempt's error is an OperationError")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{ChildContextError, ChildContextErrorKind, OperationErrorKind};

    fn op_error(kind: ChildContextErrorKind) -> OperationError {
        OperationError::from_kind(OperationErrorKind::ChildContext(ChildContextError::new(
            kind, None,
        )))
    }

    /// Exhaustion makes the last attempt's error the child failure's
    /// direct source; a failed wait goes under `with_retry delay wait
    /// failed`. An unwrapped block returns either error unchanged.
    #[test]
    fn loop_failure_maps_to_child_and_operation_errors() {
        let exhausted = LoopFailure::Exhausted(op_error(ChildContextErrorKind::ChildFailed))
            .into_child_error()
            .into_source();
        let last = exhausted
            .downcast_ref::<OperationError>()
            .expect("the attempt error is the direct source");
        assert!(matches!(last.kind(), OperationErrorKind::ChildContext(_)));

        let waited = LoopFailure::Wait(op_error(ChildContextErrorKind::Internal))
            .into_child_error()
            .into_source();
        assert_eq!(waited.to_string(), "with_retry delay wait failed");
        assert!(
            std::error::Error::source(&*waited)
                .and_then(|s| s.downcast_ref::<OperationError>())
                .is_some()
        );

        let err = LoopFailure::Exhausted(op_error(ChildContextErrorKind::ChildFailed))
            .into_operation_error();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::ChildContext(child)
                if matches!(child.kind(), ChildContextErrorKind::ChildFailed)
        ));
        let err =
            LoopFailure::Wait(op_error(ChildContextErrorKind::Internal)).into_operation_error();
        assert!(matches!(
            err.kind(),
            OperationErrorKind::ChildContext(child)
                if matches!(child.kind(), ChildContextErrorKind::Internal)
        ));
    }

    /// Only an unwrapped or virtual block with a non-empty name prefixes its
    /// attempts.
    #[test]
    fn attempt_names() {
        assert_eq!(attempt_name(Some("b"), RetryWrap::None, 2), "b-attempt-2");
        assert_eq!(attempt_name(Some(""), RetryWrap::None, 2), "attempt-2");
        assert_eq!(attempt_name(None, RetryWrap::None, 2), "attempt-2");
        assert_eq!(attempt_name(Some("b"), RetryWrap::Child, 2), "attempt-2");
        assert_eq!(
            attempt_name(Some("b"), RetryWrap::Virtual, 2),
            "b-attempt-2"
        );
        assert_eq!(attempt_name(Some(""), RetryWrap::Virtual, 2), "attempt-2");
        assert_eq!(attempt_name(None, RetryWrap::Virtual, 2), "attempt-2");
    }
}
