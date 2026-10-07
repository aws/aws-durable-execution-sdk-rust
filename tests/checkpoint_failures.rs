//! Checkpoint write failures unwind the handler, never surface to it
//! (issue #43).
//!
//! Drives handlers through [`LocalRunner`] with injected checkpoint
//! failures and asserts the three acceptance tests the issue names:
//! - a step whose result the service permanently rejects fails the
//!   execution after ONE body execution, not at the execution timeout;
//! - a handler that wraps a step in a catch cannot observe a checkpoint
//!   API failure;
//! - a step whose result fails serialization runs its body exactly once
//!   across an invocation boundary, and replay yields the recorded `FAIL`.
//!
//! Plus the retryable half of the contract: an exhausted transient
//! checkpoint failure fails the INVOCATION (no further writes), the
//! service re-invokes, and the execution converges on the re-run.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::indexing_slicing)] // reason: test assertions

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::LocalRunner;

/// The wire error type a checkpoint-write failure records: internal to
/// the SDK, asserted here through the public `TestResult` surface.
const CHECKPOINT_FAILED: &str = "CheckpointFailedError";

type StepOutcomeFuture = Pin<Box<dyn Future<Output = Result<String, durable::BoxError>> + Send>>;

fn step_outcome_handler(
    body_runs: Arc<AtomicU32>,
) -> impl Fn(serde_json::Value, durable::DurableContext) -> StepOutcomeFuture {
    move |_event, ctx| {
        let body_runs = Arc::clone(&body_runs);
        Box::pin(async move {
            let outcome = ctx
                .step(move |_| {
                    let body_runs = Arc::clone(&body_runs);
                    async move {
                        body_runs.fetch_add(1, Ordering::SeqCst);
                        Ok("step-outcome".to_owned())
                    }
                })
                .name("the-step")
                .await?;
            Ok::<_, durable::BoxError>(outcome)
        })
    }
}

/// Issue #43 acceptance test 1: a step whose result the service
/// permanently rejects fails the execution after one body execution, not
/// at the execution timeout.
///
/// `fail_checkpoints_after(1, 1)` lets the step's START write through and
/// permanently rejects its SUCCEED write (the oversized-result shape). The
/// old behavior looped: body runs, write rejected, invocation errors,
/// re-invoke finds `Started`, body re-runs: until the execution timeout,
/// side effects firing once per lap. Now the SDK persists a small terminal
/// `FAIL` for the step and fails the execution in the same invocation.
#[tokio::test]
async fn permanently_rejected_result_fails_execution_after_one_body_run() {
    let body_runs = Arc::new(AtomicU32::new(0));
    let body_runs_h = Arc::clone(&body_runs);

    let result = LocalRunner::new()
        .fail_checkpoints_after(1, 1)
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs_h);
                async move {
                    let out = ctx
                        .step(move |_| {
                            let body_runs = Arc::clone(&body_runs);
                            async move {
                                body_runs.fetch_add(1, Ordering::SeqCst);
                                Ok("rejected-by-service".to_owned())
                            }
                        })
                        .name("permanently-rejected")
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(
        body_runs.load(Ordering::SeqCst),
        1,
        "the body must execute exactly once — a permanent rejection must \
         not re-run it once per invocation lap"
    );
    assert_eq!(
        result.invocation_count(),
        1,
        "the execution must die in the invocation that saw the rejection, \
         not spin until the execution timeout"
    );
    assert!(result.is_failure(), "the execution must fail");
    assert_eq!(
        result.error_type(),
        Some(CHECKPOINT_FAILED),
        "the FAILED envelope carries the checkpoint-failure type"
    );

    // The terminal FAIL persisted: the step's record claims exactly what
    // executed (it ran, and its outcome could not be recorded), rather
    // than a dangling `Started`.
    let step_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("permanently-rejected"))
        .expect("the step's operation record exists");
    assert!(
        step_op.failed(),
        "a terminal FAIL is recorded for the step whose result was \
         rejected, ending the re-execution loop after one lap; got {}",
        step_op.status()
    );
    assert_eq!(step_op.error_type(), Some(CHECKPOINT_FAILED));
}

/// Issue #43 acceptance test 2: a handler that wraps a step in a catch
/// cannot observe a checkpoint API failure.
///
/// The step's SUCCEED write is permanently rejected. Pre-#43 the failure
/// surfaced as a catchable `OperationError`, letting the handler branch on
/// a decision no checkpoint records (replay divergence). Now the handler
/// future is dropped at the await point, neither the catch branch nor any
/// code after the step runs, and the execution fails.
#[tokio::test]
async fn catch_cannot_observe_checkpoint_api_failure() {
    let caught = Arc::new(AtomicU32::new(0));
    let caught_h = Arc::clone(&caught);
    let after_step = Arc::new(AtomicU32::new(0));
    let after_step_h = Arc::clone(&after_step);

    let result = LocalRunner::new()
        .fail_checkpoints_after(1, 1)
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let caught = Arc::clone(&caught_h);
                let after_step = Arc::clone(&after_step_h);
                async move {
                    // A catch-all around the step: the checkpoint failure
                    // must not be observable here.
                    let step_result = ctx
                        .step(|_| async { Ok("value".to_owned()) })
                        .name("caught-step")
                        .await;
                    if let Ok(v) = step_result {
                        after_step.fetch_add(1, Ordering::SeqCst);
                        Ok::<_, durable::BoxError>(v)
                    } else {
                        caught.fetch_add(1, Ordering::SeqCst);
                        Ok("recovered-from-checkpoint-failure".to_owned())
                    }
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(
        caught.load(Ordering::SeqCst),
        0,
        "the catch branch must never observe a checkpoint API failure"
    );
    assert_eq!(
        after_step.load(Ordering::SeqCst),
        0,
        "no code after the step may run — the handler is dropped at the \
         await point"
    );
    assert!(
        result.is_failure(),
        "the execution fails; the handler's 'recovered' value must not win"
    );
    assert_eq!(result.error_type(), Some(CHECKPOINT_FAILED));
}

/// A serdes whose serialize side always fails: the local, deterministic
/// failure shape (as opposed to a service-side rejection).
#[derive(Debug)]
struct AlwaysFailSerialize;

impl durable::Serdes<String> for AlwaysFailSerialize {
    // reason: exercises the async-fn impl form user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn serialize(
        &self,
        _value: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Err("this result cannot be serialized".into())
    }

    // reason: exercises the async-fn impl form user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn deserialize(
        &self,
        wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(wire)
    }
}

/// Issue #43 acceptance test 3: a step whose result fails serialization
/// runs its body exactly once across an invocation boundary, and replay
/// yields the recorded `FAIL`.
///
/// A serialization failure is local and deterministic, so it stays
/// catchable, but the SDK persists the `FAIL` checkpoint BEFORE yielding
/// it. The handler catches it, then suspends on a wait; the next
/// invocation replays the step from the recorded `FAIL` instead of
/// re-running the body.
#[tokio::test]
async fn serialization_failure_persists_fail_and_replays_without_rerun() {
    let body_runs = Arc::new(AtomicU32::new(0));
    let body_runs_h = Arc::clone(&body_runs);
    let caught = Arc::new(AtomicU32::new(0));
    let caught_h = Arc::clone(&caught);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs_h);
                let caught = Arc::clone(&caught_h);
                async move {
                    let step_result = ctx
                        .step(move |_| {
                            let body_runs = Arc::clone(&body_runs);
                            async move {
                                body_runs.fetch_add(1, Ordering::SeqCst);
                                Ok("unserializable".to_owned())
                            }
                        })
                        .name("bad-serdes")
                        .serdes(AlwaysFailSerialize)
                        .await;

                    // Catchable: a serialization failure is a user-facing
                    // error. Count how many times the handler observes it
                    // (once live, once from replay).
                    let err = step_result.expect_err("serialization must fail");
                    caught.fetch_add(1, Ordering::SeqCst);
                    let live_message = err.to_string();

                    // Cross an invocation boundary: the wait suspends and
                    // the runner re-invokes; the step must replay its
                    // recorded FAIL without re-running the body.
                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;

                    Ok::<_, durable::BoxError>(live_message)
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.invocation_count() >= 2,
        "the wait must force a second invocation; got {}",
        result.invocation_count()
    );
    assert_eq!(
        body_runs.load(Ordering::SeqCst),
        1,
        "the body runs exactly once across the invocation boundary — \
         replay yields the recorded FAIL instead of re-running it"
    );
    assert_eq!(
        caught.load(Ordering::SeqCst),
        u32::try_from(result.invocation_count()).expect("small count"),
        "every invocation observes the failure: live once, then from the \
         recorded FAIL"
    );
    assert!(
        result.is_success(),
        "the handler recovered from the caught failure; the execution \
         succeeds: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );

    // The FAIL was persisted BEFORE the error was yielded.
    let step_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("bad-serdes"))
        .expect("the step's operation record exists");
    assert!(
        step_op.failed(),
        "the serialization failure is recorded as a terminal FAIL; got {}",
        step_op.status()
    );
    assert!(
        step_op
            .error_message()
            .is_some_and(|m| m.contains("cannot be serialized")),
        "the recorded FAIL carries the serialization error: {:?}",
        step_op.error_message()
    );
}

/// The retryable half of the #43 contract: a checkpoint failure that
/// exhausts the client's internal retries fails the INVOCATION with no
/// further writes. The service re-invokes (the interruption recovery
/// path), replay finds the step still `Started`, the body re-runs under
/// the documented `AtLeastOncePerRetry` contract, and the execution
/// converges when the channel recovers.
#[tokio::test]
async fn retryable_exhaustion_fails_invocation_and_reinvocation_converges() {
    let body_runs = Arc::new(AtomicU32::new(0));
    let body_runs_h = Arc::clone(&body_runs);

    let result = LocalRunner::new()
        .fail_checkpoints_after_retryable(1, 1)
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs_h);
                async move {
                    let out = ctx
                        .step(move |_| {
                            let body_runs = Arc::clone(&body_runs);
                            async move {
                                body_runs.fetch_add(1, Ordering::SeqCst);
                                Ok("eventually-recorded".to_owned())
                            }
                        })
                        .name("transient-channel")
                        .await?;
                    Ok::<_, durable::BoxError>(out)
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_success(),
        "the execution converges once the channel recovers: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&"eventually-recorded".to_owned()));
    assert_eq!(
        result.invocation_count(),
        2,
        "the exhausted transient failure costs exactly one invocation"
    );
    assert_eq!(
        body_runs.load(Ordering::SeqCst),
        2,
        "the interrupted attempt re-runs the body (AtLeastOncePerRetry): \
         the record stayed Started because nothing further was written"
    );
    let step_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("transient-channel"))
        .expect("the step's operation record exists");
    assert!(
        step_op.succeeded(),
        "the re-run's SUCCEED write goes through; got {}",
        step_op.status()
    );
}

/// A serdes whose serialize side always fails, generic over the value type
/// (the local, deterministic failure shape for any operation).
#[derive(Debug)]
struct FailSerialize;

impl<T: Send + 'static> durable::Serdes<T> for FailSerialize {
    // reason: exercises the async-fn impl form user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn serialize(
        &self,
        _value: T,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Err("this result cannot be serialized".into())
    }

    // reason: exercises the async-fn impl form user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn deserialize(
        &self,
        _wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<T, durable::BoxError> {
        Err("unreachable: nothing was ever serialized".into())
    }
}

/// The wire error type a local serialization failure records on its
/// terminal `FAIL`: the replay discriminator that keeps the failure's
/// classification stable across invocations.
const SERIALIZATION_FAILED: &str = "SerializationError";

/// The kind a step error reports, as a comparable label (the enum's
/// `Debug` rendering: its variants are `#[non_exhaustive]`, so they
/// cannot be pattern-matched outside the crate).
fn step_kind_label(err: &durable::OperationError) -> String {
    match err.kind() {
        durable::OperationErrorKind::Step(step_err) => format!("{:?}", step_err.kind()),
        other => format!("non-step: {other}"),
    }
}

/// Review finding 1: the live path yields
/// `StepErrorKind::SerializationFailed` after persisting the FAIL, and
/// replay must reconstruct the SAME kind from the recorded
/// `SerializationError` type: a handler that branches on the kind takes
/// the same path live and replayed.
#[tokio::test]
async fn step_serialization_failure_kind_is_replay_equivalent() {
    let kinds = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let kinds_h = Arc::clone(&kinds);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let kinds = Arc::clone(&kinds_h);
                async move {
                    let step_result = ctx
                        .step(|_| async { Ok("unserializable".to_owned()) })
                        .name("bad-serdes")
                        .serdes(FailSerialize)
                        .await;
                    let err = step_result.expect_err("serialization must fail");
                    kinds
                        .lock()
                        .expect("test mutex")
                        .push(step_kind_label(&err));

                    // Cross an invocation boundary so the second entry is
                    // reconstructed from the recorded FAIL.
                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, durable::BoxError>("recovered".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    let kinds = kinds.lock().expect("test mutex");
    assert!(
        kinds.len() >= 2,
        "need one live and at least one replayed observation, got {kinds:?}"
    );
    assert!(
        kinds.iter().all(|k| k == "SerializationFailed"),
        "live and replayed kinds must both be SerializationFailed: {kinds:?}"
    );

    // The recorded FAIL carries the serialization discriminator.
    let step_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("bad-serdes"))
        .expect("the step's operation record exists");
    assert_eq!(step_op.error_type(), Some(SERIALIZATION_FAILED));
}

/// An ordinary terminal step failure classifies as
/// `StepErrorKind::RetriesExhausted` on the live path, and replay must
/// reconstruct the SAME kind carrying the SAME attempt count from the
/// recorded FAIL: a handler that branches on the kind (or the attempt)
/// creates the same operations live and replayed.
#[tokio::test]
async fn step_retries_exhausted_kind_is_replay_equivalent() {
    let kinds = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let kinds_h = Arc::clone(&kinds);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let kinds = Arc::clone(&kinds_h);
                async move {
                    let step_result = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom".into()) })
                        .name("always-fails")
                        .retry_strategy(|_err, attempt| {
                            if attempt >= 2 {
                                durable::RetryDecision::Stop
                            } else {
                                durable::RetryDecision::Retry {
                                    delay: Duration::from_secs(1),
                                }
                            }
                        })
                        .await;
                    let err = step_result.expect_err("the step must fail terminally");
                    // The full Debug rendering carries both the variant and
                    // its attempt payload, so equality across invocations
                    // asserts kind AND attempt equivalence.
                    kinds
                        .lock()
                        .expect("test mutex")
                        .push(step_kind_label(&err));

                    // Cross an invocation boundary so the next observation
                    // is reconstructed from the recorded FAIL.
                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, durable::BoxError>("recovered".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    let kinds = kinds.lock().expect("test mutex");
    assert!(
        kinds.len() >= 2,
        "need one live and at least one replayed observation, got {kinds:?}"
    );
    assert!(
        kinds[0].contains("RetriesExhausted"),
        "the live kind must be RetriesExhausted: {kinds:?}"
    );
    assert!(
        kinds[0].contains("attempts: 2"),
        "the live kind must carry the failing attempt: {kinds:?}"
    );
    assert!(
        kinds.windows(2).all(|w| w[0] == w[1]),
        "live and replayed kinds (including attempts) must match: {kinds:?}"
    );
}

/// Review finding 2 (child context): a child whose result fails
/// serialization persists a terminal FAIL BEFORE yielding, so the closure
/// runs exactly once across an invocation boundary and replay yields the
/// recorded failure with the same kind.
#[tokio::test]
async fn child_serialization_failure_persists_fail_and_replays_without_rerun() {
    let closure_runs = Arc::new(AtomicU32::new(0));
    let closure_runs_h = Arc::clone(&closure_runs);
    let kinds = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let kinds_h = Arc::clone(&kinds);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                let kinds = Arc::clone(&kinds_h);
                async move {
                    let child_result = ctx
                        .run_in_child_context(move |_child| {
                            let closure_runs = Arc::clone(&closure_runs);
                            async move {
                                closure_runs.fetch_add(1, Ordering::SeqCst);
                                Ok("unserializable".to_owned())
                            }
                        })
                        .name("bad-child")
                        .serdes(FailSerialize)
                        .await;
                    let err = child_result.expect_err("serialization must fail");
                    let label = match err.kind() {
                        durable::OperationErrorKind::ChildContext(child_err) => {
                            format!("{:?}", child_err.kind())
                        }
                        other => format!("non-child: {other}"),
                    };
                    kinds.lock().expect("test mutex").push(label);

                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, durable::BoxError>("recovered".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        closure_runs.load(Ordering::SeqCst),
        1,
        "the child closure runs exactly once — replay yields the recorded FAIL"
    );
    let kinds = kinds.lock().expect("test mutex");
    assert!(kinds.len() >= 2, "live + replayed observations: {kinds:?}");
    assert!(
        kinds.windows(2).all(|w| w[0] == w[1]),
        "live and replayed child error kinds must match: {kinds:?}"
    );

    let child_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("bad-child"))
        .expect("the child's operation record exists");
    assert!(child_op.failed(), "got {}", child_op.status());
    assert_eq!(child_op.error_type(), Some(SERIALIZATION_FAILED));
}

/// Review finding 2 (map, normal nesting): an item whose result fails
/// serialization persists a terminal FAIL for its child record and
/// settles as a failed `BatchItem`, the same shape a replay of that
/// record produces, so the item closure runs exactly once and live and
/// replayed batches agree.
#[tokio::test]
async fn map_item_serialization_failure_records_fail_and_replays_as_failed_item() {
    let closure_runs = Arc::new(AtomicU32::new(0));
    let closure_runs_h = Arc::clone(&closure_runs);
    let observed = Arc::new(std::sync::Mutex::new(Vec::<(String, Option<String>)>::new()));
    let observed_h = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                let observed = Arc::clone(&observed_h);
                async move {
                    let batch = ctx
                        .map(vec![1_u32], move |_child, _item, _idx| {
                            let closure_runs = Arc::clone(&closure_runs);
                            async move {
                                closure_runs.fetch_add(1, Ordering::SeqCst);
                                Ok("unserializable".to_owned())
                            }
                        })
                        .name("bad-batch")
                        .serdes(FailSerialize)
                        .await_batch()
                        .await?;
                    let item = batch.items.first().expect("one item settled");
                    observed
                        .lock()
                        .expect("test mutex")
                        .push((format!("{:?}", item.status), item.error_type.clone()));

                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, durable::BoxError>("recovered".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        closure_runs.load(Ordering::SeqCst),
        1,
        "the item closure runs exactly once — replay reconstructs the \
         failed item from the recorded FAIL"
    );
    let observed = observed.lock().expect("test mutex");
    assert!(
        observed.len() >= 2,
        "live + replayed observations: {observed:?}"
    );
    assert!(
        observed
            .iter()
            .all(|(status, error_type)| status == "Failed"
                && error_type.as_deref() == Some(SERIALIZATION_FAILED)),
        "every observation is the same recorded serialization failure: {observed:?}"
    );
}

/// Review finding 2 (map, FLAT nesting): a flat item has no per-child
/// record, so its serialization failure settles as a failed `BatchItem`
/// recorded inside the parent batch's summary: replay reconstructs the
/// same failed item from the parent record without re-running the closure.
#[tokio::test]
async fn flat_item_serialization_failure_is_recorded_in_batch_summary() {
    use durable::builders::map_parallel::NestingMode;

    let closure_runs = Arc::new(AtomicU32::new(0));
    let closure_runs_h = Arc::clone(&closure_runs);
    let observed = Arc::new(std::sync::Mutex::new(Vec::<(String, Option<String>)>::new()));
    let observed_h = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let closure_runs = Arc::clone(&closure_runs_h);
                let observed = Arc::clone(&observed_h);
                async move {
                    let batch = ctx
                        .map(vec![1_u32], move |_child, _item, _idx| {
                            let closure_runs = Arc::clone(&closure_runs);
                            async move {
                                closure_runs.fetch_add(1, Ordering::SeqCst);
                                Ok("unserializable".to_owned())
                            }
                        })
                        .name("flat-batch")
                        .nesting(NestingMode::Flat)
                        .serdes(FailSerialize)
                        .await_batch()
                        .await?;
                    let item = batch.items.first().expect("one item settled");
                    observed
                        .lock()
                        .expect("test mutex")
                        .push((format!("{:?}", item.status), item.error_type.clone()));

                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, durable::BoxError>("recovered".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        closure_runs.load(Ordering::SeqCst),
        1,
        "the flat item closure runs exactly once — the failure is recorded \
         in the parent summary and replays from it"
    );
    let observed = observed.lock().expect("test mutex");
    assert!(
        observed.len() >= 2,
        "live + replayed observations: {observed:?}"
    );
    assert!(
        observed
            .iter()
            .all(|(status, error_type)| status == "Failed"
                && error_type.as_deref() == Some(SERIALIZATION_FAILED)),
        "every observation is the same recorded serialization failure: {observed:?}"
    );
}

/// Review finding 2 (`wait_for_condition`): a state serialization failure
/// persists a terminal FAIL before yielding `SerializationFailed`, so the
/// check runs exactly once across an invocation boundary and replay
/// reconstructs the same kind.
#[tokio::test]
async fn wfc_state_serialization_failure_persists_fail_and_replays_without_rerun() {
    let check_runs = Arc::new(AtomicU32::new(0));
    let check_runs_h = Arc::clone(&check_runs);
    let kinds = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let kinds_h = Arc::clone(&kinds);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let check_runs = Arc::clone(&check_runs_h);
                let kinds = Arc::clone(&kinds_h);
                async move {
                    let wfc_result = ctx
                        .wait_for_condition(
                            move |_ctx, state: u32| {
                                let check_runs = Arc::clone(&check_runs);
                                async move {
                                    check_runs.fetch_add(1, Ordering::SeqCst);
                                    Ok(state + 1)
                                }
                            },
                            0_u32,
                        )
                        .name("bad-wfc")
                        .serdes(FailSerialize)
                        .await;
                    let err = wfc_result.expect_err("state serialization must fail");
                    let label = match err.kind() {
                        durable::OperationErrorKind::WaitForCondition(wfc_err) => {
                            format!("{:?}", wfc_err.kind())
                        }
                        other => format!("non-wfc: {other}"),
                    };
                    kinds.lock().expect("test mutex").push(label);

                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, durable::BoxError>("recovered".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(
        check_runs.load(Ordering::SeqCst),
        1,
        "the check runs exactly once — replay yields the recorded FAIL"
    );
    let kinds = kinds.lock().expect("test mutex");
    assert!(kinds.len() >= 2, "live + replayed observations: {kinds:?}");
    assert!(
        kinds.iter().all(|k| k == "SerializationFailed"),
        "live and replayed kinds must both be SerializationFailed: {kinds:?}"
    );

    let wfc_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("bad-wfc"))
        .expect("the wfc operation record exists");
    assert!(wfc_op.failed(), "got {}", wfc_op.status());
    assert_eq!(wfc_op.error_type(), Some(SERIALIZATION_FAILED));
}

/// Review finding 3, non-retryable half: a buffered outcome write whose
/// contributor was dropped, the shape of a lost `race`/`select_ok` branch
/// or any dropped `DurableFuture`, is rejected non-retryably. The SDK
/// must write a terminal FAIL for the affected operation and fail the
/// execution, instead of reporting completion while the operation's
/// record claims less than what executed.
#[tokio::test]
async fn nonretryable_flush_failure_terminalizes_dropped_contributor_and_fails_execution() {
    let body_runs = Arc::new(AtomicU32::new(0));
    let body_runs_h = Arc::clone(&body_runs);

    let result = LocalRunner::new()
        .checkpoint_batching()
        .fail_checkpoints_after(1, 1)
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs_h);
                async move {
                    let runs_at_entry = body_runs.load(Ordering::SeqCst);
                    let body_runs_step = Arc::clone(&body_runs);
                    // Start the step eagerly, exactly as a `race` starts its
                    // branches; the returned DurableFuture is dropped below
                    // without being awaited, exactly as a `race` drops its
                    // losers.
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

                    // Wait for the body to have run, then give its SUCCEED
                    // write time to join the coalescing buffer before the
                    // contributor is dropped.
                    while body_runs.load(Ordering::SeqCst) == runs_at_entry {
                        tokio::time::sleep(Duration::from_millis(2)).await;
                    }
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    drop(orphan);

                    Ok::<_, durable::BoxError>("handler-done".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(body_runs.load(Ordering::SeqCst), 1, "one body execution");
    assert_eq!(
        result.invocation_count(),
        1,
        "a permanent rejection must not re-invoke"
    );
    assert!(
        result.is_failure(),
        "the execution must fail — the handler's completion must not win \
         while the orphaned operation's record claims less than executed"
    );
    assert_eq!(result.error_type(), Some(CHECKPOINT_FAILED));

    // The terminal FAIL persisted for the orphaned operation.
    let orphan_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("orphan"))
        .expect("the orphaned step's operation record exists");
    assert!(
        orphan_op.failed(),
        "a terminal FAIL is recorded for the operation whose buffered \
         outcome was rejected; got {}",
        orphan_op.status()
    );
    assert_eq!(orphan_op.error_type(), Some(CHECKPOINT_FAILED));
}

/// Review finding 3, retryable half: a buffered outcome write whose
/// contributor was dropped fails retryably. The invocation fails with no
/// further writes, even though the handler completed, and the
/// re-invocation converges under the documented `AtLeastOncePerRetry`
/// interruption contract.
#[tokio::test]
async fn retryable_flush_failure_fails_invocation_and_reinvocation_converges() {
    let body_runs = Arc::new(AtomicU32::new(0));
    let body_runs_h = Arc::clone(&body_runs);

    let result = LocalRunner::new()
        .checkpoint_batching()
        .fail_checkpoints_after_retryable(1, 1)
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs_h);
                async move {
                    let runs_at_entry = body_runs.load(Ordering::SeqCst);
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

                    while body_runs.load(Ordering::SeqCst) == runs_at_entry {
                        tokio::time::sleep(Duration::from_millis(2)).await;
                    }
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    drop(orphan);

                    Ok::<_, durable::BoxError>("handler-done".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_success(),
        "the execution converges once the channel recovers: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&"handler-done".to_owned()));
    assert_eq!(
        result.invocation_count(),
        2,
        "the exhausted transient failure costs exactly one invocation — \
         reporting SUCCEEDED would have claimed a record the service never \
         received"
    );
    assert_eq!(
        body_runs.load(Ordering::SeqCst),
        2,
        "the orphaned operation re-runs on the re-invocation \
         (AtLeastOncePerRetry): nothing further was written for it"
    );
}

/// A checkpoint response that did not include a checkpoint token must
/// stop checkpointing and report `PENDING`, even when it is discovered only
/// AFTER the handler itself already returned a value: the handler's own
/// completed outcome must never be reported once its outcome write was
/// withdrawn.
///
/// `withdraw_checkpoints_after(1, 1)` lets the step's START write through
/// and answers its SUCCEED write (the batched outcome, flushed at
/// end-of-invocation) with a token-less response.
#[tokio::test]
async fn late_token_withdrawal_after_handler_returns_reports_pending() {
    let body_runs = Arc::new(AtomicU32::new(0));
    let body_runs_h = Arc::clone(&body_runs);

    let result = LocalRunner::new()
        .checkpoint_batching()
        .withdraw_checkpoints_after(1, 1)
        .run(
            move |_event: serde_json::Value, ctx: durable::DurableContext| {
                let body_runs = Arc::clone(&body_runs_h);
                async move {
                    let runs_at_entry = body_runs.load(Ordering::SeqCst);
                    let body_runs_step = Arc::clone(&body_runs);
                    // Start the step eagerly and never await it, exactly as
                    // a `race` starts its branches; the handler returns its
                    // own result below while this background step's
                    // outcome write is still in flight.
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

                    // Wait for the body to have run, then give its
                    // SUCCEED write time to join the coalescing buffer
                    // before the handler returns its own value.
                    //
                    // The 30ms sleep below is a synchronization assumption,
                    // not a proof: it assumes the orphan step's SUCCEED
                    // write reaches the coalescing buffer within that
                    // window. `LocalRunner`/`InMemoryExecutionClient` expose
                    // no deterministic hook for pending-buffer occupancy
                    // (that state is crate-private, `#[cfg(test)]`-only, and
                    // not reachable from this integration test's crate), so
                    // there is nothing stabler to poll here today. Under
                    // contention this can under-wait and let the test pass
                    // without exercising the intended race; it cannot
                    // false-fail (a withdrawal injected before the write
                    // joins the buffer is still a withdrawal).
                    while body_runs.load(Ordering::SeqCst) == runs_at_entry {
                        tokio::time::sleep(Duration::from_millis(2)).await;
                    }
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    drop(orphan);

                    Ok::<_, durable::BoxError>("handler-done".to_owned())
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_suspended(),
        "a token-less checkpoint response must stop checkpointing this invocation \
         and report PENDING instead of the handler's own completed outcome: \
         {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(
        result.output(),
        None,
        "the handler's own return value must never be reported once its \
         checkpointing was withdrawn"
    );
    assert!(
        !result.is_failure(),
        "a withdrawn token is not a failure of either the invocation or the \
         execution"
    );
    assert_eq!(
        result.invocation_count(),
        1,
        "the withdrawn invocation reports PENDING with nothing further to \
         advance, so the runner must not keep re-invoking it"
    );

    let orphan_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("orphan"))
        .expect("the orphaned step's operation record exists");
    assert_eq!(
        orphan_op.status(),
        "Started",
        "the orphan's outcome write was withdrawn, so it must remain in its \
         replay-safe non-terminal state (not Succeeded, and not any other \
         terminal state either, which would equally prevent replay); got {}",
        orphan_op.status()
    );
}

/// `LocalRunner::session` pause/resume (experimental): a checkpoint
/// response without a token means the invocation may checkpoint no
/// further, so pausing before the first checkpoint call suspends the
/// invocation with PENDING at that very first attempt. The accepted
/// call's updates are still recorded; everything after it is never even
/// attempted, so it replays once the session resumes.
#[tokio::test]
async fn session_pause_before_run_suspends_and_resume_replays_the_execution() {
    let a_runs = Arc::new(AtomicU32::new(0));
    let b_runs = Arc::new(AtomicU32::new(0));

    let session = LocalRunner::new().session();
    assert!(!session.is_paused());
    session.pause();
    assert!(session.is_paused());

    let handler = {
        let a_runs = Arc::clone(&a_runs);
        let b_runs = Arc::clone(&b_runs);
        move |_event: serde_json::Value, ctx: durable::DurableContext| {
            let a_runs = Arc::clone(&a_runs);
            let b_runs = Arc::clone(&b_runs);
            async move {
                let a = ctx
                    .step(move |_| {
                        let a_runs = Arc::clone(&a_runs);
                        async move {
                            a_runs.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, durable::BoxError>(1_i32)
                        }
                    })
                    .name("a")
                    .await?;
                let b = ctx
                    .step(move |_| {
                        let b_runs = Arc::clone(&b_runs);
                        async move {
                            b_runs.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, durable::BoxError>(2_i32)
                        }
                    })
                    .name("b")
                    .await?;
                Ok::<_, durable::BoxError>(a + b)
            }
        }
    };

    let result = session.run(handler.clone(), serde_json::Value::Null).await;

    assert!(
        result.is_suspended(),
        "a checkpoint response withheld before any work was attempted \
         must suspend the invocation with PENDING, not fail it: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), None);
    assert!(!result.is_failure());
    assert_eq!(
        a_runs.load(Ordering::SeqCst),
        0,
        "the paused response arrived before step a's body could run, so \
         the body never ran this invocation"
    );
    assert_eq!(
        b_runs.load(Ordering::SeqCst),
        0,
        "step b's write was never attempted once the invocation suspended \
         on step a's withheld token"
    );
    assert!(
        !result.operations().is_empty(),
        "the accepted checkpoint's updates must be recorded even though \
         its response withheld the token"
    );
    assert!(
        result.operations().iter().any(|op| op.name() == Some("a")),
        "step a's accepted START write survives the pause"
    );
    assert!(
        !result.operations().iter().any(|op| op.name() == Some("b")),
        "step b was never attempted, so it has no recorded operation"
    );

    session.resume();
    assert!(!session.is_paused());
    session.resume();
    assert!(
        !session.is_paused(),
        "resuming an already-resumed session has no effect"
    );

    let result = session.run(handler, serde_json::Value::Null).await;

    assert!(
        result.is_success(),
        "resuming lets the next run drive the execution to completion: \
         {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&3));
    assert_eq!(
        a_runs.load(Ordering::SeqCst),
        1,
        "step a's accepted START write was not re-sent; its body ran \
         exactly once, on the resumed invocation"
    );
    assert_eq!(
        b_runs.load(Ordering::SeqCst),
        1,
        "step b runs exactly once, on the resumed invocation"
    );
}

/// Pausing an already-paused session, and `is_paused` immediately after
/// each call, with no execution in progress: the latch is idempotent and
/// observable on its own, independent of any `run` call.
#[test]
fn session_pause_resume_are_idempotent_and_is_paused_reflects_state() {
    let session = LocalRunner::new().session();
    assert!(!session.is_paused());

    session.pause();
    assert!(session.is_paused());
    session.pause();
    assert!(session.is_paused(), "pausing twice stays paused");

    session.resume();
    assert!(!session.is_paused());
    session.resume();
    assert!(!session.is_paused(), "resuming twice stays resumed");
}

/// Pausing while a step's completion write is still sitting in the
/// `checkpoint_delay` coalescing buffer: the write is accepted once it
/// flushes, and its outcome is recorded in the backend's state, but the
/// response carries no token, so the current invocation still suspends.
/// The resumed invocation reuses the recorded outcome rather than
/// rerunning the step's body.
///
/// Synchronization note: the watcher's 10ms sleep only has to win a race
/// against the buffer's OWN 150ms flush timer, not against an immediate
/// write, so this is far less timing-sensitive than racing a live
/// checkpoint call directly. It can still in principle under-wait and let
/// the test pass without exercising the intended race; it cannot
/// false-fail (a pause that lands before the buffer flushes is still a
/// pause).
#[tokio::test]
async fn session_pause_mid_flight_keeps_accepted_write_and_replays_the_rest() {
    let body_runs = Arc::new(AtomicU32::new(0));
    let body_runs_h = Arc::clone(&body_runs);

    let session = Arc::new(
        LocalRunner::new()
            .checkpoint_delay(Duration::from_millis(150))
            .session(),
    );
    let session_watcher = Arc::clone(&session);
    let pause_after_step_body_runs = async move {
        while body_runs_h.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
        session_watcher.pause();
    };

    let run_until_pause = session.run(
        step_outcome_handler(Arc::clone(&body_runs)),
        serde_json::Value::Null,
    );
    let (result, ()) = tokio::join!(run_until_pause, pause_after_step_body_runs);

    assert!(
        result.is_suspended(),
        "the step's SUCCEED write was accepted once the coalescing buffer \
         flushed, but the paused response carried no token, so the \
         invocation must suspend with PENDING, not report the step's \
         own outcome: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), None);
    assert_eq!(
        body_runs.load(Ordering::SeqCst),
        1,
        "the step's body ran exactly once before the pause took effect"
    );
    let step_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("the-step"))
        .expect(
            "the step's accepted updates are recorded even though the \
             response carried no token",
        );
    assert_eq!(
        step_op.status(),
        "Succeeded",
        "a paused checkpoint still applies the accepted update to the \
         backend's state, it just answers without a token: {}",
        step_op.status()
    );

    session.resume();

    let result = session
        .run(
            step_outcome_handler(Arc::clone(&body_runs)),
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.is_success(),
        "resuming the session lets the next run drive the suspended \
         execution to completion: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&"step-outcome".to_owned()));
    assert_eq!(
        body_runs.load(Ordering::SeqCst),
        1,
        "the step's SUCCEED update was already applied to the backend's \
         state while paused, so the resumed replay reuses the recorded \
         outcome instead of rerunning the body"
    );
}
