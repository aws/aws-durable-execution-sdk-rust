//! The history events and the invocation records of `TestResult`, through
//! the public API only.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::{Duration, SystemTime};

use aws_durable_execution_sdk as durable;
use durable::lambda_types::{Event, EventType};
use durable::test_util::{CallbackFailure, CallbackReply, LocalRunner};
use durable::{BoxError, DurableContext, RetryDecision};

/// The event types of `events`, in order.
fn types(events: &[Event]) -> Vec<EventType> {
    events
        .iter()
        .filter_map(|event| event.event_type().cloned())
        .collect()
}

/// The time of a service timestamp.
fn time(t: &durable::lambda_types::DateTime) -> SystemTime {
    SystemTime::try_from(*t).expect("a representable time")
}

/// The README quick-start handler: a step, a 2 s wait, then a step.
#[tokio::test]
async fn step_wait_step_event_sequence() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                let name = ctx
                    .step(|_| async { Ok("world".to_owned()) })
                    .name("fetch-name")
                    .await?;
                ctx.wait(Duration::from_secs(2)).name("cooldown").await?;
                let greeting = ctx
                    .step(move |_| async move { Ok(format!("hello, {name}")) })
                    .name("format")
                    .await?;
                Ok::<_, BoxError>(greeting)
            },
            (),
        )
        .await
        .expect("runner");

    let events = result.history_events();
    assert_eq!(
        types(events),
        [
            EventType::ExecutionStarted,
            EventType::StepStarted,
            EventType::StepSucceeded,
            EventType::WaitStarted,
            EventType::InvocationCompleted,
            EventType::WaitSucceeded,
            EventType::StepStarted,
            EventType::StepSucceeded,
            EventType::InvocationCompleted,
            EventType::ExecutionSucceeded,
        ]
    );
    let ids: Vec<i32> = events.iter().map(Event::event_id).collect();
    assert_eq!(ids, (1..=10).collect::<Vec<_>>());

    let of_type = |event_type: EventType| {
        events
            .iter()
            .find(|event| event.event_type() == Some(&event_type))
            .expect("the event is recorded")
    };
    let started = of_type(EventType::WaitStarted);
    let succeeded = of_type(EventType::WaitSucceeded);
    let started_at = time(started.event_timestamp().expect("a timestamp"));
    let succeeded_at = time(succeeded.event_timestamp().expect("a timestamp"));
    assert!(
        succeeded_at >= started_at + Duration::from_secs(2),
        "{succeeded_at:?} is less than 2 s after {started_at:?}"
    );
    let scheduled_end = started
        .wait_started_details()
        .expect("wait started details")
        .scheduled_end_timestamp();
    assert_eq!(time(scheduled_end), started_at + Duration::from_secs(2));
}

/// A step that fails once with a 1 s retry, then succeeds.
#[tokio::test]
async fn retried_step_records_failure_events() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.step(|step| async move {
                    if step.attempt() == 1 {
                        return Err::<u32, BoxError>("first attempt fails".into());
                    }
                    Ok(1)
                })
                .name("flaky")
                .retry_strategy(|_err, _attempt| RetryDecision::Retry {
                    delay: Duration::from_secs(1),
                })
                .await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let step = result.operation("flaky").expect("the step");
    assert_eq!(
        types(step.events()),
        [
            EventType::StepStarted,
            EventType::StepFailed,
            EventType::StepStarted,
            EventType::StepSucceeded,
        ]
    );

    let events = result.history_events();
    let failed_at = events
        .iter()
        .position(|event| event.event_type() == Some(&EventType::StepFailed))
        .expect("a StepFailed event");
    assert!(
        events
            .iter()
            .skip(failed_at)
            .any(|event| event.event_type() == Some(&EventType::InvocationCompleted)),
        "an InvocationCompleted follows the StepFailed"
    );

    let details = events
        .get(failed_at)
        .and_then(Event::step_failed_details)
        .expect("step failed details");
    let message = details
        .error()
        .and_then(|error| error.payload())
        .and_then(|payload| payload.error_message());
    assert_eq!(message, Some("first attempt fails"));
    assert_eq!(
        details
            .retry_details()
            .and_then(durable::lambda_types::RetryDetails::next_attempt_delay_seconds),
        Some(1)
    );
}

/// A `wait_for_condition` named `poll` whose check adds 1 to a state that
/// starts at 0, and whose strategy continues with a 1 s delay while the
/// state is below 2.
async fn poll_handler((): (), ctx: DurableContext) -> Result<i32, BoxError> {
    use durable::builders::wait_for_condition::WaitDecision;
    Ok(ctx
        .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
        .wait_strategy_fn(|state: i32, _attempt| {
            if state < 2 {
                WaitDecision::continue_with(Duration::from_secs(1))
            } else {
                WaitDecision::complete()
            }
        })
        .name("poll")
        .await?)
}

/// A check that continues records `StepSucceeded` with the new state and
/// the delay, as the service does, and the next attempt time is the start
/// of `poll` plus that delay.
#[tokio::test]
async fn condition_continue_records_step_succeeded() {
    let result = LocalRunner::new()
        .run(poll_handler, ())
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&2));
    assert_eq!(
        types(result.history_events()),
        [
            EventType::ExecutionStarted,
            EventType::StepStarted,
            EventType::StepSucceeded,
            EventType::InvocationCompleted,
            EventType::StepStarted,
            EventType::StepSucceeded,
            EventType::InvocationCompleted,
            EventType::ExecutionSucceeded,
        ]
    );

    let poll = result.operation("poll").expect("the condition");
    let succeeded: Vec<_> = poll
        .events()
        .iter()
        .filter_map(Event::step_succeeded_details)
        .map(|details| {
            let retry = details.retry_details().expect("retry details");
            (
                details
                    .result()
                    .and_then(|r| r.payload())
                    .map(ToOwned::to_owned),
                retry.current_attempt(),
                retry.next_attempt_delay_seconds(),
            )
        })
        .collect();
    assert_eq!(
        succeeded,
        [
            (Some("1".to_owned()), 1, Some(1)),
            (Some("2".to_owned()), 2, None),
        ]
    );

    let step = match poll.details() {
        durable::test_util::TestOperationDetails::Step(step) => Some(step),
        _ => None,
    }
    .expect("a step operation");
    let start = poll.start_time().expect("a start time");
    assert_eq!(
        step.next_attempt_time(),
        Some(start + Duration::from_secs(1))
    );
}

/// A `wait_for_callback` whose callback receives a failure.
#[tokio::test]
async fn callback_failure_event_carries_error() {
    let result = LocalRunner::new()
        .callback(CallbackReply::failure(CallbackFailure::new(
            "Declined", "no",
        )))
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.wait_for_callback::<String>(|_step, _id| async move { Ok(()) })
                    .name("approval")
                    .await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let events = result.history_events();
    let failed_at = events
        .iter()
        .position(|event| event.event_type() == Some(&EventType::CallbackFailed))
        .expect("a CallbackFailed event");
    let error_type = events
        .get(failed_at)
        .and_then(Event::callback_failed_details)
        .and_then(|details| details.error())
        .and_then(|error| error.payload())
        .and_then(|payload| payload.error_type());
    assert_eq!(error_type, Some("Declined"));
    assert!(
        events
            .iter()
            .skip(failed_at)
            .any(|event| event.event_type() == Some(&EventType::ContextFailed)),
        "a ContextFailed follows the CallbackFailed"
    );
}

/// The events of a step inside a child context carry the context's ID as
/// their parent ID.
#[tokio::test]
async fn child_events_carry_parent_id() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.run_in_child_context(|child| async move {
                    child.step(|_| async { Ok(1_u32) }).name("inner").await?;
                    Ok(())
                })
                .name("outer")
                .await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let outer = result.operation("outer").expect("the child context");
    let inner = result.operation("inner").expect("the step");
    assert_ne!(inner.events(), &[]);
    for event in inner.events() {
        assert_eq!(event.parent_id(), Some(outer.id()));
    }
}

/// The invocation records of a run with one wait.
#[tokio::test]
async fn invocations_are_public() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.wait(Duration::from_secs(1)).await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let invocations = result.invocations();
    let first = invocations.first().expect("a first invocation");
    let second = invocations.get(1).expect("a second invocation");
    assert_eq!(first.request_id(), Some("local-invocation-1"));
    assert_eq!(
        second.start_time(),
        first.start_time().map(|t| t + Duration::from_secs(1))
    );
}
