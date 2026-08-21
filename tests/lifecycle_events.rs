//! Lifecycle-event contract tests (issue #18).
//!
//! Drives handlers through [`LocalRunner`] with a capturing JSON subscriber
//! and asserts the documented span/event names and fields from
//! [`durable::observability`] appear in the emitted output: a step (live and
//! replayed), a wait, a retrying step, and the execution-level
//! started/resumed/suspended events.

#![cfg(feature = "test-util")]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::observability::{TARGET, event_names, field_names, span_names};
use durable::test_util::LocalRunner;
use tracing_subscriber::layer::SubscriberExt;

/// A `MakeWriter` that captures subscriber output in a shared buffer.
#[derive(Clone)]
struct CaptureWriter(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut inner) = self.0.lock() {
            inner.extend_from_slice(buf);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
    type Writer = CaptureWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Installs a JSON capture subscriber (all levels, flattened fields) and
/// returns the shared buffer.
fn capture_subscriber() -> (Arc<Mutex<Vec<u8>>>, tracing::subscriber::DefaultGuard) {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .flatten_event(true)
        .with_span_list(true)
        .with_current_span(true)
        .with_writer(CaptureWriter(Arc::clone(&buffer)));
    let subscriber = tracing_subscriber::registry().with(fmt_layer);
    let guard = tracing::subscriber::set_default(subscriber);
    (buffer, guard)
}

/// Renders the captured buffer as a string.
fn captured(buffer: &Arc<Mutex<Vec<u8>>>) -> String {
    buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    )
}

/// Parses the captured output into JSON lines whose `message` equals the
/// given event name and whose `target` is the lifecycle target.
fn lifecycle_events(output: &str, event_name: &str) -> Vec<serde_json::Value> {
    output
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| {
            v.get("message").and_then(serde_json::Value::as_str) == Some(event_name)
                && v.get("target").and_then(serde_json::Value::as_str) == Some(TARGET)
        })
        .collect()
}

/// Selects the events of the given name that carry a specific field value.
fn events_with_field<'a>(
    events: &'a [serde_json::Value],
    field: &str,
    value: &str,
) -> Vec<&'a serde_json::Value> {
    events
        .iter()
        .filter(|v| v.get(field).and_then(serde_json::Value::as_str) == Some(value))
        .collect()
}

/// A step (live + replayed) and a wait: the documented events fire with the
/// documented fields, exactly once each across the whole execution.
#[tokio::test]
#[allow(clippy::too_many_lines)] // reason: one end-to-end walkthrough of the whole contract
async fn step_and_wait_lifecycle_events_follow_the_contract() {
    let (buffer, _guard) = capture_subscriber();

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                let greeting = ctx
                    .step(|_| async {
                        tracing::info!("inside the step body");
                        Ok("hello".to_owned())
                    })
                    .name("greet")
                    .await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                Ok::<_, durable::BoxError>(greeting)
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&"hello".to_owned()));
    assert!(
        result.invocation_count() >= 2,
        "the wait must suspend and resume, got {} invocation(s)",
        result.invocation_count()
    );

    let output = captured(&buffer);

    // operation_started: once for the step, once for the wait — replay on
    // the resumed invocation must NOT re-emit them.
    let started = lifecycle_events(&output, event_names::OPERATION_STARTED);
    let step_started = events_with_field(&started, field_names::OPERATION_TYPE, "Step");
    let wait_started = events_with_field(&started, field_names::OPERATION_TYPE, "Wait");
    assert_eq!(
        step_started.len(),
        1,
        "exactly one Step operation_started. Got: {output}"
    );
    assert_eq!(
        wait_started.len(),
        1,
        "exactly one Wait operation_started. Got: {output}"
    );

    // The step start carries the full documented field set.
    let step_start = step_started
        .first()
        .copied()
        .unwrap_or(&serde_json::Value::Null);
    assert_eq!(
        step_start
            .get(field_names::OPERATION_NAME)
            .and_then(serde_json::Value::as_str),
        Some("greet"),
        "operationName must carry the .name() label. Got: {step_start}"
    );
    assert_eq!(
        step_start
            .get(field_names::ATTEMPT)
            .and_then(serde_json::Value::as_u64),
        Some(1),
        "first attempt is 1-based. Got: {step_start}"
    );
    assert_eq!(
        step_start
            .get(field_names::IS_REPLAY)
            .and_then(serde_json::Value::as_bool),
        Some(false),
        "a live start is not replay. Got: {step_start}"
    );
    let arn = step_start
        .get(field_names::EXECUTION_ARN)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    assert!(
        !arn.is_empty(),
        "executionArn must be present. Got: {step_start}"
    );
    let op_id = step_start
        .get(field_names::OPERATION_ID)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    assert_eq!(
        op_id.len(),
        64,
        "operationId is the SHA-256 hex wire ID. Got: {step_start}"
    );
    assert!(
        step_start
            .get(field_names::REQUEST_ID)
            .and_then(serde_json::Value::as_str)
            .is_some(),
        "requestId must be present (LocalRunner synthesizes an empty one). Got: {step_start}"
    );

    // operation_succeeded: exactly once, for the step (the wait's success
    // is a backend timer, not an SDK checkpoint).
    let succeeded = lifecycle_events(&output, event_names::OPERATION_SUCCEEDED);
    let step_succeeded = events_with_field(&succeeded, field_names::OPERATION_TYPE, "Step");
    assert_eq!(
        step_succeeded.len(),
        1,
        "exactly one Step operation_succeeded. Got: {output}"
    );

    // operation_replayed: once for the step and once for the wait, both on
    // the resumed invocation, with isReplay = true.
    let replayed = lifecycle_events(&output, event_names::OPERATION_REPLAYED);
    let step_replayed = events_with_field(&replayed, field_names::OPERATION_TYPE, "Step");
    let wait_replayed = events_with_field(&replayed, field_names::OPERATION_TYPE, "Wait");
    assert_eq!(
        step_replayed.len(),
        1,
        "exactly one Step operation_replayed. Got: {output}"
    );
    assert_eq!(
        wait_replayed.len(),
        1,
        "exactly one Wait operation_replayed. Got: {output}"
    );
    let step_replay = step_replayed
        .first()
        .copied()
        .unwrap_or(&serde_json::Value::Null);
    assert_eq!(
        step_replay
            .get(field_names::IS_REPLAY)
            .and_then(serde_json::Value::as_bool),
        Some(true),
        "a replay hit carries isReplay = true. Got: {step_replay}"
    );
    assert_eq!(
        step_replay
            .get(field_names::OPERATION_NAME)
            .and_then(serde_json::Value::as_str),
        Some("greet"),
        "the replayed step keeps its name. Got: {step_replay}"
    );
    assert_eq!(
        step_replay
            .get(field_names::OPERATION_ID)
            .and_then(serde_json::Value::as_str),
        Some(op_id),
        "replay reports the same wire operation ID as the live start"
    );

    // Execution-level events: exactly one started, at least one resumed,
    // exactly one suspended (only the wait suspends this handler).
    assert_eq!(
        lifecycle_events(&output, event_names::EXECUTION_STARTED).len(),
        1,
        "exactly one execution_started. Got: {output}"
    );
    assert!(
        !lifecycle_events(&output, event_names::EXECUTION_RESUMED).is_empty(),
        "the resumed invocation emits execution_resumed. Got: {output}"
    );
    assert_eq!(
        lifecycle_events(&output, event_names::EXECUTION_SUSPENDED).len(),
        1,
        "the suspending invocation emits execution_suspended. Got: {output}"
    );

    // Span-name contract: the live step body runs inside a
    // `durable_operation` span nested in the handler's `durable_execution`
    // span — visible on the span scope of the event logged in the body.
    let in_step_line = output
        .lines()
        .find(|line| line.contains("inside the step body"))
        .unwrap_or_default();
    assert!(
        in_step_line.contains(span_names::OPERATION),
        "the in-step event must sit inside the durable_operation span. Got: {in_step_line}"
    );
    assert!(
        in_step_line.contains(span_names::EXECUTION),
        "the in-step event must sit inside the durable_execution span. Got: {in_step_line}"
    );
}

/// A retrying step emits `operation_retry_scheduled` per scheduled retry —
/// with the failing attempt number, the delay, and the error — then
/// `operation_succeeded` on the attempt that passes.
#[tokio::test]
async fn retrying_step_emits_retry_scheduled_events() {
    let (buffer, _guard) = capture_subscriber();

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                let value = ctx
                    .step(|step_ctx| async move {
                        let attempt = step_ctx.attempt();
                        if attempt < 3 {
                            return Err(format!("transient failure {attempt}").into());
                        }
                        Ok(attempt)
                    })
                    .name("flaky")
                    .retry_strategy(|_err, attempt| {
                        if attempt >= 3 {
                            durable::RetryDecision::Stop
                        } else {
                            durable::RetryDecision::Retry {
                                delay: Duration::from_secs(1),
                            }
                        }
                    })
                    .await?;
                Ok::<_, durable::BoxError>(value)
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&3));

    let output = captured(&buffer);

    let retries = lifecycle_events(&output, event_names::OPERATION_RETRY_SCHEDULED);
    assert_eq!(
        retries.len(),
        2,
        "attempts 1 and 2 each schedule a retry. Got: {output}"
    );
    let attempts: Vec<u64> = retries
        .iter()
        .filter_map(|v| {
            v.get(field_names::ATTEMPT)
                .and_then(serde_json::Value::as_u64)
        })
        .collect();
    assert_eq!(
        attempts,
        vec![1, 2],
        "retry events carry the attempt that failed. Got: {output}"
    );
    for retry in &retries {
        assert_eq!(
            retry
                .get(field_names::DELAY_SECONDS)
                .and_then(serde_json::Value::as_u64),
            Some(1),
            "retry events carry the recorded delay. Got: {retry}"
        );
        assert!(
            retry
                .get(field_names::ERROR)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|e| e.contains("transient failure")),
            "retry events carry the error message. Got: {retry}"
        );
    }

    // Each attempt records a fresh start; only the last one succeeds.
    let started = lifecycle_events(&output, event_names::OPERATION_STARTED);
    let step_started = events_with_field(&started, field_names::OPERATION_TYPE, "Step");
    assert_eq!(
        step_started.len(),
        3,
        "each of the three attempts records a start. Got: {output}"
    );
    let succeeded = lifecycle_events(&output, event_names::OPERATION_SUCCEEDED);
    assert_eq!(
        events_with_field(&succeeded, field_names::OPERATION_TYPE, "Step").len(),
        1,
        "only the final attempt succeeds. Got: {output}"
    );
    assert!(
        lifecycle_events(&output, event_names::OPERATION_FAILED).is_empty(),
        "no permanent failure is recorded. Got: {output}"
    );
}

/// A step that exhausts its retries emits `operation_failed` with the error.
#[tokio::test]
async fn failing_step_emits_operation_failed() {
    let (buffer, _guard) = capture_subscriber();

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                ctx.step(|_| async { Err::<u32, durable::BoxError>("boom".into()) })
                    .name("doomed")
                    .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
                    .await
                    .map_err(durable::BoxError::from)
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(result.output().is_none(), "the handler must fail");

    let output = captured(&buffer);
    let failed = lifecycle_events(&output, event_names::OPERATION_FAILED);
    assert_eq!(
        failed.len(),
        1,
        "exactly one operation_failed. Got: {output}"
    );
    let event = failed.first().unwrap_or(&serde_json::Value::Null);
    assert_eq!(
        event
            .get(field_names::OPERATION_NAME)
            .and_then(serde_json::Value::as_str),
        Some("doomed"),
        "the failure carries the operation name. Got: {event}"
    );
    assert!(
        event
            .get(field_names::ERROR)
            .and_then(serde_json::Value::as_str)
            .is_some_and(|e| e.contains("boom")),
        "the failure carries the error message. Got: {event}"
    );
    assert!(
        lifecycle_events(&output, event_names::OPERATION_RETRY_SCHEDULED).is_empty(),
        "Stop schedules no retry. Got: {output}"
    );
}

/// The execution-level events are span events of the handler's
/// `durable_execution` span: `execution_started` / `execution_resumed` are
/// emitted while the span is entered before the handler runs, and
/// `execution_suspended` while it is entered after the handler future is
/// dropped — which is what lets the documented OpenTelemetry bridge export
/// them on the execution span.
#[tokio::test]
async fn execution_events_are_span_events_of_the_execution_span() {
    let (buffer, _guard) = capture_subscriber();

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                Ok::<_, durable::BoxError>(7_u32)
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&7));
    let output = captured(&buffer);

    for event_name in [
        event_names::EXECUTION_STARTED,
        event_names::EXECUTION_RESUMED,
        event_names::EXECUTION_SUSPENDED,
    ] {
        let events = lifecycle_events(&output, event_name);
        assert!(
            !events.is_empty(),
            "{event_name} must be emitted. Got: {output}"
        );
        for event in &events {
            let current_span = event
                .get("span")
                .and_then(|span| span.get("name"))
                .and_then(serde_json::Value::as_str);
            assert_eq!(
                current_span,
                Some(span_names::EXECUTION),
                "{event_name} must be a span event of the durable_execution span. Got: {event}"
            );
        }
    }
}

/// `operation_replayed` covers the terminal replay paths beyond step and
/// wait: a child context, a `wait_for_condition`, and a callback each emit
/// it — with the documented identity fields — when a resumed invocation
/// returns their recorded outcome without re-running them.
#[tokio::test]
async fn replayed_events_cover_child_condition_and_callback() {
    let (buffer, _guard) = capture_subscriber();

    let result = LocalRunner::new()
        .callback_success(&"yes".to_owned())
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                let doubled = ctx
                    .run_in_child_context(|child| async move {
                        let base = child.step(|_| async { Ok(21_u32) }).name("base").await?;
                        Ok(base * 2)
                    })
                    .name("sub")
                    .await?;

                let polls = ctx
                    .wait_for_condition(|_ctx, state: i32| async move { Ok(state + 1) }, 0)
                    .wait_strategy_fn(|state: i32, _attempt| {
                        if state >= 1 {
                            durable::builders::wait_for_condition::WaitDecision::complete()
                        } else {
                            durable::builders::wait_for_condition::WaitDecision::continue_with(
                                Duration::from_secs(1),
                            )
                        }
                    })
                    .name("poll")
                    .await?;

                let approval: String = ctx
                    .wait_for_callback::<String, _, _>(
                        |_step_ctx, _callback_id| async move { Ok(()) },
                    )
                    .name("approve")
                    .await?;

                // A final suspension forces one more resume, on which every
                // operation above — including the wait_for_callback context,
                // which only records Succeed on the invocation that consumed
                // the callback — replays from its terminal record.
                ctx.wait(Duration::from_secs(1)).name("final-pause").await?;

                Ok::<_, durable::BoxError>(format!("{doubled}-{polls}-{approval}"))
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&"42-1-yes".to_owned()));
    assert!(
        result.invocation_count() >= 2,
        "the callback must suspend and resume, got {} invocation(s)",
        result.invocation_count()
    );

    let output = captured(&buffer);
    let replayed = lifecycle_events(&output, event_names::OPERATION_REPLAYED);

    // (operationSubType, operationName) for each replay path under test.
    for (sub_type, name) in [
        ("RunInChildContext", "sub"),
        ("WaitForCondition", "poll"),
        ("WaitForCallback", "approve"),
    ] {
        let matching: Vec<&serde_json::Value> =
            events_with_field(&replayed, field_names::OPERATION_SUB_TYPE, sub_type)
                .into_iter()
                .filter(|v| {
                    v.get(field_names::OPERATION_NAME)
                        .and_then(serde_json::Value::as_str)
                        == Some(name)
                })
                .collect();
        assert!(
            !matching.is_empty(),
            "operation_replayed must fire for {sub_type} '{name}'. Got: {output}"
        );
        for event in &matching {
            assert_eq!(
                event
                    .get(field_names::IS_REPLAY)
                    .and_then(serde_json::Value::as_bool),
                Some(true),
                "a replay hit carries isReplay = true. Got: {event}"
            );
            assert!(
                event
                    .get(field_names::EXECUTION_ARN)
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|arn| !arn.is_empty()),
                "executionArn must be present. Got: {event}"
            );
            assert!(
                event
                    .get(field_names::ATTEMPT)
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|attempt| attempt >= 1),
                "attempt is 1-based. Got: {event}"
            );
        }
    }
}

/// A rejected checkpoint emits no record-transition lifecycle event: the
/// events claim a transition was recorded, so they fire only after the
/// checkpoint that persists it succeeds. The step whose START write is
/// rejected unwinds the handler through the unrecoverable path (issue
/// #43) — the execution fails, and no `operation_started` is emitted.
#[tokio::test]
async fn rejected_checkpoint_emits_no_record_transition_event() {
    let (buffer, _guard) = capture_subscriber();

    let result = LocalRunner::new()
        .fail_next_checkpoints(1)
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                ctx.step(|_| async { Ok(1_u32) })
                    .name("rejected-start")
                    .await
                    .map_err(durable::BoxError::from)
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.output().is_none(),
        "the rejected START checkpoint must fail the step and the handler"
    );

    let output = captured(&buffer);
    for event_name in [
        event_names::OPERATION_STARTED,
        event_names::OPERATION_SUCCEEDED,
        event_names::OPERATION_FAILED,
        event_names::OPERATION_RETRY_SCHEDULED,
    ] {
        assert!(
            lifecycle_events(&output, event_name).is_empty(),
            "a rejected checkpoint must emit no {event_name}. Got: {output}"
        );
    }
}

/// A serdes whose serialize side works (so the live step records normally)
/// and whose deserialize side succeeds exactly once — covering the live
/// path's round-trip — then always fails, simulating a payload that decodes
/// during the original invocation but is corrupt at replay time.
#[derive(Debug)]
struct FailOnReplayDeserializeSerdes(Arc<std::sync::atomic::AtomicU32>);

impl durable::Serdes<u32> for FailOnReplayDeserializeSerdes {
    // reason: exercises the async-fn impl form user code writes
    #[allow(clippy::unused_async_trait_impl)]
    async fn serialize(
        &self,
        value: u32,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(serde_json::to_string(&value)?)
    }

    // reason: exercises the async-fn impl form user code writes
    #[allow(clippy::unused_async_trait_impl)]
    async fn deserialize(
        &self,
        wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<u32, durable::BoxError> {
        let call = self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call == 0 {
            // The live path's round-trip decode.
            Ok(serde_json::from_str(&wire)?)
        } else {
            Err("injected replay deserialize failure".into())
        }
    }
}

/// REGRESSION: `operation_replayed` claims a recorded terminal outcome was
/// returned, so it must be emitted only after the recorded payload is
/// successfully reconstructed. A step whose recorded result fails to
/// deserialize on replay surfaces an error and emits NO `operation_replayed`
/// — while its live-path events from the first invocation remain intact.
#[tokio::test]
async fn failed_replay_decode_emits_no_operation_replayed() {
    let (buffer, _guard) = capture_subscriber();
    let decode_calls = Arc::new(std::sync::atomic::AtomicU32::new(0));

    let result = LocalRunner::new()
        .run(
            {
                let decode_calls = Arc::clone(&decode_calls);
                move |_event: serde_json::Value, ctx: durable::DurableContext| {
                    let decode_calls = Arc::clone(&decode_calls);
                    async move {
                        let value = ctx
                            .step(|_| async { Ok(7_u32) })
                            .name("decays-on-replay")
                            .serdes(FailOnReplayDeserializeSerdes(decode_calls))
                            .await?;
                        // The wait suspends the execution, so the resume
                        // replays the step — and the replay decode fails.
                        ctx.wait(Duration::from_secs(1))
                            .name("force-resume")
                            .await?;
                        Ok::<_, durable::BoxError>(value)
                    }
                }
            },
            serde_json::Value::Null,
        )
        .await;

    assert!(
        result.output().is_none(),
        "the failed replay decode must fail the execution"
    );
    assert!(
        result.invocation_count() >= 2,
        "the wait must suspend so the step is replayed, got {} invocation(s)",
        result.invocation_count()
    );

    let output = captured(&buffer);

    // The live first attempt recorded normally.
    let started = lifecycle_events(&output, event_names::OPERATION_STARTED);
    assert!(
        !events_with_field(&started, field_names::OPERATION_NAME, "decays-on-replay").is_empty(),
        "the live step start must have been recorded and emitted. Got: {output}"
    );

    // But no replay event fired: decoding the recorded outcome failed, so
    // no recorded terminal outcome was actually returned.
    let replayed = lifecycle_events(&output, event_names::OPERATION_REPLAYED);
    assert!(
        events_with_field(&replayed, field_names::OPERATION_NAME, "decays-on-replay").is_empty(),
        "a failed replay decode must not emit operation_replayed. Got: {output}"
    );
}
