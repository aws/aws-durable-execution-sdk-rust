//! The plugin end to end, through the SDK's `LocalRunner`.
//!
//! A recording exporter keeps every record the plugin exports. The tests
//! assert the records' content and order, not their times: local times
//! are the run start plus virtual seconds.

#![expect(clippy::expect_used, clippy::indexing_slicing)] // reason: test assertions

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use aws_durable_execution_sdk_telemetry_exporter::{
    ContentConfig, ContentSetting, DurableExecutionTelemetry, DurableExecutionTelemetryBuilder,
    EmitMode, ExportFuture, OperationDetail, OperationOverride, OperationsFormat,
    TelemetryExporter, TelemetryRecord, render_value,
};
use durable::test_util::LocalRunner;
use durable::{BoxError, DurableContext, Options};
use serde_json::{Value, json};

/// Keeps the `operationsByName` and `operations` forms of every record.
#[derive(Clone, Default)]
struct Recording {
    records: Arc<Mutex<Vec<Value>>>,
}

impl Recording {
    fn records(&self) -> Vec<Value> {
        self.records.lock().expect("lock").clone()
    }
}

impl TelemetryExporter for Recording {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        let value = render_value(record, OperationsFormat::Both).expect("render");
        self.records.lock().expect("lock").push(value);
        Box::pin(std::future::ready(Ok(())))
    }
}

/// Fails every export.
struct Failing;

impl TelemetryExporter for Failing {
    fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(std::future::ready(Err("export failed".into())))
    }
}

/// Takes one millisecond per export. Keeps each record's status and the
/// largest number of live tokio tasks that an export observed.
#[derive(Clone, Default)]
struct Slow {
    statuses: Arc<Mutex<Vec<Value>>>,
    peak_tasks: Arc<AtomicUsize>,
}

impl TelemetryExporter for Slow {
    fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
        Box::pin(async move {
            let alive = tokio::runtime::Handle::current()
                .metrics()
                .num_alive_tasks();
            self.peak_tasks.fetch_max(alive, Ordering::Relaxed);
            tokio::time::sleep(Duration::from_millis(1)).await;
            let value = render_value(record, OperationsFormat::ByName).expect("render");
            self.statuses
                .lock()
                .expect("lock")
                .push(value["status"].clone());
            Ok(())
        })
    }
}

fn runner(telemetry: DurableExecutionTelemetry) -> LocalRunner {
    LocalRunner::new().options(
        Options::builder()
            .plugin(telemetry)
            .build()
            .expect("valid options"),
    )
}

fn recorded(builder: DurableExecutionTelemetryBuilder) -> (DurableExecutionTelemetry, Recording) {
    let recording = Recording::default();
    let telemetry = builder.exporter(recording.clone()).build();
    (telemetry, recording)
}

async fn one_step(event: Value, ctx: DurableContext) -> Result<Value, BoxError> {
    let n = event["n"].as_i64().unwrap_or_default();
    let doubled = ctx
        .step(move |_| async move { Ok(n * 2) })
        .name("double")
        .await?;
    Ok(json!({ "doubled": doubled }))
}

async fn failing_step(_event: Value, ctx: DurableContext) -> Result<Value, BoxError> {
    ctx.step(|_| async move { Err::<i64, BoxError>("step boom".into()) })
        .name("explode")
        .await?;
    Ok(Value::Null)
}

async fn two_waits(_event: Value, ctx: DurableContext) -> Result<String, BoxError> {
    ctx.wait(Duration::from_secs(1)).name("first").await?;
    ctx.wait(Duration::from_secs(1)).name("second").await?;
    Ok("done".to_owned())
}

async fn mapped(_event: Value, ctx: DurableContext) -> Result<Vec<String>, BoxError> {
    let items = vec!["a".to_owned(), "b".to_owned()];
    let results = ctx
        .map(items, |child, item, _idx| async move {
            child
                .step(move |_| async move { Ok(format!("Hello, {item}!")) })
                .name("greet")
                .await
                .map_err(Into::into)
        })
        .name("process-items")
        .max_concurrency(1)
        .item_namer(|_item, idx| format!("item-{idx}"))
        .await?;
    Ok(results)
}

fn keys(value: &Value) -> Vec<&str> {
    let mut keys: Vec<&str> = value
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys
}

#[tokio::test]
async fn a_succeeded_execution_emits_one_complete_record() {
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder());
    let result = runner(telemetry)
        .run(one_step, json!({ "n": 21 }))
        .await
        .expect("runner");
    assert!(result.is_success());

    let records = recording.records();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record["recordType"], "DurableExecutionTelemetry");
    assert_eq!(record["schemaVersion"], "1.0");
    assert_eq!(record["status"], "SUCCEEDED");
    assert_eq!(record["input"], json!({ "n": 21 }));
    assert_eq!(record["output"], json!({ "doubled": 42 }));
    assert_eq!(record["functionName"], "local-test");
    assert_eq!(record["region"], "us-west-2");
    assert_eq!(record["accountId"], "000000000000");
    assert!(record["durationMs"].is_i64());
    assert!(record.get("error").is_none());
    assert!(record.get("truncated").is_none());

    // The local backend reports no attempt for a first-try success, so
    // `maxAttempt` is absent here. The service reports attempt 1, which the
    // telemetry conformance tests check.
    let double = &record["operationsByName"]["double"];
    assert_eq!(
        keys(double),
        [
            "count",
            "failedCount",
            "maxDurationMs",
            "minDurationMs",
            "status",
            "subType",
            "totalDurationMs",
            "type"
        ]
    );
    assert_eq!(double["type"], "STEP");
    assert_eq!(double["status"], "SUCCEEDED");
    assert_eq!(double["count"], 1);
    assert_eq!(double["failedCount"], 0);
    assert_eq!(double["subType"], "Step");

    let operations = record["operations"].as_array().expect("operations");
    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0]["name"], "double");
    assert!(operations[0].get("result").is_none());
}

#[tokio::test]
async fn a_failed_execution_carries_the_error_and_no_output() {
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder());
    let result = runner(telemetry)
        .run(failing_step, json!({}))
        .await
        .expect("runner");
    assert!(!result.is_success());

    let records = recording.records();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    assert_eq!(record["status"], "FAILED");
    assert!(record.get("output").is_none());
    let message = record["error"]["message"].as_str().expect("message");
    assert!(message.contains("step boom"), "{message}");
    let explode = &record["operationsByName"]["explode"];
    assert_eq!(explode["status"], "FAILED");
    assert_eq!(explode["failedCount"], 1);
    assert!(explode["error"]["message"].is_string());
}

#[tokio::test]
async fn include_errors_false_omits_the_operation_error_and_keeps_the_execution_error() {
    let content = ContentConfig::new().include_errors(false);
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder().content(content));
    let result = runner(telemetry)
        .run(failing_step, json!({}))
        .await
        .expect("runner");
    assert!(!result.is_success());

    let records = recording.records();
    assert_eq!(records.len(), 1);
    let record = &records[0];
    let message = record["error"]["message"].as_str().expect("message");
    assert!(message.contains("step boom"), "{message}");
    assert_eq!(record["operations"][0]["status"], "FAILED");
    assert!(record["operations"][0].get("error").is_none());
    let explode = &record["operationsByName"]["explode"];
    assert_eq!(explode["status"], "FAILED");
    assert!(explode.get("error").is_none());
}

#[tokio::test]
async fn on_complete_skips_the_pending_invocations() {
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder());
    let result = runner(telemetry)
        .run(two_waits, json!({}))
        .await
        .expect("runner");
    assert!(result.is_success());
    assert!(result.invocations().len() > 1);

    let records = recording.records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["status"], "SUCCEEDED");
    assert_eq!(records[0]["output"], "done");
    assert_eq!(records[0]["operationsByName"]["first"]["type"], "WAIT");
    assert_eq!(records[0]["operationsByName"]["second"]["type"], "WAIT");
}

#[tokio::test]
async fn on_change_emits_running_records_then_the_terminal_record_last() {
    let (telemetry, recording) =
        recorded(DurableExecutionTelemetry::builder().emit_mode(EmitMode::OnChange));
    let result = runner(telemetry)
        .run(two_waits, json!({}))
        .await
        .expect("runner");
    assert!(result.invocations().len() > 1);

    let records = recording.records();
    let (last, earlier) = records.split_last().expect("records");
    assert_eq!(last["status"], "SUCCEEDED");
    assert!(earlier.iter().all(|r| r["status"] == "RUNNING"));
    assert!(earlier.iter().all(|r| r.get("output").is_none()));
    assert!(earlier.iter().all(|r| r["input"] == json!({})));

    // A record lists every operation so far. So a later record never lists
    // fewer operations than an earlier one.
    let counts: Vec<usize> = records
        .iter()
        .map(|r| r["operations"].as_array().expect("operations").len())
        .collect();
    assert!(
        counts.windows(2).all(|pair| pair[0] <= pair[1]),
        "{counts:?}"
    );
    assert_eq!(counts.last(), Some(&2), "{counts:?}");
}

#[tokio::test]
async fn on_change_emits_a_running_record_for_an_operation_change() {
    let (telemetry, recording) =
        recorded(DurableExecutionTelemetry::builder().emit_mode(EmitMode::OnChange));
    runner(telemetry)
        .run(one_step, json!({ "n": 21 }))
        .await
        .expect("runner");

    // The invocation-start record lists no step, and the invocation-end
    // record has an end time. So only an operation change produces a record
    // that lists the step and has no end time.
    let records = recording.records();
    assert!(
        records
            .iter()
            .any(|r| r.get("endTime").is_none() && r["operationsByName"].get("double").is_some()),
        "{records:?}"
    );
}

#[tokio::test]
async fn on_change_with_a_slow_exporter_keeps_the_task_count_bounded() {
    const STEPS: usize = 500;
    let slow = Slow::default();
    let telemetry = DurableExecutionTelemetry::builder()
        .emit_mode(EmitMode::OnChange)
        .exporter(slow.clone())
        .build();
    let many_steps = |_event: Value, ctx: DurableContext| async move {
        for n in 0..STEPS {
            ctx.step(move |_| async move { Ok(n) })
                .name("count")
                .await?;
        }
        Ok::<_, BoxError>(Value::Null)
    };
    runner(telemetry)
        .run(many_steps, json!({}))
        .await
        .expect("runner");

    // Each step changes its operation at least once, and each export takes
    // longer than a step. One pump task exports for the whole scheduler.
    // So the number of live tasks does not grow with the number of steps.
    // The observed peak is 3 tasks. The limit of 10 leaves room for tasks
    // that the runner may add.
    let peak = slow.peak_tasks.load(Ordering::Relaxed);
    assert!(peak <= 10, "{peak} live tasks");

    let statuses = slow.statuses.lock().expect("lock").clone();
    let (last, earlier) = statuses.split_last().expect("records");
    assert_eq!(last, "SUCCEEDED");
    assert!(earlier.iter().all(|status| status == "RUNNING"));
}

#[tokio::test]
async fn on_change_emits_a_running_record_when_the_invocation_starts() {
    let (telemetry, recording) =
        recorded(DurableExecutionTelemetry::builder().emit_mode(EmitMode::OnChange));
    // The scheduler's pump task exports the invocation-start record. The
    // handler yields before its first operation, so the pump runs first.
    let yields_then_steps = |event: Value, ctx: DurableContext| async move {
        tokio::task::yield_now().await;
        one_step(event, ctx).await
    };
    runner(telemetry)
        .run(yields_then_steps, json!({ "n": 21 }))
        .await
        .expect("runner");

    let records = recording.records();
    let (first, last) = (
        records.first().expect("records"),
        records.last().expect("records"),
    );
    // The handler has not run yet, so the step is not in the record.
    assert_eq!(first["status"], "RUNNING");
    assert_eq!(first["input"], json!({ "n": 21 }));
    assert!(first["operationsByName"].get("double").is_none(), "{first}");
    assert_eq!(last["operationsByName"]["double"]["type"], "STEP");
}

#[tokio::test]
async fn on_failure_emits_only_for_a_failed_execution() {
    let (telemetry, recording) =
        recorded(DurableExecutionTelemetry::builder().emit_mode(EmitMode::OnFailure));
    runner(telemetry.clone())
        .run(one_step, json!({ "n": 1 }))
        .await
        .expect("runner");
    assert_eq!(recording.records().len(), 0);

    runner(telemetry)
        .run(failing_step, json!({}))
        .await
        .expect("runner");
    let records = recording.records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["status"], "FAILED");
}

#[tokio::test]
async fn a_zero_sampling_rate_emits_nothing() {
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder().sampling_rate(0.0));
    let result = runner(telemetry)
        .run(one_step, json!({ "n": 1 }))
        .await
        .expect("runner");
    assert!(result.is_success());
    assert_eq!(recording.records().len(), 0);
}

#[tokio::test]
async fn top_level_lists_only_the_map_and_full_tree_lists_its_children() {
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder());
    runner(telemetry)
        .run(mapped, json!({}))
        .await
        .expect("runner");
    let records = recording.records();
    assert_eq!(keys(&records[0]["operationsByName"]), ["process-items"]);
    assert_eq!(
        records[0]["operationsByName"]["process-items"]["type"],
        "CONTEXT"
    );
    assert!(
        records[0]["operationsByName"]["process-items"]
            .get("maxAttempt")
            .is_none()
    );

    let (telemetry, recording) =
        recorded(DurableExecutionTelemetry::builder().operation_detail(OperationDetail::FullTree));
    runner(telemetry)
        .run(mapped, json!({}))
        .await
        .expect("runner");
    let records = recording.records();
    let by_name = &records[0]["operationsByName"];
    assert_eq!(
        keys(by_name),
        ["greet", "item-0", "item-1", "process-items"]
    );
    assert_eq!(by_name["greet"]["count"], 2);
    assert_eq!(by_name["item-0"]["type"], "CONTEXT");
}

#[tokio::test]
async fn content_settings_shape_the_payloads() {
    let content = ContentConfig::new()
        .input(ContentSetting::Omit)
        .output(ContentSetting::transform(|value: Value| {
            Ok(json!({ "kept": value["doubled"].clone() }))
        }))
        .operation_override(OperationOverride::new("double").result(|value: Value| Ok(value)));
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder().content(content));
    runner(telemetry)
        .run(one_step, json!({ "n": 2 }))
        .await
        .expect("runner");
    let records = recording.records();
    let record = &records[0];
    assert!(record.get("input").is_none());
    assert_eq!(record["output"], json!({ "kept": 4 }));
    assert_eq!(record["operationsByName"]["double"]["result"], 4);
    assert_eq!(record["operations"][0]["result"], 4);
}

#[tokio::test]
async fn an_excluded_operation_is_absent() {
    let content =
        ContentConfig::new().operation_override(OperationOverride::new("double").exclude());
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder().content(content));
    runner(telemetry)
        .run(one_step, json!({ "n": 2 }))
        .await
        .expect("runner");
    let records = recording.records();
    assert_eq!(records[0]["operations"], json!([]));
    assert_eq!(records[0]["operationsByName"], json!({}));
}

#[tokio::test]
async fn a_failing_exporter_does_not_fail_the_execution() {
    let recording = Recording::default();
    let telemetry = DurableExecutionTelemetry::builder()
        .exporter(Failing)
        .exporter(recording.clone())
        .build();
    let result = runner(telemetry)
        .run(one_step, json!({ "n": 3 }))
        .await
        .expect("runner");
    assert!(result.is_success());
    assert_eq!(recording.records().len(), 1);
}

async fn flaky(_event: Value, ctx: DurableContext) -> Result<u32, BoxError> {
    let attempt = ctx
        .step(|step_ctx| async move {
            let attempt = step_ctx.attempt();
            if attempt < 3 {
                return Err(format!("transient failure on attempt {attempt}").into());
            }
            Ok(attempt)
        })
        .name("flaky-call")
        .retry_strategy(|_err, _attempt| durable::RetryDecision::retry(Duration::from_secs(1)))
        .await?;
    Ok(attempt)
}

/// Runs the name `work` twice with the step `other` between the two runs.
/// The first `work` fails and the second succeeds.
async fn repeated_name(_event: Value, ctx: DurableContext) -> Result<u32, BoxError> {
    let first = ctx
        .step(|_| async move { Err::<u32, BoxError>("first run fails".into()) })
        .name("work")
        .retry_strategy(|_err, _attempt| durable::RetryDecision::Stop)
        .await;
    assert!(first.is_err());
    ctx.step(|_| async move { Ok(1_u32) }).name("other").await?;
    let second = ctx.step(|_| async move { Ok(2_u32) }).name("work").await?;
    Ok(second)
}

/// The three steps run in one invocation. The local runner's virtual clock
/// does not move during an invocation. So the three steps have the same
/// start time, and the start time alone does not order them. The record
/// lists them in the order they ran, and the summary of the repeated name
/// reports its last run.
#[tokio::test]
async fn operations_with_equal_start_times_keep_the_order_they_ran_in() {
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder());
    let result = runner(telemetry)
        .run(repeated_name, json!({}))
        .await
        .expect("runner");
    assert!(result.is_success());
    let records = recording.records();
    assert_eq!(records.len(), 1);
    let operations = records[0]["operations"].as_array().expect("array");
    let starts: Vec<_> = operations.iter().map(|op| &op["startTime"]).collect();
    assert!(starts[0].is_string());
    assert!(
        starts.iter().all(|start| *start == starts[0]),
        "the test needs equal start times: {starts:?}"
    );
    let ran: Vec<_> = operations
        .iter()
        .map(|op| (op["name"].as_str(), op["status"].as_str()))
        .collect();
    assert_eq!(
        ran,
        [
            (Some("work"), Some("FAILED")),
            (Some("other"), Some("SUCCEEDED")),
            (Some("work"), Some("SUCCEEDED")),
        ]
    );
    let work = &records[0]["operationsByName"]["work"];
    assert_eq!(work["count"], 2);
    assert_eq!(work["status"], "SUCCEEDED");
}

#[tokio::test]
async fn a_retried_step_reports_its_highest_attempt() {
    let (telemetry, recording) = recorded(DurableExecutionTelemetry::builder());
    let result = runner(telemetry)
        .run(flaky, json!({}))
        .await
        .expect("runner");
    assert!(result.is_success());
    let records = recording.records();
    assert_eq!(records.len(), 1);
    let flaky = &records[0]["operationsByName"]["flaky-call"];
    assert_eq!(flaky["status"], "SUCCEEDED");
    assert!(
        flaky["maxAttempt"].as_u64().is_some_and(|n| n >= 2),
        "{flaky}"
    );
}
