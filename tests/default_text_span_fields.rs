//! The text format of the default subscriber shows each field of a span
//! once, with the value recorded last.
//!
//! This file holds one test that installs a subscriber, so no other test in
//! its binary installs one first.
//!
//! The helper reads the process environment. Changing the environment of a
//! running test is `unsafe` and races with other threads. So the test runs
//! this binary again as a child process with a controlled environment. The
//! child installs the subscriber, runs a handler under `LocalRunner`, and
//! prints what the writer captured.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::panic)] // reason: test assertions

use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::lambda_runtime::tracing::subscriber;
use durable::test_util::LocalRunner;

/// Set in the child process. The test then installs the subscriber instead
/// of starting a child.
const CHILD_VAR: &str = "DURABLE_SDK_TEXT_SPAN_FIELDS_CHILD";

/// Precedes each line the child captured, so the parent can find it in the
/// child's output.
const CAPTURED_PREFIX: &str = "captured: ";

/// The variables the helper reads.
const LOG_VARS: [&str; 3] = ["AWS_LAMBDA_LOG_LEVEL", "RUST_LOG", "AWS_LAMBDA_LOG_FORMAT"];

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

impl<'a> subscriber::fmt::MakeWriter<'a> for CaptureWriter {
    type Writer = CaptureWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// The handler the child runs: steps around a wait, a child context that
/// logs, and a user span whose field is recorded again.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<(), durable::BoxError> {
    ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
    tracing::info!("between");
    ctx.wait(Duration::from_secs(1)).name("pause").await?;
    ctx.step(|_| async { Ok(2_u32) }).name("b").await?;
    ctx.run_in_child_context(|child| async move {
        child.step(|_| async { Ok(3_u32) }).name("inner").await?;
        tracing::info!("in-child");
        Ok(())
    })
    .name("child")
    .await?;
    let phase = tracing::info_span!("phase", stage = "one", stable = "keep");
    phase.record("stage", "two");
    phase.in_scope(|| tracing::info!("in-phase"));
    tracing::info!("after");
    Ok(())
}

/// Runs in the child: installs the subscriber, runs [`handler`], and prints
/// each captured line with [`CAPTURED_PREFIX`].
fn child() {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let installed =
        durable::init_default_subscriber_with_writer(CaptureWriter(Arc::clone(&buffer)));
    assert!(installed.is_ok(), "{installed:?}");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    let result = runtime
        .block_on(LocalRunner::new().run(handler, serde_json::Value::Null))
        .expect("runner");
    assert!(result.is_success(), "{result}");
    assert!(result.invocation_count() >= 2, "the wait suspends");

    let output = buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    );
    for line in output.lines() {
        println!("{CAPTURED_PREFIX}{line}");
    }
}

/// Runs this test in a child process with no log variable set, and returns
/// the lines the child's writer captured, without ANSI color sequences.
fn captured_by_child() -> Vec<String> {
    let exe = std::env::current_exe().expect("test binary path");
    let mut command = Command::new(exe);
    command.args([
        "--exact",
        "default_text_shows_each_span_field_once",
        "--nocapture",
        "--test-threads=1",
    ]);
    for var in LOG_VARS {
        command.env_remove(var);
    }
    command.env(CHILD_VAR, "1");

    let output = command.output().expect("child process");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "child failed: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
        .lines()
        .filter_map(|line| {
            // The harness may print the test name on the same line first.
            line.find(CAPTURED_PREFIX)
                .and_then(|start| line.get(start + CAPTURED_PREFIX.len()..))
        })
        .map(strip_ansi)
        .collect()
}

/// `line` without its ANSI SGR sequences (`ESC [ ... m`).
fn strip_ansi(line: &str) -> String {
    let mut text = String::with_capacity(line.len());
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for skipped in chars.by_ref() {
                if skipped == 'm' {
                    break;
                }
            }
        } else {
            text.push(c);
        }
    }
    text
}

/// The spans of a text line: each span's name and its fields, as
/// `(field name, value)` pairs in the order the line shows them.
///
/// A span shows as `name{field=value field=value}`. A quoted value may hold
/// a space or a brace, so the parser tracks quotes.
fn spans(line: &str) -> Vec<(String, Vec<(String, String)>)> {
    let mut spans = Vec::new();
    let mut rest = line;
    while let Some(open) = rest.find('{') {
        let name = rest
            .get(..open)
            .and_then(|head| head.rsplit([' ', ':']).next())
            .unwrap_or_default()
            .to_owned();
        let body = rest.get(open + 1..).unwrap_or_default();
        let mut fields = Vec::new();
        let mut token = String::new();
        let mut quoted = false;
        let mut escaped = false;
        let mut end = body.len();
        for (index, c) in body.char_indices() {
            if quoted {
                token.push(c);
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    quoted = false;
                }
                continue;
            }
            match c {
                '"' => {
                    quoted = true;
                    token.push(c);
                }
                ' ' | '}' => {
                    if let Some((field, value)) = token.split_once('=') {
                        fields.push((field.to_owned(), value.to_owned()));
                    }
                    token.clear();
                    if c == '}' {
                        end = index;
                        break;
                    }
                }
                _ => token.push(c),
            }
        }
        spans.push((name, fields));
        rest = body.get(end + 1..).unwrap_or_default();
    }
    spans
}

/// The SDK's spans of a line: those that carry an `isReplay` field.
fn sdk_spans(spans: &[(String, Vec<(String, String)>)]) -> Vec<&Vec<(String, String)>> {
    spans
        .iter()
        .map(|(_, fields)| fields)
        .filter(|fields| fields.iter().any(|(name, _)| name == "isReplay"))
        .collect()
}

#[test]
fn default_text_shows_each_span_field_once() {
    if std::env::var_os(CHILD_VAR).is_some() {
        child();
        return;
    }

    let lines = captured_by_child();
    let line_of = |message: &str| {
        lines
            .iter()
            .find(|line| line.ends_with(&format!(": {message}")))
            .unwrap_or_else(|| panic!("no {message} line in {lines:#?}"))
            .clone()
    };

    // No span of any line shows a field name twice.
    for line in &lines {
        for (span, fields) in spans(line) {
            let mut names: Vec<&str> = fields.iter().map(|(name, _)| name.as_str()).collect();
            let count = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), count, "span {span} repeats a field: {line}");
        }
    }

    // Each SDK span of a line logged after its namespace's replay shows
    // `isReplay=false`, and no other value.
    for message in ["between", "in-child", "after"] {
        let line = line_of(message);
        let spans = spans(&line);
        let sdk = sdk_spans(&spans);
        assert!(!sdk.is_empty(), "no SDK span: {line}");
        for fields in sdk {
            let values: Vec<&str> = fields
                .iter()
                .filter(|(name, _)| name == "isReplay")
                .map(|(_, value)| value.as_str())
                .collect();
            assert_eq!(values, ["false"], "{line}");
        }
    }

    // The user span shows `stage` once, with its recorded value, before
    // `stable`.
    let line = line_of("in-phase");
    let spans = spans(&line);
    let (_, phase) = spans
        .iter()
        .find(|(name, _)| name == "phase")
        .unwrap_or_else(|| panic!("no phase span: {line}"));
    let names: Vec<&str> = phase.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["stage", "stable"], "{line}");
    assert_eq!(
        phase.first().map(|(_, value)| value.as_str()),
        Some("\"two\""),
        "{line}"
    );
}
