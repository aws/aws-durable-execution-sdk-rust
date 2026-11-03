//! `init_default_subscriber_with_writer` installs a subscriber that writes
//! to the given writer.
//!
//! This file holds one test, so no other test in its binary installs a
//! subscriber first.
//!
//! The helper reads the process environment. Changing the environment of a
//! running test is `unsafe` and races with other threads. So the test runs
//! this binary again as a child process with a controlled environment. The
//! child installs the subscriber, logs, and prints what the writer captured.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::process::Command;
use std::sync::{Arc, Mutex};

use aws_durable_execution_sdk as durable;
use durable::lambda_runtime::tracing::subscriber;

/// Set in the child process. The test then installs the subscriber instead
/// of starting children.
const CHILD_VAR: &str = "DURABLE_SDK_SUBSCRIBER_TEST_CHILD";

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

/// Runs in the child: installs the subscriber, logs `x` at `INFO`, and
/// prints each captured line with [`CAPTURED_PREFIX`].
fn child() {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let installed =
        durable::init_default_subscriber_with_writer(CaptureWriter(Arc::clone(&buffer)));
    assert!(installed.is_ok(), "{installed:?}");

    tracing::info!("x");

    let output = buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    );
    for line in output.lines() {
        println!("{CAPTURED_PREFIX}{line}");
    }
}

/// Runs this test in a child process whose only log variables are `vars`,
/// and returns the lines the child's writer captured.
fn captured_by_child(vars: &[(&str, &str)]) -> Vec<String> {
    let exe = std::env::current_exe().expect("test binary path");
    let mut command = Command::new(exe);
    command.args([
        "--exact",
        "init_with_writer_installs_and_writes",
        "--nocapture",
        "--test-threads=1",
    ]);
    for var in LOG_VARS {
        command.env_remove(var);
    }
    command.envs(vars.iter().copied()).env(CHILD_VAR, "1");

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
        .map(str::to_owned)
        .collect()
}

#[test]
fn init_with_writer_installs_and_writes() {
    if std::env::var_os(CHILD_VAR).is_some() {
        child();
        return;
    }

    // With no log variable set, the level is `INFO` and the format is text.
    let lines = captured_by_child(&[]);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines.first().is_some_and(|line| line.contains('x')),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .all(|line| serde_json::from_str::<serde_json::Value>(line).is_err()),
        "{lines:?}"
    );

    // The helper reads the level from the environment.
    let lines = captured_by_child(&[("AWS_LAMBDA_LOG_LEVEL", "ERROR")]);
    assert!(lines.is_empty(), "{lines:?}");

    // The helper reads the format from the environment.
    let lines = captured_by_child(&[("AWS_LAMBDA_LOG_FORMAT", "json")]);
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert!(
        lines
            .iter()
            .all(|line| serde_json::from_str::<serde_json::Value>(line).is_ok()),
        "{lines:?}"
    );
}
