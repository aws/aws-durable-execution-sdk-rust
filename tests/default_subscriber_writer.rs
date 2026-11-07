//! `init_default_subscriber_with_writer` installs a subscriber that writes
//! to the given writer.
//!
//! This file holds one test that installs a subscriber, so no other test in
//! its binary installs one first.
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

    // The helper reads the format from the environment. The JSON record
    // has exactly `timestamp`, `level` and `message`, in that order.
    let started = unix_millis(std::time::SystemTime::now());
    let lines = captured_by_child(&[("AWS_LAMBDA_LOG_FORMAT", "json")]);
    let finished = unix_millis(std::time::SystemTime::now());
    assert_eq!(lines.len(), 1, "{lines:?}");
    let line = lines.first().expect("one line");
    let record: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(line).expect("a JSON object");
    assert_eq!(record.len(), 3, "{line}");
    // The values are `INFO`, `x` and a timestamp, so no value contains a
    // key's quoted name, and the first match of each is its key.
    let position = |key: &str| line.find(&format!("\"{key}\":")).expect("the key");
    assert!(
        position("timestamp") < position("level") && position("level") < position("message"),
        "{line}"
    );
    assert_eq!(
        record.get("level"),
        Some(&serde_json::json!("INFO")),
        "{line}"
    );
    assert_eq!(
        record.get("message"),
        Some(&serde_json::json!("x")),
        "{line}"
    );
    let timestamp = record
        .get("timestamp")
        .and_then(serde_json::Value::as_str)
        .expect("a timestamp string");
    // The child formatted the record between the parent's two clock reads,
    // and the timestamp cuts the time to whole milliseconds.
    let millis = js_timestamp_millis(timestamp).expect("a valid UTC timestamp");
    assert!(started <= millis && millis <= finished, "{line}");
}

/// Milliseconds since the Unix epoch of `time`, cut to whole milliseconds.
fn unix_millis(time: std::time::SystemTime) -> i64 {
    let since = time
        .duration_since(std::time::UNIX_EPOCH)
        .expect("a time after the epoch");
    i64::try_from(since.as_millis()).expect("milliseconds fit in i64")
}

/// The milliseconds since the Unix epoch of `text`, when `text` is a valid
/// UTC time of the form `YYYY-MM-DDTHH:MM:SS.sssZ`.
fn js_timestamp_millis(text: &str) -> Option<i64> {
    let shape_ok = text.len() == 24
        && text.char_indices().all(|(i, c)| match i {
            4 | 7 => c == '-',
            10 => c == 'T',
            13 | 16 => c == ':',
            19 => c == '.',
            23 => c == 'Z',
            _ => c.is_ascii_digit(),
        });
    if !shape_ok {
        return None;
    }
    let number = |range: std::ops::Range<usize>| -> Option<i64> { text.get(range)?.parse().ok() };
    let (year, month, day) = (number(0..4)?, number(5..7)?, number(8..10)?);
    let (hour, minute, second) = (number(11..13)?, number(14..16)?, number(17..19)?);
    let millis = number(20..23)?;
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let month_days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return None,
    };
    if !(1..=month_days).contains(&day) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    // Howard Hinnant's `days_from_civil`.
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let year_of_era = y - era * 400;
    let day_of_year = (153 * m + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    Some(((days * 24 + hour) * 60 + minute) * 60_000 + second * 1000 + millis)
}

#[test]
fn js_timestamp_millis_accepts_only_valid_times() {
    assert_eq!(js_timestamp_millis("1970-01-01T00:00:00.000Z"), Some(0));
    assert_eq!(
        js_timestamp_millis("2023-11-14T22:13:20.123Z"),
        Some(1_700_000_000_123)
    );
    assert_eq!(
        js_timestamp_millis("2024-02-29T23:59:59.999Z"),
        Some(1_709_251_199_999)
    );
    for invalid in [
        "9999-99-99T99:99:99.999Z",
        "2023-02-29T00:00:00.000Z",
        "2024-13-01T00:00:00.000Z",
        "2024-00-01T00:00:00.000Z",
        "2024-04-31T00:00:00.000Z",
        "2024-01-00T00:00:00.000Z",
        "2024-01-01T24:00:00.000Z",
        "2024-01-01T00:60:00.000Z",
        "2024-01-01T00:00:60.000Z",
        "2024-01-01T00:00:00.000+00:00",
        "2024-01-01 00:00:00.000Z",
    ] {
        assert_eq!(js_timestamp_millis(invalid), None, "{invalid}");
    }
}
