//! Shrinks a record to an exporter's size limit.

use std::borrow::Cow;
use std::io;

use crate::record::{OperationRecord, TelemetryRecord};
use crate::render::{OperationsFormat, RecordView};

/// Counts the bytes written to it and discards them.
struct ByteCounter(usize);

impl io::Write for ByteCounter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 = self.0.saturating_add(buf.len());
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    /// How many records this thread has rendered to measure them.
    static RENDERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The UTF-8 size of a view's JSON, or `None` when it does not serialize.
fn json_size(view: &RecordView<'_>) -> Option<usize> {
    #[cfg(test)]
    RENDERS.with(|count| count.set(count.get() + 1));
    let mut counter = ByteCounter(0);
    serde_json::to_writer(&mut counter, view).ok()?;
    Some(counter.0)
}

/// The operation indices, oldest start time first. An operation without a
/// start time counts as the newest, so it is dropped last. Equal start
/// times keep the list order.
fn oldest_first(operations: &[OperationRecord]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..operations.len()).collect();
    order.sort_by_key(|&index| {
        let start = operations.get(index).and_then(|op| op.start_ms);
        (start.is_none(), start)
    });
    order
}

/// The working state of one truncation.
struct Candidate {
    record: TelemetryRecord,
    kept: Vec<bool>,
    format: OperationsFormat,
    max_bytes: usize,
}

impl Candidate {
    fn fits(&self) -> bool {
        let operations: Vec<&OperationRecord> = self
            .record
            .operations
            .iter()
            .zip(&self.kept)
            .filter_map(|(op, kept)| kept.then_some(op))
            .collect();
        let view = RecordView {
            record: &self.record,
            operations: &operations,
            format: self.format,
        };
        json_size(&view).is_some_and(|size| size <= self.max_bytes)
    }
}

/// Returns `record` shrunk to fit `max_bytes`, measured on the JSON that
/// `format` renders.
///
/// The record is returned unchanged when `max_bytes` is `None` or zero,
/// when it already fits, or when it does not serialize. Otherwise the
/// function drops parts in this order until the record fits:
///
/// 1. operation results, oldest operation first; each such operation is
///    marked `truncated: true`;
/// 2. whole operations, oldest first, counted in `droppedOperations`;
/// 3. the execution input (`droppedInput: true`), then the output
///    (`droppedOutput: true`).
///
/// The identity fields, the times, the status and each error's name and
/// message are never dropped. When anything was dropped, the returned
/// record has `truncated: true`. When nothing could be dropped, the
/// record is returned unchanged, even though it is still too large.
///
/// The JavaScript exporter first drops error `cause` chains. The SDK's
/// errors carry no cause, so this function has no such step.
pub fn truncate_record(
    record: &TelemetryRecord,
    max_bytes: Option<usize>,
    format: OperationsFormat,
) -> Cow<'_, TelemetryRecord> {
    let Some(max_bytes) = max_bytes.filter(|&max| max > 0) else {
        return Cow::Borrowed(record);
    };
    let operations: Vec<&OperationRecord> = record.operations.iter().collect();
    let initial = json_size(&RecordView {
        record,
        operations: &operations,
        format,
    });
    if initial.is_none_or(|size| size <= max_bytes) {
        return Cow::Borrowed(record);
    }

    let mut candidate = Candidate {
        record: record.clone(),
        kept: vec![true; record.operations.len()],
        format,
        max_bytes,
    };
    candidate.record.truncated = true;
    let order = oldest_first(&record.operations);
    let mut dropped_any = false;

    // `fits()` serializes the whole record. An operation without a result
    // leaves the record unchanged, so a size check before it would repeat
    // the previous answer. So the loop measures the record only before an
    // operation that has a result to drop.
    for &index in &order {
        let has_result = candidate
            .record
            .operations
            .get(index)
            .is_some_and(|op| op.result.is_some());
        if !has_result {
            continue;
        }
        if candidate.fits() {
            break;
        }
        if let Some(op) = candidate.record.operations.get_mut(index) {
            op.result = None;
            op.truncated = true;
            dropped_any = true;
        }
    }

    for &index in &order {
        if candidate.fits() {
            break;
        }
        if let Some(kept) = candidate.kept.get_mut(index)
            && *kept
        {
            *kept = false;
            candidate.record.dropped_operations += 1;
            dropped_any = true;
        }
    }

    if !candidate.fits() && candidate.record.input.is_some() {
        candidate.record.input = None;
        candidate.record.dropped_input = true;
        dropped_any = true;
    }
    if !candidate.fits() && candidate.record.output.is_some() {
        candidate.record.output = None;
        candidate.record.dropped_output = true;
        dropped_any = true;
    }

    if !dropped_any {
        return Cow::Borrowed(record);
    }
    let Candidate {
        mut record, kept, ..
    } = candidate;
    let mut kept = kept.into_iter();
    record.operations.retain(|_| kept.next().unwrap_or(true));
    Cow::Owned(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::render_json;
    use crate::render::tests::{op, record};
    use serde_json::json;

    fn size(record: &TelemetryRecord, format: OperationsFormat) -> usize {
        render_json(record, format).map_or(0, |s| s.len())
    }

    fn timed(name: &str, start_ms: Option<i64>, result: Option<&str>) -> OperationRecord {
        let mut o = op(name, "SUCCEEDED");
        o.start_ms = start_ms;
        o.result = result.map(|r| json!(r));
        o
    }

    #[test]
    fn no_limit_returns_the_record() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        assert!(matches!(
            truncate_record(&r, None, OperationsFormat::Array),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            truncate_record(&r, Some(0), OperationsFormat::Array),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn a_record_that_fits_is_returned_unchanged() {
        let r = record(vec![op("a", "SUCCEEDED")]);
        let limit = size(&r, OperationsFormat::Array);
        assert!(matches!(
            truncate_record(&r, Some(limit), OperationsFormat::Array),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn results_are_dropped_oldest_first() {
        let big = "x".repeat(200);
        let r = record(vec![
            timed("new", Some(20), Some(&big)),
            timed("old", Some(10), Some(&big)),
        ]);
        let limit = size(&r, OperationsFormat::Array) - 100;
        let out = truncate_record(&r, Some(limit), OperationsFormat::Array);
        assert!(out.truncated);
        let old = out
            .operations
            .iter()
            .find(|o| o.name.as_deref() == Some("old"));
        let new = out
            .operations
            .iter()
            .find(|o| o.name.as_deref() == Some("new"));
        assert_eq!(
            old.map(|o| (o.result.is_none(), o.truncated)),
            Some((true, true))
        );
        assert_eq!(
            new.map(|o| (o.result.is_some(), o.truncated)),
            Some((true, false))
        );
        assert_eq!(out.dropped_operations, 0);
        assert!(size(&out, OperationsFormat::Array) <= limit);
    }

    #[test]
    fn result_dropping_stops_at_the_first_fit_among_operations_without_results() {
        let big = "x".repeat(200);
        let r = record(vec![
            timed("new", Some(40), Some(&big)),
            timed("plain-late", Some(30), None),
            timed("old", Some(20), Some(&big)),
            timed("plain-early", Some(10), None),
        ]);
        let limit = size(&r, OperationsFormat::Array) - 100;
        let out = truncate_record(&r, Some(limit), OperationsFormat::Array);
        let state: Vec<_> = out
            .operations
            .iter()
            .map(|o| (o.name.as_deref(), o.result.is_some(), o.truncated))
            .collect();
        assert_eq!(
            state,
            [
                (Some("new"), true, false),
                (Some("plain-late"), false, false),
                (Some("old"), false, true),
                (Some("plain-early"), false, false),
            ]
        );
        assert_eq!(out.dropped_operations, 0);
        assert!(size(&out, OperationsFormat::Array) <= limit);
    }

    #[test]
    fn operations_are_dropped_oldest_first_after_results() {
        let r = record(vec![
            timed("unstarted", None, None),
            timed("new", Some(20), None),
            timed("old", Some(10), None),
        ]);
        let mut without_one = r.clone();
        without_one
            .operations
            .retain(|o| o.name.as_deref() != Some("old"));
        without_one.truncated = true;
        without_one.dropped_operations = 1;
        let limit = size(&without_one, OperationsFormat::Array);
        let out = truncate_record(&r, Some(limit), OperationsFormat::Array);
        let names: Vec<_> = out
            .operations
            .iter()
            .filter_map(|o| o.name.clone())
            .collect();
        assert_eq!(names, ["unstarted", "new"]);
        assert_eq!(out.dropped_operations, 1);
        assert!(!out.dropped_input);
    }

    #[test]
    fn an_unstarted_operation_is_dropped_last() {
        let r = record(vec![
            timed("unstarted", None, None),
            timed("old", Some(10), None),
        ]);
        let mut only_unstarted = r.clone();
        only_unstarted
            .operations
            .retain(|o| o.name.as_deref() == Some("unstarted"));
        only_unstarted.truncated = true;
        only_unstarted.dropped_operations = 1;
        let limit = size(&only_unstarted, OperationsFormat::Array);
        let out = truncate_record(&r, Some(limit), OperationsFormat::Array);
        let names: Vec<_> = out
            .operations
            .iter()
            .filter_map(|o| o.name.clone())
            .collect();
        assert_eq!(names, ["unstarted"]);
    }

    #[test]
    fn input_then_output_are_dropped_last() {
        let mut r = record(vec![op("a", "SUCCEEDED")]);
        r.input = Some(json!("i".repeat(300)));
        r.output = Some(json!("o".repeat(300)));
        let mut no_input = r.clone();
        no_input.operations.clear();
        no_input.input = None;
        no_input.truncated = true;
        no_input.dropped_operations = 1;
        no_input.dropped_input = true;
        let limit = size(&no_input, OperationsFormat::Array);
        let out = truncate_record(&r, Some(limit), OperationsFormat::Array);
        assert!(out.dropped_input);
        assert!(!out.dropped_output);
        assert!(out.output.is_some());

        let out = truncate_record(&r, Some(10), OperationsFormat::Array);
        assert!(out.dropped_input);
        assert!(out.dropped_output);
        assert!(out.input.is_none());
        assert!(out.output.is_none());
        assert_eq!(out.operations, Vec::new());
    }

    #[test]
    fn a_record_with_nothing_to_drop_is_returned_unchanged() {
        let mut r = record(Vec::new());
        r.input = None;
        r.output = None;
        assert!(matches!(
            truncate_record(&r, Some(10), OperationsFormat::Array),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn the_limit_is_measured_on_the_rendered_form() {
        // The by-name form of one operation is longer than its array form,
        // so a limit between the two sizes truncates only the by-name form.
        let mut o = op("a-long-operation-name", "SUCCEEDED");
        o.duration_ms = Some(1);
        o.attempt = Some(1);
        let mut r = record(vec![o]);
        r.input = None;
        r.output = None;
        let array = size(&r, OperationsFormat::Array);
        let by_name = size(&r, OperationsFormat::ByName);
        assert!(by_name > array, "{by_name} {array}");
        let limit = array;
        assert!(matches!(
            truncate_record(&r, Some(limit), OperationsFormat::Array),
            Cow::Borrowed(_)
        ));
        let out = truncate_record(&r, Some(limit), OperationsFormat::ByName);
        assert_eq!(out.dropped_operations, 1);
    }

    /// `count` operations named `step-0`, `step-1` and so on, started in
    /// that order, each with a duration and an attempt.
    fn steps(count: i64) -> TelemetryRecord {
        let operations = (0..count)
            .map(|i| {
                let mut o = timed(&format!("step-{i}"), Some(i), None);
                o.duration_ms = Some(3);
                o.attempt = Some(1);
                o
            })
            .collect();
        record(operations)
    }

    /// The number of renders `truncate_record` makes for one call.
    fn renders(r: &TelemetryRecord, limit: usize, format: OperationsFormat) -> usize {
        RENDERS.with(|count| count.set(0));
        let out = truncate_record(r, Some(limit), format);
        assert!(out.dropped_operations > 0);
        RENDERS.with(std::cell::Cell::get)
    }

    #[test]
    fn many_dropped_operations_take_a_few_renders() {
        let r = steps(2_000);
        // The CloudWatch Logs limit and the by-name form.
        let by_name = renders(&r, 256_000, OperationsFormat::ByName);
        assert!(by_name < 40, "by name: {by_name} renders");
        for format in [OperationsFormat::Array, OperationsFormat::Both] {
            let count = renders(&r, size(&r, format) / 2, format);
            assert!(count < 40, "{format:?}: {count} renders");
        }
    }

    #[test]
    fn byte_counter_flush_is_a_no_op() {
        let mut counter = ByteCounter(3);
        assert!(io::Write::flush(&mut counter).is_ok());
        assert_eq!(counter.0, 3);
    }
}
