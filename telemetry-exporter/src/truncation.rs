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

    /// Clears the results of the first `count` operations in
    /// `with_results` and restores the other results from `original`.
    fn clear_results(&mut self, original: &TelemetryRecord, with_results: &[usize], count: usize) {
        for (position, &index) in with_results.iter().enumerate() {
            let source = original.operations.get(index);
            if let (Some(op), Some(source)) = (self.record.operations.get_mut(index), source) {
                op.truncated = source.truncated || position < count;
                op.result = (position >= count).then(|| source.result.clone()).flatten();
            }
        }
    }

    /// Drops the first `count` operations in `order` and keeps the others.
    fn drop_operations(&mut self, order: &[usize], base: usize, count: usize) {
        for (position, &index) in order.iter().enumerate() {
            if let Some(kept) = self.kept.get_mut(index) {
                *kept = position >= count;
            }
        }
        self.record.dropped_operations = base.saturating_add(count);
    }
}

/// Returns the smallest count in `low..=high` for which `fits` returned
/// true, or `None` when no count it measured fits.
///
/// The search assumes that dropping more never makes the record larger.
/// Some records break this. For example, a name's summary in the by-name
/// form shows a result and an error only when one operation has that name.
/// So a drop that leaves one operation with a name can make the record
/// larger. Then the search can return more than the fewest drops that fit.
/// A count is returned only after `fits` returned true for it. So the
/// chosen record never goes over the limit.
fn first_fit(low: usize, high: usize, mut fits: impl FnMut(usize) -> bool) -> Option<usize> {
    let (mut low, mut high, mut found) = (low, high.saturating_add(1), None);
    while low < high {
        let mid = low + (high - low) / 2;
        if fits(mid) {
            found = Some(mid);
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    found
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
    let with_results: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&index| {
            record
                .operations
                .get(index)
                .is_some_and(|op| op.result.is_some())
        })
        .collect();
    let base = record.dropped_operations;

    // Each size check renders the whole record. So one check per dropped
    // part costs time that grows with the square of the operation count.
    // Each phase therefore binary searches the number of parts to drop.
    // With no result cleared, the record is the input marked truncated,
    // which does not fit. So the search over results starts at one.
    let cleared = first_fit(1, with_results.len(), |count| {
        candidate.clear_results(record, &with_results, count);
        candidate.fits()
    });
    let mut fitted = cleared.is_some();
    let cleared = cleared.unwrap_or(with_results.len());
    candidate.clear_results(record, &with_results, cleared);
    let mut dropped_any = cleared > 0;

    if !fitted {
        let dropped = first_fit(1, order.len(), |count| {
            candidate.drop_operations(&order, base, count);
            candidate.fits()
        });
        fitted = dropped.is_some();
        let dropped = dropped.unwrap_or(order.len());
        candidate.drop_operations(&order, base, dropped);
        dropped_any |= dropped > 0;
    }

    if !fitted && candidate.record.input.is_some() {
        candidate.record.input = None;
        candidate.record.dropped_input = true;
        dropped_any = true;
        fitted = candidate.fits();
    }
    if !fitted && candidate.record.output.is_some() {
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
    use crate::record::ErrorRecord;
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
    fn repeated_names_near_the_limit_take_a_few_renders() {
        // The older half of the operations share one name. In the by-name
        // form, dropping one of them changes only that name's counters.
        let mut r = steps(2_000);
        for o in r.operations.iter_mut().take(1_000) {
            o.name = Some("same".to_owned());
        }
        for format in [
            OperationsFormat::ByName,
            OperationsFormat::Array,
            OperationsFormat::Both,
        ] {
            let count = renders(&r, size(&r, format) - 100, format);
            assert!(count < 40, "{format:?}: {count} renders");
        }
    }

    /// The truncation before the binary search: it checks the size before
    /// each dropped part, so it finds the fewest drops in every case.
    fn one_at_a_time(
        r: &TelemetryRecord,
        max_bytes: usize,
        format: OperationsFormat,
    ) -> TelemetryRecord {
        let mut candidate = Candidate {
            record: r.clone(),
            kept: vec![true; r.operations.len()],
            format,
            max_bytes,
        };
        candidate.record.truncated = true;
        let order = oldest_first(&r.operations);
        for &index in &order {
            if candidate.fits() {
                break;
            }
            if let Some(op) = candidate.record.operations.get_mut(index)
                && op.result.is_some()
            {
                op.result = None;
                op.truncated = true;
            }
        }
        for &index in &order {
            if candidate.fits() {
                break;
            }
            if let Some(kept) = candidate.kept.get_mut(index) {
                *kept = false;
                candidate.record.dropped_operations += 1;
            }
        }
        if !candidate.fits() && candidate.record.input.is_some() {
            candidate.record.input = None;
            candidate.record.dropped_input = true;
        }
        if !candidate.fits() && candidate.record.output.is_some() {
            candidate.record.output = None;
            candidate.record.dropped_output = true;
        }
        let mut kept = candidate.kept.into_iter();
        candidate
            .record
            .operations
            .retain(|_| kept.next().unwrap_or(true));
        candidate.record
    }

    /// A small deterministic random source (xorshift64).
    struct Random(u64);

    impl Random {
        fn below(&mut self, bound: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            usize::try_from(self.0).unwrap_or(0) % bound
        }

        fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
            items
                .get(self.below(items.len()))
                .copied()
                .unwrap_or_default()
        }

        fn text(&mut self, min: usize) -> String {
            const PARTS: [&str; 6] = ["a", "\"", "\\", "é", "😀", "\n"];
            let length = min + self.below(30);
            (0..length).map(|_| self.pick(&PARTS)).collect()
        }
    }

    /// A random record. Every result has at least `min_result` characters.
    fn random_record(random: &mut Random, min_result: usize) -> TelemetryRecord {
        const NAMES: [&str; 5] = ["", "same", "q\"uote", "back\\slash", "ünï"];
        let count = 1 + random.below(40);
        let operations = (0..count)
            .map(|i| {
                let name = match random.below(3) {
                    0 => random.pick(&NAMES).to_owned(),
                    1 => random.text(0),
                    _ => format!("step-{i}"),
                };
                let start = (random.below(5) > 0).then(|| random.below(20).try_into().unwrap_or(0));
                let mut o = timed(&name, start, None);
                o.id = format!("op-{i}");
                if random.below(2) == 0 {
                    o.result = Some(json!(random.text(min_result)));
                }
                if random.below(4) == 0 {
                    o.error = Some(ErrorRecord {
                        name: random.text(0),
                        message: random.text(0),
                    });
                }
                o
            })
            .collect();
        let mut r = record(operations);
        r.input = (random.below(3) > 0).then(|| json!(random.text(0)));
        r.output = (random.below(3) > 0).then(|| json!(random.text(0)));
        r
    }

    const FORMATS: [OperationsFormat; 3] = [
        OperationsFormat::Array,
        OperationsFormat::ByName,
        OperationsFormat::Both,
    ];

    #[test]
    fn random_records_stay_under_the_limit_and_drop_oldest_first() {
        let mut random = Random(0x5eed);
        for _ in 0..300 {
            let r = random_record(&mut random, 0);
            for format in FORMATS {
                let full = size(&r, format);
                let limit = 1 + random.below(full + 50);
                let out = truncate_record(&r, Some(limit), format);
                let nothing_left =
                    out.operations.is_empty() && out.input.is_none() && out.output.is_none();
                assert!(size(&out, format) <= limit || nothing_left);

                // The kept operations are the newest ones, in list order.
                let order = oldest_first(&r.operations);
                let dropped = out.dropped_operations;
                let mut kept: Vec<usize> = order.iter().copied().skip(dropped).collect();
                kept.sort_unstable();
                let ids: Vec<_> = out.operations.iter().map(|o| o.id.clone()).collect();
                let expected: Vec<_> = kept
                    .iter()
                    .filter_map(|&i| r.operations.get(i).map(|o| o.id.clone()))
                    .collect();
                assert_eq!(ids, expected);

                // The cleared results belong to the oldest operations.
                let cleared: Vec<bool> = order
                    .iter()
                    .filter_map(|&i| r.operations.get(i))
                    .filter(|o| o.result.is_some())
                    .map(|source| {
                        out.operations
                            .iter()
                            .find(|o| o.id == source.id)
                            .is_none_or(|o| o.result.is_none())
                    })
                    .collect();
                assert!(cleared.windows(2).all(|pair| pair != [false, true]));
            }
        }
    }

    #[test]
    fn the_array_form_drops_the_same_parts_as_one_at_a_time() {
        // In the array form, each dropped part makes the record smaller,
        // because every result is longer than `"truncated":true`. So the
        // binary search finds the fewest drops.
        let mut random = Random(0xa77a);
        for _ in 0..300 {
            let r = random_record(&mut random, 20);
            let format = OperationsFormat::Array;
            let limit = 1 + random.below(size(&r, format) + 50);
            let out = truncate_record(&r, Some(limit), format).into_owned();
            let expected = one_at_a_time(&r, limit, format);
            if expected.truncated && size(&r, format) > limit {
                assert_eq!(out, expected);
            }
        }
    }

    #[test]
    fn only_clearing_every_result_fits_and_no_operation_is_dropped() {
        let big = "x".repeat(200);
        let r = record(vec![
            timed("a", Some(10), Some(&big)),
            timed("b", Some(20), Some(&big)),
            timed("c", Some(30), Some(&big)),
        ]);
        let mut cleared = r.clone();
        cleared.truncated = true;
        for o in &mut cleared.operations {
            o.result = None;
            o.truncated = true;
        }
        let limit = size(&cleared, OperationsFormat::Array);
        let out = truncate_record(&r, Some(limit), OperationsFormat::Array).into_owned();
        assert_eq!(out, cleared);
    }

    #[test]
    fn only_dropping_every_operation_fits_and_the_input_is_kept() {
        let mut r = record(vec![
            timed("a", Some(10), None),
            timed("b", Some(20), None),
            timed("c", Some(30), None),
        ]);
        r.input = Some(json!("i".repeat(300)));
        let mut empty = r.clone();
        empty.operations.clear();
        empty.truncated = true;
        empty.dropped_operations = 3;
        let limit = size(&empty, OperationsFormat::Array);
        let out = truncate_record(&r, Some(limit), OperationsFormat::Array).into_owned();
        assert_eq!(out, empty);
        assert!(!out.dropped_input);
    }

    #[test]
    fn first_fit_measures_the_upper_bound() {
        assert_eq!(first_fit(1, 5, |count| count == 5), Some(5));
        assert_eq!(first_fit(1, 5, |_| false), None);
    }

    #[test]
    fn byte_counter_flush_is_a_no_op() {
        let mut counter = ByteCounter(3);
        assert!(io::Write::flush(&mut counter).is_ok());
        assert_eq!(counter.0, 3);
    }
}
