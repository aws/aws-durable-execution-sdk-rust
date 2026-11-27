//! Serializes exports and keeps only the latest record of each invocation.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::VecDeque;
use std::future::poll_fn;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::task::Poll;

use tokio::sync::oneshot;

use crate::contain::{contained, describe_error, log_contained};
use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::truncation::{Measure, truncate_measured};

/// Locks a standard mutex. A panic while the lock was held leaves plain
/// data that is still consistent, so a poisoned lock is used as is.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One invocation's place in the export queue.
///
/// A record is a complete snapshot of the execution. So a newer record
/// replaces a pending older one, and the older one is never exported.
#[derive(Debug, Default)]
pub(crate) struct Slot {
    state: Mutex<SlotState>,
}

#[derive(Debug, Default)]
struct SlotState {
    /// The latest record that the pump has not taken yet.
    pending: Option<TelemetryRecord>,
    /// True while the slot is in the scheduler's queue. So the queue holds
    /// a slot at most once.
    queued: bool,
    /// True while the slot has a record that is queued or being exported.
    outstanding: bool,
    /// The drains that wait for the slot's latest record.
    waiters: Vec<oneshot::Sender<()>>,
}

/// Ends one export of a slot when it is dropped.
///
/// Without a newer pending record, it clears `outstanding` and releases
/// the slot's drains. With one, the drains wait for that record.
///
/// It runs on drop, so it also runs when the pump is dropped during the
/// export. The record of that export is then lost. Without this, the slot
/// would stay outstanding with no record, and its drains would wait for
/// an export that no pump runs.
struct Settle<'a>(&'a Slot);

impl Drop for Settle<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.0.state);
        if state.pending.is_none() {
            state.outstanding = false;
            for waiter in state.waiters.drain(..) {
                // The drain may have been dropped. Nobody waits then.
                let _ = waiter.send(());
            }
        }
    }
}

/// The work that waits for the pump, and the pump's claim.
#[derive(Default)]
struct Queue {
    /// The slots that have a pending record, oldest first.
    slots: VecDeque<Arc<Slot>>,
    /// The flush requests that wait for the pump's next flush.
    flush_waiters: Vec<oneshot::Sender<()>>,
    /// The token of the pump. It is dead when no pump runs.
    pump: Weak<()>,
}

/// The pump holds this token. Dropping the token ends the claim. So a
/// pump that is dropped before it finishes, or that panics, does not keep
/// the scheduler claimed.
type PumpToken = Arc<()>;

/// Returns a token when no pump runs, and `None` when one does.
fn claim_pump(queue: &mut Queue) -> Option<PumpToken> {
    if queue.pump.strong_count() > 0 {
        return None;
    }
    let token = PumpToken::default();
    queue.pump = Arc::downgrade(&token);
    Some(token)
}

/// What the pump does in one turn.
enum Turn {
    /// Export the record of this slot, if any, then serve the flush
    /// requests.
    Work(Option<Arc<Slot>>),
    /// Nothing waits. The claim has ended.
    Idle,
}

/// The exporters of one plugin factory, and the pump that calls them.
///
/// Every invocation that the factory serves shares the scheduler. The
/// pump is one task that exports one record at a time and runs every
/// flush. So an exporter never sees two calls of the same factory at
/// once. The scheduler has at most one pump. So the number of export
/// tasks grows neither with the number of invocations nor with the number
/// of records they schedule.
///
/// The pump serves the slots oldest first. A slot that is scheduled again
/// goes to the back of the queue. So invocations take turns.
///
/// No invocation owns the pump's task. The SDK ends an invocation's hook
/// work when that invocation ends, and the pump also exports the records
/// of other invocations. So the pump runs on a task of its own.
pub(crate) struct Scheduler {
    exporters: Vec<Arc<dyn TelemetryExporter>>,
    queue: Mutex<Queue>,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("exporters", &self.exporters.len())
            .finish_non_exhaustive()
    }
}

/// Polls `future` to completion. A panic while polling or dropping the
/// future becomes an error.
async fn contained_future(future: impl Future<Output = ()>) -> Result<(), String> {
    let mut future = Some(Box::pin(future));
    poll_fn(|cx| {
        let polled = match future.as_mut() {
            Some(active) => contained(|| active.as_mut().poll(cx)),
            None => Err("polled after completion".to_owned()),
        };
        if matches!(polled, Ok(Poll::Pending)) {
            return Poll::Pending;
        }
        // A future that panicked is not polled again. Dropping it runs
        // the drop guards of the code that panicked.
        let dropped = contained(|| drop(future.take()));
        Poll::Ready(polled.and(dropped))
    })
    .await
}

/// Polls every future to completion and returns each outcome in order.
///
/// Every step that runs an exporter's code is contained: the poll, the
/// formatting of the error it returned, the drop of that error, and the
/// drop of the future. Each step has its own catch. A panic in the poll
/// or in the future's drop becomes that future's error. So the other
/// futures are still polled to completion.
async fn settle_all<'a>(futures: Vec<Result<ExportFuture<'a>, String>>) -> Vec<Result<(), String>> {
    let mut results: Vec<Option<Result<(), String>>> = Vec::with_capacity(futures.len());
    let mut running: Vec<Option<ExportFuture<'a>>> = Vec::with_capacity(futures.len());
    for future in futures {
        match future {
            Ok(future) => {
                running.push(Some(future));
                results.push(None);
            }
            Err(error) => {
                running.push(None);
                results.push(Some(Err(error)));
            }
        }
    }
    poll_fn(|cx| {
        let mut waiting = false;
        for (future, result) in running.iter_mut().zip(results.iter_mut()) {
            let Some(active) = future else { continue };
            // The catch covers the poll only. The error that the future
            // returned leaves the catch unformatted, and `describe_error`
            // formats it and drops it in two separate catches.
            let polled = contained(|| active.as_mut().poll(cx));
            let outcome = match polled {
                Ok(Poll::Pending) => {
                    waiting = true;
                    continue;
                }
                Ok(Poll::Ready(Ok(()))) => Ok(()),
                Ok(Poll::Ready(Err(error))) => Err(describe_error(error)),
                Err(error) => Err(error),
            };
            let dropped = contained(|| drop(future.take()));
            *result = Some(outcome.and(dropped));
        }
        if waiting {
            Poll::Pending
        } else {
            Poll::Ready(())
        }
    })
    .await;
    #[expect(
        clippy::expect_used,
        reason = "poll_fn returns Ready only after every future settled, and each settled future stored its result"
    )]
    results
        .into_iter()
        .map(|result| result.expect("every future settled"))
        .collect()
}

impl Scheduler {
    pub(crate) fn new(exporters: Vec<Arc<dyn TelemetryExporter>>) -> Self {
        Self {
            exporters,
            queue: Mutex::default(),
        }
    }

    /// Replaces the slot's pending record, and starts the pump unless one
    /// runs. It does not wait for the export.
    ///
    /// A record that is scheduled during an export adds no task and no
    /// queue entry: the slot is in the queue at most once.
    pub(crate) fn schedule(self: &Arc<Self>, slot: &Arc<Slot>, record: TelemetryRecord) {
        let enqueue = {
            let mut state = lock(&slot.state);
            state.pending = Some(record);
            state.outstanding = true;
            !std::mem::replace(&mut state.queued, true)
        };
        let mut queue = lock(&self.queue);
        if enqueue {
            queue.slots.push_back(Arc::clone(slot));
        }
        self.ensure_pump(&mut queue);
    }

    /// Ends an invocation's exports. Builds and schedules the final record
    /// when `build` is given. Then waits for the slot's drain and for a
    /// flush.
    ///
    /// `build` runs before this returns, not inside the returned future.
    /// A panic in `build` schedules nothing. The future still waits for
    /// the drain and the flush. So the records that were scheduled earlier
    /// are exported and flushed. The panic then becomes the future's
    /// error.
    pub(crate) fn finish<'a>(
        self: &'a Arc<Self>,
        slot: &'a Arc<Slot>,
        build: Option<impl FnOnce() -> TelemetryRecord>,
    ) -> impl Future<Output = Result<(), String>> + 'a {
        let scheduled = build.map_or(Ok(()), |build| {
            contained(build).map(|record| self.schedule(slot, record))
        });
        async move {
            self.drain(slot).await;
            self.flush().await;
            scheduled.map_err(|error| format!("telemetry record was not built: {error}"))
        }
    }

    /// Waits until the slot's latest record was handed to every exporter.
    /// Returns at once when the slot has no record to wait for.
    pub(crate) async fn drain(self: &Arc<Self>, slot: &Slot) {
        let exported = {
            let mut state = lock(&slot.state);
            if !state.outstanding {
                return;
            }
            let (waiter, exported) = oneshot::channel();
            state.waiters.push(waiter);
            exported
        };
        // A pump that was dropped left its queued slots without a pump.
        self.ensure_pump(&mut lock(&self.queue));
        // The slot keeps the sender until it releases it. So this never
        // fails.
        let _ = exported.await;
    }

    /// Flushes every exporter. The pump runs the flush, so it overlaps no
    /// export.
    ///
    /// The requests that wait when the pump starts a flush share that
    /// flush. A request that arrives during a flush waits for the next
    /// one: its record may have reached the exporters after the flush
    /// read their buffers.
    pub(crate) async fn flush(self: &Arc<Self>) {
        let (waiter, flushed) = oneshot::channel();
        {
            let mut queue = lock(&self.queue);
            queue.flush_waiters.push(waiter);
            self.ensure_pump(&mut queue);
        }
        // An error means the pump was dropped after it took this request.
        // The flush did not run, which the caller treats like a failed
        // flush.
        let _ = flushed.await;
    }

    /// Starts the pump on its own task unless one runs. Outside a tokio
    /// runtime it starts nothing, and the work waits for the next call
    /// inside one.
    #[expect(
        clippy::disallowed_methods,
        reason = "the SDK's task gateway is private to the SDK crate, and its hook work belongs to one invocation; the pump serves every invocation and creates no durable operation"
    )]
    fn ensure_pump(self: &Arc<Self>, queue: &mut Queue) {
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        if let Some(token) = claim_pump(queue) {
            drop(tokio::spawn(Arc::clone(self).run_pump(token)));
        }
    }

    /// Takes the pump's next work. Without work, it ends the pump's claim.
    ///
    /// The check and the end of the claim hold one lock. Work that
    /// arrives later then finds the scheduler unclaimed, and its caller
    /// starts a new pump. If the claim ended only when the pump dropped
    /// its token, work that arrives between the check and the drop would
    /// find the scheduler claimed, and no pump would serve it.
    fn begin_turn(&self) -> Turn {
        let mut queue = lock(&self.queue);
        if queue.slots.is_empty() && queue.flush_waiters.is_empty() {
            queue.pump = Weak::new();
            return Turn::Idle;
        }
        Turn::Work(queue.slots.pop_front())
    }

    /// Runs one turn after another until nothing waits. The future owns
    /// the token, so dropping the future ends the claim.
    ///
    /// A panic in a turn ends that turn only, and the pump takes the next
    /// turn. The exporters' code is already contained, so such a panic
    /// comes from elsewhere, such as the application's log subscriber.
    /// If the panic ended the pump, the drains that wait for the queued
    /// slots would wait until a later call starts a new pump.
    ///
    /// The loop still ends after such a panic. `begin_turn` removed the
    /// turn's slot from the queue, and the turn takes the flush requests
    /// before it calls other code. So a turn that panics leaves less work
    /// than it found. Dropping the turn releases the slot's drains and
    /// drops the flush requests it took, which those requesters treat like
    /// a failed flush.
    async fn run_pump(self: Arc<Self>, _token: PumpToken) {
        while let Turn::Work(slot) = self.begin_turn() {
            if let Err(error) = contained_future(self.run_turn(slot)).await {
                // The log calls the application's subscriber, which may be
                // the code that panicked.
                log_contained(|| {
                    tracing::warn!(target: crate::TARGET, %error, "telemetry pump turn failed");
                });
            }
        }
    }

    /// Exports the record of `slot`, if any, then serves the flush
    /// requests.
    ///
    /// A flush therefore waits for at most one export that no drain waits
    /// for, however many records are queued.
    async fn run_turn(&self, slot: Option<Arc<Slot>>) {
        if let Some(slot) = slot {
            self.export_slot(&slot).await;
        }
        let mut flushers = std::mem::take(&mut lock(&self.queue).flush_waiters);
        if flushers.is_empty() {
            return;
        }
        // A drain that waits belongs to an invocation end, which asks
        // for its flush only after the drain returns. So these records
        // are exported before the flush, and the ends that have asked
        // by then share it. The count bounds the pass: a slot that is
        // scheduled again during the pass cannot hold the flush back.
        for _ in 0..self.awaited_slots() {
            let Some(slot) = self.pop_awaited_slot() else {
                break;
            };
            self.export_slot(&slot).await;
        }
        flushers.append(&mut lock(&self.queue).flush_waiters);
        self.flush_all().await;
        for flusher in flushers {
            // The requester may have been dropped. Nobody waits then.
            let _ = flusher.send(());
        }
    }

    /// The number of queued slots that a drain waits for.
    fn awaited_slots(&self) -> usize {
        let queue = lock(&self.queue);
        queue
            .slots
            .iter()
            .filter(|slot| !lock(&slot.state).waiters.is_empty())
            .count()
    }

    /// Removes the oldest queued slot that a drain waits for.
    ///
    /// The pump removes one slot at a time and exports it at once. So a
    /// pump that is dropped leaves every other slot in the queue for the
    /// next pump.
    fn pop_awaited_slot(&self) -> Option<Arc<Slot>> {
        let mut queue = lock(&self.queue);
        let index = queue
            .slots
            .iter()
            .position(|slot| !lock(&slot.state).waiters.is_empty())?;
        queue.slots.remove(index)
    }

    /// Exports the pending record of a slot that the pump removed from
    /// the queue, then releases the slot's drains.
    async fn export_slot(&self, slot: &Slot) {
        let record = {
            let mut state = lock(&slot.state);
            // From here, a record that is scheduled puts the slot back in
            // the queue.
            state.queued = false;
            state.pending.take()
        };
        let _settle = Settle(slot);
        if let Some(record) = record {
            self.export(&record).await;
        }
    }

    /// Hands `record` to every exporter. Only the pump calls it.
    ///
    /// An exporter whose limit query panics gets no `export` call: the
    /// size it accepts is unknown.
    async fn export(&self, record: &TelemetryRecord) {
        let records: Vec<Result<Cow<'_, TelemetryRecord>, String>> = self
            .exporters
            .iter()
            .enumerate()
            .map(|(index, exporter)| {
                let max_bytes = contained(|| exporter.max_record_size_bytes())?;
                let panicked = Cell::new(false);
                let text = |record: &TelemetryRecord| {
                    // A panic in the measure marks the exporter as failed.
                    // An error means the record cannot be measured.
                    match contained(|| exporter.measured_text(record)) {
                        Ok(Ok(text)) => Some(text.len()),
                        Ok(Err(error)) => {
                            let error = describe_error(error);
                            log_contained(|| {
                                tracing::warn!(target: crate::TARGET, exporter = index, %error, "telemetry record could not be measured");
                            });
                            None
                        }
                        Err(_) => {
                            panicked.set(true);
                            None
                        }
                    }
                };
                let truncated = truncate_measured(record, max_bytes, Measure::Text(&text));
                if panicked.get() {
                    Err("panicked".to_owned())
                } else {
                    Ok(truncated)
                }
            })
            .collect();
        let futures = self
            .exporters
            .iter()
            .zip(&records)
            .map(|(exporter, record)| match record {
                Ok(record) => contained(|| exporter.export(record)),
                Err(error) => Err(error.clone()),
            })
            .collect();
        for (index, result) in settle_all(futures).await.into_iter().enumerate() {
            if let Err(error) = result {
                log_contained(|| {
                    tracing::warn!(target: crate::TARGET, exporter = index, %error, "telemetry export failed");
                });
            }
        }
    }

    /// Flushes every exporter. Only the pump calls it.
    async fn flush_all(&self) {
        let futures = self
            .exporters
            .iter()
            .map(|exporter| contained(|| exporter.flush()))
            .collect();
        for (index, result) in settle_all(futures).await.into_iter().enumerate() {
            if let Err(error) = result {
                log_contained(|| {
                    tracing::warn!(target: crate::TARGET, exporter = index, %error, "telemetry flush failed");
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contain::tests::{
        PanicsOnDisplay, PanicsOnDisplayAndDrop, PanicsOnDrop, PanicsOnEvent,
    };
    use crate::render::OperationsFormat;
    use crate::render::tests::{op, record};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct Recorder {
        exported: Mutex<Vec<String>>,
        flushes: AtomicUsize,
        limit: Option<usize>,
    }

    impl TelemetryExporter for Recorder {
        fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(async move {
                lock(&self.exported).push(record.emitted_at.clone());
                Ok(())
            })
        }

        fn flush(&self) -> ExportFuture<'_> {
            self.flushes.fetch_add(1, Ordering::SeqCst);
            Box::pin(std::future::ready(Ok(())))
        }

        fn max_record_size_bytes(&self) -> Option<usize> {
            self.limit
        }
    }

    struct Failing;

    impl TelemetryExporter for Failing {
        fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(std::future::ready(Err("export failed".into())))
        }

        fn flush(&self) -> ExportFuture<'_> {
            Box::pin(std::future::ready(Err("flush failed".into())))
        }
    }

    struct PanicsOnCall;

    #[expect(
        clippy::panic,
        reason = "the tests check that a panicking exporter is contained"
    )]
    impl TelemetryExporter for PanicsOnCall {
        fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
            panic!("export panicked")
        }

        fn flush(&self) -> ExportFuture<'_> {
            Box::pin(async { panic!("flush panicked") })
        }
    }

    /// An exporter whose every step fails in a way that runs its code after
    /// the poll returned: `export` returns an error whose formatting
    /// panics, and `flush` returns an error whose drop panics.
    struct FailsWithPanickingError;

    impl TelemetryExporter for FailsWithPanickingError {
        fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(std::future::ready(Err(PanicsOnDisplay.into())))
        }

        fn flush(&self) -> ExportFuture<'_> {
            Box::pin(std::future::ready(Err(PanicsOnDrop.into())))
        }
    }

    /// An exporter whose `export` and `flush` both return an error whose
    /// formatting panics and whose drop panics.
    struct FailsWithDoublyPanickingError;

    impl TelemetryExporter for FailsWithDoublyPanickingError {
        fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(std::future::ready(Err(PanicsOnDisplayAndDrop.into())))
        }

        fn flush(&self) -> ExportFuture<'_> {
            Box::pin(std::future::ready(Err(PanicsOnDisplayAndDrop.into())))
        }
    }

    /// An exporter whose futures panic when they are dropped.
    struct ReturnsFuturesThatPanicOnDrop;

    impl TelemetryExporter for ReturnsFuturesThatPanicOnDrop {
        fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(PanicsOnDrop)
        }

        fn flush(&self) -> ExportFuture<'_> {
            Box::pin(PanicsOnDrop)
        }
    }

    /// An exporter whose size limit query panics.
    struct PanicsOnLimit;

    #[expect(
        clippy::panic,
        reason = "the tests check that a panicking exporter is contained"
    )]
    impl TelemetryExporter for PanicsOnLimit {
        fn export<'a>(&'a self, _record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn max_record_size_bytes(&self) -> Option<usize> {
            panic!("limit panicked")
        }
    }

    /// A waker that panics when it is woken.
    struct PanicsOnWake;

    impl std::task::Wake for PanicsOnWake {
        #[expect(
            clippy::panic,
            reason = "the test checks that a panic in a turn is contained"
        )]
        fn wake(self: Arc<Self>) {
            panic!("wake panicked")
        }
    }

    /// Schedules one record on each of two slots, then waits for both
    /// drains at once and for a flush. Both drains are registered before
    /// the pump's first turn. So the second drain completes only when the
    /// pump that exported the first record also exports the second.
    async fn export_two_slots(scheduler: &Arc<Scheduler>) {
        let (a, b) = (Arc::new(Slot::default()), Arc::new(Slot::default()));
        scheduler.schedule(&a, stamped("a"));
        scheduler.schedule(&b, stamped("b"));
        let work = async {
            tokio::join!(scheduler.drain(&a), scheduler.drain(&b));
            scheduler.flush().await;
        };
        let finished = tokio::time::timeout(std::time::Duration::from_secs(5), work).await;
        assert!(finished.is_ok(), "the drains and the flush completed");
    }

    /// An exporter whose every export waits for one permit of `gate`.
    struct Gated {
        recorder: Recorder,
        gate: tokio::sync::Semaphore,
    }

    impl Gated {
        fn closed() -> Arc<Self> {
            Arc::new(Self {
                recorder: Recorder::default(),
                gate: tokio::sync::Semaphore::new(0),
            })
        }
    }

    impl TelemetryExporter for Gated {
        fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
            Box::pin(async move {
                self.gate.acquire().await?.forget();
                self.recorder.export(record).await
            })
        }

        fn flush(&self) -> ExportFuture<'_> {
            self.recorder.flush()
        }
    }

    fn stamped(emitted_at: &str) -> TelemetryRecord {
        let mut r = record(Vec::new());
        r.emitted_at = emitted_at.to_owned();
        r
    }

    fn scheduler_of(exporter: Arc<dyn TelemetryExporter>) -> Arc<Scheduler> {
        Arc::new(Scheduler::new(vec![exporter]))
    }

    #[expect(clippy::expect_used, reason = "the test needs a runtime")]
    fn current_thread_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime")
    }

    fn alive_tasks() -> usize {
        tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks()
    }

    #[tokio::test]
    async fn a_newer_record_replaces_a_pending_one() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, stamped("1"));
        scheduler.schedule(&slot, stamped("2"));
        scheduler.drain(&slot).await;
        assert_eq!(*lock(&recorder.exported), ["2"]);
        assert!(format!("{scheduler:?}").contains("exporters: 1"));
    }

    #[tokio::test]
    async fn schedule_exports_without_being_awaited() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, stamped("1"));
        assert!(lock(&recorder.exported).is_empty());
        tokio::task::yield_now().await;
        assert_eq!(*lock(&recorder.exported), ["1"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn drain_without_a_record_returns_and_flush_still_flushes() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        scheduler.drain(&Slot::default()).await;
        scheduler.flush().await;
        assert!(lock(&recorder.exported).is_empty());
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failing_or_panicking_exporter_does_not_stop_the_others() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![
            Arc::new(Failing),
            Arc::new(PanicsOnCall),
            recorder.clone(),
        ]));
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, stamped("1"));
        scheduler.drain(&slot).await;
        scheduler.flush().await;
        assert_eq!(*lock(&recorder.exported), ["1"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_error_whose_formatting_or_drop_panics_stops_neither_the_other_exporter_nor_the_pump()
     {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![
            Arc::new(FailsWithPanickingError),
            recorder.clone(),
        ]));
        export_two_slots(&scheduler).await;
        assert_eq!(*lock(&recorder.exported), ["a", "b"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_error_whose_formatting_and_drop_panic_stops_neither_the_other_exporter_nor_the_pump()
     {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![
            Arc::new(FailsWithDoublyPanickingError),
            recorder.clone(),
        ]));
        export_two_slots(&scheduler).await;
        assert_eq!(*lock(&recorder.exported), ["a", "b"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_subscriber_that_panics_on_a_failure_log_ends_no_turn() {
        let events = Arc::new(AtomicUsize::new(0));
        let _subscriber = test_tracing::install_test_subscriber(PanicsOnEvent(events.clone()));
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![Arc::new(Failing), recorder.clone()]));
        export_two_slots(&scheduler).await;
        assert_eq!(*lock(&recorder.exported), ["a", "b"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
        // Two failed exports and one failed flush. A turn that the panic
        // ended would add the pump's own event.
        assert_eq!(events.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn a_future_whose_drop_panics_stops_neither_the_other_exporter_nor_the_pump() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![
            Arc::new(ReturnsFuturesThatPanicOnDrop),
            recorder.clone(),
        ]));
        export_two_slots(&scheduler).await;
        assert_eq!(*lock(&recorder.exported), ["a", "b"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_limit_query_that_panics_stops_neither_the_other_exporter_nor_the_pump() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![
            Arc::new(PanicsOnLimit),
            recorder.clone(),
        ]));
        export_two_slots(&scheduler).await;
        assert_eq!(*lock(&recorder.exported), ["a", "b"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_turn_that_panics_outside_the_exporters_does_not_stop_the_pump() {
        // The subscriber panics on the pump's log of the failed turn.
        let events = Arc::new(AtomicUsize::new(0));
        let _subscriber = test_tracing::install_test_subscriber(PanicsOnEvent(events.clone()));
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        let (a, b) = (Arc::new(Slot::default()), Arc::new(Slot::default()));
        scheduler.schedule(&a, stamped("a"));
        scheduler.schedule(&b, stamped("b"));
        // The pump wakes this drain after it exported record `a`. The
        // waker panics, which is a panic in the turn and outside every
        // exporter.
        let waker = std::task::Waker::from(Arc::new(PanicsOnWake));
        let mut drain_a = Box::pin(scheduler.drain(&a));
        let polled = drain_a
            .as_mut()
            .poll(&mut std::task::Context::from_waker(&waker));
        assert!(polled.is_pending());
        // This drain and this flush are requested before the pump's first
        // turn, and no call after the panic starts a pump. So they complete
        // only when the pump that panicked also takes the next turn.
        let waits = async { tokio::join!(scheduler.drain(&b), scheduler.flush()) };
        let finished = tokio::time::timeout(std::time::Duration::from_secs(5), waits).await;
        assert!(finished.is_ok(), "the second drain and the flush completed");
        assert_eq!(*lock(&recorder.exported), ["a", "b"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
        assert_eq!(events.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    #[expect(
        clippy::panic,
        reason = "the test checks that a panicking build is contained"
    )]
    async fn a_final_build_that_panics_keeps_the_earlier_record_and_the_flush() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, stamped("a"));
        let finished = scheduler
            .finish(&slot, Some(|| panic!("build panicked")))
            .await;
        assert_eq!(
            finished,
            Err("telemetry record was not built: panicked".to_owned())
        );
        assert_eq!(*lock(&recorder.exported), ["a"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn each_exporter_gets_its_own_truncation() {
        struct Sizes(Mutex<Vec<bool>>, Option<usize>);
        impl TelemetryExporter for Sizes {
            fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
                lock(&self.0).push(record.truncated);
                Box::pin(std::future::ready(Ok(())))
            }
            fn max_record_size_bytes(&self) -> Option<usize> {
                self.1
            }
            fn operations_format(&self) -> OperationsFormat {
                OperationsFormat::ByName
            }
        }
        let small = Arc::new(Sizes(Mutex::default(), Some(10)));
        let large = Arc::new(Sizes(Mutex::default(), None));
        let scheduler = Arc::new(Scheduler::new(vec![small.clone(), large.clone()]));
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, record(Vec::new()));
        scheduler.drain(&slot).await;
        assert_eq!(*lock(&small.0), [true]);
        assert_eq!(*lock(&large.0), [false]);
    }

    type MeasureFn = fn(&TelemetryRecord) -> Result<String, aws_durable_execution_sdk::BoxError>;

    /// An exporter whose size limit is 100 bytes and whose measured text
    /// comes from `measure`. It records each exported record's
    /// `truncated` flag and its operation count.
    struct Measured {
        exported: Mutex<Vec<(bool, usize)>>,
        measure: MeasureFn,
    }

    impl Measured {
        fn new(measure: MeasureFn) -> Arc<Self> {
            Arc::new(Self {
                exported: Mutex::default(),
                measure,
            })
        }
    }

    impl TelemetryExporter for Measured {
        fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
            lock(&self.exported).push((record.truncated, record.operations.len()));
            Box::pin(std::future::ready(Ok(())))
        }

        fn max_record_size_bytes(&self) -> Option<usize> {
            Some(100)
        }

        fn measured_text(
            &self,
            record: &TelemetryRecord,
        ) -> Result<String, aws_durable_execution_sdk::BoxError> {
            (self.measure)(record)
        }
    }

    async fn export_one(exporter: Arc<dyn TelemetryExporter>, record: TelemetryRecord) {
        let scheduler = scheduler_of(exporter);
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, record);
        scheduler.drain(&slot).await;
    }

    #[tokio::test]
    async fn the_limit_is_measured_on_the_text_the_exporter_returns() {
        // The record's own JSON is far over 100 bytes. The measured text
        // is one byte per operation. So the record fits, and the same
        // record with 101 operations does not.
        let exporter = Measured::new(|record| Ok("x".repeat(record.operations.len())));
        let ops = |count: usize| {
            (0..count)
                .map(|n| op(&format!("op{n}"), "SUCCEEDED"))
                .collect()
        };
        export_one(exporter.clone(), record(ops(100))).await;
        export_one(exporter.clone(), record(ops(101))).await;
        // The second record drops its oldest operation to reach 100 bytes.
        assert_eq!(*lock(&exporter.exported), [(false, 100), (true, 100)]);
    }

    #[tokio::test]
    async fn a_default_exporter_is_measured_on_its_operations_format_json() {
        struct Default100;
        impl TelemetryExporter for Default100 {
            fn export<'a>(&'a self, record: &'a TelemetryRecord) -> ExportFuture<'a> {
                assert!(record.truncated);
                Box::pin(std::future::ready(Ok(())))
            }
            fn max_record_size_bytes(&self) -> Option<usize> {
                Some(100)
            }
        }
        let expected = crate::render::render_json(&record(Vec::new()), OperationsFormat::Array)
            .map(|json| json.len());
        assert!(expected.is_ok_and(|size| size > 100));
        export_one(Arc::new(Default100), record(Vec::new())).await;
    }

    #[tokio::test]
    async fn a_record_that_cannot_be_measured_is_exported_unchanged() {
        let exporter = Measured::new(|_| Err("no text".into()));
        let ops = (0..5).map(|n| op(&format!("op{n}"), "SUCCEEDED")).collect();
        export_one(exporter.clone(), record(ops)).await;
        assert_eq!(*lock(&exporter.exported), [(false, 5)]);
    }

    #[tokio::test]
    async fn a_measure_that_panics_skips_only_that_exporter() {
        #[expect(
            clippy::panic,
            reason = "the test checks that a panicking measure is contained"
        )]
        fn panics(_: &TelemetryRecord) -> Result<String, aws_durable_execution_sdk::BoxError> {
            panic!("measure panicked")
        }
        let panics = Measured::new(panics);
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![panics.clone(), recorder.clone()]));
        export_two_slots(&scheduler).await;
        assert_eq!(*lock(&panics.exported), []);
        assert_eq!(*lock(&recorder.exported), ["a", "b"]);
    }

    #[tokio::test]
    async fn records_scheduled_during_an_export_start_no_task() {
        let gated = Gated::closed();
        let scheduler = scheduler_of(gated.clone());
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, stamped("0"));
        // The pump takes record 0 and waits for the gate.
        tokio::task::yield_now().await;
        assert_eq!(alive_tasks(), 1);
        for n in 1..=500 {
            scheduler.schedule(&slot, stamped(&n.to_string()));
            assert_eq!(alive_tasks(), 1, "record {n}");
        }
        gated.gate.add_permits(2);
        scheduler.drain(&slot).await;
        assert_eq!(*lock(&gated.recorder.exported), ["0", "500"]);
    }

    #[tokio::test]
    async fn slots_are_exported_oldest_first_and_a_rescheduled_slot_goes_last() {
        let gated = Gated::closed();
        let scheduler = scheduler_of(gated.clone());
        let (a, b, c) = (
            Arc::new(Slot::default()),
            Arc::new(Slot::default()),
            Arc::new(Slot::default()),
        );
        scheduler.schedule(&a, stamped("a1"));
        // The pump takes a1 and waits for the gate.
        tokio::task::yield_now().await;
        scheduler.schedule(&b, stamped("b1"));
        scheduler.schedule(&c, stamped("c1"));
        scheduler.schedule(&a, stamped("a2"));
        // b keeps its place in the queue, and b2 replaces b1.
        scheduler.schedule(&b, stamped("b2"));
        gated.gate.add_permits(4);
        scheduler.drain(&a).await;
        assert_eq!(*lock(&gated.recorder.exported), ["a1", "b2", "c1", "a2"]);
    }

    #[tokio::test]
    async fn a_flush_first_exports_the_record_that_a_drain_waits_for() {
        let gated = Gated::closed();
        let scheduler = scheduler_of(gated.clone());
        let (first, unawaited, awaited) = (
            Arc::new(Slot::default()),
            Arc::new(Slot::default()),
            Arc::new(Slot::default()),
        );
        scheduler.schedule(&first, stamped("first"));
        // The pump takes the first record and waits for the gate.
        tokio::task::yield_now().await;
        scheduler.schedule(&unawaited, stamped("unawaited"));
        scheduler.schedule(&awaited, stamped("awaited"));
        // `join!` polls in order. So the drain and the flush request both
        // wait before the gate opens.
        tokio::join!(scheduler.drain(&awaited), scheduler.flush(), async {
            gated.gate.add_permits(3);
        });
        scheduler.drain(&unawaited).await;
        assert_eq!(
            *lock(&gated.recorder.exported),
            ["first", "awaited", "unawaited"]
        );
        assert_eq!(gated.recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_record_scheduled_after_the_pump_ended_starts_a_new_pump() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, stamped("1"));
        scheduler.drain(&slot).await;
        tokio::task::yield_now().await;
        assert_eq!(alive_tasks(), 0);
        scheduler.schedule(&slot, stamped("2"));
        scheduler.drain(&slot).await;
        assert_eq!(*lock(&recorder.exported), ["1", "2"]);
    }

    #[test]
    fn an_idle_turn_ends_the_claim_while_the_pump_still_holds_its_token() {
        let scheduler = Scheduler::new(Vec::new());
        let token = claim_pump(&mut lock(&scheduler.queue));
        assert!(token.is_some());
        assert!(claim_pump(&mut lock(&scheduler.queue)).is_none());
        assert!(matches!(scheduler.begin_turn(), Turn::Idle));
        assert!(claim_pump(&mut lock(&scheduler.queue)).is_some());
        drop(token);
    }

    #[test]
    fn a_pump_dropped_during_an_export_releases_the_drain_and_the_claim() {
        let gated = Gated::closed();
        let scheduler = scheduler_of(gated.clone());
        let slot = Arc::new(Slot::default());
        let mut drain = Box::pin(scheduler.drain(&slot));
        let first = current_thread_runtime();
        first.block_on(async {
            scheduler.schedule(&slot, stamped("1"));
            // The pump takes record 1 and waits for the gate.
            tokio::task::yield_now().await;
            let waits = poll_fn(|cx| Poll::Ready(drain.as_mut().poll(cx).is_pending())).await;
            assert!(waits);
        });
        // Dropping the runtime drops the pump during the export.
        drop(first);
        current_thread_runtime().block_on(async {
            drain.await;
            gated.gate.add_permits(1);
            scheduler.schedule(&slot, stamped("2"));
            scheduler.drain(&slot).await;
        });
        assert_eq!(*lock(&gated.recorder.exported), ["2"]);
    }

    #[test]
    fn a_pump_dropped_before_it_ran_leaves_the_record_for_the_next_pump() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        let slot = Arc::new(Slot::default());
        let first = current_thread_runtime();
        first.block_on(async { scheduler.schedule(&slot, stamped("1")) });
        drop(first);
        assert!(lock(&recorder.exported).is_empty());
        current_thread_runtime().block_on(scheduler.drain(&slot));
        assert_eq!(*lock(&recorder.exported), ["1"]);
    }

    #[test]
    fn a_record_scheduled_outside_a_runtime_waits_for_the_next_call_inside_one() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = scheduler_of(recorder.clone());
        let slot = Arc::new(Slot::default());
        scheduler.schedule(&slot, stamped("1"));
        current_thread_runtime().block_on(scheduler.drain(&slot));
        assert_eq!(*lock(&recorder.exported), ["1"]);
    }

    #[tokio::test]
    async fn settle_all_waits_for_a_pending_future() {
        let (tx, rx) = oneshot::channel::<()>();
        let waiting: ExportFuture<'static> = Box::pin(async move {
            rx.await.map_err(|e| e.to_string())?;
            Ok(())
        });
        let settle = settle_all(vec![Ok(waiting)]);
        let sender = async move {
            tokio::task::yield_now().await;
            let _ = tx.send(());
        };
        let (results, ()) = tokio::join!(settle, sender);
        assert_eq!(results, [Ok(())]);
    }

    #[test]
    fn a_poisoned_lock_is_still_usable() {
        let mutex = Arc::new(Mutex::new(1));
        let poisoner = Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock();
            #[expect(clippy::panic, reason = "the test poisons the lock")]
            {
                panic!("poison");
            }
        })
        .join();
        assert_eq!(*lock(&mutex), 1);
    }
}
