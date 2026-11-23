//! Serializes exports and keeps only the latest record of each invocation.

use std::borrow::Cow;
use std::future::{Future, poll_fn};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::task::Poll;

use crate::exporter::{ExportFuture, TelemetryExporter};
use crate::record::TelemetryRecord;
use crate::truncation::truncate_record;

/// Locks a standard mutex. A panic while the lock was held leaves plain
/// data that is still consistent, so a poisoned lock is used as is.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One invocation's place in the export queue.
///
/// A record is a complete snapshot of the execution. So a newer record
/// replaces a pending older one, and the older one is never exported.
///
/// A pump is a future that exports the slot's pending records until the
/// slot is empty. A slot has at most one pump at a time. So the number of
/// export tasks of an invocation does not grow with the number of records
/// it schedules.
#[derive(Debug, Default)]
pub(crate) struct Slot {
    state: Mutex<SlotState>,
}

#[derive(Debug, Default)]
struct SlotState {
    pending: Option<TelemetryRecord>,
    /// The token of the slot's pump. It is dead when the slot has no pump.
    pump: Weak<()>,
}

/// The pump of a slot holds this token. Dropping the token ends the
/// slot's claim. So a pump that is dropped before it finishes, or that
/// panics, does not keep the slot claimed.
type PumpToken = Arc<()>;

impl Slot {
    /// Replaces the pending record.
    pub(crate) fn schedule(&self, record: TelemetryRecord) {
        lock(&self.state).pending = Some(record);
    }

    fn take(&self) -> Option<TelemetryRecord> {
        lock(&self.state).pending.take()
    }

    /// Returns a token when the slot has no pump, and `None` when it has
    /// one.
    fn claim_pump(&self) -> Option<PumpToken> {
        let mut state = lock(&self.state);
        if state.pump.strong_count() > 0 {
            return None;
        }
        let token = PumpToken::default();
        state.pump = Arc::downgrade(&token);
        Some(token)
    }

    /// Takes the pending record for the slot's pump. Without a pending
    /// record, it ends the pump's claim and returns `None`.
    ///
    /// The check and the end of the claim hold one lock. A record that is
    /// scheduled later then finds the slot unclaimed, and its caller
    /// starts a new pump. If the claim ended only when the pump dropped
    /// its token, a record scheduled between the check and the drop would
    /// find the slot claimed, and no pump would export it.
    fn take_for_pump(&self) -> Option<TelemetryRecord> {
        let mut state = lock(&self.state);
        let record = state.pending.take();
        if record.is_none() {
            state.pump = Weak::new();
        }
        record
    }
}

/// The exporters of one plugin factory, and the lock that serializes them.
///
/// Every invocation that the factory serves shares the scheduler. Each
/// export and each flush holds the lock. So an exporter never sees two
/// calls of the same factory at once. The lock is fair, so invocations
/// take turns in the order they asked.
pub(crate) struct Scheduler {
    exporters: Vec<Arc<dyn TelemetryExporter>>,
    turn: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("exporters", &self.exporters.len())
            .finish_non_exhaustive()
    }
}

/// Polls every future to completion and returns each outcome in order. A
/// panic while creating or polling a future becomes that future's error.
async fn settle_all<'a>(
    futures: Vec<std::thread::Result<ExportFuture<'a>>>,
) -> Vec<Result<(), String>> {
    let mut results: Vec<Option<Result<(), String>>> = Vec::with_capacity(futures.len());
    let mut running: Vec<Option<ExportFuture<'a>>> = Vec::with_capacity(futures.len());
    for future in futures {
        if let Ok(future) = future {
            running.push(Some(future));
            results.push(None);
        } else {
            running.push(None);
            results.push(Some(Err("panicked".to_owned())));
        }
    }
    poll_fn(|cx| {
        let mut waiting = false;
        for (future, result) in running.iter_mut().zip(results.iter_mut()) {
            let Some(active) = future else { continue };
            match catch_unwind(AssertUnwindSafe(|| active.as_mut().poll(cx))) {
                Ok(Poll::Pending) => waiting = true,
                Ok(Poll::Ready(outcome)) => {
                    *result = Some(outcome.map_err(|error| error.to_string()));
                    *future = None;
                }
                Err(_) => {
                    *result = Some(Err("panicked".to_owned()));
                    *future = None;
                }
            }
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
            turn: tokio::sync::Mutex::new(()),
        }
    }

    /// Exports the slot's pending record, if it still has one when its
    /// turn comes. Another task may have exported it in the meantime.
    pub(crate) async fn export_pending(&self, slot: &Slot) {
        let _turn = self.turn.lock().await;
        if let Some(record) = slot.take() {
            self.export(&record).await;
        }
    }

    /// Hands `record` to every exporter. The caller holds the lock.
    async fn export(&self, record: &TelemetryRecord) {
        let records: Vec<Cow<'_, TelemetryRecord>> = self
            .exporters
            .iter()
            .map(|exporter| {
                truncate_record(
                    record,
                    exporter.max_record_size_bytes(),
                    exporter.operations_format(),
                )
            })
            .collect();
        let futures = self
            .exporters
            .iter()
            .zip(&records)
            .map(|(exporter, record)| catch_unwind(AssertUnwindSafe(|| exporter.export(record))))
            .collect();
        for (index, result) in settle_all(futures).await.into_iter().enumerate() {
            if let Err(error) = result {
                tracing::warn!(target: crate::TARGET, exporter = index, %error, "telemetry export failed");
            }
        }
    }

    /// Flushes every exporter.
    pub(crate) async fn flush(&self) {
        let _turn = self.turn.lock().await;
        let futures = self
            .exporters
            .iter()
            .map(|exporter| catch_unwind(AssertUnwindSafe(|| exporter.flush())))
            .collect();
        for (index, result) in settle_all(futures).await.into_iter().enumerate() {
            if let Err(error) = result {
                tracing::warn!(target: crate::TARGET, exporter = index, %error, "telemetry flush failed");
            }
        }
    }

    /// Exports the slot's pending record, then flushes every exporter.
    ///
    /// The SDK delivers every operation change, and waits for its work,
    /// before `on_invocation_end`. So no record is scheduled on the slot
    /// while this runs, and one round exports the last record.
    pub(crate) async fn drain(&self, slot: &Slot) {
        self.export_pending(slot).await;
        self.flush().await;
    }

    /// The pump of `slot`: a `'static` future that exports the slot's
    /// pending record, then every record that is scheduled before the slot
    /// is empty. Returns `None` when the slot already has a pump, which
    /// then exports the pending record.
    ///
    /// The pump releases the lock after each record. So other invocations
    /// and [`Self::drain`] take their turn between two records.
    pub(crate) fn pump(
        self: &Arc<Self>,
        slot: &Arc<Slot>,
    ) -> Option<impl Future<Output = ()> + Send + 'static> {
        let token = slot.claim_pump()?;
        Some(Arc::clone(self).run_pump(Arc::clone(slot), token))
    }

    /// Exports the slot's records until the slot is empty. The future
    /// owns the token, so dropping the future ends the slot's claim.
    async fn run_pump(self: Arc<Self>, slot: Arc<Slot>, _token: PumpToken) {
        loop {
            let _turn = self.turn.lock().await;
            let Some(record) = slot.take_for_pump() else {
                return;
            };
            self.export(&record).await;
        }
    }

    /// Starts the slot's pump on its own task, unless the slot already has
    /// a pump, and does not wait for it.
    ///
    /// The pump takes the same lock as every other export. So
    /// [`Self::drain`] waits for an export that the pump has started.
    #[expect(
        clippy::disallowed_methods,
        reason = "the SDK's task gateway is private to the SDK crate; this task creates no durable operation, and drain waits for its export through the lock"
    )]
    pub(crate) fn spawn_pump(self: &Arc<Self>, slot: &Arc<Slot>) {
        drop(self.pump(slot).map(tokio::spawn));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::OperationsFormat;
    use crate::render::tests::record;
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

    #[expect(
        clippy::expect_used,
        reason = "the test fails when the slot already has a pump"
    )]
    fn new_pump(scheduler: &Arc<Scheduler>, slot: &Arc<Slot>) -> impl Future<Output = ()> {
        scheduler.pump(slot).expect("the slot has no pump")
    }

    fn stamped(emitted_at: &str) -> TelemetryRecord {
        let mut r = record(Vec::new());
        r.emitted_at = emitted_at.to_owned();
        r
    }

    #[tokio::test]
    async fn a_newer_record_replaces_a_pending_one() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Scheduler::new(vec![recorder.clone()]);
        let slot = Slot::default();
        slot.schedule(stamped("1"));
        slot.schedule(stamped("2"));
        scheduler.export_pending(&slot).await;
        scheduler.export_pending(&slot).await;
        assert_eq!(*lock(&recorder.exported), ["2"]);
    }

    #[tokio::test]
    async fn drain_exports_and_flushes() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Scheduler::new(vec![recorder.clone()]);
        let slot = Slot::default();
        slot.schedule(stamped("1"));
        scheduler.drain(&slot).await;
        assert_eq!(*lock(&recorder.exported), ["1"]);
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn drain_without_a_record_still_flushes() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Scheduler::new(vec![recorder.clone()]);
        scheduler.drain(&Slot::default()).await;
        assert!(lock(&recorder.exported).is_empty());
        assert_eq!(recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_failing_or_panicking_exporter_does_not_stop_the_others() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Scheduler::new(vec![
            Arc::new(Failing),
            Arc::new(PanicsOnCall),
            recorder.clone(),
        ]);
        let slot = Slot::default();
        slot.schedule(stamped("1"));
        scheduler.drain(&slot).await;
        assert_eq!(*lock(&recorder.exported), ["1"]);
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
        let scheduler = Scheduler::new(vec![small.clone(), large.clone()]);
        let slot = Slot::default();
        slot.schedule(record(Vec::new()));
        scheduler.export_pending(&slot).await;
        assert_eq!(*lock(&small.0), [true]);
        assert_eq!(*lock(&large.0), [false]);
    }

    #[tokio::test]
    async fn the_pump_exports_the_pending_record() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![recorder.clone()]));
        let slot = Arc::new(Slot::default());
        slot.schedule(stamped("1"));
        new_pump(&scheduler, &slot).await;
        assert_eq!(*lock(&recorder.exported), ["1"]);
        assert!(format!("{scheduler:?}").contains("exporters: 1"));
    }

    #[tokio::test]
    async fn records_scheduled_during_an_export_start_no_second_pump() {
        let gated = Gated::closed();
        let scheduler = Arc::new(Scheduler::new(vec![gated.clone()]));
        let slot = Arc::new(Slot::default());
        slot.schedule(stamped("0"));
        let pump = new_pump(&scheduler, &slot);
        // `join!` polls the pump first. The pump takes record 0 and waits
        // for the gate. The second future then runs to its end.
        tokio::join!(pump, async {
            for n in 1..=500 {
                slot.schedule(stamped(&n.to_string()));
                assert!(scheduler.pump(&slot).is_none(), "record {n}");
            }
            gated.gate.add_permits(2);
        });
        assert_eq!(*lock(&gated.recorder.exported), ["0", "500"]);
    }

    #[tokio::test]
    async fn a_record_scheduled_after_the_pump_finished_starts_a_new_pump() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![recorder.clone()]));
        let slot = Arc::new(Slot::default());
        slot.schedule(stamped("1"));
        new_pump(&scheduler, &slot).await;
        slot.schedule(stamped("2"));
        new_pump(&scheduler, &slot).await;
        assert_eq!(*lock(&recorder.exported), ["1", "2"]);
    }

    #[test]
    fn an_empty_slot_ends_the_claim_while_the_pump_still_holds_its_token() {
        let slot = Slot::default();
        let token = slot.claim_pump();
        assert!(token.is_some());
        assert!(slot.claim_pump().is_none());
        assert!(slot.take_for_pump().is_none());
        assert!(slot.claim_pump().is_some());
        drop(token);
    }

    #[test]
    fn a_pump_dropped_before_it_ran_does_not_keep_the_slot_claimed() {
        let scheduler = Arc::new(Scheduler::new(Vec::new()));
        let slot = Arc::new(Slot::default());
        drop(new_pump(&scheduler, &slot));
        assert!(scheduler.pump(&slot).is_some());
    }

    #[tokio::test]
    async fn drain_exports_a_record_scheduled_while_the_pump_exports() {
        let gated = Gated::closed();
        let scheduler = Arc::new(Scheduler::new(vec![gated.clone()]));
        let slot = Arc::new(Slot::default());
        slot.schedule(stamped("1"));
        let pump = new_pump(&scheduler, &slot);
        tokio::join!(pump, async {
            slot.schedule(stamped("2"));
            gated.gate.add_permits(2);
            scheduler.drain(&slot).await;
        });
        assert_eq!(*lock(&gated.recorder.exported), ["1", "2"]);
        assert_eq!(gated.recorder.flushes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn spawn_pump_exports_without_being_awaited() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![recorder.clone()]));
        let slot = Arc::new(Slot::default());
        slot.schedule(stamped("1"));
        scheduler.spawn_pump(&slot);
        assert!(lock(&recorder.exported).is_empty());
        tokio::task::yield_now().await;
        assert_eq!(*lock(&recorder.exported), ["1"]);
    }

    #[tokio::test]
    async fn settle_all_waits_for_a_pending_future() {
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
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
