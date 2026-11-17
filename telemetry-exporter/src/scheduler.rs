//! Serializes exports and keeps only the latest record of each invocation.

use std::borrow::Cow;
use std::future::{Future, poll_fn};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
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
#[derive(Debug, Default)]
pub(crate) struct Slot {
    state: Mutex<SlotState>,
}

#[derive(Debug, Default)]
struct SlotState {
    pending: Option<TelemetryRecord>,
    revision: u64,
}

impl Slot {
    /// Replaces the pending record.
    pub(crate) fn schedule(&self, record: TelemetryRecord) {
        let mut state = lock(&self.state);
        state.pending = Some(record);
        state.revision = state.revision.wrapping_add(1);
    }

    fn take(&self) -> Option<TelemetryRecord> {
        lock(&self.state).pending.take()
    }

    /// The number of records scheduled so far, and whether one is pending.
    fn status(&self) -> (u64, bool) {
        let state = lock(&self.state);
        (state.revision, state.pending.is_some())
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
    results
        .into_iter()
        .map(|result| result.unwrap_or(Ok(())))
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
        let Some(record) = slot.take() else {
            return;
        };
        let records: Vec<Cow<'_, TelemetryRecord>> = self
            .exporters
            .iter()
            .map(|exporter| {
                truncate_record(
                    &record,
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

    /// Exports the slot's pending record and flushes, until no record was
    /// scheduled during the round.
    ///
    /// A record scheduled while a flush runs may miss that flush. So the
    /// loop runs another round when the revision changed.
    pub(crate) async fn drain(&self, slot: &Slot) {
        loop {
            let (before, _) = slot.status();
            self.export_pending(slot).await;
            self.flush().await;
            let (after, pending) = slot.status();
            if after == before && !pending {
                return;
            }
        }
    }

    /// A `'static` future that exports the slot's pending record.
    pub(crate) fn export_pending_owned(
        self: &Arc<Self>,
        slot: &Arc<Slot>,
    ) -> impl Future<Output = ()> + Send + 'static {
        let scheduler = Arc::clone(self);
        let slot = Arc::clone(slot);
        async move { scheduler.export_pending(&slot).await }
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
    async fn the_owned_future_exports_the_pending_record() {
        let recorder = Arc::new(Recorder::default());
        let scheduler = Arc::new(Scheduler::new(vec![recorder.clone()]));
        let slot = Arc::new(Slot::default());
        slot.schedule(stamped("1"));
        scheduler.export_pending_owned(&slot).await;
        assert_eq!(*lock(&recorder.exported), ["1"]);
        assert!(format!("{scheduler:?}").contains("exporters: 1"));
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
