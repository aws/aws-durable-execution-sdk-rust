//! Thread-default `tracing` captures for the SDK's tests.
//!
//! `tracing-core` caches one `Interest` per callsite for the whole process.
//! While only one dispatcher is registered, a callsite's first use asks the
//! default subscriber of the thread that reaches it. On a thread with no
//! default, that is the no-op subscriber, which answers `Interest::never()`,
//! and the event macro then skips the callsite on every thread, including
//! the thread of a capture installed later.
//!
//! This crate registers two dispatchers before `main`, through `ctor`. Their
//! subscriber answers `Interest::sometimes()` for every callsite and enables
//! nothing. With more than one dispatcher registered, every callsite caches
//! the combined interest of all of them, which is `sometimes`, so each event
//! asks the emitting thread's current default whether it is enabled. The
//! two dispatchers never become a default, and the global default stays
//! whatever the test sets.
//!
//! Install every test capture with [`install_test_subscriber`].

use std::sync::OnceLock;

use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::{DefaultGuard, Interest};
use tracing::{Dispatch, Event, Metadata, Subscriber};

/// Makes `subscriber` the current thread's default until the returned guard
/// drops.
///
/// The binary that links this crate runs the pre-main initializer, so the
/// subscriber receives every event and span it enables on this thread,
/// whichever thread reached the callsite first.
pub fn install_test_subscriber<S>(subscriber: S) -> DefaultGuard
where
    S: Subscriber + Send + Sync + 'static,
{
    #[expect(clippy::disallowed_methods)]
    // reason: this is the one place that installs a test capture.
    tracing::subscriber::set_default(subscriber)
}

/// A subscriber that leaves every callsite undecided and enables nothing.
#[derive(Debug)]
struct UndecidedSubscriber;

impl Subscriber for UndecidedSubscriber {
    fn register_callsite(&self, _metadata: &'static Metadata<'static>) -> Interest {
        Interest::sometimes()
    }

    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        false
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, _event: &Event<'_>) {}

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

/// The two kept dispatchers. A registered dispatcher stays registered while
/// a `Dispatch` holding it lives, so these keep more than one dispatcher
/// registered for the life of the binary.
static KEPT: OnceLock<[Dispatch; 2]> = OnceLock::new();

/// Registers the two kept dispatchers and rebuilds the interest of every
/// callsite registered before them.
///
/// `ctor` runs it before `main`. It touches only the `tracing` registries
/// and a `OnceLock`, which need no runtime setup.
#[ctor::ctor(unsafe)]
fn keep_undecided_dispatchers() {
    KEPT.get_or_init(|| {
        [
            Dispatch::new(UndecidedSubscriber),
            Dispatch::new(UndecidedSubscriber),
        ]
    });
    tracing::callsite::rebuild_interest_cache();
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex, PoisonError};

    use super::*;

    /// A capture that keeps the metadata of every event it receives.
    #[derive(Clone, Default)]
    struct Seen(Arc<Mutex<Vec<&'static Metadata<'static>>>>);

    impl Seen {
        fn names(&self) -> Vec<&'static str> {
            let seen = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            seen.iter().map(|metadata| metadata.name()).collect()
        }
    }

    impl Subscriber for Seen {
        fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _span: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }
        fn record(&self, _span: &Id, _values: &Record<'_>) {}
        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut seen = self.0.lock().unwrap_or_else(PoisonError::into_inner);
            seen.push(event.metadata());
        }
        fn enter(&self, _span: &Id) {}
        fn exit(&self, _span: &Id) {}
    }

    fn probe() {
        tracing::error!(name: "isolation-probe", "probe");
    }

    #[test]
    fn the_kept_dispatchers_exist_before_any_test() {
        assert!(KEPT.get().is_some());
    }

    #[test]
    fn the_undecided_subscriber_enables_nothing_and_stays_undecided() {
        let seen = Seen::default();
        {
            let _guard = install_test_subscriber(seen.clone());
            probe();
        }
        let seen = seen.0.lock().unwrap_or_else(PoisonError::into_inner);
        let metadata = seen.first().copied();
        assert!(metadata.is_some());
        if let Some(metadata) = metadata {
            assert!(
                UndecidedSubscriber
                    .register_callsite(metadata)
                    .is_sometimes()
            );
            assert!(!UndecidedSubscriber.enabled(metadata));
        }
    }

    #[test]
    fn a_capture_receives_events_only_from_its_own_thread() {
        let seen = Seen::default();
        let _guard = install_test_subscriber(seen.clone());
        assert!(std::thread::spawn(probe).join().is_ok());
        assert_eq!(seen.names(), Vec::<&str>::new());
        probe();
        assert_eq!(seen.names(), ["isolation-probe"]);
    }
}
