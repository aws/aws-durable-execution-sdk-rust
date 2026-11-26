//! The panic boundary around code that this crate did not write.

use std::panic::{AssertUnwindSafe, catch_unwind};

use aws_durable_execution_sdk::BoxError;

/// Runs `f` and returns its value. A panic in `f` becomes an error.
///
/// Every call into an exporter or a transform goes through it. Both are
/// code that this crate did not write. Without the catch, a panic in an
/// exporter would end the pump, and the other exporters would not get the
/// record. A panic in a transform would end the hook that builds the
/// record.
///
/// The panic payload is also the caller's value, and its drop can panic.
/// So the payload is dropped inside a second catch. That drop's own
/// payload is not dropped at all: its drop could panic in turn.
pub(crate) fn contained<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        if let Err(second) = catch_unwind(AssertUnwindSafe(move || drop(payload))) {
            std::mem::forget(second);
        }
        "panicked".to_owned()
    })
}

/// Formats `error`, drops it, and returns the text.
///
/// Formatting and dropping both run code that this crate did not write,
/// and either can panic. A panic unwinds the closure it starts in, and
/// the unwind drops what that closure owns. A second panic during an
/// unwind aborts the process. So the first catch only borrows the error:
/// a panic in its formatting drops nothing. The second catch then owns
/// the error and drops it.
///
/// The text is `"panicked"` when the formatting panics.
pub(crate) fn describe_error(error: BoxError) -> String {
    let message = contained(|| error.to_string());
    let _ = contained(move || drop(error));
    message.unwrap_or_else(|panicked| panicked)
}

/// Emits one log event. A panic in the application's `tracing` subscriber
/// is caught and ignored.
///
/// A log call runs the application's subscriber, which is code that this
/// crate did not write. Without the catch, a panic there would end the
/// hook or the pump turn that logs.
pub(crate) fn log_contained(emit: impl FnOnce()) {
    let _ = contained(emit);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::Poll;

    /// An error whose formatting panics.
    #[derive(Debug)]
    pub(crate) struct PanicsOnDisplay;

    impl std::fmt::Display for PanicsOnDisplay {
        #[expect(
            clippy::panic,
            reason = "the tests check that a panicking error is contained"
        )]
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("display panicked")
        }
    }

    impl std::error::Error for PanicsOnDisplay {}

    /// A value whose drop panics. It is an error, a future and a panic
    /// payload.
    #[derive(Debug)]
    pub(crate) struct PanicsOnDrop;

    impl Drop for PanicsOnDrop {
        #[expect(
            clippy::panic,
            reason = "the tests check that a panicking drop is contained"
        )]
        fn drop(&mut self) {
            panic!("drop panicked")
        }
    }

    impl std::fmt::Display for PanicsOnDrop {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("drop panics")
        }
    }

    impl std::error::Error for PanicsOnDrop {}

    impl Future for PanicsOnDrop {
        type Output = Result<(), BoxError>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> Poll<Self::Output> {
            Poll::Ready(Ok(()))
        }
    }

    /// An error whose formatting panics and whose drop panics.
    #[derive(Debug)]
    pub(crate) struct PanicsOnDisplayAndDrop;

    impl std::fmt::Display for PanicsOnDisplayAndDrop {
        #[expect(
            clippy::panic,
            reason = "the tests check that a panicking error is contained"
        )]
        fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            panic!("display panicked")
        }
    }

    impl Drop for PanicsOnDisplayAndDrop {
        #[expect(
            clippy::panic,
            reason = "the tests check that a panicking drop is contained"
        )]
        fn drop(&mut self) {
            panic!("drop panicked")
        }
    }

    impl std::error::Error for PanicsOnDisplayAndDrop {}

    /// A subscriber that counts every event and panics on it.
    #[derive(Default)]
    pub(crate) struct PanicsOnEvent(pub(crate) Arc<AtomicUsize>);

    impl tracing::Subscriber for PanicsOnEvent {
        fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }

        fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}

        fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}

        #[expect(
            clippy::panic,
            reason = "the test checks that a panicking subscriber is contained"
        )]
        fn event(&self, _event: &tracing::Event<'_>) {
            self.0.fetch_add(1, Ordering::SeqCst);
            panic!("subscriber panicked")
        }

        fn enter(&self, _span: &tracing::span::Id) {}

        fn exit(&self, _span: &tracing::span::Id) {}
    }

    #[test]
    #[expect(
        clippy::panic,
        reason = "the test checks that a panicking payload drop is contained"
    )]
    fn a_panic_payload_whose_drop_panics_is_contained() {
        let result = contained(|| std::panic::panic_any(PanicsOnDrop));
        assert_eq!(result, Err::<(), _>("panicked".to_owned()));
    }
}
