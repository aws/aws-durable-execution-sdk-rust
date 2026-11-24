//! The panic boundary around code that this crate did not write.

use std::panic::{AssertUnwindSafe, catch_unwind};

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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
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
        type Output = Result<(), aws_durable_execution_sdk::BoxError>;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> Poll<Self::Output> {
            Poll::Ready(Ok(()))
        }
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
