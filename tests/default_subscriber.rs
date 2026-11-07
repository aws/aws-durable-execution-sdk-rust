//! The default subscriber helper and the `lambda_runtime` re-export path.
//!
//! These tests build every subscriber through
//! `aws_durable_execution_sdk::lambda_runtime::tracing::subscriber`, never
//! through a direct `tracing-subscriber` import. So a dependency update that
//! breaks that path fails to compile here.

use std::sync::{Arc, Mutex};

use aws_durable_execution_sdk as durable;
use durable::lambda_runtime::tracing::subscriber;
use subscriber::Layer as _;
use subscriber::layer::SubscriberExt as _;

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

/// The helper installs a global subscriber with `try_init`. So the first
/// call succeeds, and the second call returns the error that `try_init`
/// returns when a global subscriber is already set.
#[test]
fn second_init_returns_error() {
    assert!(durable::init_default_subscriber().is_ok());
    assert!(durable::init_default_subscriber().is_err());
}

/// A subscriber built outside the crate, from the re-exported
/// `tracing-subscriber`, accepts `ReplayFilterLayer` as a layer filter and
/// writes a record outside any replay once.
#[test]
fn replay_filter_composes_with_the_runtime_reexport() {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = subscriber::registry().with(
        subscriber::fmt::layer()
            .with_writer(CaptureWriter(Arc::clone(&buffer)))
            .with_filter(durable::ReplayFilterLayer::new()),
    );

    {
        let _guard = test_tracing::install_test_subscriber(subscriber);
        tracing::info!("composed-record");
    };

    let output = buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    );
    assert_eq!(output.matches("composed-record").count(), 1, "{output}");
}
