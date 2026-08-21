//! End-to-end replay-suppressed logging (issue #10).
//!
//! Drives a handler that logs before and after a durable wait through
//! [`LocalRunner`] with [`ReplayFilterLayer`] installed, and asserts each
//! handler-level log line is emitted exactly once across the whole
//! execution — the resumed invocation replays the code before the wait
//! without re-emitting its log line.

#![cfg(all(feature = "test-util", feature = "replay-filter"))]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk_rust::{self as durable, ReplayFilterLayer};
use durable::test_util::LocalRunner;
use tracing_subscriber::Layer as _;
use tracing_subscriber::layer::SubscriberExt;

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

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CaptureWriter {
    type Writer = CaptureWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[tokio::test]
async fn handler_level_log_emitted_exactly_once_across_invocations() {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(CaptureWriter(Arc::clone(&buffer)))
        .with_filter(ReplayFilterLayer::new());
    let subscriber = tracing_subscriber::registry().with(fmt_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                tracing::info!("replay-log-before-wait");
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                tracing::info!("replay-log-after-wait");
                Ok::<_, durable::BoxError>("done".to_owned())
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&"done".to_owned()));
    assert!(
        result.invocation_count() >= 2,
        "the execution must suspend on the wait and resume, got {} invocation(s)",
        result.invocation_count()
    );

    let output = buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    );

    assert_eq!(
        output.matches("replay-log-before-wait").count(),
        1,
        "the pre-wait log line must be emitted exactly once across {} invocations. Got: {output}",
        result.invocation_count()
    );
    assert_eq!(
        output.matches("replay-log-after-wait").count(),
        1,
        "the post-wait log line must be emitted exactly once across {} invocations. Got: {output}",
        result.invocation_count()
    );
}

/// A resumed `run_in_child_context` (a Started composite parent) must not
/// re-emit handler-level logs before it, nor child-body logs before the
/// nested wait: the child namespace carries its own replay-aware span, and
/// the root span treats a started composite record as still-replaying.
#[tokio::test]
async fn child_context_pre_wait_logs_emitted_exactly_once() {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(CaptureWriter(Arc::clone(&buffer)))
        .with_filter(ReplayFilterLayer::new());
    let subscriber = tracing_subscriber::registry().with(fmt_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                tracing::info!("handler-before-child");
                let value = ctx
                    .run_in_child_context(|child| async move {
                        tracing::info!("child-before-wait");
                        child
                            .wait(Duration::from_secs(1))
                            .name("child-pause")
                            .await?;
                        tracing::info!("child-after-wait");
                        Ok("child-done".to_owned())
                    })
                    .name("nested")
                    .await?;
                tracing::info!("handler-after-child");
                Ok::<_, durable::BoxError>(value)
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&"child-done".to_owned()));
    assert!(
        result.invocation_count() >= 2,
        "the execution must suspend on the nested wait and resume, got {} invocation(s)",
        result.invocation_count()
    );

    let output = buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    );

    for marker in [
        "handler-before-child",
        "child-before-wait",
        "child-after-wait",
        "handler-after-child",
    ] {
        assert_eq!(
            output.matches(marker).count(),
            1,
            "`{marker}` must be emitted exactly once across {} invocations. Got: {output}",
            result.invocation_count()
        );
    }
}

/// A resumed map branch must not re-emit its pre-wait log lines: each
/// branch runs in its own namespace with its own replay-aware span, so a
/// branch that is still replaying suppresses its logs independently of the
/// root handler span and of sibling branches.
#[tokio::test]
async fn map_branch_pre_wait_logs_emitted_exactly_once() {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(CaptureWriter(Arc::clone(&buffer)))
        .with_filter(ReplayFilterLayer::new());
    let subscriber = tracing_subscriber::registry().with(fmt_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                tracing::info!("handler-before-map");
                let items = vec![10_u32, 20];
                let outputs = ctx
                    .map(items, |child, item, idx| async move {
                        tracing::info!("branch-{idx}-before-wait");
                        child.wait(Duration::from_secs(1)).name("pause").await?;
                        tracing::info!("branch-{idx}-after-wait");
                        Ok(item)
                    })
                    .name("fan-out")
                    .await?;
                tracing::info!("handler-after-map");
                Ok::<_, durable::BoxError>(outputs)
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&vec![10_u32, 20]));
    assert!(
        result.invocation_count() >= 2,
        "the execution must suspend on the branch waits and resume, got {} invocation(s)",
        result.invocation_count()
    );

    let output = buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    );

    for marker in [
        "handler-before-map",
        "branch-0-before-wait",
        "branch-0-after-wait",
        "branch-1-before-wait",
        "branch-1-after-wait",
        "handler-after-map",
    ] {
        assert_eq!(
            output.matches(marker).count(),
            1,
            "`{marker}` must be emitted exactly once across {} invocations. Got: {output}",
            result.invocation_count()
        );
    }
}

/// A replayed TERMINAL flat batch must leave the caller's replay flag
/// correct for the code after it. In [`durable::builders::map_parallel::NestingMode::Flat`] the
/// synthetic child positions carry no checkpoint records, so minting the
/// terminal batch parent flips the namespace span to `isReplay=false`;
/// skipping those positions must re-derive the flag from the next logical
/// caller operation (here: the outer wait, which has a record), or the
/// handler log line between the batch and the wait is re-emitted on the
/// resumed invocation.
#[tokio::test]
async fn post_flat_batch_marker_emitted_exactly_once_on_terminal_replay() {
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let fmt_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(CaptureWriter(Arc::clone(&buffer)))
        .with_filter(ReplayFilterLayer::new());
    let subscriber = tracing_subscriber::registry().with(fmt_layer);
    let _guard = tracing::subscriber::set_default(subscriber);

    let result = LocalRunner::new()
        .run(
            |_event: serde_json::Value, ctx: durable::DurableContext| async move {
                // The flat batch completes in the FIRST invocation (no
                // suspension inside the items), so the resumed invocation
                // replays it from its terminal parent record alone.
                let outputs = ctx
                    .map(
                        vec![1_u32, 2],
                        |_child, item, _idx| async move { Ok(item * 10) },
                    )
                    .name("flat-fan-out")
                    .nesting(durable::builders::map_parallel::NestingMode::Flat)
                    .await?;
                tracing::info!("handler-after-flat-batch");
                // The outer wait suspends the first invocation, forcing a
                // resume that replays the terminal flat batch above.
                ctx.wait(Duration::from_secs(1)).name("outer-pause").await?;
                tracing::info!("handler-after-outer-wait");
                Ok::<_, durable::BoxError>(outputs)
            },
            serde_json::Value::Null,
        )
        .await;

    assert_eq!(result.output(), Some(&vec![10_u32, 20]));
    assert!(
        result.invocation_count() >= 2,
        "the execution must suspend on the outer wait and resume, got {} invocation(s)",
        result.invocation_count()
    );

    let output = buffer.lock().map_or_else(
        |_| String::new(),
        |b| String::from_utf8_lossy(&b).to_string(),
    );

    for marker in ["handler-after-flat-batch", "handler-after-outer-wait"] {
        assert_eq!(
            output.matches(marker).count(),
            1,
            "`{marker}` must be emitted exactly once across {} invocations. Got: {output}",
            result.invocation_count()
        );
    }
}
