//! End-to-end async-path test for [`FileSystemSerdes`] (issue #20).
//!
//! `FileSystemSerdes` performs blocking `std::fs` I/O behind the sync
//! [`Serdes`] trait. The SDK routes every serdes invocation reached from
//! async code through `tokio::task::spawn_blocking`, so a slow filesystem
//! never stalls the executor. This test drives a full execution — live run,
//! suspension, and replay — with `FileSystemSerdes` attached, on a
//! single-threaded runtime where any inline blocking would also be the
//! quickest to misbehave, and verifies the round trip through the real
//! filesystem.

#![cfg(feature = "test-util")]
#![allow(clippy::expect_used)] // reason: test assertions with descriptive messages

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::test_util::LocalRunner;
use durable::{FileSystemSerdes, Serdes, SerdesContext};

/// Fresh temp dir per test, cleaned up on drop even when the test fails.
struct TempBase(std::path::PathBuf);

impl TempBase {
    fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("fs_serdes_async_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        Self(dir)
    }

    fn path(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }
}

impl Drop for TempBase {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A step with `FileSystemSerdes` attached round-trips its result through
/// the filesystem across a suspension: the live invocation serializes (file
/// write, on the blocking pool), and the replay invocation deserializes
/// (file read, on the blocking pool) — all from a current-thread runtime.
#[tokio::test]
async fn step_round_trips_through_filesystem_across_replay() {
    let base = TempBase::new("step_replay");
    let base_path = base.path();

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let base_path = base_path.clone();
                async move {
                    let value = ctx
                        .step(|_| async { Ok("stored on the filesystem".to_owned()) })
                        .name("fs-step")
                        .serdes(FileSystemSerdes::new(base_path))
                        .await?;
                    // Suspend so the execution resumes in a second invocation
                    // and the step result replays through deserialize.
                    ctx.wait(Duration::from_secs(1)).name("pause").await?;
                    Ok::<_, durable::BoxError>(value)
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "execution should succeed: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(
        result.output(),
        Some(&"stored on the filesystem".to_owned())
    );
    assert!(
        result.invocation_count() >= 2,
        "the wait must split the execution into live + replay invocations"
    );

    // The step's checkpoint payload must be the file-pointer envelope, and
    // the pointed-at file must hold the value — proof the serdes actually
    // ran against the filesystem rather than being bypassed.
    let step_op = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("fs-step"))
        .expect("step operation must be recorded");
    let envelope: serde_json::Value =
        serde_json::from_str(step_op.result().expect("step must checkpoint a result"))
            .expect("checkpoint payload must be the serdes envelope");
    let file_path = envelope
        .get("file")
        .and_then(serde_json::Value::as_str)
        .expect("Always mode must store a file pointer");
    let on_disk = std::fs::read_to_string(file_path).expect("pointed-at file must exist");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&on_disk).expect("file holds JSON"),
        serde_json::json!("stored on the filesystem"),
    );
}

/// A serdes whose blocking `serialize` would deadlock a current-thread
/// runtime if invoked inline: it parks its thread until a second task gets
/// to run. With `spawn_blocking` routing, the executor stays free, the
/// unblocking task runs, and the operation completes.
#[derive(Debug)]
struct ExecutorProbeSerdes {
    unblocked: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Serdes for ExecutorProbeSerdes {
    fn serialize(
        &self,
        value: &serde_json::Value,
        _context: &SerdesContext,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        // Wait (bounded) for the async task on the executor to flip the
        // flag. Inline execution on a current-thread runtime would never
        // observe the flip: the executor would be stuck right here.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !self.unblocked.load(std::sync::atomic::Ordering::SeqCst) {
            if std::time::Instant::now() > deadline {
                return Err("executor never ran the unblocking task — \
                     serdes must not run inline on the runtime"
                    .into());
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(value.to_string())
    }
}

/// The serdes call demonstrably runs OFF the executor: while the sync
/// `serialize` blocks its thread, an async task on the (single-threaded)
/// runtime still gets polled. If the SDK invoked the serdes inline, this
/// test would time out instead of completing.
#[tokio::test]
async fn serdes_call_does_not_block_the_executor() {
    let unblocked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag_setter = std::sync::Arc::clone(&unblocked);

    // Flip the flag from the executor shortly after the step serializes.
    // This task can only run if the executor thread is not blocked inside
    // the serdes.
    let handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        flag_setter.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let probe = ExecutorProbeSerdes {
                    unblocked: std::sync::Arc::clone(&unblocked),
                };
                async move {
                    ctx.step(|_| async { Ok(42_u32) })
                        .name("probed")
                        .serdes(probe)
                        .await
                        .map_err(durable::BoxError::from)
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "step must complete because the serdes runs on the blocking pool: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&42));
    handle.await.expect("unblocker task must finish");
}
