//! A `FileSystemSerdes` preview, built with `build_preview`, lands in the
//! file envelope and replays through `LocalRunner`.
//!
//! The tests use only the public API of the crate, plus `serde_json` to
//! read the returned envelope text.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::serdes::{
    FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode, PreviewConfig, PreviewMode,
    RetryableSerdesError, Serdes, SerdesContext, build_preview,
};
use durable::test_util::LocalRunner;
use durable::{BoxError, DurableContext};

/// A directory under the system temp dir that no other test uses.
fn scratch_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "fs_serdes_preview_{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ))
}

/// Counts the regular files under `dir`, recursively.
fn count_files(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() { count_files(&path) } else { 1 }
        })
        .sum()
}

fn value() -> serde_json::Value {
    serde_json::json!({"id": "abc", "secret": "s3cr3t", "other": "ignored"})
}

/// The serdes every test uses: `Always` mode, with a preview that shows
/// `id` and masks `secret`.
fn preview_serdes(dir: &Path) -> FileSystemSerdes {
    let config = PreviewConfig::builder(PreviewMode::ExcludeAll)
        .include("id")
        .mask("secret")
        .build();
    FileSystemSerdes::with_config(
        dir.to_string_lossy().into_owned(),
        FileSystemSerdesConfig::builder()
            .storage_mode(FileSystemSerdesMode::Always)
            .preview(move |json| Ok(build_preview(json, &config)))
            .build(),
    )
}

#[tokio::test]
async fn preview_from_build_preview_lands_in_the_envelope() {
    let dir = scratch_dir("envelope");
    let serdes = preview_serdes(&dir);

    let envelope = Serdes::<serde_json::Value>::serialize(
        &serdes,
        value(),
        SerdesContext::new("op-1", "arn:test"),
    )
    .await
    .expect("serialize");
    let parsed: serde_json::Value = serde_json::from_str(&envelope).expect("envelope JSON");
    assert_eq!(
        parsed.get("preview"),
        Some(&serde_json::json!({"id": "abc", "secret": "***"})),
        "{envelope}"
    );

    let output: serde_json::Value = serdes
        .deserialize(envelope, SerdesContext::new("op-1", "arn:test"))
        .await
        .expect("deserialize");
    assert_eq!(output, value());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn step_with_preview_replays_through_local_runner() {
    let dir = scratch_dir("replay");
    let serdes = preview_serdes(&dir);

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                async move {
                    let stored = ctx
                        .step(|_| async { Ok(value()) })
                        .name("store")
                        .serdes(serdes)
                        .await?;
                    ctx.wait(Duration::from_secs(1)).name("pause").await?;
                    Ok::<_, BoxError>(stored)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        result.output(),
        Some(&value()),
        "{:?}",
        result.error_message()
    );
    assert_eq!(result.invocation_count(), 2);
    let recorded = result
        .operation("store")
        .and_then(|op| op.result())
        .expect("the step records its envelope");
    let parsed: serde_json::Value = serde_json::from_str(recorded).expect("envelope JSON");
    assert_eq!(
        parsed.get("preview"),
        Some(&serde_json::json!({"id": "abc", "secret": "***"})),
        "{recorded}"
    );
}

#[tokio::test]
async fn preview_error_marked_retryable_ends_invocation() {
    let dir = scratch_dir("retryable");
    let calls = Arc::new(AtomicUsize::new(0));
    // The number of files in `dir` when the hook runs again, in the next
    // invocation.
    let files_at_retry = Arc::new(Mutex::new(None));
    let serdes = {
        let calls = Arc::clone(&calls);
        let files_at_retry = Arc::clone(&files_at_retry);
        let dir = dir.clone();
        FileSystemSerdes::with_config(
            dir.to_string_lossy().into_owned(),
            FileSystemSerdesConfig::builder()
                .storage_mode(FileSystemSerdesMode::Always)
                .preview(move |_json| {
                    if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                        return Err(RetryableSerdesError::new("preview store down").into());
                    }
                    *files_at_retry.lock().expect("lock") = Some(count_files(&dir));
                    Ok(None)
                })
                .build(),
        )
    };

    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                async move {
                    let stored = ctx
                        .step(|_| async { Ok(value()) })
                        .name("store")
                        .serdes(serdes)
                        .await?;
                    Ok::<_, BoxError>(stored)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    let files_after_run = count_files(&dir);
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        result.output(),
        Some(&value()),
        "{:?}",
        result.error_message()
    );
    assert_eq!(result.invocation_count(), 2);
    let first = result
        .invocations()
        .first()
        .and_then(|invocation| invocation.error())
        .expect("the first invocation ends with a fault");
    let message = first.error_message().unwrap_or_default();
    assert!(message.contains("serdes serialize failed"), "{message}");
    assert_eq!(first.error_type(), None, "an invocation fault has no type");
    assert_eq!(
        *files_at_retry.lock().expect("lock"),
        Some(0),
        "the first invocation wrote no payload file"
    );
    assert_eq!(files_after_run, 1, "the second invocation wrote one file");
}
