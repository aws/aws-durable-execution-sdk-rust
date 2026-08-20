//! End-to-end async-path test for [`FileSystemSerdes`] (issues #20, #37).
//!
//! `FileSystemSerdes` performs blocking `std::fs` I/O. Under the generic
//! async [`Serdes`] trait the SDK awaits the serdes future directly, and
//! the IMPLEMENTATION owns its scheduling: `FileSystemSerdes` moves each
//! complete call into one `tokio::task::spawn_blocking` task, so a slow
//! filesystem never stalls the executor. This test drives a full
//! execution — live run, suspension, and replay — with `FileSystemSerdes`
//! attached, on a single-threaded runtime where any inline blocking would
//! also be the quickest to misbehave, and verifies the round trip through
//! the real filesystem.

#![cfg(feature = "test-util")]
#![allow(clippy::expect_used)] // reason: test assertions with descriptive messages

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::Serdes;
use durable::serdes::FileSystemSerdes;
use durable::test_util::LocalRunner;

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

/// A serdes whose blocking work would deadlock a current-thread runtime if
/// run inline: it parks a thread until a second task gets to run. Following
/// the [`Serdes`] scheduling contract, the IMPLEMENTATION moves that work
/// into `tokio::task::spawn_blocking` itself, so the executor stays free,
/// the unblocking task runs, and the operation completes.
#[derive(Debug)]
struct ExecutorProbeSerdes {
    unblocked: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Serdes<u32> for ExecutorProbeSerdes {
    async fn serialize(
        &self,
        value: u32,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        let unblocked = std::sync::Arc::clone(&self.unblocked);
        // Implementation-controlled blocking: the ENTIRE blocking wait runs
        // inside one spawn_blocking task, per the trait's contract.
        tokio::task::spawn_blocking(move || -> Result<String, durable::BoxError> {
            // Wait (bounded) for the async task on the executor to flip the
            // flag. Inline execution on a current-thread runtime would never
            // observe the flip: the executor would be stuck right here.
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while !unblocked.load(std::sync::atomic::Ordering::SeqCst) {
                if std::time::Instant::now() > deadline {
                    return Err("executor never ran the unblocking task — \
                         blocking serdes work must run on the blocking pool"
                        .into());
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(serde_json::to_string(&value)?)
        })
        .await
        .map_err(|e| -> durable::BoxError { format!("join error: {e}").into() })?
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<u32, durable::BoxError> {
        Ok(serde_json::from_str(&wire)?)
    }
}

/// The blocking serdes work demonstrably runs OFF the executor: while the
/// blocking closure parks its thread, an async task on the
/// (single-threaded) runtime still gets polled. An implementation that ran
/// the same wait inline would time out instead of completing.
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

/// A type-specific custom serdes over one concrete type.
#[derive(Debug)]
struct PrefixSerdes;

impl Serdes<u32> for PrefixSerdes {
    async fn serialize(
        &self,
        value: u32,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(format!("N:{value}"))
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<u32, durable::BoxError> {
        Ok(wire
            .strip_prefix("N:")
            .ok_or_else(|| -> durable::BoxError { "missing N: prefix".into() })?
            .parse()?)
    }
}

/// `IntoFuture` erases the serdes type: futures configured with different
/// serdes implementations (default `JsonSerdes`, a type-specific custom
/// format, and a shared `Arc<S>`) are all `DurableFuture<u32>` and coexist
/// in ONE combinator input collection, through a real execution.
#[tokio::test]
async fn mixed_serdes_futures_coexist_in_one_combinator() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: durable::DurableContext| async move {
                let shared = std::sync::Arc::new(PrefixSerdes);
                let futures: Vec<durable::DurableFuture<u32>> = vec![
                    // Default JsonSerdes.
                    ctx.step(|_| async { Ok(1_u32) }).name("json").future(),
                    // Type-specific custom serdes.
                    ctx.step(|_| async { Ok(2_u32) })
                        .name("custom")
                        .serdes(PrefixSerdes)
                        .future(),
                    // Shared Arc<S> instance.
                    ctx.step(|_| async { Ok(3_u32) })
                        .name("shared")
                        .serdes(shared)
                        .future(),
                ];
                let values = ctx.try_join_all(futures).name("all").await?;
                Ok::<_, durable::BoxError>(values.iter().sum::<u32>())
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "mixed-serdes combinator must succeed: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&6));

    // The custom-serdes steps stored their non-JSON wire forms.
    let stored: Vec<Option<&str>> = ["custom", "shared"]
        .iter()
        .map(|name| {
            result
                .operations()
                .iter()
                .find(|op| op.name() == Some(name))
                .and_then(|op| op.result())
        })
        .collect();
    assert_eq!(
        stored,
        vec![Some("N:2"), Some("N:3")],
        "custom serdes wire forms must reach the checkpoint"
    );
}

/// A `Send` but non-`Sync` step output compiles and round-trips through a
/// real operation with `FileSystemSerdes` attached: the owned-value serdes
/// API moves the value into the blocking task, so `T: Send` is the only
/// bound the operation needs — never `T: Sync`.
#[tokio::test]
async fn send_not_sync_output_round_trips_through_operation() {
    // `Cell` is `Send` but not `Sync`.
    #[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
    struct NotSync {
        value: String,
        #[serde(skip)]
        #[allow(dead_code)] // reason: exists only to make the type non-Sync
        cell: std::cell::Cell<u8>,
    }

    let base = TempBase::new("not_sync_step");
    let base_path = base.path();

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let base_path = base_path.clone();
                async move {
                    let out: NotSync = ctx
                        .step(|_| async {
                            Ok(NotSync {
                                value: "send-not-sync".to_owned(),
                                cell: std::cell::Cell::new(3),
                            })
                        })
                        .name("not-sync")
                        .serdes(FileSystemSerdes::new(base_path))
                        .await?;
                    Ok::<_, durable::BoxError>(out.value)
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "non-Sync output must round-trip: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&"send-not-sync".to_owned()));
}

// ── custom-serdes outputs without serde traits (issue #37) ──────────────
//
// A type-specific serdes pairs with the operation's actual Rust type, so
// the builders whose execution paths need only `S: Serdes<O>` must accept
// an output type that implements NEITHER `Serialize` NOR
// `DeserializeOwned`.

/// Deliberately implements no serde traits — only the custom serdes below
/// knows its wire form.
struct NoSerde {
    text: String,
}

/// A custom wire format for [`NoSerde`]: the raw text, no JSON anywhere.
#[derive(Debug, Clone)]
struct NoSerdeSerdes;

impl Serdes<NoSerde> for NoSerdeSerdes {
    async fn serialize(
        &self,
        value: NoSerde,
        _context: durable::serdes::SerdesContext,
    ) -> Result<String, durable::BoxError> {
        Ok(value.text)
    }

    async fn deserialize(
        &self,
        wire: String,
        _context: durable::serdes::SerdesContext,
    ) -> Result<NoSerde, durable::BoxError> {
        Ok(NoSerde { text: wire })
    }
}

/// Compile-only coverage: `create_callback` and `with_retry` builders (the
/// `future`/`spawn` conversions and `IntoFuture`) resolve for an output
/// type with no serde implementations at all. The bounds resolving IS the
/// assertion, so the function is never called.
#[allow(dead_code)] // reason: compile-only bound coverage, never invoked
fn non_serde_output_builders_compile(ctx: &durable::DurableContext) {
    // create_callback: no `DeserializeOwned` on the payload type.
    let _cb: durable::DurableFuture<durable::builders::callback::Callback<NoSerde>> = ctx
        .create_callback::<NoSerde>()
        .serdes(NoSerdeSerdes)
        .future();

    // with_retry: no `Serialize`/`DeserializeOwned` on the block output.
    let _wr: durable::DurableFuture<NoSerde> = ctx
        .with_retry(|_child| async {
            Ok(NoSerde {
                text: "compile-only".to_owned(),
            })
        })
        .serdes(NoSerdeSerdes)
        .future();
}

/// A `with_retry` block whose output has no serde implementations
/// round-trips through a real execution: the configured custom serdes
/// carries the value both for each attempt's nested child context and for
/// the block's own result.
#[tokio::test]
async fn with_retry_round_trips_non_serde_output_through_custom_serdes() {
    let result = LocalRunner::new()
        .run(
            |_event: (), ctx: durable::DurableContext| async move {
                let out: NoSerde = ctx
                    .with_retry(|_child| async {
                        Ok(NoSerde {
                            text: "no-serde-traits".to_owned(),
                        })
                    })
                    .name("no-serde-block")
                    .serdes(NoSerdeSerdes)
                    .await?;
                Ok::<_, durable::BoxError>(out.text)
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "non-serde output must round-trip: {:?} / {:?}",
        result.error_type(),
        result.error_message()
    );
    assert_eq!(result.output(), Some(&"no-serde-traits".to_owned()));
}
