//! A serdes failure marked with `RetryableSerdesError` ends the
//! invocation and records nothing; the service invokes the function
//! again. An unmarked failure keeps its permanent path.
//!
//! Every test drives a handler through [`LocalRunner`], which invokes the
//! function again after an invocation fault, as the service does.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::map_parallel::NestingMode;
use durable::builders::wait_for_condition::WaitDecision;
use durable::serdes::{
    FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode, RetryableSerdesError, Serdes,
    SerdesContext,
};
use durable::test_util::{CallbackReply, LocalRunner, TestResult};
use durable::{BoxError, DurableContext, RetryDecision, StepSemantics};

/// How an injected failure is reported.
#[derive(Clone, Copy)]
enum Mark {
    /// `RetryableSerdesError::new("injected failure")`.
    Marked,
    /// The plain error `"injected failure"`.
    Unmarked,
    /// A user error whose `source()` is a `RetryableSerdesError`.
    Wrapped,
}

/// Which 0-based calls of one direction fail.
#[derive(Clone)]
enum Calls {
    None,
    At(Vec<usize>),
    Every,
}

impl Calls {
    fn contains(&self, index: usize) -> bool {
        match self {
            Self::None => false,
            Self::At(indexes) => indexes.contains(&index),
            Self::Every => true,
        }
    }
}

/// A user error that wraps the marker, so the SDK must walk the chain.
#[derive(Debug)]
struct UserError(RetryableSerdesError);

impl std::fmt::Display for UserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("user serdes error")
    }
}

impl std::error::Error for UserError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// Delegates to JSON and fails the configured calls.
#[derive(Clone)]
struct InjectedFailures {
    mark: Mark,
    serialize_fails: Calls,
    deserialize_fails: Calls,
    serialize_calls: Arc<AtomicUsize>,
    deserialize_calls: Arc<AtomicUsize>,
}

impl InjectedFailures {
    fn new(mark: Mark) -> Self {
        Self {
            mark,
            serialize_fails: Calls::None,
            deserialize_fails: Calls::None,
            serialize_calls: Arc::new(AtomicUsize::new(0)),
            deserialize_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn serialize_at(mut self, indexes: &[usize]) -> Self {
        self.serialize_fails = Calls::At(indexes.to_vec());
        self
    }

    fn serialize_every(mut self) -> Self {
        self.serialize_fails = Calls::Every;
        self
    }

    fn deserialize_at(mut self, indexes: &[usize]) -> Self {
        self.deserialize_fails = Calls::At(indexes.to_vec());
        self
    }

    fn failure(&self) -> BoxError {
        match self.mark {
            Mark::Marked => Box::new(RetryableSerdesError::new("injected failure")),
            Mark::Unmarked => "injected failure".into(),
            Mark::Wrapped => Box::new(UserError(RetryableSerdesError::new("injected failure"))),
        }
    }
}

impl<T> Serdes<T> for InjectedFailures
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    fn serialize(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        let call = self.serialize_calls.fetch_add(1, Ordering::SeqCst);
        let result = if self.serialize_fails.contains(call) {
            Err(self.failure())
        } else {
            serde_json::to_string(&value).map_err(Into::into)
        };
        std::future::ready(result)
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<T, BoxError>> + Send {
        let call = self.deserialize_calls.fetch_add(1, Ordering::SeqCst);
        let result = if self.deserialize_fails.contains(call) {
            Err(self.failure())
        } else {
            serde_json::from_str(&wire).map_err(Into::into)
        };
        std::future::ready(result)
    }
}

/// Counts body runs and runs of the handler code right after the
/// operation.
#[derive(Clone, Default)]
struct Counters {
    body: Arc<AtomicUsize>,
    after: Arc<AtomicUsize>,
}

impl Counters {
    fn body(&self) -> usize {
        self.body.load(Ordering::SeqCst)
    }

    fn after(&self) -> usize {
        self.after.load(Ordering::SeqCst)
    }
}

fn assert_no_failed_operation<O>(result: &TestResult<O>) {
    let failed: Vec<_> = result
        .operations()
        .iter()
        .filter(|op| op.failed())
        .collect();
    assert!(failed.is_empty(), "no operation may fail: {failed:?}");
}

/// The kind a step error reports, as its `Debug` label.
fn step_kind_label(err: &durable::OperationError) -> String {
    match err.kind() {
        durable::OperationErrorKind::Step(step_err) => format!("{:?}", step_err.kind()),
        other => format!("non-step: {other}"),
    }
}

/// Runs one step named `encode-me` or `decode-me` with `serdes` and
/// returns its value.
async fn run_one_step(
    runner: LocalRunner,
    name: &'static str,
    serdes: InjectedFailures,
    counters: Counters,
) -> TestResult<String> {
    runner
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let counters = counters.clone();
                async move {
                    let body = Arc::clone(&counters.body);
                    let value = ctx
                        .step(move |_| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok("step-value".to_owned())
                            }
                        })
                        .name(name)
                        .serdes(serdes)
                        .await?;
                    counters.after.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

#[tokio::test]
async fn step_marked_serialize_failure_reruns_body() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).serialize_at(&[0]);
    let result = run_one_step(LocalRunner::new(), "encode-me", serdes, counters.clone()).await;

    assert_eq!(result.output().map(String::as_str), Some("step-value"));
    assert_eq!(result.invocation_count(), 2);
    assert_eq!(counters.body(), 2, "the marked failure recorded no outcome");
    assert_no_failed_operation(&result);
    assert_eq!(counters.after(), 1);
}

#[tokio::test]
async fn step_marked_serialize_failure_at_most_once_takes_interrupted_path() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).serialize_at(&[0]);
    let counters_h = counters.clone();
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let body = Arc::clone(&counters_h.body);
                let after = Arc::clone(&counters_h.after);
                async move {
                    let outcome = ctx
                        .step(move |_| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok("step-value".to_owned())
                            }
                        })
                        .name("encode-me")
                        .semantics(StepSemantics::AtMostOncePerRetry)
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .serdes(serdes)
                        .await;
                    after.fetch_add(1, Ordering::SeqCst);
                    let err = outcome.expect_err("the interrupted attempt fails");
                    Ok::<_, BoxError>(step_kind_label(&err))
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

    assert_eq!(result.output().map(String::as_str), Some("Interrupted"));
    assert_eq!(counters.body(), 1);
    assert_eq!(counters.after(), 1);
    let step = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("encode-me"))
        .expect("the step is recorded");
    assert!(step.failed(), "{step:?}");
}

#[tokio::test]
async fn step_marked_live_decode_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).deserialize_at(&[0]);
    let result = run_one_step(LocalRunner::new(), "decode-me", serdes, counters.clone()).await;

    assert_eq!(result.output().map(String::as_str), Some("step-value"));
    assert_eq!(result.invocation_count(), 2);
    assert_eq!(counters.body(), 1, "the SUCCEED was recorded");
    assert_eq!(counters.after(), 1);
}

#[tokio::test]
async fn step_marked_replay_decode_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).deserialize_at(&[1]);
    let counters_h = counters.clone();
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let body = Arc::clone(&counters_h.body);
                let after = Arc::clone(&counters_h.after);
                async move {
                    let value = ctx
                        .step(move |_| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok("step-value".to_owned())
                            }
                        })
                        .name("decode-me")
                        .serdes(serdes)
                        .await?;
                    after.fetch_add(1, Ordering::SeqCst);
                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 3);
    assert_eq!(counters.body(), 1);
    assert_eq!(
        counters.after(),
        2,
        "the code after the step runs in the first and the last invocation, not in the faulting one"
    );
}

#[tokio::test]
async fn step_permanent_marked_failure_records_no_outcome() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).serialize_every();
    let result = run_one_step(
        LocalRunner::new().max_invocations(3),
        "encode-me",
        serdes,
        counters.clone(),
    )
    .await;

    assert!(result.is_suspended(), "{:?}", result.error_message());
    assert_eq!(result.error_type(), Some("RuntimeError"));
    let message = result.error_message().expect("the result names the fault");
    assert!(
        message.contains("serdes serialize failed for operation \"encode-me\""),
        "{message}"
    );
    let step = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("encode-me"))
        .expect("the step START is recorded");
    assert_eq!(step.status(), durable::OperationStatus::Started);
    assert_no_failed_operation(&result);
    assert_eq!(
        counters.body(),
        3,
        "each of the three invocations reruns the body"
    );
    assert_eq!(counters.after(), 0);
}

#[tokio::test]
async fn step_unmarked_serialize_failure_records_fail_once() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Unmarked).serialize_at(&[0]);
    let counters_h = counters.clone();
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let body = Arc::clone(&counters_h.body);
                let after = Arc::clone(&counters_h.after);
                async move {
                    let outcome = ctx
                        .step(move |_| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok("step-value".to_owned())
                            }
                        })
                        .name("encode-me")
                        .serdes(serdes)
                        .await;
                    after.fetch_add(1, Ordering::SeqCst);
                    let err = outcome.expect_err("an unmarked failure is catchable");
                    Ok::<_, BoxError>(step_kind_label(&err))
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

    assert_eq!(
        result.output().map(String::as_str),
        Some("SerializationFailed")
    );
    assert_eq!(result.invocation_count(), 1);
    assert_eq!(counters.body(), 1);
    assert_eq!(counters.after(), 1);
    let step = result
        .operations()
        .iter()
        .find(|op| op.name() == Some("encode-me"))
        .expect("the step is recorded");
    assert!(step.failed(), "{step:?}");
    assert_eq!(step.error_type(), Some("SerializationError"));
}

#[tokio::test]
async fn step_marker_inside_user_error_is_found() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Wrapped).serialize_at(&[0]);
    let result = run_one_step(LocalRunner::new(), "encode-me", serdes, counters.clone()).await;

    assert_eq!(result.output().map(String::as_str), Some("step-value"));
    assert_eq!(result.invocation_count(), 2);
    assert_eq!(counters.body(), 2);
    assert_no_failed_operation(&result);
    assert_eq!(counters.after(), 1);
}

/// Runs one child context with `serdes`, then optionally a wait.
async fn run_one_child(
    serdes: InjectedFailures,
    counters: Counters,
    wait_after: bool,
) -> TestResult<u32> {
    LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let body = Arc::clone(&counters.body);
                let after = Arc::clone(&counters.after);
                async move {
                    let value = ctx
                        .run_in_child_context(move |_child| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok(7_u32)
                            }
                        })
                        .name("child")
                        .serdes(serdes)
                        .await?;
                    after.fetch_add(1, Ordering::SeqCst);
                    if wait_after {
                        ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    }
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

#[tokio::test]
async fn child_marked_serialize_failure_reruns_body() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).serialize_at(&[0]);
    let result = run_one_child(serdes, counters.clone(), false).await;

    assert_eq!(result.output(), Some(&7));
    assert_eq!(counters.body(), 2);
    assert_no_failed_operation(&result);
    assert_eq!(counters.after(), 1);
}

#[tokio::test]
async fn child_marked_replay_decode_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).deserialize_at(&[1]);
    let result = run_one_child(serdes, counters.clone(), true).await;

    assert_eq!(result.output(), Some(&7));
    assert_eq!(result.invocation_count(), 3);
    assert_eq!(counters.body(), 1);
    assert_eq!(
        counters.after(),
        2,
        "the code after the child runs in the first and the last invocation, not in the faulting one"
    );
}

#[tokio::test]
async fn create_callback_marked_decode_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).deserialize_at(&[0]);
    let counters_h = counters.clone();
    let result = LocalRunner::new()
        .callback(CallbackReply::success(&"ok").expect("encodes"))
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let counters = counters_h.clone();
                async move {
                    let callback = ctx
                        .create_callback::<String>()
                        .name("approval")
                        .serdes(serdes)
                        .await?;
                    // A callback has no body. Count the handler code that
                    // runs between creating the callback and awaiting it.
                    counters.body.fetch_add(1, Ordering::SeqCst);
                    let value = callback.result().await?;
                    counters.after.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

    assert_eq!(result.output().map(String::as_str), Some("ok"));
    assert_no_failed_operation(&result);
    assert_eq!(
        counters.body(),
        2,
        "the faulting invocation and the next one each reach the code between creating and awaiting the callback"
    );
    assert_eq!(counters.after(), 1, "the faulting decode returns nothing");
}

/// Runs a `wait_for_condition` that counts to 2 with a 1 s delay.
async fn run_count_to_two(serdes: InjectedFailures, counters: Counters) -> TestResult<u32> {
    LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let body = Arc::clone(&counters.body);
                let after = Arc::clone(&counters.after);
                async move {
                    let count = ctx
                        .wait_for_condition(
                            move |_ctx, state: u32| {
                                let body = Arc::clone(&body);
                                async move {
                                    body.fetch_add(1, Ordering::SeqCst);
                                    Ok(state + 1)
                                }
                            },
                            0,
                        )
                        .wait_strategy_fn(|state: u32, _attempt| {
                            if state >= 2 {
                                WaitDecision::complete()
                            } else {
                                WaitDecision::continue_with(Duration::from_secs(1))
                            }
                        })
                        .name("count")
                        .serdes(serdes)
                        .await?;
                    after.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, BoxError>(count)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

#[tokio::test]
async fn wait_for_condition_marked_serialize_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).serialize_at(&[0]);
    let result = run_count_to_two(serdes, counters.clone()).await;

    assert_eq!(result.output(), Some(&2));
    assert_no_failed_operation(&result);
    assert_eq!(
        counters.body(),
        3,
        "the first check reruns because its marked failure recorded nothing; then the second check runs"
    );
    assert_eq!(counters.after(), 1);
}

#[tokio::test]
async fn wait_for_condition_marked_decode_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).deserialize_at(&[0]);
    let result = run_count_to_two(serdes, counters.clone()).await;

    assert_eq!(result.output(), Some(&2));
    assert_no_failed_operation(&result);
    assert_eq!(
        counters.body(),
        3,
        "the first check reruns because its marked decode failure recorded nothing; then the second check runs"
    );
    assert_eq!(counters.after(), 1);
}

/// Runs a map over `[1, 2]` with `serdes` as the item serdes and returns
/// the item results, failing on any failed item.
async fn run_map(
    serdes: InjectedFailures,
    nesting: NestingMode,
    counters: Counters,
) -> TestResult<Vec<u32>> {
    LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let body = Arc::clone(&counters.body);
                let after = Arc::clone(&counters.after);
                async move {
                    let batch = ctx
                        .map(vec![1_u32, 2], move |_child, item, _idx| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok(item * 10)
                            }
                        })
                        .name("items")
                        .nesting(nesting)
                        .serdes(serdes)
                        .await_batch()
                        .await?;
                    after.fetch_add(1, Ordering::SeqCst);
                    let mut values = Vec::new();
                    for item in batch.items {
                        let value = item
                            .result
                            .ok_or_else(|| format!("item {} has no result", item.index))?;
                        values.push(value);
                    }
                    Ok::<_, BoxError>(values)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner")
}

#[tokio::test]
async fn map_item_marked_serialize_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).serialize_at(&[0]);
    let result = run_map(serdes, NestingMode::Normal, counters.clone()).await;

    assert_eq!(result.output(), Some(&vec![10, 20]));
    assert_no_failed_operation(&result);
    assert_eq!(
        counters.body(),
        3,
        "both items run; the item whose encode failed runs again, the recorded item does not"
    );
    assert_eq!(counters.after(), 1);
}

#[tokio::test]
async fn map_flat_item_marked_decode_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).deserialize_at(&[0]);
    let result = run_map(serdes, NestingMode::Flat, counters.clone()).await;

    assert_eq!(result.output(), Some(&vec![10, 20]));
    assert_eq!(
        counters.body(),
        4,
        "flat items have no child checkpoints, so the next invocation reruns both items to rebuild their values"
    );
    assert_eq!(counters.after(), 1);
}

#[tokio::test]
async fn map_result_serdes_marked_decode_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).deserialize_at(&[0]);
    let counters_h = counters.clone();
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let body = Arc::clone(&counters_h.body);
                let after = Arc::clone(&counters_h.after);
                async move {
                    let values = ctx
                        .map(vec![1_u32, 2], move |_child, item, _idx| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok(item * 10)
                            }
                        })
                        .name("items")
                        .result_serdes(serdes)
                        .await?;
                    after.fetch_add(1, Ordering::SeqCst);
                    ctx.wait(Duration::from_secs(1)).name("boundary").await?;
                    Ok::<_, BoxError>(values)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 3);
    assert_eq!(
        counters.body(),
        2,
        "the result serdes decodes only on replay, after both items were recorded"
    );
    assert_eq!(
        counters.after(),
        2,
        "the code after the map runs in the first and the last invocation, not in the faulting one"
    );
}

#[tokio::test]
async fn parallel_branch_marked_serialize_failure_reinvokes() {
    let counters = Counters::default();
    let serdes = InjectedFailures::new(Mark::Marked).serialize_at(&[0]);
    let counters_h = counters.clone();
    let result = LocalRunner::new()
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = serdes.clone();
                let one = Arc::clone(&counters_h.body);
                let two = Arc::clone(&counters_h.body);
                let after = Arc::clone(&counters_h.after);
                async move {
                    let branches: Vec<durable::Branch<u32>> = vec![
                        durable::Branch::new("one", move |_child| async move {
                            one.fetch_add(1, Ordering::SeqCst);
                            Ok(1_u32)
                        }),
                        durable::Branch::new("two", move |_child| async move {
                            two.fetch_add(1, Ordering::SeqCst);
                            Ok(2_u32)
                        }),
                    ];
                    let values = ctx
                        .parallel(branches)
                        .name("fan-out")
                        .serdes(serdes)
                        .await?;
                    after.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, BoxError>(values)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");

    assert_eq!(result.output(), Some(&vec![1, 2]));
    assert_eq!(
        counters.body(),
        3,
        "both branches run; the branch whose encode failed runs again, the recorded branch does not"
    );
    assert_eq!(counters.after(), 1);
}

#[tokio::test]
async fn filesystem_serdes_write_failure_ends_invocation() {
    // A base_path that is a regular file makes `create_dir_all` fail.
    let base = std::env::temp_dir().join(format!(
        "serdes_faults_file_base_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    std::fs::write(&base, "not a directory").expect("create the file");
    let base_path = base.to_string_lossy().into_owned();

    let counters = Counters::default();
    let counters_h = counters.clone();
    let result = LocalRunner::new()
        .max_invocations(2)
        .run(
            move |_event: serde_json::Value, ctx: DurableContext| {
                let serdes = FileSystemSerdes::with_config(
                    base_path.clone(),
                    FileSystemSerdesConfig::builder()
                        .storage_mode(FileSystemSerdesMode::Always)
                        .build(),
                );
                let body = Arc::clone(&counters_h.body);
                let after = Arc::clone(&counters_h.after);
                async move {
                    let value = ctx
                        .step(move |_| {
                            let body = Arc::clone(&body);
                            async move {
                                body.fetch_add(1, Ordering::SeqCst);
                                Ok("step-value".to_owned())
                            }
                        })
                        .name("store")
                        .serdes(serdes)
                        .await?;
                    after.fetch_add(1, Ordering::SeqCst);
                    Ok::<_, BoxError>(value)
                }
            },
            serde_json::Value::Null,
        )
        .await
        .expect("runner");
    let _ = std::fs::remove_file(&base);

    assert!(result.is_suspended(), "{:?}", result.error_message());
    let message = result.error_message().expect("the result names the fault");
    assert!(message.contains("serdes serialize failed"), "{message}");
    assert_no_failed_operation(&result);
    assert_eq!(
        counters.body(),
        2,
        "each of the two invocations reruns the body"
    );
    assert_eq!(counters.after(), 0);
}
