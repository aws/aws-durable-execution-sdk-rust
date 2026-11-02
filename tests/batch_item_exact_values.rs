//! A map item or parallel branch whose value holds an integer outside the
//! 64-bit range replays unchanged, and the batch record holds its exact
//! digits.
//!
//! Each test runs the batch, records what it returned, and then waits 1 s.
//! So the batch runs live in the first invocation and replays from its
//! record in the second.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::serdes::{FileSystemSerdes, SerdesContext};
use durable::test_util::{LocalRunner, TestResult};
use durable::{BoxError, Branch, DurableContext, JsonSerdes, PassthroughSerdes, Serdes};

/// The digits of `u128::MAX`.
const U128_MAX_TEXT: &str = "340282366920938463463374607431768211455";

/// What a batch returned in each invocation, and the run's result.
struct Run<T> {
    result: TestResult<Vec<T>>,
    seen: Vec<Vec<T>>,
}

impl<T> Run<T> {
    /// The stored payload of the batch operation named `b`.
    fn payload(&self) -> String {
        self.result
            .operation("b")
            .expect("the batch operation is recorded")
            .result()
            .expect("the batch operation has a payload")
            .to_owned()
    }
}

/// Runs a handler that maps one item returning `value` with the result
/// serdes `result_serdes()` builds, records what the map returned in each
/// invocation, and then waits 1 s.
async fn run_map_with<T, RS>(
    value: T,
    result_serdes: impl Fn() -> RS + Send + Sync + 'static,
) -> Run<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + Sync + 'static,
    RS: Serdes<String>,
{
    let seen: Arc<Mutex<Vec<Vec<T>>>> = Arc::default();
    let seen_in_handler = Arc::clone(&seen);
    let result_serdes = Arc::new(result_serdes);
    let result = LocalRunner::new()
        .run(
            move |(): (), ctx: DurableContext| {
                let seen = Arc::clone(&seen_in_handler);
                let value = value.clone();
                let result_serdes = Arc::clone(&result_serdes);
                async move {
                    let values = ctx
                        .map(vec![1_u32], move |_child, _item, _idx| {
                            let value = value.clone();
                            async move { Ok(value) }
                        })
                        .name("b")
                        .result_serdes(result_serdes())
                        .await?;
                    seen.lock().expect("lock").push(values.clone());
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(values)
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    let seen = seen.lock().expect("lock").clone();
    Run { result, seen }
}

/// Runs a handler that maps `values` with the default result serdes,
/// records what the map returned in each invocation, and then waits 1 s.
async fn run_map<T>(values: Vec<T>) -> Run<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone + Send + Sync + 'static,
{
    let seen: Arc<Mutex<Vec<Vec<T>>>> = Arc::default();
    let seen_in_handler = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |(): (), ctx: DurableContext| {
                let seen = Arc::clone(&seen_in_handler);
                let values = values.clone();
                async move {
                    let out = ctx
                        .map(values, |_child, item, _idx| async move { Ok(item) })
                        .name("b")
                        .await?;
                    seen.lock().expect("lock").push(out.clone());
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(out)
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    let seen = seen.lock().expect("lock").clone();
    Run { result, seen }
}

/// Asserts that the batch returned `[value]` in both invocations and that
/// the execution succeeded with it.
fn assert_replays<T>(run: &Run<T>, value: &T)
where
    T: PartialEq + std::fmt::Debug,
{
    assert!(
        run.result.is_success(),
        "the execution succeeds: {}",
        run.result
    );
    assert_eq!(run.seen.len(), 2, "two invocations ran the batch");
    for seen in &run.seen {
        assert_eq!(seen.as_slice(), std::slice::from_ref(value));
    }
    assert_eq!(
        run.result.output().map(Vec::as_slice),
        Some(std::slice::from_ref(value))
    );
}

#[tokio::test]
async fn map_u128_max_item_replays() {
    let run = run_map(vec![u128::MAX]).await;
    assert_replays(&run, &u128::MAX);
    let payload = run.payload();
    assert!(
        payload.contains(&format!(r#""result":{U128_MAX_TEXT}}}"#)),
        "the record holds the exact digits unquoted: {payload}"
    );
}

#[tokio::test]
async fn map_i128_min_item_replays() {
    let run = run_map(vec![i128::MIN]).await;
    assert_replays(&run, &i128::MIN);
    let payload = run.payload();
    assert!(
        payload.contains(&format!(r#""result":{}}}"#, i128::MIN)),
        "the record holds the exact digits unquoted: {payload}"
    );
}

#[tokio::test]
async fn parallel_u128_max_branch_replays() {
    let seen: Arc<Mutex<Vec<Vec<u128>>>> = Arc::default();
    let seen_in_handler = Arc::clone(&seen);
    let result = LocalRunner::new()
        .run(
            move |(): (), ctx: DurableContext| {
                let seen = Arc::clone(&seen_in_handler);
                async move {
                    let values = ctx
                        .parallel(vec![Branch::new(
                            "only",
                            |_child| async move { Ok(u128::MAX) },
                        )])
                        .name("b")
                        .await?;
                    seen.lock().expect("lock").push(values.clone());
                    ctx.wait(Duration::from_secs(1)).name("after").await?;
                    Ok::<_, BoxError>(values)
                }
            },
            (),
        )
        .await
        .expect("the run finishes");
    let run = Run {
        result,
        seen: seen.lock().expect("lock").clone(),
    };
    assert_replays(&run, &u128::MAX);
    let payload = run.payload();
    assert!(
        payload.contains(&format!(r#""result":{U128_MAX_TEXT}}}"#)),
        "the record holds the exact digits unquoted: {payload}"
    );
}

#[tokio::test]
async fn default_record_is_the_json_text() {
    let run = run_map(vec!["a".to_owned(), "b".to_owned()]).await;
    assert!(run.result.is_success(), "{}", run.result);
    assert_eq!(
        run.payload(),
        concat!(
            r#"{"all":[{"index":0,"status":"SUCCEEDED","name":"map-item-0","result":"a"},"#,
            r#"{"index":1,"status":"SUCCEEDED","name":"map-item-1","result":"b"}],"#,
            r#""completionReason":"ALL_COMPLETED"}"#
        )
    );
}

#[tokio::test]
async fn default_result_serdes_is_passthrough() {
    let run = run_map_with(u128::MAX, || PassthroughSerdes).await;
    assert_replays(&run, &u128::MAX);
    assert!(run.payload().starts_with(r#"{"all":"#), "{}", run.payload());
}

#[tokio::test]
async fn json_result_serdes_keeps_the_value_exact() {
    let run = run_map_with(u128::MAX, || JsonSerdes).await;
    assert_replays(&run, &u128::MAX);
    let payload = run.payload();
    let record: String =
        serde_json::from_str(&payload).expect("JsonSerdes stores a quoted JSON string");
    assert!(
        record.contains(&format!(r#""result":{U128_MAX_TEXT}}}"#)),
        "the quoted record holds the exact digits: {record}"
    );
}

/// A fresh temp directory per test, removed on drop.
struct TempBase(std::path::PathBuf);

impl Drop for TempBase {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[tokio::test]
async fn filesystem_result_serdes_keeps_the_value_exact() {
    let base = TempBase(
        std::env::temp_dir().join(format!("batch_item_exact_values_{}", std::process::id())),
    );
    let _ = std::fs::remove_dir_all(&base.0);
    let base_path = base.0.to_string_lossy().into_owned();
    let run = run_map_with(u128::MAX, move || FileSystemSerdes::new(base_path.clone())).await;
    assert_replays(&run, &u128::MAX);
    assert!(
        !run.payload().contains(U128_MAX_TEXT),
        "the payload is a file reference: {}",
        run.payload()
    );
}

/// A `Serdes<String>` that adds a prefix and removes it on replay.
struct PrefixSerdes;

const PREFIX: &str = "prefix:";

impl Serdes<String> for PrefixSerdes {
    fn serialize(
        &self,
        value: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(Ok(format!("{PREFIX}{value}")))
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(
            wire.strip_prefix(PREFIX)
                .map(str::to_owned)
                .ok_or_else(|| BoxError::from("the payload has no prefix")),
        )
    }
}

#[tokio::test]
async fn prefix_result_serdes_keeps_the_value_exact() {
    let run = run_map_with(u128::MAX, || PrefixSerdes).await;
    assert_replays(&run, &u128::MAX);
    assert!(
        run.payload().starts_with(r#"prefix:{"all":"#),
        "{}",
        run.payload()
    );
}

/// A type-agnostic serdes that passes its value through
/// `serde_json::Value`.
struct ValueSerdes;

impl<T> Serdes<T> for ValueSerdes
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    fn serialize(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        std::future::ready(
            serde_json::to_value(value)
                .and_then(|value| serde_json::to_string(&value))
                .map_err(Into::into),
        )
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<T, BoxError>> + Send {
        std::future::ready(
            serde_json::from_str::<serde_json::Value>(&wire)
                .and_then(serde_json::from_value)
                .map_err(Into::into),
        )
    }
}

#[tokio::test]
async fn value_result_serdes_keeps_the_value_exact() {
    let run = run_map_with(u128::MAX, || ValueSerdes).await;
    assert_replays(&run, &u128::MAX);
}

/// A type-agnostic CBOR serdes. The CBOR bytes are stored as hex text.
struct CborSerdes;

impl<T> Serdes<T> for CborSerdes
where
    T: serde::Serialize + serde::de::DeserializeOwned + Send + 'static,
{
    fn serialize(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<String, BoxError>> + Send {
        let mut bytes = Vec::new();
        let result = ciborium::into_writer(&value, &mut bytes)
            .map(|()| {
                bytes.iter().fold(String::new(), |mut hex, byte| {
                    let _ = write!(hex, "{byte:02x}");
                    hex
                })
            })
            .map_err(Into::into);
        std::future::ready(result)
    }

    fn deserialize(
        &self,
        wire: String,
        _context: SerdesContext,
    ) -> impl Future<Output = Result<T, BoxError>> + Send {
        let bytes: Result<Vec<u8>, BoxError> = (0..wire.len())
            .step_by(2)
            .map(|at| {
                wire.get(at..at + 2)
                    .ok_or_else(|| BoxError::from("odd hex length"))
                    .and_then(|pair| u8::from_str_radix(pair, 16).map_err(Into::into))
            })
            .collect();
        std::future::ready(
            bytes.and_then(|bytes| ciborium::from_reader(bytes.as_slice()).map_err(Into::into)),
        )
    }
}

#[tokio::test]
async fn cbor_result_serdes_keeps_the_value_exact() {
    let run = run_map_with(u128::MAX, || CborSerdes).await;
    assert_replays(&run, &u128::MAX);
    assert!(
        !run.payload().contains(U128_MAX_TEXT),
        "the payload is CBOR hex: {}",
        run.payload()
    );
}
