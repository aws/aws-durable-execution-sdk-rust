//! The public handler bounds do not require a `'static` handler future.
//!
//! The SDK runs the handler as the invocation's root task, which needs a
//! `'static` future. The SDK builds that future itself: an adapter that
//! owns the handler, the event and the context, and awaits the handler's
//! future inside itself. So the handler's own future may borrow. The
//! helpers below copy today's bounds of `wrap`, `run`, `run_with_options`
//! and `LocalRunner::run` exactly, without `Fut: 'static`; this file
//! compiles only while the public bounds accept them.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use aws_durable_execution_sdk as durable;
use durable::test_util::{LocalRunner, TestResult, TestRunnerError};
use durable::{BoxError, DurableContext, Options};
use serde::Serialize;
use serde::de::DeserializeOwned;

fn forward_wrap<F, E, Fut, O>(handler: F, options: Options)
where
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    E: for<'de> serde::Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    drop(durable::wrap(handler, options));
}

async fn forward_run<F, E, Fut, O>(handler: F) -> Result<(), lambda_runtime::Error>
where
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    E: for<'de> serde::Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    durable::run(handler).await
}

async fn forward_run_with_options<F, E, Fut, O>(
    handler: F,
    options: Options,
) -> Result<(), lambda_runtime::Error>
where
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    E: for<'de> serde::Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    durable::run_with_options(handler, options).await
}

async fn forward_local_run<E, O, F, Fut>(
    runner: &LocalRunner,
    handler: F,
    event: E,
) -> Result<TestResult<O>, TestRunnerError>
where
    E: Serialize + DeserializeOwned + Send + 'static,
    O: Serialize + DeserializeOwned + Send + 'static,
    F: Fn(E, DurableContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
{
    runner.run(handler, event).await
}

fn assert_send<T: Send>(_: &T) {}

/// The futures of the `run` and `run_with_options` helpers are `Send` for
/// every handler the bounds accept. The futures are never polled.
fn run_futures_are_send<F, E, Fut, O>(handler: F)
where
    F: Fn(E, DurableContext) -> Fut + Clone + Send + Sync + 'static,
    E: for<'de> serde::Deserialize<'de> + Send + 'static,
    Fut: Future<Output = Result<O, BoxError>> + Send,
    O: Serialize + Send + 'static,
{
    let run = forward_run(handler.clone());
    assert_send(&run);
    let run_with_options = forward_run_with_options(handler, Options::default());
    assert_send(&run_with_options);
}

/// A handler future with a non-`'static` lifetime parameter: it returns
/// its input plus 1.
struct Borrowing<'a> {
    value: u32,
    _borrow: PhantomData<&'a ()>,
}

impl Future for Borrowing<'_> {
    type Output = Result<u32, BoxError>;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(Ok(self.value + 1))
    }
}

/// Runs a handler whose future type borrows for `'a`.
#[expect(clippy::extra_unused_lifetimes)]
// reason: the handler future names `'a`, a lifetime that is not `'static`;
// that is what the test checks the bounds accept.
async fn run_borrowing<'a>() -> Result<TestResult<u32>, TestRunnerError> {
    forward_local_run(
        &LocalRunner::new(),
        |value: u32, _ctx: DurableContext| Borrowing::<'a> {
            value,
            _borrow: PhantomData,
        },
        41,
    )
    .await
}

#[tokio::test]
async fn handler_bounds_are_unchanged() {
    forward_wrap(
        |event: serde_json::Value, _ctx: DurableContext| async move { Ok::<_, BoxError>(event) },
        Options::default(),
    );
    run_futures_are_send(
        |event: serde_json::Value, _ctx: DurableContext| async move { Ok::<_, BoxError>(event) },
    );

    let borrowing = run_borrowing().await.expect("runner");
    assert!(borrowing.is_success(), "{:?}", borrowing.error_message());
    assert_eq!(borrowing.output(), Some(&42));

    let one_step = forward_local_run(
        &LocalRunner::new(),
        |value: u32, ctx: DurableContext| async move {
            let next = ctx.step(move |_| async move { Ok(value + 1) }).await?;
            Ok::<_, BoxError>(next)
        },
        1,
    )
    .await
    .expect("runner");
    assert!(one_step.is_success(), "{:?}", one_step.error_message());
    assert_eq!(one_step.output(), Some(&2));
}
