//! Integration tests for chained invokes against targets registered on
//! `LocalRunner`, through the crate's public API.
//!
//! `register_function` registers a plain target, and
//! `register_durable_function` registers a durable target that runs as its
//! own execution. The runner runs each pending invoke's target once,
//! between the caller's invocations, and the caller's replay reads the
//! recorded outcome.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{LocalRunner, TestOperation, TestResult};
use durable::{BoxError, DurableContext, TypedError, WireError};

/// The one chained invoke record of `result`.
fn invoke_record<O>(result: &TestResult<O>) -> &TestOperation {
    let mut invokes = result
        .operations()
        .iter()
        .filter(|op| op.op_type() == "ChainedInvoke");
    let invoke = invokes.next().expect("a chained invoke record");
    assert!(invokes.next().is_none(), "one chained invoke record");
    invoke
}

/// A plain target's result reaches the caller, and a later operation in
/// the caller replays the invoke without running the target again.
#[tokio::test]
async fn plain_target_result_survives_replay() {
    let calls = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&calls);
    let result = LocalRunner::new()
        .register_function("double", move |x: i32| {
            let counter = Arc::clone(&counter);
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(x * 2)
            }
        })
        .run(
            |n: i32, ctx: DurableContext| async move {
                let doubled = ctx.invoke::<i32, _>("double", n).name("double").await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                let plus = ctx
                    .step(move |_| async move { Ok(doubled + 1) })
                    .name("inc")
                    .await?;
                Ok::<_, BoxError>(plus)
            },
            20_i32,
        )
        .await;

    assert_eq!(result.output(), Some(&41), "{:?}", result.error_message());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.invocation_count(), 3);
    let invoke = invoke_record(&result);
    assert_eq!(invoke.status(), "Succeeded");
    assert_eq!(invoke.result(), Some("40"));
}

/// A plain target's typed error reaches the caller with its type and
/// error data.
#[tokio::test]
async fn plain_target_typed_error_reaches_caller() {
    let result = LocalRunner::new()
        .register_function("pay", |(): ()| async move {
            Err::<(), BoxError>(
                TypedError::with_type("PaymentError", "declined")
                    .with_error_data("{\"code\":7}")
                    .into(),
            )
        })
        .run(
            |(): (), ctx: DurableContext| async move {
                let err = ctx
                    .invoke::<(), _>("pay", ())
                    .await
                    .expect_err("the invoke fails");
                let wire = err.wire();
                Ok::<_, BoxError>((
                    wire.and_then(WireError::error_type).map(ToOwned::to_owned),
                    wire.and_then(WireError::error_data).map(ToOwned::to_owned),
                ))
            },
            (),
        )
        .await;

    assert_eq!(
        result.output(),
        Some(&(
            Some("PaymentError".to_owned()),
            Some("{\"code\":7}".to_owned())
        )),
        "{:?}",
        result.error_message()
    );
    let invoke = invoke_record(&result);
    assert_eq!(invoke.status(), "Failed");
    assert_eq!(invoke.error_type(), Some("PaymentError"));
}

/// A durable target steps and waits as its own execution, and its result
/// reaches the caller after one more caller invocation.
#[tokio::test]
async fn durable_target_runs_as_its_own_execution() {
    let result = LocalRunner::new()
        .register_durable_function("workflow", |n: u32, ctx: DurableContext| async move {
            let next = ctx.step(move |_| async move { Ok(n + 1) }).await?;
            ctx.wait(Duration::from_secs(5)).await?;
            let doubled = ctx.step(move |_| async move { Ok(next * 2) }).await?;
            Ok::<_, BoxError>(doubled)
        })
        .run(
            |n: u32, ctx: DurableContext| async move {
                let out = ctx.invoke::<u32, _>("workflow", n).await?;
                Ok::<_, BoxError>(out)
            },
            1_u32,
        )
        .await;

    assert_eq!(result.output(), Some(&4), "{:?}", result.error_message());
    assert_eq!(result.invocation_count(), 2);
    // The nested execution's operations stay separate.
    let invoke = invoke_record(&result);
    assert_eq!(invoke.status(), "Succeeded");
    assert_eq!(invoke.result(), Some("4"));
}

/// A durable target that waits for a callback no reply answers leaves the
/// invoke started, and the caller's suspended message names it.
#[tokio::test]
async fn suspended_durable_target_names_the_pending_invoke() {
    let result = LocalRunner::new()
        .register_durable_function("approval", |(): (), ctx: DurableContext| async move {
            let callback = ctx.create_callback::<bool>().await?;
            let approved = callback.result().await?;
            Ok::<_, BoxError>(approved)
        })
        .run(
            |(): (), ctx: DurableContext| async move {
                let approved = ctx.invoke::<bool, _>("approval", ()).await?;
                Ok::<_, BoxError>(approved)
            },
            (),
        )
        .await;

    assert!(result.is_suspended(), "{:?}", result.error_message());
    assert_eq!(invoke_record(&result).status(), "Started");
    let message = result.error_message().expect("a suspended message");
    assert!(
        message.ends_with("; chained invokes still pending for: \"approval\""),
        "{message}"
    );
}

/// An invoke of a name nothing is registered under fails the run with a
/// message naming the function.
#[tokio::test]
async fn unregistered_target_fails_the_run() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.invoke::<(), _>("child-fn", ()).await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await;

    assert!(result.is_failure(), "{:?}", result.error_message());
    let message = result.error_message().expect("a failure message");
    assert!(
        message.contains(
            "No function found for function name \"child-fn\". Register it on the LocalRunner."
        ),
        "{message}"
    );
}
