//! `wait_for_callback`, `invoke` and `CloudRunner::run` take the output
//! type as their only explicit type parameter. The submitter is a
//! `Submitter`, an invoke input needs only `Serialize`, and an invoke takes
//! a `PayloadSerializer`.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::builders::Submitter;
use durable::serdes::SerdesContext;
use durable::test_util::{CallbackReply, LocalRunner};
use durable::{BoxError, DurableContext, DurableFuture, PayloadSerializer, RetryDecision};
use serde_json::value::RawValue;

/// An input type that derives `Serialize` and no `Deserialize`.
#[derive(serde::Serialize)]
struct Charge {
    amount: u32,
}

/// A payload serializer that uppercases the JSON of its input.
struct UppercaseJson;

impl PayloadSerializer for UppercaseJson {
    // reason: exercises the async-fn impl form user code writes
    #[expect(clippy::unused_async_trait_impl)]
    async fn serialize_payload<T>(
        &self,
        value: T,
        _context: SerdesContext,
    ) -> Result<String, BoxError>
    where
        T: serde::Serialize + Send + 'static,
    {
        Ok(serde_json::to_string(&value)?.to_uppercase())
    }
}

fn assert_send_static<T: Send + 'static>(_: &T) {}

/// The submitter's closure parameters stay unannotated, and method calls on
/// both compile.
#[tokio::test]
async fn wait_for_callback_takes_only_the_output_type() {
    let result = LocalRunner::new()
        .callback_for("approval", CallbackReply::success(&"yes").expect("reply"))
        .run(
            |(): (), ctx: DurableContext| async move {
                let answer = ctx
                    .wait_for_callback::<String>(|step_ctx, callback_id| async move {
                        assert!(step_ctx.attempt() >= 1);
                        assert_ne!(callback_id, "");
                        let _ = callback_id.len();
                        Ok(())
                    })
                    .name("approval")
                    .await?;
                Ok::<_, BoxError>(answer)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output().map(String::as_str),
        Some("yes"),
        "{:?}",
        result.error_message()
    );
}

/// A submitter that fails once and succeeds on its retry returns the queued
/// reply.
#[tokio::test]
async fn wait_for_callback_submitter_retries() {
    let attempts = Arc::new(AtomicU32::new(0));
    let counter = Arc::clone(&attempts);
    let result = LocalRunner::new()
        .callback_for("approval", CallbackReply::success(&"yes").expect("reply"))
        .run(
            move |(): (), ctx: DurableContext| {
                let counter = Arc::clone(&counter);
                async move {
                    let answer = ctx
                        .wait_for_callback::<String>(move |step_ctx, _callback_id| {
                            let counter = Arc::clone(&counter);
                            async move {
                                counter.fetch_add(1, Ordering::SeqCst);
                                if step_ctx.attempt() == 1 {
                                    return Err::<(), BoxError>("first attempt fails".into());
                                }
                                Ok(())
                            }
                        })
                        .name("approval")
                        .submitter_retry(|_err, attempt| {
                            if attempt >= 2 {
                                RetryDecision::Stop
                            } else {
                                RetryDecision::retry(Duration::from_secs(1))
                            }
                        })
                        .await?;
                    Ok::<_, BoxError>(answer)
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output().map(String::as_str),
        Some("yes"),
        "{:?}",
        result.error_message()
    );
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn invoke_takes_only_the_output_type() {
    let result = LocalRunner::new()
        .register_function("double", |x: i32| async move { Ok(x * 2) })
        .run(
            |(): (), ctx: DurableContext| async move {
                let doubled = ctx.invoke::<i32>("double", 21).await?;
                Ok::<_, BoxError>(doubled)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(result.output(), Some(&42), "{:?}", result.error_message());
}

#[tokio::test]
async fn invoke_input_needs_only_serialize() {
    let result = LocalRunner::new()
        .register_function("echo", |input: serde_json::Value| async move { Ok(input) })
        .run(
            |(): (), ctx: DurableContext| async move {
                let echoed: serde_json::Value = ctx.invoke("echo", Charge { amount: 100 }).await?;
                Ok::<_, BoxError>(echoed)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output(),
        Some(&serde_json::json!({"amount": 100})),
        "{:?}",
        result.error_message()
    );
}

#[tokio::test]
async fn custom_payload_serializer_encodes_the_input() {
    let result = LocalRunner::new()
        .register_function("echo", |s: String| async move { Ok(s) })
        .run(
            |(): (), ctx: DurableContext| async move {
                let direct = ctx
                    .invoke::<String>("echo", "hello")
                    .name("direct")
                    .payload_serdes(UppercaseJson)
                    .await?;
                let shared = ctx
                    .invoke::<String>("echo", "hello")
                    .name("shared")
                    .payload_serdes(Arc::new(UppercaseJson))
                    .await?;
                Ok::<_, BoxError>((direct, shared))
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output(),
        Some(&("HELLO".to_owned(), "HELLO".to_owned())),
        "{:?}",
        result.error_message()
    );
}

#[tokio::test]
async fn raw_json_input_is_sent_unchanged() {
    let result = LocalRunner::new()
        .register_function("echo", |input: serde_json::Value| async move { Ok(input) })
        .run(
            |(): (), ctx: DurableContext| async move {
                let raw: Box<RawValue> = RawValue::from_string(r#"{"amount":100}"#.to_owned())?;
                let echoed = ctx.invoke::<serde_json::Value>("echo", raw).await?;
                Ok::<_, BoxError>(echoed)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output(),
        Some(&serde_json::json!({"amount": 100})),
        "{:?}",
        result.error_message()
    );
}

/// Both builders are `Send + 'static`, and both run after the context they
/// came from was moved and dropped.
#[tokio::test]
async fn builders_outlive_the_context_they_came_from() {
    let result = LocalRunner::new()
        .callback_for("approval", CallbackReply::success(&"yes").expect("reply"))
        .register_function("double", |x: i32| async move { Ok(x * 2) })
        .run(
            |(): (), ctx: DurableContext| async move {
                let approval = ctx
                    .wait_for_callback::<String>(|_step_ctx, _callback_id| async { Ok(()) })
                    .name("approval");
                let doubled = ctx.invoke::<i32>("double", 21).name("double");
                assert_send_static(&approval);
                assert_send_static(&doubled);
                let moved = ctx;
                drop(moved);
                let answer = approval.await?;
                let value = doubled.await?;
                Ok::<_, BoxError>((answer, value))
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output(),
        Some(&("yes".to_owned(), 42)),
        "{:?}",
        result.error_message()
    );
}

/// A generic helper bounded by `Submitter` returns the operation's future.
fn approval<F: Submitter>(ctx: &DurableContext, submitter: F) -> DurableFuture<String> {
    ctx.wait_for_callback::<String>(submitter)
        .name("approval")
        .future()
}

#[tokio::test]
async fn submitter_bounds_a_generic_helper() {
    let result = LocalRunner::new()
        .callback_for("approval", CallbackReply::success(&"yes").expect("reply"))
        .run(
            |(): (), ctx: DurableContext| async move {
                let answer = approval(&ctx, |_step_ctx, _callback_id| async { Ok(()) }).await?;
                Ok::<_, BoxError>(answer)
            },
            (),
        )
        .await
        .expect("runner");
    assert_eq!(
        result.output().map(String::as_str),
        Some("yes"),
        "{:?}",
        result.error_message()
    );
}
