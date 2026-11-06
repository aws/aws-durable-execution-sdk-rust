//! Integration tests for a checkpoint response without a `CheckpointToken`.
//!
//! Such a response means the service accepts no further checkpoints from
//! the invocation. The SDK sends nothing more and answers `PENDING`, and
//! the next invocation replays the accepted updates.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use aws_durable_execution_sdk as durable;
use aws_sdk_lambda::operation::checkpoint_durable_execution::CheckpointDurableExecutionOutput;
use aws_smithy_mocks::{RuleMode, mock, mock_client};
use durable::test_util::LocalRunner;
use durable::{BoxError, DurableContext, StepSemantics, lambda_runtime};

/// A port of the Python SDK's `test_checkpoint_without_token_answers_pending`:
/// `wrap` over a Lambda client whose checkpoint response has no token. The
/// handler's one step sends one request and the invocation answers
/// `PENDING`.
#[tokio::test]
async fn wrap_with_tokenless_lambda_client_answers_pending() {
    let rule = mock!(aws_sdk_lambda::Client::checkpoint_durable_execution)
        .then_output(|| CheckpointDurableExecutionOutput::builder().build());
    let state = mock!(aws_sdk_lambda::Client::get_durable_execution_state).then_output(|| {
        aws_sdk_lambda::operation::get_durable_execution_state::GetDurableExecutionStateOutput::builder()
            .set_operations(Some(Vec::new()))
            .build()
            .expect("valid output")
    });
    let client = mock_client!(aws_sdk_lambda, RuleMode::MatchAny, [&rule, &state]);
    let options = durable::Options::builder()
        .lambda_client(client)
        .build()
        .expect("options build");
    let service = durable::wrap(
        |_event: serde_json::Value, ctx: DurableContext| async move {
            let value = ctx
                .step(|_| async { Ok("x".to_owned()) })
                .name("s1")
                .await?;
            Ok::<String, BoxError>(value)
        },
        options,
    );
    let payload: durable::InvocationPayload = serde_json::from_value(serde_json::json!({
        "DurableExecutionArn": "arn:aws:lambda:us-east-1:1:function:test",
        "CheckpointToken": "token-0",
        "InitialExecutionState": {
            "Operations": [{
                "Id": "root",
                "Type": "EXECUTION",
                "Status": "STARTED",
                "ExecutionDetails": { "InputPayload": "{}" }
            }]
        }
    }))
    .expect("valid payload");
    let response = service(lambda_runtime::LambdaEvent::new(
        payload,
        lambda_runtime::Context::default(),
    ))
    .await
    .expect("the invocation answers a response");
    assert_eq!(
        serde_json::to_value(&response).expect("the response serializes"),
        serde_json::json!({ "Status": "PENDING" })
    );
    assert_eq!(rule.num_calls(), 1);
    assert_eq!(state.num_calls(), 0);
}

/// Step, `wait(1s)`, step. The handler of
/// `withdraw_checkpoints_after_reinvokes_under_run`.
async fn step_wait_step(_event: (), ctx: DurableContext) -> Result<u32, BoxError> {
    let first = ctx.step(|_| async { Ok(1_u32) }).name("first").await?;
    ctx.wait(Duration::from_secs(1)).name("pause").await?;
    let second = ctx
        .step(move |_| async move { Ok(first + 1) })
        .name("second")
        .await?;
    Ok(second)
}

#[tokio::test]
async fn withdraw_checkpoints_after_reinvokes_under_run() {
    let plain = LocalRunner::new()
        .run(step_wait_step, ())
        .await
        .expect("runner");
    let withdrawn = LocalRunner::new()
        .withdraw_checkpoints_after(1)
        .run(step_wait_step, ())
        .await
        .expect("runner");
    assert!(plain.is_success());
    assert!(withdrawn.is_success());
    assert_eq!(withdrawn.output(), plain.output());
    assert_eq!(withdrawn.invocation_count(), plain.invocation_count() + 1);
}

#[tokio::test]
async fn at_most_once_start_in_withdrawn_request_not_rerun() {
    let runs = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&runs);
    let result = LocalRunner::new()
        .withdraw_checkpoints_after(0)
        .run(
            move |_event: (), ctx: DurableContext| {
                let counted = Arc::clone(&counted);
                async move {
                    let value = ctx
                        .step(move |_| {
                            let counted = Arc::clone(&counted);
                            async move {
                                counted.fetch_add(1, Ordering::SeqCst);
                                Ok(7_u32)
                            }
                        })
                        .name("once")
                        .semantics(StepSemantics::AtMostOncePerRetry)
                        .await?;
                    Ok::<u32, BoxError>(value)
                }
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output(), Some(&7));
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1,
        "the step body runs once across both invocations"
    );
    assert!(result.invocation_count() >= 2);
}

/// A failed request is not a withdrawal: a retryable failure of the chosen
/// request re-invokes as a fault, and the run takes the invocations it
/// takes with the failure alone.
#[tokio::test]
async fn failed_request_is_not_withdrawn() {
    let failed = LocalRunner::new()
        .fail_checkpoints_after_retryable(0, 1)
        .run(step_wait_step, ())
        .await
        .expect("runner");
    let both = LocalRunner::new()
        .fail_checkpoints_after_retryable(0, 1)
        .withdraw_checkpoints_after(0)
        .run(step_wait_step, ())
        .await
        .expect("runner");
    assert!(both.is_success());
    assert_eq!(both.output(), failed.output());
    assert_eq!(both.invocation_count(), failed.invocation_count());
}

/// A nested durable run does not inherit the withdrawal.
#[tokio::test]
async fn nested_run_does_not_withdraw() {
    let nested_invocations = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&nested_invocations);
    let result = LocalRunner::new()
        .withdraw_checkpoints_after(0)
        .register_durable_function("target", move |_event: (), ctx: DurableContext| {
            let counted = Arc::clone(&counted);
            async move {
                counted.fetch_add(1, Ordering::SeqCst);
                let value = ctx.step(|_| async { Ok(5_u32) }).name("inner").await?;
                Ok::<u32, BoxError>(value)
            }
        })
        .run(
            |_event: (), ctx: DurableContext| async move {
                let value = ctx.invoke::<u32, _>("target", ()).name("call").await?;
                Ok::<u32, BoxError>(value)
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{:?}", result.error_message());
    assert_eq!(result.output(), Some(&5));
    assert_eq!(nested_invocations.load(Ordering::SeqCst), 1);
}
