//! Integration test for the live controls of a `LocalRunner` run, through
//! the public API only: a test waits for an operation, pauses the run while
//! a step runs, resumes it, and answers a `wait_for_callback`.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::test_util::{CallbackReply, LocalRunner, WaitingStatus};
use durable::{BoxError, DurableContext, OperationStatus};

/// Step `reserve-stock` (200 ms), the `wait_for_callback`
/// `manager-approval`, and step `ship-order`.
async fn pause_resume_handler(
    event: serde_json::Value,
    ctx: DurableContext,
) -> Result<serde_json::Value, BoxError> {
    let order_id = event
        .get("orderId")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("order-1")
        .to_owned();
    let id = order_id.clone();
    let reservation = ctx
        .step(move |_| async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            Ok(format!("reserved-{id}"))
        })
        .name("reserve-stock")
        .await?;
    let approval = ctx
        .wait_for_callback::<String, _, _>(|_step_ctx, _callback_id| async { Ok(()) })
        .name("manager-approval")
        .await?;
    let shipment = ctx
        .step(move |_| async move { Ok(format!("shipped-{order_id}")) })
        .name("ship-order")
        .await?;
    Ok(serde_json::json!({
        "reservation": reservation,
        "approval": approval,
        "shipment": shipment,
    }))
}

#[tokio::test]
async fn public_run_and_operation_handles() {
    let mut run = LocalRunner::new().run(
        pause_resume_handler,
        serde_json::json!({"orderId": "order-7"}),
    );
    let reserve = run.operation("reserve-stock");
    let approval = run.operation("manager-approval");
    reserve
        .wait_for_data(WaitingStatus::Started)
        .await
        .expect("started");
    run.pause().await.expect("paused");
    assert_eq!(
        reserve.snapshot().map(|op| op.status()),
        Some(OperationStatus::Succeeded)
    );
    assert!(approval.snapshot().is_none());
    assert!(
        tokio::time::timeout(Duration::from_millis(50), &mut run)
            .await
            .is_err(),
        "the paused run does not finish"
    );
    run.resume().await.expect("resumed");
    approval
        .wait_for_data(WaitingStatus::Submitted)
        .await
        .expect("submitted");
    approval
        .send_callback(CallbackReply::success(&"approved").expect("encodes"))
        .await
        .expect("sent");
    let result = run.await.expect("runner");
    assert_eq!(
        result.output(),
        Some(&serde_json::json!({
            "reservation": "reserved-order-7",
            "approval": "approved",
            "shipment": "shipped-order-7",
        }))
    );
}
