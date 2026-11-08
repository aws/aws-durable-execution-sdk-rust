//! The operation times and the printable table of `TestResult`, through the
//! public API only.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::{Duration, SystemTime};

use aws_durable_execution_sdk as durable;
use durable::test_util::{CallbackReply, LocalRunner, PrintColumns, TestOperationDetails};
use durable::{BoxError, DurableContext, OperationKind, OperationStatus, OperationSubType};

/// A wait's start, end and scheduled end come from the virtual clock, which
/// starts at the time the run started.
#[tokio::test]
async fn operation_times_are_public() {
    let before = SystemTime::now();
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
                ctx.wait(Duration::from_secs(10)).name("pause").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let pause = result.operation("pause").expect("the wait");
    assert_eq!(pause.kind(), OperationKind::Wait);
    assert_eq!(pause.status(), OperationStatus::Succeeded);
    let start = pause.start_time().expect("a start time");
    assert!(start >= before, "{start:?} is before {before:?}");
    let ten_later = Some(start + Duration::from_secs(10));
    assert_eq!(pause.end_time(), ten_later);
    let scheduled_end = match pause.details() {
        TestOperationDetails::Wait(wait) => wait.scheduled_end_time(),
        _ => None,
    };
    assert_eq!(scheduled_end, ten_later);
}

/// `display_with` prints the selected columns of the table.
#[tokio::test]
async fn display_is_public() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.step(|_| async { Ok(1_u32) }).name("fetch").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let mut columns = PrintColumns::default();
    columns.parent_id = false;
    columns.sub_type = false;
    columns.start_time = false;
    columns.end_time = false;
    columns.duration = false;
    assert_eq!(
        result.display_with(columns).to_string(),
        "name | type | status\nfetch | STEP | SUCCEEDED\n"
    );
}

/// `result_json` decodes a step's JSON result into the caller's type, and
/// reports no result for a wait.
#[tokio::test]
async fn result_json_decodes_a_step_result() {
    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Receipt {
        order: String,
        total: u32,
    }
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.step(|_| async {
                    Ok(Receipt {
                        order: "o-1".to_owned(),
                        total: 42,
                    })
                })
                .name("charge")
                .await?;
                ctx.wait(Duration::from_secs(1)).name("pause").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    let charge = result.operation("charge").expect("the step");
    assert_eq!(
        charge.result_json::<Receipt>().expect("decodes"),
        Some(Receipt {
            order: "o-1".to_owned(),
            total: 42,
        })
    );
    assert!(charge.result_json::<u32>().is_err());
    let pause = result.operation("pause").expect("the wait");
    assert_eq!(pause.result_json::<Receipt>().expect("no result"), None);
}

/// `wait_duration` reads a wait's recorded duration, and is `None` for a
/// step.
#[tokio::test]
async fn wait_duration_reads_only_a_wait() {
    let result = LocalRunner::new()
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.wait(Duration::from_secs(10)).name("pause").await?;
                ctx.step(|_| async { Ok(1_u32) }).name("a").await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");

    assert_eq!(
        result.operation("pause").expect("the wait").wait_duration(),
        Some(Duration::from_secs(10))
    );
    assert_eq!(
        result.operation("a").expect("the step").wait_duration(),
        None
    );
}

/// Only the context of a `wait_for_callback` is marked: not its callback
/// child, a `create_callback`, a plain child context, or a step whose custom
/// subtype is `WaitForCallback`.
#[tokio::test]
async fn is_wait_for_callback_marks_only_its_context() {
    let result = LocalRunner::new()
        .callback_for("approve", CallbackReply::success(&"yes").expect("encodes"))
        .callback_for("cb", CallbackReply::success(&"ok").expect("encodes"))
        .run(
            |(): (), ctx: DurableContext| async move {
                ctx.wait_for_callback::<String>(|_step, _callback_id| async { Ok(()) })
                    .name("approve")
                    .await?;
                let cb = ctx.create_callback::<String>().name("cb").await?;
                cb.result().await?;
                ctx.run_in_child_context(|child| async move {
                    child
                        .step(|_| async { Ok(1_u32) })
                        .name("inner")
                        .await
                        .map_err(BoxError::from)
                })
                .name("child")
                .await?;
                ctx.step(|_| async { Ok(2_u32) })
                    .name("custom")
                    .sub_type(OperationSubType::new("WaitForCallback")?)
                    .await?;
                Ok::<_, BoxError>(())
            },
            (),
        )
        .await
        .expect("runner");
    assert!(result.is_success(), "{result}");

    let approve = result.operation("approve").expect("the wait_for_callback");
    assert!(approve.is_wait_for_callback());
    for name in ["cb", "child", "custom"] {
        let op = result.operation(name).expect(name);
        assert!(!op.is_wait_for_callback(), "{name}");
    }
    let inner_callback = result
        .operations()
        .iter()
        .find(|op| op.kind() == OperationKind::Callback && op.parent_id() == Some(approve.id()))
        .expect("the callback of approve");
    assert!(!inner_callback.is_wait_for_callback());
}
