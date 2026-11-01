//! The operation times and the printable table of `TestResult`, through the
//! public API only.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used)] // reason: test assertions

use std::time::{Duration, SystemTime};

use aws_durable_execution_sdk as durable;
use durable::test_util::{LocalRunner, PrintColumns, TestOperationDetails};
use durable::{BoxError, DurableContext, OperationKind, OperationStatus};

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
