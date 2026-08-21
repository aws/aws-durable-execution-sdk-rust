//! Integration tests for the combinator error surface (issue #28).
//!
//! Verifies that `race` surfaces a losing branch's failure as
//! `CombinatorErrorKind::FirstSettledFailed`, with the same variant and
//! message live and on replay, and that `race` and `select_ok` agree on
//! `CombinatorErrorKind::EmptyInput` for empty input.

#![cfg(feature = "test-util")]
#![expect(clippy::expect_used, clippy::indexing_slicing)] // reason: test assertions

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;
use durable::test_util::LocalRunner;
use durable::{CombinatorErrorKind, DurableFuture, OperationErrorKind, RetryDecision};

/// Renders the combinator error kind a handler observed, so the test can
/// compare the live observation against the replayed one.
fn describe_combinator_error(err: &durable::OperationError) -> String {
    match err.kind() {
        OperationErrorKind::Combinator(ce) => match ce.kind() {
            CombinatorErrorKind::FirstSettledFailed { .. } => {
                // The loser travels as the error's source; flatten it the
                // documented way.
                format!("FirstSettledFailed:{ce:#}")
            }
            CombinatorErrorKind::EmptyInput { .. } => "EmptyInput".to_owned(),
            other => format!("other:{other:?}"),
        },
        other => format!("non-combinator:{other:?}"),
    }
}

/// The race loser's error identity survives replay: the live invocation
/// and the replayed invocation both observe `FirstSettledFailed` carrying
/// the losing step's message.
#[tokio::test]
async fn race_loser_error_identity_survives_replay() {
    let observed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    // The loser fails immediately; the slow branch never wins.
                    let loser = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom-loser".into()) })
                        .name("loser")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();
                    let slow = ctx
                        .step(|_| async {
                            tokio::time::sleep(Duration::from_secs(30)).await;
                            Ok("slow".to_owned())
                        })
                        .name("slow")
                        .future();

                    let race_err = match ctx.race([loser, slow]).name("race").await {
                        Ok(winner) => {
                            return Err(format!("race unexpectedly won: {winner}").into());
                        }
                        Err(err) => err,
                    };
                    if let Ok(mut seen) = observed.lock() {
                        seen.push(describe_combinator_error(&race_err));
                    }

                    // Suspend so the next invocation REPLAYS the failed race
                    // from its checkpoint record.
                    ctx.wait(Duration::from_secs(1)).name("suspend").await?;

                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "handler should complete: {:?}",
        result.error_message()
    );
    assert!(
        result.invocation_count() >= 2,
        "the wait must force at least one replay, got {} invocation(s)",
        result.invocation_count()
    );

    let seen = observed.lock().expect("observation lock").clone();
    assert!(
        seen.len() >= 2,
        "the race error must be observed live and on replay: {seen:?}"
    );
    for (i, entry) in seen.iter().enumerate() {
        assert!(
            entry.starts_with("FirstSettledFailed:"),
            "invocation {i} must observe FirstSettledFailed: {entry}"
        );
        assert!(
            entry.contains("boom-loser"),
            "invocation {i} must preserve the loser's message: {entry}"
        );
    }
    // Live and replayed observations are identical (same variant, same message).
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "live and replayed errors must be indistinguishable: {seen:?}"
    );
}

/// `race([])` and `select_ok([])` both fail with `EmptyInput`.
#[tokio::test]
async fn race_and_select_ok_agree_on_empty_input() {
    let observed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    let Err(race_err) = ctx
                        .race(Vec::<DurableFuture<String>>::new())
                        .name("empty-race")
                        .await
                    else {
                        return Err("race([]) unexpectedly succeeded".into());
                    };
                    let Err(select_err) = ctx
                        .select_ok(Vec::<DurableFuture<String>>::new())
                        .name("empty-select")
                        .await
                    else {
                        return Err("select_ok([]) unexpectedly succeeded".into());
                    };
                    if let Ok(mut seen) = observed.lock() {
                        seen.push(describe_combinator_error(&race_err));
                        seen.push(describe_combinator_error(&select_err));
                    }
                    Ok::<_, durable::BoxError>("done".to_owned())
                }
            },
            (),
        )
        .await;

    assert!(
        result.is_success(),
        "handler should complete: {:?}",
        result.error_message()
    );
    let seen = observed.lock().expect("observation lock").clone();
    assert_eq!(
        seen,
        vec!["EmptyInput".to_owned(), "EmptyInput".to_owned()],
        "race([]) and select_ok([]) must both yield EmptyInput"
    );
}
