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

use aws_durable_execution_sdk as durable;
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

/// Renders a `try_join_all` failure as its variant plus the failed index,
/// so live and replayed observations compare structurally.
fn describe_join_failure(err: &durable::OperationError) -> String {
    match err.kind() {
        OperationErrorKind::Combinator(ce) => match ce.kind() {
            CombinatorErrorKind::JoinFailed(details) => {
                format!("JoinFailed@{}", details.failed_index())
            }
            other => format!("other:{other:?}"),
        },
        other => format!("non-combinator:{other:?}"),
    }
}

/// A `try_join_all` failure classifies as `JoinFailed` carrying the failed
/// input's index on the live path, and replay must reconstruct the SAME
/// variant and index from the recorded discriminator: a handler that
/// branches on either takes the same path live and replayed.
#[tokio::test]
async fn try_join_all_failure_kind_and_index_survive_replay() {
    let observed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    let ok = ctx
                        .step(|_| async { Ok("fine".to_owned()) })
                        .name("ok")
                        .future();
                    let bad = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom-join".into()) })
                        .name("bad")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();

                    let join_err = match ctx.try_join_all([ok, bad]).name("gather").await {
                        Ok(values) => {
                            return Err(format!("unexpectedly succeeded: {values:?}").into());
                        }
                        Err(err) => err,
                    };
                    if let Ok(mut seen) = observed.lock() {
                        seen.push(describe_join_failure(&join_err));
                    }

                    // Suspend so the next invocation replays the failure.
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
    let seen = observed.lock().expect("observation lock").clone();
    assert!(
        seen.len() >= 2,
        "the failure must be observed live and on replay: {seen:?}"
    );
    assert_eq!(
        seen[0], "JoinFailed@1",
        "the live failure names the failing input's index: {seen:?}"
    );
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "live and replayed kinds (including the index) must match: {seen:?}"
    );
}

/// A `select_ok` whose inputs all fail classifies as `AllFailed` with one
/// loser per input on the live path, and replay must reconstruct the SAME
/// variant with the SAME loser count from the recorded discriminator.
#[tokio::test]
async fn select_ok_all_failed_kind_and_loser_count_survive_replay() {
    let observed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    let first = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom-a".into()) })
                        .name("first")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();
                    let second = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom-b".into()) })
                        .name("second")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();

                    let select_err = match ctx.select_ok([first, second]).name("pick").await {
                        Ok(value) => {
                            return Err(format!("unexpectedly succeeded: {value}").into());
                        }
                        Err(err) => err,
                    };
                    let label = match select_err.kind() {
                        OperationErrorKind::Combinator(ce) => match ce.kind() {
                            CombinatorErrorKind::AllFailed { .. } => {
                                format!("AllFailed:{}", ce.failures().len())
                            }
                            other => format!("other:{other:?}"),
                        },
                        other => format!("non-combinator:{other:?}"),
                    };
                    if let Ok(mut seen) = observed.lock() {
                        seen.push(label);
                    }

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
    let seen = observed.lock().expect("observation lock").clone();
    assert!(
        seen.len() >= 2,
        "the failure must be observed live and on replay: {seen:?}"
    );
    assert_eq!(
        seen[0], "AllFailed:2",
        "the live failure carries one loser per input: {seen:?}"
    );
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "live and replayed kinds (including loser count) must match: {seen:?}"
    );
}

/// A rejected `join_all` outcome's wire identity (`error_type`, message)
/// survives the checkpoint round-trip: the replayed `Settled::Rejected`
/// carries the SAME recorded wire record the live outcome carried, not a
/// synthetic untyped reconstruction (issue #41).
#[tokio::test]
async fn join_all_rejected_wire_identity_survives_replay() {
    let observed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    let good = ctx
                        .step(|_| async { Ok("fine".to_owned()) })
                        .name("good")
                        .future();
                    let bad = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom-settled".into()) })
                        .name("bad-settled")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();

                    let settled = ctx.join_all([good, bad]).name("collect").await?;
                    let rejected_wires: Vec<String> = settled
                        .iter()
                        .filter_map(|outcome| match outcome {
                            durable::Settled::Rejected(err) => Some(format!(
                                "{}:{}",
                                err.wire()
                                    .and_then(durable::WireError::error_type)
                                    .unwrap_or("<none>"),
                                err.wire()
                                    .and_then(durable::WireError::error_message)
                                    .unwrap_or("<none>"),
                            )),
                            _ => None,
                        })
                        .collect();
                    if let Ok(mut seen) = observed.lock() {
                        seen.push(rejected_wires.join("|"));
                    }

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
    let seen = observed.lock().expect("observation lock").clone();
    assert!(
        seen.len() >= 2,
        "the settled set must be observed live and on replay: {seen:?}"
    );
    assert!(
        seen[0].contains("boom-settled") && !seen[0].starts_with("<none>"),
        "the live rejected outcome carries a typed wire record: {seen:?}"
    );
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "live and replayed rejected wire identities must match: {seen:?}"
    );
}

/// Renders one loser's complete wire failure record, whether the loser is
/// the live error itself (an `OperationError` carrying its recorded wire
/// record) or a replay-reconstructed `ReplayedFailure`.
fn describe_loser(source: &(dyn std::error::Error + 'static)) -> String {
    let wire = source
        .downcast_ref::<durable::OperationError>()
        .and_then(|op| op.wire().cloned())
        .or_else(|| {
            source
                .downcast_ref::<durable::ReplayedFailure>()
                .map(|replayed| replayed.wire().clone())
        });
    wire.map_or_else(
        || "no-wire".to_owned(),
        |w| {
            format!(
                "type={:?} msg={:?} data={:?} trace={:?}",
                w.error_type(),
                w.error_message(),
                w.error_data(),
                w.stack_trace()
            )
        },
    )
}

/// Each `select_ok` loser's error identity survives replay individually:
/// the live invocation and the replayed invocation observe the SAME
/// per-loser `error_type`, message, `error_data`, and `stack_trace` for
/// every input, not clones of the aggregate record.
#[tokio::test]
async fn select_ok_all_failed_loser_identities_survive_replay() {
    let observed: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let observed_clone = Arc::clone(&observed);

    let result = LocalRunner::new()
        .run(
            move |_event: (), ctx: durable::DurableContext| {
                let observed = Arc::clone(&observed_clone);
                async move {
                    let first = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom-first".into()) })
                        .name("first")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();
                    let second = ctx
                        .step(|_| async { Err::<String, durable::BoxError>("boom-second".into()) })
                        .name("second")
                        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
                        .future();

                    let err = match ctx.select_ok([first, second]).name("all-fail").await {
                        Ok(winner) => {
                            return Err(format!("select_ok unexpectedly won: {winner}").into());
                        }
                        Err(err) => err,
                    };
                    let OperationErrorKind::Combinator(ce) = err.kind() else {
                        return Err(format!("expected a combinator error: {err:?}").into());
                    };
                    if !matches!(ce.kind(), CombinatorErrorKind::AllFailed { .. }) {
                        return Err(format!("expected AllFailed: {:?}", ce.kind()).into());
                    }
                    let losers: Vec<String> = ce
                        .failures()
                        .iter()
                        .map(|loser| describe_loser(&**loser))
                        .collect();
                    if let Ok(mut seen) = observed.lock() {
                        seen.push(losers);
                    }

                    // Suspend so the next invocation REPLAYS the failed
                    // select_ok from its checkpoint record.
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
    let seen = observed.lock().expect("observation lock").clone();
    assert!(
        seen.len() >= 2,
        "the losers must be observed live and on replay: {} observation(s)",
        seen.len()
    );
    for (i, losers) in seen.iter().enumerate() {
        assert_eq!(losers.len(), 2, "invocation {i} must observe both losers");
        assert!(
            losers[0].contains("boom-first"),
            "invocation {i} loser 0 keeps its own message: {}",
            losers[0]
        );
        assert!(
            losers[1].contains("boom-second"),
            "invocation {i} loser 1 keeps its own message: {}",
            losers[1]
        );
        assert_ne!(
            losers[0], losers[1],
            "invocation {i} losers must be individual records, not clones"
        );
    }
    // Live and replayed observations are identical, field for field,
    // loser for loser.
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "live and replayed losers must be indistinguishable: {seen:#?}"
    );
}
