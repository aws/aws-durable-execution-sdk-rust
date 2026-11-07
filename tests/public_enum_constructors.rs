//! The data-bearing variants of the decision and backoff enums that callers
//! build are `#[non_exhaustive]`: a caller builds them with constructors and
//! matches them with open patterns.

use std::time::Duration;

use aws_durable_execution_sdk::RetryDecision;
use aws_durable_execution_sdk::builders::Backoff;
use aws_durable_execution_sdk::builders::map_parallel::{CompletionDecision, CompletionOutcome};
use aws_durable_execution_sdk::builders::wait_for_condition::WaitDecision;

/// Every constructor builds its variant, and every open pattern matches it.
#[test]
fn constructors_and_open_patterns_work() {
    let exponential = match Backoff::exponential(2.0) {
        Backoff::Exponential { rate, .. } => Some(rate),
        _ => None,
    };
    assert_eq!(exponential, Some(2.0));

    let linear = match Backoff::linear(Duration::from_secs(2)) {
        Backoff::Linear { increment, .. } => Some(increment),
        _ => None,
    };
    assert_eq!(linear, Some(Duration::from_secs(2)));

    let retry = match RetryDecision::retry(Duration::from_secs(3)) {
        RetryDecision::Retry { delay, .. } => Some(delay),
        _ => None,
    };
    assert_eq!(retry, Some(Duration::from_secs(3)));

    let cont = match WaitDecision::continue_with(Duration::from_secs(4)) {
        WaitDecision::Continue { delay, .. } => Some(delay),
        _ => None,
    };
    assert_eq!(cont, Some(Duration::from_secs(4)));

    let exhausted = match WaitDecision::exhausted("done") {
        WaitDecision::Exhausted { reason, .. } => Some(reason),
        _ => None,
    };
    assert_eq!(exhausted.as_deref(), Some("done"));

    let complete = match CompletionDecision::complete(CompletionOutcome::Failed) {
        CompletionDecision::Complete { 0: outcome, .. } => Some(outcome),
        _ => None,
    };
    assert_eq!(complete, Some(CompletionOutcome::Failed));
}
