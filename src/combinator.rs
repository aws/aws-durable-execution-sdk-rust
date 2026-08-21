//! Combinator operation execution engine.
//!
//! Implements `try_join_all`, `join_all`, `select_ok`, and `race`:
//! each is a durable operation that uses the child-context checkpoint
//! machinery to freeze the combined result. On replay the recorded
//! winner/collection is returned without re-running the constituent
//! operations.
//!
//! Semantics:
//! - `try_join_all` — await all; fail fast on the first error.
//! - `join_all` — await all; collect each outcome as `Settled` (never short-circuits).
//! - `select_ok` — return the first success; fail only if all branches fail.
//! - `race` — return the first settled outcome (success or failure). A
//!   failure winner surfaces as `CombinatorErrorKind::FirstSettledFailed`.
//!
//! Empty input: `try_join_all` and `join_all` resolve to an empty
//! collection (matching `futures-rs`); `select_ok` and `race` fail with
//! `CombinatorErrorKind::EmptyInput` since no winner can exist.
//!
//! Losers are dropped (cancelled) when a combinator resolves; each
//! combinator runs inside a child context so the combined result is
//! checkpointed atomically.
//!
//! Suspension is isolated per input: each constituent runs in its own
//! suspension scope ([`spawn_constituent`]), so a parked input — a pending
//! wait, an unresolved callback — never suspends the caller's scope. The
//! combined outcome is recorded the moment the completion condition is met
//! (a winner for `race`/`select_ok`, a first error for `try_join_all`),
//! whatever the losing inputs are doing. The combinator itself suspends
//! only when no input can make progress: everything still pending has
//! durably parked and the completion condition is unmet (issue #49).

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::context::DurableContext;
use crate::driver::TaskOwnership;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, CombinatorError, CombinatorErrorKind, OperationError,
    OperationErrorKind,
};
use crate::future::{DurableFuture, Settled};

/// Wire sub-type for combinator operations (shared with child context
/// since combinators ARE child-context ops with a combinator-flavored closure).
pub(crate) const COMBINATOR_SUB_TYPE: &str = "RunInChildContext";

/// Wire `error_type` for a combinator failure whose specific kind carries
/// no dedicated wire discriminator (`JoinFailed`, `AllFailed`, internal
/// errors). Replay reconstructs these as `CombinatorErrorKind::Internal`.
const COMBINATOR_ERROR_TYPE: &str = "CombinatorError";

/// Wire `error_type` recording that a `race` settled first on a failure.
///
/// [`replay_combinator_failure`] maps this back to
/// [`CombinatorErrorKind::FirstSettledFailed`], so the live and replay
/// paths surface the same variant.
const FIRST_SETTLED_FAILED_ERROR_TYPE: &str = "CombinatorError.FirstSettledFailed";

/// Wire `error_type` recording an empty-input failure (`race` and
/// `select_ok` called with no futures).
///
/// [`replay_combinator_failure`] maps this back to
/// [`CombinatorErrorKind::EmptyInput`], so the live and replay paths
/// surface the same variant.
const EMPTY_INPUT_ERROR_TYPE: &str = "CombinatorError.EmptyInput";

// ────────────────────────────────────────────────────────────────────────────
// TryJoinAll (fail-fast concurrent join)
// ────────────────────────────────────────────────────────────────────────────

/// Internal execution state for `try_join_all`.
pub(crate) struct TryJoinAllExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) futures: Vec<DurableFuture<O>>,
}

impl<O: Serialize + DeserializeOwned + Send + 'static> TryJoinAllExecution<O> {
    /// Executes the `try_join_all` combinator.
    ///
    /// Live path: awaits all futures concurrently, fails fast on first error,
    /// checkpoints the combined `Vec<O>` result.
    /// Replay path: returns the frozen result from the checkpoint log.
    #[expect(clippy::too_many_lines)] // reason: validation adds lines but the flow reads better flat
    pub(crate) async fn execute(self) -> Result<Vec<O>, OperationError> {
        // Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path: check checkpoint log. The validated view covers the
        // non-terminal branches without cloning.
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Decode FIRST, then emit `operation_replayed`: a corrupt
                    // payload surfaces as an error without the event.
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let value = replay_vec_success::<O>(payload.as_ref())?;
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(value);
                }
                CheckpointStatus::Failed => {
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    let wire = self
                        .ctx
                        .checkpoint_wire_error(&positional_id)
                        .unwrap_or_default();
                    return Err(replay_combinator_failure(wire, &wire_id));
                }
                _ => {} // Started/Pending: fall through to re-execute
            }
        } else {
            // First invocation: checkpoint START.
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Live path: run all futures concurrently, fail-fast on first error.
        // Each constituent runs under its own suspension scope (see
        // `drive_constituent`), so a parked input surfaces as `Suspended`
        // here instead of parking the caller's scope.
        let count = self.futures.len();
        let mut results: Vec<Option<O>> = (0..count).map(|_| None).collect();
        let mut join_set = tokio::task::JoinSet::new();
        // Maps a task's id back to its input index so a `JoinError` (panic
        // or cancellation loses the task's payload, including the index) is
        // still attributed to the correct position.
        let mut task_index: std::collections::HashMap<tokio::task::Id, usize> =
            std::collections::HashMap::with_capacity(count);

        for (idx, future) in self.futures.into_iter().enumerate() {
            let abort = spawn_constituent(&self.ctx, &mut join_set, idx, future);
            task_index.insert(abort.id(), idx);
        }

        let mut first_error: Option<(usize, OperationError)> = None;
        let mut parked = 0_usize;
        while let Some(task_result) = join_set.join_next_with_id().await {
            match task_result {
                Ok((task_id, (idx, ConstituentOutcome::Settled(Ok(value))))) => {
                    task_index.remove(&task_id);
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = Some(value);
                    }
                }
                Ok((task_id, (idx, ConstituentOutcome::Settled(Err(op_err))))) => {
                    task_index.remove(&task_id);
                    first_error = Some((idx, op_err));
                    // Abort remaining tasks (fail-fast + loser-drop).
                    join_set.abort_all();
                    break;
                }
                Ok((task_id, (_idx, ConstituentOutcome::Parked))) => {
                    // The input parked its own scope; the join cannot
                    // complete this invocation, but runnable siblings keep
                    // going (they checkpoint their own results) and a
                    // settling error can still fail fast.
                    task_index.remove(&task_id);
                    parked += 1;
                }
                Err(join_err) => {
                    // JoinError means the task panicked or was cancelled.
                    // Recover the input index from the task id.
                    let Some(idx) = task_index.remove(&join_err.id()) else {
                        return Err(combinator_internal_error(
                            "task terminated with an unrecognized task id",
                        ));
                    };
                    first_error = Some((
                        idx,
                        combinator_internal_error(&format!("task join failed: {join_err}")),
                    ));
                    join_set.abort_all();
                    break;
                }
            }
        }

        if let Some((failed_index, op_err)) = first_error {
            // Checkpoint FAIL. The wire record is derived from the loser
            // itself, so `error_data` and identity pass through.
            let wire = wire_error_from_loser(&op_err);
            checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
            // The loser is preserved as an error, reachable via source().
            return Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::JoinFailed(crate::error::JoinFailed::new(failed_index)),
                    vec![Box::new(op_err)],
                ),
            ))
            .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
            .with_wire(wire));
        }

        // No error, but at least one input is durably parked: nothing left
        // here can make progress, so the combinator itself suspends. The
        // settled inputs recorded their own checkpoints, so a later
        // invocation replays them instantly and resumes only the parked
        // ones. No combined outcome is recorded — the completion condition
        // was not met.
        if parked > 0 {
            return Ok(self.ctx.suspend_now().await);
        }

        // All succeeded — collect results in order.
        let mut collected: Vec<O> = Vec::with_capacity(count);
        for (idx, opt) in results.into_iter().enumerate() {
            match opt {
                Some(v) => collected.push(v),
                None => {
                    return Err(combinator_internal_error(&format!(
                        "internal: slot {idx} not filled despite no error"
                    )));
                }
            }
        }

        // Serialize and checkpoint SUCCESS.
        let serialized = serde_json::to_string(&collected)
            .map_err(|e| combinator_internal_error(&format!("serialization failed: {e}")))?;
        checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized).await?;

        Ok(collected)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// JoinAll (settled collection — never short-circuits)
// ────────────────────────────────────────────────────────────────────────────

/// Internal execution state for `join_all`.
pub(crate) struct JoinAllExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) futures: Vec<DurableFuture<O>>,
}

/// Serializable representation of a settled result for checkpointing.
///
/// Error-aware serdes: rejected results are serialized as
/// `{ status: "rejected", message }` so that error information
/// survives the checkpoint round-trip. Fulfilled results serialize as
/// `{ status: "fulfilled", value }`.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "status")]
enum SerializedSettled<O> {
    #[serde(rename = "fulfilled")]
    Fulfilled { value: O },
    #[serde(rename = "rejected")]
    Rejected { message: String },
}

impl<O: Serialize + DeserializeOwned + Send + 'static> JoinAllExecution<O> {
    /// Executes the `join_all` combinator.
    ///
    /// Live path: awaits ALL futures (never short-circuits), collects each
    /// as `Settled::Fulfilled(O)` or `Settled::Rejected(OperationError)`.
    /// Checkpoints with error-aware serdes so Err values survive round-trip.
    /// Replay path: returns frozen `Vec<Settled<O>>`.
    #[expect(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
    pub(crate) async fn execute(self) -> Result<Vec<Settled<O>>, OperationError> {
        // Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path. The validated view covers the non-terminal branches
        // without cloning.
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Decode FIRST, then emit `operation_replayed`: a corrupt
                    // payload surfaces as an error without the event.
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let value = replay_settled_success::<O>(payload.as_ref())?;
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(value);
                }
                CheckpointStatus::Failed => {
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    let wire = self
                        .ctx
                        .checkpoint_wire_error(&positional_id)
                        .unwrap_or_default();
                    return Err(replay_combinator_failure(wire, &wire_id));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Live path: run all futures, never short-circuit. Each constituent
        // runs under its own suspension scope (see `drive_constituent`), so
        // a parked input surfaces as `Parked` here instead of parking the
        // caller's scope.
        let count = self.futures.len();
        let mut settled: Vec<Option<Settled<O>>> = (0..count).map(|_| None).collect();
        let mut join_set = tokio::task::JoinSet::new();
        // Maps a task's id back to its input index so a `JoinError` is
        // attributed to the correct slot instead of the first empty one.
        let mut task_index: std::collections::HashMap<tokio::task::Id, usize> =
            std::collections::HashMap::with_capacity(count);

        for (idx, future) in self.futures.into_iter().enumerate() {
            let abort = spawn_constituent(&self.ctx, &mut join_set, idx, future);
            task_index.insert(abort.id(), idx);
        }

        let mut parked = 0_usize;
        while let Some(task_result) = join_set.join_next_with_id().await {
            match task_result {
                Ok((task_id, (idx, ConstituentOutcome::Settled(Ok(value))))) => {
                    task_index.remove(&task_id);
                    if let Some(slot) = settled.get_mut(idx) {
                        *slot = Some(Settled::Fulfilled(value));
                    }
                }
                Ok((task_id, (idx, ConstituentOutcome::Settled(Err(op_err))))) => {
                    task_index.remove(&task_id);
                    if let Some(slot) = settled.get_mut(idx) {
                        *slot = Some(Settled::Rejected(op_err));
                    }
                }
                Ok((task_id, (_idx, ConstituentOutcome::Parked))) => {
                    // The input parked its own scope; its slot stays empty.
                    // Siblings keep running — a parked input never blocks
                    // them — and the decision is made after the drain.
                    task_index.remove(&task_id);
                    parked += 1;
                }
                Err(join_err) => {
                    // Task panicked or was cancelled — record as rejected at
                    // ITS OWN index, recovered from the task id.
                    let msg = format!("task join failed: {join_err}");
                    let Some(idx) = task_index.remove(&join_err.id()) else {
                        return Err(combinator_internal_error(
                            "task terminated with an unrecognized task id",
                        ));
                    };
                    if let Some(slot) = settled.get_mut(idx) {
                        *slot = Some(Settled::Rejected(combinator_internal_error(&msg)));
                    }
                }
            }
        }

        // At least one input is durably parked: the collection cannot
        // complete this invocation. Everything runnable already settled
        // (and checkpointed its own outcome), so suspend and let a later
        // invocation resume only the parked inputs. No combined outcome is
        // recorded — `join_all` completes only when every input settles.
        if parked > 0 {
            return Ok(self.ctx.suspend_now().await);
        }

        // Collect in order.
        let collected: Vec<Settled<O>> = settled
            .into_iter()
            .map(|opt| {
                opt.unwrap_or_else(|| {
                    Settled::Rejected(combinator_internal_error("future did not complete"))
                })
            })
            .collect();

        // Serialize with error-aware serdes. A rejected slot's error is
        // flattened into the payload here — the checkpoint stores text,
        // and replay rebuilds a synthetic source from it.
        let serialized = {
            let serializable: Vec<SerializedSettled<&O>> = collected
                .iter()
                .map(|s| match s {
                    Settled::Fulfilled(v) => SerializedSettled::Fulfilled { value: v },
                    Settled::Rejected(err) => SerializedSettled::Rejected {
                        message: crate::error::chain_string(err),
                    },
                })
                .collect();

            serde_json::to_string(&serializable)
                .map_err(|e| combinator_internal_error(&format!("serialization failed: {e}")))?
        };
        checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized).await?;

        Ok(collected)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SelectOk (first success wins)
// ────────────────────────────────────────────────────────────────────────────

/// Internal execution state for `select_ok`.
pub(crate) struct SelectOkExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) futures: Vec<DurableFuture<O>>,
}

impl<O: Serialize + DeserializeOwned + Send + 'static> SelectOkExecution<O> {
    /// Executes the `select_ok` combinator.
    ///
    /// Live path: races all futures; returns the first success.
    /// If all fail, returns `CombinatorError::AllFailed` with all error messages.
    /// Losers are dropped (cancelled) on first success.
    /// Replay path: returns the frozen winner.
    #[expect(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path. The validated view covers the non-terminal branches
        // without cloning.
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Decode FIRST, then emit `operation_replayed`: a corrupt
                    // payload surfaces as an error without the event.
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let value = replay_single_success::<O>(payload.as_ref())?;
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(value);
                }
                CheckpointStatus::Failed => {
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    let wire = self
                        .ctx
                        .checkpoint_wire_error(&positional_id)
                        .unwrap_or_default();
                    return Err(replay_combinator_failure(wire, &wire_id));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Empty input — there is no future that could succeed. Fail
        // explicitly with `EmptyInput` (matching `race`) rather than an
        // `AllFailed` carrying zero errors.
        if self.futures.is_empty() {
            let wire = crate::error::wire_error_manual(
                EMPTY_INPUT_ERROR_TYPE,
                "select_ok called with no futures",
            );
            checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
            // Attach the checkpointed context so the live error matches
            // what a replay of this record reconstructs.
            return Err(combinator_empty_input_error()
                .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire));
        }

        // Live path: race all futures, keep going until one succeeds or all
        // fail. Each constituent runs under its own suspension scope (see
        // `drive_constituent`), so a parked input surfaces as `Parked` here
        // instead of parking the caller's scope after a sibling succeeded.
        let count = self.futures.len();
        let mut join_set = tokio::task::JoinSet::new();
        // Maps a task's id back to its input index so a `JoinError` is
        // attributed to the correct position in the aggregate error list.
        let mut task_index: std::collections::HashMap<tokio::task::Id, usize> =
            std::collections::HashMap::with_capacity(count);

        for (idx, future) in self.futures.into_iter().enumerate() {
            let abort = spawn_constituent(&self.ctx, &mut join_set, idx, future);
            task_index.insert(abort.id(), idx);
        }

        let mut errors: Vec<(usize, crate::error::Source)> = Vec::new();
        let mut parked = 0_usize;

        while let Some(task_result) = join_set.join_next_with_id().await {
            match task_result {
                Ok((_task_id, (_idx, ConstituentOutcome::Settled(Ok(value))))) => {
                    // First success — cancel losers (drop via abort_all).
                    // Parked losers only ever parked their own constituent
                    // scopes, so recording the winner is not blocked by them.
                    join_set.abort_all();

                    // Checkpoint SUCCESS with the winner.
                    let serialized = serde_json::to_string(&value).map_err(|e| {
                        combinator_internal_error(&format!("serialization failed: {e}"))
                    })?;
                    checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized)
                        .await?;
                    return Ok(value);
                }
                Ok((task_id, (idx, ConstituentOutcome::Settled(Err(op_err))))) => {
                    task_index.remove(&task_id);
                    errors.push((idx, Box::new(op_err)));
                }
                Ok((task_id, (_idx, ConstituentOutcome::Parked))) => {
                    // The input parked its own scope; it may still succeed on
                    // a later invocation, so it neither wins nor counts as a
                    // failure now.
                    task_index.remove(&task_id);
                    parked += 1;
                }
                Err(join_err) => {
                    let Some(idx) = task_index.remove(&join_err.id()) else {
                        return Err(combinator_internal_error(
                            "task terminated with an unrecognized task id",
                        ));
                    };
                    errors.push((
                        idx,
                        crate::error::ContextualError::source_from(
                            "task join failed",
                            Box::new(join_err) as crate::error::Source,
                        ),
                    ));
                }
            }
        }

        // No success yet, and at least one input is durably parked: it may
        // still deliver the success, so recording `AllFailed` now would be
        // premature. Suspend; the settled failures replay instantly from
        // their own checkpoints and only the parked inputs resume.
        if parked > 0 {
            return Ok(self.ctx.suspend_now().await);
        }

        // All failed — build the aggregate error keeping every loser (in
        // input order) as an error rather than a flattened string.
        errors.sort_by_key(|(idx, _)| *idx);
        let losers: Vec<crate::error::Source> = errors.into_iter().map(|(_, err)| err).collect();
        let wire = crate::error::wire_error_manual(
            COMBINATOR_ERROR_TYPE,
            format!("all {} futures failed", losers.len()),
        );

        checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;

        Err(
            OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
                CombinatorErrorKind::AllFailed,
                losers,
            )))
            .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
            .with_wire(wire),
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Race (first settled wins, success or failure)
// ────────────────────────────────────────────────────────────────────────────

/// Internal execution state for `race`.
pub(crate) struct RaceExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) futures: Vec<DurableFuture<O>>,
}

impl<O: Serialize + DeserializeOwned + Send + 'static> RaceExecution<O> {
    /// Executes the `race` combinator.
    ///
    /// Live path: races all futures; returns the first settled (success OR
    /// failure). Losers are dropped (cancelled).
    /// Replay path: returns the frozen result.
    #[expect(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path. The validated view covers the non-terminal branches
        // without cloning.
        if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded => {
                    // Decode FIRST, then emit `operation_replayed`: a corrupt
                    // payload surfaces as an error without the event.
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let value = replay_single_success::<O>(payload.as_ref())?;
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    return Ok(value);
                }
                CheckpointStatus::Failed => {
                    self.ctx.emit_operation_replayed(
                        &wire_id,
                        self.name.as_deref(),
                        "Context",
                        Some(COMBINATOR_SUB_TYPE),
                        view.attempt,
                    );
                    let wire = self
                        .ctx
                        .checkpoint_wire_error(&positional_id)
                        .unwrap_or_default();
                    return Err(replay_combinator_failure(wire, &wire_id));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Empty input — there is no future that could settle. Fail
        // explicitly with `EmptyInput` (matching `select_ok`).
        if self.futures.is_empty() {
            let wire = crate::error::wire_error_manual(
                EMPTY_INPUT_ERROR_TYPE,
                "race called with no futures",
            );
            checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
            // Attach the checkpointed context so the live error matches
            // what a replay of this record reconstructs.
            return Err(combinator_empty_input_error()
                .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire));
        }

        // Live path: race all futures, first settled wins. Each constituent
        // runs under its own suspension scope (see `drive_constituent`), so
        // a losing input's park surfaces as `Parked` here instead of
        // suspending the caller after the winner already settled — the
        // defect that let replay pick a different winner (issue #49).
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, future) in self.futures.into_iter().enumerate() {
            spawn_constituent(&self.ctx, &mut join_set, idx, future);
        }

        // Wait for the first SETTLED task; parked inputs do not settle and
        // do not block a later sibling from winning.
        while let Some(task_result) = join_set.join_next().await {
            let settled_result = match task_result {
                Ok((_idx, ConstituentOutcome::Parked)) => {
                    // The input parked its own scope; keep waiting for a
                    // sibling to settle.
                    continue;
                }
                Ok((_idx, ConstituentOutcome::Settled(result))) => Ok(result),
                Err(join_err) => Err(join_err),
            };

            // Cancel all remaining losers.
            join_set.abort_all();

            match settled_result {
                Ok(Ok(value)) => {
                    // Winner is a success.
                    let serialized = serde_json::to_string(&value).map_err(|e| {
                        combinator_internal_error(&format!("serialization failed: {e}"))
                    })?;
                    checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized)
                        .await?;
                    return Ok(value);
                }
                Ok(Err(op_err)) => {
                    // Winner is a failure — race propagates it. The losing
                    // error is preserved as the combinator error's source;
                    // the wire `error_type` carries the discriminator so
                    // replay reproduces the same variant, while the message,
                    // `error_data`, and `stack_trace` derive from the loser
                    // itself (pass-through, with a fresh capture only when
                    // the chain recorded none).
                    let wire = crate::error::wire_error_with_type(
                        &op_err,
                        FIRST_SETTLED_FAILED_ERROR_TYPE,
                    );
                    checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
                    return Err(OperationError::from_kind(OperationErrorKind::Combinator(
                        CombinatorError::new(
                            CombinatorErrorKind::FirstSettledFailed,
                            vec![Box::new(op_err)],
                        ),
                    ))
                    .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                    .with_wire(wire));
                }
                Err(join_err) => {
                    let wire = crate::error::wire_error_manual(
                        COMBINATOR_ERROR_TYPE,
                        format!("task join failed: {join_err}"),
                    );
                    checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
                    // Attach the checkpointed context so the live error
                    // matches what a replay of this record reconstructs.
                    return Err(OperationError::from_kind(OperationErrorKind::Combinator(
                        CombinatorError::new(
                            CombinatorErrorKind::Internal,
                            vec![crate::error::ContextualError::source_from(
                                "task join failed",
                                Box::new(join_err) as crate::error::Source,
                            )],
                        ),
                    ))
                    .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                    .with_wire(wire));
                }
            }
        }

        // Every input parked without settling: no winner can be decided
        // this invocation, so the race itself suspends. Nothing is
        // recorded — the winner is frozen only once an input settles.
        Ok(self.ctx.suspend_now().await)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ────────────────────────────────────────────────────────────────────────────

/// Blesses the current task for task-ownership checks.
///
/// Call this at the top of a spawned async block (in `JoinSet` tasks) to
/// register the task with the ownership guard. This is a shared helper to
/// ensure the obligation cannot be forgotten when adding new spawn sites.
///
/// Does nothing if called outside a tokio spawned task (`try_id()` returns `None`).
pub(crate) fn bless_current_task(task_ownership: &TaskOwnership) {
    if let Some(task_id) = tokio::task::try_id() {
        task_ownership.bless_task(task_id);
    }
}

/// How one combinator input ended within this invocation.
enum ConstituentOutcome<O> {
    /// The input resolved with a success or a failure.
    Settled(Result<O, OperationError>),
    /// The input durably parked; only a later invocation can resume it.
    Parked,
}

/// Spawns one combinator input onto `join_set`, isolated in its own
/// suspension scope.
///
/// The input's park is redirected onto a fresh child scope (via
/// [`DurableFuture::set_park_scope`]) which
/// [`drive_scope`](crate::driver::drive_scope) observes on the spawned
/// task, so a parking input completes its task with
/// [`ConstituentOutcome::Parked`] instead of parking the caller's scope.
/// This is what keeps a losing input's suspension from ending the
/// invocation after a winner already settled (issue #49), and what lets
/// the combinator decide for itself when no input can make progress.
fn spawn_constituent<O: Send + 'static>(
    ctx: &DurableContext,
    join_set: &mut tokio::task::JoinSet<(usize, ConstituentOutcome<O>)>,
    idx: usize,
    future: DurableFuture<O>,
) -> tokio::task::AbortHandle {
    use crate::driver::{ScopeOutcome, drive_scope};

    let scope = std::sync::Arc::new(ctx.suspension_signal().new_child_scope());
    future.set_park_scope(std::sync::Arc::clone(&scope));
    let task_ownership = ctx.task_ownership().clone();
    join_set.spawn(async move {
        bless_current_task(&task_ownership);
        match drive_scope(future, scope).await {
            ScopeOutcome::Completed(result) => (idx, ConstituentOutcome::Settled(result)),
            ScopeOutcome::Suspended => (idx, ConstituentOutcome::Parked),
        }
    })
}

/// Refuses a combinator's terminal checkpoint once an execution-fatal error
/// has been recorded.
///
/// Called at the top of [`checkpoint_succeed`] and [`checkpoint_fail`], which
/// every combinator's live path funnels through. A replay identity mismatch
/// is recorded on the shared fatal slot eagerly — when the mismatching
/// constituent [`DurableFuture`] was finalized (see `preflight_identity!` in
/// `builders`) — so by the time a short-circuiting combinator (`select_ok`,
/// `race`, `try_join_all`) is ready to record a winner, the slot already
/// says the recorded history contradicts the handler. Writing a SUCCEED (or
/// an unrelated FAIL) checkpoint at that point would store an outcome the
/// execution can never legitimately replay; instead the combinator returns
/// the fatal as its own error and the invocation driver fails the execution
/// with the dedicated `NonDeterministicExecutionError`.
fn fatal_gate(ctx: &DurableContext) -> Result<(), OperationError> {
    if let Some(fatal) = ctx.suspension_signal().fatal_error() {
        return Err(combinator_internal_error(&fatal.error_message));
    }
    Ok(())
}

/// Checkpoints a START action for a combinator operation.
async fn checkpoint_start(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
) -> Result<(), OperationError> {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(COMBINATOR_SUB_TYPE.to_owned())
        .action(OperationAction::Start);

    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43) — combinator START: no user code ran for the
        // combined operation itself, so no terminal FAIL is needed;
        // re-invocation reconverges on the same write.
        return ctx
            .checkpoint_failure_unrecoverable(wire_id, err, None)
            .await;
    }
    Ok(())
}

/// Checkpoints a SUCCEED action for a combinator operation.
///
/// Refuses (via [`fatal_gate`]) once an execution-fatal error is recorded.
async fn checkpoint_succeed(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    payload: &str,
) -> Result<(), OperationError> {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    fatal_gate(ctx)?;

    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(COMBINATOR_SUB_TYPE.to_owned())
        .action(OperationAction::Succeed)
        .payload(payload.to_owned());

    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43) — combinator SUCCEED: the winning branch(es) ran,
        // so their side effects need a recorded outcome. A permanent
        // rejection persists a small terminal FAIL before the execution
        // fails.
        let cwire = crate::error::checkpoint_failure_wire(&err);
        let terminal = build_combinator_fail_update(ctx, wire_id, name, &cwire);
        return ctx
            .checkpoint_failure_unrecoverable(wire_id, err, Some(terminal))
            .await;
    }
    Ok(())
}

/// Checkpoints a FAIL action for a combinator operation.
///
/// Refuses (via [`fatal_gate`]) once an execution-fatal error is recorded:
/// the dedicated non-determinism error must reach the driver rather than a
/// stringified combinator failure.
async fn checkpoint_fail(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    error: &crate::error::WireError,
) -> Result<(), OperationError> {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    fatal_gate(ctx)?;

    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(COMBINATOR_SUB_TYPE.to_owned())
        .action(OperationAction::Fail)
        .error(error.to_error_object());

    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43) — combinator FAIL: branch bodies ran; the failed
        // FAIL write routes unrecoverable with a minimal terminal FAIL
        // retry (the original carried the branch error's payload).
        let cwire = crate::error::checkpoint_failure_wire(&err);
        let terminal = build_combinator_fail_update(ctx, wire_id, name, &cwire);
        return ctx
            .checkpoint_failure_unrecoverable(wire_id, err, Some(terminal))
            .await;
    }
    Ok(())
}

/// Builds a combinator `FAIL` update carrying `wire` as its error — the
/// terminal record persisted when the combinator's own outcome write was
/// permanently rejected (issue #43).
fn build_combinator_fail_update(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    wire: &crate::error::WireError,
) -> aws_sdk_lambda::types::OperationUpdate {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(COMBINATOR_SUB_TYPE.to_owned())
        .action(OperationAction::Fail)
        .error(wire.to_error_object());
    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }
    #[expect(clippy::expect_used)] // reason: all required fields are set above
    builder
        .build()
        .expect("all required OperationUpdate fields set")
}

/// Creates a combinator internal `OperationError`; the message becomes
/// the source frame, keeping the kind a pure classification.
fn combinator_internal_error(message: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
        CombinatorErrorKind::Internal,
        vec![message.to_owned().into()],
    )))
}

/// Creates the `EmptyInput` combinator `OperationError` shared by `race`
/// and `select_ok` (live and replay paths).
fn combinator_empty_input_error() -> OperationError {
    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
        CombinatorErrorKind::EmptyInput,
        Vec::new(),
    )))
}

/// Derives the wire failure record for a losing future's error,
/// preserving pass-through identity (`error_type`, `error_data`) and
/// flattening the message once.
fn wire_error_from_loser(op_err: &OperationError) -> crate::error::WireError {
    crate::error::wire_error_for(op_err, COMBINATOR_ERROR_TYPE)
}

/// Replays a successful `Vec<O>` from a checkpoint record.
fn replay_vec_success<O: DeserializeOwned>(
    result: Option<&String>,
) -> Result<Vec<O>, OperationError> {
    let payload = result
        .ok_or_else(|| combinator_internal_error("replay: succeeded record has no payload"))?;
    serde_json::from_str(payload)
        .map_err(|e| combinator_internal_error(&format!("replay deserialization failed: {e}")))
}

/// Replays a successful single `O` from a checkpoint record.
fn replay_single_success<O: DeserializeOwned>(
    result: Option<&String>,
) -> Result<O, OperationError> {
    let payload = result
        .ok_or_else(|| combinator_internal_error("replay: succeeded record has no payload"))?;
    serde_json::from_str(payload)
        .map_err(|e| combinator_internal_error(&format!("replay deserialization failed: {e}")))
}

/// Replays a successful `Vec<Settled<O>>` with error-aware deserialization.
fn replay_settled_success<O: DeserializeOwned>(
    result: Option<&String>,
) -> Result<Vec<Settled<O>>, OperationError> {
    let payload = result
        .ok_or_else(|| combinator_internal_error("replay: succeeded record has no payload"))?;
    let serialized: Vec<SerializedSettled<O>> = serde_json::from_str(payload)
        .map_err(|e| combinator_internal_error(&format!("replay deserialization failed: {e}")))?;

    // Restore into Settled<O> — Rejected items become OperationErrors
    // carrying the original message.
    let settled = serialized
        .into_iter()
        .map(|s| match s {
            SerializedSettled::Fulfilled { value } => Settled::Fulfilled(value),
            SerializedSettled::Rejected { message } => {
                let wire = crate::error::WireError::new(None::<String>, Some(message));
                Settled::Rejected(
                    OperationError::from_kind(OperationErrorKind::ChildContext(
                        ChildContextError::new(
                            ChildContextErrorKind::ChildFailed,
                            Some(crate::error::ReplayedFailure::source_from(wire.clone())),
                        ),
                    ))
                    .with_wire(wire),
                )
            }
        })
        .collect();
    Ok(settled)
}

/// Replays a failed combinator from a checkpoint record.
///
/// The wire `error_type` is the variant discriminator: failures recorded
/// as [`FIRST_SETTLED_FAILED_ERROR_TYPE`] or [`EMPTY_INPUT_ERROR_TYPE`]
/// reconstruct the same [`CombinatorErrorKind`] variant the live path
/// produced, so an error observed live and the same error observed on
/// replay are indistinguishable. Everything else (including records
/// written before these discriminators existed) reconstructs as
/// `Internal` carrying the recorded message.
fn replay_combinator_failure(wire: crate::error::WireError, wire_id: &str) -> OperationError {
    if wire.error_type() == Some(EMPTY_INPUT_ERROR_TYPE) {
        return combinator_empty_input_error()
            .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
            .with_wire(wire);
    }
    let kind = if wire.error_type() == Some(FIRST_SETTLED_FAILED_ERROR_TYPE) {
        CombinatorErrorKind::FirstSettledFailed
    } else {
        CombinatorErrorKind::Internal
    };
    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
        kind,
        vec![crate::error::ReplayedFailure::source_from(wire.clone())],
    )))
    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
    .with_wire(wire)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[expect(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::items_after_statements
)] // reason: test assertions
mod tests {
    use super::*;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Helper to create a test context with the given checkpoint log.
    fn test_ctx(log: CheckpointLog) -> DurableContext {
        DurableContext::new_root(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
        )
    }

    /// Helper to create a test context backed by a mock client (for live paths).
    fn test_ctx_with_client(log: CheckpointLog) -> DurableContext {
        use crate::client::{CheckpointOutput, ClientError, ExecutionClient, GetStateOutput};
        use std::future::Future;
        use std::pin::Pin;

        #[derive(Debug)]
        struct MockClient;

        impl ExecutionClient for MockClient {
            fn checkpoint(
                &self,
                _arn: &str,
                _token: &str,
                _updates: Vec<aws_sdk_lambda::types::OperationUpdate>,
            ) -> Pin<Box<dyn Future<Output = Result<CheckpointOutput, ClientError>> + Send + '_>>
            {
                Box::pin(async {
                    Ok(CheckpointOutput {
                        checkpoint_token: "token-2".to_owned(),
                        updated_operations: vec![],
                        next_marker: None,
                    })
                })
            }

            fn get_state(
                &self,
                _arn: &str,
                _token: &str,
            ) -> Pin<Box<dyn Future<Output = Result<GetStateOutput, ClientError>> + Send + '_>>
            {
                Box::pin(async { Ok(GetStateOutput { operations: vec![] }) })
            }
        }

        DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:123456789012:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            Arc::new(MockClient),
            "token-1".to_owned(),
        )
    }

    /// Helper to create a succeeded checkpoint record for a combinator.
    fn succeeded_record(positional_id: &str, result: &str) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Succeeded,
                result: Some(result.to_owned()),
                error_type: None,
                error_message: None,
                error_data: None,
                stack_trace: None,
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                invoke_error_data: None,
                invoke_stack_trace: None,
                replay_children: false,
                callback_id: None,
                op_type: None,
                sub_type: None,
                op_name: None,
            },
        )
    }

    /// Helper to create a failed checkpoint record for a combinator.
    fn failed_record(
        positional_id: &str,
        err_type: &str,
        err_msg: &str,
    ) -> (String, CheckpointRecord) {
        let wire_id = crate::engine::compute_wire_id_public(positional_id);
        (
            wire_id.clone(),
            CheckpointRecord {
                id: wire_id,
                status: CheckpointStatus::Failed,
                result: None,
                error_type: Some(err_type.to_owned()),
                error_message: Some(err_msg.to_owned()),
                error_data: None,
                stack_trace: None,
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                invoke_error_data: None,
                invoke_stack_trace: None,
                replay_children: false,
                callback_id: None,
                op_type: None,
                sub_type: None,
                op_name: None,
            },
        )
    }

    // === try_join_all tests ===

    #[tokio::test]
    async fn try_join_all_success_live() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let a = DurableFuture::from_async(async { Ok(1) });
        let b = DurableFuture::from_async(async { Ok(2) });
        let c = DurableFuture::from_async(async { Ok(3) });

        let op_id = ctx.mint_id();
        let exec = TryJoinAllExecution {
            ctx,
            op_id,
            name: Some("gather".to_owned()),
            futures: vec![a, b, c],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn try_join_all_fail_fast() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let a = DurableFuture::from_async(async { Ok(1) });
        let b: DurableFuture<i32> = DurableFuture::from_async(async {
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["boom".to_owned().into()],
                ),
            )))
        });
        let c = DurableFuture::from_async(async {
            // This should be cancelled due to fail-fast.
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            Ok(3)
        });

        let op_id = ctx.mint_id();
        let exec = TryJoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b, c],
        };
        let err = exec.execute().await.unwrap_err();
        assert!(matches!(err.kind(), OperationErrorKind::Combinator(_)));
    }

    #[tokio::test]
    async fn try_join_all_replay_success() {
        // Simulate replay: checkpoint log has a succeeded record for positional "1".
        let (wire_id, record) = succeeded_record("1", "[10,20,30]");
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id(); // Claims "1"
        let exec = TryJoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![], // No futures needed — replay returns stored result.
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, vec![10, 20, 30]);
    }

    #[tokio::test]
    async fn try_join_all_replay_failure() {
        let (wire_id, record) =
            failed_record("1", "CombinatorError", "join failed at index 1: boom");
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let exec = TryJoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        assert!(matches!(err.kind(), OperationErrorKind::Combinator(_)));
    }

    // === panic attribution tests ===
    //
    // A panicking future surfaces as a `JoinError` with no payload; these
    // tests pin that the error is attributed to the panicking future's OWN
    // index even when a slower, earlier-indexed future is still running.

    #[tokio::test]
    async fn try_join_all_panic_reports_panicking_index() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        // Index 0 is slow; index 1 panics first.
        let a = DurableFuture::from_async(async {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok(1)
        });
        let b: DurableFuture<i32> = DurableFuture::from_async(async { panic!("boom-idx1") });

        let op_id = ctx.mint_id();
        let exec = TryJoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => match ce.kind() {
                CombinatorErrorKind::JoinFailed(details) => {
                    assert_eq!(
                        details.failed_index(),
                        1,
                        "panic must be attributed to the panicking future's index"
                    );
                }
                other => panic!("expected JoinFailed, got: {other:?}"),
            },
            other => panic!("expected Combinator, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn join_all_panic_lands_in_its_own_slot() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        // Index 0 is slow and succeeds; index 1 panics BEFORE index 0
        // completes — the rejection must land in slot 1, not slot 0.
        let a = DurableFuture::from_async(async {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok(1)
        });
        let b: DurableFuture<i32> = DurableFuture::from_async(async { panic!("boom-idx1") });

        let op_id = ctx.mint_id();
        let exec = JoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(
            matches!(&result[0], Settled::Fulfilled(1)),
            "slow earlier future must keep its own slot"
        );
        assert!(
            matches!(&result[1], Settled::Rejected(_)),
            "panic must land in the panicking future's slot"
        );
    }

    #[tokio::test]
    async fn select_ok_panic_error_keeps_input_order() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        // Index 0 panics immediately; index 1 fails slowly. All fail, and the
        // aggregate error list must be in INPUT order (panic first).
        let a: DurableFuture<String> = DurableFuture::from_async(async { panic!("boom-idx0") });
        let b: DurableFuture<String> = DurableFuture::from_async(async {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["err-idx1".to_owned().into()],
                ),
            )))
        });

        let op_id = ctx.mint_id();
        let exec = SelectOkExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => match ce.kind() {
                CombinatorErrorKind::AllFailed => {
                    let errors: Vec<String> = ce
                        .failures()
                        .iter()
                        .map(|e| crate::error::chain_string(&**e))
                        .collect();
                    assert_eq!(errors.len(), 2);
                    assert!(
                        errors[0].contains("task join failed"),
                        "index 0 (the panic) must sort first: {errors:?}"
                    );
                    assert!(
                        errors[1].contains("err-idx1"),
                        "index 1 must sort second: {errors:?}"
                    );
                }
                other => panic!("expected AllFailed, got: {other:?}"),
            },
            other => panic!("expected Combinator, got: {other:?}"),
        }
    }

    // === join_all tests ===

    #[tokio::test]
    async fn join_all_mixed_outcomes() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let a = DurableFuture::from_async(async { Ok(1) });
        let b: DurableFuture<i32> = DurableFuture::from_async(async {
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["task-b-failed".to_owned().into()],
                ),
            )))
        });
        let c = DurableFuture::from_async(async { Ok(3) });

        let op_id = ctx.mint_id();
        let exec = JoinAllExecution {
            ctx,
            op_id,
            name: Some("collect".to_owned()),
            futures: vec![a, b, c],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result.len(), 3);
        assert!(matches!(&result[0], Settled::Fulfilled(1)));
        assert!(matches!(&result[1], Settled::Rejected(_)));
        assert!(matches!(&result[2], Settled::Fulfilled(3)));
    }

    #[tokio::test]
    async fn join_all_replay_with_error_aware_serdes() {
        // Simulate replay with mixed settled results including rejections.
        let payload = r#"[{"status":"fulfilled","value":42},{"status":"rejected","message":"oops"},{"status":"fulfilled","value":99}]"#;
        let (wire_id, record) = succeeded_record("1", payload);
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let exec = JoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result.len(), 3);
        assert!(matches!(&result[0], Settled::Fulfilled(42)));
        assert!(matches!(&result[1], Settled::Rejected(e) if format!("{e:#}").contains("oops")));
        assert!(matches!(&result[2], Settled::Fulfilled(99)));
    }

    // === select_ok tests ===

    #[tokio::test]
    async fn select_ok_first_success() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        // b fails, a succeeds — a should win.
        let a = DurableFuture::from_async(async { Ok("winner".to_owned()) });
        let b: DurableFuture<String> = DurableFuture::from_async(async {
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["fail".to_owned().into()],
                ),
            )))
        });

        let op_id = ctx.mint_id();
        let exec = SelectOkExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, "winner");
    }

    #[tokio::test]
    async fn select_ok_all_failed_aggregate() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let a: DurableFuture<String> = DurableFuture::from_async(async {
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["err-a".to_owned().into()],
                ),
            )))
        });
        let b: DurableFuture<String> = DurableFuture::from_async(async {
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["err-b".to_owned().into()],
                ),
            )))
        });

        let op_id = ctx.mint_id();
        let exec = SelectOkExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => match ce.kind() {
                CombinatorErrorKind::AllFailed => {
                    assert_eq!(ce.failures().len(), 2);
                }
                other => panic!("expected AllFailed, got: {other:?}"),
            },
            other => panic!("expected Combinator, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn select_ok_replay_returns_winner() {
        let payload = r#""the-winner""#;
        let (wire_id, record) = succeeded_record("1", payload);
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let exec = SelectOkExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![], // No futures needed on replay.
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, "the-winner");
    }

    // === race tests ===

    #[tokio::test]
    async fn race_first_settled_success() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let a = DurableFuture::from_async(async { Ok("fast".to_owned()) });
        let b = DurableFuture::from_async(async {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            Ok("slow".to_owned())
        });

        let op_id = ctx.mint_id();
        let exec = RaceExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, "fast");
    }

    #[tokio::test]
    async fn race_failure_wins() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        // The failure resolves first (no delay); the success is slow.
        let a: DurableFuture<String> = DurableFuture::from_async(async {
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["fast-fail".to_owned().into()],
                ),
            )))
        });
        let b = DurableFuture::from_async(async {
            tokio::time::sleep(std::time::Duration::from_secs(10)).await;
            Ok("slow-success".to_owned())
        });

        let op_id = ctx.mint_id();
        let exec = RaceExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => match ce.kind() {
                CombinatorErrorKind::FirstSettledFailed => {
                    let message = crate::error::chain_string(ce);
                    assert!(
                        message.contains("fast-fail"),
                        "loser's error must be preserved: {message}"
                    );
                }
                other => panic!("expected FirstSettledFailed, got: {other:?}"),
            },
            other => panic!("expected Combinator, got: {other:?}"),
        }
        // The recorded first-settled failure derives from the loser: the
        // discriminator names the variant, and a stack trace is present
        // (pass-through from the loser, or captured at the record site).
        let wire = err.wire().expect("first-settled failure carries wire");
        assert_eq!(
            wire.error_type(),
            Some("CombinatorError.FirstSettledFailed")
        );
        assert!(
            !wire.stack_trace().is_empty(),
            "first-settled failure record must carry a stack trace"
        );
    }

    #[tokio::test]
    async fn race_replay_failure_reproduces_first_settled_failed() {
        // A race that settled on a failure records the FirstSettledFailed
        // discriminator on the wire; replay must reconstruct the SAME
        // variant the live path produced.
        let (wire_id, record) = failed_record(
            "1",
            "CombinatorError.FirstSettledFailed",
            "step failed: fast-fail",
        );
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let exec = RaceExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => match ce.kind() {
                CombinatorErrorKind::FirstSettledFailed => {
                    let message = crate::error::chain_string(ce);
                    assert!(
                        message.contains("fast-fail"),
                        "replayed message must match the recorded one: {message}"
                    );
                }
                other => panic!("expected FirstSettledFailed, got: {other:?}"),
            },
            other => panic!("expected Combinator, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn race_empty_input() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());

        let op_id = ctx.mint_id();
        let exec = RaceExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => {
                assert!(
                    matches!(ce.kind(), CombinatorErrorKind::EmptyInput),
                    "expected EmptyInput, got: {:?}",
                    ce.kind()
                );
            }
            other => panic!("expected Combinator, got: {other:?}"),
        }
        // The live error carries the checkpointed context — the same
        // operation id, status, and wire record a replay reconstructs.
        assert!(err.operation_id().is_some(), "live EmptyInput has op id");
        assert_eq!(err.status(), Some("FAILED"));
        let wire = err.wire().expect("live EmptyInput carries wire record");
        assert_eq!(wire.error_type(), Some("CombinatorError.EmptyInput"));
        assert!(
            !wire.stack_trace().is_empty(),
            "manually constructed live failure records capture a stack"
        );
    }

    #[tokio::test]
    async fn select_ok_empty_input() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());

        let op_id = ctx.mint_id();
        let exec = SelectOkExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => {
                assert!(
                    matches!(ce.kind(), CombinatorErrorKind::EmptyInput),
                    "expected EmptyInput, got: {:?}",
                    ce.kind()
                );
            }
            other => panic!("expected Combinator, got: {other:?}"),
        }
        // Live/replay parity: the checkpointed context is attached.
        assert!(err.operation_id().is_some(), "live EmptyInput has op id");
        assert_eq!(err.status(), Some("FAILED"));
        let wire = err.wire().expect("live EmptyInput carries wire record");
        assert_eq!(wire.error_type(), Some("CombinatorError.EmptyInput"));
        assert!(!wire.stack_trace().is_empty());
    }

    #[tokio::test]
    async fn empty_input_replay_reproduces_empty_input() {
        // A recorded empty-input failure replays as EmptyInput — the same
        // variant the live path produced — for both race and select_ok.
        let (wire_id, record) = failed_record(
            "1",
            "CombinatorError.EmptyInput",
            "race called with no futures",
        );
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let exec = RaceExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        match err.kind() {
            OperationErrorKind::Combinator(ce) => {
                assert!(
                    matches!(ce.kind(), CombinatorErrorKind::EmptyInput),
                    "expected EmptyInput, got: {:?}",
                    ce.kind()
                );
            }
            other => panic!("expected Combinator, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn try_join_all_empty_returns_empty_vec() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());

        let op_id = ctx.mint_id();
        let exec = TryJoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let result = exec.execute().await.unwrap();
        assert!(result.is_empty(), "try_join_all([]) must be Ok(empty)");
    }

    #[tokio::test]
    async fn join_all_empty_returns_empty_vec() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());

        let op_id = ctx.mint_id();
        let exec = JoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let result = exec.execute().await.unwrap();
        assert!(result.is_empty(), "join_all([]) must be Ok(empty)");
    }

    #[tokio::test]
    async fn race_replay_returns_recorded_winner() {
        let payload = r#""replay-winner""#;
        let (wire_id, record) = succeeded_record("1", payload);
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let exec = RaceExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![], // Replay — no futures run.
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, "replay-winner");
    }

    #[tokio::test]
    async fn race_loser_drop_cancels() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = Arc::clone(&cancelled);

        // Loser future: sets flag on drop via a guard.
        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let a = DurableFuture::from_async(async { Ok("winner".to_owned()) });
        let b = DurableFuture::from_async(async move {
            let _guard = DropGuard(cancelled_clone);
            // Never completes.
            std::future::pending::<()>().await;
            Ok("never".to_owned())
        });

        let op_id = ctx.mint_id();
        let exec = RaceExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, "winner");

        // Give tokio a moment to clean up the aborted task.
        tokio::task::yield_now().await;
        assert!(
            cancelled.load(Ordering::SeqCst),
            "loser future should have been dropped (cancelled)"
        );
    }

    #[tokio::test]
    async fn select_ok_loser_drop_cancels() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let cancelled = Arc::new(AtomicBool::new(false));
        let cancelled_clone = Arc::clone(&cancelled);

        struct DropGuard(Arc<AtomicBool>);
        impl Drop for DropGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let a = DurableFuture::from_async(async { Ok("winner".to_owned()) });
        let b = DurableFuture::from_async(async move {
            let _guard = DropGuard(cancelled_clone);
            std::future::pending::<()>().await;
            Ok("never".to_owned())
        });

        let op_id = ctx.mint_id();
        let exec = SelectOkExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![a, b],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result, "winner");

        tokio::task::yield_now().await;
        assert!(
            cancelled.load(Ordering::SeqCst),
            "loser future should have been dropped (cancelled)"
        );
    }

    #[tokio::test]
    async fn combinator_ids_deterministic_under_reversed_order() {
        // Two contexts created identically should produce the same IDs
        // regardless of which combinator resolves first.
        let ctx1 = test_ctx_with_client(CheckpointLog::empty());
        let ctx2 = test_ctx_with_client(CheckpointLog::empty());

        // Both mint the same sequence of IDs.
        let id1 = ctx1.mint_id();
        let id2 = ctx2.mint_id();
        assert_eq!(id1.positional(), id2.positional());
        assert_eq!(id1.wire(), id2.wire());

        // Second ID.
        let id1b = ctx1.mint_id();
        let id2b = ctx2.mint_id();
        assert_eq!(id1b.positional(), id2b.positional());
        assert_eq!(id1b.wire(), id2b.wire());
    }

    #[tokio::test]
    async fn tokio_join_interop_with_combinator_builders() {
        // Verifies that tokio::join! works with DurableFuture (the output
        // of combinator builders converted via IntoFuture).
        let _ctx = test_ctx_with_client(CheckpointLog::empty());

        // Create two futures that resolve immediately.
        let f1 = DurableFuture::from_async(async { Ok(1_i32) });
        let f2 = DurableFuture::from_async(async { Ok(2_i32) });

        // tokio::join! accepts futures directly.
        let (r1, r2) = tokio::join!(f1, f2);
        assert_eq!(r1.unwrap(), 1);
        assert_eq!(r2.unwrap(), 2);
    }

    // === Task-ownership tests ===
    //
    // These verify that combinator-spawned JoinSet tasks are blessed and
    // do NOT trigger ownership errors when the guard is active (context
    // created inside tokio::spawn where try_id() returns Some).
    //
    // Each branch calls `ctx.enforce_task_ownership()` — the same check
    // that every durable operation (step, invoke, etc.) performs at entry.
    // Without the bless_current_task() call in the combinator spawn sites,
    // these tests FAIL with ownership errors.

    #[tokio::test]
    async fn try_join_all_ownership_blessed() {
        // Create context inside a spawned task so the ownership guard is active.
        let result = tokio::spawn(async {
            let ctx = test_ctx_with_client(CheckpointLog::empty());

            // Each branch performs an ownership check (simulating a durable op).
            let ctx_a = ctx.clone();
            let a = DurableFuture::from_async(async move {
                ctx_a.enforce_task_ownership()?;
                Ok(1)
            });
            let ctx_b = ctx.clone();
            let b = DurableFuture::from_async(async move {
                ctx_b.enforce_task_ownership()?;
                Ok(2)
            });

            let op_id = ctx.mint_id();
            let exec = TryJoinAllExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![a, b],
            };
            exec.execute().await
        })
        .await
        .unwrap();

        // Must succeed — no ownership errors.
        let values = result.unwrap();
        assert_eq!(values, vec![1, 2]);
    }

    #[tokio::test]
    async fn join_all_ownership_blessed_no_rejections() {
        // Create context inside a spawned task so the ownership guard is active.
        let result = tokio::spawn(async {
            let ctx = test_ctx_with_client(CheckpointLog::empty());

            // Each branch calls enforce_task_ownership — without blessing,
            // every branch would be Rejected with an ownership error.
            let ctx_a = ctx.clone();
            let a = DurableFuture::from_async(async move {
                ctx_a.enforce_task_ownership()?;
                Ok(10)
            });
            let ctx_b = ctx.clone();
            let b = DurableFuture::from_async(async move {
                ctx_b.enforce_task_ownership()?;
                Ok(20)
            });
            let ctx_c = ctx.clone();
            let c = DurableFuture::from_async(async move {
                ctx_c.enforce_task_ownership()?;
                Ok(30)
            });

            let op_id = ctx.mint_id();
            let exec = JoinAllExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![a, b, c],
            };
            exec.execute().await
        })
        .await
        .unwrap();

        // Must succeed with all branches fulfilled — none rejected due to ownership.
        let settled = result.unwrap();
        assert_eq!(settled.len(), 3);
        for (i, s) in settled.iter().enumerate() {
            assert!(
                matches!(s, Settled::Fulfilled(_)),
                "branch {i} must be Fulfilled, not Rejected with ownership error"
            );
        }
        assert!(matches!(&settled[0], Settled::Fulfilled(10)));
        assert!(matches!(&settled[1], Settled::Fulfilled(20)));
        assert!(matches!(&settled[2], Settled::Fulfilled(30)));
    }

    #[tokio::test]
    async fn select_ok_ownership_blessed() {
        let result = tokio::spawn(async {
            let ctx = test_ctx_with_client(CheckpointLog::empty());

            let ctx_a = ctx.clone();
            let a = DurableFuture::from_async(async move {
                ctx_a.enforce_task_ownership()?;
                Ok("winner".to_owned())
            });
            let ctx_b = ctx.clone();
            let b = DurableFuture::from_async(async move {
                ctx_b.enforce_task_ownership()?;
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                Ok("slower".to_owned())
            });

            let op_id = ctx.mint_id();
            let exec = SelectOkExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![a, b],
            };
            exec.execute().await
        })
        .await
        .unwrap();

        // Must succeed — no ownership errors.
        let value = result.unwrap();
        assert_eq!(value, "winner");
    }

    #[tokio::test]
    async fn race_ownership_blessed() {
        let result = tokio::spawn(async {
            let ctx = test_ctx_with_client(CheckpointLog::empty());

            let ctx_a = ctx.clone();
            let a = DurableFuture::from_async(async move {
                ctx_a.enforce_task_ownership()?;
                Ok("fast".to_owned())
            });
            let ctx_b = ctx.clone();
            let b = DurableFuture::from_async(async move {
                ctx_b.enforce_task_ownership()?;
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                Ok("slow".to_owned())
            });

            let op_id = ctx.mint_id();
            let exec = RaceExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![a, b],
            };
            exec.execute().await
        })
        .await
        .unwrap();

        // Must succeed — no ownership errors.
        let value = result.unwrap();
        assert_eq!(value, "fast");
    }
}
