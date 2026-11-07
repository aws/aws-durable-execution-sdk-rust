//! Combinator operation execution engine.
//!
//! Implements `try_join_all`, `join_all`, `select_ok`, and `race`:
//! each is a durable operation that uses the child-context checkpoint
//! machinery to freeze the combined result. On replay the recorded
//! winner/collection is returned without re-running the constituent
//! operations.
//!
//! A combined result whose serialized form has more than
//! [`CHECKPOINT_SIZE_LIMIT_BYTES`] bytes is not stored. The SUCCEED record
//! carries `ReplayChildren` instead, with the payload `""` for
//! `try_join_all` and `join_all`, and a [`RecordedWinner`] for `race` and
//! `select_ok`. Replay of such a record drives every input again
//! (`try_join_all`, `join_all`) or only the recorded winner (`race`,
//! `select_ok`), and writes nothing.
//!
//! Semantics:
//! - `try_join_all`: await all; fail fast on the first error.
//! - `join_all`: await all; collect each outcome as a `Result` (never short-circuits).
//! - `select_ok`: return the first success; fail only if all branches fail.
//! - `race`: return the first settled outcome (success or failure). A
//!   failure winner surfaces as `CombinatorErrorKind::FirstSettledFailed`.
//!
//! Empty input: `try_join_all` and `join_all` resolve to an empty
//! collection (matching `futures-rs`); `select_ok` and `race` fail with
//! `CombinatorErrorKind::EmptyInput` since no winner can exist.
//!
//! When `try_join_all`, `select_ok` or `race` resolves, it cancels every
//! input that is still running, and a cancelled input stops at its next
//! `.await`. `join_all` cancels nothing. Each combinator runs inside a
//! child context so the combined result is checkpointed atomically.
//!
//! A combinator's first run returns the value decoded from the payload it
//! just recorded, with the same helper its replay path uses. So the first
//! run returns what a replay of the record returns: a `join_all` rejection
//! is `ChildContext(ChildFailed)` on both paths, and a value whose serde
//! form omits data omits it on both paths.
//!
//! Each input runs on its own task in the combinator's join set
//! ([`spawn_constituent`]). An input that waits (a pending wait, an
//! unresolved callback) stays pending there, and continues in the same
//! invocation when a checkpoint response completes it. The combined outcome
//! is recorded the moment the completion condition is met (a winner for
//! `race`/`select_ok`, a first error for `try_join_all`), whatever the
//! losing inputs are doing; the losers are then aborted. When every
//! remaining input waits and nothing else runs, the invocation coordinator
//! suspends the invocation (issue #49).

use std::future::Future;
use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::child::CHECKPOINT_SIZE_LIMIT_BYTES;
use crate::context::{DurableContext, SubmittedWrite};
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, CombinatorError, CombinatorErrorKind, OperationError,
    OperationErrorKind,
};
use crate::future::DurableFuture;
use crate::task::{NoRuntime, TrackedJoinSet};

/// Wire sub-type for combinator operations (shared with child context
/// since combinators ARE child-context ops with a combinator-flavored closure).
pub(crate) const COMBINATOR_SUB_TYPE: &str = "RunInChildContext";

/// Wire `error_type` for an internal combinator failure whose kind
/// carries no dedicated discriminator. Replay reconstructs it as
/// `CombinatorErrorKind::Internal`.
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

/// Wire `error_type` prefix recording that a `try_join_all` input failed.
///
/// The failed input's index is appended after the trailing dot
/// (`CombinatorError.JoinFailed.3`): the index is a structural fact of
/// [`crate::error::JoinFailed`] that must survive replay, and the
/// discriminator is an SDK-authored wire field replay may interpret
/// (`error_data` stays opaque here: a join failure's record may carry
/// pass-through external payloads). [`replay_combinator_failure`] maps the
/// prefix back to [`CombinatorErrorKind::JoinFailed`] carrying the
/// recorded index, so the live and replay paths surface the same variant
/// and index.
const JOIN_FAILED_ERROR_TYPE_PREFIX: &str = "CombinatorError.JoinFailed.";

/// Wire `error_type` recording that every `select_ok` input failed.
///
/// The record's `error_data` carries the SDK-authored losers payload: a
/// JSON array of one [`SerializedLoser`] per input, in input order, each
/// holding the loser's own recorded wire failure record. The `AllFailed`
/// record is written only by the live path below (never assembled from
/// pass-through external payloads), so this is SDK-authored data guarded
/// by an SDK-authored discriminator. [`replay_combinator_failure`] maps
/// the discriminator back to [`CombinatorErrorKind::AllFailed`] and
/// rebuilds one [`crate::error::ReplayedFailure`] per recorded loser, so
/// each replayed `failures()` entry carries the same `error_type`,
/// message, `error_data`, and `stack_trace` its live counterpart did.
const ALL_FAILED_ERROR_TYPE: &str = "CombinatorError.AllFailed";

/// Per-loser record persisted (as a JSON array) in an `AllFailed`
/// aggregate record's `error_data`.
///
/// Mirrors [`SerializedOutcome::Rejected`]: the loser's complete wire
/// failure record (`error_type`, message, `error_data`, `stack_trace`),
/// so each loser's identity survives the checkpoint round-trip rather
/// than only the aggregate's summary of the losers' messages. Every field is optional
/// with a default so hand-written or trimmed payloads still decode, and
/// an absent field round-trips as absent rather than as an empty value.
/// Replay only stores and exposes these fields (via
/// [`crate::error::ReplayedFailure`]); it never dispatches on them.
#[derive(serde::Serialize, serde::Deserialize)]
struct SerializedLoser {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error_data: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    stack_trace: Vec<String>,
}

impl SerializedLoser {
    /// Projects a loser's wire failure record into its persisted form.
    fn from_wire(wire: &crate::error::WireError) -> Self {
        Self {
            error_type: wire.error_type().map(str::to_owned),
            message: wire.error_message().map(str::to_owned),
            error_data: wire.error_data().map(str::to_owned),
            stack_trace: wire.stack_trace().to_vec(),
        }
    }

    /// Rebuilds the loser's wire failure record from its persisted form.
    fn into_wire(self) -> crate::error::WireError {
        crate::error::WireError::new(self.error_type, self.message)
            .with_error_data(self.error_data)
            .with_stack_trace(self.stack_trace)
    }
}

/// The payload of an oversized `race` or `select_ok` SUCCEED record.
///
/// The winner's value is larger than [`CHECKPOINT_SIZE_LIMIT_BYTES`], so
/// the record cannot store it. It stores the winner's 0-based input index
/// instead, and the record carries `ReplayChildren`. Replay then drives only
/// that input again (see [`replay_recorded_winner`]).
#[derive(serde::Serialize, serde::Deserialize)]
struct RecordedWinner {
    winner: usize,
}

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
    /// Replay path: returns the frozen result from the checkpoint log, or,
    /// for a record with `ReplayChildren`, drives the inputs again without
    /// writing.
    #[expect(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
    pub(crate) async fn execute(self) -> Result<Vec<O>, OperationError> {
        // Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path: check checkpoint log. The validated view covers the
        // non-terminal branches without cloning.
        //
        // A SUCCEEDED record with `ReplayChildren` stores no result, because
        // the result was larger than the payload limit. Every input has its
        // own record, so the combinator rebuilds the result by driving the
        // inputs again (`reconstruct`). Reconstruction writes no checkpoint
        // and emits no `operation_replayed` event, as child-context
        // reconstruction does.
        let run = if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded if view.replay_children => Run::Reconstruct,
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
                // Started/Pending/Ready: run the inputs again.
                _ => Run::Resumed,
            }
        } else {
            Run::Live
        };
        let reconstruct = run == Run::Reconstruct;

        // Live path: run all futures concurrently, fail-fast on first error.
        let count = self.futures.len();
        let mut results: Vec<Option<O>> = (0..count).map(|_| None).collect();
        let Running {
            mut join_set,
            start,
        } = start_inputs(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            self.futures,
        )
        .await?;

        // The join: wait for every input, and on the first failure stop the
        // inputs still running (fail-fast + loser-drop).
        let join = async move {
            while let Some((_, joined)) = join_set.join_next_with_id().await {
                // A panicked input resumes its panic here.
                match joined.into_value() {
                    (idx, Ok(value)) => {
                        if let Some(slot) = results.get_mut(idx) {
                            *slot = Some(value);
                        }
                    }
                    (idx, Err(op_err)) => {
                        join_set.abort_all();
                        return Err((idx, op_err));
                    }
                }
            }
            Ok(results)
        };
        let joined = run_join(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            join,
            join_failed_wire,
        )
        .await;
        let results = match joined {
            Ok(results) => results,
            Err(((failed_index, op_err), wire)) => {
                // Checkpoint FAIL with the wire the end hooks saw.
                if !reconstruct {
                    checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), start, &wire)
                        .await?;
                }
                // The loser is preserved as an error, reachable via source().
                return Err(OperationError::from_kind(OperationErrorKind::Combinator(
                    CombinatorError::new(
                        CombinatorErrorKind::JoinFailed(crate::error::JoinFailed::new(
                            failed_index,
                        )),
                        vec![Box::new(op_err)],
                    ),
                ))
                .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire));
            }
        };

        // All succeeded: collect results in order.
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

        // Serialize and checkpoint SUCCESS. Reconstruction serializes too,
        // so it returns what the live path returned, and writes nothing.
        let serialized = serialize_result(&collected)?;
        if !reconstruct {
            checkpoint_collection_succeed(
                &self.ctx,
                &wire_id,
                self.name.as_deref(),
                start,
                &serialized,
            )
            .await?;
        }
        drop(collected);

        // Return the value decoded from the payload just written, so the
        // first run returns what a replay of this record returns: a value
        // whose serde form drops data loses it on both paths, and a decode
        // failure is the same `Combinator(Internal)` error on both paths.
        replay_vec_success::<O>(Some(&serialized))
    }
}

/// Internal execution state for `join_all`.
pub(crate) struct JoinAllExecution<O> {
    pub(crate) ctx: DurableContext,
    pub(crate) op_id: OperationId,
    pub(crate) name: Option<String>,
    pub(crate) futures: Vec<DurableFuture<O>>,
}

/// Serializable representation of one `join_all` outcome for checkpointing.
///
/// Error-aware serdes: rejected results are serialized as
/// `{ status: "rejected", ... }` carrying the outcome's full wire error
/// record (`error_type`, `message`, `error_data`, `stack_trace`), so the
/// failure's identity survives the checkpoint round-trip rather than
/// only its message (issue #41). The `message` is the outcome record's
/// own message, or, for an outcome without a record, its recorded
/// message (see [`wire_message`](crate::error::wire_message)). Fulfilled results serialize
/// as `{ status: "fulfilled", value }`. The error fields beyond the
/// message need defaults because the writer skips a `None` or empty
/// field.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "status")]
enum SerializedOutcome<O> {
    #[serde(rename = "fulfilled")]
    Fulfilled { value: O },
    #[serde(rename = "rejected")]
    Rejected {
        message: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_type: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_data: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        stack_trace: Vec<String>,
    },
}

impl<O: Serialize + DeserializeOwned + Send + 'static> JoinAllExecution<O> {
    /// Executes the `join_all` combinator.
    ///
    /// Live path: awaits ALL futures (never short-circuits), collects each
    /// as `Ok(O)` or `Err(OperationError)`.
    /// Checkpoints with error-aware serdes so Err values survive round-trip.
    /// Replay path: returns frozen `Vec<Result<O, OperationError>>`, or,
    /// for a record with `ReplayChildren`, drives the inputs again without
    /// writing.
    #[expect(clippy::too_many_lines)] // reason: replay/live paths and per-status replay events read better as one flow
    pub(crate) async fn execute(self) -> Result<Vec<Result<O, OperationError>>, OperationError> {
        // Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path. The validated view covers the non-terminal branches
        // without cloning.
        //
        // A SUCCEEDED record with `ReplayChildren` stores no result, because
        // the result was larger than the payload limit. Every input has its
        // own record, so the combinator rebuilds the result by driving the
        // inputs again (`reconstruct`). Reconstruction writes no checkpoint
        // and emits no `operation_replayed` event, as child-context
        // reconstruction does.
        let run = if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                CheckpointStatus::Succeeded if view.replay_children => Run::Reconstruct,
                CheckpointStatus::Succeeded => {
                    // Decode FIRST, then emit `operation_replayed`: a corrupt
                    // payload surfaces as an error without the event.
                    let payload = self.ctx.checkpoint_result_payload(&positional_id);
                    let value = replay_outcomes_success::<O>(payload.as_ref())?;
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
                // Started/Pending/Ready: run the inputs again.
                _ => Run::Resumed,
            }
        } else {
            Run::Live
        };
        let reconstruct = run == Run::Reconstruct;

        // Live path: run all futures, never short-circuit.
        let count = self.futures.len();
        let mut outcomes: Vec<Option<Result<O, OperationError>>> =
            (0..count).map(|_| None).collect();
        let Running {
            mut join_set,
            start,
        } = start_inputs(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            self.futures,
        )
        .await?;

        // The join: wait for every input; it never short-circuits.
        let join = async move {
            while let Some((_, joined)) = join_set.join_next_with_id().await {
                // A panicked input resumes its panic here.
                let (idx, outcome) = joined.into_value();
                if let Some(slot) = outcomes.get_mut(idx) {
                    *slot = Some(outcome);
                }
            }
            Ok(outcomes)
        };
        let outcomes = match run_join(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            join,
            crate::plugin::never_fails,
        )
        .await
        {
            Ok(outcomes) => outcomes,
            Err((never, _)) => match never {},
        };

        // Collect in order.
        let collected: Vec<Result<O, OperationError>> = outcomes
            .into_iter()
            .map(|opt| {
                opt.unwrap_or_else(|| Err(combinator_internal_error("future did not complete")))
            })
            .collect();

        // Serialize with error-aware serdes. A rejected slot records its
        // full wire error (derived from the loser itself, so identity and
        // `error_data` pass through), and replay rebuilds a synthetic
        // source carrying those fields (issue #41).
        let serialized = {
            let serializable: Vec<SerializedOutcome<&O>> = collected
                .iter()
                .map(|s| match s {
                    Ok(v) => SerializedOutcome::Fulfilled { value: v },
                    Err(err) => {
                        // Pass the outcome's own recorded wire record
                        // through verbatim when it carries one (the normal
                        // case: a failed constituent operation attaches its
                        // terminal wire record); derive one only for a
                        // wire-less error (e.g. an internal join failure).
                        let wire = err.wire().cloned().unwrap_or_else(|| {
                            crate::error::wire_error_for(err, COMBINATOR_ERROR_TYPE)
                        });
                        SerializedOutcome::Rejected {
                            message: wire
                                .error_message()
                                .map_or_else(|| crate::error::chain_string(err), str::to_owned),
                            error_type: wire.error_type().map(str::to_owned),
                            error_data: wire.error_data().map(str::to_owned),
                            stack_trace: wire.stack_trace().to_vec(),
                        }
                    }
                })
                .collect();

            serialize_result(&serializable)?
        };
        if !reconstruct {
            checkpoint_collection_succeed(
                &self.ctx,
                &wire_id,
                self.name.as_deref(),
                start,
                &serialized,
            )
            .await?;
        }
        drop(collected);

        // Return the outcomes decoded from the payload just written, so the
        // first run returns what a replay of this record returns: every
        // rejected item is a `ChildContext(ChildFailed)` error whose
        // `wire()` carries the input's recorded failure, on both paths.
        replay_outcomes_success::<O>(Some(&serialized))
    }
}

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
    /// On the first success it cancels every input that is still running;
    /// a cancelled input stops at its next `.await`.
    /// Replay path: returns the frozen winner, or, for a record with
    /// `ReplayChildren`, drives only the recorded winner again.
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path. The validated view covers the non-terminal branches
        // without cloning.
        let run = if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                // An oversized winner: the record stores the winner's index
                // and carries `ReplayChildren`. Drive only that input again.
                CheckpointStatus::Succeeded if view.replay_children => {
                    return replay_recorded_winner(&self.ctx, &positional_id, self.futures).await;
                }
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
                // Started/Pending/Ready: run the inputs again.
                _ => Run::Resumed,
            }
        } else {
            Run::Live
        };

        let empty = self.futures.is_empty();
        let Running {
            mut join_set,
            start,
        } = start_inputs(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            self.futures,
        )
        .await?;

        // The join: race all inputs, and keep going until one succeeds or
        // all fail. On the first success it stops the inputs still
        // running (loser-drop), so a waiting loser does not block
        // recording the winner. An empty input list has no input that
        // could succeed: it fails with `EmptyInput` (matching `race`)
        // rather than an `AllFailed` carrying zero errors.
        let join = async move {
            if empty {
                return Err(FirstFailure::Empty("select_ok called with no futures"));
            }
            let mut errors: Vec<(usize, crate::error::Source)> = Vec::new();
            while let Some((_, joined)) = join_set.join_next_with_id().await {
                // A panicked input resumes its panic here.
                match joined.into_value() {
                    (idx, Ok(value)) => {
                        join_set.abort_all();
                        return Ok((idx, value));
                    }
                    (idx, Err(op_err)) => {
                        errors.push((idx, Box::new(op_err)));
                    }
                }
            }
            // Every input failed: keep every loser, in input order, as an
            // error rather than a flattened string.
            errors.sort_by_key(|(idx, _)| *idx);
            Err(FirstFailure::AllFailed(
                errors.into_iter().map(|(_, err)| err).collect(),
            ))
        };
        let joined = run_join(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            join,
            first_failure_wire,
        )
        .await;
        settle_first(&self.ctx, &wire_id, self.name.as_deref(), start, joined).await
    }
}

/// Writes the outcome of a live `select_ok` or `race` join: the winner's
/// SUCCEED, or the FAIL with the wire the end hooks saw, and returns the
/// value a replay of that record returns.
async fn settle_first<O: Serialize + DeserializeOwned + Send + 'static>(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    start: Option<SubmittedWrite>,
    joined: Result<(usize, O), (FirstFailure, crate::error::WireError)>,
) -> Result<O, OperationError> {
    match joined {
        Ok((idx, value)) => {
            // Checkpoint SUCCESS with the winner.
            let serialized = serialize_result(&value)?;
            checkpoint_winner_succeed(ctx, wire_id, name, start, idx, &serialized).await?;
            drop(value);
            // Return the winner decoded from the payload just written, so
            // the first run returns what replay returns.
            replay_single_success::<O>(Some(&serialized))
        }
        Err((failure, wire)) => {
            checkpoint_fail(ctx, wire_id, name, start, &wire).await?;
            // Attach the checkpointed context so the live error matches
            // what a replay of this record reconstructs.
            let error = match failure {
                FirstFailure::Empty(_) => combinator_empty_input_error(),
                FirstFailure::AllFailed(losers) => {
                    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
                        CombinatorErrorKind::AllFailed,
                        losers,
                    )))
                }
                // The losing error is preserved as the combinator error's
                // source.
                FirstFailure::Settled(op_err) => {
                    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
                        CombinatorErrorKind::FirstSettledFailed,
                        vec![Box::new(op_err)],
                    )))
                }
            };
            Err(error
                .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire))
        }
    }
}

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
    /// failure), and cancels every input that is still running; a
    /// cancelled input stops at its next `.await`.
    /// Replay path: returns the frozen result, or, for a record with
    /// `ReplayChildren`, drives only the recorded winner again.
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path. The validated view covers the non-terminal branches
        // without cloning.
        let run = if let Some(view) = self.ctx.checkpoint_view_validated(
            &positional_id,
            &wire_id,
            "Context",
            Some(COMBINATOR_SUB_TYPE),
            self.name.as_deref(),
        )? {
            match view.status {
                // An oversized winner: the record stores the winner's index
                // and carries `ReplayChildren`. Drive only that input again.
                CheckpointStatus::Succeeded if view.replay_children => {
                    return replay_recorded_winner(&self.ctx, &positional_id, self.futures).await;
                }
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
                // Started/Pending/Ready: run the inputs again.
                _ => Run::Resumed,
            }
        } else {
            Run::Live
        };

        let empty = self.futures.is_empty();
        let Running {
            mut join_set,
            start,
        } = start_inputs(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            self.futures,
        )
        .await?;

        // The join: race all inputs, first settled wins, and stop the
        // inputs still running. A losing input that waits stays pending and
        // never suspends the invocation once the winner settled (issue
        // #49): the winner is recorded, and the losers are stopped. An
        // empty input list has no input that could settle: it fails with
        // `EmptyInput` (matching `select_ok`).
        let join = async move {
            if empty {
                return Err(FirstFailure::Empty("race called with no futures"));
            }
            // Wait for the first SETTLED task. When every input waits, the
            // race stays pending here.
            #[expect(clippy::expect_used)]
            // reason: the input list is not empty here, and the set yields
            // one outcome per input it started (a refused start never
            // yields), so the join never reports an empty set.
            let (_, joined) = join_set
                .join_next_with_id()
                .await
                .expect("a non-empty race joins an input");
            // A panicked input resumes its panic here.
            let (idx, settled_result) = joined.into_value();
            join_set.abort_all();
            settled_result
                .map(|value| (idx, value))
                .map_err(FirstFailure::Settled)
        };
        let joined = run_join(
            &self.ctx,
            &positional_id,
            &wire_id,
            self.name.as_deref(),
            run,
            join,
            first_failure_wire,
        )
        .await;
        settle_first(&self.ctx, &wire_id, self.name.as_deref(), start, joined).await
    }
}

/// The test hook called with an input's index immediately before the
/// input's retained remainder is spawned.
#[cfg(test)]
type RemainderRecorder = Box<dyn Fn(usize)>;

#[cfg(test)]
thread_local! {
    /// This thread's [`RemainderRecorder`], when a test installed one.
    static REMAINDER_RECORDER: std::cell::RefCell<Option<RemainderRecorder>> =
        const { std::cell::RefCell::new(None) };
}

/// Calls the remainder recorder of this thread, when a test installed one.
#[cfg(test)]
fn record_remainder(idx: usize) {
    REMAINDER_RECORDER.with(|recorder| {
        if let Some(recorder) = &*recorder.borrow() {
            recorder(idx);
        }
    });
}

/// Spawns one combinator input onto `join_set`.
///
/// The input future runs on its own task; an input that waits stays pending
/// in the set until a checkpoint response completes it. The join set
/// blesses the task before its first poll. When no Tokio runtime is
/// running, the input is dropped and the combinator returns the error.
fn spawn_constituent<O: Send + 'static>(
    ctx: &DurableContext,
    join_set: &mut TrackedJoinSet<(usize, Result<O, OperationError>)>,
    idx: usize,
    future: DurableFuture<O>,
) -> Result<(), OperationError> {
    // The input runs in the context that created it, not the combinator's,
    // as a JS input promise exists before the combinator.
    join_set
        .spawn(future.task_context(), ctx.task_ownership(), async move {
            (idx, future.await)
        })
        .map(drop)
        .map_err(NoRuntime::operation_error)
}

/// The message of the context usage error for the first still-unclaimed
/// combinator input, or `None` when every input claimed its ID.
///
/// A combinator claims its operation ID on its first poll or at `.spawn()`.
/// An outer combinator polls its inputs on tasks it spawns, so two
/// unclaimed inner combinators would claim their IDs in scheduler order,
/// and the IDs would differ between runs. JS rejects the same handler with
/// its context usage error. So every path that would start inputs checks
/// this before it begins any input. It records nothing:
/// [`reject_unclaimed`] does, once the combinator's own start step allows.
///
/// `outer` is the positional ID of the outer combinator.
fn unclaimed_input_message<O: Send + 'static>(
    outer: &str,
    futures: &[DurableFuture<O>],
) -> Option<String> {
    futures
        .iter()
        .find(|future| future.is_unclaimed_combinator())
        .and_then(|future| future.unclaimed_input_message(outer))
}

/// Records the context usage error `message` as execution-fatal, and
/// returns it. No input runs, and the outer combinator writes neither
/// `SUCCEED` nor `FAIL`.
fn reject_unclaimed(ctx: &DurableContext, message: &str) -> OperationError {
    ctx.coordinator()
        .record_fatal("Error".to_owned(), message.to_owned());
    combinator_internal_error(message)
}

/// How a combinator runs its inputs in this cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Run {
    /// No record: queue `START`, then run the inputs.
    Live,
    /// A `STARTED`, `PENDING` or `READY` record: fire the replay start
    /// hook, then run the inputs.
    Resumed,
    /// A `SUCCEEDED` record with `ReplayChildren`: run the inputs again to
    /// rebuild the result, and write nothing.
    Reconstruct,
}

/// A combinator's running inputs, and its retained `START` write.
struct Running<O> {
    join_set: TrackedJoinSet<(usize, Result<O, OperationError>)>,
    /// The combinator's `START` write, which its outcome write awaits.
    /// `None` when the cycle queued no `START`.
    start: Option<SubmittedWrite>,
}

/// Begins the combinator's inputs, then runs its own start step, then
/// spawns the inputs' remainders, in that order:
///
/// 1. A still-unclaimed combinator input is detected first (see
///    [`unclaimed_input_message`]). Reconstruction records and returns the
///    error at once. Otherwise the combinator runs its own start step,
///    waits for its `START` write when it queued one, and only then
///    records and returns the error. A failed `START` write ends the
///    invocation instead, so its failure wins. No input begins.
/// 2. Each input's start phase begins, in input order. A lazy operation
///    whose execution begins with the start step runs it now, on this
///    task (see [`DurableFuture::begin_in_creator_context`]). A lazy invoke or
///    `wait_for_condition` is spawned unchanged, and this waits until it
///    has been polled once, but not for its `START`. Every other input,
///    such as a handle that `.spawn()` returned, is left as it is.
/// 3. The combinator's own start step: without a record, its `START` is
///    reserved, `on_operation_start` fires, and the `START` is released
///    without waiting for the response, as a JS child context does. A
///    resumable record fires the replay start hook instead.
/// 4. The retained remainders of step 2 are spawned, in input order.
async fn start_inputs<O: Send + 'static>(
    ctx: &DurableContext,
    positional_id: &str,
    wire_id: &str,
    name: Option<&str>,
    run: Run,
    futures: Vec<DurableFuture<O>>,
) -> Result<Running<O>, OperationError> {
    if let Some(message) = unclaimed_input_message(positional_id, &futures) {
        drop(futures);
        if let Some(start) = combinator_start(ctx, wire_id, name, run)
            && let Err(err) = start.await
        {
            return ctx.checkpoint_failure_unrecoverable(err).await;
        }
        return Err(reject_unclaimed(ctx, &message));
    }

    let mut join_set = constituent_set(ctx);
    let mut remainders = Vec::with_capacity(futures.len());
    for (idx, mut future) in futures.into_iter().enumerate() {
        if future.starts_on_first_poll() {
            let (polled, acknowledged) = tokio::sync::oneshot::channel();
            // The input's first poll runs in the context that created it,
            // as its remainder does (see `spawn_constituent`).
            let task_cx = future.task_context();
            let future = FirstPoll {
                future,
                polled: Some(polled),
            };
            join_set
                .spawn(
                    task_cx,
                    ctx.task_ownership(),
                    async move { (idx, future.await) },
                )
                .map_err(NoRuntime::operation_error)?;
            // A dropped sender means the task stopped before its first
            // poll: there is nothing to wait for.
            let _polled = acknowledged.await;
        } else {
            future.begin_in_creator_context();
            remainders.push((idx, future));
        }
    }

    let start = combinator_start(ctx, wire_id, name, run);

    for (idx, future) in remainders {
        #[cfg(test)]
        record_remainder(idx);
        spawn_constituent(ctx, &mut join_set, idx, future)?;
    }
    Ok(Running { join_set, start })
}

/// An input future that reports its first poll through `polled`.
struct FirstPoll<O> {
    future: DurableFuture<O>,
    polled: Option<tokio::sync::oneshot::Sender<()>>,
}

impl<O: Send + 'static> Future for FirstPoll<O> {
    type Output = Result<O, OperationError>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = &mut *self;
        let poll = std::pin::Pin::new(&mut this.future).poll(cx);
        if let Some(polled) = this.polled.take() {
            // The combinator may have stopped waiting; nothing to report.
            let _sent = polled.send(());
        }
        poll
    }
}

/// The combinator's own start step for `run` (see [`start_inputs`]).
/// Returns the retained `START` write when it queued one.
fn combinator_start(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    run: Run,
) -> Option<SubmittedWrite> {
    match run {
        Run::Live => Some(checkpoint_start(ctx, wire_id, name)),
        Run::Resumed => {
            combinator_resumed(ctx, wire_id, name);
            None
        }
        Run::Reconstruct => None,
    }
}

/// Creates the join set of one combinator's inputs, counted on the
/// invocation's task tracker.
fn constituent_set<O: Send + 'static>(
    ctx: &DurableContext,
) -> TrackedJoinSet<(usize, Result<O, OperationError>)> {
    TrackedJoinSet::new(Arc::clone(ctx.coordinator()))
}

/// Refuses a combinator's terminal checkpoint once an execution-fatal error
/// has been recorded.
///
/// Called at the top of [`checkpoint_succeed`] and [`checkpoint_fail`], which
/// every combinator's live path funnels through. A replay identity mismatch
/// is recorded on the shared fatal slot eagerly: when the mismatching
/// constituent [`DurableFuture`] was finalized, or, for a combinator
/// constituent, when it claimed its ID (see `preflight_identity!` in
/// `builders`), so by the time a short-circuiting combinator (`select_ok`,
/// `race`, `try_join_all`) is ready to record a winner, the slot already
/// says the recorded history contradicts the handler. Writing a SUCCEED (or
/// an unrelated FAIL) checkpoint at that point would store an outcome the
/// execution can never legitimately replay; instead the combinator returns
/// the fatal as its own error and the invocation driver fails the execution
/// with the dedicated `NonDeterministicExecutionError`.
fn fatal_gate(ctx: &DurableContext) -> Result<(), OperationError> {
    if let Some(fatal) = ctx.coordinator().fatal_error() {
        return Err(combinator_internal_error(&fatal.error_message));
    }
    Ok(())
}

/// Queues the `START` of a combinator operation, as the start step of a
/// child context does (see [`crate::start`]): the `START` is reserved,
/// `on_operation_start` fires, and the `START` is released. Returns the
/// write without waiting for it, so no request carries the `START` before
/// the hook dispatch returns.
fn checkpoint_start(ctx: &DurableContext, wire_id: &str, name: Option<&str>) -> SubmittedWrite {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(COMBINATOR_SUB_TYPE.to_owned())
        .action(OperationAction::Start);

    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    let reserved = ctx.reserve_checkpoint_updates(vec![update]);
    if let Some(plugins) = ctx.plugins() {
        let info = combinator_info(ctx, wire_id, name)
            .with_status(crate::OperationStatus::Started)
            .with_replay(false);
        crate::plugin::operation_started(plugins, &info);
    }
    reserved.release()
}

/// The plugin info of the combinator `wire_id`, created in `ctx`.
fn combinator_info(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
) -> crate::plugin::OperationInfo {
    crate::plugin::OperationInfo::for_operation(
        ctx,
        wire_id,
        crate::OperationKind::Context,
        COMBINATOR_SUB_TYPE,
        name,
    )
}

/// Runs a combinator's join `body`, and derives the wire of its error.
///
/// On a live or resumed cycle the join is a user function: it runs
/// through the runner, under its caller's active body, with no scope, and
/// its info is the one `on_operation_start` received. A reconstruction
/// (`ReplayChildren`) runs the body directly, with no frame and no
/// user-function hook.
async fn run_join<T, E, B>(
    ctx: &DurableContext,
    positional_id: &str,
    wire_id: &str,
    name: Option<&str>,
    run: Run,
    body: B,
    wire: fn(&E) -> crate::error::WireError,
) -> Result<T, (E, crate::error::WireError)>
where
    B: Future<Output = Result<T, E>>,
{
    if run == Run::Reconstruct {
        return body.await.map_err(|error| {
            let derived = wire(&error);
            (error, derived)
        });
    }
    let info = combinator_info(ctx, wire_id, name);
    let info = if run == Run::Live {
        info.with_status(crate::OperationStatus::Started)
            .with_replay(false)
    } else {
        info.with_replay(true)
    };
    crate::plugin::run_user_function_at(
        ctx.shared_plugins(),
        Arc::new(info),
        crate::context::current_body(),
        positional_id.to_owned(),
        tracing::Span::current(),
        move || body,
        wire,
    )
    .await
    .into_parts()
}

/// The wire of a failed `try_join_all`: the `JoinFailed` discriminator
/// with the failed input's index appended, so replay reconstructs the
/// same variant and index; the message, `error_data` and `stack_trace`
/// derive from the loser itself (pass-through; a chain that recorded no
/// frames records none).
fn join_failed_wire((index, error): &(usize, OperationError)) -> crate::error::WireError {
    crate::error::wire_error_with_type(error, &format!("{JOIN_FAILED_ERROR_TYPE_PREFIX}{index}"))
}

/// Why a `select_ok` or `race` join failed.
enum FirstFailure {
    /// The input list was empty.
    Empty(&'static str),
    /// `select_ok`: every input failed, in input order.
    AllFailed(Vec<crate::error::Source>),
    /// `race`: the first input to settle failed.
    Settled(OperationError),
}

/// The wire of a failed `select_ok` or `race` join.
fn first_failure_wire(failure: &FirstFailure) -> crate::error::WireError {
    match failure {
        FirstFailure::Empty(message) => {
            crate::error::WireError::new(Some(EMPTY_INPUT_ERROR_TYPE), Some(*message))
        }
        FirstFailure::AllFailed(losers) => all_failed_wire(losers),
        // The discriminator lets replay reproduce the same variant, while
        // the message, `error_data` and `stack_trace` derive from the
        // loser itself.
        FirstFailure::Settled(error) => {
            crate::error::wire_error_with_type(error, FIRST_SETTLED_FAILED_ERROR_TYPE)
        }
    }
}

/// The wire of an `AllFailed` `select_ok`: the discriminator, a message
/// naming each loser's recorded message (see
/// [`wire_message`](crate::error::wire_message)) in input order, and, as
/// `error_data`, each loser's complete wire failure record (a JSON array
/// of [`SerializedLoser`]), so replay rebuilds one loser per input with
/// the same fields the live failures carried.
fn all_failed_wire(losers: &[crate::error::Source]) -> crate::error::WireError {
    let loser_summary = losers
        .iter()
        .enumerate()
        .map(|(idx, err)| format!("[{idx}] {}", crate::error::wire_message(&**err)))
        .collect::<Vec<_>>()
        .join("; ");
    // Pass each loser's own recorded wire record through verbatim when it
    // carries one (the normal case: a failed constituent operation
    // attaches its terminal wire record); derive one only for a wire-less
    // error (e.g. an internal join failure). A derived record keeps the
    // loser's flattened chain as its message.
    let serialized_losers: Vec<SerializedLoser> = losers
        .iter()
        .map(|err| {
            match err
                .downcast_ref::<OperationError>()
                .and_then(|op| op.wire())
            {
                Some(wire) => SerializedLoser::from_wire(wire),
                None => SerializedLoser {
                    message: Some(crate::error::chain_string(&**err)),
                    ..SerializedLoser::from_wire(&crate::error::wire_error_for(
                        &**err,
                        COMBINATOR_ERROR_TYPE,
                    ))
                },
            }
        })
        .collect();
    // `SerializedLoser` holds only strings, options of strings and a
    // vector of strings. JSON serialization of those types cannot fail,
    // so this `expect` never fires.
    #[expect(clippy::expect_used)]
    let losers_json = serde_json::to_string(&serialized_losers)
        .expect("a SerializedLoser always serializes to JSON");
    crate::error::WireError::new(
        Some(ALL_FAILED_ERROR_TYPE),
        Some(format!(
            "all {} futures failed: {loser_summary}",
            losers.len()
        )),
    )
    .with_error_data(Some(losers_json))
}

/// Fires `on_operation_start` with `is_replay` `true` for a combinator
/// whose record exists and is not terminal.
fn combinator_resumed(ctx: &DurableContext, wire_id: &str, name: Option<&str>) {
    if let Some(plugins) = ctx.plugins() {
        let info = combinator_info(ctx, wire_id, name).with_replay(true);
        crate::plugin::operation_started(plugins, &info);
    }
}

/// Serializes a combinator result to its JSON string.
///
/// The live path writes this string (or, when it is oversized, a smaller
/// stand-in) and returns the value decoded from it. Reconstruction of an
/// oversized record computes the same string and returns the value decoded
/// from it, so both paths return the same value.
fn serialize_result<T: Serialize + ?Sized>(value: &T) -> Result<String, OperationError> {
    serde_json::to_string(value)
        .map_err(|e| combinator_internal_error(&format!("serialization failed: {e}")))
}

/// Checkpoints the SUCCEED of a `try_join_all` or `join_all`.
///
/// A result whose serialized string has more than
/// [`CHECKPOINT_SIZE_LIMIT_BYTES`] bytes does not fit in the record. So the
/// record stores the payload `""` and carries `ReplayChildren`, as JS and the
/// Rust child context do, and replay drives the inputs again. A smaller
/// result is stored inline, and carries `ReplayChildren` within the plugin
/// child operations depth of `ctx`, the caller's context.
async fn checkpoint_collection_succeed(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    start: Option<SubmittedWrite>,
    serialized: &str,
) -> Result<(), OperationError> {
    if serialized.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
        checkpoint_succeed(ctx, wire_id, name, start, "", true).await
    } else {
        let preserve = ctx.preserves_children(1);
        checkpoint_succeed(ctx, wire_id, name, start, serialized, preserve).await
    }
}

/// Checkpoints the SUCCEED of a `race` or `select_ok` whose input `winner`
/// won.
///
/// A winner whose serialized value has more than
/// [`CHECKPOINT_SIZE_LIMIT_BYTES`] bytes does not fit in the record. So the
/// record stores a [`RecordedWinner`] with the winner's index and carries
/// `ReplayChildren`, and replay drives only that input again. A smaller
/// value is stored inline.
async fn checkpoint_winner_succeed(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    start: Option<SubmittedWrite>,
    winner: usize,
    serialized: &str,
) -> Result<(), OperationError> {
    if serialized.len() > CHECKPOINT_SIZE_LIMIT_BYTES {
        #[expect(clippy::expect_used)]
        // reason: a struct holding one `usize` always serializes to JSON.
        let record =
            serde_json::to_string(&RecordedWinner { winner }).expect("a RecordedWinner serializes");
        checkpoint_succeed(ctx, wire_id, name, start, &record, true).await
    } else {
        checkpoint_succeed(ctx, wire_id, name, start, serialized, false).await
    }
}

/// Rebuilds the value of a `race` or `select_ok` whose SUCCEEDED record
/// carries `ReplayChildren` and a [`RecordedWinner`] payload.
///
/// 1. A still-unclaimed combinator input fails the execution (see
///    [`unclaimed_input_message`]), before any input is driven.
/// 2. The payload names the winner's input index. A missing payload, or one
///    that does not decode, is an internal error.
/// 3. An index past the inputs is an internal error.
/// 4. Every other input is dropped without being polled: a lazy input never
///    runs, and a spawned one is aborted. The winner is awaited directly, so
///    its error, or its wait for the service, is the combinator's.
/// 5. The value is serialized and decoded, as the live path does, so both
///    paths return the same value.
///
/// It writes no checkpoint and emits no `operation_replayed` event, as
/// child-context reconstruction does.
async fn replay_recorded_winner<O: Serialize + DeserializeOwned + Send + 'static>(
    ctx: &DurableContext,
    positional_id: &str,
    futures: Vec<DurableFuture<O>>,
) -> Result<O, OperationError> {
    if let Some(message) = unclaimed_input_message(positional_id, &futures) {
        return Err(reject_unclaimed(ctx, &message));
    }
    let RecordedWinner { winner } = ctx
        .checkpoint_result_payload(positional_id)
        .and_then(|payload| serde_json::from_str(&payload).ok())
        .ok_or_else(|| combinator_internal_error("replay: malformed winner record"))?;
    let count = futures.len();
    // `nth` drops the inputs before the winner, and the iterator drops the
    // inputs after it at the end of this statement.
    let Some(input) = futures.into_iter().nth(winner) else {
        return Err(combinator_internal_error(&format!(
            "replay: recorded winner index {winner} is out of range for {count} inputs"
        )));
    };
    let value = input.await?;
    let serialized = serialize_result(&value)?;
    drop(value);
    replay_single_success::<O>(Some(&serialized))
}

/// Checkpoints a SUCCEED action for a combinator operation.
///
/// The update always carries `payload`, including `""`. When
/// `replay_children` is true, it also carries `ContextOptions` with
/// `replay_children(true)`: the record does not store the combined result,
/// and replay rebuilds it from the inputs.
///
/// Refuses (via [`fatal_gate`]) once an execution-fatal error is recorded.
async fn checkpoint_succeed(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    start: Option<SubmittedWrite>,
    payload: &str,
    replay_children: bool,
) -> Result<(), OperationError> {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    ctx.await_retained_start(wire_id, start, OperationAction::Succeed)
        .await?;

    fatal_gate(ctx)?;

    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(COMBINATOR_SUB_TYPE.to_owned())
        .action(OperationAction::Succeed)
        .payload(payload.to_owned());

    if replay_children {
        builder = builder.context_options(
            aws_sdk_lambda::types::ContextOptions::builder()
                .replay_children(true)
                .build(),
        );
    }
    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        return ctx.checkpoint_failure_unrecoverable(err).await;
    }
    if let Some(plugins) = ctx.plugins() {
        let info = combinator_info(ctx, wire_id, name)
            .with_status(crate::OperationStatus::Succeeded)
            .with_result(Some(payload.to_owned()))
            .with_replay(false);
        crate::plugin::operation_ended(plugins, &info);
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
    start: Option<SubmittedWrite>,
    error: &crate::error::WireError,
) -> Result<(), OperationError> {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

    ctx.await_retained_start(wire_id, start, OperationAction::Fail)
        .await?;

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
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }
    // Within the plugin child operations depth, a failure record carries
    // `ReplayChildren`, as JS writes it.
    if ctx.preserves_children(1) {
        builder = builder.context_options(
            aws_sdk_lambda::types::ContextOptions::builder()
                .replay_children(true)
                .build(),
        );
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        return ctx.checkpoint_failure_unrecoverable(err).await;
    }
    if let Some(plugins) = ctx.plugins() {
        let info = combinator_info(ctx, wire_id, name)
            .with_status(crate::OperationStatus::Failed)
            .with_error(Some(error.clone()))
            .with_replay(false);
        crate::plugin::operation_ended(plugins, &info);
    }
    Ok(())
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

/// Replays a successful `join_all` record as `Vec<Result<O, OperationError>>`
/// with error-aware deserialization.
fn replay_outcomes_success<O: DeserializeOwned>(
    result: Option<&String>,
) -> Result<Vec<Result<O, OperationError>>, OperationError> {
    let payload = result
        .ok_or_else(|| combinator_internal_error("replay: succeeded record has no payload"))?;
    let serialized: Vec<SerializedOutcome<O>> = serde_json::from_str(payload)
        .map_err(|e| combinator_internal_error(&format!("replay deserialization failed: {e}")))?;

    // Restore each outcome: rejected items become `Err(OperationError)`s
    // whose synthetic sources carry the recorded wire fields
    // (`error_type`, message, `error_data`, `stack_trace`), not just the
    // recorded message (issue #41).
    let outcomes = serialized
        .into_iter()
        .map(|s| match s {
            SerializedOutcome::Fulfilled { value } => Ok(value),
            SerializedOutcome::Rejected {
                message,
                error_type,
                error_data,
                stack_trace,
            } => {
                let wire = crate::error::WireError::new(error_type, Some(message))
                    .with_error_data(error_data)
                    .with_stack_trace(stack_trace);
                Err(OperationError::from_kind(OperationErrorKind::ChildContext(
                    ChildContextError::new(
                        ChildContextErrorKind::ChildFailed,
                        Some(crate::error::ReplayedFailure::source_from(wire.clone())),
                    ),
                ))
                .with_wire(wire))
            }
        })
        .collect();
    Ok(outcomes)
}

/// Replays a failed combinator from a checkpoint record.
///
/// The wire `error_type` is the variant discriminator: failures recorded
/// as [`FIRST_SETTLED_FAILED_ERROR_TYPE`], [`EMPTY_INPUT_ERROR_TYPE`],
/// [`JOIN_FAILED_ERROR_TYPE_PREFIX`] (carrying the failed input's index
/// as its suffix), or [`ALL_FAILED_ERROR_TYPE`] reconstruct the same
/// [`CombinatorErrorKind`] variant the live path produced, so an error
/// observed live and the same error observed on replay are
/// indistinguishable. For `AllFailed`, the record's `error_data` carries
/// the SDK-authored losers payload (one [`SerializedLoser`] per input, in
/// input order), and each recorded loser rebuilds as its own
/// [`crate::error::ReplayedFailure`] with the same wire fields the live
/// loser carried. An `AllFailed` record matches only when its
/// `error_data` decodes as a non-empty losers array. Everything else
/// reconstructs as `Internal` carrying the recorded message.
fn replay_combinator_failure(wire: crate::error::WireError, wire_id: &str) -> OperationError {
    if wire.error_type() == Some(EMPTY_INPUT_ERROR_TYPE) {
        return combinator_empty_input_error()
            .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
            .with_wire(wire);
    }
    let recorded_join_index = wire
        .error_type()
        .and_then(|t| t.strip_prefix(JOIN_FAILED_ERROR_TYPE_PREFIX))
        .and_then(|suffix| suffix.parse::<usize>().ok());
    let (kind, failures) = if let Some(failed_index) = recorded_join_index {
        (
            CombinatorErrorKind::JoinFailed(crate::error::JoinFailed::new(failed_index)),
            vec![crate::error::ReplayedFailure::source_from(wire.clone())],
        )
    } else if wire.error_type() == Some(ALL_FAILED_ERROR_TYPE)
        && let Some(losers) = wire
            .error_data()
            .and_then(|data| serde_json::from_str::<Vec<SerializedLoser>>(data).ok())
            .filter(|losers| !losers.is_empty())
    {
        // One ReplayedFailure per input, each carrying its loser's own
        // recorded wire fields. The payload is SDK-authored (see
        // [`ALL_FAILED_ERROR_TYPE`]).
        let failures = losers
            .into_iter()
            .map(|loser| crate::error::ReplayedFailure::source_from(loser.into_wire()))
            .collect();
        (CombinatorErrorKind::AllFailed, failures)
    } else if wire.error_type() == Some(FIRST_SETTLED_FAILED_ERROR_TYPE) {
        (
            CombinatorErrorKind::FirstSettledFailed,
            vec![crate::error::ReplayedFailure::source_from(wire.clone())],
        )
    } else {
        (
            CombinatorErrorKind::Internal,
            vec![crate::error::ReplayedFailure::source_from(wire.clone())],
        )
    };
    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
        kind, failures,
    )))
    .with_operation(wire_id, CheckpointStatus::Failed.wire_str())
    .with_wire(wire)
}

#[cfg(test)]
#[expect(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::items_after_statements
)]
// reason: test assertions
// Tests deliberately spawn foreign (unblessed) tasks to exercise runtime
// behavior; production spawning is confined to the src/task.rs gateway.
#[expect(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::client::InMemoryExecutionClient;
    use crate::engine::{CheckpointLog, CheckpointRecord, CheckpointStatus};
    use std::future::Future;
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
        use crate::client::{
            CheckpointOutput, ExecutionClient, ExecutionClientError, GetStateOutput,
        };
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
            ) -> Pin<
                Box<
                    dyn Future<Output = Result<CheckpointOutput, ExecutionClientError>> + Send + '_,
                >,
            > {
                Box::pin(async {
                    Ok(CheckpointOutput {
                        checkpoint_token: Some("token-2".to_owned()),
                        updated_operations: vec![],
                        next_marker: None,
                    })
                })
            }

            fn get_state(
                &self,
                _arn: &str,
                _token: &str,
                _marker: Option<&str>,
            ) -> Pin<
                Box<dyn Future<Output = Result<GetStateOutput, ExecutionClientError>> + Send + '_>,
            > {
                Box::pin(async { Ok(GetStateOutput::new(vec![], None)) })
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
                due_at: None,
                op_name: None,
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
            },
        )
    }

    /// Helper to create a failed checkpoint record for a combinator.
    fn failed_record(
        positional_id: &str,
        err_type: &str,
        err_msg: &str,
    ) -> (String, CheckpointRecord) {
        failed_record_with_data(positional_id, err_type, err_msg, None)
    }

    /// Helper to create a failed checkpoint record carrying an
    /// `error_data` payload (e.g. an `AllFailed` losers payload).
    fn failed_record_with_data(
        positional_id: &str,
        err_type: &str,
        err_msg: &str,
        error_data: Option<String>,
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
                error_data,
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
                due_at: None,
                op_name: None,
                parent_id: None,
                start_timestamp: None,
                end_timestamp: None,
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
            futures: vec![], // No futures needed: replay returns stored result.
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

    // === panic propagation tests ===
    //
    // A panic in a combinator input fails the invocation: the combinator
    // resumes it with its original payload and records nothing for it.

    /// A context over an in-memory client, whose recorded updates a test
    /// reads.
    fn recording_ctx(log: CheckpointLog) -> (DurableContext, Arc<InMemoryExecutionClient>) {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token-1".to_owned(),
        );
        (ctx, client)
    }

    /// Drives the combinator `run` builds through the invocation driver,
    /// asserts that the driver returned a panic, and returns its `&'static str`
    /// payload and whether any recorded update has action `FAIL`.
    async fn panic_of<F, Fut>(run: F) -> (Option<&'static str>, bool)
    where
        F: FnOnce(DurableContext) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let ownership = Arc::clone(ctx.task_ownership());
        let signal = Arc::clone(ctx.coordinator());
        let root = ctx.clone();
        let handler = run(ctx);
        let outcome = crate::driver::drive_outcome(
            async move {
                handler.await;
                Ok::<_, crate::error::WireError>("done".to_owned())
            },
            ownership,
            signal,
            root,
            None,
        )
        .await;
        let Err(payload) = outcome else {
            panic!("the input panic must reach the invocation driver, got {outcome:?}");
        };
        let wrote_fail = client
            .recorded_updates()
            .iter()
            .any(|u| u.action() == &aws_sdk_lambda::types::OperationAction::Fail);
        (
            crate::task::test_support::str_payload(payload.as_ref()),
            wrote_fail,
        )
    }

    #[tokio::test]
    async fn try_join_all_input_panic_propagates_payload() {
        let (message, wrote_fail) = panic_of(|ctx| async move {
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
            let _ = exec.execute().await;
        })
        .await;
        assert_eq!(message, Some("boom-idx1"));
        assert!(!wrote_fail, "a panic records no FAIL");
    }

    #[tokio::test]
    async fn join_all_input_panic_propagates_payload() {
        let (message, wrote_fail) = panic_of(|ctx| async move {
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
            let _ = exec.execute().await;
        })
        .await;
        assert_eq!(message, Some("boom-idx1"));
        assert!(!wrote_fail, "a panic records no FAIL");
    }

    #[tokio::test]
    async fn select_ok_input_panic_propagates_payload() {
        let (message, wrote_fail) = panic_of(|ctx| async move {
            let a: DurableFuture<String> = DurableFuture::from_async(async { panic!("boom-idx0") });
            let b: DurableFuture<String> = DurableFuture::from_async(std::future::pending());
            let op_id = ctx.mint_id();
            let exec = SelectOkExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![a, b],
            };
            let _ = exec.execute().await;
        })
        .await;
        assert_eq!(message, Some("boom-idx0"));
        assert!(!wrote_fail, "a panic records no FAIL");
    }

    #[tokio::test]
    async fn race_input_panic_propagates_payload() {
        let (message, wrote_fail) = panic_of(|ctx| async move {
            let a: DurableFuture<String> = DurableFuture::from_async(async { panic!("boom-idx0") });
            let b: DurableFuture<String> = DurableFuture::from_async(std::future::pending());
            let op_id = ctx.mint_id();
            let exec = RaceExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![a, b],
            };
            let _ = exec.execute().await;
        })
        .await;
        assert_eq!(message, Some("boom-idx0"));
        assert!(!wrote_fail, "a panic records no FAIL");
    }

    // === replay of recorded failures ===

    /// A `try_join_all` record whose error type names the failed input's
    /// index replays as `JoinFailed` with that index.
    #[tokio::test]
    async fn try_join_all_join_failed_record_replays() {
        let (wire_id, record) = failed_record("1", "CombinatorError.JoinFailed.1", "input failed");
        let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let op_id = ctx.mint_id();
        let exec = TryJoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        let OperationErrorKind::Combinator(ce) = err.kind() else {
            panic!("expected Combinator, got: {:?}", err.kind());
        };
        let CombinatorErrorKind::JoinFailed(details) = ce.kind() else {
            panic!("expected JoinFailed, got: {:?}", ce.kind());
        };
        assert_eq!(details.failed_index(), 1);
    }

    /// The live path writes fulfilled outcomes with the value and rejected
    /// outcomes with the input's recorded error type and message.
    #[tokio::test]
    async fn join_all_live_payload_keeps_wire_format() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let ok = DurableFuture::from_async(async { Ok(1_u32) });
        let bad: DurableFuture<u32> = DurableFuture::from_async(async {
            Err(
                OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["boom".to_owned().into()],
                )))
                .with_wire(crate::error::WireError::new(
                    Some("Error".to_owned()),
                    Some("boom".to_owned()),
                )),
            )
        });
        let op_id = ctx.mint_id();
        JoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![ok, bad],
        }
        .execute()
        .await
        .expect("the combinator itself succeeds");

        assert_eq!(
            succeed_payload(&client),
            r#"[{"status":"fulfilled","value":1},{"status":"rejected","message":"boom","error_type":"Error"}]"#
        );
    }

    // === no Tokio runtime ===
    //
    // Each combinator returns the no-runtime error when it cannot start an
    // input's task, instead of panicking.

    /// Polls the combinator `run` builds once on a thread with no Tokio
    /// runtime, and asserts that it resolved to the no-runtime error.
    fn assert_no_runtime_error<F, Fut, T>(run: F)
    where
        F: FnOnce(DurableContext) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T, OperationError>>,
        T: Send + std::fmt::Debug + 'static,
    {
        // The context is built on the polling thread, so that thread owns it.
        let polled = crate::task::test_support::poll_once_without_runtime(move || {
            run(test_ctx_with_client(CheckpointLog::empty()))
        });
        let std::task::Poll::Ready(Err(err)) = polled else {
            panic!("expected an immediate error, got {polled:?}");
        };
        assert!(
            format!("{err:#}").contains(crate::task::NO_RUNTIME_MESSAGE),
            "unexpected error: {err:#}"
        );
    }

    /// Two inputs that each succeed at once.
    fn two_inputs() -> Vec<DurableFuture<i32>> {
        vec![
            DurableFuture::from_async(async { Ok(1) }),
            DurableFuture::from_async(async { Ok(2) }),
        ]
    }

    /// An invoke or `wait_for_condition` input is spawned to be polled
    /// once before the combinator's start step. Without a runtime that
    /// spawn fails, and the combinator returns the error.
    #[test]
    fn first_poll_input_without_runtime_returns_error() {
        assert_no_runtime_error(|ctx| {
            let op_id = ctx.mint_id();
            let input = DurableFuture::for_first_poll_operation(&ctx, async { Ok(1) });
            TryJoinAllExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![input],
            }
            .execute()
        });
    }

    #[test]
    fn try_join_all_without_runtime_returns_error() {
        assert_no_runtime_error(|ctx| {
            let op_id = ctx.mint_id();
            TryJoinAllExecution {
                ctx,
                op_id,
                name: None,
                futures: two_inputs(),
            }
            .execute()
        });
    }

    #[test]
    fn join_all_without_runtime_returns_error() {
        assert_no_runtime_error(|ctx| {
            let op_id = ctx.mint_id();
            JoinAllExecution {
                ctx,
                op_id,
                name: None,
                futures: two_inputs(),
            }
            .execute()
        });
    }

    #[test]
    fn select_ok_without_runtime_returns_error() {
        assert_no_runtime_error(|ctx| {
            let op_id = ctx.mint_id();
            SelectOkExecution {
                ctx,
                op_id,
                name: None,
                futures: two_inputs(),
            }
            .execute()
        });
    }

    #[test]
    fn race_without_runtime_returns_error() {
        assert_no_runtime_error(|ctx| {
            let op_id = ctx.mint_id();
            RaceExecution {
                ctx,
                op_id,
                name: None,
                futures: two_inputs(),
            }
            .execute()
        });
    }

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
        assert!(matches!(&result[0], Ok(1)));
        assert!(result[1].is_err());
        assert!(matches!(&result[2], Ok(3)));
    }

    #[tokio::test]
    async fn join_all_replay_with_error_aware_serdes() {
        // Simulate replay with mixed outcomes including rejections.
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
        assert!(matches!(&result[0], Ok(42)));
        assert!(matches!(&result[1], Err(e) if format!("{e:#}").contains("oops")));
        assert!(matches!(&result[2], Ok(99)));
    }

    // === select_ok tests ===

    #[tokio::test]
    async fn select_ok_first_success() {
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        // b fails, a succeeds: a should win.
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
        // The aggregate record persists each loser's own wire record (in
        // input order) in its `error_data`, so replay can rebuild the
        // individual losers rather than clones of the aggregate.
        let wire = err.wire().expect("live AllFailed carries wire record");
        let payload = wire.error_data().expect("AllFailed persists losers");
        let losers: Vec<SerializedLoser> =
            serde_json::from_str(payload).expect("losers payload decodes");
        assert_eq!(losers.len(), 2);
        assert!(
            losers[0]
                .message
                .as_deref()
                .is_some_and(|m| m.contains("err-a")),
            "loser 0 keeps its own message: {:?}",
            losers[0].message
        );
        assert!(
            losers[1]
                .message
                .as_deref()
                .is_some_and(|m| m.contains("err-b")),
            "loser 1 keeps its own message: {:?}",
            losers[1].message
        );
    }

    /// Replaying a recorded `AllFailed` rebuilds EACH loser from the
    /// persisted losers payload: every `failures()` entry carries its own
    /// recorded `error_type`, message, `error_data`, and `stack_trace`,
    /// not clones of the aggregate record.
    #[tokio::test]
    async fn select_ok_all_failed_replay_rebuilds_each_recorded_loser() {
        let losers_payload = serde_json::to_string(&vec![
            SerializedLoser {
                error_type: Some("TypeA".to_owned()),
                message: Some("boom-a".to_owned()),
                error_data: Some(r#"{"a":1}"#.to_owned()),
                stack_trace: vec!["frame-a0".to_owned(), "frame-a1".to_owned()],
            },
            SerializedLoser {
                error_type: Some("TypeB".to_owned()),
                message: Some("boom-b".to_owned()),
                error_data: None,
                stack_trace: Vec::new(),
            },
        ])
        .expect("payload serializes");
        let (wire_id, record) = failed_record_with_data(
            "1",
            "CombinatorError.AllFailed",
            "all 2 futures failed: [0] boom-a; [1] boom-b",
            Some(losers_payload),
        );
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let exec = SelectOkExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![], // No futures needed on replay.
        };
        let err = exec.execute().await.unwrap_err();
        let OperationErrorKind::Combinator(ce) = err.kind() else {
            panic!("expected Combinator, got: {:?}", err.kind());
        };
        assert!(
            matches!(ce.kind(), CombinatorErrorKind::AllFailed),
            "expected AllFailed, got: {:?}",
            ce.kind()
        );
        let failures = ce.failures();
        assert_eq!(failures.len(), 2);
        let replayed = |idx: usize| {
            failures[idx]
                .downcast_ref::<crate::error::ReplayedFailure>()
                .expect("replayed loser is a ReplayedFailure")
                .wire()
                .clone()
        };
        let first = replayed(0);
        assert_eq!(first.error_type(), Some("TypeA"));
        assert_eq!(first.error_message(), Some("boom-a"));
        assert_eq!(first.error_data(), Some(r#"{"a":1}"#));
        assert_eq!(
            first.stack_trace(),
            ["frame-a0".to_owned(), "frame-a1".to_owned()]
        );
        let second = replayed(1);
        assert_eq!(second.error_type(), Some("TypeB"));
        assert_eq!(second.error_message(), Some("boom-b"));
        assert_eq!(second.error_data(), None);
        assert_eq!(second.stack_trace(), &[] as &[String]);
    }

    /// An `AllFailed` record whose `error_data` is absent, not JSON, or an
    /// empty losers array is not an `AllFailed` record the SDK writes. It
    /// replays as `Internal` with one failure carrying the recorded
    /// message, as any record with an unrecognized error type does.
    #[tokio::test]
    async fn select_ok_all_failed_replay_without_valid_losers_is_internal() {
        let message = "all 3 futures failed: [0] a; [1] b; [2] c";
        for error_data in [None, Some("not json".to_owned()), Some("[]".to_owned())] {
            let (wire_id, record) = failed_record_with_data(
                "1",
                "CombinatorError.AllFailed",
                message,
                error_data.clone(),
            );
            let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));

            let op_id = ctx.mint_id();
            let a: DurableFuture<String> =
                DurableFuture::from_async(async { panic!("never runs") });
            let b: DurableFuture<String> =
                DurableFuture::from_async(async { panic!("never runs") });
            let c: DurableFuture<String> =
                DurableFuture::from_async(async { panic!("never runs") });
            let exec = SelectOkExecution::<String> {
                ctx,
                op_id,
                name: None,
                futures: vec![a, b, c],
            };
            let err = exec.execute().await.unwrap_err();
            let OperationErrorKind::Combinator(ce) = err.kind() else {
                panic!("expected Combinator, got: {:?}", err.kind());
            };
            assert!(
                matches!(ce.kind(), CombinatorErrorKind::Internal),
                "{error_data:?}: expected Internal, got: {:?}",
                ce.kind()
            );
            assert_eq!(ce.failures().len(), 1, "{error_data:?}");
            let failure = ce.failures()[0]
                .downcast_ref::<crate::error::ReplayedFailure>()
                .expect("replayed failure is a ReplayedFailure")
                .wire();
            assert_eq!(failure.error_message(), Some(message), "{error_data:?}");
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
        // discriminator names the variant. The loser recorded no frames,
        // and the SDK captures none.
        let wire = err.wire().expect("first-settled failure carries wire");
        assert_eq!(
            wire.error_type(),
            Some("CombinatorError.FirstSettledFailed")
        );
        assert!(
            wire.stack_trace().is_empty(),
            "the SDK captures no stack trace for a first-settled failure"
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
        // The live error carries the checkpointed context: the same
        // operation id, status, and wire record a replay reconstructs.
        assert!(err.operation_id().is_some(), "live EmptyInput has op id");
        assert_eq!(err.status(), Some("FAILED"));
        let wire = err.wire().expect("live EmptyInput carries wire record");
        assert_eq!(wire.error_type(), Some("CombinatorError.EmptyInput"));
        assert!(
            wire.stack_trace().is_empty(),
            "the SDK captures no stack trace for an empty input"
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
        assert_eq!(wire.stack_trace(), [] as [String; 0]);
    }

    #[tokio::test]
    async fn empty_input_replay_reproduces_empty_input() {
        // A recorded empty-input failure replays as EmptyInput, the same
        // variant the live path produced, for both race and select_ok.
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
            futures: vec![], // Replay: no futures run.
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
    // These verify that combinator input tasks are blessed and do NOT
    // trigger ownership errors when the guard is active (context created
    // inside tokio::spawn where try_id() returns Some).
    //
    // Each branch calls `ctx.enforce_task_ownership()`: the same check
    // that every durable operation (step, invoke, etc.) performs at entry.
    // Without the blessing in the task gateway's wrapper, these tests FAIL
    // with ownership errors.

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

        // Must succeed, no ownership errors.
        let values = result.unwrap();
        assert_eq!(values, vec![1, 2]);
    }

    #[tokio::test]
    async fn join_all_ownership_blessed_no_rejections() {
        // Create context inside a spawned task so the ownership guard is active.
        let result = tokio::spawn(async {
            let ctx = test_ctx_with_client(CheckpointLog::empty());

            // Each branch calls enforce_task_ownership: without blessing,
            // every branch would be an Err with an ownership error.
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

        // Must succeed with every branch Ok: none is an Err due to ownership.
        let settled = result.unwrap();
        assert_eq!(settled.len(), 3);
        for (i, s) in settled.iter().enumerate() {
            assert!(
                s.is_ok(),
                "branch {i} must be Ok, not Err with an ownership error"
            );
        }
        assert!(matches!(&settled[0], Ok(10)));
        assert!(matches!(&settled[1], Ok(20)));
        assert!(matches!(&settled[2], Ok(30)));
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

        // Must succeed, no ownership errors.
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

        // Must succeed, no ownership errors.
        let value = result.unwrap();
        assert_eq!(value, "fast");
    }

    // === live value equals replayed value (issue #111) ===

    /// A value whose serde form omits `cached`, so a JSON round trip drops it.
    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct Tagged {
        value: u32,
        #[serde(skip)]
        cached: Option<String>,
    }

    /// An input that resolves to a `Tagged` carrying live-only data.
    fn tagged_input() -> DurableFuture<Tagged> {
        DurableFuture::from_async(async {
            Ok(Tagged {
                value: 7,
                cached: Some("live-only".to_owned()),
            })
        })
    }

    /// An input that fails with a typed wire record.
    fn failing_input<O: Send + 'static>() -> DurableFuture<O> {
        DurableFuture::from_async(async {
            Err(
                OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
                    CombinatorErrorKind::Internal,
                    vec!["input failed".to_owned().into()],
                )))
                .with_wire(crate::error::WireError::new(
                    Some("InputFailure"),
                    Some("input failed"),
                )),
            )
        })
    }

    /// Returns the payload of the one SUCCEED update the client recorded.
    fn succeed_payload(client: &InMemoryExecutionClient) -> String {
        let succeeds: Vec<String> = client
            .recorded_updates()
            .iter()
            .filter(|u| u.action() == &aws_sdk_lambda::types::OperationAction::Succeed)
            .map(|u| u.payload().unwrap_or_default().to_owned())
            .collect();
        assert_eq!(succeeds.len(), 1, "one SUCCEED update: {succeeds:?}");
        succeeds[0].clone()
    }

    /// A context whose log holds a SUCCEED record at positional ID "1"
    /// carrying `payload`, so the first combinator replays it.
    fn replay_ctx(payload: &str) -> DurableContext {
        let (wire_id, record) = succeeded_record("1", payload);
        test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]))
    }

    #[tokio::test]
    async fn try_join_all_live_value_equals_replay() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let live = TryJoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![tagged_input(), tagged_input()],
        }
        .execute()
        .await
        .unwrap();
        let payload = succeed_payload(&client);

        let ctx = replay_ctx(&payload);
        let op_id = ctx.mint_id();
        let replayed = TryJoinAllExecution::<Tagged> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        }
        .execute()
        .await
        .unwrap();

        assert_eq!(live, replayed);
        assert!(live.iter().all(|t| t.cached.is_none() && t.value == 7));
    }

    /// Renders a `join_all` result so a live and a replayed run compare.
    fn render_settled(settled: &[Result<Tagged, OperationError>]) -> Vec<String> {
        settled
            .iter()
            .map(|s| match s {
                Ok(t) => format!("ok:{}:{:?}", t.value, t.cached),
                Err(err) => format!(
                    "err:{:?}:{:?}:{:?}",
                    err.kind(),
                    err.wire().and_then(crate::error::WireError::error_type),
                    err.wire().and_then(crate::error::WireError::error_message),
                ),
            })
            .collect()
    }

    /// Runs `join_all` live over one `Tagged` input and one failing input,
    /// and returns the result with the SUCCEED payload it wrote.
    async fn join_all_live() -> (Vec<Result<Tagged, OperationError>>, String) {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let live = JoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![tagged_input(), failing_input()],
        }
        .execute()
        .await
        .unwrap();
        (live, succeed_payload(&client))
    }

    #[tokio::test]
    async fn join_all_live_value_equals_replay() {
        let (live, payload) = join_all_live().await;

        let ctx = replay_ctx(&payload);
        let op_id = ctx.mint_id();
        let replayed = JoinAllExecution::<Tagged> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        }
        .execute()
        .await
        .unwrap();

        assert_eq!(render_settled(&live), render_settled(&replayed));
        assert!(live.iter().all(|s| match s {
            Ok(t) => t.cached.is_none(),
            Err(_) => true,
        }));
    }

    #[tokio::test]
    async fn join_all_live_rejection_is_child_failed() {
        let (live, payload) = join_all_live().await;

        let Some(Err(err)) = live.get(1) else {
            panic!("item 1 must be rejected: {live:?}");
        };
        let OperationErrorKind::ChildContext(child) = err.kind() else {
            panic!("expected a ChildContext error, got {:?}", err.kind());
        };
        assert!(matches!(child.kind(), ChildContextErrorKind::ChildFailed));

        let recorded: serde_json::Value = serde_json::from_str(&payload).unwrap();
        let recorded_type = recorded[1]["error_type"].as_str();
        assert_eq!(recorded_type, Some("InputFailure"));
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_type),
            recorded_type
        );
    }

    #[tokio::test]
    async fn select_ok_live_value_equals_replay() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let live = SelectOkExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![tagged_input()],
        }
        .execute()
        .await
        .unwrap();
        let payload = succeed_payload(&client);

        let ctx = replay_ctx(&payload);
        let op_id = ctx.mint_id();
        let replayed = SelectOkExecution::<Tagged> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        }
        .execute()
        .await
        .unwrap();

        assert_eq!(live, replayed);
        assert_eq!(live.cached, None);
    }

    #[tokio::test]
    async fn race_live_value_equals_replay() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let live = RaceExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![tagged_input()],
        }
        .execute()
        .await
        .unwrap();
        let payload = succeed_payload(&client);

        let ctx = replay_ctx(&payload);
        let op_id = ctx.mint_id();
        let replayed = RaceExecution::<Tagged> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        }
        .execute()
        .await
        .unwrap();

        assert_eq!(live, replayed);
        assert_eq!(live.cached, None);
    }

    /// A type whose `Serialize` writes `0` and whose `Deserialize` always
    /// fails, so the payload a combinator writes cannot be decoded.
    #[derive(Debug)]
    struct Unreadable;

    impl Serialize for Unreadable {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            serializer.serialize_u32(0)
        }
    }

    impl<'de> serde::Deserialize<'de> for Unreadable {
        fn deserialize<D: serde::Deserializer<'de>>(_deserializer: D) -> Result<Self, D::Error> {
            Err(serde::de::Error::custom("unreadable"))
        }
    }

    /// Returns the message of a `Combinator(Internal)` error.
    fn internal_message(err: &OperationError) -> String {
        let OperationErrorKind::Combinator(ce) = err.kind() else {
            panic!("expected a Combinator error, got {:?}", err.kind());
        };
        assert!(
            matches!(ce.kind(), CombinatorErrorKind::Internal),
            "expected Internal, got {:?}",
            ce.kind()
        );
        ce.failures()[0].to_string()
    }

    #[tokio::test]
    async fn try_join_all_live_decode_failure_matches_replay() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let live = TryJoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![DurableFuture::from_async(async { Ok(Unreadable) })],
        }
        .execute()
        .await
        .unwrap_err();
        let payload = succeed_payload(&client);
        assert_eq!(payload, "[0]");

        let ctx = replay_ctx(&payload);
        let op_id = ctx.mint_id();
        let replayed = TryJoinAllExecution::<Unreadable> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        }
        .execute()
        .await
        .unwrap_err();

        let live_message = internal_message(&live);
        assert_eq!(live_message, internal_message(&replayed));
        assert!(
            live_message.starts_with("replay deserialization failed"),
            "{live_message}"
        );
    }

    // === oversized combined result (issue #112) ===

    /// Returns the one SUCCEED update the client recorded.
    fn succeed_update(client: &InMemoryExecutionClient) -> aws_sdk_lambda::types::OperationUpdate {
        let succeeds: Vec<aws_sdk_lambda::types::OperationUpdate> = client
            .recorded_updates()
            .into_iter()
            .filter(|u| u.action() == &aws_sdk_lambda::types::OperationAction::Succeed)
            .collect();
        assert_eq!(succeeds.len(), 1, "one SUCCEED update: {succeeds:?}");
        succeeds[0].clone()
    }

    /// Returns `ContextOptions.replay_children` of `update`, `None` when the
    /// update carries no `ContextOptions`.
    fn replay_children_of(update: &aws_sdk_lambda::types::OperationUpdate) -> Option<bool> {
        update
            .context_options()
            .and_then(aws_sdk_lambda::types::ContextOptions::replay_children)
    }

    /// An input that resolves to `value`.
    fn ready<O: Send + 'static>(value: O) -> DurableFuture<O> {
        DurableFuture::from_async(async move { Ok(value) })
    }

    /// An input that never resolves.
    fn never<O: Send + 'static>() -> DurableFuture<O> {
        DurableFuture::from_async(std::future::pending())
    }

    /// An input that panics when it is polled.
    fn must_not_run<O: Send + 'static>() -> DurableFuture<O> {
        DurableFuture::from_async(async { panic!("this input must not run") })
    }

    /// An input that sets `flag` when it runs and returns `value`.
    fn flagging(flag: &Arc<AtomicBool>, value: i32) -> DurableFuture<i32> {
        let flag = Arc::clone(flag);
        DurableFuture::from_async(async move {
            flag.store(true, Ordering::SeqCst);
            Ok(value)
        })
    }

    /// The single update with `action` the client recorded.
    fn depth_update(
        client: &InMemoryExecutionClient,
        action: &aws_sdk_lambda::types::OperationAction,
    ) -> aws_sdk_lambda::types::OperationUpdate {
        let found: Vec<_> = client
            .recorded_updates()
            .into_iter()
            .filter(|u| u.action() == action)
            .collect();
        assert_eq!(found.len(), 1, "one {action:?} update: {found:?}");
        found[0].clone()
    }

    /// Runs one combinator execution on a root with depth `depth`, and
    /// returns the client.
    async fn depth_run<T>(
        depth: u32,
        run: impl FnOnce(DurableContext, OperationId) -> T,
    ) -> Arc<InMemoryExecutionClient>
    where
        T: std::future::Future<Output = ()>,
    {
        let (ctx, client) = DurableContext::test_context_with_depth(depth);
        let op_id = ctx.mint_id();
        run(ctx, op_id).await;
        client
    }

    #[tokio::test]
    async fn preserved_try_join_all_success_has_flag() {
        let client = depth_run(1, |ctx, op_id| async move {
            let value = TryJoinAllExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![ready(1), ready(2)],
            }
            .execute()
            .await
            .unwrap();
            assert_eq!(value, vec![1, 2]);
        })
        .await;
        let update = succeed_update(&client);
        assert_eq!(replay_children_of(&update), Some(true));
        assert_eq!(update.payload(), Some("[1,2]"));
    }

    #[tokio::test]
    async fn preserved_join_all_success_has_flag() {
        let run = |depth| {
            depth_run(depth, |ctx, op_id| async move {
                let outcomes = JoinAllExecution {
                    ctx,
                    op_id,
                    name: None,
                    futures: vec![ready(1), failing_input::<i32>()],
                }
                .execute()
                .await
                .unwrap();
                assert_eq!(outcomes.len(), 2);
            })
        };
        let plain = succeed_update(&*run(0).await);
        let update = succeed_update(&*run(1).await);
        assert_eq!(replay_children_of(&plain), None);
        assert_eq!(replay_children_of(&update), Some(true));
        assert_eq!(update.payload(), plain.payload());
    }

    #[tokio::test]
    async fn preserved_race_success_has_no_flag() {
        let client = depth_run(1, |ctx, op_id| async move {
            let value = RaceExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![never(), ready(20)],
            }
            .execute()
            .await
            .unwrap();
            assert_eq!(value, 20);
        })
        .await;
        let update = succeed_update(&client);
        assert_eq!(update.payload(), Some("20"));
        assert!(update.context_options().is_none());
    }

    #[tokio::test]
    async fn preserved_select_ok_success_has_no_flag() {
        let client = depth_run(1, |ctx, op_id| async move {
            let value = SelectOkExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![never(), ready(20)],
            }
            .execute()
            .await
            .unwrap();
            assert_eq!(value, 20);
        })
        .await;
        let update = succeed_update(&client);
        assert_eq!(update.payload(), Some("20"));
        assert!(update.context_options().is_none());
    }

    #[tokio::test]
    async fn preserved_try_join_all_failure_has_flag() {
        let client = depth_run(1, |ctx, op_id| async move {
            let result = TryJoinAllExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![failing_input::<i32>()],
            }
            .execute()
            .await;
            assert!(result.is_err());
        })
        .await;
        let update = depth_update(&client, &aws_sdk_lambda::types::OperationAction::Fail);
        assert_eq!(replay_children_of(&update), Some(true));
    }

    #[tokio::test]
    async fn preserved_race_failure_has_flag() {
        let client = depth_run(1, |ctx, op_id| async move {
            let result = RaceExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![never(), failing_input::<i32>()],
            }
            .execute()
            .await;
            assert!(result.is_err());
        })
        .await;
        let update = depth_update(&client, &aws_sdk_lambda::types::OperationAction::Fail);
        assert_eq!(replay_children_of(&update), Some(true));
    }

    #[tokio::test]
    async fn preserved_select_ok_failure_has_flag() {
        let client = depth_run(1, |ctx, op_id| async move {
            let result = SelectOkExecution {
                ctx,
                op_id,
                name: None,
                futures: vec![failing_input::<i32>(), failing_input::<i32>()],
            }
            .execute()
            .await;
            assert!(result.is_err());
        })
        .await;
        let update = depth_update(&client, &aws_sdk_lambda::types::OperationAction::Fail);
        assert_eq!(replay_children_of(&update), Some(true));
    }

    /// A recording context whose log holds a SUCCEEDED combinator record
    /// at positional ID "1" with `ReplayChildren` and `payload`.
    fn replay_children_ctx(
        payload: Option<&str>,
    ) -> (DurableContext, Arc<InMemoryExecutionClient>) {
        let (wire_id, mut record) = succeeded_record("1", "");
        record.result = payload.map(str::to_owned);
        record.replay_children = true;
        recording_ctx(CheckpointLog::from_records(vec![(wire_id, record)]))
    }

    /// Runs `try_join_all` over one `String` input of `len` bytes and
    /// returns the SUCCEED update.
    async fn try_join_all_one_string(len: usize) -> aws_sdk_lambda::types::OperationUpdate {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let value = TryJoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![ready("x".repeat(len))],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, vec!["x".repeat(len)]);
        succeed_update(&client)
    }

    #[tokio::test]
    async fn succeed_write_inline_at_limit() {
        // `["…"]` adds 4 bytes, so the JSON has exactly 262 144 bytes.
        let update = try_join_all_one_string(262_140).await;
        let payload = update.payload().unwrap();
        assert_eq!(payload.len(), 262_144);
        assert_eq!(payload, format!("[\"{}\"]", "x".repeat(262_140)));
        assert_eq!(replay_children_of(&update), None);
    }

    #[tokio::test]
    async fn succeed_write_replay_children_above_limit() {
        let update = try_join_all_one_string(262_141).await;
        assert_eq!(update.payload(), Some(""));
        assert_eq!(replay_children_of(&update), Some(true));
    }

    #[tokio::test]
    async fn join_all_oversized_writes_replay_children() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let big = "x".repeat(262_145);
        let value = JoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![ready(big.clone())],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value.len(), 1);
        assert_eq!(value[0].as_ref().unwrap(), &big);
        let update = succeed_update(&client);
        assert_eq!(update.payload(), Some(""));
        assert_eq!(replay_children_of(&update), Some(true));
    }

    #[tokio::test]
    async fn race_oversized_records_winner() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let big = "x".repeat(262_145);
        let value = RaceExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![never(), ready(big.clone())],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, big);
        let update = succeed_update(&client);
        assert_eq!(update.payload(), Some("{\"winner\":1}"));
        assert_eq!(replay_children_of(&update), Some(true));
    }

    #[tokio::test]
    async fn select_ok_oversized_records_winner() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let big = "x".repeat(262_145);
        let value = SelectOkExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![never(), ready(big.clone())],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, big);
        let update = succeed_update(&client);
        assert_eq!(update.payload(), Some("{\"winner\":1}"));
        assert_eq!(replay_children_of(&update), Some(true));
    }

    #[tokio::test]
    async fn race_small_result_writes_inline() {
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        let op_id = ctx.mint_id();
        let value = RaceExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![never(), ready("small".to_owned())],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, "small");
        let update = succeed_update(&client);
        assert_eq!(update.payload(), Some("\"small\""));
        assert_eq!(replay_children_of(&update), None);
    }

    #[tokio::test]
    async fn try_join_all_replay_children_reconstructs_without_writing() {
        let (ctx, client) = replay_children_ctx(Some(""));
        let op_id = ctx.mint_id();
        let value = TryJoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![ready(1), ready(2)],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, vec![1, 2]);
        assert_eq!(client.recorded_updates().len(), 0, "no update");
    }

    #[tokio::test]
    async fn try_join_all_replay_children_input_failure_writes_nothing() {
        let (ctx, client) = replay_children_ctx(Some(""));
        let op_id = ctx.mint_id();
        let err = TryJoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![ready(1), failing_input()],
        }
        .execute()
        .await
        .unwrap_err();
        let OperationErrorKind::Combinator(ce) = err.kind() else {
            panic!("expected a Combinator error, got {:?}", err.kind());
        };
        let CombinatorErrorKind::JoinFailed(failed) = ce.kind() else {
            panic!("expected JoinFailed, got {:?}", ce.kind());
        };
        assert_eq!(failed.failed_index(), 1);
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_type),
            Some("CombinatorError.JoinFailed.1")
        );
        assert_eq!(client.recorded_updates().len(), 0, "no update");
    }

    #[tokio::test]
    async fn join_all_replay_children_reconstructs_without_writing() {
        let (ctx, client) = replay_children_ctx(Some(""));
        let op_id = ctx.mint_id();
        let value = JoinAllExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![ready(1), failing_input()],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value.len(), 2);
        assert_eq!(value[0].as_ref().unwrap(), &1);
        let err = value[1].as_ref().unwrap_err();
        assert!(
            matches!(
                err.kind(),
                OperationErrorKind::ChildContext(c) if matches!(c.kind(), ChildContextErrorKind::ChildFailed)
            ),
            "expected ChildContext(ChildFailed), got {:?}",
            err.kind()
        );
        assert_eq!(client.recorded_updates().len(), 0, "no update");
    }

    #[tokio::test]
    async fn race_replay_children_drives_only_recorded_winner() {
        let (ctx, client) = replay_children_ctx(Some("{\"winner\":1}"));
        let flag = Arc::new(AtomicBool::new(false));
        let op_id = ctx.mint_id();
        let value = RaceExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![flagging(&flag, 10), ready(20)],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, 20);
        assert!(!flag.load(Ordering::SeqCst), "input 0 must not run");
        assert_eq!(client.recorded_updates().len(), 0, "no update");
    }

    #[tokio::test]
    async fn select_ok_replay_children_drives_only_recorded_winner() {
        let (ctx, client) = replay_children_ctx(Some("{\"winner\":1}"));
        let flag = Arc::new(AtomicBool::new(false));
        let op_id = ctx.mint_id();
        let value = SelectOkExecution {
            ctx,
            op_id,
            name: None,
            futures: vec![flagging(&flag, 10), ready(20)],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, 20);
        assert!(!flag.load(Ordering::SeqCst), "input 0 must not run");
        assert_eq!(client.recorded_updates().len(), 0, "no update");
    }

    /// Runs `race` over two inputs against a `ReplayChildren` record with
    /// `payload`, and returns the `Combinator(Internal)` error's text.
    async fn race_replay_children_error(payload: Option<&str>) -> String {
        let (ctx, client) = replay_children_ctx(payload);
        let op_id = ctx.mint_id();
        let err = RaceExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![must_not_run(), must_not_run()],
        }
        .execute()
        .await
        .unwrap_err();
        internal_message(&err);
        assert_eq!(client.recorded_updates().len(), 0, "no update");
        format!("{err:#}")
    }

    #[tokio::test]
    async fn race_replay_children_rejects_out_of_range_winner() {
        let text = race_replay_children_error(Some("{\"winner\":5}")).await;
        assert!(
            text.contains("recorded winner index 5 is out of range for 2 inputs"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn race_replay_children_rejects_malformed_winner_record() {
        let text = race_replay_children_error(Some("{}")).await;
        assert!(text.contains("malformed winner record"), "{text}");
    }

    #[tokio::test]
    async fn race_replay_children_rejects_missing_winner_record() {
        let text = race_replay_children_error(None).await;
        assert!(text.contains("malformed winner record"), "{text}");
    }

    #[tokio::test]
    async fn race_replay_children_returns_winner_error_unchanged() {
        let (ctx, client) = replay_children_ctx(Some("{\"winner\":0}"));
        let op_id = ctx.mint_id();
        let err = RaceExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![failing_input(), must_not_run()],
        }
        .execute()
        .await
        .unwrap_err();
        assert_eq!(internal_message(&err), "input failed");
        assert_eq!(
            err.wire().and_then(crate::error::WireError::error_type),
            Some("InputFailure")
        );
        assert_eq!(client.recorded_updates().len(), 0, "no update");
    }

    #[tokio::test]
    async fn try_join_all_replay_without_replay_children_decodes_payload() {
        let (wire_id, record) = succeeded_record("1", "[1,2]");
        assert!(!record.replay_children);
        let (ctx, client) = recording_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let op_id = ctx.mint_id();
        let value = TryJoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![must_not_run(), must_not_run()],
        }
        .execute()
        .await
        .unwrap();
        assert_eq!(value, vec![1, 2]);
        assert_eq!(client.recorded_updates().len(), 0, "no update");
    }

    // === unclaimed combinator inputs (issue #149) ===

    /// The outer combinator of a test-20 case.
    #[derive(Clone, Copy, Debug)]
    enum Outer {
        TryJoinAll,
        JoinAll,
        SelectOk,
        Race,
    }

    /// The record the outer combinator finds at positional ID "1".
    #[derive(Clone, Copy, Debug)]
    enum Recorded {
        Nothing,
        Started,
        Pending,
        ReplayChildren,
    }

    /// Test 20: on every path that would start its inputs, an outer
    /// combinator fails on a still-unclaimed combinator input before it
    /// polls any input, and it writes START only when no record exists.
    #[tokio::test]
    async fn unclaimed_input_check_covers_every_spawn_path() {
        let outers = [
            Outer::TryJoinAll,
            Outer::JoinAll,
            Outer::SelectOk,
            Outer::Race,
        ];
        let records = [
            Recorded::Nothing,
            Recorded::Started,
            Recorded::Pending,
            Recorded::ReplayChildren,
        ];
        for outer in outers {
            for recorded in records {
                let log = match recorded {
                    Recorded::Nothing => CheckpointLog::empty(),
                    Recorded::Started | Recorded::Pending => {
                        let (wire_id, mut record) = succeeded_record("1", "");
                        record.result = None;
                        record.status = if matches!(recorded, Recorded::Started) {
                            CheckpointStatus::Started
                        } else {
                            CheckpointStatus::Pending
                        };
                        CheckpointLog::from_records(vec![(wire_id, record)])
                    }
                    Recorded::ReplayChildren => {
                        let payload = match outer {
                            Outer::TryJoinAll | Outer::JoinAll => "",
                            Outer::SelectOk | Outer::Race => r#"{"winner":0}"#,
                        };
                        let (wire_id, mut record) = succeeded_record("1", payload);
                        record.replay_children = true;
                        CheckpointLog::from_records(vec![(wire_id, record)])
                    }
                };
                let (ctx, client) = recording_ctx(log);
                let polled = Arc::new(AtomicBool::new(false));
                let inner = ctx.race(vec![flagging(&polled, 1)]).name("inner").future();
                // The claimed input comes first, so the check must skip it.
                let inputs = vec![flagging(&polled, 2), inner];
                let failed = match outer {
                    Outer::TryJoinAll => ctx.try_join_all(inputs).await.is_err(),
                    Outer::JoinAll => ctx.join_all(inputs).await.is_err(),
                    Outer::SelectOk => ctx.select_ok(inputs).await.is_err(),
                    Outer::Race => ctx.race(inputs).await.is_err(),
                };
                let case = format!("{outer:?} with {recorded:?}");
                assert!(failed, "{case}: the outer combinator fails");
                let fatal = ctx.coordinator().fatal_error().expect("a fatal error");
                assert_eq!(fatal.error_type, "Error", "{case}");
                assert_eq!(
                    fatal.error_message,
                    "context usage error in \"race\": the combinator was created on context \
                     \"root\", but combinator \"1\" started it as an input. Start the inner \
                     combinator with .spawn() and pass the returned handle to the outer \
                     combinator.",
                    "{case}"
                );
                assert!(!polled.load(Ordering::SeqCst), "{case}: no input is polled");
                assert_eq!(
                    ctx.mint_id().positional(),
                    "2",
                    "{case}: the inner combinator claims nothing"
                );
                let actions: Vec<_> = client
                    .recorded_updates()
                    .iter()
                    .map(|u| u.action().clone())
                    .collect();
                let expected = if matches!(recorded, Recorded::Nothing) {
                    vec![aws_sdk_lambda::types::OperationAction::Start]
                } else {
                    vec![]
                };
                assert_eq!(actions, expected, "{case}: START only without a record");
            }
        }
    }
    /// The four outer combinators, in the order the tests below run them.
    const OUTERS: [Outer; 4] = [
        Outer::TryJoinAll,
        Outer::JoinAll,
        Outer::SelectOk,
        Outer::Race,
    ];

    /// Awaits the combinator `outer`, named `all`, over `inputs`, and
    /// drops its result.
    fn run_outer<O>(
        ctx: &DurableContext,
        outer: Outer,
        inputs: Vec<DurableFuture<O>>,
    ) -> std::pin::Pin<Box<dyn Future<Output = ()> + Send>>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        use std::future::IntoFuture;
        match outer {
            Outer::TryJoinAll => {
                let f = ctx.try_join_all(inputs).name("all").into_future();
                Box::pin(async move { drop(f.await) })
            }
            Outer::JoinAll => {
                let f = ctx.join_all(inputs).name("all").into_future();
                Box::pin(async move { drop(f.await) })
            }
            Outer::SelectOk => {
                let f = ctx.select_ok(inputs).name("all").into_future();
                Box::pin(async move { drop(f.await) })
            }
            Outer::Race => {
                let f = ctx.race(inputs).name("all").into_future();
                Box::pin(async move { drop(f.await) })
            }
        }
    }

    /// Starts the combinator `outer`, named `all`, over `inputs` with its
    /// own `.spawn()`. The returned future resolves to whether it
    /// succeeded.
    fn spawn_outer<O>(
        ctx: &DurableContext,
        outer: Outer,
        inputs: Vec<DurableFuture<O>>,
    ) -> std::pin::Pin<Box<dyn Future<Output = bool> + Send>>
    where
        O: Serialize + DeserializeOwned + Send + 'static,
    {
        match outer {
            Outer::TryJoinAll => {
                let h = ctx.try_join_all(inputs).name("all").spawn();
                Box::pin(async move { h.await.is_ok() })
            }
            Outer::JoinAll => {
                let h = ctx.join_all(inputs).name("all").spawn();
                Box::pin(async move { h.await.is_ok() })
            }
            Outer::SelectOk => {
                let h = ctx.select_ok(inputs).name("all").spawn();
                Box::pin(async move { h.await.is_ok() })
            }
            Outer::Race => {
                let h = ctx.race(inputs).name("all").spawn();
                Box::pin(async move { h.await.is_ok() })
            }
        }
    }

    /// Waits up to two seconds for `ready`. Returns whether it held.
    async fn within_two_seconds(mut ready: impl FnMut() -> bool) -> bool {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        while !ready() {
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        true
    }

    /// One submission the recorder saw: the update's name, ID and action,
    /// and the first-poll flag of its write.
    type Submitted = (
        String,
        String,
        aws_sdk_lambda::types::OperationAction,
        Arc<AtomicBool>,
    );

    /// Installs a submission recorder on `ctx` and returns what it records.
    fn record_submissions(ctx: &DurableContext) -> Arc<std::sync::Mutex<Vec<Submitted>>> {
        let submissions: Arc<std::sync::Mutex<Vec<Submitted>>> = Arc::default();
        let recorded = Arc::clone(&submissions);
        ctx.set_submission_recorder(Arc::new(move |updates, flag| {
            let mut recorded = recorded.lock().unwrap();
            for u in updates {
                recorded.push((
                    u.name().unwrap_or("-").to_owned(),
                    u.id().to_owned(),
                    u.action().clone(),
                    Arc::clone(flag),
                ));
            }
        }));
        submissions
    }

    /// The outcome a test-7 case expects the combinator to write.
    #[derive(Clone, Copy, Debug)]
    enum Written {
        /// Both inputs succeed: `SUCCEED`.
        Succeed,
        /// Both inputs fail: `FAIL`.
        Fail,
        /// No inputs (`select_ok` and `race`): `FAIL`.
        Empty,
    }

    impl Written {
        fn action(self) -> aws_sdk_lambda::types::OperationAction {
            match self {
                Self::Succeed => aws_sdk_lambda::types::OperationAction::Succeed,
                Self::Fail | Self::Empty => aws_sdk_lambda::types::OperationAction::Fail,
            }
        }

        fn inputs(self) -> Vec<DurableFuture<i32>> {
            match self {
                Self::Succeed => vec![ready(1), ready(2)],
                Self::Fail => vec![failing_input(), failing_input()],
                Self::Empty => Vec::new(),
            }
        }
    }

    /// Every test-7 case. `join_all` records a failed input as an item and
    /// never writes `FAIL`, and only `select_ok` and `race` fail on no
    /// inputs.
    fn outcome_cases() -> Vec<(Outer, Written)> {
        let mut cases: Vec<(Outer, Written)> =
            OUTERS.iter().map(|o| (*o, Written::Succeed)).collect();
        cases.extend(
            [Outer::TryJoinAll, Outer::SelectOk, Outer::Race]
                .into_iter()
                .map(|o| (o, Written::Fail)),
        );
        cases.extend(
            [Outer::SelectOk, Outer::Race]
                .into_iter()
                .map(|o| (o, Written::Empty)),
        );
        cases
    }

    /// Runs one case of test 7. With `fail_start`, the first checkpoint
    /// call, which carries the combinator's `START`, fails.
    async fn outcome_case(outer: Outer, written: Written, fail_start: bool) -> Result<(), String> {
        use aws_sdk_lambda::types::OperationAction;
        let (ctx, client) = recording_ctx(CheckpointLog::empty());
        if fail_start {
            client.enqueue_checkpoint_response(crate::client::TestResponse::NonRetryableError(
                "START rejected".to_owned(),
            ));
        }
        let gate = Arc::new(tokio::sync::Notify::new());
        let held = Arc::clone(&gate);
        ctx.set_pre_lock_gate(Arc::new(move |updates| {
            updates
                .iter()
                .any(|u| u.action() == &OperationAction::Start && u.name() == Some("all"))
                .then(|| Arc::clone(&held))
        }));
        let submissions = record_submissions(&ctx);
        let entries: Arc<std::sync::Mutex<Vec<(String, OperationAction)>>> = Arc::default();
        let entered = Arc::clone(&entries);
        ctx.set_settler_entry_recorder(Arc::new(move |wire_id, action| {
            entered.lock().unwrap().push((wire_id.to_owned(), action));
        }));
        let wire = crate::engine::compute_wire_id_public("1");
        let action = written.action();
        let outcomes = |submissions: &[Submitted]| {
            submissions
                .iter()
                .filter(|(_, id, a, _)| {
                    *id == wire && matches!(a, OperationAction::Succeed | OperationAction::Fail)
                })
                .count()
        };
        let mut combinator = run_outer(&ctx, outer, written.inputs());

        let marker = within_two_seconds(|| {
            entries
                .lock()
                .unwrap()
                .contains(&(wire.clone(), action.clone()))
        });
        tokio::select! {
            () = &mut combinator => return Err("resolved before the settler-entry marker".to_owned()),
            reached = marker => if !reached {
                return Err("no settler-entry marker within two seconds".to_owned());
            },
        }
        let start_polled = submissions
            .lock()
            .unwrap()
            .iter()
            .find(|(name, _, a, _)| name == "all" && a == &OperationAction::Start)
            .is_some_and(|(_, _, _, flag)| flag.load(Ordering::SeqCst));
        if !start_polled {
            return Err("the retained START was not polled".to_owned());
        }
        if outcomes(&submissions.lock().unwrap()) != 0 {
            return Err("an outcome was submitted while START was held".to_owned());
        }
        gate.notify_one();
        if fail_start {
            if tokio::time::timeout(std::time::Duration::from_millis(500), &mut combinator)
                .await
                .is_ok()
            {
                return Err("resolved after a failed START".to_owned());
            }
            let fatal = ctx
                .coordinator()
                .fatal_error()
                .ok_or_else(|| "no fatal error".to_owned())?;
            let expected = crate::error::checkpoint_execution_failure_wire(
                &crate::client::ExecutionClientError::non_retryable("START rejected".to_owned()),
            );
            if fatal.error_type != "CheckpointUnrecoverableExecutionError"
                || Some(fatal.error_message.as_str()) != expected.error_message()
            {
                return Err(format!("fatal {fatal:?}"));
            }
            if outcomes(&submissions.lock().unwrap()) != 0 {
                return Err("an outcome was submitted after a failed START".to_owned());
            }
        } else {
            tokio::time::timeout(std::time::Duration::from_secs(2), &mut combinator)
                .await
                .map_err(|_| "did not resolve after the release".to_owned())?;
            if outcomes(&submissions.lock().unwrap()) != 1 {
                return Err("no outcome after the release".to_owned());
            }
            let recorded: Vec<OperationAction> = client
                .recorded_updates()
                .iter()
                .filter(|u| u.id() == wire)
                .map(|u| u.action().clone())
                .collect();
            if recorded != [OperationAction::Start, action] {
                return Err(format!("recorded {recorded:?}"));
            }
        }
        Ok(())
    }

    /// Test 7: a combinator reaches its outcome path while its `START` is
    /// held, and writes its `SUCCEED` or `FAIL` only after the `START`
    /// write completes, and neither when it fails.
    #[tokio::test]
    async fn combinator_outcome_waits_for_retained_start_write() {
        let mut failures = Vec::new();
        for (outer, written) in outcome_cases() {
            for fail_start in [false, true] {
                if let Err(err) = outcome_case(outer, written, fail_start).await {
                    failures.push(format!(
                        "{outer:?} {written:?} fail_start={fail_start}: {err}"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "failed cases: {failures:#?}");
    }

    /// A record at `positional` with `status`, of the record type
    /// `op_type` and sub-type `sub_type`, named `name`.
    fn record_of(
        positional: &str,
        status: CheckpointStatus,
        op_type: &str,
        sub_type: &str,
        name: &str,
    ) -> (String, CheckpointRecord) {
        let (wire_id, mut record) = succeeded_record(positional, "");
        record.result = None;
        record.status = status;
        record.op_type = Some(op_type.to_owned());
        record.sub_type = Some(sub_type.to_owned());
        record.op_name = Some(name.to_owned());
        (wire_id, record)
    }

    /// Rule 7.3's join row: a combinator's join gets the info its
    /// `on_operation_start` received, unchanged: no replay when fresh, a
    /// replay when it resumes an earlier invocation's STARTED record.
    #[tokio::test]
    async fn live_join_info_is_the_combinator_info() {
        use crate::plugin::recorder::{UfHooks, context};
        let join = |ctx: &DurableContext| {
            let a = ctx.step(|_| async { Ok(1_u32) }).name("a").future();
            let b = ctx.step(|_| async { Ok(2_u32) }).name("b").future();
            ctx.try_join_all([a, b]).name("all")
        };
        for (run, records) in [
            (1, Vec::new()),
            (
                2,
                vec![record_of(
                    "3",
                    CheckpointStatus::Started,
                    "Context",
                    COMBINATOR_SUB_TYPE,
                    "all",
                )],
            ),
        ] {
            let hooks = UfHooks::default();
            let (ctx, _client) = context(&hooks, records);
            assert_eq!(join(&ctx).await.unwrap(), [1, 2], "run {run}");
            let start = hooks.find("all", "start");
            assert_eq!(start.replay, run == 2, "run {run}");
            assert_eq!(start.id, crate::engine::compute_wire_id_public("3"));
            let join = hooks.user_function_of("all", true);
            for record in &join {
                assert_eq!(record.replay, run == 2, "run {run}: {}", record.hook);
                assert_eq!(
                    record.info, start.info,
                    "run {run}: {} gets the combinator's info",
                    record.hook
                );
            }
        }
    }

    /// `(type, message)` of an error.
    type TypeAndMessage = (Option<String>, Option<String>);

    /// The end hook's wire of the join `name`, and the error object of its
    /// FAIL, when the join failed.
    fn join_end(
        hooks: &crate::plugin::recorder::UfHooks,
        client: &InMemoryExecutionClient,
        name: &str,
    ) -> Option<(TypeAndMessage, TypeAndMessage)> {
        use crate::plugin::recorder::{type_and_message, update_error, update_of};
        let end = hooks.find(name, "uf_end");
        let wire = end.wire.as_ref()?;
        let fail = update_of(
            &client.recorded_updates(),
            name,
            &aws_sdk_lambda::types::OperationAction::Fail,
        )
        .expect("the join's FAIL");
        Some((type_and_message(wire), update_error(&fail)))
    }

    /// Rule 11's join rows. Each join fails on one failing input, except
    /// `join_all`, which ends `Ok`; empty `select_ok` and `race` fail with
    /// `EmptyInput`. Each failing join's FAIL stores its end hook's wire.
    /// A `race`'s started loser gets a start hook and no end hook.
    #[tokio::test]
    async fn combinator_join_outcomes() {
        use crate::plugin::recorder::{UfHooks, context, update_of};
        let ok = |ctx: &DurableContext| ctx.step(|_| async { Ok(1_u32) }).name("ok").future();
        let bad = |ctx: &DurableContext| {
            ctx.step(|_| async { Err::<u32, crate::BoxError>("boom".into()) })
                .name("bad")
                .retry_strategy(|_err, _attempt| crate::RetryDecision::Stop)
                .future()
        };
        // Each join ends with `expected` as its end hook's error type.
        let failing = |expected: &str, hooks: &UfHooks, client: &InMemoryExecutionClient| {
            hooks.user_function_of("j", true);
            let (end, stored) = join_end(hooks, client, "j").expect("an Err outcome");
            assert_eq!(end.0.as_deref(), Some(expected), "{end:?}");
            assert_eq!(
                end, stored,
                "{expected}: the FAIL stores the end hook's wire"
            );
        };

        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let inputs = [ok(&ctx), bad(&ctx)];
        drop(ctx.try_join_all(inputs).name("j").await.unwrap_err());
        failing("CombinatorError.JoinFailed.1", &hooks, &client);

        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let inputs = [ok(&ctx), bad(&ctx)];
        let outcomes = ctx.join_all(inputs).name("j").await.unwrap();
        assert!(outcomes.first().is_some_and(Result::is_ok));
        // The item's recorded wire is `wire_error_for(err,
        // "CombinatorError")` of the failed input: the step's recorded
        // type, which takes precedence over that fallback, and its message.
        // The returned item is decoded from that record.
        let item = outcomes
            .get(1)
            .and_then(|o| o.as_ref().err())
            .expect("input 1 failed");
        let item = crate::plugin::recorder::type_and_message(item.wire().expect("a wire"));
        assert_eq!(item, (Some("Error".to_owned()), Some("boom".to_owned())));
        hooks.user_function_of("j", true);
        assert_eq!(hooks.find("j", "uf_end").ok, Some(true));
        let succeed = update_of(
            &client.recorded_updates(),
            "j",
            &aws_sdk_lambda::types::OperationAction::Succeed,
        );
        assert!(succeed.is_some(), "join_all records a SUCCEED");

        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let inputs = [bad(&ctx)];
        drop(ctx.select_ok(inputs).name("j").await.unwrap_err());
        failing(ALL_FAILED_ERROR_TYPE, &hooks, &client);

        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let inputs = [bad(&ctx)];
        drop(ctx.race(inputs).name("j").await.unwrap_err());
        failing(FIRST_SETTLED_FAILED_ERROR_TYPE, &hooks, &client);

        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let none: [DurableFuture<u32>; 0] = [];
        drop(ctx.select_ok(none).name("j").await.unwrap_err());
        failing(EMPTY_INPUT_ERROR_TYPE, &hooks, &client);

        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let none: [DurableFuture<u32>; 0] = [];
        drop(ctx.race(none).name("j").await.unwrap_err());
        failing(EMPTY_INPUT_ERROR_TYPE, &hooks, &client);

        // A race whose started loser pends: the winner settles once the
        // loser's body runs.
        let hooks = UfHooks::default();
        let (ctx, _client) = context(&hooks, Vec::new());
        let started = Arc::new(tokio::sync::Notify::new());
        let waits = Arc::clone(&started);
        let winner = ctx
            .step(move |_| async move {
                waits.notified().await;
                Ok(1_u32)
            })
            .name("winner")
            .future();
        let loser = ctx
            .step(move |_| async move {
                started.notify_one();
                std::future::pending::<()>().await;
                Ok(2_u32)
            })
            .name("loser")
            .future();
        let value = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            ctx.race([winner, loser]).name("j").into_future(),
        )
        .await
        .expect("the race ends within 5 s")
        .unwrap();
        assert_eq!(value, 1);
        hooks.user_function_of("loser", false);
        hooks.user_function_of("winner", true);
        hooks.user_function_of("j", true);
        assert_eq!(hooks.find("j", "uf_end").ok, Some(true));
    }

    /// Rule 11: an empty `try_join_all` and `join_all` run their join as a
    /// user function, which ends `Ok`, and record a SUCCEED with `[]`.
    #[tokio::test]
    async fn empty_collection_joins_fire_join_hooks() {
        use crate::plugin::recorder::{UfHooks, context, update_of};
        let hooks = UfHooks::default();
        let (ctx, client) = context(&hooks, Vec::new());
        let none: [DurableFuture<u32>; 0] = [];
        assert_eq!(
            ctx.try_join_all(none).name("t").await.unwrap(),
            Vec::<u32>::new()
        );
        let none: [DurableFuture<u32>; 0] = [];
        assert!(ctx.join_all(none).name("j").await.unwrap().is_empty());
        let updates = client.recorded_updates();
        for name in ["t", "j"] {
            hooks.user_function_of(name, true);
            assert_eq!(hooks.find(name, "uf_end").ok, Some(true), "{name}");
            let succeed = update_of(
                &updates,
                name,
                &aws_sdk_lambda::types::OperationAction::Succeed,
            )
            .expect("the join's SUCCEED");
            assert_eq!(succeed.payload(), Some("[]"), "{name}");
        }
    }

    /// A root context over `log` with an in-memory client, the coalescer
    /// `window` and `plugins`.
    fn plugin_ctx(
        log: CheckpointLog,
        window: Option<std::time::Duration>,
        plugins: Arc<crate::plugin::InvocationPlugins>,
    ) -> (DurableContext, Arc<InMemoryExecutionClient>) {
        let client = Arc::new(InMemoryExecutionClient::new(Vec::new()));
        let ctx = DurableContext::new_root_with_client_and_defaults(
            "arn:aws:lambda:us-east-1:1:function:test".to_owned(),
            lambda_runtime::Context::default(),
            Arc::new(log),
            Arc::clone(&client) as Arc<dyn crate::client::ExecutionClient>,
            "token-1".to_owned(),
            window,
            0,
            Some(plugins),
        );
        (ctx, client)
    }

    /// A plugin whose user-function wrap enters a `plugin_user_function`
    /// span whose `trace_id` is the info's ID, as the README's Example 2
    /// does.
    #[derive(Clone)]
    struct TraceSpan;

    impl crate::plugin::Plugin for TraceSpan {
        fn wrap_user_function(&self, info: &crate::plugin::OperationInfo, next: &mut dyn FnMut()) {
            let span = tracing::info_span!("plugin_user_function", trace_id = %info.id());
            span.in_scope(next);
        }
    }

    impl crate::plugin::PluginFactory for TraceSpan {
        fn create_plugin(
            &self,
            _info: &crate::plugin::InvocationInfo,
        ) -> Result<Option<Box<dyn crate::plugin::Plugin>>, crate::BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    /// A record of the body `run_join` runs has the caller's span, then the
    /// plugin's span, as its stored ancestry; the body runs under the
    /// caller's active body; the plugin's trace ID is the join's
    /// `OperationInfo::id()`, and the innermost SDK span is the caller's
    /// namespace span with the caller's correlation operation ID.
    #[tokio::test]
    async fn run_join_body_record_is_under_the_plugin_span() {
        use tracing::Instrument as _;
        let (buffer, _guard) = crate::plugin::test_support::capture();
        let (ctx, _client) = plugin_ctx(
            CheckpointLog::empty(),
            None,
            crate::plugin::test_support::plugins_of(TraceSpan),
        );
        // A caller namespace with its own correlation operation ID, so a
        // join that replaced the caller's span would change the record.
        let caller = ctx.replay_span().in_scope(|| {
            crate::tracing_layer::scoped_execution_span(
                "arn:aws:lambda:us-east-1:1:function:test",
                "req-caller",
                None,
                "caller-op",
                Some("caller-name"),
                false,
            )
        });
        let joined = crate::context::run_body(
            crate::context::ActiveBody::Context("caller".to_owned()),
            run_join(
                &ctx,
                "7",
                "join-wire",
                Some("j"),
                Run::Live,
                async {
                    tracing::info!("join body");
                    Ok::<_, std::convert::Infallible>(crate::context::current_body())
                },
                crate::plugin::never_fails,
            ),
        )
        .instrument(caller)
        .await;
        let Ok(active) = joined;
        assert!(
            matches!(&active, crate::context::ActiveBody::Context(id) if id == "caller"),
            "{active:?}"
        );
        let output = String::from_utf8_lossy(&buffer.lock().unwrap()).to_string();
        let record = output
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .find(|v| v["message"] == "join body")
            .unwrap_or_else(|| panic!("no join body record in {output}"));
        let spans = record["spans"].as_array().expect("a span list");
        let names: Vec<&str> = spans.iter().filter_map(|s| s["name"].as_str()).collect();
        assert_eq!(
            names,
            [
                "durable_execution",
                "durable_execution",
                "plugin_user_function"
            ],
            "{record}"
        );
        assert_eq!(spans[2]["trace_id"], "join-wire", "{record}");
        // The innermost SDK span is the caller's, with its exact operation
        // ID, and no span carries the join's ID as a correlation field.
        assert_eq!(spans[1]["operationId"], "caller-op", "{record}");
        assert_eq!(spans[1]["operationName"], "caller-name", "{record}");
        assert_eq!(spans[1]["requestId"], "req-caller", "{record}");
        assert!(spans[0].get("operationId").is_none(), "{record}");
        assert!(spans[2].get("operationId").is_none(), "{record}");
        assert_eq!(record.get("operationId"), None, "{record}");
    }

    /// Test 13: on a `PENDING` or `READY` combinator record, the replay
    /// start hooks of resuming waits fire before the combinator's.
    #[tokio::test]
    async fn resumable_combinator_inputs_start_before_combinator_replay_hook() {
        let mut failures = Vec::new();
        for status in [CheckpointStatus::Pending, CheckpointStatus::Ready] {
            for outer in OUTERS {
                let hooks = crate::plugin::test_support::OpHooks::default();
                let log = CheckpointLog::from_records(vec![
                    record_of("1", CheckpointStatus::Started, "Wait", "Wait", "a"),
                    record_of("2", CheckpointStatus::Started, "Wait", "Wait", "b"),
                    record_of("3", status.clone(), "Context", COMBINATOR_SUB_TYPE, "all"),
                ]);
                let (ctx, _client) = plugin_ctx(log, None, hooks.plugins());
                let a = ctx
                    .wait(std::time::Duration::from_secs(1))
                    .name("a")
                    .future();
                let b = ctx
                    .wait(std::time::Duration::from_secs(2))
                    .name("b")
                    .future();
                let replays = || -> Vec<String> {
                    hooks
                        .records()
                        .into_iter()
                        .filter(|r| r.hook == "start" && r.replay)
                        .filter_map(|r| r.name)
                        .collect()
                };
                let combinator = run_outer(&ctx, outer, vec![a, b]);
                tokio::select! {
                    () = combinator => {}
                    _ = within_two_seconds(|| replays().len() >= 3) => {}
                }
                let replays = replays();
                if replays != ["a", "b", "all"] {
                    failures.push(format!("{outer:?} {status:?}: {replays:?}"));
                }
            }
        }
        assert!(failures.is_empty(), "failed cases: {failures:#?}");
    }

    /// Test 14: a combinator's own `.spawn()` starts its inputs, then
    /// itself, on its task, in input order.
    #[tokio::test(flavor = "current_thread")]
    async fn spawned_combinator_starts_inputs_on_its_task_in_order() {
        let mut failures = Vec::new();
        for outer in OUTERS {
            let (ctx, _client) = recording_ctx(CheckpointLog::empty());
            let submissions = record_submissions(&ctx);
            let starts = || -> Vec<String> {
                submissions
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(_, _, a, _)| a == &aws_sdk_lambda::types::OperationAction::Start)
                    .map(|(name, _, _, _)| name.clone())
                    .collect()
            };
            let a = ctx.step(|_| async { Ok(1_i32) }).name("a").future();
            let b = ctx.step(|_| async { Ok(2_i32) }).name("b").future();
            let handle = spawn_outer(&ctx, outer, vec![a, b]);
            let at_spawn = starts();
            for _ in 0..10 {
                if starts().len() >= 3 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            let after = starts();
            if !at_spawn.is_empty() || after != ["a", "b", "all"] {
                failures.push(format!("{outer:?}: at spawn {at_spawn:?}, after {after:?}"));
            }
            let _succeeded = handle.await;
        }
        assert!(failures.is_empty(), "failed cases: {failures:#?}");
    }

    /// The state of [`LogHooks`]'s blocking start hook.
    #[derive(Default)]
    struct HookGate {
        entered: bool,
        released: bool,
    }

    /// A plugin that logs `hook <name>` when `on_operation_start` begins,
    /// and `hook-return <name>` as its last statement. With a gate, the
    /// hook of `all` blocks in `block_in_place` between the two until the
    /// gate is released or five seconds pass.
    #[derive(Clone, Default)]
    struct LogHooks {
        log: Arc<std::sync::Mutex<Vec<String>>>,
        gate: Option<Arc<(std::sync::Mutex<HookGate>, std::sync::Condvar)>>,
    }

    impl LogHooks {
        fn entries(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }

        fn position(&self, entry: &str) -> Option<usize> {
            self.entries().iter().position(|e| e == entry)
        }

        fn release(&self) {
            if let Some(gate) = &self.gate {
                gate.0.lock().unwrap().released = true;
                gate.1.notify_all();
            }
        }

        fn entered(&self) -> bool {
            self.gate
                .as_ref()
                .is_some_and(|gate| gate.0.lock().unwrap().entered)
        }
    }

    impl crate::plugin::Plugin for LogHooks {
        fn on_operation_start(
            &self,
            info: &crate::plugin::OperationInfo,
        ) -> crate::plugin::HookWork {
            let name = info.name().unwrap_or("-");
            self.log.lock().unwrap().push(format!("hook {name}"));
            if name == "all"
                && let Some(gate) = &self.gate
            {
                tokio::task::block_in_place(|| {
                    let (state, condvar) = &**gate;
                    let mut state = state.lock().unwrap();
                    state.entered = true;
                    let _unused = condvar
                        .wait_timeout_while(state, std::time::Duration::from_secs(5), |s| {
                            !s.released
                        })
                        .unwrap();
                });
            }
            self.log.lock().unwrap().push(format!("hook-return {name}"));
            Ok(None)
        }
    }

    impl crate::plugin::PluginFactory for LogHooks {
        fn create_plugin(
            &self,
            _info: &crate::plugin::InvocationInfo,
        ) -> Result<Option<Box<dyn crate::plugin::Plugin>>, crate::BoxError> {
            Ok(Some(Box::new(self.clone())))
        }
    }

    /// Test 15: while the combinator's start hook dispatch blocks, the
    /// client receives no `START` of the combinator, on a multi-thread
    /// runtime, with checkpoint batching off and on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn combinator_queue_then_hook_blocks_start_send_until_hook_dispatch() {
        let mut failures = Vec::new();
        for outer in OUTERS {
            for window in [None, Some(std::time::Duration::ZERO)] {
                let hooks = LogHooks {
                    gate: Some(Arc::default()),
                    ..LogHooks::default()
                };
                let (ctx, client) = plugin_ctx(
                    CheckpointLog::empty(),
                    window,
                    crate::plugin::test_support::plugins_of(hooks.clone()),
                );
                let sent = || {
                    client.recorded_updates().iter().any(|u| {
                        u.action() == &aws_sdk_lambda::types::OperationAction::Start
                            && u.name() == Some("all")
                    })
                };
                let handle = spawn_outer(&ctx, outer, vec![ready(1_i32), ready(2_i32)]);
                let entered = within_two_seconds(|| hooks.entered()).await;
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                let sent_while_blocked = sent();
                hooks.release();
                let succeeded = tokio::time::timeout(std::time::Duration::from_secs(5), handle)
                    .await
                    .unwrap_or(false);
                let sent_after = sent();
                if !entered || sent_while_blocked || !sent_after || !succeeded {
                    failures.push(format!(
                        "{outer:?} window={window:?}: entered={entered} \
                         sent_while_blocked={sent_while_blocked} sent_after={sent_after} \
                         succeeded={succeeded}"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "failed cases: {failures:#?}");
    }

    /// A response-first input of test 16.
    #[derive(Clone, Copy, Debug)]
    enum ResponseFirst {
        Wait,
        AtMostOnceStep,
    }

    impl ResponseFirst {
        /// The input `name` on `ctx`. A step's body never returns, so no
        /// input settles before both hooks fire.
        fn input(self, ctx: &DurableContext, name: &str) -> DurableFuture<()> {
            match self {
                Self::Wait => ctx
                    .wait(std::time::Duration::from_secs(1))
                    .name(name)
                    .future(),
                Self::AtMostOnceStep => ctx
                    .step(|_| std::future::pending::<Result<(), crate::BoxError>>())
                    .name(name)
                    .semantics(crate::StepSemantics::AtMostOncePerRetry)
                    .future(),
            }
        }
    }

    /// Test 16: the remainders of response-first inputs are spawned after
    /// the combinator's start hook returns, so their hooks follow it.
    #[tokio::test(flavor = "current_thread")]
    async fn response_first_remainders_start_after_combinator_start_hook() {
        let mut failures = Vec::new();
        for outer in OUTERS {
            for kind in [ResponseFirst::Wait, ResponseFirst::AtMostOnceStep] {
                let hooks = LogHooks::default();
                let (ctx, _client) = plugin_ctx(
                    CheckpointLog::empty(),
                    None,
                    crate::plugin::test_support::plugins_of(hooks.clone()),
                );
                let log = Arc::clone(&hooks.log);
                REMAINDER_RECORDER.with(|recorder| {
                    *recorder.borrow_mut() = Some(Box::new(move |idx| {
                        let name = if idx == 0 { "a" } else { "b" };
                        log.lock().unwrap().push(format!("remainder {name}"));
                    }));
                });
                let inputs = vec![kind.input(&ctx, "a"), kind.input(&ctx, "b")];
                let combinator = run_outer(&ctx, outer, inputs);
                tokio::select! {
                    () = combinator => {}
                    _ = within_two_seconds(|| {
                        hooks.position("hook a").is_some() && hooks.position("hook b").is_some()
                    }) => {}
                }
                REMAINDER_RECORDER.with(|recorder| *recorder.borrow_mut() = None);
                let at = |entry: &str| hooks.position(entry);
                let ordered = ["hook all", "hook-return all", "remainder a", "remainder b"]
                    .windows(2)
                    .all(|pair| matches!((at(pair[0]), at(pair[1])), (Some(x), Some(y)) if x < y));
                let inputs_after = ["hook a", "hook b"].iter().all(|entry| {
                    matches!((at("hook-return all"), at(entry)), (Some(x), Some(y)) if x < y)
                });
                if !ordered || !inputs_after {
                    failures.push(format!("{outer:?} {kind:?}: {:?}", hooks.entries()));
                }
            }
        }
        assert!(failures.is_empty(), "failed cases: {failures:#?}");
    }
    /// An operation error with `message` as its record's message.
    fn loser_with_message(message: &str) -> crate::error::Source {
        Box::new(
            OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::new(
                CombinatorErrorKind::Internal,
                vec![format!("{message} source").into()],
            )))
            .with_wire(crate::error::WireError::new(Some("Error"), Some(message))),
        )
    }

    #[test]
    fn all_failed_message_names_each_loser_message() {
        let wire = all_failed_wire(&[loser_with_message("a"), loser_with_message("b")]);
        assert_eq!(
            wire.error_message(),
            Some("all 2 futures failed: [0] a; [1] b")
        );
    }

    /// A wire-less loser: the outer summary takes its recorded message,
    /// while its stored record keeps its flattened chain.
    #[test]
    fn all_failed_preserves_serialized_loser_message() {
        let inner = crate::error::WireError::new(Some("InnerType"), Some("a"))
            .with_error_data(Some("loser-data"))
            .with_stack_trace(vec!["loser-frame".to_owned()]);
        let loser = crate::error::ContextualError::source_from(
            "context",
            crate::error::recorded_step_error(inner),
        );
        let wire = all_failed_wire(&[loser]);
        assert_eq!(wire.error_message(), Some("all 1 futures failed: [0] a"));
        assert_eq!(
            wire.error_data(),
            Some(
                r#"[{"error_type":"CombinatorError","message":"context: operation error: step: execution failed: a","error_data":"loser-data","stack_trace":["loser-frame"]}]"#
            )
        );
    }

    /// A wire-less virtual child that propagates a failed step: the
    /// serialized rejected outcome carries the step's recorded message.
    #[tokio::test]
    async fn join_all_rejected_outcome_records_the_propagated_message() {
        let (ctx, _client) = recording_ctx(CheckpointLog::empty());
        let input = ctx
            .run_in_child_context(|child| async move {
                Ok(child
                    .step(|_| async { Err::<u32, crate::BoxError>("a".into()) })
                    .retry_strategy(|_err, _attempt| crate::RetryDecision::Stop)
                    .await?)
            })
            .virtual_context()
            .future();
        let outcomes = ctx.join_all([input]).await.expect("join_all succeeds");
        let [Err(rejected)] = outcomes.as_slice() else {
            panic!("expected one rejected outcome, got {outcomes:?}");
        };
        let wire = rejected
            .wire()
            .expect("a rejected outcome carries its record");
        assert_eq!(wire.error_type(), Some("ChildContextError"));
        assert_eq!(wire.error_message(), Some("a"));
    }
}

#[cfg(test)]
mod coordinator_tests {
    use crate::driver::InvocationOutcome;
    use crate::driver::test_support::{
        ScriptedClient, drive_handler, named, operation_like, scripted_ctx, wire_err,
    };
    use aws_sdk_lambda::types::{OperationAction, OperationStatus, OperationType};
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test(start_paused = true)]
    async fn dropped_race_loser_retires_its_registration() {
        let ctx = scripted_ctx(&Arc::new(ScriptedClient::empty()));
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let fast = ctx
                .step(|_| async {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok(1_u32)
                })
                .name("fast")
                .future();
            let slow = ctx.wait(Duration::from_secs(3600)).name("slow").future();
            let slow = crate::DurableFuture::from_async(async move {
                slow.await?;
                Ok(0_u32)
            });
            ctx.race([fast, slow]).await.map_err(wire_err)?;
            ctx.hold_activity(ctx.coordinator().tasks().wait_idle())
                .await;
            let waits = ctx.coordinator().live_waits();
            ctx.step(|_| async { Ok(2_u32) })
                .name("more")
                .await
                .map_err(wire_err)?;
            Ok(waits.to_string())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("0".to_owned()));
    }

    #[tokio::test(start_paused = true)]
    async fn race_with_every_input_waiting_suspends() {
        let client = Arc::new(ScriptedClient::empty());
        let ctx = scripted_ctx(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let one = ctx.wait(Duration::from_secs(1)).future();
            let two = ctx.wait(Duration::from_secs(2)).future();
            ctx.race([one, two]).await.map_err(wire_err)?;
            Ok("done".to_owned())
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Pending);
        let calls = client.calls();
        // The inputs' STARTs come before the race's, and nothing follows.
        let starts: Vec<OperationType> = calls
            .iter()
            .flat_map(|(_, updates)| updates.iter())
            .inspect(|u| assert_eq!(u.action(), &OperationAction::Start))
            .map(|u| u.r#type().clone())
            .collect();
        assert_eq!(
            starts,
            [
                OperationType::Wait,
                OperationType::Wait,
                OperationType::Context
            ],
            "the two wait STARTs, then the race START, and no later call"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn waiting_input_continues_on_sibling_response() {
        let client = Arc::new(ScriptedClient::new(|updates, history| {
            if named(updates, "s", &OperationAction::Succeed).is_none() {
                return Vec::new();
            }
            named(history, "w", &OperationAction::Start)
                .map(|start| {
                    operation_like(start, OperationStatus::Succeeded)
                        .build()
                        .expect("all required fields set")
                })
                .into_iter()
                .collect()
        }));
        let ctx = scripted_ctx(&client);
        let body_client = Arc::clone(&client);
        let outcome = drive_handler(ctx, None, |ctx| async move {
            let child = ctx
                .run_in_child_context(|child| async move {
                    child.wait(Duration::from_secs(1)).name("w").await?;
                    Ok(1_u32)
                })
                .name("child")
                .future();
            let step = ctx
                .step(move |_| async move {
                    for _ in 0..1000 {
                        if body_client.received("w", &OperationAction::Start) {
                            return Ok(2_u32);
                        }
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                    Err("w never started".into())
                })
                .name("s")
                .future();
            let values = ctx.try_join_all([child, step]).await.map_err(wire_err)?;
            Ok(format!("{values:?}"))
        })
        .await
        .expect("no panic");
        assert_eq!(outcome, InvocationOutcome::Complete("[1, 2]".to_owned()));
    }
}
