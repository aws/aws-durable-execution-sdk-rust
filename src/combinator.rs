//! Combinator operation execution engine.
//!
//! Implements `try_join_all`, `join_all`, `select_ok`, and `race`:
//! each is a durable operation that uses the child-context checkpoint
//! machinery to freeze the combined result. On replay the recorded
//! winner/collection is returned without re-running the constituent
//! operations.
//!
//! Semantics:
//! - `try_join_all`: await all; fail fast on the first error.
//! - `join_all`: await all; collect each outcome as `Settled` (never short-circuits).
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

use std::sync::Arc;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, CombinatorError, CombinatorErrorKind, OperationError,
    OperationErrorKind,
};
use crate::future::{DurableFuture, Settled};
use crate::task::{NoRuntime, TrackedJoinSet};

/// Wire sub-type for combinator operations (shared with child context
/// since combinators ARE child-context ops with a combinator-flavored closure).
pub(crate) const COMBINATOR_SUB_TYPE: &str = "RunInChildContext";

/// Wire `error_type` for a combinator failure whose specific kind carries
/// no dedicated wire discriminator (internal errors, and records written
/// before the discriminators below existed). Replay reconstructs these as
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
/// message, `error_data`, and `stack_trace` its live counterpart did. A
/// record without a decodable payload (written before the payload
/// existed, or malformed) falls back to one synthetic loser per input
/// reconstructed from the aggregate record, preserving the `failures()`
/// length.
const ALL_FAILED_ERROR_TYPE: &str = "CombinatorError.AllFailed";

/// Per-loser record persisted (as a JSON array) in an `AllFailed`
/// aggregate record's `error_data`.
///
/// Mirrors [`SerializedSettled::Rejected`]: the loser's complete wire
/// failure record (`error_type`, message, `error_data`, `stack_trace`),
/// so each loser's identity survives the checkpoint round-trip rather
/// than only the aggregate's flattened summary. Every field is optional
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
                    return Err(replay_combinator_failure(
                        wire,
                        &wire_id,
                        self.futures.len(),
                    ));
                }
                _ => {} // Started/Pending: fall through to re-execute
            }
        } else {
            // First invocation: checkpoint START.
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Live path: run all futures concurrently, fail-fast on first error.
        let count = self.futures.len();
        let mut results: Vec<Option<O>> = (0..count).map(|_| None).collect();
        let mut join_set = constituent_set(&self.ctx);

        for (idx, future) in self.futures.into_iter().enumerate() {
            spawn_constituent(&self.ctx, &mut join_set, idx, future)?;
        }

        let mut first_error: Option<(usize, OperationError)> = None;
        while let Some((_, joined)) = join_set.join_next_with_id().await {
            // A panicked input resumes its panic here.
            match joined.into_value() {
                (idx, Ok(value)) => {
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = Some(value);
                    }
                }
                (idx, Err(op_err)) => {
                    first_error = Some((idx, op_err));
                    // Abort remaining tasks (fail-fast + loser-drop).
                    join_set.abort_all();
                    break;
                }
            }
        }

        if let Some((failed_index, op_err)) = first_error {
            // Checkpoint FAIL. The wire `error_type` carries the JoinFailed
            // discriminator with the failed input's index appended, so
            // replay reconstructs the same variant and index; the message,
            // `error_data`, and `stack_trace` derive from the loser itself
            // (pass-through; a chain that recorded no frames records
            // none).
            let wire = crate::error::wire_error_with_type(
                &op_err,
                &format!("{JOIN_FAILED_ERROR_TYPE_PREFIX}{failed_index}"),
            );
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

        // Serialize and checkpoint SUCCESS.
        let serialized = serde_json::to_string(&collected)
            .map_err(|e| combinator_internal_error(&format!("serialization failed: {e}")))?;
        checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized).await?;
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

/// Serializable representation of a settled result for checkpointing.
///
/// Error-aware serdes: rejected results are serialized as
/// `{ status: "rejected", ... }` carrying the outcome's full wire error
/// record (`error_type`, `message`, `error_data`, `stack_trace`), so the
/// failure's identity survives the checkpoint round-trip rather than
/// only its flattened message (issue #41). Fulfilled results serialize
/// as `{ status: "fulfilled", value }`. The error fields beyond the
/// message are optional with defaults so records written before they
/// existed still replay.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(tag = "status")]
enum SerializedSettled<O> {
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
                    return Err(replay_combinator_failure(
                        wire,
                        &wire_id,
                        self.futures.len(),
                    ));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Live path: run all futures, never short-circuit.
        let count = self.futures.len();
        let mut settled: Vec<Option<Settled<O>>> = (0..count).map(|_| None).collect();
        let mut join_set = constituent_set(&self.ctx);

        for (idx, future) in self.futures.into_iter().enumerate() {
            spawn_constituent(&self.ctx, &mut join_set, idx, future)?;
        }

        while let Some((_, joined)) = join_set.join_next_with_id().await {
            // A panicked input resumes its panic here.
            match joined.into_value() {
                (idx, Ok(value)) => {
                    if let Some(slot) = settled.get_mut(idx) {
                        *slot = Some(Settled::Fulfilled(value));
                    }
                }
                (idx, Err(op_err)) => {
                    if let Some(slot) = settled.get_mut(idx) {
                        *slot = Some(Settled::Rejected(op_err));
                    }
                }
            }
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

        // Serialize with error-aware serdes. A rejected slot records its
        // full wire error (derived from the loser itself, so identity and
        // `error_data` pass through), and replay rebuilds a synthetic
        // source carrying those fields (issue #41).
        let serialized = {
            let serializable: Vec<SerializedSettled<&O>> = collected
                .iter()
                .map(|s| match s {
                    Settled::Fulfilled(v) => SerializedSettled::Fulfilled { value: v },
                    Settled::Rejected(err) => {
                        // Pass the outcome's own recorded wire record
                        // through verbatim when it carries one (the normal
                        // case: a failed constituent operation attaches its
                        // terminal wire record); derive one only for a
                        // wire-less error (e.g. an internal join failure).
                        let wire = err.wire().cloned().unwrap_or_else(|| {
                            crate::error::wire_error_for(err, COMBINATOR_ERROR_TYPE)
                        });
                        SerializedSettled::Rejected {
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

            serde_json::to_string(&serializable)
                .map_err(|e| combinator_internal_error(&format!("serialization failed: {e}")))?
        };
        checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized).await?;
        drop(collected);

        // Return the outcomes decoded from the payload just written, so the
        // first run returns what a replay of this record returns: every
        // rejected item is a `ChildContext(ChildFailed)` error whose
        // `wire()` carries the input's recorded failure, on both paths.
        replay_settled_success::<O>(Some(&serialized))
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
                    return Err(replay_combinator_failure(
                        wire,
                        &wire_id,
                        self.futures.len(),
                    ));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Empty input: there is no future that could succeed. Fail
        // explicitly with `EmptyInput` (matching `race`) rather than an
        // `AllFailed` carrying zero errors.
        if self.futures.is_empty() {
            let wire = crate::error::WireError::new(
                Some(EMPTY_INPUT_ERROR_TYPE),
                Some("select_ok called with no futures"),
            );
            checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
            // Attach the checkpointed context so the live error matches
            // what a replay of this record reconstructs.
            return Err(combinator_empty_input_error()
                .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire));
        }

        // Live path: race all futures, keep going until one succeeds or all
        // fail.
        let mut join_set = constituent_set(&self.ctx);

        for (idx, future) in self.futures.into_iter().enumerate() {
            spawn_constituent(&self.ctx, &mut join_set, idx, future)?;
        }

        let mut errors: Vec<(usize, crate::error::Source)> = Vec::new();

        while let Some((_, joined)) = join_set.join_next_with_id().await {
            // A panicked input resumes its panic here.
            match joined.into_value() {
                (_idx, Ok(value)) => {
                    // First success: cancel losers (drop via abort_all). A
                    // waiting loser does not block recording the winner.
                    join_set.abort_all();

                    // Checkpoint SUCCESS with the winner.
                    let serialized = serde_json::to_string(&value).map_err(|e| {
                        combinator_internal_error(&format!("serialization failed: {e}"))
                    })?;
                    checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized)
                        .await?;
                    drop(value);
                    // Return the winner decoded from the payload just
                    // written, so the first run returns what replay returns.
                    return replay_single_success::<O>(Some(&serialized));
                }
                (idx, Err(op_err)) => {
                    errors.push((idx, Box::new(op_err)));
                }
            }
        }

        // All failed: build the aggregate error keeping every loser (in
        // input order) as an error rather than a flattened string. The wire
        // `error_type` carries the AllFailed discriminator so replay
        // reconstructs the same variant, the recorded message names each
        // loser's flattened chain in input order, and the record's
        // `error_data` persists each loser's COMPLETE wire failure record
        // (a JSON array of [`SerializedLoser`]), so replay rebuilds one
        // loser per input with the same `error_type`, message,
        // `error_data`, and `stack_trace` the live failures carried.
        errors.sort_by_key(|(idx, _)| *idx);
        let losers: Vec<crate::error::Source> = errors.into_iter().map(|(_, err)| err).collect();
        let loser_summary = losers
            .iter()
            .enumerate()
            .map(|(idx, err)| format!("[{idx}] {}", crate::error::chain_string(&**err)))
            .collect::<Vec<_>>()
            .join("; ");
        // Pass each loser's own recorded wire record through verbatim when
        // it carries one (the normal case: a failed constituent operation
        // attaches its terminal wire record); derive one only for a
        // wire-less error (e.g. an internal join failure).
        let serialized_losers: Vec<SerializedLoser> = losers
            .iter()
            .map(|err| {
                let wire = err
                    .downcast_ref::<OperationError>()
                    .and_then(|op| op.wire().cloned())
                    .unwrap_or_else(|| crate::error::wire_error_for(&**err, COMBINATOR_ERROR_TYPE));
                SerializedLoser::from_wire(&wire)
            })
            .collect();
        let wire = crate::error::WireError::new(
            Some(ALL_FAILED_ERROR_TYPE),
            Some(format!(
                "all {} futures failed: {loser_summary}",
                losers.len()
            )),
        )
        // A payload of plain optional strings cannot fail JSON
        // serialization in practice; if it ever did, the record simply
        // carries no payload and replay falls back to synthetic losers.
        .with_error_data(serde_json::to_string(&serialized_losers).ok());

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
    /// Replay path: returns the frozen result.
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
                    return Err(replay_combinator_failure(
                        wire,
                        &wire_id,
                        self.futures.len(),
                    ));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Empty input: there is no future that could settle. Fail
        // explicitly with `EmptyInput` (matching `select_ok`).
        if self.futures.is_empty() {
            let wire = crate::error::WireError::new(
                Some(EMPTY_INPUT_ERROR_TYPE),
                Some("race called with no futures"),
            );
            checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
            // Attach the checkpointed context so the live error matches
            // what a replay of this record reconstructs.
            return Err(combinator_empty_input_error()
                .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire));
        }

        // Live path: race all futures, first settled wins. A losing input
        // that waits stays pending and never suspends the invocation once
        // the winner settled (issue #49): the winner is recorded, and the
        // losers are aborted.
        let mut join_set = constituent_set(&self.ctx);

        for (idx, future) in self.futures.into_iter().enumerate() {
            spawn_constituent(&self.ctx, &mut join_set, idx, future)?;
        }

        // Wait for the first SETTLED task. When every input waits, the race
        // stays pending here.
        #[expect(clippy::expect_used)]
        // reason: the input list is not empty here, and the set yields one
        // outcome per input it started (a refused start never yields), so
        // the join never reports an empty set.
        let (_, joined) = join_set
            .join_next_with_id()
            .await
            .expect("a non-empty race joins an input");
        // A panicked input resumes its panic here.
        let (_idx, settled_result) = joined.into_value();

        // Cancel all remaining losers.
        join_set.abort_all();

        match settled_result {
            Ok(value) => {
                // Winner is a success.
                let serialized = serde_json::to_string(&value).map_err(|e| {
                    combinator_internal_error(&format!("serialization failed: {e}"))
                })?;
                checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized).await?;
                drop(value);
                // Return the winner decoded from the payload just written,
                // so the first run returns what replay returns.
                replay_single_success::<O>(Some(&serialized))
            }
            Err(op_err) => {
                // Winner is a failure: race propagates it. The losing
                // error is preserved as the combinator error's source;
                // the wire `error_type` carries the discriminator so
                // replay reproduces the same variant, while the message,
                // `error_data`, and `stack_trace` derive from the loser
                // itself (pass-through; a chain that recorded no frames
                // records none).
                let wire =
                    crate::error::wire_error_with_type(&op_err, FIRST_SETTLED_FAILED_ERROR_TYPE);
                checkpoint_fail(&self.ctx, &wire_id, self.name.as_deref(), &wire).await?;
                Err(OperationError::from_kind(OperationErrorKind::Combinator(
                    CombinatorError::new(
                        CombinatorErrorKind::FirstSettledFailed,
                        vec![Box::new(op_err)],
                    ),
                ))
                .with_operation(&wire_id, CheckpointStatus::Failed.wire_str())
                .with_wire(wire))
            }
        }
    }
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
    join_set
        .spawn(ctx.task_ownership(), async move { (idx, future.await) })
        .map(drop)
        .map_err(NoRuntime::operation_error)
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
/// constituent [`DurableFuture`] was finalized (see `preflight_identity!` in
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
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43): combinator START: no user code ran for the
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
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43): combinator SUCCEED: the winning branch(es) ran,
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
    if let Some(parent_wire) = ctx.parent_wire_id() {
        builder = builder.parent_id(parent_wire);
    }

    #[expect(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    if let Err(err) = ctx.checkpoint_updates(vec![update]).await {
        // Audit (#43): combinator FAIL: branch bodies ran; the failed
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

/// Builds a combinator `FAIL` update carrying `wire` as its error: the
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
    if let Some(parent_wire) = ctx.parent_wire_id() {
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

    // Restore into Settled<O>: Rejected items become OperationErrors
    // whose synthetic sources carry the recorded wire fields
    // (`error_type`, message, `error_data`, `stack_trace`), not just the
    // flattened text (issue #41).
    let settled = serialized
        .into_iter()
        .map(|s| match s {
            SerializedSettled::Fulfilled { value } => Settled::Fulfilled(value),
            SerializedSettled::Rejected {
                message,
                error_type,
                error_data,
                stack_trace,
            } => {
                let wire = crate::error::WireError::new(error_type, Some(message))
                    .with_error_data(error_data)
                    .with_stack_trace(stack_trace);
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
/// as [`FIRST_SETTLED_FAILED_ERROR_TYPE`], [`EMPTY_INPUT_ERROR_TYPE`],
/// [`JOIN_FAILED_ERROR_TYPE_PREFIX`] (carrying the failed input's index
/// as its suffix), or [`ALL_FAILED_ERROR_TYPE`] reconstruct the same
/// [`CombinatorErrorKind`] variant the live path produced, so an error
/// observed live and the same error observed on replay are
/// indistinguishable. For `AllFailed`, the record's `error_data` carries
/// the SDK-authored losers payload (one [`SerializedLoser`] per input, in
/// input order), and each recorded loser rebuilds as its own
/// [`crate::error::ReplayedFailure`] with the same wire fields the live
/// loser carried. `input_len` is the combinator's input count, used only
/// as the `AllFailed` fallback when the payload is missing or malformed:
/// every input failed by definition, so the fallback carries one
/// synthetic source per input, keeping `CombinatorError::failures().len()`
/// replay-equivalent. Everything else (including records written before
/// these discriminators existed) reconstructs as `Internal` carrying the
/// recorded message.
fn replay_combinator_failure(
    wire: crate::error::WireError,
    wire_id: &str,
    input_len: usize,
) -> OperationError {
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
    } else if wire.error_type() == Some(ALL_FAILED_ERROR_TYPE) {
        // Prefer the recorded losers payload: one ReplayedFailure per
        // input, each carrying its loser's own recorded wire fields. The
        // payload is SDK-authored (see [`ALL_FAILED_ERROR_TYPE`]) and
        // decoded defensively: a record without one (written before the
        // payload existed, or malformed) falls back to one synthetic
        // loser per input rebuilt from the aggregate record, preserving
        // the `failures()` length.
        let recorded_losers = wire
            .error_data()
            .and_then(|data| serde_json::from_str::<Vec<SerializedLoser>>(data).ok())
            .filter(|losers| !losers.is_empty());
        let failures = match recorded_losers {
            Some(losers) => losers
                .into_iter()
                .map(|loser| crate::error::ReplayedFailure::source_from(loser.into_wire()))
                .collect(),
            None => (0..input_len.max(1))
                .map(|_| crate::error::ReplayedFailure::source_from(wire.clone()))
                .collect(),
        };
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
        let outcome = crate::driver::drive_invocation(
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

    // === replay of panic records an earlier release wrote ===

    #[tokio::test]
    async fn join_failed_record_from_a_panic_still_replays() {
        let (wire_id, record) = failed_record(
            "1",
            "CombinatorError.JoinFailed.1",
            r#"task join failed: task 7 panicked with message "boom""#,
        );
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

    #[tokio::test]
    async fn join_all_panic_slot_from_an_earlier_release_still_replays() {
        let payload = r#"[{"status":"fulfilled","value":1},{"status":"rejected","message":"task join failed: task 7 panicked with message \"boom\"","error_type":"CombinatorError"}]"#;
        let (wire_id, record) = succeeded_record("1", payload);
        let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let op_id = ctx.mint_id();
        let exec = JoinAllExecution::<i32> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let result = exec.execute().await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(matches!(&result[0], Settled::Fulfilled(1)));
        let Settled::Rejected(err) = &result[1] else {
            panic!("slot 1 must be rejected, got {:?}", result[1]);
        };
        let wire = err.wire().expect("the replayed slot carries its record");
        assert_eq!(wire.error_type(), Some("CombinatorError"));
        assert_eq!(
            wire.error_message(),
            Some(r#"task join failed: task 7 panicked with message "boom""#)
        );
    }

    #[tokio::test]
    async fn select_ok_panic_loser_from_an_earlier_release_still_replays() {
        let losers_payload = serde_json::to_string(&vec![
            SerializedLoser {
                error_type: Some("CombinatorError".to_owned()),
                message: Some("task join failed".to_owned()),
                error_data: None,
                stack_trace: Vec::new(),
            },
            SerializedLoser {
                error_type: Some("Error".to_owned()),
                message: Some("err-idx1".to_owned()),
                error_data: None,
                stack_trace: Vec::new(),
            },
        ])
        .expect("payload serializes");
        let (wire_id, record) = failed_record_with_data(
            "1",
            "CombinatorError.AllFailed",
            "all 2 futures failed: [0] task join failed; [1] err-idx1",
            Some(losers_payload),
        );
        let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let op_id = ctx.mint_id();
        let exec = SelectOkExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        let OperationErrorKind::Combinator(ce) = err.kind() else {
            panic!("expected Combinator, got: {:?}", err.kind());
        };
        assert!(matches!(ce.kind(), CombinatorErrorKind::AllFailed));
        let failures = ce.failures();
        assert_eq!(failures.len(), 2);
        let first = failures[0]
            .downcast_ref::<crate::error::ReplayedFailure>()
            .expect("replayed loser is a ReplayedFailure")
            .wire();
        assert_eq!(first.error_type(), Some("CombinatorError"));
        assert_eq!(first.error_message(), Some("task join failed"));
    }

    #[tokio::test]
    async fn race_panic_record_from_an_earlier_release_still_replays() {
        let message = r#"task join failed: task 7 panicked with message "boom""#;
        let (wire_id, record) = failed_record("1", "CombinatorError", message);
        let ctx = test_ctx(CheckpointLog::from_records(vec![(wire_id, record)]));
        let op_id = ctx.mint_id();
        let exec = RaceExecution::<String> {
            ctx,
            op_id,
            name: None,
            futures: vec![],
        };
        let err = exec.execute().await.unwrap_err();
        let OperationErrorKind::Combinator(ce) = err.kind() else {
            panic!("expected Combinator, got: {:?}", err.kind());
        };
        assert!(matches!(ce.kind(), CombinatorErrorKind::Internal));
        let failure = ce.failures()[0]
            .downcast_ref::<crate::error::ReplayedFailure>()
            .expect("replayed failure is a ReplayedFailure")
            .wire();
        assert_eq!(failure.error_message(), Some(message));
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
        let ctx = test_ctx_with_client(CheckpointLog::empty());
        let polled = crate::task::test_support::poll_once_without_runtime(move || run(ctx));
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

    /// An `AllFailed` record WITHOUT a losers payload (written before the
    /// payload existed) still replays as `AllFailed` with one synthetic
    /// loser per input, preserving the `failures()` length.
    #[tokio::test]
    async fn select_ok_all_failed_replay_without_payload_falls_back() {
        let (wire_id, record) = failed_record(
            "1",
            "CombinatorError.AllFailed",
            "all 3 futures failed: [0] a; [1] b; [2] c",
        );
        let log = CheckpointLog::from_records(vec![(wire_id, record)]);
        let ctx = test_ctx(log);

        let op_id = ctx.mint_id();
        let a: DurableFuture<String> = DurableFuture::from_async(async { panic!("never runs") });
        let b: DurableFuture<String> = DurableFuture::from_async(async { panic!("never runs") });
        let c: DurableFuture<String> = DurableFuture::from_async(async { panic!("never runs") });
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
            matches!(ce.kind(), CombinatorErrorKind::AllFailed),
            "expected AllFailed, got: {:?}",
            ce.kind()
        );
        assert_eq!(
            ce.failures().len(),
            3,
            "fallback keeps one synthetic loser per input"
        );
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

        // Must succeed with all branches fulfilled: none rejected due to ownership.
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
    fn render_settled(settled: &[Settled<Tagged>]) -> Vec<String> {
        settled
            .iter()
            .map(|s| match s {
                Settled::Fulfilled(t) => format!("ok:{}:{:?}", t.value, t.cached),
                Settled::Rejected(err) => format!(
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
    async fn join_all_live() -> (Vec<Settled<Tagged>>, String) {
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
            Settled::Fulfilled(t) => t.cached.is_none(),
            _ => true,
        }));
    }

    #[tokio::test]
    async fn join_all_live_rejection_is_child_failed() {
        let (live, payload) = join_all_live().await;

        let Some(Settled::Rejected(err)) = live.get(1) else {
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
        assert_eq!(calls.len(), 3, "the race START and the two wait STARTs");
        for (_, updates) in calls.iter().skip(1) {
            assert!(
                updates
                    .iter()
                    .all(|u| u.r#type() == &OperationType::Wait
                        && u.action() == &OperationAction::Start),
                "no checkpoint call after the two wait STARTs"
            );
        }
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
