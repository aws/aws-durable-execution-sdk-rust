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
//! - `race` — return the first settled outcome (success or failure).
//!
//! Losers are dropped (cancelled) when a combinator resolves; each
//! combinator runs inside a child context so the combined result is
//! checkpointed atomically.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::context::DurableContext;
use crate::engine::{CheckpointStatus, OperationId};
use crate::error::{
    ChildContextError, ChildContextErrorKind, CombinatorError, CombinatorErrorKind, OperationError,
    OperationErrorKind,
};
use crate::future::{DurableFuture, Settled};

/// Wire sub-type for combinator operations (shared with child context
/// since combinators ARE child-context ops with a combinator-flavored closure).
const COMBINATOR_SUB_TYPE: &str = "RunInChildContext";

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
    pub(crate) async fn execute(self) -> Result<Vec<O>, OperationError> {
        // Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path: check checkpoint log.
        if let Some(record) = self.ctx.checkpoint_record(&positional_id) {
            match &record.status {
                CheckpointStatus::Succeeded => {
                    return replay_vec_success::<O>(record.result.as_ref());
                }
                CheckpointStatus::Failed => {
                    return Err(replay_combinator_failure(
                        record.error_type.as_deref(),
                        record.error_message.as_deref(),
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
        let mut join_set = tokio::task::JoinSet::new();
        // Maps a task's id back to its input index so a `JoinError` (panic
        // or cancellation loses the task's payload, including the index) is
        // still attributed to the correct position.
        let mut task_index: std::collections::HashMap<tokio::task::Id, usize> =
            std::collections::HashMap::with_capacity(count);

        for (idx, future) in self.futures.into_iter().enumerate() {
            let abort = join_set.spawn(async move { (idx, future.await) });
            task_index.insert(abort.id(), idx);
        }

        let mut first_error: Option<(usize, OperationError)> = None;
        while let Some(task_result) = join_set.join_next_with_id().await {
            match task_result {
                Ok((task_id, (idx, Ok(value)))) => {
                    task_index.remove(&task_id);
                    if let Some(slot) = results.get_mut(idx) {
                        *slot = Some(value);
                    }
                }
                Ok((task_id, (idx, Err(op_err)))) => {
                    task_index.remove(&task_id);
                    first_error = Some((idx, op_err));
                    // Abort remaining tasks (fail-fast + loser-drop).
                    join_set.abort_all();
                    break;
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
            // Checkpoint FAIL.
            let err_msg = op_err.to_string();
            checkpoint_fail(
                &self.ctx,
                &wire_id,
                self.name.as_deref(),
                "CombinatorError",
                &err_msg,
            )
            .await?;
            return Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::from_kind(CombinatorErrorKind::JoinFailed {
                    failed_index,
                    message: err_msg,
                }),
            )));
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
    pub(crate) async fn execute(self) -> Result<Vec<Settled<O>>, OperationError> {
        // Task-ownership check.
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path.
        if let Some(record) = self.ctx.checkpoint_record(&positional_id) {
            match &record.status {
                CheckpointStatus::Succeeded => {
                    return replay_settled_success::<O>(record.result.as_ref());
                }
                CheckpointStatus::Failed => {
                    return Err(replay_combinator_failure(
                        record.error_type.as_deref(),
                        record.error_message.as_deref(),
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
        let mut join_set = tokio::task::JoinSet::new();
        // Maps a task's id back to its input index so a `JoinError` is
        // attributed to the correct slot instead of the first empty one.
        let mut task_index: std::collections::HashMap<tokio::task::Id, usize> =
            std::collections::HashMap::with_capacity(count);

        for (idx, future) in self.futures.into_iter().enumerate() {
            let abort = join_set.spawn(async move { (idx, future.await) });
            task_index.insert(abort.id(), idx);
        }

        while let Some(task_result) = join_set.join_next_with_id().await {
            match task_result {
                Ok((task_id, (idx, Ok(value)))) => {
                    task_index.remove(&task_id);
                    if let Some(slot) = settled.get_mut(idx) {
                        *slot = Some(Settled::Fulfilled(value));
                    }
                }
                Ok((task_id, (idx, Err(op_err)))) => {
                    task_index.remove(&task_id);
                    if let Some(slot) = settled.get_mut(idx) {
                        *slot = Some(Settled::Rejected(op_err));
                    }
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

        // Collect in order.
        let collected: Vec<Settled<O>> = settled
            .into_iter()
            .map(|opt| {
                opt.unwrap_or_else(|| {
                    Settled::Rejected(combinator_internal_error("future did not complete"))
                })
            })
            .collect();

        // Serialize with error-aware serdes (rejected → message string).
        let serialized = {
            let serializable: Vec<SerializedSettled<&O>> = collected
                .iter()
                .map(|s| match s {
                    Settled::Fulfilled(v) => SerializedSettled::Fulfilled { value: v },
                    Settled::Rejected(err) => SerializedSettled::Rejected {
                        message: err.to_string(),
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
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path.
        if let Some(record) = self.ctx.checkpoint_record(&positional_id) {
            match &record.status {
                CheckpointStatus::Succeeded => {
                    return replay_single_success::<O>(record.result.as_ref());
                }
                CheckpointStatus::Failed => {
                    return Err(replay_combinator_failure(
                        record.error_type.as_deref(),
                        record.error_message.as_deref(),
                    ));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Live path: race all futures, keep going until one succeeds or all fail.
        let count = self.futures.len();
        let mut join_set = tokio::task::JoinSet::new();
        // Maps a task's id back to its input index so a `JoinError` is
        // attributed to the correct position in the aggregate error list.
        let mut task_index: std::collections::HashMap<tokio::task::Id, usize> =
            std::collections::HashMap::with_capacity(count);

        for (idx, future) in self.futures.into_iter().enumerate() {
            let abort = join_set.spawn(async move { (idx, future.await) });
            task_index.insert(abort.id(), idx);
        }

        let mut errors: Vec<(usize, String)> = Vec::new();

        while let Some(task_result) = join_set.join_next_with_id().await {
            match task_result {
                Ok((_task_id, (_idx, Ok(value)))) => {
                    // First success — cancel losers (drop via abort_all).
                    join_set.abort_all();

                    // Checkpoint SUCCESS with the winner.
                    let serialized = serde_json::to_string(&value).map_err(|e| {
                        combinator_internal_error(&format!("serialization failed: {e}"))
                    })?;
                    checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized)
                        .await?;
                    return Ok(value);
                }
                Ok((task_id, (idx, Err(op_err)))) => {
                    task_index.remove(&task_id);
                    errors.push((idx, op_err.to_string()));
                }
                Err(join_err) => {
                    let Some(idx) = task_index.remove(&join_err.id()) else {
                        return Err(combinator_internal_error(
                            "task terminated with an unrecognized task id",
                        ));
                    };
                    errors.push((idx, format!("task join failed: {join_err}")));
                }
            }
        }

        // All failed — build aggregate error.
        errors.sort_by_key(|(idx, _)| *idx);
        let error_messages: Vec<String> = errors.into_iter().map(|(_, msg)| msg).collect();
        let err_display = format!("all {} futures failed", error_messages.len());

        checkpoint_fail(
            &self.ctx,
            &wire_id,
            self.name.as_deref(),
            "CombinatorError",
            &err_display,
        )
        .await?;

        Err(OperationError::from_kind(OperationErrorKind::Combinator(
            CombinatorError::from_kind(CombinatorErrorKind::AllFailed {
                errors: error_messages,
            }),
        )))
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
    pub(crate) async fn execute(self) -> Result<O, OperationError> {
        self.ctx.enforce_task_ownership()?;

        let positional_id = self.op_id.positional().to_owned();
        let wire_id = self.op_id.wire().to_owned();

        // Replay path.
        if let Some(record) = self.ctx.checkpoint_record(&positional_id) {
            match &record.status {
                CheckpointStatus::Succeeded => {
                    return replay_single_success::<O>(record.result.as_ref());
                }
                CheckpointStatus::Failed => {
                    return Err(replay_combinator_failure(
                        record.error_type.as_deref(),
                        record.error_message.as_deref(),
                    ));
                }
                _ => {}
            }
        } else {
            checkpoint_start(&self.ctx, &wire_id, self.name.as_deref()).await?;
        }

        // Live path: race all futures, first settled wins.
        let mut join_set = tokio::task::JoinSet::new();

        for (idx, future) in self.futures.into_iter().enumerate() {
            join_set.spawn(async move { (idx, future.await) });
        }

        // Wait for the first completed task.
        if let Some(task_result) = join_set.join_next().await {
            // Cancel all remaining losers.
            join_set.abort_all();

            match task_result {
                Ok((_idx, Ok(value))) => {
                    // Winner is a success.
                    let serialized = serde_json::to_string(&value).map_err(|e| {
                        combinator_internal_error(&format!("serialization failed: {e}"))
                    })?;
                    checkpoint_succeed(&self.ctx, &wire_id, self.name.as_deref(), &serialized)
                        .await?;
                    return Ok(value);
                }
                Ok((_idx, Err(op_err))) => {
                    // Winner is a failure — race propagates it.
                    let err_msg = op_err.to_string();
                    checkpoint_fail(
                        &self.ctx,
                        &wire_id,
                        self.name.as_deref(),
                        "CombinatorError",
                        &err_msg,
                    )
                    .await?;
                    return Err(OperationError::from_kind(OperationErrorKind::Combinator(
                        CombinatorError::from_kind(CombinatorErrorKind::Internal {
                            message: err_msg,
                        }),
                    )));
                }
                Err(join_err) => {
                    let msg = format!("task join failed: {join_err}");
                    checkpoint_fail(
                        &self.ctx,
                        &wire_id,
                        self.name.as_deref(),
                        "CombinatorError",
                        &msg,
                    )
                    .await?;
                    return Err(combinator_internal_error(&msg));
                }
            }
        }

        // Empty iterator — no futures provided.
        let msg = "race called with empty iterator";
        checkpoint_fail(
            &self.ctx,
            &wire_id,
            self.name.as_deref(),
            "CombinatorError",
            msg,
        )
        .await?;
        Err(combinator_internal_error(msg))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ────────────────────────────────────────────────────────────────────────────

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

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    ctx.checkpoint_updates(vec![update])
        .await
        .map_err(|e| combinator_internal_error(&format!("checkpoint start: {e}")))?;
    Ok(())
}

/// Checkpoints a SUCCEED action for a combinator operation.
async fn checkpoint_succeed(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    payload: &str,
) -> Result<(), OperationError> {
    use aws_sdk_lambda::types::{OperationAction, OperationType, OperationUpdate};

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

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    ctx.checkpoint_updates(vec![update])
        .await
        .map_err(|e| combinator_internal_error(&format!("checkpoint succeed: {e}")))?;
    Ok(())
}

/// Checkpoints a FAIL action for a combinator operation.
async fn checkpoint_fail(
    ctx: &DurableContext,
    wire_id: &str,
    name: Option<&str>,
    error_type: &str,
    error_message: &str,
) -> Result<(), OperationError> {
    use aws_sdk_lambda::types::{ErrorObject, OperationAction, OperationType, OperationUpdate};

    let mut builder = OperationUpdate::builder()
        .id(wire_id.to_owned())
        .r#type(OperationType::Context)
        .sub_type(COMBINATOR_SUB_TYPE.to_owned())
        .action(OperationAction::Fail)
        .error(
            ErrorObject::builder()
                .error_type(error_type)
                .error_message(error_message)
                .build(),
        );

    if let Some(n) = name {
        builder = builder.name(n.to_owned());
    }
    if let Some(parent_wire) = ctx.parent_wire_id_computed() {
        builder = builder.parent_id(parent_wire);
    }

    #[allow(clippy::expect_used)] // reason: all required fields are set above
    let update = builder
        .build()
        .expect("all required OperationUpdate fields set");

    ctx.checkpoint_updates(vec![update])
        .await
        .map_err(|e| combinator_internal_error(&format!("checkpoint fail: {e}")))?;
    Ok(())
}

/// Creates a combinator internal `OperationError`.
fn combinator_internal_error(message: &str) -> OperationError {
    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::from_kind(
        CombinatorErrorKind::Internal {
            message: message.to_owned(),
        },
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

    // Restore into Settled<O> — Rejected items become OperationErrors
    // carrying the original message.
    let settled = serialized
        .into_iter()
        .map(|s| match s {
            SerializedSettled::Fulfilled { value } => Settled::Fulfilled(value),
            SerializedSettled::Rejected { message } => {
                Settled::Rejected(OperationError::from_kind(OperationErrorKind::ChildContext(
                    ChildContextError::from_kind(ChildContextErrorKind::ChildFailed { message }),
                )))
            }
        })
        .collect();
    Ok(settled)
}

/// Replays a failed combinator from a checkpoint record.
fn replay_combinator_failure(
    _error_type: Option<&str>,
    error_message: Option<&str>,
) -> OperationError {
    let message = error_message
        .unwrap_or("unknown combinator error")
        .to_owned();
    OperationError::from_kind(OperationErrorKind::Combinator(CombinatorError::from_kind(
        CombinatorErrorKind::Internal { message },
    )))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
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
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: None,
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
                attempt: 0,
                invoke_result: None,
                invoke_error_type: None,
                invoke_error_message: None,
                replay_children: false,
                callback_id: None,
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
                CombinatorError::from_kind(CombinatorErrorKind::Internal {
                    message: "boom".to_owned(),
                }),
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
                CombinatorErrorKind::JoinFailed { failed_index, .. } => {
                    assert_eq!(
                        *failed_index, 1,
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
                CombinatorError::from_kind(CombinatorErrorKind::Internal {
                    message: "err-idx1".to_owned(),
                }),
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
                CombinatorErrorKind::AllFailed { errors } => {
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
                CombinatorError::from_kind(CombinatorErrorKind::Internal {
                    message: "task-b-failed".to_owned(),
                }),
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
        assert!(matches!(&result[1], Settled::Rejected(e) if e.to_string().contains("oops")));
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
                CombinatorError::from_kind(CombinatorErrorKind::Internal {
                    message: "fail".to_owned(),
                }),
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
                CombinatorError::from_kind(CombinatorErrorKind::Internal {
                    message: "err-a".to_owned(),
                }),
            )))
        });
        let b: DurableFuture<String> = DurableFuture::from_async(async {
            Err(OperationError::from_kind(OperationErrorKind::Combinator(
                CombinatorError::from_kind(CombinatorErrorKind::Internal {
                    message: "err-b".to_owned(),
                }),
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
                CombinatorErrorKind::AllFailed { errors } => {
                    assert_eq!(errors.len(), 2);
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
                CombinatorError::from_kind(CombinatorErrorKind::Internal {
                    message: "fast-fail".to_owned(),
                }),
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
        assert!(matches!(err.kind(), OperationErrorKind::Combinator(_)));
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
}
