//! Keep checkpoints small for large results with [`FileSystemSerdes`] overflow.
//!
//! Checkpoint payloads have a size limit (~256 KB). A result that would exceed
//! it inline fails the checkpoint. [`FileSystemSerdes`] in
//! [`Overflow`](aws_durable_execution_sdk_rust::serdes::FileSystemSerdesMode::Overflow)
//! mode stores small values inline but spills anything over the threshold to a
//! file, checkpointing only a lightweight file-pointer envelope — so a large
//! result round-trips where an inline one would fail.
//!
//! # Durable storage requirement
//!
//! Production durable functions that suspend and replay MUST point `base_path`
//! at a durable, shared filesystem (Amazon EFS or S3 Files mounted to Lambda),
//! never Lambda's per-environment `/tmp`. This example completes in a single
//! invocation without suspending, so `/tmp` is adequate purely for the
//! demonstration; do not copy that choice into replayable code.
//!
//! [`FileSystemSerdes`]: aws_durable_execution_sdk_rust::serdes::FileSystemSerdes

use aws_durable_execution_sdk_rust as durable;
use durable::serdes::{
    FileSystemPathEncoding, FileSystemSerdes, FileSystemSerdesConfig, FileSystemSerdesMode,
};

/// Produces a ~300 KB result through the overflow serdes and returns its size.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<usize, durable::BoxError> {
    let serdes = FileSystemSerdes::with_config(
        "/tmp/durable-serdes",
        FileSystemSerdesConfig::builder()
            .storage_mode(FileSystemSerdesMode::Overflow)
            .path_encoding(FileSystemPathEncoding::Hash)
            .build(),
    );
    // ~300 KB: over the inline checkpoint limit, so overflow to a file engages.
    // An inline checkpoint of this size would fail; here the execution succeeds.
    let big = ctx
        .step(|_| async { Ok("x".repeat(300 * 1024)) })
        .name("produce-large")
        .serdes(serdes)
        .await?;
    Ok(big.len())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
