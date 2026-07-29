//! A child that returns a large result.
//!
//! A child context's result is checkpointed and replayed like any operation
//! result, including sizeable payloads. This example returns a result on the
//! order of 128 KB from a child and reports its length, showing that a large
//! value round-trips through a child without special handling in user code.
//!
//! Very large payloads that would exceed the inline checkpoint size are handled
//! by a serdes that offloads the bytes to an external store; see the
//! large-payload example in the external/serdes family for that pattern.

use aws_durable_execution_sdk_rust as durable;

/// Runs a child that produces a large (~128 KB) result and returns its length.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<usize, durable::BoxError> {
    let payload = ctx
        .run_in_child_context(|child| async move {
            let large = child
                .step(|_| async {
                    // ~128 KB: a sizeable result that still checkpoints inline.
                    Ok("x".repeat(128 * 1024))
                })
                .name("produce-large")
                .await?;
            Ok(large)
        })
        .name("large-child")
        .await?;
    Ok(payload.len())
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
