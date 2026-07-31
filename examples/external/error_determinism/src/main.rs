//! The determinism contract in practice.
//!
//! Two rules make replay safe, and this example shows both. First,
//! nondeterminism *inside* a step body is fine: the step's result is
//! checkpointed once and replayed, so a value computed from the clock is frozen
//! after the first run and never recomputed. Second, the *order* in which
//! durable operations are created must be deterministic: these steps are always
//! created in the same sequence, so each is paired with the same recorded
//! result on every replay.

use aws_durable_execution_sdk_rust as durable;

/// Mints a clock-derived token in a step, then runs two ordered steps.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    // Nondeterministic input, but captured inside the step: checkpointed once,
    // replayed thereafter — so replay sees the same token.
    let token = ctx
        .step(|_| async {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos());
            Ok(format!("token-{nanos}"))
        })
        .name("mint-token")
        .await?;

    // Deterministic creation order: these always mint the same operation ids.
    let first = ctx.step(|_| async { Ok(1) }).name("first").await?;
    let second = ctx.step(|_| async { Ok(2) }).name("second").await?;

    Ok(format!("{token}:sum={}", first + second))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
