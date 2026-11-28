//! Default step semantics: async steps, with one at-least-once key step.
//!
//! [`default_step_semantics`] sets the semantics of every step that does
//! not set its own. Here it is [`StepSemantics::AtLeastOnceAsync`]: a step
//! returns its result without waiting for the service to save it, so each
//! step has lower latency.
//!
//! An async step can run again on replay and return a different value. So a
//! step whose result becomes an idempotency key for a later step sets its
//! own [`StepSemantics::AtLeastOnce`]. That step waits until its result is
//! saved, and every later step reads the same key on every replay.
//!
//! The wait suspends the execution, so a second invocation replays the
//! steps before it and runs the step after it.
//!
//! [`default_step_semantics`]: aws_durable_execution_sdk::OptionsBuilder::default_step_semantics

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::StepSemantics;
use serde::{Deserialize, Serialize};

/// Handler output.
#[derive(Debug, Serialize, Deserialize)]
struct Output {
    /// The idempotency key every step used.
    key: String,
    /// The charge receipt.
    receipt: String,
}

/// Creates an idempotency key, waits, then charges with that key.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Output, durable::BoxError> {
    // The key feeds a later side effect, so this step waits until its
    // result is saved.
    let key = ctx
        .step(|_| async { Ok(format!("order-{}", std::process::id())) })
        .name("create-key")
        .semantics(StepSemantics::AtLeastOnce)
        .await?;

    ctx.wait(Duration::from_secs(2)).name("cool-down").await?;

    // This step uses the default, AtLeastOnceAsync.
    let charge_key = key.clone();
    let receipt = ctx
        .step(move |_| async move { Ok(format!("receipt-for-{charge_key}")) })
        .name("charge")
        .await?;

    Ok(Output { key, receipt })
}

/// The options of this function: async steps by default.
fn options() -> Result<durable::Options, durable::BoxError> {
    Ok(durable::Options::builder()
        .default_step_semantics(StepSemantics::AtLeastOnceAsync)
        .build()?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::init_default_subscriber()?;
    durable::run_with_options(handler, options()?).await
}

#[cfg(test)]
mod tests {
    use aws_durable_execution_sdk::test_util::LocalRunner;

    /// The wait splits the run into two invocations. The second replays
    /// the saved key, and the charge uses it.
    #[tokio::test]
    async fn charge_uses_the_saved_key() -> Result<(), aws_durable_execution_sdk::BoxError> {
        let result = LocalRunner::new()
            .options(super::options()?)
            .run(super::handler, serde_json::Value::Null)
            .await?;
        assert!(result.is_success());
        assert_eq!(result.invocation_count(), 2);
        let output = result.output().ok_or("no output")?;
        assert_eq!(output.receipt, format!("receipt-for-{}", output.key));
        let saved: Option<String> = result
            .operation("create-key")
            .ok_or("no create-key step")?
            .result_json()?;
        assert_eq!(saved.as_ref(), Some(&output.key));
        Ok(())
    }
}
