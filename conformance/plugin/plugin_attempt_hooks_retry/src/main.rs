//! Conformance requirement 10-3: the plugin attempt hooks fire for each
//! step attempt, with its number and outcome.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::RetryDecision;
use durable::plugin::{
    AttemptEndInfo, AttemptInfo, HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory,
};

/// Records the hooks of one invocation as `CONFPLUGIN`.
struct Recorder {
    execution_arn: String,
}

impl Recorder {
    /// Emits `fields` after `plugin` `CONFPLUGIN` and `hook` `hook`.
    fn emit(&self, hook: &str, fields: &[(&str, serde_json::Value)]) {
        let mut record = vec![("plugin", "CONFPLUGIN".into()), ("hook", hook.into())];
        record.extend(fields.iter().cloned());
        conformance::emit_record(&self.execution_arn, &record);
    }
}

/// Whether `info` is a step's: the step filter of the plugin requirements.
fn is_step(info: &OperationInfo) -> bool {
    info.sub_type() == Some("Step")
}

impl Plugin for Recorder {
    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        if is_step(info.operation()) {
            self.emit(
                "attempt-start",
                &[
                    ("n", info.attempt().into()),
                    ("op", info.operation().id().into()),
                ],
            );
        }
        Ok(None)
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        if is_step(info.operation()) {
            self.emit(
                "attempt-end",
                &[
                    ("n", info.attempt().into()),
                    ("outcome", info.outcome().wire_str().into()),
                    ("op", info.operation().id().into()),
                ],
            );
        }
        Ok(None)
    }
}

/// Creates the plugin of each invocation.
struct Factory;

impl PluginFactory for Factory {
    fn create_plugin(
        &self,
        info: &InvocationInfo,
    ) -> Result<Option<Box<dyn Plugin>>, durable::BoxError> {
        Ok(Some(Box::new(Recorder {
            execution_arn: info.execution_arn().to_owned(),
        })))
    }
}

/// Retries an attempt below 3 after 1 s, and stops at attempt 3.
fn retry_below_3(_err: &durable::StepError, attempt: u32) -> RetryDecision {
    if attempt < 3 {
        RetryDecision::retry(Duration::from_secs(1))
    } else {
        RetryDecision::Stop
    }
}

/// Handler: one step that fails its first attempt and succeeds later.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result = ctx
        .step(|step_ctx| async move {
            let n = step_ctx.attempt();
            if n == 1 {
                return Err(format!("Attempt {n} failed").into());
            }
            Ok("Operation succeeded".to_owned())
        })
        .retry_strategy(retry_below_3)
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
