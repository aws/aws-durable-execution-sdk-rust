//! Conformance requirement 10-14: the operation end hook carries a
//! successful step's serialized result and a failed step's error.

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::{OperationStatus, RetryDecision, WireError};

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
    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if is_step(info) {
            self.emit(
                "operation-end",
                &[
                    ("op", info.id().into()),
                    (
                        "status",
                        info.status().map(OperationStatus::wire_str).into(),
                    ),
                    ("result", info.result().unwrap_or("NONE").into()),
                    (
                        "error",
                        info.error()
                            .and_then(WireError::error_message)
                            .unwrap_or("NONE")
                            .into(),
                    ),
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

/// Handler: step A returns `task-a`; step B fails without a retry, and the
/// handler propagates its error.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.step(|_| async { Ok("task-a".to_owned()) }).await?;
    ctx.step(|_| async { Err::<(), durable::BoxError>("boom".into()) })
        .retry_strategy(|_err, _attempt| RetryDecision::Stop)
        .await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
