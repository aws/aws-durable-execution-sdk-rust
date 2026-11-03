//! Conformance requirement 10-20: the operation start and end hooks of a
//! step carry every operation info field, in a canonical dump.

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};

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
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        if is_step(info) {
            self.emit("operation-start", &conformance::operation_dump(info));
        }
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if is_step(info) {
            self.emit("operation-end", &conformance::operation_dump(info));
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

/// Handler: one step named `greet` that returns `task-a`.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let result = ctx
        .step(|_| async { Ok("task-a".to_owned()) })
        .name("greet")
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
