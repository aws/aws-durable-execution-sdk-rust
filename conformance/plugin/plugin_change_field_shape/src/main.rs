//! Conformance requirement 10-22: each updated operation of the
//! operation-change hook carries every operation info field, in a
//! canonical dump, with the change info's own fields.

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookWork, InvocationInfo, OperationChangeInfo, Plugin, PluginFactory};

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

impl Plugin for Recorder {
    fn on_operation_change(&self, info: &OperationChangeInfo) -> HookWork {
        for item in info.updated_operations() {
            if item.sub_type() == Some("Step") {
                let mut fields = vec![
                    ("executionArn", info.execution_arn().into()),
                    (
                        "updatedOperationsCount",
                        info.updated_operations().len().into(),
                    ),
                    ("operationsCount", info.operations().len().into()),
                    (
                        "inFullMap",
                        info.operations().contains_key(item.id()).into(),
                    ),
                ];
                fields.extend(conformance::operation_dump(item));
                self.emit("operation-change", &fields);
            }
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
