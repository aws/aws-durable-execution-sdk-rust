//! Conformance requirement 10-8: the operation-change hook reports a
//! step's terminal change once, with the step in the full operation map.

use aws_durable_execution_sdk as durable;
use durable::OperationStatus;
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
                self.emit(
                    "operation-change",
                    &[
                        ("op", item.id().into()),
                        (
                            "status",
                            item.status().map(OperationStatus::wire_str).into(),
                        ),
                        (
                            "in_full_map",
                            info.operations().contains_key(item.id()).into(),
                        ),
                    ],
                );
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

/// Handler: one step that returns a greeting.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let name = event.as_str().unwrap_or("World").to_owned();
    let result = ctx
        .step(move |_| async move { Ok(format!("Hello, {name}!")) })
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
