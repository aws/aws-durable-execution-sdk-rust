//! Conformance requirement 10-11: the plugin operation end hook reports the
//! parent ID of a step nested in a child context, and no parent for the
//! child context itself.

use aws_durable_execution_sdk as durable;
use durable::OperationStatus;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};

/// Records the hooks of one invocation as `CONFPLUGIN`.
struct Recorder {
    execution_arn: String,
}

impl Plugin for Recorder {
    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", "CONFPLUGIN".into()),
                ("hook", "operation-end".into()),
                ("op", info.id().into()),
                ("parent", info.parent_id().unwrap_or("NONE").into()),
                (
                    "status",
                    info.status().map(OperationStatus::wire_str).into(),
                ),
            ],
        );
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

/// Handler: a child context holding one step that returns a greeting.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let name = event.as_str().unwrap_or("World").to_owned();
    let result = ctx
        .run_in_child_context(|child| async move {
            Ok(child
                .step(move |_| async move { Ok(format!("Hello, {name}!")) })
                .await?)
        })
        .await?;
    Ok(serde_json::Value::String(result))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
