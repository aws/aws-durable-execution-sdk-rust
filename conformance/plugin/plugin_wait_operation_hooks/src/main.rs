//! Conformance requirement 10-10: the plugin operation start and end hooks
//! fire for a wait, with the operation ID the history records and the
//! `WAIT` type.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::{OperationKind, OperationStatus};

/// Records the hooks of one invocation as `CONFPLUGIN`.
struct Recorder {
    execution_arn: String,
    /// The factory's copy of the first-invocation flag. This case writes
    /// no `first` key, so nothing reads it.
    _first: bool,
}

impl Recorder {
    /// Emits `fields` after `plugin` `CONFPLUGIN` and `hook` `hook`.
    fn emit(&self, hook: &str, fields: &[(&str, serde_json::Value)]) {
        let mut record = vec![("plugin", "CONFPLUGIN".into()), ("hook", hook.into())];
        record.extend(fields.iter().cloned());
        conformance::emit_record(&self.execution_arn, &record);
    }
}

/// Whether `info` is a wait's: the wait filter of the plugin requirements.
fn is_wait(info: &OperationInfo) -> bool {
    info.sub_type() == Some("Wait")
}

/// The operation's type token, `null` when the SDK reports no kind.
fn type_token(info: &OperationInfo) -> serde_json::Value {
    serde_json::json!(info.kind().map(OperationKind::wire_str))
}

impl Plugin for Recorder {
    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        if is_wait(info) {
            self.emit(
                "operation-start",
                &[("op", info.id().into()), ("type", type_token(info))],
            );
        }
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if is_wait(info) {
            self.emit(
                "operation-end",
                &[
                    ("op", info.id().into()),
                    ("type", type_token(info)),
                    (
                        "status",
                        serde_json::json!(info.status().map(OperationStatus::wire_str)),
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
            _first: info.is_first_invocation(),
        })))
    }
}

/// Handler: one 2-second wait.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(2)).await?;
    Ok(serde_json::Value::Null)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
