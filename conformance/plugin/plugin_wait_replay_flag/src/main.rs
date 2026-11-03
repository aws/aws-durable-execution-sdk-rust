//! Conformance requirement 10-18: a wait still pending during a replay
//! fires the operation start hook again with the replay flag, and a
//! terminal wait never does.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::{Branch, OperationKind, OperationStatus};

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
                &[
                    ("type", type_token(info)),
                    ("name", serde_json::json!(info.name())),
                    ("replay", info.is_replay().into()),
                    ("pending", info.end_timestamp().is_none().into()),
                ],
            );
        }
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if is_wait(info) {
            self.emit(
                "operation-end",
                &[
                    ("type", type_token(info)),
                    ("name", serde_json::json!(info.name())),
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

/// Handler: a parallel named `waits` whose branch 0 waits 2 s as `short`
/// and whose branch 1 waits 8 s as `long`.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<Vec<String>, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |child: durable::DurableContext| async move {
            child.wait(Duration::from_secs(2)).name("short").await?;
            Ok("short-done".to_owned())
        }),
        Branch::new("1", |child: durable::DurableContext| async move {
            child.wait(Duration::from_secs(8)).name("long").await?;
            Ok("long-done".to_owned())
        }),
    ];
    Ok(ctx
        .parallel(branches)
        .name("waits")
        .max_concurrency(2)
        .await?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
