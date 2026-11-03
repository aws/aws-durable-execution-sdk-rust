//! Conformance requirement 10-9: a plugin sees a wait completed
//! externally between invocations as an updated operation.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::OperationStatus;
use durable::plugin::{HookFuture, InvocationInfo, Plugin, PluginFactory};

/// Creates the plugin of each invocation.
struct Factory;

/// Records each wait the service updated before this invocation.
struct Recorder {
    execution_arn: String,
    /// The factory's copy of the first-invocation flag.
    first: bool,
}

impl Plugin for Recorder {
    fn on_invocation_start<'a>(&'a self, info: &'a InvocationInfo) -> HookFuture<'a> {
        for op in info.updated_operations() {
            if op.sub_type() != Some("Wait") {
                continue;
            }
            conformance::emit_record(
                &self.execution_arn,
                &[
                    ("plugin", "CONFPLUGIN".into()),
                    ("hook", "updated-on-invoke".into()),
                    ("op", op.id().into()),
                    (
                        "status",
                        serde_json::json!(op.status().map(OperationStatus::wire_str)),
                    ),
                    ("first", self.first.into()),
                ],
            );
        }
        Box::pin(std::future::ready(Ok(())))
    }
}

impl PluginFactory for Factory {
    fn create_plugin(
        &self,
        info: &InvocationInfo,
    ) -> Result<Option<Box<dyn Plugin>>, durable::BoxError> {
        Ok(Some(Box::new(Recorder {
            execution_arn: info.execution_arn().to_owned(),
            first: info.is_first_invocation(),
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
