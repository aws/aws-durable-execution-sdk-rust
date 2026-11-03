//! Conformance requirement 10-16: the invocation-end hook runs for every
//! invocation, non-terminal on suspension and terminal at completion.

use std::time::Duration;

use aws_durable_execution_sdk as durable;
use durable::InvocationStatus;
use durable::plugin::{HookFuture, InvocationEndInfo, InvocationInfo, Plugin, PluginFactory};

/// Creates the plugin of each invocation.
struct Factory;

/// Records the invocation hooks of one invocation as `CONFPLUGIN`, with
/// the invocation's first-invocation flag on both records.
struct Recorder {
    execution_arn: String,
    first: bool,
}

impl Plugin for Recorder {
    fn on_invocation_start<'a>(&'a self, info: &'a InvocationInfo) -> HookFuture<'a> {
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", "CONFPLUGIN".into()),
                ("hook", "invocation-start".into()),
                ("first", info.is_first_invocation().into()),
            ],
        );
        Box::pin(std::future::ready(Ok(())))
    }

    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        let terminal = matches!(
            info.status(),
            InvocationStatus::Succeeded | InvocationStatus::Failed
        );
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", "CONFPLUGIN".into()),
                ("hook", "invocation-end".into()),
                ("first", self.first.into()),
                ("terminal", terminal.into()),
                ("status", info.status().wire_str().into()),
            ],
        );
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
