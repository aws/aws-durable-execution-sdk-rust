//! Conformance requirement 10-7: the plugin invocation-end hook receives
//! the FAILED status when the execution fails.

use aws_durable_execution_sdk as durable;
use durable::RetryDecision;
use durable::plugin::{HookFuture, InvocationEndInfo, InvocationInfo, Plugin, PluginFactory};

/// Creates the plugin of each invocation.
struct Factory;

/// Records the invocation hooks of one invocation as `CONFPLUGIN`.
struct Recorder {
    execution_arn: String,
    /// The factory's copy of the first-invocation flag.
    first: bool,
}

impl Plugin for Recorder {
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", "CONFPLUGIN".into()),
                ("hook", "invocation-start".into()),
                ("first", self.first.into()),
            ],
        );
        Box::pin(std::future::ready(Ok(())))
    }

    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", "CONFPLUGIN".into()),
                ("hook", "invocation-end".into()),
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

/// Handler: one step that always fails, without retries.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.step(|_| async { Err::<(), durable::BoxError>("step failed".into()) })
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
