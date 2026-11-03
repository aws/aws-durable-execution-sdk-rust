//! Conformance requirement 10-1: plugin invocation lifecycle hooks (start
//! and end on a single invocation).

use aws_durable_execution_sdk as durable;
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

/// Handler: one step that records its running line, then returns a
/// greeting.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let name = event.as_str().unwrap_or("World").to_owned();
    let execution_arn = ctx.execution_arn().to_owned();
    let result = ctx
        .step(move |_| async move {
            conformance::emit_record(
                &execution_arn,
                &[(
                    "message",
                    format!("Greeting step running for: {name}").into(),
                )],
            );
            Ok(format!("Hello, {name}!"))
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
