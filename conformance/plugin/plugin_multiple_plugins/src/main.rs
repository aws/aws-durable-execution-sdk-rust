//! Conformance requirement 10-5: multiple registered plugins all receive
//! lifecycle hooks.

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookFuture, InvocationEndInfo, InvocationInfo, Plugin, PluginFactory};

/// Creates the plugin of each invocation, which records as `label`.
struct Factory {
    label: &'static str,
}

/// Records the invocation hooks of one invocation.
struct Recorder {
    label: &'static str,
    execution_arn: String,
    /// The factory's copy of the first-invocation flag. This case writes
    /// no `first` key, so nothing reads it.
    _first: bool,
}

impl Plugin for Recorder {
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", self.label.into()),
                ("hook", "invocation-start".into()),
            ],
        );
        Box::pin(std::future::ready(Ok(())))
    }

    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", self.label.into()),
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
            label: self.label,
            execution_arn: info.execution_arn().to_owned(),
            _first: info.is_first_invocation(),
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
    let options = durable::Options::builder()
        .plugin(Factory {
            label: "CONFPLUGIN-A",
        })
        .plugin(Factory {
            label: "CONFPLUGIN-B",
        })
        .build()?;
    durable::run_with_options(handler, options).await
}
