//! Conformance requirement 10-4: a plugin whose every hook fails changes
//! neither the execution's result nor its history.

use aws_durable_execution_sdk as durable;
use durable::plugin::{
    AttemptEndInfo, AttemptInfo, HookFuture, HookWork, InvocationEndInfo, InvocationInfo,
    OperationInfo, Plugin, PluginFactory,
};

/// Records each hook as `CONFPLUGIN-FAULTY`, then fails it.
struct Faulty {
    execution_arn: String,
}

impl Faulty {
    /// Emits the record of `hook`.
    fn emit(&self, hook: &str) {
        conformance::emit_record(
            &self.execution_arn,
            &[
                ("plugin", "CONFPLUGIN-FAULTY".into()),
                ("hook", hook.into()),
            ],
        );
    }

    /// Emits the record of `hook` when `info` is a step's, and fails.
    fn fail_operation_hook(&self, hook: &str, info: &OperationInfo) -> HookWork {
        if info.sub_type() == Some("Step") {
            self.emit(hook);
        }
        Err("CONFPLUGIN-FAULTY".into())
    }
}

impl Plugin for Faulty {
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        self.emit("invocation-start");
        Box::pin(async { Err("CONFPLUGIN-FAULTY".into()) })
    }

    fn on_invocation_end<'a>(&'a self, _info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.emit("invocation-end");
        Box::pin(async { Err("CONFPLUGIN-FAULTY".into()) })
    }

    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        self.fail_operation_hook("operation-start", info)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        self.fail_operation_hook("operation-end", info)
    }

    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        self.fail_operation_hook("attempt-start", info.operation())
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        self.fail_operation_hook("attempt-end", info.operation())
    }
}

/// Creates the faulty plugin of each invocation.
struct FaultyFactory;

impl PluginFactory for FaultyFactory {
    fn create_plugin(
        &self,
        info: &InvocationInfo,
    ) -> Result<Option<Box<dyn Plugin>>, durable::BoxError> {
        Ok(Some(Box::new(Faulty {
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
    let options = durable::Options::builder().plugin(FaultyFactory).build()?;
    durable::run_with_options(handler, options).await
}
