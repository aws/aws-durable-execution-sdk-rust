//! Conformance requirement 10-17: a healthy plugin registered after a
//! faulty one receives every hook, and the execution still succeeds.

use aws_durable_execution_sdk as durable;
use durable::OperationStatus;
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

/// Records the hooks of one invocation as `CONFPLUGIN-HEALTHY`.
struct Healthy {
    execution_arn: String,
    /// The factory's copy of the first-invocation flag.
    first: bool,
}

impl Healthy {
    /// Emits `fields` after `plugin` `CONFPLUGIN-HEALTHY` and `hook` `hook`.
    fn emit(&self, hook: &str, fields: &[(&str, serde_json::Value)]) {
        let mut record = vec![
            ("plugin", "CONFPLUGIN-HEALTHY".into()),
            ("hook", hook.into()),
        ];
        record.extend(fields.iter().cloned());
        conformance::emit_record(&self.execution_arn, &record);
    }
}

impl Plugin for Healthy {
    fn on_invocation_start<'a>(&'a self, _info: &'a InvocationInfo) -> HookFuture<'a> {
        self.emit("invocation-start", &[("first", self.first.into())]);
        Box::pin(std::future::ready(Ok(())))
    }

    fn on_invocation_end<'a>(&'a self, info: &'a InvocationEndInfo) -> HookFuture<'a> {
        self.emit(
            "invocation-end",
            &[("status", info.status().wire_str().into())],
        );
        Box::pin(std::future::ready(Ok(())))
    }

    fn on_operation_start(&self, info: &OperationInfo) -> HookWork {
        if info.sub_type() == Some("Step") {
            self.emit("operation-start", &[("op", info.id().into())]);
        }
        Ok(None)
    }

    fn on_operation_end(&self, info: &OperationInfo) -> HookWork {
        if info.sub_type() == Some("Step") {
            self.emit(
                "operation-end",
                &[
                    ("op", info.id().into()),
                    (
                        "status",
                        info.status().map(OperationStatus::wire_str).into(),
                    ),
                ],
            );
        }
        Ok(None)
    }

    fn on_attempt_start(&self, info: &AttemptInfo) -> HookWork {
        if info.operation().sub_type() == Some("Step") {
            self.emit("attempt-start", &[("op", info.operation().id().into())]);
        }
        Ok(None)
    }

    fn on_attempt_end(&self, info: &AttemptEndInfo) -> HookWork {
        if info.operation().sub_type() == Some("Step") {
            self.emit(
                "attempt-end",
                &[
                    ("op", info.operation().id().into()),
                    ("outcome", info.outcome().wire_str().into()),
                ],
            );
        }
        Ok(None)
    }
}

/// Creates the healthy plugin of each invocation.
struct HealthyFactory;

impl PluginFactory for HealthyFactory {
    fn create_plugin(
        &self,
        info: &InvocationInfo,
    ) -> Result<Option<Box<dyn Plugin>>, durable::BoxError> {
        Ok(Some(Box::new(Healthy {
            execution_arn: info.execution_arn().to_owned(),
            first: info.is_first_invocation(),
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
        .plugin(FaultyFactory)
        .plugin(HealthyFactory)
        .build()?;
    durable::run_with_options(handler, options).await
}
