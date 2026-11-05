//! Conformance requirement 10-12: the plugin user-function hooks fire for
//! each parallel branch, with the parallel operation as the parent.

use aws_durable_execution_sdk as durable;
use durable::plugin::{HookWork, InvocationInfo, OperationInfo, Plugin, PluginFactory};
use durable::{Branch, DurableContext, WireError};

/// The sub-type of a parallel branch's child context.
const PARALLEL_BRANCH: &str = "ParallelBranch";

/// Records the user-function hooks of parallel branches as `CONFPLUGIN`.
struct Recorder {
    execution_arn: String,
}

impl Plugin for Recorder {
    fn on_user_function_start(&self, info: &OperationInfo) -> HookWork {
        if info.sub_type() == Some(PARALLEL_BRANCH) {
            conformance::emit_record(
                &self.execution_arn,
                &[
                    ("plugin", "CONFPLUGIN".into()),
                    ("hook", "fn-start".into()),
                    ("op", info.id().into()),
                    ("parent", info.parent_id().unwrap_or("NONE").into()),
                ],
            );
        }
        Ok(None)
    }

    fn on_user_function_end(
        &self,
        info: &OperationInfo,
        outcome: Result<(), &WireError>,
    ) -> HookWork {
        if info.sub_type() == Some(PARALLEL_BRANCH) {
            let outcome = if outcome.is_ok() {
                "SUCCEEDED"
            } else {
                "FAILED"
            };
            conformance::emit_record(
                &self.execution_arn,
                &[
                    ("plugin", "CONFPLUGIN".into()),
                    ("hook", "fn-end".into()),
                    ("op", info.id().into()),
                    ("parent", info.parent_id().unwrap_or("NONE").into()),
                    ("outcome", outcome.into()),
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
        })))
    }
}

/// Handler: a parallel named "parallel" with two branches that run one at a
/// time and return a constant each.
async fn handler(
    _event: serde_json::Value,
    ctx: DurableContext,
) -> Result<Vec<String>, durable::BoxError> {
    let branches: Vec<Branch<String>> = vec![
        Branch::new("0", |_: DurableContext| async { Ok("task-1".to_owned()) }),
        Branch::new("1", |_: DurableContext| async { Ok("task-2".to_owned()) }),
    ];
    Ok(ctx
        .parallel(branches)
        .name("parallel")
        .max_concurrency(1)
        .await?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    let options = durable::Options::builder().plugin(Factory).build()?;
    durable::run_with_options(handler, options).await
}
