//! Conformance handler for requirement 7-11: wait_for_callback with JSON result.
//! Callback payload is JSON object `{"status":"approved"}`, handler returns the status field.

use aws_durable_execution_sdk_rust as durable;
use serde::{Deserialize, Serialize};

/// The structured callback payload.
#[derive(Deserialize, Serialize)]
struct ApprovalResult {
    status: String,
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(|name: String, ctx: durable::DurableContext| async move {
        let result: ApprovalResult = ctx
            .wait_for_callback::<ApprovalResult, _, _>(|_step_ctx, _cb_id| async { Ok(()) })
            .name(&name)
            .await?;
        Ok(result.status)
    })
    .await
}
