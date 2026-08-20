//! Companion target for `invoke_tenant_id`: a tenant-isolated durable function
//! that echoes its input.
//!
//! The handler is identical to `invoke_target`; tenant isolation is configured
//! on the deployed function (`TenancyConfig` in the SAM template), not in
//! handler code. The caller passes a tenant id through
//! [`InvokeBuilder::tenant_id`](aws_durable_execution_sdk_rust::builders::InvokeBuilder::tenant_id),
//! and the platform routes the invocation to the matching tenant partition.

use std::time::Duration;

use aws_durable_execution_sdk_rust as durable;

/// Echoes the input event unchanged after a one-second durable wait.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    ctx.wait(Duration::from_secs(1)).name("settle").await?;
    Ok(event)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
