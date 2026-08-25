//! Chained invoke scoped to a tenant.
//!
//! For a multi-tenant callee,
//! [`InvokeBuilder::tenant_id`](aws_durable_execution_sdk::builders::InvokeBuilder::tenant_id)
//! routes the invocation to a specific tenant's partition. The callee must be
//! deployed with tenant isolation enabled (see the companion
//! `invoke_target_tenant` and its `TenancyConfig`). The tenant id and payload
//! arrive together in the event.

use aws_durable_execution_sdk as durable;
use serde::Deserialize;

/// Event carrying the tenant to isolate to and the payload to forward.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Input {
    /// Tenant identifier the callee should be scoped to.
    tenant_id: String,
    /// Payload forwarded to the target function.
    payload: serde_json::Value,
}

/// Invokes the tenant-isolated echo target for the requested tenant.
async fn handler(
    event: Input,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let target = std::env::var("TARGET_FUNCTION_NAME")
        .map_err(|e| -> durable::BoxError { e.to_string().into() })?;
    let receipt = ctx
        .invoke::<serde_json::Value, _>(&target, event.payload)
        .tenant_id(event.tenant_id)
        .name("delegate")
        .await?;
    Ok(receipt)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
