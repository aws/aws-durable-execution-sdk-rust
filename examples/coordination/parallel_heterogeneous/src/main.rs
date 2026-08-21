//! Heterogeneous fan-out: differently-typed operations in parallel.
//!
//! [`parallel`](aws_durable_execution_sdk_rust::DurableContext::parallel)
//! requires a single branch result type. When you want to run operations that
//! return *different* types concurrently, the idiom is to hold the operation
//! builders and drive them with `tokio::join!`. Each builder claims its
//! operation identity at creation, so joining them is replay-safe: polling
//! order cannot change which operation is which.
//!
//! This is the Rust analogue of a variadic, mixed-type `Promise.all`. The
//! variadic-tuple sugar is not expressible in Rust's type system, but the
//! capability, concurrent, heterogeneous durable operations, is fully
//! preserved through `tokio::join!` over held builders.
//!
//! This example joins a numeric step and a string step, then combines them.

use aws_durable_execution_sdk_rust as durable;

/// Joins two differently-typed steps concurrently and combines their results.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    // Builders are lazy handles with already-claimed identities; tokio::join!
    // polls them concurrently and returns both results.
    let count = ctx.step(|_| async { Ok(7u32) }).name("count");
    let status = ctx
        .step(|_| async { Ok("ready".to_owned()) })
        .name("status");

    let (count, status) = tokio::join!(count, status);
    Ok(format!("{}: {}", count?, status?))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
