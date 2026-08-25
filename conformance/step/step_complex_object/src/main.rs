//! Conformance requirement 1-4: returning a complex nested object.

use aws_durable_execution_sdk as durable;
use serde::{Deserialize, Serialize};

/// Input event.
#[derive(Debug, Deserialize)]
struct Input {
    /// User name.
    name: String,
    /// Tags list.
    tags: Vec<String>,
}

/// Nested user object.
#[derive(Debug, Serialize, Deserialize)]
struct User {
    /// Name field.
    name: String,
    /// Tags field.
    tags: Vec<String>,
}

/// Response object.
#[derive(Debug, Serialize, Deserialize)]
struct Response {
    /// Nested user.
    user: User,
    /// Count of tags.
    count: usize,
}

/// Handler: step returning a nested object with arrays.
async fn handler(
    event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<serde_json::Value, durable::BoxError> {
    let input: Input = serde_json::from_value(event)?;
    let result = ctx
        .step(move |_| async move {
            let count = input.tags.len();
            Ok(Response {
                user: User {
                    name: input.name,
                    tags: input.tags,
                },
                count,
            })
        })
        .await?;
    Ok(serde_json::to_value(result)?)
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    lambda_runtime::tracing::init_default_subscriber();
    durable::run(handler).await
}
