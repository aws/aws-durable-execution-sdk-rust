//! Conformance handler for requirement 4-18: two callbacks: create both then wait in order.
//! Create A, create B, wait for A, wait for B, return both.

use aws_durable_execution_sdk_rust as durable;
use serde::Serialize;
use std::collections::HashMap;

/// Output: map with keys "a" and "b".
#[derive(Serialize)]
struct Output(HashMap<String, String>);

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |names: Vec<String>, ctx: durable::DurableContext| async move {
            let name_a = names.first().ok_or("missing name A")?;
            let name_b = names.get(1).ok_or("missing name B")?;

            let cb_a = ctx
                .create_callback::<String>()
                .name(name_a.as_str())
                .await?;
            let cb_b = ctx
                .create_callback::<String>()
                .name(name_b.as_str())
                .await?;

            let result_a = cb_a.result().await?;
            let result_b = cb_b.result().await?;

            let mut map = HashMap::new();
            map.insert("a".to_owned(), result_a);
            map.insert("b".to_owned(), result_b);
            Ok(map)
        },
    )
    .await
}
