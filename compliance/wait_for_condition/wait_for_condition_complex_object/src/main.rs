//! Conformance handler for requirement 6-9: complex object state.

use aws_durable_execution_sdk_rust as durable;
use durable::WaitDecision;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Serialize, Deserialize)]
struct PollState {
    status: String,
    attempts: i32,
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::run(
        |_input: serde_json::Value, ctx: durable::DurableContext| async move {
            let result = ctx
                .wait_for_condition(
                    |_ctx, mut state: PollState| async move {
                        state.attempts += 1;
                        if state.attempts >= 2 {
                            state.status = "DONE".to_owned();
                        }
                        Ok(state)
                    },
                    PollState {
                        status: "PENDING".to_owned(),
                        attempts: 0,
                    },
                )
                .wait_strategy_fn(Box::new(|state: PollState, _attempt| {
                    if state.status == "DONE" {
                        WaitDecision::complete()
                    } else {
                        WaitDecision::continue_with(Duration::from_secs(1))
                    }
                }))
                .await?;
            Ok(result)
        },
    )
    .await
}
