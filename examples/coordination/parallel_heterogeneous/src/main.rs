//! Heterogeneous fan-out: branches of different result types in one
//! parallel operation.
//!
//! [`parallel_mixed`](aws_durable_execution_sdk::DurableContext::parallel_mixed)
//! runs branches of different result types in one parallel operation. Each
//! branch is added to a [`BranchSet`](aws_durable_execution_sdk::BranchSet),
//! which returns a [`BranchHandle`](aws_durable_execution_sdk::BranchHandle)
//! typed with that branch's result. After the `.await` of `parallel_mixed`,
//! each handle returns its own type.
//!
//! This example runs a numeric branch and a string branch, then combines
//! their results.

use aws_durable_execution_sdk as durable;

/// Runs two differently-typed branches in one parallel operation and
/// combines their results.
async fn handler(
    _event: serde_json::Value,
    ctx: durable::DurableContext,
) -> Result<String, durable::BoxError> {
    let mut branches = durable::BranchSet::new();
    let count = branches.add("count", |child| async move {
        Ok(child.step(|_| async { Ok(7u32) }).name("count").await?)
    });
    let status = branches.add("status", |child| async move {
        Ok(child
            .step(|_| async { Ok("ready".to_owned()) })
            .name("status")
            .await?)
    });

    ctx.parallel_mixed(branches).name("fan-out").await?;
    Ok(format!(
        "{}: {}",
        count.into_result()?,
        status.into_result()?
    ))
}

#[tokio::main]
async fn main() -> Result<(), lambda_runtime::Error> {
    durable::init_default_subscriber()?;
    durable::run(handler).await
}

#[cfg(test)]
mod tests {
    use aws_durable_execution_sdk::test_util::LocalRunner;

    /// The handler returns the count branch's value and the status
    /// branch's value, each read through its own typed handle.
    #[tokio::test]
    async fn outputs_count_and_status() -> Result<(), aws_durable_execution_sdk::BoxError> {
        let result = LocalRunner::new()
            .run(super::handler, serde_json::Value::Null)
            .await?;
        assert_eq!(result.output().map(String::as_str), Some("7: ready"));
        Ok(())
    }

    /// This workspace builds `serde_json` without `preserve_order`, so a
    /// `serde_json::Map` writes its keys in alphabetical order. The SDK's
    /// own test graph enables the feature, so only a workspace like this
    /// one can show that the SDK's record text does not rely on it.
    #[test]
    fn serde_json_map_is_not_preserve_order() {
        let mut map = serde_json::Map::new();
        map.insert("z".to_owned(), 0.into());
        map.insert("a".to_owned(), 0.into());
        assert_eq!(
            serde_json::Value::Object(map).to_string(),
            r#"{"a":0,"z":0}"#
        );
    }

    /// An oversized one-branch `parallel_mixed` with a caller summary
    /// records its summary with JS's key order.
    #[tokio::test]
    async fn oversized_parallel_mixed_summary_has_js_key_order()
    -> Result<(), aws_durable_execution_sdk::BoxError> {
        let result = LocalRunner::new()
            .run(
                |_event: serde_json::Value, ctx: aws_durable_execution_sdk::DurableContext| async move {
                    let mut branches = aws_durable_execution_sdk::BranchSet::new();
                    let big = branches.add("big", |_child| async { Ok("x".repeat(300 * 1024)) });
                    ctx.parallel_mixed(branches)
                        .name("oversized")
                        .summary(|_batch| "ok".to_owned())
                        .await?;
                    Ok::<_, aws_durable_execution_sdk::BoxError>(big.into_result()?.len())
                },
                serde_json::Value::Null,
            )
            .await?;
        assert_eq!(result.output(), Some(&(300 * 1024)));
        let operation = result
            .operation("oversized")
            .ok_or("the parallel_mixed operation was recorded")?;
        assert_eq!(
            operation.result(),
            Some(
                r#"{"type":"ParallelResult","totalCount":1,"successCount":1,"failureCount":0,"startedCount":0,"completionReason":"ALL_COMPLETED","status":"SUCCEEDED","itemStatuses":"S","summary":"ok"}"#
            )
        );
        Ok(())
    }
}
