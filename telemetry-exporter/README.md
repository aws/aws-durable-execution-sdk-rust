# Durable Execution Telemetry for the Rust SDK

`aws-durable-execution-sdk-telemetry-exporter` is a plugin for the
[AWS Lambda Durable Functions Rust SDK](../README.md). It builds one
telemetry record per durable execution and sends the record to one or more
exporters.

A record is a complete snapshot of one execution. It holds the execution's
identity, status, start and end times, input, output, error, and
operations. The schema is the one the JavaScript, Python and Java telemetry
exporters write: `recordType` `DurableExecutionTelemetry`, `schemaVersion`
`1.0`.

The crate is not published to crates.io yet. Depend on it by path or by git.

## Usage

Register `DurableExecutionTelemetry` as a plugin factory:

```rust,no_run
use aws_durable_execution_sdk as durable;
use aws_durable_execution_sdk_telemetry_exporter::DurableExecutionTelemetry;

async fn handler(event: u32, ctx: durable::DurableContext) -> Result<u32, durable::BoxError> {
    let next = ctx
        .step(move |_| async move { Ok(event + 1) })
        .name("increment")
        .await?;
    Ok(next)
}

#[tokio::main]
async fn main() -> Result<(), durable::BoxError> {
    let options = durable::Options::builder()
        .plugin(DurableExecutionTelemetry::builder().build())
        .build()?;
    durable::run_with_options(handler, options).await
}
```

Without an exporter, the plugin writes each record as one JSON line to
standard output. Lambda sends standard output to the function's log group.

## Exporters

| Exporter | Cargo feature | Destination | IAM permissions |
| --- | --- | --- | --- |
| `LambdaLogExporter` (default) | None | Standard output, one JSON line per record | None |
| `CloudWatchLogsExporter` | `cloudwatch-logs` | An existing log group you choose, one stream per UTC day | `logs:CreateLogStream`, `logs:PutLogEvents` |
| `RedshiftExporter` | `redshift` | A Redshift table you choose, one row per execution | `redshift-data:ExecuteStatement`, plus `redshift-serverless:GetCredentials` (Serverless) or `redshift:GetClusterCredentialsWithIAM` (provisioned cluster) |

`CloudWatchLogsExporter` does not create the log group. Create the group
before the function runs.

No cargo feature is on by default. An exporter that needs a service client
is behind its own feature, so a build compiles only the clients of the
exporters it enables. To use `CloudWatchLogsExporter` or `RedshiftExporter`,
enable its feature:

```toml
aws-durable-execution-sdk-telemetry-exporter = { path = "...", features = ["cloudwatch-logs", "redshift"] }
```

`LambdaLogExporter` and `CloudWatchLogsExporter` write the
`operationsByName` form of the record. `RedshiftExporter` writes the
`operations` array. `LambdaLogExporter` and `CloudWatchLogsExporter`
truncate a record larger than 256,000 bytes. `RedshiftExporter` truncates a
record larger than 1,000,000 bytes. Truncation drops operation
results first, then whole operations, then the input, then the output. A
truncated record carries `truncated: true` and the counts of what it
dropped.

Each size check serializes the whole record. So truncation does not check
the size after every drop. Each phase binary searches the number of
results, then the number of operations, to drop. So one truncation
serializes the record about 2 × log2(n) times for n operations.
Truncation accepts a size only after it serialized that record. So the
record never goes over the limit, as long as dropping everything that can
be dropped makes it fit.

The search assumes that dropping more never makes the record larger. Some
records break this. So truncation can drop more results and operations
than the fewest that fit. For example, in the array form, clearing a short
result such as `true` adds `"truncated":true`, so the record grows. A
record with one old result of 2,000 bytes and 1,999 results of `true`,
`1`, `"ok"` or `null` is 1 byte over the limit. Clearing the old result
alone fits. Truncation clears all 2,000 results and drops 53 operations.

Implement the `TelemetryExporter` trait to send records anywhere else. One
task that the plugin factory owns makes every export and every flush, one at
a time. So an exporter never receives two calls at once. An exporter error is
logged as a warning. It never fails the execution.

To unit test an exporter, build a record with `TelemetryRecord::new` and
`OperationRecord::new`, then assign the fields the test needs.

### `RedshiftExporter`

`RedshiftExporter` runs one `MERGE` statement per record with the Redshift
Data API. It matches rows on `execution_arn`, so the table holds one row per
execution. The table must exist. Its columns are `execution_arn`,
`execution_name`, `function_name`, `status`, `start_time`, `end_time`,
`duration_ms`, `record_json` (type `SUPER`) and `emitted_at`.

The builder needs `workgroup_name` (Redshift Serverless) or
`cluster_identifier` (provisioned cluster), and `build()` returns an error
without one. `database` is the builder's argument. `db_user`, `secret_arn`,
`region` and `client` are optional. `table` defaults to
`durable_execution_telemetry` and `schema` to `public`. Both must match
`^[a-zA-Z_][a-zA-Z0-9_]*$`, and `build()` returns an error otherwise.

The Data API runs a statement asynchronously. The exporter does not wait for
the statement to finish. An export succeeds when Redshift accepts the
statement.

```rust,no_run
# #[cfg(feature = "redshift")]
# fn build() -> Result<(), aws_durable_execution_sdk::BoxError> {
use aws_durable_execution_sdk_telemetry_exporter::{
    DurableExecutionTelemetry, RedshiftExporter,
};

let exporter = RedshiftExporter::builder("dev")
    .workgroup_name("my-workgroup")
    .build()?;
let telemetry = DurableExecutionTelemetry::builder()
    .exporter(exporter)
    .build();
# drop(telemetry);
# Ok(())
# }
```

## Configuration

| Builder method | Default | Effect |
| --- | --- | --- |
| `emit_mode` | `EmitMode::OnComplete` | `OnComplete` emits when the execution succeeds or fails. `OnFailure` emits only when it fails. `OnChange` also emits a `RUNNING` record when each invocation starts and on every checkpoint response that changes an operation, and a record at every invocation end. The invocation-start record is exported in the background, so the handler does not wait for it. A newer record that is ready first replaces it. |
| `sampling_rate` | `1.0` | The fraction of executions that emit records. The decision hashes the execution ARN, so every invocation of an execution makes the same decision. |
| `operation_detail` | `OperationDetail::TopLevel` | `TopLevel` lists operations without a parent. `FullTree` also lists map items, parallel branches and child-context operations. |
| `content` | input, output and errors; no operation results | `ContentConfig` includes, omits or transforms the input and output, and sets per-operation overrides. |

A content transform receives the payload as JSON. If the transform returns
an error or panics, the record omits the payload. So a failing redaction
never writes the original value.

```rust
use aws_durable_execution_sdk_telemetry_exporter::{
    ContentConfig, ContentSetting, DurableExecutionTelemetry, EmitMode, OperationOverride,
};
use serde_json::json;

let telemetry = DurableExecutionTelemetry::builder()
    .emit_mode(EmitMode::OnFailure)
    .content(
        ContentConfig::new()
            .input(ContentSetting::Omit)
            .output(ContentSetting::Omit)
            .operation_override(
                OperationOverride::new("fetch-user")
                    .result(|value| Ok(json!({ "userId": value.get("userId").cloned() }))),
            ),
    )
    .build();
```

## Development

The crate is its own cargo workspace. The SDK's dependency allowlist does
not include `aws-sdk-cloudwatchlogs` or `aws-sdk-redshiftdata`, so the crate is not a member of the
root workspace. `make workspaces WORKSPACES=telemetry-exporter` checks the
formatting, builds, lints, runs the tests and builds the docs. It lints,
tests and builds the docs twice: with no feature and with every feature.
