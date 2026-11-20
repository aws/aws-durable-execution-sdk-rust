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

| Exporter | Destination | IAM permissions |
| --- | --- | --- |
| `LambdaLogExporter` (default) | Standard output, one JSON line per record | None |

The exporter writes the `operationsByName` form of the record. It
truncates a record larger than 256,000 bytes. Truncation drops operation
results first, then whole operations, then the input, then the output. A
truncated record carries `truncated: true` and the counts of what it
dropped.

Implement the `TelemetryExporter` trait to send records anywhere else. Every
export and every flush holds one lock that the plugin factory owns. So an
exporter never receives two calls at once. An exporter error is logged as a
warning. It never fails the execution.

To unit test an exporter, build a record with `TelemetryRecord::new` and
`OperationRecord::new`, then assign the fields the test needs.

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

The crate is its own cargo workspace. Exporters that need an AWS service
client depend on crates that the SDK's dependency allowlist does not
include, so the crate is not a member of the root workspace.
`make workspaces WORKSPACES=telemetry-exporter` checks the formatting,
builds, lints, runs the tests and builds the docs.
