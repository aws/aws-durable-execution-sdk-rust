# Examples

Deployable durable functions for the Rust SDK. Each example is a real Lambda
function on the `provided.al2023` runtime and doubles as an end-to-end smoke
test against the live service, using the same packaging story as
`compliance/` (a shared cargo workspace, one `bootstrap` per example staged
under `publish/`).

Examples are user-facing documentation. Every one is a single honest workload
for one pattern — no artificial scaffolding, no option menus, no branching
demo harnesses. The mapping from the JavaScript SDK's examples tree (which
examples are ported, which are covered by a representative, and which do not
apply to Rust) is tracked in [`docs/porting-map.md`](../docs/porting-map.md).

## Families

| Family | Scope |
| --- | --- |
| `basics/` | fundamental `step`, `wait`, retry, and no-op handler patterns |
| `coordination/` | `run_in_child_context`, `parallel`, `map`, the four combinators, concurrent fan-out via `.spawn()`, determinism/replay behaviors |
| `external/` | `invoke`, `create_callback`, `wait_for_callback`, `wait_for_condition`, serdes, large payloads, `tracing` logging, and the capstone comprehensive example |

## Build and deploy

```sh
# Build a family (default: all families). Produces publish/<example>/bootstrap.
./build_examples.sh basics

# Configure AWS credentials for a test account, then deploy a family:
sam deploy --template-file template_basics.yaml \
    --stack-name dex-rust-examples-basics \
    --resolve-s3 --capabilities CAPABILITY_IAM --region us-west-2
```

`build_examples.sh` mirrors `compliance/build_examples.sh` exactly: one shared
`cargo build` over the requested families, a skip-if-unchanged guard keyed on
git HEAD + clean tree + a per-family stamp, and a `Makefile`-per-bootstrap for
SAM's `BuildMethod: makefile`. It is a separate cargo workspace
(`examples/Cargo.toml`) so its heavier dependency graph never enters the SDK's
root `make check`.

## Smoke-test notes

- **Callbacks are driven externally.** After an execution suspends on a
  callback, complete it with
  `aws lambda send-durable-execution-callback-success`; the execution then
  finishes SUCCEEDED.
- **Expected-FAILED example.** `external/handler_error` returns an error from
  the handler and finishes FAILED deterministically — a FAILED terminal state
  is its pass condition. The two callback-timeout examples catch their
  timeouts and finish SUCCEEDED.
- **Companion callees.** `external/invoke_target` and
  `external/invoke_target_tenant` are deployed in the same stack; callers
  reach them through the `TARGET_FUNCTION_NAME` environment variable
  (`${Target.Arn}:$LATEST`).
