# Examples

Deployable durable functions for the Rust SDK. Each example is a real Lambda
function on the `provided.al2023` runtime and doubles as an end-to-end smoke
test against the live service, using the same packaging story as
`conformance/` (a shared cargo workspace, one `bootstrap` per example staged
under `publish/`).

Examples are user-facing documentation. Every one is a single honest workload
for one pattern, no artificial scaffolding, no option menus, no branching
demo harnesses.

## Families

| Family | Scope |
| --- | --- |
| `basics/` | fundamental `step`, `wait`, retry, and no-op handler patterns |
| `coordination/` | `run_in_child_context` (one child in `child_basic`, concurrent child fan-out joined with `try_join_all` in `child_fanout`), `parallel`, `map`, the four combinators, concurrent fan-out via `.spawn()`, determinism/replay behaviors |
| `external/` | `invoke`, `create_callback`, `wait_for_callback`, `wait_for_condition`, serdes, large payloads, `tracing` logging, and the capstone comprehensive example |

## Build and deploy

Prerequisite: [cargo-lambda](https://www.cargo-lambda.info/), the build tool
the Lambda Developer Guide documents for
[packaging Rust functions](https://docs.aws.amazon.com/lambda/latest/dg/rust-package.html).
It is the single build path for this workspace and for `conformance/`, used
identically on a workstation and in CI, so a local pass and a CI pass mean the
same thing. A plain `cargo build` is not a substitute: it links against the
build host's glibc, and a binary built on a host newer than the
`provided.al2023` runtime (glibc 2.34) fails to start there at all.

```sh
pip3 install cargo-lambda     # or: cargo install cargo-lambda
```

```sh
# Build a family (default: all families). Produces publish/<example>/bootstrap.
./build_examples.sh basics

# Configure AWS credentials for a test account, then deploy a family. The
# templates do not create an IAM role: every deploy MUST pass the ARN of an
# existing Lambda execution role (with the durable-execution permissions:
# lambda:CheckpointDurableExecution, lambda:GetDurableExecutionState) via
# --parameter-overrides.
sam deploy --template-file template_basics.yaml \
    --stack-name dex-rust-examples-basics \
    --resolve-s3 --region us-west-2 \
    --parameter-overrides ExecutionRoleArn=arn:aws:iam::<account>:role/<lambda-execution-role>
```

`build_examples.sh` mirrors `conformance/build_examples.sh` exactly: one shared
`cargo lambda build` over the requested families, a skip-if-unchanged guard
keyed on git HEAD + clean tree + a per-family stamp, and a `Makefile`-per-bootstrap
for SAM's `BuildMethod: makefile`. It is a separate cargo workspace
(`examples/Cargo.toml`) so its heavier dependency graph never enters the SDK's
root `make check`.

## Smoke-test notes

The CI cloud-test workflow (`.github/workflows/cloud-tests.yml`) deploys all
three family stacks and runs `cloud/run_cloud_tests.sh`, which invokes every
directly-invokable example and asserts the terminal states below.

- **Callbacks are driven externally.** After an execution suspends on a
  callback, complete it with
  `aws lambda send-durable-execution-callback-success`; the execution then
  finishes SUCCEEDED.
- **Expected-FAILED example.** `external/handler_error` returns an error from
  the handler and finishes FAILED deterministically: a FAILED terminal state
  is its pass condition. The two callback-timeout examples catch their
  timeouts and finish SUCCEEDED.
- **Companion callees.** `external/invoke_target` and
  `external/invoke_target_tenant` are deployed in the same stack; callers
  reach them through the `TARGET_FUNCTION_NAME` environment variable
  (`${Target.Arn}:$LATEST`).
