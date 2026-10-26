# AWS Durable Execution SDK for Rust - Agent Guide

This repository contains the AWS Durable Execution SDK for Rust, the
`aws-durable-execution-sdk` crate, used to author AWS Lambda durable
functions.

## Use the README and rustdoc, not this file, for APIs

Do not rely on this file for method signatures, builder options, or code
examples. The API reference for this SDK is the [README](README.md) and the
rustdoc comments in `src/`. To read the rustdoc in a browser, run
`cargo doc --no-deps --all-features --open`.

The [AWS Durable Execution SDK Developer Guide](https://docs.aws.amazon.com/durable-execution/)
explains the concepts that all the durable execution SDKs share. Its code
examples cover other languages, not Rust. Use it for what each operation
does, and the README and rustdoc for how to call it in Rust. Key sections:

| Topic | Link |
| --- | --- |
| Key concepts: replay, checkpoints, determinism | <https://docs.aws.amazon.com/durable-execution/getting-started/key-concepts/> |
| Steps | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/step/> |
| Waits | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/wait/> |
| Wait for condition (polling) | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/wait-for-condition/> |
| Callbacks and wait for callback | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/callback/> |
| Invoke (function chaining) | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/invoke/> |
| Parallel | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/parallel/> |
| Map | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/map/> |
| Child contexts | <https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/child-context/> |
| Errors | <https://docs.aws.amazon.com/durable-execution/sdk-reference/error-handling/errors/> |
| Retries | <https://docs.aws.amazon.com/durable-execution/sdk-reference/error-handling/retries/> |
| Serialization | <https://docs.aws.amazon.com/durable-execution/sdk-reference/state/serialization/> |
| Best practices (determinism, idempotency, state) | <https://docs.aws.amazon.com/durable-execution/patterns/best-practices/> |

For Lambda service topics such as deployment, infrastructure as code,
invocation, IAM permissions, and quotas, see the
[Lambda durable functions guide](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html).

## Critical rules: the replay model

Durable functions use checkpoint and replay. After a wait, failure, or
resume, code re-runs from the beginning. Completed steps return their
checkpointed results without re-executing, and code outside steps runs
again on every replay. This implies four rules:

1. **Code outside steps must be deterministic.** Wrap clock reads, random
   values, ID generation, calls to other services, and any other
   non-deterministic work in a step. The SDK assigns each operation its ID
   in the order the handler creates operations. So the handler must create
   the same operations in the same order on every replay. Do not create
   operations while iterating a `HashMap` or `HashSet`, because their
   iteration order changes between runs. Sort into a `Vec` first, or use a
   `BTreeMap` or `BTreeSet`.
2. **Never call durable operations inside a step.** `StepContext`, the
   argument of a step body, has no operation methods. A `DurableContext`
   clone moved into the step closure still compiles. Do not do that. On
   replay the step body does not run, so every later operation gets a
   different ID than in the first invocation. Later operations then
   receive the wrong recorded results, or the execution fails with
   `NonDeterministicExecutionError`. Use `ctx.run_in_child_context` to
   group operations.
3. **Mutations inside steps are lost on replay.** A step closure is
   `'static`, so it cannot borrow a local variable. The body can still
   change shared state through an `Arc<Mutex<_>>` or an atomic. That change
   happens only in the invocation that runs the body, not on replay. Return
   values from steps instead.
4. **Side effects outside steps repeat on every replay.** Put side effects
   in steps. Logging is a side effect too: a `tracing` event between
   operations is emitted again on every replay. The SDK does not filter
   these events by default. To suppress them, install `ReplayFilterLayer`
   (feature `replay-filter`) as the filter of your `tracing-subscriber`
   layer.

Three more rules are specific to Rust. The crate documentation lists them
under "Determinism contract" in `src/lib.rs`.

5. **Use the SDK's combinators, not `tokio::select!`.** Use `ctx.race` or
   `ctx.select_ok` over durable futures, so the SDK records which branch
   won.
6. **Use `.spawn()` on an operation's builder, not `tokio::spawn`.** A
   durable operation created on a task that the SDK did not start fails
   with an error.
7. **Do not rely on `Drop` between operations.** When an execution
   suspends, the SDK drops the handler future. So a guard's `Drop` runs at
   the suspension, not after the wait completes. Release resources in a
   step.

See [Determinism and Replay](https://docs.aws.amazon.com/durable-execution/patterns/best-practices/determinism/)
for worked examples in other languages.

## Rules for SDK code

These rules apply to the SDK code in `src/`.

- **Do not panic.** A panic in the SDK fails the customer's Lambda
  invocation. So clippy denies `unwrap`, `expect`, `panic!`, and slice
  indexing. Return an error, and use `get()` instead of an index. Tests
  may panic: `src/lib.rs` allows `unwrap` and `expect` in unit tests, and
  other test code states its exemption with `#[expect(...)]`.
- **Do not add an error path for a case that cannot happen.** No test can
  reach such a path, so it breaks
  [rule 4 of the rules for every change](CONTRIBUTING.md#4-every-branch-has-a-test-that-fails-without-it).
  When a type forces you to handle the case, call `.expect("...")` under
  `#[expect(clippy::expect_used)]`, with a comment that states why the case
  cannot happen.
- **Suppress a lint with `#[expect]`, not `#[allow]`.** Give the reason in
  a comment. An `#[expect]` that no longer matches a lint fails
  `make check`, so a stale suppression cannot stay in the code. The
  unit-test exemption in `src/lib.rs` is the one `allow`.
- **Do not call `tokio::spawn`.** The SDK's task-ownership checks and
  suspension accounting do not cover a task that `tokio::spawn` starts.
  `clippy.toml` disallows the call. Spawn through the helpers in
  `src/future.rs`.
- **Do not add a production dependency as part of another change.**
  `scripts/check-direct-deps.sh` fails on a direct dependency outside its
  allowlist. Adding one needs its own pull request. See
  [Dependency policy](CONTRIBUTING.md#dependency-policy). Dev-dependencies
  need no approval.

## Testing requirements

All changes MUST include related tests. At minimum, include **unit tests**
in a `#[cfg(test)] mod tests` block in the module under test. Add an
**integration test** under `tests/` when the change spans several
operations, affects the public API, or needs to observe replay. These tests
drive a whole handler through `LocalRunner` (feature `test-util`). For an
isolated bug fix that a unit test covers, an integration test is not
required.

Do NOT add or modify **conformance tests** without coordinating with the
team. The runner and most requirements live in a separate repository
([aws-durable-execution-conformance-tests](https://github.com/aws/aws-durable-execution-conformance-tests)).
This repository holds the Rust handlers under `conformance/`. It also holds
the requirements of three suites that started in this SDK (`history`,
`nondeterminism`, and `combinator`), under `conformance/conformance_ext/`.
If a change warrants a new conformance test, note it in the PR description
or [open an issue](https://github.com/aws/aws-durable-execution-conformance-tests/issues/new?template=new_requirement.yml)
in that repository.

## Working in this repository

- The repository holds three independent cargo workspaces: the SDK crate at
  the root (`src/` and `tests/`), the conformance handlers
  (`conformance/`), and the deployable examples (`examples/`).
- Read [CONTRIBUTING.md](CONTRIBUTING.md) before making changes, and
  follow its [rules for every change](CONTRIBUTING.md#rules-for-every-change).
  Run `make check` before you push. It is the single quality gate, and
  CONTRIBUTING.md lists what it runs.
- `make check` covers only the root workspace. After a public API change,
  also build and lint `conformance/` and `examples/`. CONTRIBUTING.md gives
  the commands.
- Build a deployable binary with cargo-lambda, not with `cargo build`. The
  `build_examples.sh` scripts in `conformance/` and `examples/` call it.
  `cargo build` links against the build host's glibc. A binary linked
  against a newer glibc than the `provided.al2023` runtime's fails to
  start there.
- When the README, the rustdoc, and the source disagree, trust the source
  in this repository and report the discrepancy.
