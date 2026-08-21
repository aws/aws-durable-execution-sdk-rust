# Contributing

We welcome bug reports, corrections, and pull requests. Read this document
first so your change arrives in a shape we can review quickly.

The [README](README.md) documents the SDK itself: the handler signature, the
operations, and how to deploy a durable function. This guide covers the rest of
the repository, the checks that guard it, and how a change gets from a branch to
`alpha`.

All work targets the `alpha` branch. Branch from it and open your pull request
against it.

## Standards and where they come from

Where established Rust guidance already answers a question, this project
follows it instead of inventing a house rule.

Formatting comes from the
[Rust Style Guide](https://doc.rust-lang.org/style-guide/), and rustfmt applies
it with no overrides. `rustfmt.toml` carries only a comment saying so, which
makes rustfmt's default output the definition of correct formatting and leaves
nothing for a reviewer to argue about.

The exported surface follows the
[Rust API Guidelines](https://rust-lang.github.io/api-guidelines/): naming,
`Debug` implementations, the shape of builders, and what a public type promises.
`Cargo.toml` records the precedence we apply when guidance conflicts, which is
rustfmt first, then the API Guidelines, then clippy's judgment.

Commit subjects follow
[Conventional Commits](https://www.conventionalcommits.org/). No gate validates
them, so a reviewer checks the subject line instead. Keep the subject under 50
characters, write it in the imperative mood, drop the trailing period, and wrap
body text at 72 characters.

The crate declares `rust-version = "1.94.1"` and edition 2024. Most CI jobs
build on stable, but a dedicated `msrv` job in `ci.yml` runs `cargo check
--all-targets --all-features` on the declared minimum, so reaching for a
feature newer than 1.94.1 fails that leg rather than silently raising the
true minimum above the declared one. Bumping the minimum supported Rust
version is an intentional, documented change: while the crate is pre-1.0, an
MSRV bump ships as a minor release (0.x → 0.x+1), never as a patch, and the
change must update every declaration in the same pull request: the root
`Cargo.toml` is the source of truth, and every `Cargo.toml` under
`compliance/` and `examples/` that declares `rust-version` (200+ manifests),
the `msrv` CI job, and the README must agree with it.
`scripts/check-msrv-consistency.sh`, wired into `make check`, enforces this
mechanically, so a bump that misses a declaration fails the quality gate
rather than leaving stale minimums behind. There is deliberately no
`rust-toolchain.toml`: pinning one would override `rustup default stable` in
every CI job (the file takes precedence over the default toolchain), quietly
turning the stable legs into MSRV legs. Contributors build on any toolchain
at or above the minimum; the `msrv` job is what guards the floor.

Four places go beyond what the standard guidance asks for, and each one exists
for a reason worth knowing:

`unsafe_code = "forbid"` in the workspace lints closes the door entirely. The
SDK drives customer futures and serializes customer data, and neither job needs
unsafe code.

`unwrap_used`, `expect_used`, `panic`, and `indexing_slicing` all deny. A panic
inside the SDK kills the customer's Lambda invocation, and clippy's default
configuration allows all four. Return an error instead, and reach for `get()`
rather than an index.

`missing_docs = "deny"` and `unreachable_pub = "deny"` mean every public item
carries a `///` comment and nothing leaks into the public surface by accident.

The script closes and enforces the production dependency allowlist. The
dependency policy section below explains what adding to it involves.

## The quality gate

`make check` is the single entry point, and it must pass before you push. It
runs eight commands in order, and any one of them failing fails the gate:

| Command | Rejects |
| --- | --- |
| `cargo fmt --check` | any file whose formatting differs from rustfmt's default output |
| `cargo clippy --all-targets --all-features -- -D warnings` | every clippy warning, including the `pedantic` and `cargo` groups, across the library and its tests |
| `cargo test` | a failing unit test or doctest in the default feature set |
| `cargo test --all-features` | a failing test behind a feature gate (`test-util`, `replay-filter`), including the integration tests those features unlock |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` | a broken intra-doc link or malformed rustdoc markup anywhere in the documented surface, including feature-gated items |
| `cargo deny check` | a dependency license outside the allowlist, a crate carrying an unpatched RustSec advisory, or an unrecognized registry source |
| `sh scripts/check-direct-deps.sh` | a direct production dependency the allowlists do not name, in either the default-feature or the all-features graph |
| `sh scripts/check-msrv-consistency.sh` | a `rust-version` declaration — in any manifest, the `msrv` CI job, or the README — that disagrees with the root `Cargo.toml`'s |

Two of those are stricter than their defaults. Clippy normally reports warnings
and exits zero; `-D warnings` turns every one into an error, so a `pedantic`
suggestion blocks the gate exactly as a correctness lint does. Rustdoc likewise
warns by default; `RUSTDOCFLAGS="-D warnings"` makes a broken link a build
failure.

The lint configuration lives in `[workspace.lints]` in each workspace's
`Cargo.toml`. The SDK, the conformance handlers, and the examples are three
independent cargo workspaces, each with its own lint table. The tables share a
common core (the panic restriction lints, `missing_docs`, `unsafe_code =
"forbid"`) but already differ in the groups they enable and the extra lints they
carry. The root workspace adds `unreachable_pub`, `missing_debug_implementations`,
`unused_qualifications`, and the `cargo` clippy group; the compliance and examples
workspaces omit those. A lint policy change that you intend to apply across the
project must update all applicable `Cargo.toml` files, and the pull request should
say why.

`make check` deliberately leaves out one thing that CI still runs: it covers
the SDK workspace only. Both feature-gated surfaces are already inside the
gate — `cargo test --all-features` runs the `test-util` and `replay-filter`
tests, so there is no separate feature-testing step to remember.

`compliance/` and `examples/` are separate cargo workspaces whose heavier
dependency graph the root gate deliberately excludes so it stays fast. CI
compiles and lints both, which catches a public API change that breaks them.
Do the same locally when you change the public surface:

```sh
(cd compliance && cargo build --all-targets && cargo clippy --all-targets -- -D warnings)
(cd examples   && cargo build --all-targets && cargo clippy --all-targets -- -D warnings)
```

## Dependency policy

Eight crates make up the always-on direct production dependency set:
`aws-config`, `aws-sdk-lambda`, `lambda_runtime`, `serde`, `serde_json`,
`sha2`, `tokio`, and `tracing`. One more is approved as an optional,
feature-gated production dependency: `tracing-subscriber`, which only the
`replay-filter` feature pulls in, because [`ReplayFilterLayer`] implements
that crate's `Filter` trait and cannot exist without it. A default build
never compiles or ships it.

`scripts/check-direct-deps.sh` reads `cargo tree` twice and fails if either
graph steps outside its allowlist: the default-feature graph may contain only
the eight always-on crates, and the `--all-features` graph may add only the
approved optional crates. The second pass is what closes the loophole where
an unapproved crate rides in behind a feature flag, so the gate turns red the
moment someone adds a dependency of either kind.

Adding a crate to either list is a decision, not a routine change. The SDK
ships inside the customer's Lambda package and every always-on dependency
enlarges it, widens the advisory surface, and constrains what the customer can
resolve. An optional dependency avoids the package-size cost for consumers who
do not enable its feature, but it still widens the advisory surface and still
needs the same approval. Open a pull request that changes the allowlists in
the script on its own, explain what the crate buys that the current set
cannot, and expect the discussion to be about whether we take on the
dependency at all. Development dependencies carry none of
this weight: `deny.toml` sets `exclude-dev = true` and you may add one freely.

[`ReplayFilterLayer`]: https://github.com/aws/aws-durable-execution-sdk-rust/blob/alpha/src/tracing_layer.rs

`cargo deny check` handles the rest of the supply chain. It enforces a permissive
license allowlist, so a dependency arriving under an unlisted license fails, and
it checks the whole graph against the RustSec advisory database. `deny.toml`
carries a short list of advisory exceptions for transitive crates we cannot fix
ourselves while waiting for upstream releases. Each one needs a comment naming
the advisory and the reason. The `[bans]` section is deliberately empty because
cargo-deny's ban allowlist matches the entire transitive graph, which the AWS
SDK tree makes impractical; the script is what closes the allowlist instead.

## Tests

`cargo test` runs the unit tests that live beside the code in `src/` and every
doctest in the public documentation. Both count toward the gate, so a doc example
that stops compiling breaks the build. Mark an example `no_run` when it needs
AWS credentials or a live service, and keep it compiling rather than turning it
into a `text` block.

Write unit tests in a `#[cfg(test)] mod tests` block in the module under test.
The README describes `LocalRunner`, which drives a whole handler through
simulated invocations in process, and it is the right tool for a test that spans
several operations or needs to observe replay.

## The conformance suite

`compliance/` holds Lambda handlers in nine suites that a language-agnostic
runner drives against the live service. The runner starts each handler's
execution, then compares the recorded history against the language-independent
durable execution contract, which is how we know the Rust implementation
satisfies the service specification rather than merely passing its own tests.
Each suite has a SAM template at `compliance/template_<suite>.yaml`.
`compliance/template_smoke.yaml` is a single hello handler that checks a
deployment works at all, so the runner does not validate it and CI skips it.

Running the suite costs real time and real resources. A cold build of all
handlers produces roughly two gigabytes of artifacts, and each suite deploys a
CloudFormation stack of Lambda functions and runs executions that include
timers. CI splits the nine suites across parallel jobs; on a workstation, build
and run one suite at a time.

You need a Rust toolchain,
[cargo-lambda](https://www.cargo-lambda.info/), the
[AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html),
Python, an AWS account you can deploy to, and the ARN of a Lambda execution role
in that account. The quickest way to get one is the template this repository
ships — a one-time deploy with IAM-capable credentials; the test stacks
themselves never create IAM resources, so day-to-day runs need no IAM
permissions:

```sh
aws cloudformation deploy \
    --template-file scripts/test-execution-role.yaml \
    --stack-name durable-sdk-test-execution-role \
    --capabilities CAPABILITY_IAM
```

Its `RoleArn` output is what you pass as `ExecutionRoleArn` below. If you build
the role yourself instead, it needs the union of what all suites use:

- CloudWatch Logs write access (to receive handler output).
- `lambda:CheckpointDurableExecution` and `lambda:GetDurableExecutionState`
  (the two durable execution service actions the SDK calls).
- `lambda:InvokeFunction` on your test functions (the invoke suite and the
  chained-invoke examples call other functions in their stack).
- `lambda:SendDurableExecutionCallbackSuccess`, `...CallbackFailure`, and
  `...CallbackHeartbeat` (callback handlers).
- DynamoDB read and write access to the stack's Attempts table (the step and
  child suites deploy a DynamoDB table and their retry handlers record attempt
  counts through it).

CI pins cargo-lambda to an exact version; match it if you are chasing a
difference between a local run and a CI run.

Use cargo-lambda rather than `cargo build`. It builds through cargo-zigbuild,
which pins the glibc version the handler links against, so the binary starts on
the `provided.al2023` runtime whatever host built it. A natively linked binary
from a modern host dies at startup instead.

```sh
pip3 install cargo-lambda
pip install "git+https://github.com/aws/aws-durable-execution-conformance-tests.git@main#subdirectory=packages/aws-durable-execution-conformance-tests"

cd compliance
./build_examples.sh step          # build one suite; omit the argument to build every template

python -m aws_durable_execution_conformance_tests.app \
    --template template_step.yaml \
    --language rust \
    --suite step \
    --name conf-rust-step-local \
    --region us-west-2 \
    --parameter-overrides ExecutionRoleArn=arn:aws:iam::111122223333:role/your-lambda-execution-role \
    --history-dir history-step \
    --report junit \
    --report-file report-step
```

`build_examples.sh` skips the build when git HEAD is unchanged, the tree is
clean, and no source file is newer than the last run, so a repeat invocation
costs nothing. A dirty tree always rebuilds.

## The cloud example suite

`examples/` holds deployable examples in three families (basics, coordination,
external), and they double as an end-to-end test.
`cloud/run_cloud_tests.sh` invokes every directly-invokable example, drives the
callback examples with `aws lambda send-durable-execution-callback-success`, and
asserts that each execution reaches the terminal state its row in the script's
expectations table names. Some examples pass by reaching `FAILED`, which is why
the table is data rather than a blanket assertion.

The prerequisites match the conformance suite, plus `jq`. The whole run takes
tens of minutes because several examples wait on real timers; the CI job caps it
at 90.

Build and deploy all three family stacks, then run the harness:

```sh
cd examples
./build_examples.sh

ROLE=arn:aws:iam::111122223333:role/your-lambda-execution-role

sam deploy --template-file template_basics.yaml \
    --stack-name dex-rust-examples-basics \
    --resolve-s3 --region us-west-2 --no-confirm-changeset \
    --parameter-overrides ExecutionRoleArn=$ROLE

sam deploy --template-file template_coordination.yaml \
    --stack-name dex-rust-examples-coordination \
    --resolve-s3 --region us-west-2 --no-confirm-changeset \
    --parameter-overrides ExecutionRoleArn=$ROLE

sam deploy --template-file template_external.yaml \
    --stack-name dex-rust-examples-external \
    --resolve-s3 --region us-west-2 --no-confirm-changeset \
    --parameter-overrides ExecutionRoleArn=$ROLE

./cloud/run_cloud_tests.sh
```

The harness reads the stack names from environment variables
`STACK_BASICS`, `STACK_COORDINATION`, and `STACK_EXTERNAL`, defaulting to the
`dex-rust-examples-<family>` pattern shown above. Override them if you deploy
under different names. Delete the stacks when you finish; they hold live Lambda
functions. `examples/README.md` documents the family layout and the per-example
expectations.

## Raising a pull request

Branch from `alpha`, keep the change focused on one thing, and open the pull
request against `alpha`. Discuss anything substantial in an issue first so you do
not spend a weekend on a design we cannot take.

Run `make check` before you push, and run the two extra workspace builds when
you have changed the public API. A pull request that fails the gate wastes a CI
run and a reviewer's first pass.

CI runs four workflows on a pull request. `ci.yml` splits `make check` into a
`check` job and a `dependencies` job, verifies the declared minimum Rust
version with an `msrv` job, guards the public API surface with an
`external-types` job (no foreign type may reach the public API unless the
allowlist under `[package.metadata.cargo_check_external_types]` in
`Cargo.toml` names it deliberately) and a `semver` job (cargo-semver-checks
compares the pull request against its base branch, so a breaking API change
needs the matching version bump), proves every feature combination compiles
with a `feature-powerset` job (cargo-hack), then builds and lints the
`compliance` and `examples` workspaces.
`codeql.yml` runs static analysis over the Rust sources.
`conformance-tests.yml` runs the ten conformance suites when the SDK or the
handlers change. `cloud-tests.yml` deploys the example stacks and runs the cloud
harness. Everything must be green before merge. The conformance and cloud
workflows need credentials for the project's test account, which a pull request
from a fork cannot obtain, so a maintainer runs those for you.

A reviewer reads for the things a gate cannot see. Documentation on a new public
item should explain when to use it rather than restate its signature. A change
must preserve determinism, since the SDK claims operation IDs at the call site,
so any control flow deciding which operations to create may depend only on the
event and on results the service already recorded. A new error path returns an
error rather than panicking, and a new operation follows the builder shape the
existing operations use so the API stays predictable. The reviewer also asks
whether the change belongs in the SDK at all, or whether an example teaches it
better.

## Reporting bugs and requesting features

Use the GitHub issue tracker. Check the open and recently closed issues first.
A report is most useful when it names the SDK version, includes a reproducible
case, and describes anything unusual about the deployment.

Report a potential security issue through the
[AWS vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/)
rather than a public issue.

## Code of conduct

This project has adopted the
[Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct). The
[Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) answers most
questions, and opensource-codeofconduct@amazon.com takes the rest.

## Licensing

The [LICENSE](LICENSE) file carries the project's Apache-2.0 license. We will
ask you to confirm the licensing of your contribution.
