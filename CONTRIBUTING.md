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
[Conventional Commits](https://www.conventionalcommits.org/). Nothing enforces
this mechanically, so a reviewer checks the subject line instead. Keep the subject under 50
characters, write it in the imperative mood, drop the trailing period, and wrap
body text at 72 characters.

The crate declares `rust-version = "1.94.1"` and edition 2024. Most CI jobs
build on stable, but a dedicated `msrv` job in `ci.yml` runs `cargo check
--all-targets --all-features` on the declared minimum with warnings denied,
so reaching for a feature newer than 1.94.1 — or tripping a warning only the
older compiler emits — fails that leg rather than silently raising the
true minimum above the declared one. `make check` runs the same check
locally through `scripts/check-msrv.sh`, which reads the version from the
root `Cargo.toml` and asks you to `rustup toolchain install` the minimum if
it is missing, so the local gate sees what CI's `msrv` job sees. Bumping the minimum supported Rust
version is an intentional, documented change: while the crate is pre-1.0, an
MSRV bump ships as a minor release (0.x → 0.x+1), never as a patch, and the
change must update every declaration in the same pull request: the root
`Cargo.toml` is the source of truth, and every `Cargo.toml` under
`conformance/` and `examples/` that declares `rust-version` (200+ manifests),
the `msrv` CI job, and the README must agree with it.
`scripts/check-msrv-consistency.sh`, wired into `make check`, enforces this
mechanically, so a bump that misses a declaration fails `make check`
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

## `make check`

`make check` is the single entry point, and it must pass before you push. It
runs nine commands in order, and any one of them failing fails the run:

| Command | Rejects |
| --- | --- |
| `cargo fmt --check` | any file whose formatting differs from rustfmt's default output |
| `cargo clippy --all-targets --all-features -- -D warnings` | every clippy warning, including the `pedantic` and `cargo` groups, across the library and its tests |
| `cargo test` | a failing unit test or doctest in the default feature set |
| `cargo test --all-features` | a failing test behind a feature gate (`test-util`, `replay-filter`), including the integration tests those features unlock |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` | a broken intra-doc link or malformed rustdoc markup anywhere in the documented surface, including feature-gated items |
| `cargo deny check` | a dependency license outside the allowlist, a crate carrying an unpatched RustSec advisory, or an unrecognized registry source |
| `sh scripts/check-direct-deps.sh` | a direct production dependency the allowlists do not name, in either the default-feature or the all-features graph |
| `sh scripts/check-msrv-consistency.sh` | a `rust-version` declaration (in any manifest, the `msrv` CI job, or the README) that disagrees with the root `Cargo.toml`'s |
| `sh scripts/check-msrv.sh` | code that does not compile warning-free on the declared minimum supported Rust version, matching CI's `msrv` job; if that toolchain is not installed, it fails with a `rustup toolchain install` hint rather than running on the wrong compiler |

Two of those are stricter than their defaults. Clippy normally reports warnings
and exits zero; `-D warnings` turns every one into an error, so a `pedantic`
suggestion blocks `make check` exactly as a correctness lint does. Rustdoc likewise
warns by default; `RUSTDOCFLAGS="-D warnings"` makes a broken link a build
failure.

The lint configuration lives in `[workspace.lints]` in each workspace's
`Cargo.toml`. The SDK, the conformance handlers, and the examples are three
independent cargo workspaces, each with its own lint table. The tables share a
common core (the panic restriction lints, `missing_docs`, `unsafe_code =
"forbid"`) but already differ in the groups they enable and the extra lints they
carry. The root workspace adds `unreachable_pub`, `missing_debug_implementations`,
`unused_qualifications`, and the `cargo` clippy group; the conformance and examples
workspaces omit those. A lint policy change that you intend to apply across the
project must update all applicable `Cargo.toml` files, and the pull request should
say why.

`make check` deliberately leaves out one thing that CI still runs: it covers
the SDK workspace only. Both feature-gated surfaces are already inside it:
`cargo test --all-features` runs the `test-util` and `replay-filter`
tests, so there is no separate feature-testing step to remember.

`conformance/` and `examples/` are separate cargo workspaces whose heavier
dependency graph `make check` deliberately excludes so it stays fast. CI
compiles and lints both, which catches a public API change that breaks them.
Do the same locally when you change the public surface:

```sh
(cd conformance && cargo build --all-targets && cargo clippy --all-targets -- -D warnings)
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
an unapproved crate rides in behind a feature flag, so `make check` fails the
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

[`ReplayFilterLayer`]: https://github.com/aws/aws-durable-execution-sdk-rust/blob/main/src/tracing_layer.rs

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
doctest in the public documentation. Both count toward `make check`, so a doc
example that stops compiling breaks the build. Mark an example `no_run` when it needs
AWS credentials or a live service, and keep it compiling rather than turning it
into a `text` block.

Write unit tests in a `#[cfg(test)] mod tests` block in the module under test.
The README describes `LocalRunner`, which drives a whole handler through
simulated invocations in process, and it is the right tool for a test that spans
several operations or needs to observe replay.

## The conformance suite

`conformance/` holds Lambda handlers in nine suites that a language-agnostic
runner drives against the live service. The runner starts each handler's
execution, then compares the recorded history against the language-independent
durable execution contract, which is how we know the Rust implementation
satisfies the service specification rather than merely passing its own tests.
Each suite has a SAM template at `conformance/template_<suite>.yaml`.
`conformance/template_smoke.yaml` is a single hello handler that checks a
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
ships: a one-time deploy with IAM-capable credentials; the test stacks
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

cd conformance
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
you have changed the public API. A pull request that fails `make check` wastes a CI
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
`conformance` and `examples` workspaces.
`codeql.yml` runs static analysis over the Rust sources.
`conformance-tests.yml` runs the ten conformance suites when the SDK or the
handlers change. `cloud-tests.yml` deploys the example stacks and runs the cloud
harness. Everything must be green before merge. The conformance and cloud
workflows need credentials for the project's test account, which a pull request
from a fork cannot obtain, so a maintainer runs those for you.

A reviewer reads for the things no automated check can see. Documentation on a new public
item should explain when to use it rather than restate its signature. A change
must preserve determinism, since the SDK claims operation IDs at the call site,
so any control flow deciding which operations to create may depend only on the
event and on results the service already recorded. A new error path returns an
error rather than panicking, and a new operation follows the builder shape the
existing operations use so the API stays predictable. The reviewer also asks
whether the change belongs in the SDK at all, or whether an example teaches it
better.

### Rules for every change

These rules apply to every package in this repository, to people and to coding agents. A
package can add its own rules in its `AGENTS.md` or README. Many pull requests are now
written with AI help, and AI-written changes tend to add logic. These rules keep each
change as small as the problem.

#### 1. Keep it simple

1. Do not add a mechanism only to make a rare case faster. Correctness is required. Speed
   in a rare case is not.
2. Do not add configuration that nobody asked for.
3. For a new feature, prefer the simpler option, even if it loses a minor capability of a
   more complex option. An existing feature has users, so removing a capability of it is
   a breaking change and follows the release rules instead.

#### 2. Simplify before you add

1. Every new condition, branch or option adds cases that readers and tests must cover.
2. A bug or a missing use case often exists because the existing logic is more complex
   than the problem. A special case hides another case, or two paths do the same job in
   different ways.
3. So the first question for a bug fix or a feature is: can a simpler version of the
   existing logic cover it?
4. A change that fixes the bug or covers the use case and removes or merges logic is
   preferred to a change that adds logic. This holds even when the simpler change
   touches more lines.
5. If a change must add logic, its description states which simpler options were
   considered and why each one does not work.

#### 3. Write a design note before the code

1. A review finds a wrong approach only after the code exists. Rewriting code costs more
   than rewriting a note.
2. So a new feature, a new option or a change to a public API starts with a design note.
   A bug fix that changes a public API needs one too.
3. The note states, in this order:
   1. the problem, in one or two sentences, without a solution;
   2. the cause;
   3. the options considered, including a simplification of existing logic (rule 2);
   4. the chosen option and the reason;
   5. what the change removes, if anything.
4. A maintainer agrees to the note before the code is written. The note goes in the
   issue or in the pull request description.

#### 4. Every branch has a test that fails without it

1. Line coverage does not prove that a branch matters. A test can run a branch whose
   result changes no assertion.
2. So for each condition and each branch in a change, some test must fail when the branch
   is removed or inverted.
3. The reviewer asks of each condition: which test fails if this is deleted? A branch
   without an answer is removed, or it gets a test.
4. Defensive code for a case that cannot happen has no such test. So it is removed.

#### 5. A bug fix starts with a failing test

1. A test that reproduces the bug proves that the bug exists. The same test proves that
   the fix removes it.
2. So a bug fix first adds a test that fails on the current code. The fix then makes it
   pass. The pull request description shows the test failing before the fix, or the
   commit order shows it.
3. The description names the root cause, not only the symptom. The fix changes the code
   at the cause. Example: a value arrives in an unexpected unit. The fix converts it once,
   where it enters the code. It does not change every place that reads it.
4. A change that edits the expectation of an existing test lists the edited test and the
   reason in its description.

#### 6. Mark unverified assumptions

1. A change often depends on behavior outside the package: the service, Lambda, the
   language runtime or another library.
2. An assumption about that behavior can be wrong, and the tests then pass anyway,
   because a fake of the outside component encodes the same assumption.
3. So every assumption that a change depends on cites its evidence: a run against the
   real component, a document or a test against the real component.
4. An assumption without evidence is marked "unverified" in the pull request description.
   The reviewer decides whether a check against the real component is needed before
   approval.

#### 7. List the decisions that the task did not cover

1. An issue or a design note never covers every detail. So the contributor makes some
   decisions alone.
2. A decision that is hidden in the code cannot be reviewed.
3. So the pull request description lists every decision that the issue or the design
   note did not cover, one line each, with the reason. Examples: a default value, an
   error message, a changed edge case, a skipped test.
4. A maintainer accepts or rejects each listed decision.

#### 8. Repeated review findings mean a redesign

A review finding is a change request or a blocking issue from a reviewer, a person or an
agent. A round is one review of a revision, followed by the contributor's changes.

1. A correct and simple change usually passes review after one round of fixes.
2. Sometimes a reviewer finds new problems in the code that fixed the previous findings.
   Each fix then adds a case, a condition or a special path.
3. This pattern has a few common causes:
   - the feature itself is wrong;
   - the implementation is wrong or too complex;
   - the change fixes a symptom instead of the cause.
4. More fixes on the same design add complexity and break rule 1.
5. So the trigger is this: the contributor addressed every finding of a round, and the
   next review finds new blocking issues. At the trigger, the contributor stops fixing.
6. The contributor then returns to the design phase:
   1. State the problem again in one or two sentences, without the current solution.
   2. Find the cause. Ask why the problem happens until the answer is a fact about the
      system, not about the code that the change added.
   3. List the review findings of all rounds. Check whether one cause explains several of
      them.
   4. Consider simpler options, including not adding the feature.
   5. Write the chosen design and the reason in the pull request, and get the reviewer's
      agreement before writing new code.
7. The new design may keep parts of the old code. It may also replace the change
   completely. A smaller change that solves the cause is better than a larger change
   that passes review.
8. A coding agent that hits the trigger stops. It reports the findings of all rounds and
   its analysis of the cause, and it waits for a decision before it continues.

#### 9. Reviewers enforce these rules

1. A reviewer, a person or an agent, checks every change against these rules and the
   package's own rules, not only for correctness.
2. A change can be correct and still break a rule. Examples: it adds logic where a
   simplification works (rule 2), it has no design note (rule 3), or it has a branch that
   no test needs (rule 4).
3. Change requests on such a change produce a better version of the wrong approach. So
   for a change that breaks a rule, the reviewer does not ask for detailed code changes.
4. Instead, the reviewer pushes back on the approach. The review names the rule, states
   how the change breaks it, and asks the contributor to return to the design phase
   (rule 8, step 6). The reviewer can suggest a simpler direction.
5. The reviewer approves only a change that follows every rule. A reviewer who wants an
   exception names the rule and the reason in the review, and a maintainer decides.

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
