# Each target runs its commands as recipe lines, one command per line.
# Make runs each line in its own shell and stops the target at the first
# line that exits non-zero. So any failing command fails its target.
# CONTRIBUTING.md describes the targets and their inputs.
#
# The first line of each check target checks its tools. scripts/tools.sh
# holds the tool pins and the `require` check, so a recipe line sources it
# and calls `require` in the same shell.
#
# This file uses only features of GNU make 3.81, the version macOS ships.

# The ci.yml jobs that run on a push to main, plus the tool pin check.
CI_TARGETS = pins check-sdk msrv dependencies external-types feature-powerset workspaces

# The cargo workspaces the workspaces job builds. CI's matrix passes one,
# for example `make workspaces WORKSPACES=conformance`.
WORKSPACES ?= conformance examples
WORKSPACE_TARGETS = $(addprefix workspace-,$(WORKSPACES))

.PHONY: check ci $(CI_TARGETS) $(WORKSPACE_TARGETS) \
	test-role conformance conformance-build conformance-run cloud-tests

# The quality gate for every change: the check-sdk, dependencies and msrv
# jobs of ci.yml. It stops at the first failing target; `make -k check`
# runs the other targets too.
check: check-sdk dependencies msrv

# Every target in CI_TARGETS, with the environment ci.yml sets for every
# job. `make check` does not set it, so a contributor's incremental build
# cache survives `make check`. A variable set on the sub-make's command
# line is exported to its recipes. -k runs every target even after one
# fails, as CI runs its jobs independently, and the sub-make still exits
# non-zero when any target failed.
ci:
	$(MAKE) -k RUSTFLAGS='-D warnings' CARGO_PROFILE_DEV_DEBUG=0 $(CI_TARGETS)

# One target per ci.yml job. ci.yml calls these.

# Fails when scripts/tools.sh and a workflow name different versions of a
# pinned tool.
pins:
	. scripts/tools.sh && check_pins

check-sdk:
	. scripts/tools.sh && require cargo rustfmt clippy
	cargo fmt --check
	cargo clippy --workspace --all-targets --all-features -- -D warnings
	cargo test --workspace
	cargo test --workspace --all-features
	RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features
	sh scripts/check-msrv-consistency.sh

msrv:
	. scripts/tools.sh && require cargo msrv-toolchain
	sh scripts/check-msrv.sh

dependencies:
	. scripts/tools.sh && require cargo cargo-deny
	cargo deny --all-features check
	sh scripts/check-direct-deps.sh

# CI's workflow-wide RUSTFLAGS='-D warnings' belongs to the stable lint
# gate. A pinned nightly's extra lints must not fail the API check, so the
# command clears RUSTFLAGS.
external-types:
	. scripts/tools.sh && require cargo nightly-toolchain cargo-check-external-types
	. scripts/tools.sh && RUSTFLAGS='' cargo "+$$NIGHTLY_TOOLCHAIN" check-external-types --all-features

feature-powerset:
	. scripts/tools.sh && require cargo cargo-hack
	cargo hack check --feature-powerset --all-targets

workspaces: $(WORKSPACE_TARGETS)

# A static pattern rule, because make skips pattern rules for phony targets.
$(WORKSPACE_TARGETS): workspace-%:
	. scripts/tools.sh && require cargo clippy
	cd $* && cargo build --all-targets
	cd $* && cargo clippy --all-targets -- -D warnings

# Targets that deploy to AWS. They need TEST_ACCOUNT_ID.
test-role:
	sh scripts/test-role.sh
conformance:
	sh scripts/conformance.sh all
conformance-build:
	sh scripts/conformance.sh build
conformance-run:
	sh scripts/conformance.sh run
cloud-tests:
	sh scripts/cloud-tests.sh run
