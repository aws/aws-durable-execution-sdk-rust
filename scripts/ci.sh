#!/bin/sh
# ci.sh: the commands of every ci.yml job that runs on a push to main.
# Each ci.yml job calls its make target, and `make ci` runs all of them, so
# a local run executes the same commands as CI. ci.yml's semver job runs
# only on pull requests, so it is not here.
#
# Usage:
#   sh scripts/ci.sh JOB...   run the named jobs (make check, make <job>)
#   sh scripts/ci.sh ci       run every job with CI's environment (make ci)
#
# Jobs: check-sdk, msrv, dependencies, external-types, feature-powerset,
# workspaces (named after the ci.yml jobs), and pins (the tool-version
# consistency check in scripts/tools.sh).
#
# The tools of every named job are checked before the first job starts.
# Inside a job, the first failing command ends the job, as a failing CI
# step does. The next job still runs, because CI runs its jobs
# independently. A summary follows the last job, and the exit status is 1
# when any job failed.
#
# No command here runs under a time limit, so these jobs need no GNU
# timeout. A caller that nobody watches sets the limit itself, for example
# `timeout 3600 make ci`. In CI, each job's timeout-minutes is the limit.
#
# POSIX sh, no bashisms.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/tools.sh

# The workspaces job builds these cargo workspaces. CI's matrix passes one.
WORKSPACES=${WORKSPACES:-conformance examples}

check_sdk() (
    set -ex
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test
    cargo test --all-features
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
    sh scripts/check-msrv-consistency.sh
)

msrv() (
    set -ex
    sh scripts/check-msrv.sh
)

dependencies() (
    set -ex
    cargo deny --all-features check
    sh scripts/check-direct-deps.sh
)

external_types() (
    set -ex
    # CI's workflow-wide RUSTFLAGS="-D warnings" belongs to the stable lint
    # gate. A pinned nightly's extra lints must not fail the API check.
    RUSTFLAGS='' cargo "+$NIGHTLY_TOOLCHAIN" check-external-types --all-features
)

feature_powerset() (
    set -ex
    cargo hack check --feature-powerset --all-targets
)

workspaces() (
    set -ex
    for ws in $WORKSPACES; do
        (cd "$ws" && cargo build --all-targets && cargo clippy --all-targets -- -D warnings)
    done
)

# job_tools JOB: the tools JOB needs, as names scripts/tools.sh knows.
job_tools() {
    case "$1" in
        pins) echo "" ;;
        check-sdk) echo "cargo rustfmt clippy" ;;
        msrv) echo "cargo msrv-toolchain" ;;
        dependencies) echo "cargo cargo-deny" ;;
        external-types) echo "cargo nightly-toolchain cargo-check-external-types" ;;
        feature-powerset) echo "cargo cargo-hack" ;;
        workspaces) echo "cargo clippy" ;;
        *) echo "ci.sh: unknown job '$1'" >&2; exit 2 ;;
    esac
}

run_job() {
    case "$1" in
        pins) check_pins ;;
        check-sdk) check_sdk ;;
        msrv) msrv ;;
        dependencies) dependencies ;;
        external-types) external_types ;;
        feature-powerset) feature_powerset ;;
        workspaces) workspaces ;;
    esac
}

if [ "$#" -eq 0 ]; then
    echo "usage: sh scripts/ci.sh JOB... | ci" >&2
    exit 2
fi

if [ "$*" = ci ]; then
    # ci.yml sets these for every job. `make check` does not, so a
    # contributor's incremental build cache survives it.
    export RUSTFLAGS="-D warnings" CARGO_PROFILE_DEV_DEBUG=0
    set -- pins check-sdk msrv dependencies external-types feature-powerset workspaces
fi

tools=""
for job in "$@"; do
    tools="$tools $(job_tools "$job")" || exit 2
done
# shellcheck disable=SC2086
require $tools

summary=""
failed=0
for job in "$@"; do
    echo
    echo "=== $job ==="
    start=$(date +%s)
    if run_job "$job"; then
        result=PASS
    else
        result=FAIL
        failed=1
    fi
    summary="$summary$(printf '%-5s %-17s %6ss' "$result" "$job" "$(($(date +%s) - start))")
"
done

echo
echo "=== summary ==="
printf '%s' "$summary"
exit "$failed"
