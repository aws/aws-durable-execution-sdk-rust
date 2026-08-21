#!/bin/sh
# check-msrv-consistency.sh: Mechanical gate for MSRV declaration consistency.
#
# The root Cargo.toml's rust-version is the single source of truth for the
# MSRV. But rust-version is also declared by every compliance handler and
# example crate (200+ manifests), quoted by the msrv CI job in ci.yml, and
# stated in the README. A bump that misses any of them leaves a stale,
# misleading declaration behind, exactly the drift this script exists to
# catch. It fails if:
#
#   1. Any Cargo.toml in the repository declares a rust-version different
#      from the root manifest's.
#   2. The msrv job in .github/workflows/ci.yml does not install and name
#      that same version.
#   3. README.md does not mention that same version.
#
# Wired into `make check`. See CONTRIBUTING.md, "MSRV" for the bump policy
# (pre-1.0 MSRV bumps ship as minor releases, never patches).
#
# POSIX sh, no bashisms.

set -e

FAIL=0

# Source of truth: the root manifest

MSRV=$(sed -n 's/^rust-version = "\(.*\)"$/\1/p' Cargo.toml)

if [ -z "$MSRV" ]; then
    echo "msrv-consistency FAILED: could not read rust-version from the root Cargo.toml." >&2
    exit 1
fi

# 1. Every manifest that declares rust-version must agree

# find(1) rather than a checked-in list: new crates under compliance/ or
# examples/ are covered automatically. target/ holds vendored copies of
# third-party manifests and is excluded.
MISMATCHES=$(find . -path ./target -prune -o -name Cargo.toml -print \
    | while IFS= read -r manifest; do
        grep -H '^rust-version = ' "$manifest" 2>/dev/null || true
    done \
    | grep -v "rust-version = \"$MSRV\"" || true)

if [ -n "$MISMATCHES" ]; then
    echo "msrv-consistency FAILED: manifests declaring a rust-version other than the root's ($MSRV):"
    echo ""
    echo "$MISMATCHES" | sed 's/^/  /'
    FAIL=1
fi

# 2. The msrv CI job must pin the same version

CI_WORKFLOW=.github/workflows/ci.yml

if ! grep -q "rustup toolchain install $MSRV" "$CI_WORKFLOW"; then
    echo "msrv-consistency FAILED: the msrv job in $CI_WORKFLOW does not install toolchain $MSRV."
    FAIL=1
fi

if ! grep -q "rustup default $MSRV" "$CI_WORKFLOW"; then
    echo "msrv-consistency FAILED: the msrv job in $CI_WORKFLOW does not default to toolchain $MSRV."
    FAIL=1
fi

# 3. The README must state the same version

if ! grep -q "$MSRV" README.md; then
    echo "msrv-consistency FAILED: README.md does not mention the MSRV ($MSRV)."
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "An MSRV bump must update every declaration in the same pull request:"
    echo "the root Cargo.toml, every Cargo.toml under compliance/ and examples/"
    echo "that declares rust-version, the msrv job in $CI_WORKFLOW, and README.md."
    echo "See CONTRIBUTING.md for the bump policy."
    exit 1
fi

echo "msrv-consistency OK: all rust-version declarations, the msrv CI job, and the README agree on $MSRV."
