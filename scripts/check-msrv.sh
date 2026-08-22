#!/bin/sh
# check-msrv.sh: The MSRV leg of the local quality gate.
#
# CI's msrv job compiles the SDK on the declared minimum supported Rust
# version with warnings denied, so code that leans on a newer compiler --
# or that trips a warning only the older compiler emits -- fails CI even
# when every stable-toolchain leg passes. Without this leg, `make check`
# on the contributor's default toolchain can stay green while CI's msrv
# job goes red (and the reverse). This script runs the same check locally
# so `make check` sees what CI sees.
#
# The version is not hard-coded here: it is read from the root
# Cargo.toml's rust-version, the same source of truth that
# scripts/check-msrv-consistency.sh enforces, so an MSRV bump propagates
# to this leg by updating that one declaration.
#
# Wired into `make check`. POSIX sh, no bashisms.

set -e

# Source of truth: the root manifest

MSRV=$(sed -n 's/^rust-version = "\(.*\)"$/\1/p' Cargo.toml)

if [ -z "$MSRV" ]; then
    echo "msrv-check FAILED: could not read rust-version from the root Cargo.toml." >&2
    exit 1
fi

# Graceful guard: name the fix rather than letting cargo fail obscurely.

if ! command -v rustup >/dev/null 2>&1; then
    echo "msrv-check FAILED: rustup is not on PATH, so the MSRV toolchain ($MSRV) cannot be selected." >&2
    echo "" >&2
    echo "Install rustup (https://rustup.rs), then run:" >&2
    echo "" >&2
    echo "    rustup toolchain install $MSRV" >&2
    exit 1
fi

if ! rustup toolchain list | grep -q "^$MSRV-"; then
    echo "msrv-check FAILED: Rust toolchain $MSRV is not installed." >&2
    echo "" >&2
    echo "make check compiles the SDK on the minimum supported Rust version" >&2
    echo "($MSRV) with warnings denied, matching CI's msrv job. Install the" >&2
    echo "toolchain with:" >&2
    echo "" >&2
    echo "    rustup toolchain install $MSRV" >&2
    exit 1
fi

echo "msrv-check: RUSTFLAGS=\"-D warnings\" cargo +$MSRV check --all-targets --all-features"
RUSTFLAGS="-D warnings" cargo "+$MSRV" check --all-targets --all-features
echo "msrv-check OK: the SDK compiles warning-free on Rust $MSRV."
