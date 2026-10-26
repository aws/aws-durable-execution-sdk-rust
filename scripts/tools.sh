# tools.sh: the tool versions CI pins, and the checks that use them.
# Sourced by the other scripts, never run on its own. POSIX sh.
#
# The workflows install these tools themselves. GitHub reads an action
# input, a cache key, or a job name only as literal text, so a workflow
# cannot read the pins from this file without an extra step in every job.
# So each pin is written twice, here and in the workflow, and check_pins
# (run by `make ci`) fails when the two disagree. That is one function
# instead of a load-the-pins step per job.
#
# Two versions are not pinned here. Stable Rust is not pinned at all,
# because CI installs the newest stable on every run. The MSRV comes from
# the root Cargo.toml, and scripts/check-msrv-consistency.sh checks the
# workflow against it.

CARGO_HACK_VERSION=0.6.45
CARGO_CHECK_EXTERNAL_TYPES_VERSION=0.5.0
CARGO_DENY_VERSION=0.20.2
CARGO_LAMBDA_VERSION=1.8.6
NIGHTLY_TOOLCHAIN=nightly-2026-03-20

# require TOOL...
# Checks every TOOL before anything runs. A tool that is missing, or that
# reports a version other than CI's pin, is listed with the command that
# installs the right one, and the calling script exits with status 1.
require() {
    tools_failed=0
    for tool in "$@"; do
        case "$tool" in
            cargo-hack)
                require_version cargo-hack "$CARGO_HACK_VERSION" \
                    "cargo install --locked cargo-hack --version $CARGO_HACK_VERSION" \
                    cargo hack --version ;;
            cargo-check-external-types)
                require_version cargo-check-external-types "$CARGO_CHECK_EXTERNAL_TYPES_VERSION" \
                    "cargo install --locked cargo-check-external-types --version $CARGO_CHECK_EXTERNAL_TYPES_VERSION" \
                    cargo-check-external-types --version ;;
            cargo-deny)
                require_version cargo-deny "$CARGO_DENY_VERSION" \
                    "cargo install --locked cargo-deny --version $CARGO_DENY_VERSION" \
                    cargo deny --version ;;
            cargo-lambda)
                require_version cargo-lambda "$CARGO_LAMBDA_VERSION" \
                    "pip install \"cargo-lambda==$CARGO_LAMBDA_VERSION\"" \
                    cargo lambda --version ;;
            nightly-toolchain)
                require_toolchain "$NIGHTLY_TOOLCHAIN" ;;
            msrv-toolchain)
                require_toolchain "$(sed -n 's/^rust-version = "\(.*\)"$/\1/p' Cargo.toml)" ;;
            rustfmt)
                cargo fmt --version > /dev/null 2>&1 ||
                    tool_missing "rustfmt is not installed" "rustup component add rustfmt" ;;
            clippy)
                cargo clippy --version > /dev/null 2>&1 ||
                    tool_missing "clippy is not installed" "rustup component add clippy" ;;
            *)
                command -v "$tool" > /dev/null 2>&1 ||
                    tool_missing "$tool is not installed" "$(install_hint "$tool")" ;;
        esac
    done
    if [ "$tools_failed" -ne 0 ]; then
        echo "Nothing was run. Install the tools above, then run the target again." >&2
        exit 1
    fi
}

install_hint() {
    case "$1" in
        cargo) echo "install Rust with rustup: https://rustup.rs" ;;
        timeout) echo "install GNU coreutils (macOS: brew install coreutils, then put its gnubin directory on PATH)" ;;
        aws) echo "see https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" ;;
        sam) echo "see https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html" ;;
        python3) echo "install Python 3 from https://www.python.org/downloads/" ;;
        *) echo "install $1 with your package manager" ;;
    esac
}

tool_missing() {
    echo "$1. Install it with:" >&2
    echo "    $2" >&2
    tools_failed=1
}

# require_version BINARY VERSION HINT COMMAND...
# COMMAND prints "<name> <version> ..." on its first line. It is the form
# the tool accepts: cargo-lambda answers --version only as a cargo
# subcommand, cargo-check-external-types only when run directly.
require_version() {
    binary=$1
    want=$2
    hint=$3
    shift 3
    if ! command -v "$binary" > /dev/null 2>&1; then
        tool_missing "$binary is not installed; CI uses $want" "$hint"
        return
    fi
    have=$("$@" 2> /dev/null | awk 'NR == 1 { print $2 }')
    if [ "$have" != "$want" ]; then
        tool_missing "$binary is version ${have:-unknown}; CI uses $want" "$hint"
    fi
}

require_toolchain() {
    if ! command -v rustup > /dev/null 2>&1; then
        tool_missing "rustup is not installed, so toolchain $1 cannot be selected" "see https://rustup.rs"
    elif ! rustup toolchain list | grep -q "^$1-"; then
        tool_missing "Rust toolchain $1 is not installed" "rustup toolchain install $1 --profile minimal"
    fi
}

# check_pins
# Fails when a workflow names a version of a pinned tool other than the pin
# above, or does not name the tool at all. Every occurrence counts, so a
# workflow that bumps one of two mentions also fails.
check_pins() {
    pins_failed=0
    check_pin 'cargo-hack@' "$CARGO_HACK_VERSION" .github/workflows/ci.yml
    check_pin 'cargo-deny@' "$CARGO_DENY_VERSION" .github/workflows/ci.yml
    check_pin 'cargo-check-external-types --version ' "$CARGO_CHECK_EXTERNAL_TYPES_VERSION" .github/workflows/ci.yml
    check_pin 'cargo-check-external-types-' "$CARGO_CHECK_EXTERNAL_TYPES_VERSION" .github/workflows/ci.yml
    check_pin 'nightly-' "${NIGHTLY_TOOLCHAIN#nightly-}" .github/workflows/ci.yml
    check_pin 'cargo-lambda==' "$CARGO_LAMBDA_VERSION" \
        .github/workflows/conformance-tests.yml .github/workflows/cloud-tests.yml
    if [ "$pins_failed" -ne 0 ]; then
        echo "Update the workflow and scripts/tools.sh together." >&2
        return 1
    fi
    echo "tool-pins OK: the workflows and scripts/tools.sh agree."
}

# check_pin PREFIX VERSION FILE...
check_pin() {
    prefix=$1
    want=$2
    shift 2
    for file in "$@"; do
        found=$(grep -oE "${prefix}[0-9][0-9.-]*[0-9]" "$file" | sort -u)
        if [ "$found" != "$prefix$want" ]; then
            echo "tool-pins FAILED: $file names '${found:-nothing}' where scripts/tools.sh pins '$prefix$want'." >&2
            pins_failed=1
        fi
    done
}
