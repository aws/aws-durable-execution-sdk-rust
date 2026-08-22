#!/bin/sh
# Build script for Rust Durable Execution conformance test handlers.
#
# conformance/ is a single cargo WORKSPACE (conformance/Cargo.toml): every
# handler is a member, so the whole suite shares ONE target dir
# (conformance/target) and ONE resolve of the SDK + aws-sdk graph. This script
# runs a single `cargo lambda build` over the requested members and stages each
# handler's binary into publish/<handler>/bootstrap for SAM deployment on the
# provided.al2023 runtime.
#
# Fast-build tuning lives in conformance/Cargo.toml [profile.release] (handlers
# opt-level 1, deps opt-level 3 via package."*"), scoped to this workspace:
# the SDK's own release profile is untouched.
#
# Why cargo-lambda and not a plain `cargo build`: a natively linked binary
# requires the glibc symbol versions of the machine that built it. The
# provided.al2023 runtime ships glibc 2.34, so a binary built on a newer host
# (for example the GitHub ubuntu-latest runner, glibc 2.39) dies at startup
# with `/lib64/libc.so.6: version 'GLIBC_2.39' not found` and the execution
# fails with `Runtime exited with error: exit status 1`. cargo-lambda builds
# through cargo-zigbuild, which pins the glibc version the binary links
# against, so the SAME command produces a runtime-compatible artifact on every
# host: a local pass and a CI pass mean the same thing. cargo-lambda is the
# route documented in the Lambda Developer Guide:
# https://docs.aws.amazon.com/lambda/latest/dg/rust-package.html
#
# Requires:
#   - Rust toolchain
#   - cargo-lambda (pip3 install cargo-lambda, or cargo install cargo-lambda)
#
# Usage:
#   ./build_examples.sh [operation...]
#
# Operations (default: all found in this directory):
#   smoke, step, wait, callback, child, invoke, parallel,
#   wait_for_callback, wait_for_condition, map, history,
#   nondeterminism, combinator
#
# Examples:
#   ./build_examples.sh smoke          # build only the smoke suite
#   ./build_examples.sh step wait      # build two suites
#   ./build_examples.sh                # build everything
#
# Skip-if-unchanged: a second invocation with the same operations, an
# unchanged git HEAD, a clean working tree, all bootstraps present, and no
# source file newer than the last build is a no-op. A dirty tree, a new
# commit, a touched source file, or a missing bootstrap always rebuilds.

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PUBLISH_DIR="$SCRIPT_DIR/publish"
LAMBDA_DIR="$SCRIPT_DIR/target/lambda"
STAMP="$PUBLISH_DIR/.built-at"

cd "$SCRIPT_DIR"

if ! command -v cargo-lambda > /dev/null 2>&1; then
    cat >&2 << 'EOF'
Error: cargo-lambda is not installed.

It is the single build path for this workspace, locally and in CI: it pins the
glibc version the handler binaries link against so they start on the
provided.al2023 runtime regardless of which host built them.

Install it with one of:
    pip3 install cargo-lambda
    cargo install cargo-lambda
EOF
    exit 1
fi

# Resolve the requested operations
if [ $# -gt 0 ]; then
    operations="$*"
else
    # `build` is scratch the conformance runner leaves behind (gitignored),
    # not a suite; discovering it as an operation fails the cargo build.
    # conformance_ext is the Python runner-extension package, not a handler
    # suite: exclude it from auto-discovery.
    operations=$(find . -maxdepth 1 -mindepth 1 -type d \
        ! -name publish ! -name src ! -name target ! -name build ! -name '.*' \
        ! -name conformance_ext \
        -exec basename {} \;)
fi
# Stable signature (sorted, single-space-joined) for the skip stamp.
ops_sig=$(printf '%s\n' $operations | sort | tr '\n' ' ')

# Collect the handler package + directory list for these operations
pkgs=""
handler_dirs=""
for op in $operations; do
    [ -d "$op" ] || { echo "Error: unknown operation '$op'." >&2; exit 1; }
    for dir in "$op"/*/; do
        [ -d "$dir" ] || continue
        pkgs="$pkgs -p $(basename "$dir")"
        handler_dirs="$handler_dirs $dir"
    done
done
[ -n "$handler_dirs" ] || { echo "Error: no handlers found for: $operations" >&2; exit 1; }

# Skip-if-unchanged guard
sha=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo nogit)
dirty=$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null || true)

should_skip() {
    # Never skip on a dirty tree.
    [ -z "$dirty" ] || return 1
    # Need a prior stamp for the same SHA and the same operation set.
    [ -f "$STAMP" ] || return 1
    [ "$(sed -n '1p' "$STAMP")" = "$sha" ] || return 1
    [ "$(sed -n '2p' "$STAMP")" = "$ops_sig" ] || return 1
    # Every requested bootstrap must actually exist.
    for dir in $handler_dirs; do
        [ -f "$PUBLISH_DIR/$(basename "$dir")/bootstrap" ] || return 1
    done
    # No handler source, SDK source, or manifest newer than the stamp.
    newer=$(find "$SCRIPT_DIR" "$SCRIPT_DIR/../src" "$SCRIPT_DIR/../Cargo.toml" \
        \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$STAMP" \
        ! -path "$SCRIPT_DIR/target/*" ! -path "$SCRIPT_DIR/publish/*" \
        2>/dev/null | head -1)
    [ -z "$newer" ] || return 1
    return 0
}

if should_skip; then
    echo "Nothing changed since last build (HEAD $sha, clean tree, ops: $ops_sig)."
    echo "Skipping. (delete $STAMP or touch a source file to force a rebuild.)"
    exit 0
fi

# Single shared-workspace build over just the requested members
echo "Building operations: $operations"
# cargo-lambda stages every binary it finds in the target dir into
# target/lambda/<bin>/bootstrap, so a leftover binary from an earlier build of
# other members would look like fresh output. Clear the staging dir first: only
# what this invocation builds may be staged.
rm -rf "$LAMBDA_DIR"
# --x86-64 is explicit rather than host-derived so an aarch64 workstation does
# not silently produce arm64 binaries for the x86_64 SAM functions.
# shellcheck disable=SC2086
cargo lambda build --release --x86-64 $pkgs

# Stage each handler binary into publish/<handler>/bootstrap
for dir in $handler_dirs; do
    handler=$(basename "$dir")
    op=$(basename "$(dirname "$dir")")
    out="$PUBLISH_DIR/$handler"

    # The binary name is the crate name with '-' normalized to '_'.
    bin_name=$(grep '^name' "$dir/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/' | tr '-' '_')
    src_bin="$LAMBDA_DIR/$bin_name/bootstrap"
    if [ ! -f "$src_bin" ]; then
        echo "Error: built binary not found for $handler at $src_bin" >&2
        exit 1
    fi

    mkdir -p "$out"
    cp "$src_bin" "$out/bootstrap"

    # SAM builds provided.al2023 functions through BuildMethod: makefile.
    # Logical IDs are the PascalCase form of the handler dir name.
    logical_id=$(echo "$handler" | awk -F_ '{for (i = 1; i <= NF; i++) printf "%s%s", toupper(substr($i, 1, 1)), substr($i, 2)}')
    cat > "$out/Makefile" << MKEOF
.PHONY: build-$logical_id

build-$logical_id:
	cp -r . \$(ARTIFACTS_DIR)/
	rm -f \$(ARTIFACTS_DIR)/Makefile
MKEOF
    echo "Staged $op/$handler -> publish/$handler/bootstrap"
done

# Record the build stamp (SHA + operation signature); mtime = now
mkdir -p "$PUBLISH_DIR"
{ echo "$sha"; echo "$ops_sig"; } > "$STAMP"

echo "Build complete. Binaries in $PUBLISH_DIR"
