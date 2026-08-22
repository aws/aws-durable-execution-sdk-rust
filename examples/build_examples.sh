#!/bin/sh
# Build script for the Rust Durable Execution examples.
#
# examples/ is a single cargo WORKSPACE (examples/Cargo.toml): every example is
# a member, so the whole set shares ONE target dir (examples/target) and ONE
# resolve of the SDK graph. This script runs a single `cargo lambda build` over
# the requested families and stages each example's binary into
# publish/<example>/bootstrap for SAM deployment on the provided.al2023
# runtime. It mirrors conformance/build_examples.sh (same layout, same
# skip-guard, same Makefile-per-bootstrap contract); the SDK's own release
# profile is untouched: fast-build tuning lives in examples/Cargo.toml.
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
#   ./build_examples.sh [family...]
#
# Families (default: all found in this directory):
#   basics
#
# Examples:
#   ./build_examples.sh basics     # build only the basics family
#   ./build_examples.sh            # build everything
#
# Skip-if-unchanged: a second invocation with the same families, an unchanged
# git HEAD, a clean working tree, all bootstraps present, and no source file
# newer than the last build is a no-op. A dirty tree, a new commit, a touched
# source file, or a missing bootstrap always rebuilds.

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
glibc version the example binaries link against so they start on the
provided.al2023 runtime regardless of which host built them.

Install it with one of:
    pip3 install cargo-lambda
    cargo install cargo-lambda
EOF
    exit 1
fi

# Resolve the requested families
if [ $# -gt 0 ]; then
    families="$*"
else
    families=$(find . -maxdepth 1 -mindepth 1 -type d \
        ! -name publish ! -name target ! -name '.*' \
        -exec basename {} \;)
fi
# Stable signature (sorted, single-space-joined) for the skip stamp.
fams_sig=$(printf '%s\n' $families | sort | tr '\n' ' ')

# Collect the example package + directory list for these families
pkgs=""
example_dirs=""
for fam in $families; do
    [ -d "$fam" ] || { echo "Error: unknown family '$fam'." >&2; exit 1; }
    for dir in "$fam"/*/; do
        [ -d "$dir" ] || continue
        pkgs="$pkgs -p $(basename "$dir")"
        example_dirs="$example_dirs $dir"
    done
done
[ -n "$example_dirs" ] || { echo "Error: no examples found for: $families" >&2; exit 1; }

# Skip-if-unchanged guard
sha=$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo nogit)
dirty=$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null || true)

should_skip() {
    # Never skip on a dirty tree.
    [ -z "$dirty" ] || return 1
    # Need a prior stamp for the same SHA and the same family set.
    [ -f "$STAMP" ] || return 1
    [ "$(sed -n '1p' "$STAMP")" = "$sha" ] || return 1
    [ "$(sed -n '2p' "$STAMP")" = "$fams_sig" ] || return 1
    # Every requested bootstrap must actually exist.
    for dir in $example_dirs; do
        [ -f "$PUBLISH_DIR/$(basename "$dir")/bootstrap" ] || return 1
    done
    # No example source, SDK source, or manifest newer than the stamp.
    newer=$(find "$SCRIPT_DIR" "$SCRIPT_DIR/../src" "$SCRIPT_DIR/../Cargo.toml" \
        \( -name '*.rs' -o -name 'Cargo.toml' \) -newer "$STAMP" \
        ! -path "$SCRIPT_DIR/target/*" ! -path "$SCRIPT_DIR/publish/*" \
        2>/dev/null | head -1)
    [ -z "$newer" ] || return 1
    return 0
}

if should_skip; then
    echo "Nothing changed since last build (HEAD $sha, clean tree, families: $fams_sig)."
    echo "Skipping. (delete $STAMP or touch a source file to force a rebuild.)"
    exit 0
fi

# Single shared-workspace build over just the requested members
echo "Building families: $families"
# cargo-lambda stages every binary it finds in the target dir into
# target/lambda/<bin>/bootstrap, so a leftover binary from an earlier build of
# other members would look like fresh output. Clear the staging dir first: only
# what this invocation builds may be staged.
rm -rf "$LAMBDA_DIR"
# --x86-64 is explicit rather than host-derived so an aarch64 workstation does
# not silently produce arm64 binaries for the x86_64 SAM functions.
# shellcheck disable=SC2086
cargo lambda build --release --x86-64 $pkgs

# Stage each example binary into publish/<example>/bootstrap
for dir in $example_dirs; do
    example=$(basename "$dir")
    fam=$(basename "$(dirname "$dir")")
    out="$PUBLISH_DIR/$example"

    # The binary name is the crate name with '-' normalized to '_'.
    bin_name=$(grep '^name' "$dir/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/' | tr '-' '_')
    src_bin="$LAMBDA_DIR/$bin_name/bootstrap"
    if [ ! -f "$src_bin" ]; then
        echo "Error: built binary not found for $example at $src_bin" >&2
        exit 1
    fi

    mkdir -p "$out"
    cp "$src_bin" "$out/bootstrap"

    # SAM builds provided.al2023 functions through BuildMethod: makefile.
    # Logical IDs are the PascalCase form of the example dir name.
    logical_id=$(echo "$example" | awk -F_ '{for (i = 1; i <= NF; i++) printf "%s%s", toupper(substr($i, 1, 1)), substr($i, 2)}')
    cat > "$out/Makefile" << MKEOF
.PHONY: build-$logical_id

build-$logical_id:
	cp -r . \$(ARTIFACTS_DIR)/
	rm -f \$(ARTIFACTS_DIR)/Makefile
MKEOF
    echo "Staged $fam/$example -> publish/$example/bootstrap"
done

# Record the build stamp (SHA + family signature); mtime = now
mkdir -p "$PUBLISH_DIR"
{ echo "$sha"; echo "$fams_sig"; } > "$STAMP"

echo "Build complete. Binaries in $PUBLISH_DIR"
