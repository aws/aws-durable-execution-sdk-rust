#!/bin/sh
# Build script for Rust Durable Execution conformance test handlers.
#
# compliance/ is a single cargo WORKSPACE (compliance/Cargo.toml): every
# handler is a member, so the whole suite shares ONE target dir
# (compliance/target) and ONE resolve of the SDK + aws-sdk graph. This script
# runs a single `cargo build` over the requested members and stages each
# handler's binary into publish/<handler>/bootstrap for SAM deployment on the
# provided.al2023 runtime.
#
# Fast-build tuning lives in compliance/Cargo.toml [profile.release] (handlers
# opt-level 1, deps opt-level 3 via package."*"), scoped to this workspace —
# the SDK's own release profile is untouched.
#
# Requires:
#   - Rust toolchain with the x86_64-unknown-linux-gnu target (native on AL2023)
#
# Usage:
#   ./build_examples.sh [operation...]
#
# Operations (default: all found in this directory):
#   smoke, step, wait, callback, child, invoke, parallel,
#   wait_for_callback, wait_for_condition, map
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
TARGET="x86_64-unknown-linux-gnu"
RELEASE_DIR="$SCRIPT_DIR/target/$TARGET/release"
STAMP="$PUBLISH_DIR/.built-at"

cd "$SCRIPT_DIR"

# ---- resolve the requested operations ----
if [ $# -gt 0 ]; then
    operations="$*"
else
    operations=$(find . -maxdepth 1 -mindepth 1 -type d \
        ! -name publish ! -name src ! -name target ! -name '.*' \
        -exec basename {} \;)
fi
# Stable signature (sorted, single-space-joined) for the skip stamp.
ops_sig=$(printf '%s\n' $operations | sort | tr '\n' ' ')

# ---- collect the handler package + directory list for these operations ----
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

# ---- skip-if-unchanged guard ----
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

# ---- single shared-workspace build over just the requested members ----
echo "Building operations: $operations"
# shellcheck disable=SC2086
cargo build --release --target "$TARGET" $pkgs

# ---- stage each handler binary into publish/<handler>/bootstrap ----
for dir in $handler_dirs; do
    handler=$(basename "$dir")
    op=$(basename "$(dirname "$dir")")
    out="$PUBLISH_DIR/$handler"

    # The binary name is the crate name with '-' normalized to '_'.
    bin_name=$(grep '^name' "$dir/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/' | tr '-' '_')
    src_bin="$RELEASE_DIR/$bin_name"
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

# ---- record the build stamp (SHA + operation signature); mtime = now ----
mkdir -p "$PUBLISH_DIR"
{ echo "$sha"; echo "$ops_sig"; } > "$STAMP"

echo "Build complete. Binaries in $PUBLISH_DIR"
