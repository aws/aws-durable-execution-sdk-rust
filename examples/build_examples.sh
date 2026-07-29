#!/bin/sh
# Build script for the Rust Durable Execution examples.
#
# examples/ is a single cargo WORKSPACE (examples/Cargo.toml): every example is
# a member, so the whole set shares ONE target dir (examples/target) and ONE
# resolve of the SDK graph. This script runs a single `cargo build` over the
# requested families and stages each example's binary into
# publish/<example>/bootstrap for SAM deployment on the provided.al2023
# runtime. It mirrors compliance/build_examples.sh (same layout, same
# skip-guard, same Makefile-per-bootstrap contract); the SDK's own release
# profile is untouched — fast-build tuning lives in examples/Cargo.toml.
#
# Requires:
#   - Rust toolchain with the x86_64-unknown-linux-gnu target (native on AL2023)
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
TARGET="x86_64-unknown-linux-gnu"
RELEASE_DIR="$SCRIPT_DIR/target/$TARGET/release"
STAMP="$PUBLISH_DIR/.built-at"

cd "$SCRIPT_DIR"

# ---- resolve the requested families ----
if [ $# -gt 0 ]; then
    families="$*"
else
    families=$(find . -maxdepth 1 -mindepth 1 -type d \
        ! -name publish ! -name target ! -name '.*' \
        -exec basename {} \;)
fi
# Stable signature (sorted, single-space-joined) for the skip stamp.
fams_sig=$(printf '%s\n' $families | sort | tr '\n' ' ')

# ---- collect the example package + directory list for these families ----
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

# ---- skip-if-unchanged guard ----
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

# ---- single shared-workspace build over just the requested members ----
echo "Building families: $families"
# shellcheck disable=SC2086
cargo build --release --target "$TARGET" $pkgs

# ---- stage each example binary into publish/<example>/bootstrap ----
for dir in $example_dirs; do
    example=$(basename "$dir")
    fam=$(basename "$(dirname "$dir")")
    out="$PUBLISH_DIR/$example"

    # The binary name is the crate name with '-' normalized to '_'.
    bin_name=$(grep '^name' "$dir/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/' | tr '-' '_')
    src_bin="$RELEASE_DIR/$bin_name"
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

# ---- record the build stamp (SHA + family signature); mtime = now ----
mkdir -p "$PUBLISH_DIR"
{ echo "$sha"; echo "$fams_sig"; } > "$STAMP"

echo "Build complete. Binaries in $PUBLISH_DIR"
