#!/bin/sh
# check-direct-deps.sh — Mechanical gate for the closed production dependency allowlist.
#
# cargo-deny's [bans] allow list checks the ENTIRE dependency graph (including
# transitives), which is impractical with the aws-sdk tree (~400 crates).
# This script enforces the spec's allowlist on DIRECT production dependencies
# only: any workspace member adding a [dependencies] entry not in the
# allowlists causes a non-zero exit and a clear error message.
#
# Two graphs are checked, because an optional dependency hides from the
# default-feature graph:
#
#   1. Default features — only the ALWAYS_ON allowlist may appear. This is
#      the graph every consumer gets, so an optional crate leaking into it
#      (e.g. via a stray non-optional feature edge) fails here.
#   2. --all-features — ALWAYS_ON plus OPTIONAL_ALLOWLIST may appear. This
#      closes the loophole where an unapproved crate rides in behind a
#      feature flag and never shows up in pass 1.
#
# Wired into `make check`. See also deny.toml for the rationale comment.
#
# POSIX sh — no bashisms.

set -e

# Temporary files, created once and cleaned up on any exit path. Keeping
# them at the top level (rather than per call) lets a single trap own them.
TREE_FILE=$(mktemp /tmp/check-direct-deps-tree.XXXXXX)
TREE_ERR_FILE=$(mktemp /tmp/check-direct-deps-tree-err.XXXXXX)
VIOLATIONS_FILE=$(mktemp /tmp/check-direct-deps-violations.XXXXXX)
trap 'rm -f "$TREE_FILE" "$TREE_ERR_FILE" "$VIOLATIONS_FILE"' EXIT INT HUP TERM

# The 8 always-on production dependencies per the design spec.
ALWAYS_ON="aws-config aws-sdk-lambda lambda_runtime serde serde_json sha2 tokio tracing"

# Optional, feature-gated production dependencies. Each entry needs the same
# policy approval as ALWAYS_ON (see CONTRIBUTING.md, "Dependency policy") and
# a note here naming the feature that gates it:
#   tracing-subscriber — gated by the `replay-filter` feature; provides the
#     Filter/Layer traits ReplayFilterLayer implements. Absent from every
#     default-feature build.
OPTIONAL_ALLOWLIST="tracing-subscriber"

# check_graph <allowlist> <label> [extra cargo-tree args...]
# Parses `cargo tree --workspace --depth 1 -e normal` for the given feature
# selection and reports any direct dependency of a workspace member that the
# allowlist does not name. Returns non-zero on violation.
check_graph() {
    allowlist=$1
    label=$2
    shift 2

    # Run `cargo tree` on its own and capture its output to a file, so its
    # exit status is checked directly. Feeding it straight into the parser
    # via a pipeline would take the pipeline's status from the parser group,
    # letting a failed (or absent) cargo silently produce an empty graph
    # that both passes trivially. Fail closed instead.
    if ! cargo tree --workspace --depth 1 -e normal --prefix none "$@" \
        > "$TREE_FILE" 2> "$TREE_ERR_FILE"; then
        echo "direct-dep-allowlist FAILED ($label) — 'cargo tree' itself failed; cannot verify the dependency graph:" >&2
        cat "$TREE_ERR_FILE" >&2
        return 1
    fi

    # An empty graph means cargo told us nothing to check, which is never
    # true for this workspace. Treat it as a failure rather than a pass.
    if [ ! -s "$TREE_FILE" ]; then
        echo "direct-dep-allowlist FAILED ($label) — 'cargo tree' produced no output; cannot verify the dependency graph." >&2
        return 1
    fi

    # `cargo tree --workspace --depth 1 -e normal` outputs workspace members
    # (identifiable by a local path in parentheses) as section roots, with
    # their direct normal dependencies listed beneath. We parse this to
    # identify which direct deps belong to OUR workspace members only.
    {
        current_member=""
        while IFS= read -r line; do
            # Skip blank lines (section separators).
            case "$line" in
                "") current_member=""; continue ;;
            esac

            # Workspace members have a local path in parentheses, e.g.:
            #   aws-durable-execution-sdk-rust v0.1.0 (/path/to/crate)
            # Duplicates (shown as dep of another member) have (*) and should
            # NOT start a new section.
            case "$line" in
                *"(*)"*) continue ;;
                *"(/"*")"*)
                    # This is a workspace member line — extract name (first field).
                    current_member=$(echo "$line" | awk '{print $1}')
                    # Skip the compliance crate — it's test infrastructure, not production.
                    case "$current_member" in
                        compliance) current_member=""; continue ;;
                    esac
                    continue
                    ;;
            esac

            # If we're not inside a workspace member section, skip.
            [ -z "$current_member" ] && continue

            # Extract the dependency crate name (first field).
            dep=$(echo "$line" | awk '{print $1}')

            # Skip workspace-internal path dependencies (they also have local paths).
            case "$line" in
                *"(/"*")"*) continue ;;
            esac

            # Check against allowlist.
            allowed=0
            for a in $allowlist; do
                if [ "$dep" = "$a" ]; then
                    allowed=1
                    break
                fi
            done

            if [ "$allowed" -eq 0 ]; then
                echo "VIOLATION:$current_member:$dep"
            fi
        done
    } < "$TREE_FILE" > "$VIOLATIONS_FILE"

    if [ -s "$VIOLATIONS_FILE" ]; then
        echo "direct-dep-allowlist FAILED ($label) — unapproved direct production dependencies detected:"
        echo ""
        while IFS=: read -r _ member dep; do
            echo "  ERROR: crate '$dep' is a direct dependency of workspace member '$member' but is NOT in the allowlist for this graph."
        done < "$VIOLATIONS_FILE"
        echo ""
        echo "Always-on allowlist: $ALWAYS_ON"
        echo "Optional (feature-gated) allowlist: $OPTIONAL_ALLOWLIST"
        echo "To add a new production dependency, update the allowlists in scripts/check-direct-deps.sh"
        echo "and obtain design-spec approval. See CONTRIBUTING.md and deny.toml for policy."
        return 1
    fi

    return 0
}

# Pass 1: the default-feature graph must contain only the always-on crates.
check_graph "$ALWAYS_ON" "default features"

# Pass 2: with every feature enabled, only always-on plus approved optional
# crates may appear.
check_graph "$ALWAYS_ON $OPTIONAL_ALLOWLIST" "--all-features" --all-features

echo "direct-dep-allowlist OK — default-feature graph limited to the always-on allowlist;"
echo "all-features graph adds only approved optional crates ($OPTIONAL_ALLOWLIST)."
