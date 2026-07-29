#!/bin/sh
# check-direct-deps.sh — Mechanical gate for the closed production dependency allowlist.
#
# cargo-deny's [bans] allow list checks the ENTIRE dependency graph (including
# transitives), which is impractical with the aws-sdk tree (~400 crates).
# This script enforces the spec's allowlist on DIRECT production dependencies
# only: any workspace member adding a [dependencies] entry not in ALLOWLIST
# causes a non-zero exit and a clear error message.
#
# Wired into `make check`. See also deny.toml for the rationale comment.
#
# POSIX sh — no bashisms.

set -e

# The 8 approved production dependencies per the design spec.
ALLOWLIST="aws-config aws-sdk-lambda lambda_runtime serde serde_json sha2 tokio tracing"

violations=""
current_member=""

# `cargo tree --workspace --depth 1 -e normal` outputs workspace members
# (identifiable by a local path in parentheses) as section roots, with their
# direct normal dependencies listed beneath. We parse this to identify which
# direct deps belong to OUR workspace members only.
cargo tree --workspace --depth 1 -e normal --prefix none 2>/dev/null | while IFS= read -r line; do
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
    for a in $ALLOWLIST; do
        if [ "$dep" = "$a" ]; then
            allowed=1
            break
        fi
    done

    if [ "$allowed" -eq 0 ]; then
        echo "VIOLATION:$current_member:$dep"
    fi
done > /tmp/check-direct-deps-violations.$$

if [ -s /tmp/check-direct-deps-violations.$$ ]; then
    echo "direct-dep-allowlist FAILED — unapproved direct production dependencies detected:"
    echo ""
    while IFS=: read -r _ member dep; do
        echo "  ERROR: crate '$dep' is a direct dependency of workspace member '$member' but is NOT in the production allowlist."
    done < /tmp/check-direct-deps-violations.$$
    echo ""
    echo "Approved allowlist: $ALLOWLIST"
    echo "To add a new production dependency, update ALLOWLIST in scripts/check-direct-deps.sh"
    echo "and obtain design-spec approval. See deny.toml for policy."
    rm -f /tmp/check-direct-deps-violations.$$
    exit 1
fi

rm -f /tmp/check-direct-deps-violations.$$
echo "direct-dep-allowlist OK — all direct production dependencies are on the allowlist."
