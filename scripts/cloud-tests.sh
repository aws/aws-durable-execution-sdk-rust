#!/bin/sh
# cloud-tests.sh: build the examples, deploy the three family stacks, run
# examples/cloud/run_cloud_tests.sh against them, and delete the stacks, the
# way cloud-tests.yml does.
#
# Usage:
#   sh scripts/cloud-tests.sh run       (make cloud-tests)
#   sh scripts/cloud-tests.sh teardown  delete the stacks of RUN_ID
#
# The stacks are rust-examples-<RUN_ID>-<family>. The AWS inputs are
# described in scripts/aws.sh. The first failing step ends the run. The
# stacks are deleted on every exit path: success, a failed step, a step that
# hits its timeout, and SIGINT (Ctrl-C), SIGTERM or SIGHUP. A per-step
# summary follows, and the exit status is 1 when any step failed.
#
# POSIX sh, no bashisms.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/tools.sh
. scripts/aws.sh

FAMILIES="basics coordination external"

stack() {
    echo "rust-examples-$RUN_ID-$1"
}

all_stacks() {
    for family in $FAMILIES; do
        stack "$family"
    done
}

# bounded DIR SECONDS COMMAND...: run COMMAND in DIR under a timeout, in the
# background, so a signal reaches this script's trap at once instead of
# after the command ends.
bounded() {
    dir=$1
    seconds=$2
    shift 2
    (cd "$dir" && exec timeout "$seconds" "$@") &
    child=$!
    wait "$child"
    rc=$?
    child=""
    return "$rc"
}

# step NAME COMMAND...: run one step and add its result to the summary.
step() {
    name=$1
    shift
    echo
    echo "=== $name ==="
    start=$(date +%s)
    "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then result=PASS; else result="FAIL (exit $rc)"; failed=1; fi
    summary="$summary$(printf '%-22s %-15s %6ss' "$name" "$result" "$(($(date +%s) - start))")
"
    return "$rc"
}

deploy() {
    family=$1
    bounded examples 1800 sam build --template-file "template_$family.yaml" \
        --build-dir ".aws-sam/build-$family" &&
        bounded examples 1800 sam deploy \
            --template-file ".aws-sam/build-$family/template.yaml" \
            --stack-name "$(stack "$family")" \
            --resolve-s3 \
            --no-confirm-changeset \
            --no-fail-on-empty-changeset \
            --region "$AWS_REGION" \
            --parameter-overrides ExecutionRoleArn="$EXECUTION_ROLE_ARN"
}

# cleanup: stop the running step, delete the stacks, print the summary. Runs
# once: from the end of `run`, or from the EXIT trap on any other exit path.
cleanup() {
    trap '' INT TERM HUP
    [ -z "${cleaned:-}" ] || return 0
    cleaned=1
    if [ -n "$child" ]; then
        stop_group "$child"
        summary="$summary$(printf '%-22s %s' "$name" "INTERRUPTED")
"
        failed=1
    fi
    # shellcheck disable=SC2046
    step "delete stacks" delete_stacks $(all_stacks)
    echo
    echo "=== summary (RUN_ID $RUN_ID) ==="
    printf '%s' "$summary"
}

run() {
    require timeout cargo cargo-lambda sam aws jq
    check_account
    resolve_run_id
    resolve_role

    summary=""
    failed=0
    child=""
    trap 'rc=$?; cleanup; exit "$rc"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP

    STACK_BASICS=$(stack basics)
    STACK_COORDINATION=$(stack coordination)
    STACK_EXTERNAL=$(stack external)
    export STACK_BASICS STACK_COORDINATION STACK_EXTERNAL

    step "build examples" bounded examples 3600 ./build_examples.sh &&
        step "deploy basics" deploy basics &&
        step "deploy coordination" deploy coordination &&
        step "deploy external" deploy external &&
        step "run_cloud_tests.sh" bounded examples 5400 ./cloud/run_cloud_tests.sh

    cleanup
    trap - EXIT
    exit "$failed"
}

case "${1:-}" in
    run) run ;;
    teardown)
        require aws timeout
        [ -n "${RUN_ID:-}" ] || die "teardown needs the RUN_ID of the run whose stacks it deletes."
        check_account
        resolve_run_id
        # shellcheck disable=SC2046
        delete_stacks $(all_stacks)
        ;;
    *)
        echo "usage: sh scripts/cloud-tests.sh run | teardown" >&2
        exit 2
        ;;
esac
