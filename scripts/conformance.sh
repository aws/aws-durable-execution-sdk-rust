#!/bin/sh
# conformance.sh: build the conformance handlers, and run the suites against
# AWS with the language-agnostic runner, the way conformance-tests.yml does.
#
# Usage:
#   sh scripts/conformance.sh build     build the handlers of SUITES (make conformance-build)
#   sh scripts/conformance.sh run       run SUITES in parallel (make conformance-run)
#   sh scripts/conformance.sh all       build, then run (make conformance)
#   sh scripts/conformance.sh suites    print the default suites, one per line
#   sh scripts/conformance.sh teardown  delete the stacks of SUITES for RUN_ID
#
# SUITES (space-separated) defaults to every conformance/template_<suite>.yaml
# except template_smoke.yaml. The smoke template is one hello handler that
# checks a deployment works at all; it is not a suite the runner validates.
# The AWS inputs are described in scripts/aws.sh.
#
# For each suite, `run` writes conformance/report-<suite>.xml (JUnit),
# conformance/history-<suite>/ (execution histories) and
# conformance/history-<suite>/runner.log (the runner's output), the paths
# the workflow uploads. Then it deletes the stack of every suite, including
# a suite whose runner was killed before its own cleanup, prints the full
# log of each failed suite and a per-suite summary, and exits with status 1
# when any suite failed or a stack could not be deleted.
#
# POSIX sh, no bashisms.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/tools.sh
. scripts/aws.sh

RUNNER="git+https://github.com/aws/aws-durable-execution-conformance-tests.git@main#subdirectory=packages/aws-durable-execution-conformance-tests"

default_suites() {
    for template in conformance/template_*.yaml; do
        suite=${template#conformance/template_}
        suite=${suite%.yaml}
        [ "$suite" = smoke ] || echo "$suite"
    done
}

SUITES=${SUITES:-$(default_suites | tr '\n' ' ')}
for suite in $SUITES; do
    [ -f "conformance/template_$suite.yaml" ] ||
        die "Unknown suite '$suite': there is no conformance/template_$suite.yaml."
done

# CloudFormation stack names allow only [a-zA-Z][-a-zA-Z0-9]*.
slug() {
    echo "$1" | tr '_' '-'
}

build() {
    require timeout cargo cargo-lambda
    # shellcheck disable=SC2086
    (cd conformance && timeout --foreground 3600 ./build_examples.sh $SUITES)
}

# suite_stacks: the stack name of each suite in SUITES for RUN_ID. The
# runner prefixes --name with "conformance-tests-" (STACK_NAME_PREFIX in its
# config.py).
suite_stacks() {
    for suite in $SUITES; do
        echo "conformance-tests-conf-rust-$(slug "$suite")-$RUN_ID"
    done
}

teardown() {
    # shellcheck disable=SC2046
    delete_stacks $(suite_stacks)
}

# cleanup: stop every suite still running, then delete the stacks. Runs
# once: at the end of `run`, or from the EXIT trap on any other exit path,
# so an interrupt and a failure reach it too.
cleanup() {
    trap '' INT TERM HUP
    [ -z "${cleaned:-}" ] || return 0
    cleaned=1
    for pid in $pids; do
        stop_group "$pid"
    done
    echo
    echo "=== deleting the stacks of RUN_ID $RUN_ID ==="
    if teardown; then teardown_result=PASS; else teardown_result=FAIL; fi
    rm -rf "$venv"
}

run() {
    require timeout aws sam python3
    check_account
    resolve_run_id
    resolve_role

    pids=""
    # Absolute, because the runner starts from conformance/.
    venv=$PWD/target/conformance-venv-$RUN_ID
    teardown_result=FAIL
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP

    echo "=== installing the conformance runner into $venv ==="
    rm -rf "$venv"
    if ! timeout --foreground 300 python3 -m venv "$venv" ||
        ! timeout --foreground 900 "$venv/bin/pip" install --quiet --disable-pip-version-check "$RUNNER" ./conformance/conformance_ext; then
        die "Could not install the conformance runner."
    fi

    # Each suite runs in the background under its own timeout, which leads
    # its own process group, so cleanup's stop_group stops the runner and
    # the SAM CLI under it. Unbuffered, so a killed runner's log is whole.
    export PYTHONUNBUFFERED=1
    for suite in $SUITES; do
        rm -rf "conformance/history-$suite" "conformance/report-$suite.xml"
        mkdir -p "conformance/history-$suite"
        (cd conformance && exec timeout 3600 "$venv/bin/python" -m aws_durable_execution_conformance_tests.app \
            --template "template_$suite.yaml" \
            --language rust \
            --suite "$suite" \
            --name "conf-rust-$(slug "$suite")-$RUN_ID" \
            --region "$AWS_REGION" \
            --parameter-overrides ExecutionRoleArn="$EXECUTION_ROLE_ARN" \
            --history-dir "history-$suite" \
            --report junit \
            --report-file "report-$suite") \
            > "conformance/history-$suite/runner.log" 2>&1 &
        pids="$pids $!"
        echo "started $suite, log conformance/history-$suite/runner.log"
    done

    summary=""
    failed=""
    # shellcheck disable=SC2086
    set -- $pids
    for suite in $SUITES; do
        if wait "$1"; then
            result=PASS
        else
            result="FAIL (exit $?)"
            failed="$failed $suite"
        fi
        # A runner killed from outside leaves the SAM CLI it started running.
        kill -KILL -- "-$1" 2> /dev/null || true
        shift
        echo "finished $suite: $result"
        summary="$summary$(printf '%-20s %s' "$suite" "$result")
"
    done
    pids=""

    cleanup
    for suite in $failed; do
        echo
        echo "=== log of failed suite $suite (conformance/history-$suite/runner.log) ==="
        cat "conformance/history-$suite/runner.log"
    done
    echo
    echo "=== summary (RUN_ID $RUN_ID) ==="
    printf '%s' "$summary"
    printf '%-20s %s\n' "stack deletion" "$teardown_result"
    [ -z "$failed" ] && [ "$teardown_result" = PASS ]
}

case "${1:-}" in
    build) build ;;
    run) run ;;
    all)
        # The account is checked before the build too, so a wrong account
        # fails in seconds rather than after the build.
        require aws
        check_account
        build && run
        ;;
    suites) default_suites ;;
    teardown)
        require aws timeout
        [ -n "${RUN_ID:-}" ] || die "teardown needs the RUN_ID of the run whose stacks it deletes."
        check_account
        resolve_run_id
        teardown
        ;;
    *)
        echo "usage: sh scripts/conformance.sh build | run | all | suites | teardown" >&2
        exit 2
        ;;
esac
