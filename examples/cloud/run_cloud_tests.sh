#!/bin/sh
# Cloud smoke tests for the deployed examples.
#
# Invokes every directly-invokable example in the three deployed family
# stacks and asserts its durable execution reaches the terminal state
# documented in examples/README.md and docs/porting-map.md. Callback
# examples suspend awaiting an external completion; this harness drives
# them with `aws lambda send-durable-execution-callback-success`.
#
# Per-example expectations are DATA (the EXPECTATIONS table below), not
# code: each row names the example, the terminal status that counts as a
# pass, an optional invoke payload override, whether the example's
# callback must be driven externally, and an optional set of allowed
# Result values (pipe-separated) for nondeterministic examples.
#
# Prerequisites:
#   - The three family stacks deployed (see .github/workflows/cloud-tests.yml):
#     template_basics.yaml, template_coordination.yaml, template_external.yaml
#   - AWS credentials and region configured for the target test account
#   - jq
#
# Environment:
#   STACK_BASICS / STACK_COORDINATION / STACK_EXTERNAL  stack names
#     (default: dex-rust-examples-<family>)
#
# Usage:
#   ./run_cloud_tests.sh

set -eu

STACK_BASICS=${STACK_BASICS:-dex-rust-examples-basics}
STACK_COORDINATION=${STACK_COORDINATION:-dex-rust-examples-coordination}
STACK_EXTERNAL=${STACK_EXTERNAL:-dex-rust-examples-external}

# How often a pending durable execution is re-checked, and how long a single
# example may take to reach a terminal state after being invoked.
POLL_INTERVAL=5
EXECUTION_TIMEOUT=900

# How often and for how long to look for a suspended example's callback id
# in its execution history before giving up. Kept tight because
# create_callback_heartbeat times out if its callback sees no traffic for
# 10 seconds after creation.
CALLBACK_POLL_INTERVAL=2
CALLBACK_DISCOVERY_TIMEOUT=120
# Upper bound on a backgrounded invoke for a callback example. The function's
# own timeout ends the invocation well before this; this only guarantees the
# harness cannot block forever.
CALLBACK_INVOKE_TIMEOUT=180

DEFAULT_PAYLOAD='"cloud-test"'
CALLBACK_RESULT='"approved by cloud test"'

# ---- expectations table -------------------------------------------------
#
# family|example|expected_terminal_status|payload_override|drive|allowed_results
#
#   payload_override  JSON payload for the invoke; empty means the shared
#                     default (a JSON string, which every handler taking a
#                     String or serde_json::Value event accepts).
#   drive             `callback` if the example suspends on a callback that
#                     this harness must complete externally.
#   allowed_results   Pipe-separated set of allowed Result JSON strings for
#                     the completed execution. Empty means any result is
#                     accepted (only terminal status is asserted). Used for
#                     nondeterministic examples like combinator_race.
#
# Special cases (all documented in examples/README.md):
#   - handler_error terminates FAILED by design: FAILED is its pass state.
#   - create_callback_timeout / wait_for_callback_timeout catch their
#     timeouts and terminate SUCCEEDED.
#   - combinator_race has a nondeterministic winner; BOTH "first" and
#     "second" are valid results (the handler returns whichever step wins).
#   - step_named deserializes a typed Input struct; wait_for_condition
#     takes a bare integer threshold; invoke_tenant_id needs a tenantId.
#   - invoke_target and invoke_target_tenant are companion callees
#     exercised transitively; they are deliberately absent from this table.
EXPECTATIONS='
basics|hello_world|SUCCEEDED|||
basics|step_basic|SUCCEEDED|||
basics|step_named|SUCCEEDED|{"data":"cloud"}||
basics|step_with_retry|SUCCEEDED|||
basics|wait_basic|SUCCEEDED|||
basics|wait_named|SUCCEEDED|||
basics|multiple_waits|SUCCEEDED|||
coordination|child_basic|SUCCEEDED|||
coordination|child_failing_step|SUCCEEDED|||
coordination|child_large_data|SUCCEEDED|||
coordination|child_serdes|SUCCEEDED|||
coordination|combinator_join_all|SUCCEEDED|||
coordination|combinator_race|SUCCEEDED|||"first"|"second"
coordination|combinator_select_ok|SUCCEEDED|||
coordination|combinator_try_join_all|SUCCEEDED|||
coordination|concurrent_operations|SUCCEEDED|||
coordination|map_basic|SUCCEEDED|||
coordination|map_completion|SUCCEEDED|||
coordination|map_concurrency|SUCCEEDED|||
coordination|map_empty|SUCCEEDED|||
coordination|map_virtual|SUCCEEDED|||
coordination|parallel_basic|SUCCEEDED|||
coordination|parallel_completion|SUCCEEDED|||
coordination|parallel_empty|SUCCEEDED|||
coordination|parallel_heterogeneous|SUCCEEDED|||
coordination|parallel_virtual|SUCCEEDED|||
external|comprehensive|SUCCEEDED|||
external|create_callback_simple|SUCCEEDED||callback|
external|create_callback_heartbeat|SUCCEEDED||callback|
external|create_callback_timeout|SUCCEEDED|||
external|error_determinism|SUCCEEDED|||
external|handler_error|FAILED|||
external|invoke_simple|SUCCEEDED|||
external|invoke_tenant_id|SUCCEEDED|{"tenantId":"tenant-cloud","payload":{"greeting":"hello"}}||
external|large_payload|SUCCEEDED|||
external|logging_after_wait|SUCCEEDED|||
external|logging_levels|SUCCEEDED|||
external|no_replay_execution|SUCCEEDED|||
external|optional_results|SUCCEEDED|||
external|retry_exhaustion|SUCCEEDED|||
external|serde_basic|SUCCEEDED|||
external|serde_configure|SUCCEEDED|||
external|wait_for_callback_basic|SUCCEEDED||callback|
external|wait_for_callback_submitter|SUCCEEDED||callback|
external|wait_for_callback_timeout|SUCCEEDED|||
external|wait_for_condition|SUCCEEDED|2||
'

# ---- helpers ------------------------------------------------------------

WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT INT TERM

log() { printf '%s %s\n' "$(date -u '+%H:%M:%S')" "$*"; }

# stack_for FAMILY -> stack name
stack_for() {
    case "$1" in
        basics) echo "$STACK_BASICS" ;;
        coordination) echo "$STACK_COORDINATION" ;;
        external) echo "$STACK_EXTERNAL" ;;
        *) echo "unknown family: $1" >&2; exit 1 ;;
    esac
}

# logical_id EXAMPLE -> PascalCase logical resource id
# (same derivation build_examples.sh uses to generate SAM Makefile targets)
logical_id() {
    echo "$1" | awk -F_ '{for (i = 1; i <= NF; i++) printf "%s%s", toupper(substr($i, 1, 1)), substr($i, 2)}'
}

# function_name FAMILY EXAMPLE -> deployed (SAM-generated) function name.
# Names are discovered from the stack because the templates deliberately do
# not set FunctionName.
function_name() {
    map="$WORK_DIR/resources-$1.tsv"
    if [ ! -f "$map" ]; then
        aws cloudformation describe-stack-resources \
            --stack-name "$(stack_for "$1")" \
            --query "StackResources[?ResourceType=='AWS::Lambda::Function'].[LogicalResourceId,PhysicalResourceId]" \
            --output text > "$map"
    fi
    awk -v id="$(logical_id "$2")" '$1 == id { print $2 }' "$map"
}

# result_matches ACTUAL ALLOWED_SET -> 0 if ACTUAL is in the pipe-separated
# ALLOWED_SET, 1 otherwise.
result_matches() {
    actual="$1"
    allowed="$2"
    # Write candidates one per line and grep for an exact match.
    echo "$allowed" | tr '|' '\n' | grep -qxF "$actual"
}

# drive_callback EXAMPLE ARN: find the suspended execution's callback id in
# its history and complete it with a success result.
drive_callback() {
    deadline=$(( $(date +%s) + CALLBACK_DISCOVERY_TIMEOUT ))
    while :; do
        cb_id=$(aws lambda get-durable-execution-history \
            --durable-execution-arn "$2" --no-include-execution-data \
            --query "Events[?EventType=='CallbackStarted'].CallbackStartedDetails.CallbackId | [0]" \
            --output text)
        if [ -n "$cb_id" ] && [ "$cb_id" != "None" ]; then
            aws lambda send-durable-execution-callback-success \
                --callback-id "$cb_id" --result "$CALLBACK_RESULT" > /dev/null
            log "$1: completed callback"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            log "$1: FAIL - no callback appeared within ${CALLBACK_DISCOVERY_TIMEOUT}s"
            return 1
        fi
        sleep "$CALLBACK_POLL_INTERVAL"
    done
}

# discover_execution_arn FUNCTION STARTED_AFTER: find the durable execution a
# background invoke just started. Needed because an invoke of a function that
# parks on a callback with no timeout does not return until the callback is
# completed, so its response cannot supply the ARN we need in order to
# complete it.
discover_execution_arn() {
    deadline=$(( $(date +%s) + CALLBACK_DISCOVERY_TIMEOUT ))
    while :; do
        arn=$(aws lambda list-durable-executions-by-function \
            --function-name "$1" --statuses RUNNING \
            --started-after "$2" --reverse-order --max-items 1 \
            --query 'DurableExecutions[0].DurableExecutionArn' \
            --output text 2> /dev/null)
        if [ -n "$arn" ] && [ "$arn" != "None" ]; then
            printf '%s\n' "$arn"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            return 1
        fi
        sleep "$CALLBACK_POLL_INTERVAL"
    done
}

# ---- phase 1: invoke every example (and drive its callback if needed) ----
#
# An invoke returns once the execution reaches a terminal state or parks on a
# timer, so most examples are started here and polled to their terminal state
# in phase 2. The exception is an example that parks on a callback with no
# timeout: its invoke stays open until the callback is completed, so it is
# started in the background, its execution is discovered through the list API,
# and its callback is completed while that invoke is still open.

echo "$EXPECTATIONS" | grep -v '^[[:space:]]*$' | while IFS='|' read -r family example expected payload drive allowed_results; do
    fn=$(function_name "$family" "$example")
    if [ -z "$fn" ]; then
        echo "FAIL $example: function not found in stack $(stack_for "$family")" >> "$WORK_DIR/failures"
        continue
    fi

    meta="$WORK_DIR/invoke-$example.json"
    if [ "$drive" = "callback" ]; then
        started_after=$(date -u -d '1 minute ago' +%Y-%m-%dT%H:%M:%SZ)
        # Bounded by `timeout` rather than killed later: closing the client
        # connection early can cancel the in-flight invocation, and the parked
        # handler must stay alive long enough to observe the completed
        # callback. The function's own timeout ends the invocation first.
        timeout "$CALLBACK_INVOKE_TIMEOUT" \
            aws lambda invoke --function-name "$fn" --qualifier '$LATEST' \
            --payload "${payload:-$DEFAULT_PAYLOAD}" \
            --cli-binary-format raw-in-base64-out \
            --cli-read-timeout 0 \
            --output json "$WORK_DIR/response-$example.json" > "$meta" &
        invoke_pid=$!

        if ! arn=$(discover_execution_arn "$fn" "$started_after"); then
            wait "$invoke_pid" 2> /dev/null || true
            echo "FAIL $example: execution did not appear within ${CALLBACK_DISCOVERY_TIMEOUT}s" >> "$WORK_DIR/failures"
            continue
        fi

        printf '%s|%s|%s|%s\n' "$example" "$expected" "$arn" "$allowed_results" >> "$WORK_DIR/pending"
        log "$example: started ($arn)"

        if ! drive_callback "$example" "$arn"; then
            echo "FAIL $example: callback drive failed" >> "$WORK_DIR/failures"
        fi
        # Let the invocation finish on its own; `timeout` bounds the wait.
        wait "$invoke_pid" 2> /dev/null || true
        continue
    fi

    if ! aws lambda invoke --function-name "$fn" --qualifier '$LATEST' \
        --payload "${payload:-$DEFAULT_PAYLOAD}" \
        --cli-binary-format raw-in-base64-out \
        --output json "$WORK_DIR/response-$example.json" > "$meta"; then
        echo "FAIL $example: invoke failed" >> "$WORK_DIR/failures"
        continue
    fi

    arn=$(jq -r '.DurableExecutionArn // empty' "$meta")
    fn_error=$(jq -r '.FunctionError // empty' "$meta")

    if [ -z "$arn" ]; then
        # A handler-level error can fail before a durable execution is
        # reported; that satisfies an expected-FAILED example.
        if [ "$expected" = "FAILED" ] && [ -n "$fn_error" ]; then
            log "$example: FunctionError with no execution ARN (expected FAILED) - pass"
            continue
        fi
        echo "FAIL $example: no durable execution ARN returned (FunctionError: $fn_error)" >> "$WORK_DIR/failures"
        continue
    fi

    printf '%s|%s|%s|%s\n' "$example" "$expected" "$arn" "$allowed_results" >> "$WORK_DIR/pending"
    log "$example: started ($arn)"
done

# ---- phase 2: poll every execution to its terminal state -----------------

if [ -f "$WORK_DIR/pending" ]; then
    while IFS='|' read -r example expected arn allowed_results; do
        deadline=$(( $(date +%s) + EXECUTION_TIMEOUT ))
        while :; do
            status=$(aws lambda get-durable-execution \
                --durable-execution-arn "$arn" --query Status --output text)
            if [ "$status" != "RUNNING" ]; then
                break
            fi
            if [ "$(date +%s)" -ge "$deadline" ]; then
                status="POLL_TIMEOUT"
                break
            fi
            sleep "$POLL_INTERVAL"
        done

        if [ "$status" != "$expected" ]; then
            log "$example: $status (expected $expected) - FAIL"
            echo "FAIL $example: terminal status $status, expected $expected" >> "$WORK_DIR/failures"
            continue
        fi

        # If allowed_results is specified, fetch the execution result and
        # validate it against the allowed set.
        if [ -n "$allowed_results" ]; then
            result=$(aws lambda get-durable-execution \
                --durable-execution-arn "$arn" --query Result --output text)
            if ! result_matches "$result" "$allowed_results"; then
                log "$example: result '$result' not in allowed set ($allowed_results) - FAIL"
                echo "FAIL $example: result '$result' not in allowed set ($allowed_results)" >> "$WORK_DIR/failures"
                continue
            fi
            log "$example: $status, result '$result' in allowed set - pass"
        else
            log "$example: $status (expected $expected) - pass"
        fi
    done < "$WORK_DIR/pending"
fi

# ---- report ---------------------------------------------------------------

total=$(echo "$EXPECTATIONS" | grep -c '|' || true)
if [ -f "$WORK_DIR/failures" ]; then
    failures=$(wc -l < "$WORK_DIR/failures")
    echo
    echo "Cloud tests: $failures of $total example(s) FAILED:"
    cat "$WORK_DIR/failures"
    exit 1
fi
echo
echo "Cloud tests: all $total examples reached their expected terminal state."
