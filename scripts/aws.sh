# aws.sh: inputs and stack deletion shared by the AWS targets.
# Sourced by conformance.sh, cloud-tests.sh and test-role.sh. POSIX sh.
#
# Inputs, all from the environment (make passes its command-line variables
# through):
#   Credentials       the standard AWS chain: AWS_PROFILE locally,
#                     environment variables in CI. Never an argument.
#   TEST_ACCOUNT_ID   required. Nothing is deployed unless the credentials
#                     belong to this account.
#   AWS_REGION        default us-west-2.
#   EXECUTION_ROLE_ARN default: the RoleArn output of $ROLE_STACK.
#   RUN_ID            default local-<UTC yyyymmddHHMMSS>. Every stack name a
#                     run creates contains it, and a run deletes only those.

ROLE_STACK=durable-sdk-test-execution-role

AWS_REGION=${AWS_REGION:-us-west-2}
# The harness and the conformance runner call AWS without --region.
AWS_DEFAULT_REGION=$AWS_REGION
export AWS_REGION AWS_DEFAULT_REGION

die() {
    echo "$*" >&2
    exit 1
}

# stop_group PID: kill the process group led by PID, a `timeout` started in
# the background. Such a timeout leads its own process group, so the kill
# reaches every process under it. SIGKILL, because examples/cloud's harness
# traps SIGINT and SIGTERM and keeps running after them.
stop_group() {
    kill -KILL -- "-$1" 2> /dev/null || kill -KILL "$1" 2> /dev/null || true
    wait "$1" 2> /dev/null || true
}

# check_account: stop unless the credentials belong to TEST_ACCOUNT_ID.
check_account() {
    [ -n "${TEST_ACCOUNT_ID:-}" ] ||
        die "TEST_ACCOUNT_ID is required: the AWS account the test stacks may be deployed to."
    actual=$(aws sts get-caller-identity --query Account --output text) ||
        die "Could not read the caller identity. Check the AWS credentials (AWS_PROFILE)."
    [ "$actual" = "$TEST_ACCOUNT_ID" ] ||
        die "Expected AWS account $TEST_ACCOUNT_ID but the credentials are for $actual. Nothing was deployed."
    echo "Using AWS test account $actual in $AWS_REGION."
}

# resolve_run_id: set RUN_ID, refusing a value that is not a lowercase
# stack-name fragment.
resolve_run_id() {
    RUN_ID=${RUN_ID:-local-$(date -u +%Y%m%d%H%M%S)}
    case "$RUN_ID" in
        '' | *[!a-z0-9-]*) die "RUN_ID '$RUN_ID' may contain only lowercase letters, digits and '-'." ;;
    esac
    echo "RUN_ID=$RUN_ID"
}

# resolve_role: set EXECUTION_ROLE_ARN from the role stack unless it is set.
resolve_role() {
    if [ -z "${EXECUTION_ROLE_ARN:-}" ]; then
        EXECUTION_ROLE_ARN=$(aws cloudformation describe-stacks --stack-name "$ROLE_STACK" \
            --query "Stacks[0].Outputs[?OutputKey=='RoleArn'].OutputValue" --output text) || true
    fi
    case "$EXECUTION_ROLE_ARN" in
        '' | None) die "No execution role: set EXECUTION_ROLE_ARN, or deploy one with 'make test-role'." ;;
    esac
}

# delete_stacks STACK...: request deletion of every STACK first, so an
# interrupted wait still leaves every deletion requested, then wait for
# each. A stack that does not exist counts as deleted. Returns 1 when a
# deletion could not be confirmed.
delete_stacks() {
    for stack in "$@"; do
        aws cloudformation delete-stack --stack-name "$stack" ||
            echo "Could not request deletion of $stack." >&2
    done
    delete_failed=0
    for stack in "$@"; do
        if timeout 1800 aws cloudformation wait stack-delete-complete --stack-name "$stack"; then
            echo "No stack $stack remains."
        else
            echo "FAILED to confirm deletion of $stack." >&2
            delete_failed=1
        fi
    done
    return "$delete_failed"
}
