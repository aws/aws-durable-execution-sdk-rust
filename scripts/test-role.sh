#!/bin/sh
# test-role.sh: deploy scripts/test-execution-role.yaml as the stack the
# AWS targets read their default EXECUTION_ROLE_ARN from, and print its
# RoleArn output (make test-role).
#
# A deploy with no change succeeds and changes nothing, so running it again
# is safe. It needs IAM-capable credentials; the test stacks do not. The AWS
# inputs are described in scripts/aws.sh.
#
# POSIX sh, no bashisms.

set -u
cd "$(dirname "$0")/.." || exit 1
. scripts/tools.sh
. scripts/aws.sh

require timeout aws
check_account
timeout --foreground 900 aws cloudformation deploy \
    --template-file scripts/test-execution-role.yaml \
    --stack-name "$ROLE_STACK" \
    --capabilities CAPABILITY_IAM \
    --no-fail-on-empty-changeset ||
    die "Deploying stack $ROLE_STACK failed."
EXECUTION_ROLE_ARN=
resolve_role
echo "RoleArn: $EXECUTION_ROLE_ARN"
