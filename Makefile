.PHONY: check ci check-sdk msrv dependencies external-types feature-powerset \
	workspaces test-role conformance conformance-build conformance-run cloud-tests

# Every target calls a script in scripts/, which holds the commands.
# CONTRIBUTING.md describes the targets and their inputs.

# The quality gate for every change: the check-sdk, dependencies and msrv
# jobs of ci.yml.
check:
	sh scripts/ci.sh check-sdk dependencies msrv

# Every ci.yml job that runs on a push to main, with CI's environment.
ci:
	sh scripts/ci.sh ci

# One target per ci.yml job. ci.yml calls these.
check-sdk:
	sh scripts/ci.sh check-sdk
msrv:
	sh scripts/ci.sh msrv
dependencies:
	sh scripts/ci.sh dependencies
external-types:
	sh scripts/ci.sh external-types
feature-powerset:
	sh scripts/ci.sh feature-powerset
workspaces:
	sh scripts/ci.sh workspaces

# Targets that deploy to AWS. They need TEST_ACCOUNT_ID.
test-role:
	sh scripts/test-role.sh
conformance:
	sh scripts/conformance.sh all
conformance-build:
	sh scripts/conformance.sh build
conformance-run:
	sh scripts/conformance.sh run
cloud-tests:
	sh scripts/cloud-tests.sh run
