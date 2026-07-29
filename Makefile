.PHONY: check

# Single quality-gate entry point. Every command must pass for the
# build to be considered green. Runs: format check, lints, tests
# (including doctests), documentation build, and dependency policy.
check:
	cargo fmt --check
	cargo clippy --all-targets --all-features -- -D warnings
	cargo test
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
	cargo deny check
	sh scripts/check-direct-deps.sh
