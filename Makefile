.PHONY: check

# Single quality-gate entry point. Every command must pass for the
# build to be considered green. Runs: format check, lints, tests
# (including doctests), documentation build, dependency policy, and an
# MSRV compile check matching CI's msrv job (the version comes from the
# root Cargo.toml's rust-version, the same source of truth that
# scripts/check-msrv-consistency.sh enforces).
check:
	cargo fmt --check
	cargo clippy --all-targets --all-features -- -D warnings
	cargo test
	cargo test --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
	cargo deny check
	sh scripts/check-direct-deps.sh
	sh scripts/check-msrv-consistency.sh
	sh scripts/check-msrv.sh
