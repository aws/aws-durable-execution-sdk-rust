#!/bin/sh
# Installs the Rust build tools of the telemetry conformance handlers.
#
# The shared telemetry-suite workflow of aws-durable-execution-conformance-tests
# runs this script through its setup_command input. That input runs in the
# same shell as the prepare_command input. So this script is sourced, and the
# PATH it sets reaches the build.
#
# The CodeBuild runner image has no rustup. So the script installs the
# rustup-init that .github/actions/ensure-rustup pins, when rustup is missing.
# It then installs the stable toolchain and the cargo-lambda version that
# conformance-tests.yml installs.

RUSTUP_INIT_VERSION=1.29.1
RUSTUP_INIT_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
CARGO_LAMBDA_VERSION=1.8.6

if ! command -v rustup > /dev/null 2>&1; then
    if [ "$(uname -m)" != "x86_64" ]; then
        echo "telemetry-conformance-setup only pins the x86_64 rustup-init; runner is $(uname -m)" >&2
        return 1 2> /dev/null || exit 1
    fi
    tmp=$(mktemp -d)
    curl -fsSL --proto '=https' --tlsv1.2 \
        "https://static.rust-lang.org/rustup/archive/${RUSTUP_INIT_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
        -o "$tmp/rustup-init"
    echo "${RUSTUP_INIT_SHA256}  $tmp/rustup-init" | sha256sum -c - || { rm -rf "$tmp"; return 1 2> /dev/null || exit 1; }
    chmod +x "$tmp/rustup-init"
    "$tmp/rustup-init" -y --no-modify-path --profile minimal --default-toolchain none
    rm -rf "$tmp"
    PATH="$HOME/.cargo/bin:$PATH"
    export PATH
fi

rustup toolchain install stable --profile minimal
rustup default stable
rustup show active-toolchain
python -m pip install "cargo-lambda==${CARGO_LAMBDA_VERSION}"
