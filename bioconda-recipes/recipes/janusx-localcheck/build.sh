#!/usr/bin/env bash
set -euxo pipefail

# Lower parallelism to reduce peak memory in constrained CI/containers.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL=0
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"

# Build/install using the conda-provided maturin + Rust toolchain.
python -m pip install . \
  --no-deps \
  --no-build-isolation \
  -vv

# Collect Rust dependency licenses for Bioconda license compliance checks.
cargo-bundle-licenses --format yaml --output THIRDPARTY.yml
