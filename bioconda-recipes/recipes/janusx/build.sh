#!/usr/bin/env bash
set -euxo pipefail

# Lower parallelism to reduce peak memory in constrained CI/containers.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL=0
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"

# Isolate Cargo config from host/global ~/.cargo to avoid accidental mirror overrides
# (for example stale rsproxy settings on shared HPC nodes).
export CARGO_HOME="${PWD}/.cargo-home"
mkdir -p "${CARGO_HOME}"

if [[ -d "${PWD}/vendor" ]]; then
  cat > "${CARGO_HOME}/config.toml" <<EOF
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "${PWD}/vendor"

[net]
git-fetch-with-cli = true
retry = 6
offline = true
EOF
else
  : "${JANUSX_CARGO_REGISTRY:=sparse+https://index.crates.io/}"
  cat > "${CARGO_HOME}/config.toml" <<EOF
[source.crates-io]
registry = "${JANUSX_CARGO_REGISTRY}"

[net]
git-fetch-with-cli = true
retry = 6
EOF
fi

# Build/install using the conda-provided maturin + Rust toolchain.
python -m pip install . \
  --no-deps \
  --no-build-isolation \
  -vv

# Collect Rust dependency licenses for Bioconda license compliance checks.
cargo-bundle-licenses --format yaml --output THIRDPARTY.yml
