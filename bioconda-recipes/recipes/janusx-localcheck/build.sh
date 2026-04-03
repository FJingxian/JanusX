#!/usr/bin/env bash
set -euxo pipefail

# Lower parallelism to reduce peak memory in constrained CI/containers.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL=0
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export CARGO_NET_RETRY="${CARGO_NET_RETRY:-6}"

# Isolate Cargo config from host/global ~/.cargo to avoid accidental mirror overrides
# (for example stale rsproxy settings on shared HPC nodes).
export CARGO_HOME="${PWD}/.cargo-home"
mkdir -p "${CARGO_HOME}"
mkdir -p "${PWD}/.cargo"

# Remove mirror/env overrides inherited from parent shell/session.
unset CARGO_REGISTRIES_CRATES_IO_INDEX || true
unset CARGO_REGISTRIES_CRATES_IO_PROTOCOL || true
unset CARGO_REGISTRIES_CRATES_IO_REPLACE_WITH || true
unset CARGO_SOURCE_CRATES_IO_REPLACE_WITH || true
unset CARGO_REGISTRIES_RSPROXY_INDEX || true

if [[ -d "${PWD}/vendor" ]]; then
  # Project-local config has higher priority than parent-directory ~/.cargo config.
  cat > "${PWD}/.cargo/config.toml" <<EOF
[source.crates-io]
replace-with = "vendored-sources"

[source.vendored-sources]
directory = "${PWD}/vendor"

[net]
git-fetch-with-cli = true
retry = ${CARGO_NET_RETRY}
offline = true
EOF

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

  # Force direct crates.io (or user-provided mirror) at project level to override rsproxy.
  cat > "${PWD}/.cargo/config.toml" <<EOF
[source.crates-io]
replace-with = "janusx-direct-crates-io"

[source.janusx-direct-crates-io]
registry = "${JANUSX_CARGO_REGISTRY}"

[net]
git-fetch-with-cli = true
retry = ${CARGO_NET_RETRY}
EOF

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
