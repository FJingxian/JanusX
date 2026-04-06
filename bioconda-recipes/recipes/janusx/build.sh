#!/usr/bin/env bash
set -euxo pipefail

# Lower parallelism to reduce peak memory in constrained CI/containers.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export CARGO_INCREMENTAL=0
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export CARGO_NET_RETRY="${CARGO_NET_RETRY:-6}"
export CARGO_HTTP_TIMEOUT="${CARGO_HTTP_TIMEOUT:-900}"
export CARGO_HTTP_LOW_SPEED_LIMIT="${CARGO_HTTP_LOW_SPEED_LIMIT:-1}"
export CARGO_HTTP_MULTIPLEXING="${CARGO_HTTP_MULTIPLEXING:-false}"

# Isolate Cargo config from host/global ~/.cargo to avoid accidental mirror overrides
# (for example stale rsproxy settings on shared HPC nodes).
export HOME="${PWD}/.home"
mkdir -p "${HOME}"
if [[ -n "${JANUSX_CARGO_HOME:-}" ]]; then
  echo "[build.sh] ignore JANUSX_CARGO_HOME=${JANUSX_CARGO_HOME} to keep Cargo config isolated"
fi
export CARGO_HOME="${PWD}/.cargo-home"
mkdir -p "${CARGO_HOME}"
mkdir -p "${PWD}/.cargo"
project_cargo_config="${PWD}/.cargo/config.toml"

# Remove mirror/env overrides inherited from parent shell/session.
unset CARGO_REGISTRIES_CRATES_IO_INDEX || true
unset CARGO_REGISTRIES_CRATES_IO_PROTOCOL || true
unset CARGO_REGISTRIES_CRATES_IO_REPLACE_WITH || true
unset CARGO_SOURCE_CRATES_IO_REPLACE_WITH || true
unset CARGO_REGISTRIES_RSPROXY_INDEX || true

if [[ -d "${PWD}/vendor" ]]; then
  # Project-local config has higher priority than parent-directory ~/.cargo config.
  cat > "${project_cargo_config}" <<EOF
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
  : "${JANUSX_CARGO_REGISTRIES:=sparse+https://index.crates.io/ https://mirrors.ustc.edu.cn/crates.io-index sparse+https://rsproxy.cn/index/}"
  : "${JANUSX_CARGO_PROBE:=1}"
  : "${JANUSX_CARGO_PROBE_TIMEOUT:=10}"

  if [[ -n "${JANUSX_CARGO_REGISTRY:-}" ]]; then
    JANUSX_CARGO_REGISTRIES="${JANUSX_CARGO_REGISTRY} ${JANUSX_CARGO_REGISTRIES}"
  fi

  selected_registry=""
  selected_dl=""

  if [[ "${JANUSX_CARGO_PROBE}" == "1" ]]; then
    while IFS= read -r candidate; do
      [[ -n "${candidate}" ]] || continue
      probe_output="$(python - "${candidate}" "${JANUSX_CARGO_PROBE_TIMEOUT}" <<'PY'
import json
import sys
import urllib.request

candidate = sys.argv[1].strip()
timeout = float(sys.argv[2])
index_url = candidate
if index_url.startswith("sparse+"):
    index_url = index_url[len("sparse+"):]
index_url = index_url.rstrip("/")
cfg_url = f"{index_url}/config.json"

try:
    req = urllib.request.Request(cfg_url, headers={"User-Agent": "janusx-cargo-probe/1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        cfg = json.loads(resp.read().decode("utf-8", errors="replace"))
except Exception:
    raise SystemExit(2)

dl = (cfg.get("dl") or "").rstrip("/")
if not dl:
    raise SystemExit(3)

probe_url = f"{dl}/adler2/2.0.1/download"
try:
    req = urllib.request.Request(
        probe_url,
        headers={"User-Agent": "janusx-cargo-probe/1"},
        method="HEAD",
    )
    with urllib.request.urlopen(req, timeout=timeout):
        pass
except Exception:
    try:
        req = urllib.request.Request(
            probe_url,
            headers={"User-Agent": "janusx-cargo-probe/1"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read(1)
    except Exception:
        raise SystemExit(4)

print(f"OK\t{candidate}\t{dl}")
PY
      )" || {
        echo "[build.sh] skip unreachable cargo registry: ${candidate}"
        continue
      }

      IFS=$'\t' read -r probe_status probe_registry probe_dl <<< "${probe_output}" || true
      if [[ "${probe_status}" == "OK" && -n "${probe_registry}" ]]; then
        selected_registry="${probe_registry}"
        selected_dl="${probe_dl:-}"
        break
      fi
    done < <(printf '%s\n' ${JANUSX_CARGO_REGISTRIES})
  else
    selected_registry="${JANUSX_CARGO_REGISTRY:-${JANUSX_CARGO_REGISTRIES%% *}}"
  fi

  if [[ -z "${selected_registry}" ]]; then
    selected_registry="${JANUSX_CARGO_REGISTRY:-${JANUSX_CARGO_REGISTRIES%% *}}"
    echo "[build.sh][WARN] probe did not find reachable endpoint; fallback to: ${selected_registry}"
  fi

  JANUSX_CARGO_REGISTRY="${selected_registry}"
  export CARGO_REGISTRIES_CRATES_IO_INDEX="${JANUSX_CARGO_REGISTRY}"

  if [[ "${JANUSX_CARGO_REGISTRY}" == *"index.crates.io"* ]]; then
    # Override any inherited crates-io -> rsproxy replacement explicitly.
    # Use :443 to avoid Cargo's duplicate-source detection on canonical crates-io.
    janusx_direct_registry="sparse+https://index.crates.io:443/"
    cat > "${project_cargo_config}" <<EOF
[source.crates-io]
replace-with = "janusx-registry"

[source.janusx-registry]
registry = "${janusx_direct_registry}"

[net]
git-fetch-with-cli = true
retry = ${CARGO_NET_RETRY}
EOF

    cat > "${CARGO_HOME}/config.toml" <<EOF
[source.crates-io]
replace-with = "janusx-registry"

[source.janusx-registry]
registry = "${janusx_direct_registry}"

[net]
git-fetch-with-cli = true
retry = 6
EOF
  else
    # Explicitly override any parent replace-with (such as rsproxy) by setting
    # crates-io -> janusx-registry at project scope.
    cat > "${project_cargo_config}" <<EOF
[source.crates-io]
replace-with = "janusx-registry"

[source.janusx-registry]
registry = "${JANUSX_CARGO_REGISTRY}"

[net]
git-fetch-with-cli = true
retry = ${CARGO_NET_RETRY}
EOF

    cat > "${CARGO_HOME}/config.toml" <<EOF
[source.crates-io]
replace-with = "janusx-registry"

[source.janusx-registry]
registry = "${JANUSX_CARGO_REGISTRY}"

[net]
git-fetch-with-cli = true
retry = 6
EOF
  fi
fi

echo "[build.sh] cargo: $(cargo --version || true)"
echo "[build.sh] HOME=${HOME}"
echo "[build.sh] CARGO_HOME=${CARGO_HOME}"
echo "[build.sh] JANUSX_CARGO_REGISTRY=${JANUSX_CARGO_REGISTRY:-}"
echo "[build.sh] JANUSX_CARGO_REGISTRIES=${JANUSX_CARGO_REGISTRIES:-}"
echo "[build.sh] selected registry dl=${selected_dl:-unknown}"
echo "[build.sh] CARGO_HTTP_TIMEOUT=${CARGO_HTTP_TIMEOUT}"
echo "[build.sh] CARGO_HTTP_LOW_SPEED_LIMIT=${CARGO_HTTP_LOW_SPEED_LIMIT}"
echo "[build.sh] CARGO_HTTP_MULTIPLEXING=${CARGO_HTTP_MULTIPLEXING}"
echo "[build.sh] project cargo config:"
cat "${project_cargo_config}"

# Build/install using the conda-provided maturin + Rust toolchain.
python -m pip install . \
  --no-deps \
  --no-build-isolation \
  -vv

# Collect Rust dependency licenses for Bioconda license compliance checks.
cargo-bundle-licenses --format yaml --output THIRDPARTY.yml
