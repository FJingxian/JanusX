#!/usr/bin/env bash

set -euo pipefail

OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.33}"
OPENBLAS_BUILD_ROOT="${OPENBLAS_BUILD_ROOT:-/tmp/janusx-openblas-build}"
OPENBLAS_PREFIX="${OPENBLAS_PREFIX:-/tmp/janusx-openblas-prefix}"
OPENBLAS_TARBALL="${OPENBLAS_BUILD_ROOT}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
OPENBLAS_SRC_DIR="${OPENBLAS_BUILD_ROOT}/OpenBLAS-${OPENBLAS_VERSION}"
OPENBLAS_JOBS="${OPENBLAS_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 2)}"
OPENBLAS_BUILD_TARGET="${OPENBLAS_BUILD_TARGET:-shared}"

mkdir -p "${OPENBLAS_BUILD_ROOT}"
rm -rf "${OPENBLAS_PREFIX}" "${OPENBLAS_SRC_DIR}"

curl -L "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
  -o "${OPENBLAS_TARBALL}"
tar -xzf "${OPENBLAS_TARBALL}" -C "${OPENBLAS_BUILD_ROOT}"

arch="$(uname -m)"
make_args=(
  "BINARY=64"
  "INTERFACE64=0"
  "USE_THREAD=1"
  "NO_AFFINITY=1"
  "NO_STATIC=1"
  "NUM_THREADS=128"
  "CC=gcc"
  "FC=gfortran"
  "HOSTCC=gcc"
)

case "${arch}" in
  x86_64)
    make_args+=("TARGET=CORE2" "DYNAMIC_ARCH=1" "DYNAMIC_OLDER=1")
    ;;
  aarch64)
    make_args+=("TARGET=ARMV8" "DYNAMIC_ARCH=1")
    ;;
  *)
    make_args+=("DYNAMIC_ARCH=1")
    ;;
esac

echo "[build_linux_openblas] version=${OPENBLAS_VERSION} arch=${arch} jobs=${OPENBLAS_JOBS}"
echo "[build_linux_openblas] build_target=${OPENBLAS_BUILD_TARGET}"
echo "[build_linux_openblas] make args: ${make_args[*]}"

# Use `shared` instead of the default `all` target so wheel builds compile the
# required shared library without running OpenBLAS self-tests. The default
# target expands to `tests`, and ARM64 runners can fail in optional SBGEMM test
# binaries even when the library itself is built correctly.
make -C "${OPENBLAS_SRC_DIR}" -j"${OPENBLAS_JOBS}" "${make_args[@]}" "${OPENBLAS_BUILD_TARGET}"
make -C "${OPENBLAS_SRC_DIR}" PREFIX="${OPENBLAS_PREFIX}" "${make_args[@]}" install

if [[ ! -d "${OPENBLAS_PREFIX}/lib" ]]; then
  echo "ERROR: OpenBLAS install did not produce ${OPENBLAS_PREFIX}/lib" >&2
  exit 1
fi

cd "${OPENBLAS_PREFIX}/lib"
if [[ ! -e libopenblas.so ]]; then
  candidate="$(ls -1 libopenblas*.so* 2>/dev/null | head -n1 || true)"
  if [[ -n "${candidate}" ]]; then
    ln -sf "${candidate}" libopenblas.so
  fi
fi
if [[ ! -e libopenblas.so.0 ]]; then
  candidate="$(ls -1 libopenblas*.so* 2>/dev/null | head -n1 || true)"
  if [[ -n "${candidate}" ]]; then
    ln -sf "${candidate}" libopenblas.so.0
  fi
fi

if [[ ! -e libopenblas.so ]]; then
  echo "ERROR: OpenBLAS install did not produce libopenblas.so" >&2
  ls -al "${OPENBLAS_PREFIX}/lib" >&2
  exit 1
fi

if [[ ! -d "${OPENBLAS_PREFIX}/include" ]]; then
  echo "ERROR: OpenBLAS install did not produce ${OPENBLAS_PREFIX}/include" >&2
  exit 1
fi

export OPENBLAS_LIB_DIR="${OPENBLAS_PREFIX}/lib"
export OPENBLAS_INCLUDE_DIR="${OPENBLAS_PREFIX}/include"
export LD_LIBRARY_PATH="${OPENBLAS_LIB_DIR}:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import ctypes
import os
from pathlib import Path

lib_dir = Path(os.environ["OPENBLAS_LIB_DIR"])
path = lib_dir / "libopenblas.so"
print(f"[build_linux_openblas] installed={path}")
lib = ctypes.CDLL(str(path))
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_parallel.restype = ctypes.c_int
cfg_raw = lib.openblas_get_config()
cfg = cfg_raw.decode("utf-8", "ignore") if cfg_raw else ""
parallel = int(lib.openblas_get_parallel())
print(f"[build_linux_openblas] config={cfg}")
print(f"[build_linux_openblas] parallel={parallel}")
if parallel == 0:
    raise SystemExit("OpenBLAS source build reports no threading support")
PY

echo "OPENBLAS_LIB_DIR=${OPENBLAS_LIB_DIR}"
echo "OPENBLAS_INCLUDE_DIR=${OPENBLAS_INCLUDE_DIR}"
