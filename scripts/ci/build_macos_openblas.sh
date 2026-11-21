#!/usr/bin/env bash

set -euo pipefail

OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.33}"
OPENBLAS_BUILD_ROOT="${OPENBLAS_BUILD_ROOT:-/tmp/janusx-openblas-macos-build}"
OPENBLAS_PREFIX="${OPENBLAS_PREFIX:-/tmp/janusx-openblas-macos-prefix}"
OPENBLAS_TARBALL="${OPENBLAS_BUILD_ROOT}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz"
OPENBLAS_SRC_DIR="${OPENBLAS_BUILD_ROOT}/OpenBLAS-${OPENBLAS_VERSION}"
OPENBLAS_JOBS="${OPENBLAS_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
OPENBLAS_BUILD_TARGET="${OPENBLAS_BUILD_TARGET:-shared}"
MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"
export MACOSX_DEPLOYMENT_TARGET

mkdir -p "${OPENBLAS_BUILD_ROOT}"
rm -rf "${OPENBLAS_PREFIX}" "${OPENBLAS_SRC_DIR}"

curl -fL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
  -o "${OPENBLAS_TARBALL}"
tar -xzf "${OPENBLAS_TARBALL}" -C "${OPENBLAS_BUILD_ROOT}"

arch="$(uname -m)"
make_args=(
  "BINARY=64"
  "INTERFACE64=0"
  "USE_THREAD=1"
  "USE_OPENMP=0"
  "NO_AFFINITY=1"
  "NO_STATIC=1"
  "NUM_THREADS=128"
  "CC=clang"
  "FC=gfortran"
  "HOSTCC=clang"
)

case "${arch}" in
  arm64|aarch64)
    # macOS wheels target Apple Silicon only.  A fixed ARMV8 build avoids
    # compiling OpenBLAS' full runtime-dispatch matrix (Neoverse/SME/etc.),
    # which adds several minutes to cold CI builds without improving the
    # supported macOS arm64 binary.
    make_args+=("TARGET=ARMV8" "DYNAMIC_ARCH=0")
    ;;
  x86_64)
    make_args+=("TARGET=CORE2" "DYNAMIC_ARCH=1" "DYNAMIC_OLDER=1")
    ;;
  *)
    make_args+=("DYNAMIC_ARCH=1")
    ;;
esac

echo "[build_macos_openblas] version=${OPENBLAS_VERSION} arch=${arch} jobs=${OPENBLAS_JOBS} deployment_target=${MACOSX_DEPLOYMENT_TARGET}"
echo "[build_macos_openblas] build_target=${OPENBLAS_BUILD_TARGET}"
echo "[build_macos_openblas] make args: ${make_args[*]}"

make -C "${OPENBLAS_SRC_DIR}" -j"${OPENBLAS_JOBS}" "${make_args[@]}" "${OPENBLAS_BUILD_TARGET}"
make -C "${OPENBLAS_SRC_DIR}" PREFIX="${OPENBLAS_PREFIX}" "${make_args[@]}" install

if [[ ! -d "${OPENBLAS_PREFIX}/lib" ]]; then
  echo "ERROR: OpenBLAS install did not produce ${OPENBLAS_PREFIX}/lib" >&2
  exit 1
fi

(
  cd "${OPENBLAS_PREFIX}/lib"
  if [[ ! -e libopenblas.dylib ]]; then
    candidate="$(ls -1 libopenblas*.dylib 2>/dev/null | head -n1 || true)"
    if [[ -n "${candidate}" ]]; then
      ln -sf "${candidate}" libopenblas.dylib
    fi
  fi
  if [[ ! -e libopenblas.0.dylib ]]; then
    candidate="$(ls -1 libopenblas*.dylib 2>/dev/null | head -n1 || true)"
    if [[ -n "${candidate}" ]]; then
      ln -sf "${candidate}" libopenblas.0.dylib
    fi
  fi

  if [[ ! -e libopenblas.dylib ]]; then
    echo "ERROR: OpenBLAS install did not produce libopenblas.dylib" >&2
    ls -al "${OPENBLAS_PREFIX}/lib" >&2
    exit 1
  fi
)

if [[ ! -d "${OPENBLAS_PREFIX}/include" ]]; then
  echo "ERROR: OpenBLAS install did not produce ${OPENBLAS_PREFIX}/include" >&2
  exit 1
fi

export OPENBLAS_LIB_DIR="${OPENBLAS_PREFIX}/lib"
export OPENBLAS_INCLUDE_DIR="${OPENBLAS_PREFIX}/include"
export PKG_CONFIG_PATH="${OPENBLAS_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

python3 - <<'PY'
import ctypes
import os
from pathlib import Path
import subprocess

lib_dir = Path(os.environ["OPENBLAS_LIB_DIR"])
path = (lib_dir / "libopenblas.dylib").resolve()
print(f"[build_macos_openblas] installed={path}")

otool_out = subprocess.check_output(["otool", "-L", str(path)], text=True)
for line in otool_out.splitlines()[1:]:
    dep = line.strip().split(" (", 1)[0].strip()
    dep_name = Path(dep).name.lower()
    if dep_name.startswith(("libomp", "libgomp", "libiomp")):
        raise SystemExit(
            f"ERROR: macOS OpenBLAS unexpectedly links OpenMP runtime: {dep}"
        )

lib = ctypes.CDLL(str(path))
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_parallel.restype = ctypes.c_int
cfg_raw = lib.openblas_get_config()
cfg = cfg_raw.decode("utf-8", "ignore") if cfg_raw else ""
parallel = int(lib.openblas_get_parallel())
print(f"[build_macos_openblas] config={cfg}")
print(f"[build_macos_openblas] parallel={parallel}")
if parallel != 1:
    raise SystemExit(
        f"ERROR: macOS OpenBLAS must use pthreads (parallel=1), got parallel={parallel} (config={cfg!r})"
    )
PY

echo "OPENBLAS_LIB_DIR=${OPENBLAS_LIB_DIR}"
echo "OPENBLAS_INCLUDE_DIR=${OPENBLAS_INCLUDE_DIR}"
