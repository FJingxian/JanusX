#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  scripts/cargo_test_py.sh [cargo test args...]

Runs `cargo test` with PYO3_PYTHON and macOS libpython search paths derived
from the selected Python interpreter.

Selection order for Python:
  1. $PYO3_PYTHON
  2. $CONDA_PREFIX/bin/python
  3. python3 on PATH
  4. python on PATH

Examples:
  scripts/cargo_test_py.sh --lib
  PYO3_PYTHON="$HOME/miniconda3/envs/jxfu/bin/python" scripts/cargo_test_py.sh exact_scan_core_matches_manual_reference --lib
EOF
  exit 0
fi

py_bin="${PYO3_PYTHON:-}"
if [[ -z "$py_bin" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  py_bin="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "$py_bin" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    py_bin="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    py_bin="$(command -v python)"
  else
    echo "error: no Python interpreter found; set PYO3_PYTHON explicitly" >&2
    exit 1
  fi
fi
if [[ ! -x "$py_bin" ]]; then
  echo "error: selected Python is not executable: $py_bin" >&2
  exit 1
fi

py_info="$("$py_bin" - <<'PY'
from pathlib import Path
import sys
import sysconfig

libdir = sysconfig.get_config_var("LIBDIR")
if not libdir:
    libdir = str(Path(sys.executable).resolve().parents[1] / "lib")
version = f"{sys.version_info[0]}.{sys.version_info[1]}"
libpython = Path(libdir) / f"libpython{version}.dylib"
print(sys.executable)
print(libdir)
print(libpython)
PY
)"

py_exec="$(printf '%s\n' "$py_info" | sed -n '1p')"
py_libdir="$(printf '%s\n' "$py_info" | sed -n '2p')"
py_libpython="$(printf '%s\n' "$py_info" | sed -n '3p')"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if [[ ! -f "$py_libpython" ]]; then
    echo "error: libpython not found at $py_libpython" >&2
    echo "hint: activate the intended conda env or set PYO3_PYTHON=/path/to/python" >&2
    exit 1
  fi
  export DYLD_LIBRARY_PATH="${py_libdir}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
  export DYLD_FALLBACK_LIBRARY_PATH="${py_libdir}${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"
fi

export PYO3_PYTHON="$py_exec"
exec cargo test "$@"
