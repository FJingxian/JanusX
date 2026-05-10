#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import os
import subprocess
import sys


def _janusx_pkg_dir() -> str:
    spec = importlib.util.find_spec("janusx")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("cannot locate installed janusx package directory")
    for p in spec.submodule_search_locations:
        if p:
            return str(p)
    raise RuntimeError("cannot locate installed janusx package directory")


def _find_bundled_openblas() -> str:
    pkg_dir = _janusx_pkg_dir()
    root_parent = os.path.dirname(pkg_dir)
    cands = sorted(
        glob.glob(os.path.join(root_parent, "janusx.libs", "libopenblas*.so*"))
        + glob.glob(os.path.join(root_parent, "janusx.libs", "libopenblas*.dylib"))
    )
    if not cands:
        raise RuntimeError("bundled libopenblas not found in janusx.libs")
    return cands[0]


def _check_macos_bundled_openblas_signature_and_probe_load() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("--check-macos-bundled-openblas only supports macOS runners")

    pkg_dir = _janusx_pkg_dir()
    libs_dir = os.path.join(os.path.dirname(pkg_dir), "janusx.libs")
    dylibs = sorted(glob.glob(os.path.join(libs_dir, "*.dylib")))
    if not dylibs:
        raise RuntimeError(f"no bundled dylibs found under {libs_dir}")

    codesign_bin = "/usr/bin/codesign" if os.path.isfile("/usr/bin/codesign") else "codesign"
    for dylib in dylibs:
        proc = subprocess.run(
            [codesign_bin, "--verify", "--verbose=2", dylib],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(
            f"[check_installed_runtime] codesign_verify={os.path.basename(dylib)} rc={proc.returncode}"
        )
        if proc.returncode != 0:
            detail = (proc.stdout or "").strip()
            raise RuntimeError(
                f"codesign verification failed for {dylib}: {detail}"
            )

    libpath = str(os.environ.get("JX_OPENBLAS_LIB_PATH", "")).strip()
    if not libpath:
        libpath = _find_bundled_openblas()
    if not os.path.isfile(libpath):
        raise RuntimeError(f"JX_OPENBLAS_LIB_PATH does not exist: {libpath}")

    print(f"[check_installed_runtime] probing_cdll={libpath}")
    _ = ctypes.CDLL(libpath)
    print("[check_installed_runtime] macOS bundled OpenBLAS load probe passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify installed janusx runtime backend and KMC availability.")
    parser.add_argument(
        "--require-openblas",
        action="store_true",
        default=False,
        help="Require rust_sgemm_backend() == openblas.",
    )
    parser.add_argument(
        "--require-threaded-openblas",
        action="store_true",
        default=False,
        help="Require bundled OpenBLAS to be threaded (not SINGLE_THREADED).",
    )
    parser.add_argument(
        "--check-kmc-load",
        action="store_true",
        default=False,
        help="Try loading janusx.kmc_bind prebuilt module with CXX disabled.",
    )
    parser.add_argument(
        "--check-macos-bundled-openblas",
        action="store_true",
        default=False,
        help=(
            "On macOS, verify codesign for bundled janusx.libs/*.dylib and probe "
            "ctypes.CDLL load for bundled OpenBLAS."
        ),
    )
    args = parser.parse_args()

    need_native_import = (
        args.require_openblas or args.require_threaded_openblas or args.check_kmc_load
    )
    if need_native_import:
        import janusx.janusx as jx

        backend = jx.rust_sgemm_backend()
        print(f"[check_installed_runtime] rust_sgemm_backend={backend}")
        if args.require_openblas and backend != "openblas":
            raise SystemExit(f"Strict mode failed: expected openblas backend, got {backend!r}")

    if args.require_threaded_openblas:
        libpath = _find_bundled_openblas()
        lib = ctypes.CDLL(libpath)
        print(f"[check_installed_runtime] bundled_openblas={libpath}")
        if not hasattr(lib, "openblas_get_config") or not hasattr(lib, "openblas_get_parallel"):
            raise SystemExit(
                "Strict mode failed: missing openblas_get_config/openblas_get_parallel symbols"
            )
        lib.openblas_get_config.restype = ctypes.c_char_p
        lib.openblas_get_parallel.restype = ctypes.c_int
        cfg_raw = lib.openblas_get_config()
        cfg = cfg_raw.decode("utf-8", "ignore") if cfg_raw else ""
        parallel = int(lib.openblas_get_parallel())
        print(f"[check_installed_runtime] openblas_config={cfg}")
        print(f"[check_installed_runtime] openblas_parallel={parallel}")
        if "SINGLE_THREADED" in cfg.upper() or parallel == 0:
            raise SystemExit(
                "Strict mode failed: OpenBLAS is SINGLE_THREADED; refuse publishing low-performance wheel"
            )

    if args.check_kmc_load:
        os.environ.setdefault("JANUSX_KMC_BIND_NO_BUILD", "1")
        os.environ.setdefault("CXX", "__janusx_disabled_cxx__")
        from janusx.kmc_bind import load_kmc_bind_module

        mod = load_kmc_bind_module(verbose=False)
        print(f"[check_installed_runtime] kmc_bind_loaded={mod.__name__}")

    if args.check_macos_bundled_openblas:
        _check_macos_bundled_openblas_signature_and_probe_load()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
