#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import os
from pathlib import Path
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
        + glob.glob(os.path.join(pkg_dir, "openblas*.dll"))
        + glob.glob(os.path.join(pkg_dir, "libopenblas*.dll"))
    )
    if not cands:
        raise RuntimeError("bundled libopenblas not found in janusx.libs")
    return cands[0]


def _load_openblas_cdll(libpath: str) -> ctypes.CDLL:
    # On Windows, ensure package-local dependent DLLs (libgfortran, etc.)
    # are discoverable before loading openblas.dll.
    if sys.platform == "win32":
        pkg_dir = _janusx_pkg_dir()
        add_dir = getattr(os, "add_dll_directory", None)
        if callable(add_dir):
            add_dir(pkg_dir)
    return ctypes.CDLL(libpath)


def _check_macos_bundled_openblas_signature_and_probe_load() -> None:
    if sys.platform != "darwin":
        raise RuntimeError("--check-macos-bundled-openblas only supports macOS runners")

    pkg_dir = _janusx_pkg_dir()
    libs_dir = os.path.join(os.path.dirname(pkg_dir), "janusx.libs")
    dylibs = sorted(glob.glob(os.path.join(libs_dir, "*.dylib")))
    if not dylibs:
        raise RuntimeError(f"no bundled dylibs found under {libs_dir}")
    dylib_names = {os.path.basename(p) for p in dylibs}

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

    def _otool_deps(path: str) -> list[str]:
        proc = subprocess.run(
            ["otool", "-L", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines()[1:]]
        out: list[str] = []
        for ln in lines:
            if not ln:
                continue
            out.append(ln.split(" (", 1)[0].strip())
        return out

    libopenblas_path = _find_bundled_openblas()
    if os.path.basename(libopenblas_path) not in dylib_names:
        raise RuntimeError(
            f"bundled OpenBLAS path is not present under janusx.libs: {libopenblas_path}"
        )

    # Require a self-contained wheel-local dependency graph. If bundled
    # OpenBLAS still references host-specific @rpath libs, runtime may silently
    # pick up Homebrew/Conda copies and reintroduce mixed-BLAS crashes.
    openblas_deps = _otool_deps(libopenblas_path)
    expected_bundle_dep_names = {
        Path(dep).name
        for dep in openblas_deps
        if not dep.startswith("/usr/lib/") and not dep.startswith("/System/Library/")
    }
    missing = sorted(name for name in expected_bundle_dep_names if name not in dylib_names)
    if missing:
        raise RuntimeError(
            "bundled macOS OpenBLAS closure is incomplete; missing "
            + ", ".join(missing)
        )

    for dylib in dylibs:
        deps = _otool_deps(dylib)
        for dep in deps:
            if dep.startswith("/usr/lib/") or dep.startswith("/System/Library/"):
                continue
            dep_name = Path(dep).name
            if dep_name not in dylib_names:
                raise RuntimeError(
                    f"bundled dylib {os.path.basename(dylib)} references non-bundled dependency {dep}"
                )
            expected = f"@loader_path/{dep_name}"
            if dep != expected:
                raise RuntimeError(
                    f"bundled dylib {os.path.basename(dylib)} should reference {expected}, got {dep}"
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
    parser = argparse.ArgumentParser(description="Verify installed janusx runtime backend and Rust KMC entrypoints.")
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
        "--check-kmer-count-run",
        action="store_true",
        default=False,
        help="Require janusx.janusx to export kmer_count_run().",
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
        lib = _load_openblas_cdll(libpath)
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
        # Prefer runtime parallel flag as the decisive signal:
        #   0 => no threading, 1 => pthreads, 2 => OpenMP.
        # Some binaries may still contain "SINGLE_THREADED" text in their
        # string table, so treat it as advisory unless runtime reports
        # non-parallel behavior.
        if parallel == 0:
            raise SystemExit(
                "Strict mode failed: OpenBLAS runtime reports no threading "
                f"(openblas_get_parallel={parallel}, config={cfg!r})."
            )

    if args.check_kmc_load:
        has_symbol = hasattr(jx, "kmer_count_run")
        print(f"[check_installed_runtime] kmer_count_run={has_symbol}")
        if not has_symbol:
            raise SystemExit("Strict mode failed: janusx.janusx lacks kmer_count_run()")

    if args.check_macos_bundled_openblas:
        _check_macos_bundled_openblas_signature_and_probe_load()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
