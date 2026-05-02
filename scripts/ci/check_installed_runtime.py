#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import glob
import os


def _find_bundled_openblas() -> str:
    import janusx.janusx as jx

    root = os.path.dirname(jx.__file__)
    cands = sorted(glob.glob(os.path.join(os.path.dirname(root), "janusx.libs", "libopenblas*.so*")))
    if not cands:
        raise RuntimeError("bundled libopenblas not found in janusx.libs")
    return cands[0]


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
    args = parser.parse_args()

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
