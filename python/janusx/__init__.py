from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path


_DLL_DIR_HANDLES: list[object] = []


def _init_windows_dll_search_path() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    pkg_dir = Path(__file__).resolve().parent
    cands = [
        pkg_dir,
        pkg_dir / ".libs",
        pkg_dir.parent / "janusx.libs",
    ]

    # Optional external hints (useful for local editable/dev runs).
    for var in ("OPENBLAS_BIN_DIR", "OPENBLAS_LIB_DIR", "LIBRARY_BIN", "LIBRARY_LIB"):
        raw = str(os.environ.get(var, "")).strip()
        if raw:
            cands.append(Path(raw))
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        cp = Path(conda_prefix)
        cands.extend([cp / "Library" / "bin", cp / "Library" / "lib", cp / "DLLs", cp / "bin"])

    seen: set[str] = set()
    for d in cands:
        try:
            rd = d.resolve()
        except Exception:
            rd = d
        key = str(rd).lower()
        if key in seen:
            continue
        seen.add(key)
        if not rd.exists() or not rd.is_dir():
            continue
        try:
            h = os.add_dll_directory(str(rd))
            _DLL_DIR_HANDLES.append(h)
        except Exception:
            pass


_init_windows_dll_search_path()


def _init_macos_openblas_path() -> None:
    if sys.platform != "darwin":
        return

    # Honor explicit user override first.
    explicit = str(os.environ.get("JX_OPENBLAS_LIB_PATH", "")).strip()
    if explicit:
        return

    pkg_dir = Path(__file__).resolve().parent
    cands: list[Path] = []
    cands.extend(sorted((pkg_dir / ".dylibs").glob("libopenblas*.dylib")))
    cands.extend(sorted((pkg_dir.parent / "janusx.libs").glob("libopenblas*.dylib")))
    cands.extend(sorted((pkg_dir / ".libs").glob("libopenblas*.dylib")))
    if len(cands) == 0:
        return

    # Prefer canonical soname-style names first.
    preferred_order = [
        "libopenblas.0.dylib",
        "libopenblas.dylib",
    ]
    picked: Path | None = None
    by_name = {p.name: p for p in cands}
    for name in preferred_order:
        if name in by_name:
            picked = by_name[name]
            break
    if picked is None:
        picked = cands[0]

    os.environ.setdefault("JX_OPENBLAS_LIB_PATH", str(picked))

    # Help dependent dylib resolution for @rpath/@loader_path fallbacks.
    libs_dir = str(picked.parent)
    fallback_key = "DYLD_FALLBACK_LIBRARY_PATH"
    current = str(os.environ.get(fallback_key, "")).strip()
    if current == "":
        os.environ[fallback_key] = libs_dir
    else:
        parts = current.split(":")
        if libs_dir not in parts:
            os.environ[fallback_key] = f"{libs_dir}:{current}"


_init_macos_openblas_path()

from . import linalg as linalg


def _attach_native_linalg_alias() -> None:
    """
    Expose linear algebra convenience API on native extension namespace:
        import janusx.janusx as jx
        jx.linalg.eigh(...)
    """
    try:
        from . import janusx as _jx
    except Exception:
        return
    try:
        if not hasattr(_jx, "linalg"):
            setattr(_jx, "linalg", linalg)
    except Exception:
        pass


_attach_native_linalg_alias()


def _warn_macos_eigh_lapack_fallback() -> None:
    if sys.platform != "darwin":
        return
    # Explicit user choice: do not warn on requested Accelerate LAPACK.
    raw_pref = str(
        os.environ.get(
            "JX_RUST_EIGH_LAPACK_BACKEND",
            os.environ.get("JX_RUST_LAPACK_BACKEND", "auto"),
        )
    ).strip().lower()
    if raw_pref in ("accelerate", "veclib"):
        return
    try:
        from . import janusx as _jx
    except Exception:
        return
    try:
        probe_eigh = getattr(_jx, "rust_eigh_lapack_backend", None)
        if not callable(probe_eigh):
            return
        lapack_backend = str(probe_eigh()).strip().lower()
    except Exception:
        return

    # Auto/openblas mode expects dynamic OpenBLAS LAPACK on macOS.
    # If we land on Accelerate instead, this is a fallback path and may
    # re-introduce the known "eigh slow on Accelerate" behavior.
    if lapack_backend == "accelerate":
        warnings.warn(
            "JanusX macOS LAPACK fallback: rust_eigh is using Accelerate "
            "(expected openblas_dyn in auto/openblas mode). "
            "This may make eigen decomposition slower. "
            "Check bundled OpenBLAS dylib loading and JX_OPENBLAS_LIB_PATH.",
            RuntimeWarning,
            stacklevel=2,
        )


_warn_macos_eigh_lapack_fallback()

__all__ = ["linalg"]
