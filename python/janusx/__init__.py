from __future__ import annotations

import os
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

__all__ = ["linalg"]
