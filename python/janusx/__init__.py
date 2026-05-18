from __future__ import annotations

import os
import sys
import tempfile
import time
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
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "")).strip()
    if conda_prefix:
        cands.extend(sorted((Path(conda_prefix) / "lib").glob("libopenblas*.dylib")))
    for root in (
        Path("/opt/homebrew/opt/openblas/lib"),
        Path("/usr/local/opt/openblas/lib"),
    ):
        cands.extend(sorted(root.glob("libopenblas*.dylib")))
    if len(cands) == 0:
        return

    # Prefer canonical soname-style names first.
    preferred_order = [
        "libopenblas.0.dylib",
        "libopenblas.dylib",
    ]
    picked: Path | None = None
    for name in preferred_order:
        for cand in cands:
            if cand.name == name:
                picked = cand
                break
        if picked is not None:
            break
    if picked is None:
        picked = cands[0]

    os.environ.setdefault("JX_OPENBLAS_LIB_PATH", str(picked))


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


_EIGH_WARN_ONCE_ENV = "JANUSX_EIGH_FALLBACK_WARNED"
_EIGH_WARN_SENTINEL_TTL_SEC = 300.0


def _env_truthy(name: str) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _stderr_is_interactive() -> bool:
    try:
        return bool(getattr(sys.stderr, "isatty", lambda: False)())
    except Exception:
        return False


def _clear_current_stdout_line() -> None:
    try:
        if not bool(getattr(sys.stdout, "isatty", lambda: False)()):
            return
    except Exception:
        return
    try:
        cols = int(getattr(os, "get_terminal_size", lambda *_: os.terminal_size((80, 20)))().columns)
    except Exception:
        cols = 80
    cols = max(1, int(cols))
    try:
        sys.stdout.write("\r" + (" " * cols) + "\r")
        sys.stdout.flush()
    except Exception:
        pass


def _eigh_warn_sentinel_file() -> str:
    try:
        uid = str(int(os.getuid()))
    except Exception:
        uid = "nouid"
    return os.path.join(tempfile.gettempdir(), f".janusx_eigh_fallback_warned_{uid}")


def _is_macos_eigh_fallback_backend(backend_label: str) -> bool:
    b = str(backend_label).strip().lower()
    if b == "accelerate":
        return True
    return "accelerate" in b and b.startswith("lapack_")


def _should_check_macos_eigh_hint() -> bool:
    if sys.platform != "darwin":
        return False
    raw_pref = str(
        os.environ.get(
            "JX_RUST_EIGH_LAPACK_BACKEND",
            os.environ.get("JX_RUST_LAPACK_BACKEND", "auto"),
        )
    ).strip().lower()
    # Explicit Accelerate LAPACK request is intentional.
    if raw_pref in ("accelerate", "veclib"):
        return False
    return True


def _probe_macos_eigh_backend() -> str:
    try:
        from . import janusx as _jx
    except Exception:
        return ""
    try:
        probe_sgemm = getattr(_jx, "rust_sgemm_backend", None)
        probe_eigh = getattr(_jx, "rust_eigh_lapack_backend", None)
        if not callable(probe_sgemm):
            return ""
        sgemm_backend = str(probe_sgemm()).strip().lower()
        # OpenBLAS BLAS builds should not trigger fallback hints.
        if sgemm_backend != "accelerate":
            return ""
        if not callable(probe_eigh):
            return ""
        return str(probe_eigh()).strip().lower()
    except Exception:
        return ""


def maybe_emit_macos_eigh_fallback_hint(
    *,
    evd_backend: str | None = None,
    logger=None,
) -> bool:
    """
    Emit a gentle one-time hint when macOS eigendecomposition falls back to
    Accelerate LAPACK in auto/openblas mode.

    Returns True when the hint was emitted in this call.
    """
    if not _should_check_macos_eigh_hint():
        return False
    if sys.platform != "darwin":
        return False
    # Process-tree once-only sentinel.
    if _env_truthy(_EIGH_WARN_ONCE_ENV):
        return False
    # Cross-process once-only sentinel (pipeline/subprocess chains) is only
    # applied in non-interactive mode. Interactive CLI invocations should still
    # surface the warning once per run.
    use_cross_process_sentinel = not _stderr_is_interactive()
    sentinel_path = _eigh_warn_sentinel_file() if use_cross_process_sentinel else ""
    if use_cross_process_sentinel and os.path.exists(sentinel_path):
        try:
            age = max(0.0, float(time.time()) - float(os.path.getmtime(sentinel_path)))
            if age <= float(_EIGH_WARN_SENTINEL_TTL_SEC):
                return False
        except Exception:
            return False

    backend = (
        str(evd_backend).strip().lower()
        if evd_backend is not None and str(evd_backend).strip() != ""
        else _probe_macos_eigh_backend()
    )
    if not _is_macos_eigh_fallback_backend(backend):
        return False

    os.environ[_EIGH_WARN_ONCE_ENV] = "1"
    if use_cross_process_sentinel:
        try:
            with open(sentinel_path, "w", encoding="utf-8") as fh:
                fh.write("1\n")
        except Exception:
            pass

    backend_line = "  backend: Accelerate LAPACK"
    warning_body = (
        "OpenBLAS LAPACK was unavailable or skipped due to a runtime conflict; "
        "JanusX used Accelerate instead.\n"
        "  Set JX_RUST_EIGH_LAPACK_BACKEND=accelerate to silence this fallback."
    )
    _clear_current_stdout_line()
    if logger is not None and hasattr(logger, "info"):
        logger.info(backend_line)
        if hasattr(logger, "warning"):
            logger.warning(warning_body)
        else:
            logger.info(f"Warning: {warning_body}")
    else:
        sys.stderr.write(backend_line + "\n")
        sys.stderr.write("Warning: " + warning_body + "\n")
        sys.stderr.flush()
    return True


__all__ = ["linalg", "maybe_emit_macos_eigh_fallback_hint"]
