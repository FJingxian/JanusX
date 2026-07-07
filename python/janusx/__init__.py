from __future__ import annotations

import os
import site
import sys
import tempfile
import time
from pathlib import Path


_DLL_DIR_HANDLES: list[object] = []
_NATIVE_EXT_PATTERNS = ("janusx*.so", "janusx*.pyd")
_RUNTIME_CACHE_ROOT_ENV = "JANUSX_RUNTIME_CACHE_DIR"


def _runtime_cache_root() -> Path:
    explicit = str(os.environ.get(_RUNTIME_CACHE_ROOT_ENV, "")).strip()
    if explicit != "":
        return Path(explicit)
    try:
        uid = str(int(os.getuid()))
    except Exception:
        uid = "nouid"
    return Path(tempfile.gettempdir()) / f"janusx-runtime-{uid}"


def _path_writable_or_creatable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    probe = path / ".janusx-write-probe"
    try:
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("1")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _default_xdg_cache_home() -> Path:
    try:
        home = Path.home()
    except Exception:
        return _runtime_cache_root() / "xdg-cache"
    return home / ".cache"


def _default_mplconfigdir() -> Path:
    try:
        home = Path.home()
    except Exception:
        return _runtime_cache_root() / "mplconfig"
    return home / ".matplotlib"


def _ensure_runtime_cache_dirs() -> None:
    """
    Keep plotting/cache-heavy imports away from unwritable home directories.

    This avoids slow matplotlib/fontconfig fallback behavior in shared or
    locked-down environments (for example conda base, HPC login nodes, or
    read-only home mounts) before GWAS/GS modules import matplotlib.
    """
    cache_root = _runtime_cache_root()

    raw_xdg = str(os.environ.get("XDG_CACHE_HOME", "")).strip()
    xdg_path = Path(raw_xdg) if raw_xdg != "" else _default_xdg_cache_home()
    if not _path_writable_or_creatable(xdg_path):
        xdg_path = cache_root / "xdg-cache"
        if _path_writable_or_creatable(xdg_path):
            os.environ["XDG_CACHE_HOME"] = str(xdg_path)
    try:
        (xdg_path / "fontconfig").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    raw_mpl = str(os.environ.get("MPLCONFIGDIR", "")).strip()
    mpl_path = Path(raw_mpl) if raw_mpl != "" else _default_mplconfigdir()
    if not _path_writable_or_creatable(mpl_path):
        mpl_path = xdg_path / "matplotlib"
        if _path_writable_or_creatable(mpl_path):
            os.environ["MPLCONFIGDIR"] = str(mpl_path)


_ensure_runtime_cache_dirs()


def _package_dir_has_native_extension(pkg_dir: Path) -> bool:
    try:
        if not pkg_dir.is_dir():
            return False
    except Exception:
        return False
    for pattern in _NATIVE_EXT_PATTERNS:
        try:
            for hit in pkg_dir.glob(pattern):
                try:
                    if hit.is_file():
                        return True
                except Exception:
                    continue
        except Exception:
            continue
    return False


def _iter_janusx_package_dirs() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent
    pkg_name = pkg_dir.name
    roots: list[str] = []
    roots.extend(str(x) for x in sys.path)
    try:
        roots.extend(str(x) for x in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            roots.append(user_site)
        else:
            roots.extend(str(x) for x in user_site)
    except Exception:
        pass

    out: list[Path] = []
    seen: set[str] = set()
    for raw in roots:
        text = str(raw).strip()
        if text == "":
            text = os.getcwd()
        root = Path(text)
        candidates = [root / pkg_name]
        try:
            if root.name == pkg_name and root.is_dir():
                candidates.insert(0, root)
        except Exception:
            pass
        for cand in candidates:
            try:
                resolved = cand.resolve()
            except Exception:
                resolved = cand
            key = str(resolved)
            if key in seen or resolved == pkg_dir:
                continue
            seen.add(key)
            try:
                if resolved.is_dir():
                    out.append(resolved)
            except Exception:
                continue
    return out


def _extend_package_path_for_native_extension() -> None:
    pkg_dir = Path(__file__).resolve().parent
    if _package_dir_has_native_extension(pkg_dir):
        return
    pkg_path = globals().get("__path__")
    if pkg_path is None:
        return
    for cand in _iter_janusx_package_dirs():
        if not _package_dir_has_native_extension(cand):
            continue
        cand_s = str(cand)
        try:
            present = {str(x) for x in pkg_path}
        except Exception:
            present = set()
        if cand_s not in present:
            pkg_path.append(cand_s)
        return


_extend_package_path_for_native_extension()


def _init_windows_dll_search_path() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    pkg_dirs: list[Path] = []
    for raw in globals().get("__path__", [Path(__file__).resolve().parent]):
        try:
            pkg_dirs.append(Path(str(raw)))
        except Exception:
            continue
    if len(pkg_dirs) == 0:
        pkg_dirs = [Path(__file__).resolve().parent]

    cands: list[Path] = []
    for pkg_dir in pkg_dirs:
        cands.extend([
            pkg_dir,
            pkg_dir / ".libs",
            pkg_dir.parent / "janusx.libs",
        ])

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

    pkg_dirs: list[Path] = []
    for raw in globals().get("__path__", [Path(__file__).resolve().parent]):
        try:
            pkg_dirs.append(Path(str(raw)))
        except Exception:
            continue
    if len(pkg_dirs) == 0:
        pkg_dirs = [Path(__file__).resolve().parent]
    cands: list[Path] = []
    for pkg_dir in pkg_dirs:
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


def _should_prewarm_macos_native_eigh_backend() -> bool:
    if sys.platform != "darwin":
        return False
    raw_pref = str(
        os.environ.get(
            "JX_RUST_EIGH_LAPACK_BACKEND",
            os.environ.get("JX_RUST_LAPACK_BACKEND", "auto"),
        )
    ).strip().lower()
    # Respect explicit Accelerate requests.
    if raw_pref in ("accelerate", "veclib"):
        return False
    return True


def _prewarm_macos_native_eigh_backend() -> str:
    """
    Best-effort early native-LAPACK prewarm on macOS.

    Import the JanusX extension and probe the Rust eigh backend before heavy
    optional dependencies (SciPy/sklearn/xgboost/matplotlib) can pull a
    foreign OpenMP runtime into the process and force auto-selection back to
    Accelerate.
    """
    if not _should_prewarm_macos_native_eigh_backend():
        return ""
    try:
        from . import janusx as _jx
    except Exception:
        return ""
    try:
        probe = getattr(_jx, "rust_eigh_lapack_backend", None)
        if not callable(probe):
            return ""
        return str(probe()).strip().lower()
    except Exception:
        return ""


_ = _prewarm_macos_native_eigh_backend()

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
