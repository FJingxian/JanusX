from __future__ import annotations

import io
import math
import os
import platform
import re
from contextlib import redirect_stdout
from typing import Any


_BLAS_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_BLAS_MAX_THREAD_ENV_KEYS = (
    "OMP_MAX_THREADS",
    "MKL_MAX_THREADS",
    "OPENBLAS_MAX_THREADS",
    "NUMEXPR_MAX_THREADS",
)


def _parse_positive_env_int(name: str) -> int | None:
    raw = str(os.environ.get(name, "")).strip()
    if raw == "":
        return None
    m = re.match(r"^\s*(\d+)", raw)
    if m is None:
        return None
    val = int(m.group(1))
    return val if val > 0 else None


def _detect_cgroup_cpu_quota() -> int | None:
    # cgroup v2
    try:
        cpu_max = "/sys/fs/cgroup/cpu.max"
        if os.path.isfile(cpu_max):
            txt = open(cpu_max, "r", encoding="utf-8").read().strip().split()
            if len(txt) >= 2 and txt[0] != "max":
                quota = int(txt[0])
                period = int(txt[1])
                if quota > 0 and period > 0:
                    return max(1, int(math.ceil(float(quota) / float(period))))
    except Exception:
        pass
    # cgroup v1
    try:
        q_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        p_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        if os.path.isfile(q_path) and os.path.isfile(p_path):
            quota = int(open(q_path, "r", encoding="utf-8").read().strip())
            period = int(open(p_path, "r", encoding="utf-8").read().strip())
            if quota > 0 and period > 0:
                return max(1, int(math.ceil(float(quota) / float(period))))
    except Exception:
        pass
    return None


def detect_effective_threads() -> int:
    """
    Detect usable CPU thread count in HPC/container environments.

    Priority:
      1) Scheduler allocation vars (SLURM/PBS/LSF/SGE)
      2) CPU affinity mask
      3) cgroup CPU quota
      4) os.cpu_count()
    Then cap by common math-thread env vars when present.
    """
    scheduler_envs = [
        "SLURM_CPUS_PER_TASK",
        "PBS_NP",
        "LSB_DJOB_NUMPROC",
        "NSLOTS",
        "NCPUS",
    ]
    detected: int | None = None
    for name in scheduler_envs:
        v = _parse_positive_env_int(name)
        if v is not None:
            detected = v
            break

    # Affinity-aware fallback
    if detected is None:
        try:
            if hasattr(os, "sched_getaffinity"):
                aff = os.sched_getaffinity(0)  # type: ignore[attr-defined]
                if aff is not None and len(aff) > 0:
                    detected = int(len(aff))
        except Exception:
            pass

    # cgroup quota fallback (containers)
    if detected is None:
        detected = _detect_cgroup_cpu_quota()

    # Last fallback: host-visible cores
    if detected is None:
        detected = int(os.cpu_count() or 1)

    # Optional software caps
    cap_envs = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]
    caps = [v for v in (_parse_positive_env_int(k) for k in cap_envs) if v is not None]
    if len(caps) > 0:
        detected = min(detected, min(caps))

    return max(1, int(detected))


def apply_blas_thread_env(
    threads: int,
    *,
    set_jx_threads: bool = True,
    set_max_keys: bool = True,
) -> int:
    """
    Apply a unified BLAS/OpenMP thread cap for current process.

    Returns the normalized positive thread count.
    """
    t = max(1, int(threads))
    text = str(int(t))
    for key in _BLAS_THREAD_ENV_KEYS:
        os.environ[key] = text
    if set_max_keys:
        for key in _BLAS_MAX_THREAD_ENV_KEYS:
            os.environ[key] = text
    if set_jx_threads:
        os.environ["JX_THREADS"] = text
    # Keep MLM runtime knobs aligned with global thread cap by default.
    os.environ["JX_MLM_BLAS_THREADS"] = text
    return t


def detect_blas_backend() -> str:
    """
    Best-effort BLAS backend detection for NumPy/SciPy runtime.

    Returns one of:
      openblas / mkl / accelerate / blis / atlas / unknown
    """
    def _detect_from_threadpool_info() -> str:
        try:
            from threadpoolctl import threadpool_info  # type: ignore

            infos = list(threadpool_info())
            for rec in infos:
                text = " ".join(
                    str(rec.get(k, ""))
                    for k in ("internal_api", "user_api", "prefix", "filepath", "version")
                ).lower()
                if "openblas" in text:
                    return "openblas"
                if ("mkl" in text) or ("intel" in text and "blas" in text):
                    return "mkl"
                if "accelerate" in text or "veclib" in text:
                    return "accelerate"
                if "blis" in text:
                    return "blis"
                if "atlas" in text:
                    return "atlas"
        except Exception:
            pass
        return "unknown"

    # 1) Prefer runtime-loaded backend info when available.
    b = _detect_from_threadpool_info()
    if b != "unknown":
        return b

    # 1b) Some runtimes don't expose BLAS in threadpool_info until NumPy calls
    # into BLAS once. Trigger a tiny matmul and retry detection.
    try:
        import numpy as _np

        z = _np.zeros((1, 1), dtype=_np.float64)
        _ = z @ z
        b = _detect_from_threadpool_info()
        if b != "unknown":
            return b
    except Exception:
        pass

    # 2) Fallback to NumPy build config text.
    try:
        import numpy as _np

        buf = io.StringIO()
        with redirect_stdout(buf):
            _np.show_config()
        cfg = buf.getvalue().lower()
        name_hits = re.findall(r"^\s*name:\s*([a-z0-9_+.-]+)\s*$", cfg, flags=re.MULTILINE)
        if any("openblas" in x for x in name_hits):
            return "openblas"
        if any(("mkl" in x) or ("intel" in x) for x in name_hits):
            return "mkl"
        if any(("accelerate" in x) or ("veclib" in x) for x in name_hits):
            return "accelerate"
        if any("blis" in x for x in name_hits):
            return "blis"
        if any("atlas" in x for x in name_hits):
            return "atlas"
        # Last-resort text hints (guarded after name-based checks to avoid
        # false positives from "openblas configuration: unknown" in Accelerate builds).
        if ("libopenblas" in cfg) or ("openblas_" in cfg) or ("openblas-" in cfg):
            return "openblas"
        if ("accelerate" in cfg) or ("veclib" in cfg):
            return "accelerate"
    except Exception:
        pass

    return "unknown"


def preferred_blas_backends() -> tuple[str, ...]:
    """
    Platform-aware preferred BLAS backend order (best to acceptable).
    """
    system = platform.system().lower()
    if system == "darwin":
        # Accelerate is generally available on macOS, while OpenBLAS remains
        # the preferred backend for cross-platform consistency.
        return ("openblas", "mkl", "accelerate", "blis", "atlas", "unknown")
    if system == "windows":
        return ("openblas", "mkl", "blis", "atlas", "accelerate", "unknown")
    # Linux/other Unix-like
    return ("openblas", "mkl", "blis", "atlas", "accelerate", "unknown")


def maybe_warn_non_openblas(
    *,
    logger: Any | None = None,
    strict: bool = False,
) -> str:
    """
    Emit warning (or raise) when runtime BLAS is not OpenBLAS.

    Default behavior should be "prefer OpenBLAS, but fallback is allowed".
    Set strict=True for fail-fast enforcement.
    """
    backend = str(detect_blas_backend())
    if backend == "openblas":
        return backend
    priority = preferred_blas_backends()
    try:
        rank = int(priority.index(backend)) + 1
    except ValueError:
        rank = len(priority) + 1
    platform_name = platform.system() or "current platform"
    strict_msg = (
        f"BLAS backend detected as '{backend}', not openblas. "
        "OpenBLAS is required in strict mode. "
        "Install/use an OpenBLAS environment."
    )
    if strict:
        raise RuntimeError(strict_msg)
    prefer_chain = ", ".join(priority[:-1])
    msg = (
        f"BLAS backend detected as '{backend}'. "
        f"OpenBLAS is preferred; fallback accepted (platform={platform_name}, "
        f"priority={rank}/{len(priority)}; order: {prefer_chain})."
    )
    if logger is not None:
        try:
            logger.warning(msg)
        except Exception:
            pass
    return backend


def require_openblas_by_default() -> bool:
    """
    Compatibility helper for existing call sites.

    Default policy is warning + fallback (non-strict).
    This no longer depends on external environment flags.
    """
    return False
