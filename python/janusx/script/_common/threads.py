from __future__ import annotations

import io
import math
import os
import platform
import re
import warnings
from contextlib import contextmanager, nullcontext
from contextlib import redirect_stdout
from typing import Any

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None


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
            with open(cpu_max, "r", encoding="utf-8") as fh:
                txt = fh.read().strip().split()
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
            with open(q_path, "r", encoding="utf-8") as fh:
                quota = int(fh.read().strip())
            with open(p_path, "r", encoding="utf-8") as fh:
                period = int(fh.read().strip())
            if quota > 0 and period > 0:
                return max(1, int(math.ceil(float(quota) / float(period))))
    except Exception:
        pass
    return None


def _current_host_aliases() -> set[str]:
    aliases: set[str] = set()
    for raw in (platform.node(), getattr(os, "uname", lambda: None)() and os.uname().nodename):
        text = str(raw or "").strip().lower()
        if not text:
            continue
        aliases.add(text)
        aliases.add(text.split(".", 1)[0])
    return {x for x in aliases if x}


def _host_matches_local(raw_host: str, aliases: set[str]) -> bool:
    text = str(raw_host or "").strip().lower()
    if not text:
        return False
    short = text.split(".", 1)[0]
    return text in aliases or short in aliases


def _detect_affinity_threads() -> int | None:
    try:
        if hasattr(os, "sched_getaffinity"):
            aff = os.sched_getaffinity(0)  # type: ignore[attr-defined]
            if aff is not None and len(aff) > 0:
                return int(len(aff))
    except Exception:
        pass
    return None


def _detect_affinity_cpus() -> list[int] | None:
    try:
        if hasattr(os, "sched_getaffinity"):
            aff = os.sched_getaffinity(0)  # type: ignore[attr-defined]
            if aff is not None and len(aff) > 0:
                return sorted(int(x) for x in aff)
    except Exception:
        pass
    return None


def _detect_lsf_local_slots() -> int | None:
    raw = str(os.environ.get("LSB_MCPU_HOSTS", "")).strip().split()
    if len(raw) < 2:
        return None
    aliases = _current_host_aliases()
    for i in range(0, len(raw) - 1, 2):
        host = raw[i]
        try:
            slots = int(raw[i + 1])
        except Exception:
            continue
        if slots > 0 and _host_matches_local(host, aliases):
            return slots
    return None


def _detect_pbs_local_slots() -> int | None:
    nodefile = str(os.environ.get("PBS_NODEFILE", "")).strip()
    if not nodefile or not os.path.isfile(nodefile):
        return None
    aliases = _current_host_aliases()
    count = 0
    try:
        with open(nodefile, "r", encoding="utf-8") as fh:
            for line in fh:
                if _host_matches_local(line.strip(), aliases):
                    count += 1
    except Exception:
        return None
    return count if count > 0 else None


def _detect_sge_local_slots() -> int | None:
    hostfile = str(os.environ.get("PE_HOSTFILE", "")).strip()
    if not hostfile or not os.path.isfile(hostfile):
        return None
    aliases = _current_host_aliases()
    try:
        with open(hostfile, "r", encoding="utf-8") as fh:
            for line in fh:
                toks = line.strip().split()
                if len(toks) < 2:
                    continue
                try:
                    slots = int(toks[1])
                except Exception:
                    continue
                if slots > 0 and _host_matches_local(toks[0], aliases):
                    return slots
    except Exception:
        return None
    return None


def _first_positive_named(items: list[tuple[str, int | None]]) -> tuple[int | None, str | None]:
    for name, value in items:
        if value is not None and int(value) > 0:
            return int(value), str(name)
    return None, None


def detect_thread_budget_info() -> dict[str, Any]:
    scheduler_local_candidates = [
        ("SLURM_CPUS_ON_NODE", _parse_positive_env_int("SLURM_CPUS_ON_NODE")),
        ("LSB_MCPU_HOSTS", _detect_lsf_local_slots()),
        ("PBS_NODEFILE", _detect_pbs_local_slots()),
        ("PE_HOSTFILE", _detect_sge_local_slots()),
    ]
    scheduler_total_candidates = [
        ("SLURM_CPUS_PER_TASK", _parse_positive_env_int("SLURM_CPUS_PER_TASK")),
        ("PBS_NP", _parse_positive_env_int("PBS_NP")),
        ("LSB_DJOB_NUMPROC", _parse_positive_env_int("LSB_DJOB_NUMPROC")),
        ("NSLOTS", _parse_positive_env_int("NSLOTS")),
        ("NCPUS", _parse_positive_env_int("NCPUS")),
    ]

    scheduler_local_threads, scheduler_local_source = _first_positive_named(
        scheduler_local_candidates
    )
    scheduler_total_threads, scheduler_total_source = _first_positive_named(
        scheduler_total_candidates
    )
    affinity_cpus = _detect_affinity_cpus()
    affinity_threads = _detect_affinity_threads()
    cgroup_threads = _detect_cgroup_cpu_quota()
    host_visible_threads = max(1, int(os.cpu_count() or 1))

    effective_candidates: list[tuple[str, int]] = []
    for source, value in (
        ("scheduler_local", scheduler_local_threads),
        ("scheduler_total", scheduler_total_threads),
        ("affinity", affinity_threads),
        ("cgroup", cgroup_threads),
        ("host_visible", host_visible_threads),
    ):
        if value is not None and int(value) > 0:
            effective_candidates.append((str(source), int(value)))
    effective_threads = min(value for _, value in effective_candidates)
    effective_sources = tuple(
        source for source, value in effective_candidates if int(value) == int(effective_threads)
    )

    return {
        "scheduler_local_threads": scheduler_local_threads,
        "scheduler_local_source": scheduler_local_source,
        "scheduler_total_threads": scheduler_total_threads,
        "scheduler_total_source": scheduler_total_source,
        "affinity_cpus": affinity_cpus,
        "affinity_threads": affinity_threads,
        "cgroup_threads": cgroup_threads,
        "host_visible_threads": host_visible_threads,
        "effective_threads": int(effective_threads),
        "effective_sources": effective_sources,
    }


def format_thread_budget_summary(info: dict[str, Any]) -> str:
    def _fmt(value: Any, source: Any = None) -> str:
        if value is None:
            return "NA"
        if source is None or str(source).strip() == "":
            return str(value)
        return f"{value} [{source}]"

    effective_sources = info.get("effective_sources") or ()
    effective_src_text = ",".join(str(x) for x in effective_sources if str(x).strip())
    return (
        "scheduler total="
        f"{_fmt(info.get('scheduler_total_threads'), info.get('scheduler_total_source'))}, "
        "scheduler local="
        f"{_fmt(info.get('scheduler_local_threads'), info.get('scheduler_local_source'))}, "
        f"affinity={_fmt(info.get('affinity_threads'))}, "
        f"cgroup={_fmt(info.get('cgroup_threads'))}, "
        f"host_visible={_fmt(info.get('host_visible_threads'))}, "
        "local effective="
        f"{_fmt(info.get('effective_threads'), effective_src_text)}"
    )


def format_affinity_cpu_summary(info: dict[str, Any]) -> str:
    cpus = info.get("affinity_cpus")
    if not isinstance(cpus, (list, tuple)) or len(cpus) == 0:
        return "Affinity cpus: NA"
    vals = [int(x) for x in cpus]
    if len(vals) <= 8:
        body = ",".join(str(x) for x in vals)
    else:
        body = f"{vals[0]},{vals[1]},{vals[2]},...,{vals[-1]}"
    return f"Affinity cpus: {len(vals)} -> [{body}]"


def detect_effective_threads() -> int:
    """
    Detect usable CPU thread count in HPC/container environments.

    Use the most restrictive positive local signal among:
      1) Scheduler local-slot hints
      2) Scheduler allocation vars
      3) CPU affinity mask
      4) cgroup CPU quota
      5) os.cpu_count()

    Important:
      Pre-existing BLAS/OpenMP thread env vars are intentionally *not* used to
      shrink the detected allocation here. Those env vars are runtime caps and
      may reflect stale shell defaults (for example OPENBLAS_NUM_THREADS=8)
      rather than the scheduler allocation for the current job. CLI launchers
      apply the chosen thread budget back to BLAS/Rayon explicitly after
      detection.
    """
    return int(detect_thread_budget_info()["effective_threads"])


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


def apply_outer_thread_cap(
    threads: int,
    *,
    set_jx_threads: bool = True,
    set_max_keys: bool = True,
) -> int:
    """
    Apply a unified outer thread cap for both BLAS/OpenMP and Rust/Rayon.

    This is the preferred "single knob" mode where CLI `-t` acts as a total
    process-level cap, and inner stages may temporarily override sub-runtimes.
    """
    t = apply_blas_thread_env(
        threads,
        set_jx_threads=bool(set_jx_threads),
        set_max_keys=bool(set_max_keys),
    )
    text = str(int(t))
    os.environ["RAYON_NUM_THREADS"] = text
    os.environ["JX_MLM_RUST_THREADS"] = text
    return t


@contextmanager
def native_blas_thread_limit(limit: int):
    """
    Best-effort BLAS threadpool limit context (threadpoolctl-backed).
    """
    lim = max(1, int(limit))
    if _threadpool_limits is None:
        yield
        return
    try:
        stage_ctx = _threadpool_limits(limits=lim, user_api="blas")
    except TypeError:
        stage_ctx = _threadpool_limits(limits=lim)
    with stage_ctx:
        yield


@contextmanager
def runtime_thread_stage(
    *,
    blas_threads: int | None = None,
    rayon_threads: int | None = None,
):
    """
    Stage-scoped runtime thread control.

    - `blas_threads`: caps BLAS/OpenMP env + threadpoolctl limit.
    - `rayon_threads`: caps Rust/Rayon default pool via env.
    """
    updates: dict[str, str] = {}
    if blas_threads is not None:
        bt = str(max(1, int(blas_threads)))
        for key in _BLAS_THREAD_ENV_KEYS:
            updates[key] = bt
        updates["JX_MLM_BLAS_THREADS"] = bt
    if rayon_threads is not None:
        rt = str(max(1, int(rayon_threads)))
        updates["RAYON_NUM_THREADS"] = rt
        updates["JX_MLM_RUST_THREADS"] = rt

    old_env: dict[str, str | None] = {}
    prev_rust_blas_threads: int | None = None
    for key, val in updates.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = str(val)

    try:
        if blas_threads is not None:
            prev_rust_blas_threads = get_rust_blas_threads()
            set_rust_blas_threads(int(blas_threads))
        stage_ctx = (
            native_blas_thread_limit(int(blas_threads))
            if blas_threads is not None
            else nullcontext()
        )
        with stage_ctx:
            yield
    finally:
        if prev_rust_blas_threads is not None and prev_rust_blas_threads > 0:
            set_rust_blas_threads(int(prev_rust_blas_threads))
        for key, old_val in old_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


def detect_blas_backend() -> str:
    """
    Best-effort BLAS backend detection for NumPy/SciPy runtime.

    Returns one of:
      openblas / mkl / accelerate / blis / atlas / unknown
    """
    def _backend_from_name(text: str) -> str:
        t = str(text).lower()
        if "openblas" in t:
            return "openblas"
        if ("mkl" in t) or ("intel" in t and "blas" in t):
            return "mkl"
        if "accelerate" in t or "veclib" in t:
            return "accelerate"
        if "blis" in t:
            return "blis"
        if "atlas" in t:
            return "atlas"
        return "unknown"

    def _detect_from_threadpool_info() -> str:
        try:
            from threadpoolctl import threadpool_info  # type: ignore

            infos = list(threadpool_info())
            for rec in infos:
                text = " ".join(
                    str(rec.get(k, ""))
                    for k in ("internal_api", "user_api", "prefix", "filepath", "version")
                ).lower()
                b = _backend_from_name(text)
                if b != "unknown":
                    return b
        except Exception:
            pass
        return "unknown"

    def _detect_from_numpy_config_dict() -> str:
        try:
            import numpy as _np

            cfg_mod = getattr(_np, "__config__", None)
            cfg = getattr(cfg_mod, "CONFIG", None)
            if not isinstance(cfg, dict):
                return "unknown"

            build_deps = cfg.get("Build Dependencies", {})
            if isinstance(build_deps, dict):
                for key in ("blas", "lapack"):
                    info = build_deps.get(key, {})
                    if isinstance(info, dict):
                        b = _backend_from_name(str(info.get("name", "")))
                        if b != "unknown":
                            return b

            # Fallback on values-only deep scan (avoid false positives from keys
            # like "openblas configuration" in Accelerate entries).
            values: list[str] = []

            def _collect_vals(obj: Any) -> None:
                if isinstance(obj, dict):
                    for v in obj.values():
                        _collect_vals(v)
                    return
                if isinstance(obj, (list, tuple, set)):
                    for v in obj:
                        _collect_vals(v)
                    return
                values.append(str(obj))

            _collect_vals(cfg)
            blob = " ".join(values).lower()
            if (
                "scipy-openblas" in blob
                or "libopenblas" in blob
                or "openblas64" in blob
                or "openblas_" in blob
                or "openblas-" in blob
            ):
                return "openblas"
            b = _backend_from_name(blob)
            if b != "unknown":
                return b
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

    # 2) Prefer structured NumPy config when available (NumPy>=2).
    b = _detect_from_numpy_config_dict()
    if b != "unknown":
        return b

    # 3) Fallback to NumPy textual config.
    try:
        import numpy as _np

        buf = io.StringIO()
        # NumPy may warn "Install `pyyaml` for better output" when emitting
        # textual config. This is informational only and should not surface
        # during normal JanusX execution.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Install `pyyaml` for better output",
                category=UserWarning,
            )
            with redirect_stdout(buf):
                _np.show_config()
        cfg = buf.getvalue().lower()
        name_hits = re.findall(
            r"^\s*(?:['\"]?name['\"]?)\s*[:=]\s*['\"]?([a-z0-9_+.-]+)",
            cfg,
            flags=re.MULTILINE,
        )
        for hit in name_hits:
            b = _backend_from_name(hit)
            if b != "unknown":
                return b
        # Last-resort text hints (guarded after name-based checks to avoid
        # false positives from "openblas configuration: unknown" in Accelerate builds).
        if (
            ("scipy-openblas" in cfg)
            or ("libopenblas" in cfg)
            or ("openblas64" in cfg)
            or ("openblas_" in cfg)
            or ("openblas-" in cfg)
        ):
            return "openblas"
        b = _backend_from_name(cfg)
        if b != "unknown":
            return b
    except Exception:
        pass

    return "unknown"


def detect_rust_blas_backend() -> str:
    """
    Best-effort Rust BLAS/LAPACK backend detection from JanusX extension.

    Returns one of:
      openblas / accelerate / blas / rust / unknown
    """
    try:
        from janusx import janusx as _jx  # type: ignore

        probe = getattr(_jx, "rust_sgemm_backend", None)
        if callable(probe):
            backend = str(probe()).strip().lower()
            if backend != "":
                return backend
    except Exception:
        pass
    return "unknown"


def set_rust_blas_threads(threads: int) -> bool:
    """
    Best-effort direct BLAS thread update for Rust JanusX backend.
    We first sync process-level OpenMP/BLAS env caps, then try the direct
    Rust OpenBLAS setter (when available) for immediate effect.
    """
    t = apply_blas_thread_env(
        max(1, int(threads)),
        set_jx_threads=False,
        set_max_keys=True,
    )
    try:
        from janusx import janusx as _jx  # type: ignore

        setter = getattr(_jx, "rust_blas_set_num_threads", None)
        if callable(setter):
            return bool(setter(int(t)))
    except Exception:
        pass
    return False


def get_rust_blas_threads() -> int | None:
    """
    Best-effort query of BLAS thread count used by Rust JanusX backend.
    Returns None when unavailable.
    """
    try:
        from janusx import janusx as _jx  # type: ignore

        getter = getattr(_jx, "rust_blas_get_num_threads", None)
        if callable(getter):
            raw = int(getter())
            return raw if raw > 0 else None
    except Exception:
        pass
    return None


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
    if backend == "unknown":
        msg += (
            " Detection may be limited in this environment; install "
            "`threadpoolctl` and `pyyaml` to improve backend introspection."
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
