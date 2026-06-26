# -*- coding: utf-8 -*-
"""Process memory helpers used by Python-side workflow reporting."""

from __future__ import annotations

import ctypes
import math
import os
import sys
from contextlib import contextmanager
from typing import Optional
from typing import Union

import psutil
from janusx.gfreader import auto_mmap_window_mb, calc_decode_block_rows_from_memory_mb

try:
    import resource
except Exception:  # pragma: no cover - platform dependent
    resource = None  # type: ignore[assignment]


_MAC_PROC_PID_RUSAGE = None
_MAC_RUSAGE_INFO_V4 = None
DEFAULT_DECODE_MEMORY_GB = 1.0


def process_memory_info_bytes(
    process: Optional[psutil.Process] = None,
) -> tuple[Optional[int], Optional[int]]:
    proc = process if process is not None else psutil.Process(os.getpid())
    try:
        info = proc.memory_info()
    except Exception:
        return None, None
    rss = int(getattr(info, "rss", 0) or 0)
    vms = int(getattr(info, "vms", 0) or 0)
    return (rss if rss > 0 else None), (vms if vms > 0 else None)


def process_ru_maxrss_bytes() -> Optional[int]:
    if resource is None:
        return None
    try:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if raw <= 0:
        return None
    if sys.platform == "darwin":
        return raw
    return raw * 1024


def process_peak_footprint_bytes() -> Optional[int]:
    if sys.platform != "darwin":
        return None
    global _MAC_PROC_PID_RUSAGE, _MAC_RUSAGE_INFO_V4
    try:
        if _MAC_PROC_PID_RUSAGE is None or _MAC_RUSAGE_INFO_V4 is None:
            class RUsageInfoV4(ctypes.Structure):
                _fields_ = [
                    ("ri_uuid", ctypes.c_uint8 * 16),
                    ("ri_user_time", ctypes.c_uint64),
                    ("ri_system_time", ctypes.c_uint64),
                    ("ri_pkg_idle_wkups", ctypes.c_uint64),
                    ("ri_interrupt_wkups", ctypes.c_uint64),
                    ("ri_pageins", ctypes.c_uint64),
                    ("ri_wired_size", ctypes.c_uint64),
                    ("ri_resident_size", ctypes.c_uint64),
                    ("ri_phys_footprint", ctypes.c_uint64),
                    ("ri_proc_start_abstime", ctypes.c_uint64),
                    ("ri_proc_exit_abstime", ctypes.c_uint64),
                    ("ri_child_user_time", ctypes.c_uint64),
                    ("ri_child_system_time", ctypes.c_uint64),
                    ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
                    ("ri_child_interrupt_wkups", ctypes.c_uint64),
                    ("ri_child_pageins", ctypes.c_uint64),
                    ("ri_child_elapsed_abstime", ctypes.c_uint64),
                    ("ri_diskio_bytesread", ctypes.c_uint64),
                    ("ri_diskio_byteswritten", ctypes.c_uint64),
                    ("ri_cpu_time_qos_default", ctypes.c_uint64),
                    ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
                    ("ri_cpu_time_qos_background", ctypes.c_uint64),
                    ("ri_cpu_time_qos_utility", ctypes.c_uint64),
                    ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
                    ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
                    ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
                    ("ri_billed_system_time", ctypes.c_uint64),
                    ("ri_serviced_system_time", ctypes.c_uint64),
                    ("ri_logical_writes", ctypes.c_uint64),
                    ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
                    ("ri_instructions", ctypes.c_uint64),
                    ("ri_cycles", ctypes.c_uint64),
                    ("ri_billed_energy", ctypes.c_uint64),
                    ("ri_serviced_energy", ctypes.c_uint64),
                    ("ri_interval_max_phys_footprint", ctypes.c_uint64),
                    ("ri_runnable_time", ctypes.c_uint64),
                ]

            lib = ctypes.CDLL(None)
            fn = lib.proc_pid_rusage
            fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            fn.restype = ctypes.c_int
            _MAC_PROC_PID_RUSAGE = fn
            _MAC_RUSAGE_INFO_V4 = RUsageInfoV4

        info = _MAC_RUSAGE_INFO_V4()
        rc = int(_MAC_PROC_PID_RUSAGE(int(os.getpid()), 4, ctypes.byref(info)))
        if rc != 0:
            return None
        for name in (
            "ri_lifetime_max_phys_footprint",
            "ri_interval_max_phys_footprint",
            "ri_phys_footprint",
        ):
            value = int(getattr(info, name, 0) or 0)
            if value > 0:
                return value
    except Exception:
        return None
    return None


def finalize_peak_memory_metrics(
    process: Optional[psutil.Process] = None,
    *,
    sampled_peak_rss_bytes: Optional[int] = None,
    sampled_peak_vms_bytes: Optional[int] = None,
) -> dict[str, Optional[int]]:
    rss_now, vms_now = process_memory_info_bytes(process)
    peak_rss = max(
        int(sampled_peak_rss_bytes or 0),
        int(rss_now or 0),
        int(process_ru_maxrss_bytes() or 0),
    )
    peak_vms = max(int(sampled_peak_vms_bytes or 0), int(vms_now or 0))
    peak_footprint = process_peak_footprint_bytes()
    return {
        "rss_bytes": rss_now,
        "vms_bytes": vms_now,
        "peak_rss_bytes": peak_rss if peak_rss > 0 else None,
        "peak_vms_bytes": peak_vms if peak_vms > 0 else None,
        "peak_footprint_bytes": peak_footprint if (peak_footprint or 0) > 0 else None,
    }


def normalize_decode_memory_gb(
    memory_gb: Union[int, float, None],
    *,
    default_gb: float = DEFAULT_DECODE_MEMORY_GB,
    arg_name: str = "--memory",
) -> float:
    if memory_gb is None:
        return float(default_gb)
    gb = float(memory_gb)
    if (not math.isfinite(gb)) or gb <= 0.0:
        raise ValueError(f"{arg_name} must be a finite value > 0 GB, got {memory_gb}")
    return float(gb)


def decode_memory_gb_to_mb(
    memory_gb: Union[int, float, None],
    *,
    default_gb: float = DEFAULT_DECODE_MEMORY_GB,
    arg_name: str = "--memory",
) -> float:
    return float(
        normalize_decode_memory_gb(
            memory_gb,
            default_gb=float(default_gb),
            arg_name=str(arg_name),
        )
        * 1024.0
    )


def effective_decode_memory_mb(
    memory_mb: Union[int, float],
    *,
    needs_copy: bool = False,
) -> float:
    mb = float(memory_mb)
    if (not math.isfinite(mb)) or mb <= 0.0:
        raise ValueError(f"decode memory must be finite and > 0 MB, got {memory_mb}")
    return float(mb * (0.5 if bool(needs_copy) else 1.0))


def scale_decode_rows_for_copy(
    rows: int,
    *,
    needs_copy: bool = False,
) -> int:
    base = max(1, int(rows))
    if not bool(needs_copy):
        return base
    return max(1, int(base // 2))


def resolve_decode_block_rows(
    row_width: int,
    memory_mb: Union[int, float],
    *,
    max_rows: Union[int, None] = None,
    elem_bytes: int = 4,
    needs_copy: bool = False,
    min_rows: int = 1,
) -> int:
    rows = calc_decode_block_rows_from_memory_mb(
        int(max(1, int(row_width))),
        effective_decode_memory_mb(float(memory_mb), needs_copy=bool(needs_copy)),
        elem_bytes=int(max(1, int(elem_bytes))),
        buffers=1,
        min_rows=int(max(1, int(min_rows))),
        max_rows=(None if max_rows is None else max(1, int(max_rows))),
    )
    return max(1, int(rows if rows is not None else 1))


def resolve_decode_mmap_window_mb(
    path_or_prefix: str,
    n_samples: int,
    n_snps: int,
    memory_mb: Union[int, float],
    *,
    elem_bytes: int = 4,
    needs_copy: bool = False,
    min_rows: int = 1,
    max_rows: Union[int, None] = None,
    min_chunks: int = 2,
) -> Optional[int]:
    return auto_mmap_window_mb(
        str(path_or_prefix),
        int(n_samples),
        int(n_snps),
        effective_decode_memory_mb(float(memory_mb), needs_copy=bool(needs_copy)),
        elem_bytes=int(max(1, int(elem_bytes))),
        buffers=1,
        min_rows=int(max(1, int(min_rows))),
        max_rows=(None if max_rows is None else max(1, int(max_rows))),
        min_chunks=int(max(1, int(min_chunks))),
    )


@contextmanager
def bed_block_target_env(
    memory_mb: Union[int, float, None],
    *,
    needs_copy: bool = False,
):
    prev = os.environ.get("JX_BED_BLOCK_TARGET_MB")
    if memory_mb is not None:
        target_mb = effective_decode_memory_mb(float(memory_mb), needs_copy=bool(needs_copy))
        os.environ["JX_BED_BLOCK_TARGET_MB"] = f"{float(target_mb):.6g}"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("JX_BED_BLOCK_TARGET_MB", None)
        else:
            os.environ["JX_BED_BLOCK_TARGET_MB"] = prev


def bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)
