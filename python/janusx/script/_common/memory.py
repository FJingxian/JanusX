# -*- coding: utf-8 -*-
"""Process memory helpers used by Python-side workflow reporting."""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

import psutil

try:
    import resource
except Exception:  # pragma: no cover - platform dependent
    resource = None  # type: ignore[assignment]


_MAC_PROC_PID_RUSAGE = None
_MAC_RUSAGE_INFO_V4 = None


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


def bytes_to_gib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)
