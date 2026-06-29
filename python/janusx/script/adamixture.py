# -*- coding: utf-8 -*-
"""
JanusX: FastPop ancestry inference (Rust-kernel backend)

Examples
--------
  jx fastpop -bfile data/geno -k 8 -o out -prefix cohort
  jx fastpop -vcf data/geno.vcf.gz -k 6 -t 16
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import logging
import math
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from janusx import janusx as jxrs
from janusx.adamixture import ADAMixtureConfig, evaluate_adamixture_cverror, train_adamixture
from janusx.adamixture.core import (
    _admx_py_mem_debug,
    _is_admx_memory_limit_error,
    load_genotype_u8_matrix,
)
from janusx.gfreader import inspect_genotype_file
from ._common.interrupt import force_exit
from ._common.cli import (
    add_common_genotype_source_args,
    add_common_memory_arg,
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_snps_only_arg,
    add_common_thread_arg,
    add_common_variant_filter_args,
)
from ._common.config_render import emit_cli_configuration
from ._common.genoio import determine_genotype_source_from_args, strip_default_prefix_suffix
from ._common.cli import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.log import setup_logging
from ._common.memory import (
    DEFAULT_DECODE_MEMORY_GB,
    decode_memory_gb_to_mb,
    resolve_decode_block_rows,
)
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.progress import (
    CliStatus,
    build_rich_progress,
    log_success,
    print_failure,
    print_success,
    rich_progress_available,
    stdout_is_tty,
)
from ._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    format_requested_thread_usage,
    maybe_warn_non_openblas,
    require_openblas_by_default,
)

_DEFAULT_CVERROR_FOLDS = 5
_ADMX_PROGRESS_BAR_WIDTH = 24


def _popstruct_brand() -> dict[str, str]:
    raw = str(os.environ.get("JANUSX_POPSTRUCT_BRAND", "")).strip().lower()
    if raw == "fastpop":
        return {
            "display": "FastPop",
            "cli": "fastpop",
            "app_title": "JanusX - FastPop",
            "config_title": "FASTPOP CONFIG",
            "summary_stem": "fastpop",
            "plot_stem": "fastpop",
            "run_log_stem": "fastpop",
            "logger_name": "janusx.fastpop.bootstrap",
        }
    return {
        # Keep `jx adamixture` as a CLI/output alias, but present the module as
        # FastPop everywhere user-facing.
        "display": "FastPop",
        "cli": "adamixture",
        "app_title": "JanusX - FastPop",
        "config_title": "FASTPOP CONFIG",
        "summary_stem": "fastpop",
        "plot_stem": "fastpop",
        "run_log_stem": "fastpop",
        "logger_name": "janusx.fastpop.bootstrap",
    }


def _parse_cv_spec(cv_raw: Optional[str]) -> tuple[bool, int, str]:
    """
    Parse CV option:
      - None        -> disabled
      - ""/omitted  -> argparse const handles as default folds
      - "N>=2"      -> N-fold CVerror
      - "0"         -> disabled
    Returns: (enabled, cv_value_for_cfg, label)
    """
    if cv_raw is None:
        return False, 0, "off"
    raw = str(cv_raw).strip().lower()
    if raw in {"", "none"}:
        return False, 0, "off"
    try:
        n = int(raw)
    except Exception as ex:
        raise ValueError(
            f"Invalid -cv/--cv value: {cv_raw!r}. Use no arg or integer folds."
        ) from ex
    if n < 0:
        raise ValueError(f"-cv/--cv must be >= 0, got {n}.")
    if n == 0:
        return False, 0, "off"
    if n < 2:
        raise ValueError(f"-cv/--cv must be >= 2 when enabled, got {n}.")
    return True, int(n), f"{int(n)}-fold"


def _bool_env(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw == "":
        return bool(default)
    return raw not in {"0", "false", "no", "off"}


def _resolve_admx_rss_cap_gb_from_env() -> Optional[float]:
    for name in ("JANUSX_ADMX_MAX_FOOTPRINT_GB", "JANUSX_ADMX_MAX_RSS_GB"):
        raw = str(os.environ.get(name, "")).strip()
        if raw == "":
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if math.isfinite(value) and value > 0.0:
            return float(value)
    return None


class _AdmixtureMemoryWatchdog:
    def __init__(self, *, limit_gb: float, logger: logging.Logger, interval_s: float = 0.2) -> None:
        self.limit_gb = float(limit_gb)
        self.logger = logger
        self.interval_s = float(max(0.05, interval_s))
        self.limit_bytes = int(self.limit_gb * (1024.0**3))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._psutil = None
        self._mac_proc_pid_rusage = None
        self._mac_rusage_info_v4 = None

    def start(self) -> None:
        if not (math.isfinite(self.limit_gb) and self.limit_gb > 0.0):
            return
        try:
            import psutil  # type: ignore
        except Exception as ex:
            self.logger.warning(
                f"RSS watchdog unavailable: failed to import psutil ({ex})."
            )
            return
        self._psutil = psutil
        self._thread = threading.Thread(
            target=self._run,
            name="janusx-fastpop-rss-watchdog",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        limit_label = f"{self.limit_gb:.3f}" if self.limit_gb < 0.01 else f"{self.limit_gb:.2f}"
        self.logger.info(
            f"{'Memory limit':<22}: {limit_label} GB "
            "(0 disables; macOS uses current phys_footprint, other platforms use RSS)"
        )

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _mac_current_footprint_bytes(self) -> Optional[int]:
        if sys.platform != "darwin":
            return None
        try:
            if self._mac_proc_pid_rusage is None or self._mac_rusage_info_v4 is None:
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
                self._mac_proc_pid_rusage = fn
                self._mac_rusage_info_v4 = RUsageInfoV4
            info = self._mac_rusage_info_v4()
            rc = int(self._mac_proc_pid_rusage(int(os.getpid()), 4, ctypes.byref(info)))
            if rc != 0:
                return None
            return int(info.ri_phys_footprint)
        except Exception:
            return None

    def _proc_tree_memory_bytes(self) -> Optional[tuple[int, str]]:
        if self._psutil is None:
            return None
        try:
            proc = self._psutil.Process(os.getpid())
            rss_total = int(proc.memory_info().rss)
        except Exception:
            return None
        try:
            for child in proc.children(recursive=True):
                try:
                    rss_total += int(child.memory_info().rss)
                except Exception:
                    pass
        except Exception:
            pass
        footprint = self._mac_current_footprint_bytes()
        if footprint is not None and int(footprint) >= int(rss_total):
            return int(footprint), "phys_footprint"
        return int(rss_total), "rss"

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            mem = self._proc_tree_memory_bytes()
            if mem is None:
                continue
            mem_bytes, metric = mem
            if mem_bytes <= self.limit_bytes:
                continue
            mem_gb = float(mem_bytes) / (1024.0**3)
            limit_label = (
                f"{self.limit_gb:.3f}" if self.limit_gb < 0.01 else f"{self.limit_gb:.2f}"
            )
            brand = _popstruct_brand()
            msg = (
                f"{brand['display']} task aborted: memory limit exceeded "
                f"(limit={limit_label} GB, {metric}={mem_gb:.2f} GB)."
            )
            try:
                self.logger.error(msg)
            except Exception:
                pass
            force_exit(137, msg)


def _bed_stream_session_available(source_label: str) -> bool:
    return str(source_label).upper() == "BFILE" and (
        hasattr(jxrs, "AdmxBedTrainingSession") or hasattr(jxrs, "AdmxBedBackend")
    )


def _plink_prefix_from_pathlike(genotype_path: str) -> Optional[str]:
    raw = str(Path(str(genotype_path)).expanduser())
    prefix = raw[:-4] if raw.lower().endswith(".bed") else raw
    if Path(f"{prefix}.bed").exists() and Path(f"{prefix}.fam").exists():
        return prefix
    return None


def _try_read_plink_sample_ids(genotype_path: str) -> Optional[list[str]]:
    prefix = _plink_prefix_from_pathlike(genotype_path)
    if prefix is None:
        return None
    fam_path = Path(f"{prefix}.fam")
    sample_ids: list[str] = []
    with open(fam_path, "r", encoding="utf-8") as fr:
        for line in fr:
            toks = str(line).strip().split()
            if len(toks) >= 2:
                sample_ids.append(str(toks[1]))
            elif len(toks) == 1:
                sample_ids.append(str(toks[0]))
            else:
                sample_ids.append("")
    return sample_ids


def _open_text_auto(path: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def _lightweight_sample_count(
    genotype_path: str,
    *,
    source_label: str,
    snps_only: bool,
    maf: float,
    missing_rate: float,
) -> Optional[int]:
    src = str(source_label).strip().upper()
    if src == "BFILE":
        ids = _try_read_plink_sample_ids(genotype_path)
        return (None if ids is None or len(ids) <= 0 else int(len(ids)))

    if src == "VCF":
        with _open_text_auto(genotype_path) as fr:
            for line in fr:
                if line.startswith("#CHROM"):
                    toks = str(line).rstrip("\r\n").split()
                    return max(0, int(len(toks) - 9))
        return None

    if src == "HMP":
        with _open_text_auto(genotype_path) as fr:
            for line in fr:
                raw = str(line).strip()
                if raw == "" or raw.startswith("#"):
                    continue
                toks = raw.split()
                return max(0, int(len(toks) - 11))
        return None

    sample_ids, _n_snps = inspect_genotype_file(
        genotype_path,
        snps_only=bool(snps_only),
        maf=float(maf),
        missing_rate=float(missing_rate),
    )
    return int(len(sample_ids)) if len(sample_ids) > 0 else None


def _resolve_fastpop_chunk_rows(
    *,
    genotype_path: str,
    source_label: str,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    memory_gb: float,
) -> int:
    memory_mb = decode_memory_gb_to_mb(
        memory_gb,
        default_gb=DEFAULT_DECODE_MEMORY_GB,
        arg_name="-mem/--memory",
    )
    sample_count: Optional[int] = None
    try:
        sample_count = _lightweight_sample_count(
            genotype_path,
            source_label=source_label,
            snps_only=bool(snps_only),
            maf=float(maf),
            missing_rate=float(missing_rate),
        )
    except Exception:
        sample_count = None

    if sample_count is not None and int(sample_count) > 0:
        return int(
            max(
                1000,
                resolve_decode_block_rows(
                    int(sample_count),
                    float(memory_mb),
                    elem_bytes=4,
                    buffers=1,
                    min_rows=1000,
                ),
            )
        )

    budget_bytes = int(max(1.0, float(memory_mb)) * 1024.0 * 1024.0)
    fallback_rows = int(budget_bytes // (8192 * 4))
    return int(max(1000, fallback_rows))


def _format_fastpop_memory_cfg(memory_gb: float | int | None) -> str:
    if memory_gb is None:
        return "NA"
    return f"{float(memory_gb):.6g} GB"


def _mute_stdout_info_logs(
    logger: logging.Logger,
) -> list[tuple[logging.Handler, int]]:
    muted: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            muted.append((handler, handler.level))
            handler.setLevel(logging.WARNING)
    return muted


def _restore_handler_levels(muted: list[tuple[logging.Handler, int]]) -> None:
    for handler, level in muted:
        handler.setLevel(level)


class _AdmixtureBatchProgress:
    """Two-line rich progress for multi-K mode."""

    def __init__(
        self,
        *,
        enabled: bool,
        total_k: int,
        cv_enabled: bool,
        cv_label: str,
    ) -> None:
        self.enabled = bool(enabled and rich_progress_available())
        self.total_k = int(max(1, int(total_k)))
        self.cv_enabled = bool(cv_enabled)
        self.cv_label = str(cv_label)
        self._progress = None
        self._task_k = None
        self._task_cv = None

    def __enter__(self) -> "_AdmixtureBatchProgress":
        if not self.enabled:
            return self
        self._progress = build_rich_progress(
            description_template="[green]{task.description}",
            show_spinner=False,
            show_bar=True,
            show_percentage=True,
            show_elapsed=True,
            show_remaining=False,
            bar_width=int(_ADMX_PROGRESS_BAR_WIDTH),
            transient=False,
        )
        if self._progress is None:
            self.enabled = False
            return self
        self._progress.__enter__()
        self._task_k = self._progress.add_task("K progress", total=self.total_k)
        self._task_cv = self._progress.add_task("CV progress", total=1)
        self._progress.update(self._task_cv, completed=0, total=1)
        return self

    def start_k(self, *, k: int) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        if self.cv_enabled:
            self._progress.reset(
                self._task_cv,
                description=f"CV folds (K={int(k)})",
                total=1,
                completed=0,
            )
        else:
            self._progress.reset(
                self._task_cv,
                description=f"CV off (K={int(k)})",
                total=1,
                completed=1,
            )

    def cv_start(self, *, k: int, folds: int) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        total = max(1, int(folds))
        self._progress.reset(
            self._task_cv,
            description=f"CV folds (K={int(k)})",
            total=total,
            completed=0,
        )

    def cv_advance(self, *, steps: int = 1) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        self._progress.advance(self._task_cv, int(max(1, int(steps))))

    def cv_complete(self) -> None:
        if not self.enabled or self._progress is None or self._task_cv is None:
            return
        task = self._progress.tasks[self._task_cv]
        self._progress.update(self._task_cv, completed=task.total)

    def advance_k(self, *, steps: int = 1) -> None:
        if not self.enabled or self._progress is None or self._task_k is None:
            return
        self._progress.advance(self._task_k, int(max(1, int(steps))))

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._progress is not None:
            self._progress.__exit__(exc_type, exc, tb)
            self._progress = None


class _AdmixtureCliProgress:
    def __init__(self, logger: logging.Logger, *, enabled: bool) -> None:
        self.logger = logger
        self.enabled = bool(enabled)
        self._status: Optional[CliStatus] = None
        self._adam_max_iter = 0

    def _start(self, desc: str) -> None:
        self.close()
        self._status = CliStatus(str(desc), enabled=self.enabled)
        self._status.__enter__()

    def _update_desc(self, desc: str) -> None:
        if self._status is not None:
            self._status.desc = str(desc)

    def _complete(self, message: str) -> None:
        if self._status is not None:
            self._status.complete(str(message))
            self._status = None

    def fail(self, message: str) -> None:
        if self._status is not None:
            self._status.fail(str(message))
            self._status = None

    def close(self) -> None:
        if self._status is not None:
            self._status.__exit__(None, None, None)
            self._status = None

    def on_event(self, event: str, payload: dict[str, Any]) -> None:
        e = str(event)
        if e == "data_loaded":
            m = int(payload.get("m", 0))
            n = int(payload.get("n", 0))
            if self.enabled:
                print_success(
                    f"Loaded genotype matrix (SNPs={m}, samples={n})",
                    force_color=True,
                )
                print(f"Data shape: SNPs={m}, samples={n}", flush=True)
            return

        if e == "rsvd_start":
            self._start("RSVD")
            return
        if e == "rsvd_done":
            self._complete("RSVD ...Finished")
            return

        if e == "als_start":
            self._start("ALS initialization")
            return
        if e == "als_done":
            self._complete("ALS initialization ...Finished")
            return

        if e == "adam_start":
            self._adam_max_iter = int(payload.get("max_iter", 0))
            self._start(f"ADAM-EM iter 0/{self._adam_max_iter}")
            return
        if e == "adam_iter":
            it = int(payload.get("iteration", 0))
            ll = float(payload.get("ll", float("nan")))
            lr = float(payload.get("lr", float("nan")))
            if np.isfinite(ll) and np.isfinite(lr):
                self._update_desc(
                    f"ADAM-EM iter {it}/{self._adam_max_iter} "
                    f"ll={ll:.3f} lr={lr:.3e}"
                )
            return
        if e == "adam_done":
            self._complete("ADAM-EM optimization ...Finished")
            return


def _resolve_input(args, logger) -> tuple[str, str, str]:
    gfile, auto_prefix = determine_genotype_source_from_args(args)

    checks = []
    if args.vcf:
        checks.append(ensure_file_exists(logger, gfile, "VCF file"))
        source_label = "VCF"
    elif args.hmp:
        checks.append(ensure_file_exists(logger, gfile, "HMP file"))
        source_label = "HMP"
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, label="FILE genotype input"))
        source_label = "FILE"
    elif args.bfile:
        bprefix = gfile
        if str(gfile).lower().endswith(".bed"):
            bprefix = str(Path(gfile).with_suffix(""))
        checks.append(ensure_plink_prefix_exists(logger, bprefix, label="PLINK prefix"))
        gfile = bprefix
        auto_prefix = os.path.basename(bprefix.rstrip("/\\"))
        source_label = "BFILE"
    else:
        raise ValueError("One genotype input is required: -vcf / -hmp / -file / -bfile.")

    if not ensure_all_true(checks):
        raise FileNotFoundError("Input validation failed.")

    prefix = args.prefix or strip_default_prefix_suffix(auto_prefix)
    return gfile, source_label, prefix


def _site_line_from_obj(site: Any) -> str:
    chrom = "0"
    pos = 0
    ref = "N"
    alt = "N"
    try:
        if hasattr(site, "chrom") and hasattr(site, "pos"):
            chrom = str(getattr(site, "chrom"))
            pos = int(getattr(site, "pos"))
            if hasattr(site, "ref_allele"):
                ref = str(getattr(site, "ref_allele"))
            if hasattr(site, "alt_allele"):
                alt = str(getattr(site, "alt_allele"))
            return f"{chrom}\t{pos}\t{ref}\t{alt}"
        if isinstance(site, (tuple, list)):
            if len(site) >= 4:
                chrom = str(site[0])
                pos = int(site[1])
                ref = str(site[2])
                alt = str(site[3])
                return f"{chrom}\t{pos}\t{ref}\t{alt}"
            if len(site) >= 2:
                chrom = str(site[0])
                pos = int(site[1])
                return f"{chrom}\t{pos}\t{ref}\t{alt}"
    except Exception:
        pass
    return f"{chrom}\t{pos}\t{ref}\t{alt}"


def _safe_sample_ids_and_site_lines(
    genotype_path: str,
    *,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    expected_n: int,
    expected_m: int,
    logger: logging.Logger,
) -> tuple[list[str], Optional[list[str]]]:
    sample_ids: list[str] = []
    site_lines: Optional[list[str]] = None
    if _plink_prefix_from_pathlike(genotype_path) is not None:
        try:
            sample_ids = list(_try_read_plink_sample_ids(genotype_path) or [])
            site_lines = None
        except Exception as ex:
            logger.warning(f"Failed to read PLINK sample IDs for Q/P output: {ex}")
            sample_ids = []
            site_lines = None
    else:
        try:
            sample_ids_raw, site_rows_raw = inspect_genotype_file(
                genotype_path,
                snps_only=bool(snps_only),
                maf=float(maf),
                missing_rate=float(missing_rate),
            )
            sample_ids = [str(x) for x in sample_ids_raw]
            if isinstance(site_rows_raw, (list, tuple, np.ndarray)):
                site_lines = [_site_line_from_obj(x) for x in site_rows_raw]
        except Exception as ex:
            logger.warning(f"Failed to inspect sample/site IDs for Q/P output: {ex}")
            sample_ids = []
            site_lines = None

    if len(sample_ids) != int(expected_n):
        if len(sample_ids) > 0:
            logger.warning(
                "Sample ID count mismatch for Q output "
                f"(ids={len(sample_ids)}, expected={int(expected_n)}). "
                "Fallback to generated sample IDs."
            )
        sample_ids = [f"S{i+1}" for i in range(int(expected_n))]

    if site_lines is not None and len(site_lines) != int(expected_m):
        if len(site_lines) > 0:
            logger.warning(
                "Site row count mismatch for P.site output "
                f"(sites={len(site_lines)}, expected={int(expected_m)}). "
                "Fallback to streamed generated site rows."
            )
        site_lines = None

    return sample_ids, site_lines


def _write_q_txt_with_row_ids(
    out_path: str,
    row_ids: list[str],
    mat: np.ndarray,
) -> None:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Matrix must be 2D, got shape={arr.shape}")
    if len(row_ids) != int(arr.shape[0]):
        raise ValueError(
            f"Row ID length mismatch: ids={len(row_ids)}, rows={int(arr.shape[0])}"
        )
    with open(out_path, "w", encoding="utf-8") as fw:
        for rid, row in zip(row_ids, arr):
            vals = "\t".join(f"{float(v):.8f}" for v in row)
            fw.write(f"{rid}\t{vals}\n")


def _resolve_p_npy_dtype() -> np.dtype:
    raw = str(os.environ.get("JANUSX_ADMX_P_DTYPE", "float32")).strip().lower()
    if raw in {"float32", "f4", "single"}:
        return np.dtype(np.float32)
    if raw in {"float64", "f8", "double"}:
        return np.dtype(np.float64)
    raise ValueError(
        "Invalid JANUSX_ADMX_P_DTYPE. Use float32 or float64."
    )


def _stable_unique(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for x in items:
        s = str(x)
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _parse_tag_samples(
    tag_raw: Optional[str],
    *,
    logger: logging.Logger,
) -> list[str]:
    raw = ("" if tag_raw is None else str(tag_raw)).strip()
    if raw == "":
        return []

    p = Path(raw)
    if p.exists() and p.is_file():
        tags: list[str] = []
        with open(p, "r", encoding="utf-8") as fr:
            for line in fr:
                x = str(line).strip()
                if x == "":
                    continue
                tags.append(x)
        return _stable_unique(tags)

    parts = [x.strip() for x in raw.split(",") if x.strip() != ""]
    if len(parts) == 0:
        logger.warning(
            f"--tag specification is empty after parsing: {tag_raw!r}. "
            "No x-axis labels will be displayed."
        )
        return []
    return _stable_unique(parts)


def _render_structure_plot(
    *,
    q_mat: np.ndarray,
    q_ids: list[str],
    out_path: str,
    tag_samples: list[str],
    logger: logging.Logger,
) -> tuple[Optional[str], dict[str, Any]]:
    try:
        from janusx.bioplotkit.popstructure import plot_admixture_structure
    except Exception as ex:
        logger.warning(f"Failed to import popstructure plotting helper: {ex}")
        return None, {}

    try:
        meta = plot_admixture_structure(
            np.asarray(q_mat, dtype=np.float64),
            sample_ids=list(q_ids),
            out_path=str(out_path),
            tag_samples=list(tag_samples),
            show_xticks=False,
            dpi=300,
        )
        missing = list(meta.get("tag_missing", []))
        if len(missing) > 0:
            logger.warning(
                f"{len(missing)} tagged sample(s) were not found in Q sample IDs: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        return str(out_path), meta
    except Exception as ex:
        logger.warning(f"Failed to render structure plot: {ex}")
        return None, {}


def _write_p_site_lines(
    out_site_path: str,
    site_lines: Optional[list[str]],
    n_rows: int,
) -> None:
    with open(out_site_path, "w", encoding="utf-8") as fw:
        if site_lines is None:
            for i in range(int(n_rows)):
                fw.write(f"0\t{i+1}\tN\tN\n")
            return
        if len(site_lines) != int(n_rows):
            raise ValueError(
                f"P.site row count mismatch: sites={len(site_lines)}, rows={int(n_rows)}"
            )
        for line in site_lines:
            text = str(line).rstrip("\n")
            fw.write(f"{text}\n")


def _write_p_npy(
    out_npy_path: str,
    mat: np.ndarray,
) -> None:
    arr = np.asarray(mat)
    if arr.ndim != 2:
        raise ValueError(f"P matrix must be 2D, got shape={arr.shape}")

    dtype = _resolve_p_npy_dtype()
    chunk_rows_raw = str(os.environ.get("JANUSX_ADMX_P_MEMMAP_CHUNK", "4096")).strip()
    try:
        chunk_rows = max(1, int(chunk_rows_raw))
    except Exception:
        chunk_rows = 4096

    mm = np.lib.format.open_memmap(
        out_npy_path,
        mode="w+",
        dtype=dtype,
        shape=(int(arr.shape[0]), int(arr.shape[1])),
    )
    try:
        for start in range(0, int(arr.shape[0]), int(chunk_rows)):
            end = min(int(arr.shape[0]), int(start + int(chunk_rows)))
            mm[start:end, :] = np.asarray(arr[start:end, :], dtype=dtype)
    finally:
        del mm


def _try_write_bed_p_site_file(
    genotype_path: str,
    *,
    snps_only: bool,
    maf: float,
    missing_rate: float,
    out_site_path: str,
    logger: logging.Logger,
) -> bool:
    if not hasattr(jxrs, "AdmxBedBackend"):
        return False
    try:
        backend = jxrs.AdmxBedBackend(
            str(genotype_path),
            bool(snps_only),
            float(maf),
            float(missing_rate),
        )
        if not hasattr(backend, "write_site_file"):
            return False
        backend.write_site_file(str(out_site_path))
        return True
    except Exception as ex:
        logger.warning(
            f"Failed to stream BED P.site from backend ({ex}); fallback to generated site rows."
        )
        return False


def _parse_k_spec(k_spec: str) -> list[int]:
    """
    Parse K specification:
      - single value: "8"
      - inclusive range: "1..10" or "1:10"
      - inclusive range with step: "1..10..3", "1:10:3", or "1..10:3"
      - list: "1,5,8"
      - mixed: "1..10:3,5,8"
    """
    raw = str(k_spec or "").strip()
    if raw == "":
        raise ValueError("Empty -k/--k specification.")
    if ";" in raw:
        raise ValueError(
            "Semicolon-separated K list is not supported for shell compatibility. "
            "Use comma-separated form, e.g. -k 1,5,8."
        )

    tokens = []
    for seg in raw.split(","):
        x = seg.strip()
        if x:
            tokens.append(x)
    if len(tokens) == 0:
        raise ValueError(f"Invalid -k/--k specification: {k_spec!r}")

    values: list[int] = []
    for tok in tokens:
        if ".." in tok:
            norm = tok.replace("..", ":")
            parts = [x.strip() for x in norm.split(":") if x.strip() != ""]
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid K range token: {tok!r}")
            lo, hi = int(parts[0]), int(parts[1])
            step_abs = abs(int(parts[2])) if len(parts) == 3 else 1
            if step_abs <= 0:
                raise ValueError(f"K range step must be non-zero: {tok!r}")
            step = int(step_abs if hi >= lo else -step_abs)
            stop = int(hi + (1 if step > 0 else -1))
            values.extend(list(range(int(lo), stop, int(step))))
            continue
        if ":" in tok:
            parts = tok.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid K range token: {tok!r}")
            lo, hi = int(parts[0]), int(parts[1])
            step_abs = abs(int(parts[2])) if len(parts) == 3 else 1
            if step_abs <= 0:
                raise ValueError(f"K range step must be non-zero: {tok!r}")
            step = int(step_abs if hi >= lo else -step_abs)
            stop = int(hi + (1 if step > 0 else -1))
            values.extend(list(range(int(lo), stop, int(step))))
            continue
        values.append(int(tok))

    if len(values) == 0:
        raise ValueError(f"Invalid -k/--k specification: {k_spec!r}")

    deduped: list[int] = []
    seen = set()
    for v in values:
        if int(v) < 1:
            raise ValueError(f"K must be >= 1, got {v}.")
        if int(v) not in seen:
            deduped.append(int(v))
            seen.add(int(v))
    return deduped


def _write_cverror_summary_tsv(
    out_tsv: str,
    rows: list[dict[str, Any]],
) -> None:
    with open(out_tsv, "w", encoding="utf-8") as fw:
        fw.write(
            "\t".join(
                [
                    "K",
                    "CVerror",
                    "CVerror_SE",
                    "CV_folds",
                    "WallTime_sec",
                    "Q_output",
                    "P_npy_output",
                    "P_site_output",
                    "Structure_plot",
                    "Log_file",
                ]
            )
            + "\n"
        )
        for r in rows:
            fw.write(
                "\t".join(
                    [
                        str(int(r["k"])),
                        f"{float(r['cverror']):.6f}",
                        f"{float(r['cverror_se']):.6f}",
                        str(int(r["cv_folds"])),
                        f"{float(r['wall']):.3f}",
                        str(r["q_out"]),
                        str(r["p_npy_out"]),
                        str(r["p_site_out"]),
                        str(r.get("structure_plot", "")),
                        str(r["log_path"]),
                    ]
                )
                + "\n"
            )


def _log_cverror_summary(
    logger: logging.Logger,
    rows: list[dict[str, Any]],
    summary_tsv: str,
) -> None:
    if len(rows) == 0:
        return
    logger.info("-" * 60)
    row_fmt = "{k:<4} {cv:<10} {se:<10} {folds:<8} {time:<10}"
    logger.info(
        row_fmt.format(
            k="K",
            cv="CVerror",
            se="SE",
            folds="CVfolds",
            time="time(sec)",
        )
    )
    for r in rows:
        logger.info(
            row_fmt.format(
                k=int(r["k"]),
                cv=f"{float(r['cverror']):.6f}",
                se=f"{float(r['cverror_se']):.6f}",
                folds=int(r["cv_folds"]),
                time=f"{float(r['wall']):.2f}",
            )
        )
    best = min(rows, key=lambda x: float(x["cverror"]))
    logger.info("-" * 60)
    logger.info(
        f"Best K by CVerror: K={int(best['k'])} "
        f"(CVerror={float(best['cverror']):.6f}, SE={float(best['cverror_se']):.6f})"
    )
    logger.info(f"CVerror summary TSV: {format_path_for_display(summary_tsv)}")


def _log_outputs_summary(
    logger: logging.Logger,
    rows: list[dict[str, Any]],
) -> None:
    if len(rows) == 0:
        return
    logger.info("-" * 60)
    logger.info("Output files:")
    for r in rows:
        k = int(r["k"])
        logger.info(f"K={k:<4} Q: {format_path_for_display(str(r['q_out']))}")
        logger.info(f"K={k:<4} P.npy: {format_path_for_display(str(r['p_npy_out']))}")
        logger.info(f"K={k:<4} P.site: {format_path_for_display(str(r['p_site_out']))}")
        if r.get("structure_plot"):
            logger.info(
                f"K={k:<4} STRUCT: "
                f"{format_path_for_display(str(r['structure_plot']))}"
            )
        logger.info(f"K={k:<4} LOG: {format_path_for_display(str(r['log_path']))}")


def _run_single_k(
    *,
    genotype_path: str,
    source_label: str,
    prefix: str,
    outdir: str,
    k: int,
    k_spec: str,
    k_index: int,
    total_k: int,
    args: argparse.Namespace,
    detected_threads: int,
    requested_threads: int,
    resolved_threads: int,
    enable_spinner: bool,
    emit_config_to_stdout: bool,
    batch_progress: Optional[_AdmixtureBatchProgress],
    cv_enabled: bool,
    cv_value: int,
    cv_label: str,
    tag_samples: list[str],
    shared_g: Optional[np.ndarray],
) -> dict[str, Any]:
    brand = _popstruct_brand()
    use_batch_progress = bool(batch_progress is not None and batch_progress.enabled)
    log_path = os.path.join(outdir, f"{prefix}.{int(k)}.{brand['run_log_stem']}.log")
    logger = setup_logging(log_path)
    apply_blas_thread_env(int(resolved_threads))
    if not use_batch_progress:
        logger.info("")
    if (int(total_k) > 1) and (not use_batch_progress):
        logger.info(
            f"K scan progress: {int(k_index)}/{int(total_k)} "
            f"(K={int(k)}, spec={k_spec})"
        )

    q_out = os.path.join(outdir, f"{prefix}.{int(k)}.Q.txt")
    p_npy_out = os.path.join(outdir, f"{prefix}.{int(k)}.P.npy")
    p_site_out = os.path.join(outdir, f"{prefix}.{int(k)}.P.site")
    structure_fmt = "svg"
    structure_out = os.path.join(outdir, f"{prefix}.{int(k)}.{brand['plot_stem']}.{structure_fmt}")
    plot_enabled = not bool(getattr(args, "no_plot", False))

    model_rows = [
        ("K", int(k)),
        ("Solver", str(args.solver)),
        ("max_iter", int(args.max_iter)),
        ("check", int(args.check)),
        ("tol", float(args.tol)),
        ("CVerror", (str(cv_label) if cv_enabled else "off")),
        ("Tag labels", ("off" if len(tag_samples) == 0 else f"{len(tag_samples)} sample(s)")),
        ("Structure plot", ("on" if plot_enabled else "off")),
    ]
    if int(total_k) > 1:
        model_rows.append(("K spec", str(k_spec)))

    cfg_mute: list[tuple[logging.Handler, int]] = []
    if not emit_config_to_stdout:
        cfg_mute = _mute_stdout_info_logs(logger)
    try:
        _admx_py_mem_debug(
            "cli/run_single_k/before_config",
            f"k={int(k)} source={source_label} bed_stream={bool(_bed_stream_session_available(source_label))}",
        )
        emit_cli_configuration(
            logger,
            app_title=brand["app_title"],
            config_title=brand["config_title"],
            host=socket.gethostname(),
            sections=[
                (
                    "Input/Output",
                    [
                        ("Genotype", genotype_path),
                        ("Input type", source_label),
                        ("SNPs only", bool(args.snps_only)),
                        ("MAF threshold", float(args.maf)),
                        ("Miss threshold", float(args.geno)),
                        ("Output dir", outdir),
                        ("Prefix", prefix),
                    ],
                ),
                (
                    "Runtime",
                    [
                        (
                            "Threads",
                            format_requested_thread_usage(
                                requested_threads=int(requested_threads),
                                using_threads=int(resolved_threads),
                                detected_threads=int(detected_threads),
                            ),
                        ),
                        ("Memory", _format_fastpop_memory_cfg(getattr(args, "memory", None))),
                        ("Seed", int(args.seed)),
                    ],
                ),
                ("Model", model_rows),
            ],
            footer_rows=[
                ("Output prefix", os.path.join(outdir, prefix)),
                ("Q output", q_out),
                ("P output (.npy)", p_npy_out),
                ("P site (.site)", p_site_out),
                ("Structure plot", (structure_out if plot_enabled else "disabled")),
                ("Log file", log_path),
            ],
            emit_to_stdout=bool(emit_config_to_stdout),
        )
        _admx_py_mem_debug("cli/run_single_k/after_config", f"k={int(k)}")
    finally:
        _restore_handler_levels(cfg_mute)

    cfg = ADAMixtureConfig(
        genotype_path=genotype_path,
        k=int(k),
        outdir=outdir,
        prefix=prefix,
        seed=int(args.seed),
        threads=max(1, int(resolved_threads)),
        chunk_size=int(max(1000, int(getattr(args, "_resolved_chunk_rows", 50000)))),
        snps_only=bool(args.snps_only),
        maf=float(args.maf),
        geno=float(args.geno),
        solver=str(args.solver),
        max_iter=max(1, int(args.max_iter)),
        check=max(1, int(args.check)),
        tol=float(args.tol),
        cv=int(cv_value if cv_enabled else 0),
    )

    progress_ui: Optional[_AdmixtureCliProgress] = None
    callback = None
    muted: list[tuple[logging.Handler, int]] = []
    cverror_result: Optional[dict[str, float]] = None
    structure_plot_path: Optional[str] = None
    bed_stream_session = _bed_stream_session_available(source_label)
    t_one = time.time()
    try:
        _admx_py_mem_debug(
            "cli/run_single_k/before_progress_setup",
            f"k={int(k)} use_batch_progress={bool(use_batch_progress)} enable_spinner={bool(enable_spinner)}",
        )
        if use_batch_progress:
            muted = _mute_stdout_info_logs(logger)
        elif enable_spinner:
            progress_ui = _AdmixtureCliProgress(logger, enabled=True)
            callback = progress_ui.on_event
            muted = _mute_stdout_info_logs(logger)
        _admx_py_mem_debug(
            "cli/run_single_k/before_train",
            f"k={int(k)} bed_stream={bool(bed_stream_session)} shared_g={shared_g is not None}",
        )
        try:
            p_mat, q_mat, m, n = train_adamixture(
                cfg,
                logger,
                callback=callback,
                g_matrix=(None if bed_stream_session else shared_g),
            )
            _admx_py_mem_debug(
                "cli/run_single_k/after_train",
                f"k={int(k)} m={int(m)} n={int(n)}",
            )
        finally:
            if progress_ui is not None:
                progress_ui.close()
            if not use_batch_progress:
                _restore_handler_levels(muted)
        q_ids, p_site_lines = _safe_sample_ids_and_site_lines(
            genotype_path,
            snps_only=bool(args.snps_only),
            maf=float(args.maf),
            missing_rate=float(args.geno),
            expected_n=int(n),
            expected_m=int(m),
            logger=logger,
        )
        _write_q_txt_with_row_ids(q_out, q_ids, np.asarray(q_mat, dtype=np.float64))
        p_arr = np.asarray(p_mat, dtype=np.float32)
        site_written = False
        if bed_stream_session:
            site_written = _try_write_bed_p_site_file(
                genotype_path,
                snps_only=bool(args.snps_only),
                maf=float(args.maf),
                missing_rate=float(args.geno),
                out_site_path=p_site_out,
                logger=logger,
            )
            if site_written:
                logger.info("P.site metadata      : BED backend stream")
        if not site_written:
            _write_p_site_lines(
                p_site_out,
                p_site_lines,
                int(p_arr.shape[0]),
            )
        _write_p_npy(
            p_npy_out,
            p_arr,
        )
        if plot_enabled:
            structure_plot_path, _plot_meta = _render_structure_plot(
                q_mat=np.asarray(q_mat, dtype=np.float64),
                q_ids=list(q_ids),
                out_path=str(structure_out),
                tag_samples=list(tag_samples),
                logger=logger,
            )
        else:
            logger.info("Structure plot      : skipped (--no-plot)")
        if cv_enabled:
            cv_progress_callback = None
            if use_batch_progress:
                def _cv_progress(event: str, payload: dict[str, Any]) -> None:
                    if batch_progress is None:
                        return
                    e = str(event)
                    if e == "cv_start":
                        batch_progress.cv_start(
                            k=int(k),
                            folds=int(payload.get("folds", 1)),
                        )
                        return
                    if e == "cv_fold_done":
                        batch_progress.cv_advance(steps=1)
                        return
                    if e == "cv_done":
                        batch_progress.cv_complete()
                        return
                cv_progress_callback = _cv_progress
            cv_status: Optional[CliStatus] = None
            cv_muted: list[tuple[logging.Handler, int]] = []
            if (not use_batch_progress) and enable_spinner:
                cv_muted = _mute_stdout_info_logs(logger)
                cv_status = CliStatus("CVerror estimation", enabled=True)
                cv_status.__enter__()
            try:
                cverror_result = evaluate_adamixture_cverror(
                    cfg,
                    logger,
                    progress_callback=cv_progress_callback,
                    g_full=shared_g,
                )
                if cv_status is not None:
                    cv_status.complete("CVerror estimation ...Finished")
            except Exception:
                if cv_status is not None:
                    cv_status.fail("CVerror estimation ...Failed")
                raise
            finally:
                if cv_status is not None:
                    cv_status.__exit__(None, None, None)
                _restore_handler_levels(cv_muted)
    except Exception as e:
        _restore_handler_levels(muted)
        if progress_ui is not None:
            try:
                progress_ui.fail(f"{brand['display']} ...Failed")
            except Exception:
                pass
        if _is_admx_memory_limit_error(e):
            msg = str(e)
            logger.error(msg)
            print_failure(f"{brand['display']} memory limit exceeded", force_color=True)
            force_exit(137, msg)
        logger.exception(f"{brand['display']} failed: {e}")
        print_failure(f"{brand['display']} ...Failed", force_color=True)
        raise

    _restore_handler_levels(muted)
    wall = float(time.time() - t_one)
    end_mute: list[tuple[logging.Handler, int]] = []
    if use_batch_progress:
        end_mute = _mute_stdout_info_logs(logger)
    try:
        logger.info(f"Q output: {format_path_for_display(q_out)}")
        logger.info(f"P output (.npy): {format_path_for_display(p_npy_out)}")
        logger.info(f"P site (.site): {format_path_for_display(p_site_out)}")
        if structure_plot_path is not None:
            logger.info(
                f"Structure plot: {format_path_for_display(str(structure_plot_path))}"
            )
        if cv_enabled:
            if cverror_result is None:
                raise RuntimeError("Internal error: CVerror result is missing.")
            logger.info(
                "CVerror: "
                f"{float(cverror_result['cverror']):.6f} "
                f"(SE={float(cverror_result['cverror_se']):.6f}, "
                f"folds={int(cverror_result['cv_folds'])})"
            )
        now = datetime.now()
        logger.info("")
        log_success(logger, f"K={int(k)} finished in {wall:.2f} seconds")
        logger.info(now.strftime("%Y-%m-%d %H:%M:%S"))
    finally:
        _restore_handler_levels(end_mute)
    row = {
        "k": int(k),
        "cverror": (None if cverror_result is None else float(cverror_result["cverror"])),
        "cverror_se": (None if cverror_result is None else float(cverror_result["cverror_se"])),
        "cv_folds": (0 if cverror_result is None else int(cverror_result["cv_folds"])),
        "wall": float(wall),
        "q_out": q_out,
        "p_npy_out": p_npy_out,
        "p_site_out": p_site_out,
        "structure_plot": (None if structure_plot_path is None else str(structure_plot_path)),
        "log_path": log_path,
    }
    return row


def _build_parser() -> CliArgumentParser:
    brand = _popstruct_brand()
    cli = f"jx {brand['cli']}"
    help_profile = ("fastpop" if brand["cli"] == "fastpop" else "admixture")
    parser = CliArgumentParser(
        prog=cli,
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            f"{cli} -bfile data/geno -k 8 -o out -prefix cohort",
            f"{cli} -vcf data/geno.vcf.gz -k 1..10 -t 16",
        ]),
    )

    req_geno = parser.add_argument_group("Required Genotype Input (Choose one)")
    geno = req_geno.add_mutually_exclusive_group(required=True)
    add_common_genotype_source_args(geno, include_file=True, help_profile=help_profile)
    req_model = parser.add_argument_group("Required Nclusters")
    req_model.add_argument(
        "-k",
        "--k",
        type=str,
        required=True,
        help=(
            "K spec: single (8), range (1..10 or 1:10), "
            "stepped range (1..10..3, 1:10:3, or 1..10:3), or list (1,5,8)."
        ),
    )
    input_output = parser.add_argument_group("Input/Output Arguments")
    add_common_out_arg(input_output, default=".", help_profile="current_dir")
    add_common_prefix_arg(input_output, default=None, help_profile="simple")
    input_output.add_argument(
        "-tag",
        "--tag",
        type=str,
        default=None,
        help=(
            "Optional sample labels for structure x-axis. "
            "Use a file path (one sample ID per line) or comma-separated sample IDs."
        ),
    )
    input_output.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Skip structure plot rendering and only write Q/P/site/log outputs.",
    )
    add_common_snps_only_arg(
        input_output,
        dest="snps_only",
        default=False,
        help_profile="loading_stage",
    )
    add_common_variant_filter_args(
        input_output,
        help_profile="loading_stage",
        include_maf=True,
        include_geno=True,
        include_het=False,
        maf_default=0.02,
        geno_default=0.05,
    )
    runtime = parser.add_argument_group("Runtime Arguments")
    add_common_thread_arg(
        runtime,
        default_threads=detect_effective_threads(),
        dest="thread",
    )
    runtime.add_argument(
        "-threads",
        "--threads",
        dest="thread",
        type=int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    runtime.add_argument("-seed", "--seed", type=int, default=42, help="Random seed (default: 42).")
    add_common_memory_arg(
        runtime,
        default=DEFAULT_DECODE_MEMORY_GB,
        help_text=(
            "Working memory budget in GB for FastPop streamed genotype loading/decode. "
            "This controls internal chunk sizing rather than total process RSS; "
            "explicit -mem keeps the requested fixed budget (default: %(default)s)."
        ),
        dest="memory",
        include_hidden_legacy_single_dash_alias=True,
    )

    model = parser.add_argument_group("Model/Optimization Arguments")
    model.add_argument(
        "-solver",
        "--solver",
        type=str,
        default="adam-em",
        choices=["auto", "adam", "adam-em"],
        help="Optimization solver (default: adam-em).",
    )
    model.add_argument(
        "-max-iter",
        "--max-iter",
        dest="max_iter",
        type=int,
        default=500,
        help="Maximum optimization iterations (default: 500).",
    )
    model.add_argument(
        "-check",
        "--check",
        type=int,
        default=5,
        help=(
            "Log-likelihood check interval during Adam-EM (default: 5). "
            "Larger values reduce full-data scans and can speed up large BED runs."
        ),
    )
    model.add_argument(
        "-tol",
        "--tol",
        type=float,
        default=1e-5,
        help="Convergence tolerance (default: 1e-5).",
    )
    model.add_argument(
        "-cv",
        "--cv",
        nargs="?",
        default=None,
        const=str(_DEFAULT_CVERROR_FOLDS),
        metavar="N",
        help=(
            "Enable CVerror. No value -> 5-fold; integer N -> N-fold. "
            "Omit -cv to disable CVerror."
        ),
    )
    return parser


def main() -> None:
    brand = _popstruct_brand()
    parser = _build_parser()
    args = parser.parse_args()
    if not (0.0 <= float(args.maf) <= 0.5):
        parser.error("-maf must be within [0, 0.5].")
    if not (0.0 <= float(args.geno) <= 1.0):
        parser.error("-geno must be within [0, 1.0].")
    if float(args.tol) <= 0:
        parser.error("-tol/--tol must be > 0.")
    if int(args.max_iter) <= 0:
        parser.error("-max-iter/--max-iter must be a positive integer.")
    if int(args.check) <= 0:
        parser.error("-check/--check must be a positive integer.")
    if int(args.thread) <= 0:
        parser.error("-t/--thread must be a positive integer.")
    memory_mb = decode_memory_gb_to_mb(
        getattr(args, "memory", DEFAULT_DECODE_MEMORY_GB),
        default_gb=DEFAULT_DECODE_MEMORY_GB,
        arg_name="-mem/--memory",
    )
    os.environ["JANUSX_ADMX_LOAD_MB"] = str(int(max(1, round(float(memory_mb)))))
    try:
        k_values = _parse_k_spec(str(args.k))
    except Exception as ex:
        parser.error(str(ex))
    try:
        cv_enabled, cv_value, cv_label = _parse_cv_spec(args.cv)
    except Exception as ex:
        parser.error(str(ex))
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    resolved_threads = int(args.thread)
    if int(resolved_threads) > int(detected_threads):
        thread_capped = True
        resolved_threads = int(detected_threads)

    outdir = os.path.normpath(str(args.out or "."))
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()

    tmp_logger = logging.getLogger(brand["logger_name"])
    if not tmp_logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        tmp_logger.addHandler(h)
    tmp_logger.setLevel(logging.INFO)
    tmp_logger.propagate = False
    rss_cap_gb = _resolve_admx_rss_cap_gb_from_env()
    rss_watchdog = _AdmixtureMemoryWatchdog(
        limit_gb=float(0.0 if rss_cap_gb is None else rss_cap_gb),
        logger=tmp_logger,
    )
    try:
        rss_watchdog.start()
        genotype_path, source_label, auto_prefix = _resolve_input(args, tmp_logger)
    except Exception as e:
        print_failure(str(e), force_color=True)
        raise
    tag_samples = _parse_tag_samples(args.tag, logger=tmp_logger)

    prefix = args.prefix or auto_prefix
    multi_k = len(k_values) > 1
    enable_spinner = bool(stdout_is_tty())
    bed_stream_session = _bed_stream_session_available(source_label)
    use_geno_cache = _bool_env("JANUSX_ADMX_CACHE_GENO", True)
    effective_geno_cache = bool(use_geno_cache and (not bed_stream_session))
    args._resolved_chunk_rows = _resolve_fastpop_chunk_rows(
        genotype_path=genotype_path,
        source_label=source_label,
        snps_only=bool(args.snps_only),
        maf=float(args.maf),
        missing_rate=float(args.geno),
        memory_gb=float(getattr(args, "memory", DEFAULT_DECODE_MEMORY_GB)),
    )
    if multi_k:
        if cv_enabled:
            cv_msg = str(cv_label)
        else:
            cv_msg = "off"
        print_success(
            f"K-scan mode: {len(k_values)} K values ({args.k})",
            force_color=True,
        )
        print(
            f"CVerror={cv_msg}, solver={args.solver}, "
            f"max_iter={int(args.max_iter)}, check={int(args.check)}, "
            f"threads={int(resolved_threads)}, "
            f"memory={_format_fastpop_memory_cfg(getattr(args, 'memory', None))}, "
            f"geno_cache={'on' if effective_geno_cache else 'off'}, "
            f"bed_stream={'on' if bed_stream_session else 'off'}, "
            f"tag={'on' if len(tag_samples) > 0 else 'off'}",
            flush=True,
        )
    if bed_stream_session:
        if hasattr(jxrs, "AdmxBedTrainingSession"):
            tmp_logger.info(
                "BED Rust training session detected: model training will call fit_k() "
                "on the persistent packed BED backend instead of dense shared_g cache."
            )
        else:
            tmp_logger.info(
                "BED stream session detected: model training will use low-memory BED backend "
                "instead of dense shared_g cache."
            )
        if use_geno_cache:
            tmp_logger.info(
                "Skipping dense genotype preload because BED stream session is available."
            )
        if cv_enabled:
            tmp_logger.info(
                "CVerror will use the BED foldmask stream backend when available, "
                "avoiding dense g_full and per-fold g_train copies."
            )
    if thread_capped:
        tmp_logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(resolved_threads)}."
        )
    apply_blas_thread_env(int(resolved_threads))
    # maybe_warn_non_openblas(
    #     logger=tmp_logger,
    #     strict=require_openblas_by_default(),
    # )
    shared_g: Optional[np.ndarray] = None
    if effective_geno_cache:
        shared_g = load_genotype_u8_matrix(
            genotype_path,
            chunk_size=int(max(1000, int(getattr(args, "_resolved_chunk_rows", 50000)))),
            snps_only=bool(args.snps_only),
            maf=float(args.maf),
            missing_rate=float(args.geno),
        )
    summary_rows: list[dict[str, Any]] = []
    try:
        with _AdmixtureBatchProgress(
            enabled=bool(multi_k),
            total_k=int(len(k_values)),
            cv_enabled=bool(cv_enabled),
            cv_label=str(cv_label),
        ) as batch_progress:
            for idx, k in enumerate(k_values, start=1):
                batch_progress.start_k(k=int(k))
                row = _run_single_k(
                    genotype_path=genotype_path,
                    source_label=source_label,
                    prefix=prefix,
                    outdir=outdir,
                    k=int(k),
                    k_spec=str(args.k),
                    k_index=int(idx),
                    total_k=int(len(k_values)),
                    args=args,
                    detected_threads=int(detected_threads),
                    requested_threads=int(requested_threads),
                    resolved_threads=int(resolved_threads),
                    enable_spinner=bool(enable_spinner),
                    emit_config_to_stdout=bool(not multi_k),
                    batch_progress=batch_progress,
                    cv_enabled=bool(cv_enabled),
                    cv_value=int(cv_value),
                    cv_label=str(cv_label),
                    tag_samples=list(tag_samples),
                    shared_g=shared_g,
                )
                summary_rows.append(row)
                batch_progress.advance_k(steps=1)
    finally:
        rss_watchdog.stop()

    summary_log = os.path.join(outdir, f"{prefix}.{brand['summary_stem']}.summary.log")
    logger = setup_logging(summary_log)
    if cv_enabled:
        cv_rows = [r for r in summary_rows if r.get("cverror") is not None]
        summary_tsv = os.path.join(outdir, f"{prefix}.{brand['summary_stem']}.cverror.summary.tsv")
        _write_cverror_summary_tsv(summary_tsv, cv_rows)
        logger.info("")
        _log_cverror_summary(logger, cv_rows, summary_tsv)
    logger.info("")
    _log_outputs_summary(logger, summary_rows)

    wall = float(time.time() - t0)
    now = datetime.now()
    logger.info("")
    logger.info(f"Summary log: {format_path_for_display(summary_log)}")
    logger.info(f"Output dir: {format_path_for_display(outdir)}")
    log_success(logger, f"Finished. Total wall time: {wall:.2f} seconds")
    logger.info(now.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
