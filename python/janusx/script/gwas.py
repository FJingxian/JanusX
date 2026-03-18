# -*- coding: utf-8 -*-
"""
JanusX: High-Performance GWAS Command-Line Interface

Design overview
---------------
Models:
  - LMM     : streaming, low-memory implementation (slim.LMM)
  - LM      : streaming, low-memory implementation (slim.LM)
  - FarmCPU : in-memory implementation (pyBLUP.farmcpu) that loads the
              full genotype matrix

Execution mode (automatic)
--------------------------
  - No explicit "low-memory" flag is required.
  - LMM/LM always run in streaming mode via rust2py.gfreader.load_genotype_chunks.
  - FarmCPU always runs on the full in-memory genotype matrix.

Caching
-------
  - Genotype/GRM/PCA caches use parameterized filenames (no JSON cache metadata).
  - GRM and PCA caches for streaming LMM/LM runs:
      * genotype cache: ~{geno_prefix}.maf{maf}.geno{geno}.snp{0|1}.bed/.bim/.fam
      * GRM cache:      ~{geno_prefix}.maf{maf}.geno{geno}.grm{k}.npy (+ .id)
      * PCA cache:      ~{geno_prefix}.maf{maf}.geno{geno}.grm{k}.pc{q}.txt
  - If genotype directory is not writable, cache falls back to
    JANUSX_CACHE_DIR (configured from -o).

Covariates
----------
  - The --cov option is shared by LMM, LM, and FarmCPU.
  - Covariate files must include sample IDs in the first column.
  - Rows are aligned by sample ID intersection with genotype IDs.

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import os
import io
import math
import time
import socket
import argparse
import logging
import sys
import threading
import warnings
import multiprocessing as mp
import concurrent.futures as cf
import textwrap
from datetime import datetime
from typing import Union, Optional
import uuid
from contextlib import contextmanager
from contextlib import nullcontext

from janusx.pyBLUP.QK2 import GRM
from janusx.pyBLUP.stream_grm import (
    auto_stream_grm_chunk_size,
    build_streaming_grm_from_chunks,
)

# ---- Matplotlib backend configuration (non-interactive, server-safe) ----
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import psutil
from janusx.bioplotkit import GWASPLOT, apply_integer_yticks
from janusx.gfreader import (
    load_genotype_chunks,
    load_bed_2bit_packed,
    inspect_genotype_file,
    auto_mmap_window_mb,
)
from janusx.pyBLUP.QK2 import QK
from janusx.pyBLUP.assoc import LMM, LM, FastLMM, farmcpu
from janusx import janusx as jxrs
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    safe_expanduser,
    safe_resolve,
)
from ._common.prefetch import prefetch_iter
from ._common.status import (
    CliStatus,
    is_skip_status_text,
    print_success,
    print_warning,
    format_elapsed,
    should_animate_status,
    stdout_is_tty,
)
from ._common.progress import build_rich_progress, rich_progress_available
from ._common.threads import detect_effective_threads
from ._common.gwas_history import record_gwas_run
from ._common.genocache import configure_genotype_cache_from_out
from ._common.cjk import contains_cjk as _contains_cjk, ensure_cjk_font as _ensure_cjk_font
from ._common.genoio import (
    basename_only as _basename_only,
    determine_genotype_source_from_args as determine_genotype_source,
    read_id_file as _read_id_file,
)

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    _threadpool_limits = None
from ._common.colspec import parse_zero_based_index_specs

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False

_WARNED_GWAS_CACHE_FALLBACK_KEYS: set[str] = set()
_FASTLMM_PVE_LOW = 0.05
_FASTLMM_PVE_HIGH = 0.95
_SECTION_WIDTH = 60
_GWAS_PROGRESS_BAR_WIDTH = 30

# ======================================================================
# Basic utilities
# ======================================================================

def _section(logger:logging.Logger, title: str) -> None:
    """Emit a formatted log section header with a leading blank line."""
    logger.info("")
    logger.info("=" * _SECTION_WIDTH)
    logger.info(title)
    logger.info("=" * _SECTION_WIDTH)


def _phase_split(logger: logging.Logger) -> None:
    """Visual separator between loading stage and compute stage."""
    logger.info("-" * _SECTION_WIDTH)


def _format_progress_metric(
    done: int,
    total: int,
    memory_text: Optional[str] = None,
) -> str:
    tot = int(max(0, total))
    dn = int(max(0, done))
    if tot <= 0:
        pct = 0.0
    else:
        pct = 100.0 * float(min(dn, tot)) / float(tot)
    mem_raw = str(memory_text or "").strip()
    if mem_raw != "":
        return f"[{mem_raw:>7}]"
    return f"[{pct:5.1f}% ]"


def _fastlmm_pve_is_degenerate(pve: Optional[float]) -> bool:
    if pve is None:
        return False
    try:
        v = float(pve)
    except Exception:
        return False
    if not np.isfinite(v):
        return False
    return bool(v < _FASTLMM_PVE_LOW or v > _FASTLMM_PVE_HIGH)


def _emit_trait_header(
    logger: logging.Logger,
    trait_name: str,
    n_idv: int,
    *,
    pve: Optional[float] = None,
    use_spinner: bool = False,
    width: int = 60,
) -> None:
    """
    Emit trait header as one line: "<trait> (n=<idv>)".
    If it exceeds section width, wrap at trait/name boundary.
    """
    trait = str(trait_name)
    pve_val: Optional[float] = None
    if pve is not None:
        try:
            pve_tmp = float(pve)
            if np.isfinite(pve_tmp):
                pve_val = pve_tmp
        except Exception:
            pve_val = None
    if pve_val is None:
        n_line = f"(n={int(n_idv)})"
    else:
        n_line = f"(n={int(n_idv)}; pve={pve_val:.3f})"
    full = f"{trait} {n_line}"
    if len(full) <= int(width):
        lines = [full]
    else:
        if len(trait) <= int(width):
            lines = [trait, n_line]
        else:
            trait_lines = textwrap.wrap(
                trait,
                width=max(10, int(width)),
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines = trait_lines + [n_line]
    for ln in lines:
        if use_spinner:
            _log_file_only(logger, logging.INFO, ln)
            print(ln, flush=True)
        else:
            logger.info(ln)


def _emit_gwas_summary(
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> None:
    if len(rows) == 0:
        return

    _section(logger, "Summary")
    headers = [
        "pheno",
        "model",
        "nidv",
        "nsnp",
        "pve",
        "mem(G)",
        "Ctime(s)",
        "Vtime(s)",
    ]

    out_rows: list[list[str]] = []
    for r in _ordered_gwas_summary_rows(rows):
        model_text = str(r.get("model", "")).strip()
        if model_text.lower() in {"lowranklmm", "lrlmm"}:
            model_text = "LrLMM"
        pve_raw = r.get("pve", None)
        pve_text = "NA"
        try:
            if pve_raw is not None:
                pve_val = float(pve_raw)
                if np.isfinite(pve_val):
                    pve_text = f"{pve_val:.3f}"
        except Exception:
            pve_text = "NA"
        out_rows.append(
            [
                str(r.get("phenotype", "")),
                model_text,
                f"{int(r.get('nidv', 0))}",
                f"{int(r.get('eff_snp', 0))}",
                pve_text,
                f"{float(r.get('peak_rss_gb', 0.0)):.2f}",
                f"{float(r.get('gwas_time_s', 0.0)):.1f}",
                f"{float(r.get('viz_time_s', 0.0)):.1f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in out_rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    logger.info(header_line)
    for row in out_rows:
        logger.info("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def _gwas_model_sort_key(model_name: object) -> tuple[int, str]:
    model = str(model_name or "").strip()
    order = {
        "LM": 0,
        "LMM": 1,
        "FastLMM": 2,
        "LowRankLMM": 3,
        "LrLMM": 3,
        "LRLMM": 3,
        "Farm": 4,
        "FarmCPU": 4,
    }
    return (order.get(model, 99), model.lower())


def _ordered_gwas_summary_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        list(rows),
        key=lambda r: (
            int(r.get("pheno_col_idx", 10**9)),
            str(r.get("phenotype", "")),
            _gwas_model_sort_key(r.get("model", "")),
        ),
    )


def _ordered_saved_result_paths(
    rows: list[dict[str, object]],
    saved_paths: list[str],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in _ordered_gwas_summary_rows(rows):
        path = str(row.get("result_file", "") or "").strip()
        if path == "" or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    for path in saved_paths:
        p = str(path).strip()
        if p == "" or p in seen:
            continue
        seen.add(p)
        ordered.append(p)
    return ordered


def _log_file_only(logger: logging.Logger, level: int, message: str) -> None:
    """
    Emit a log record to file handlers only.
    Used to keep rich terminal output clean while preserving full log files.
    """
    msg = str(message)
    handled = False
    try:
        for handler in getattr(logger, "handlers", []):
            if isinstance(handler, logging.FileHandler):
                record = logger.makeRecord(
                    logger.name,
                    level,
                    __file__,
                    0,
                    msg,
                    args=(),
                    exc_info=None,
                    func=None,
                    extra=None,
                )
                handler.handle(record)
                handled = True
    except Exception:
        handled = False
    if not handled:
        logger.log(level, msg)


def _log_info(logger: logging.Logger, message: str, *, use_spinner: bool = False) -> None:
    if use_spinner:
        _log_file_only(logger, logging.INFO, str(message))
    else:
        logger.info(str(message))


def _emit_plain_info_line(
    logger: logging.Logger,
    message: str,
    *,
    use_spinner: bool = False,
) -> None:
    """
    Emit a plain (non-success-styled) info line.
    In spinner mode we print to terminal and also keep file logging.
    """
    msg = str(message)
    if use_spinner:
        _log_file_only(logger, logging.INFO, msg)
        print(msg, flush=True)
    else:
        logger.info(msg)


def _emit_warning_line(
    logger: logging.Logger,
    message: str,
    *,
    use_spinner: bool = False,
) -> None:
    msg = str(message)
    if use_spinner or stdout_is_tty():
        _log_file_only(logger, logging.WARNING, msg)
        print_warning(msg, force_color=bool(use_spinner))
    else:
        logger.warning(msg)


def _log_model_line(
    logger: logging.Logger,
    model_label: str,
    message: str,
    *,
    use_spinner: bool = False,
) -> None:
    _log_info(logger, f"{str(model_label)}: {str(message)}", use_spinner=use_spinner)


def _rich_success(
    logger: logging.Logger,
    message: str,
    *,
    use_spinner: bool = False,
    log_message: Optional[str] = None,
) -> None:
    msg = str(message)
    file_msg = str(msg if log_message is None else log_message)
    force_color = bool(use_spinner)
    if is_skip_status_text(msg):
        if use_spinner or stdout_is_tty():
            _log_file_only(logger, logging.WARNING, file_msg)
            print_warning(msg, force_color=force_color)
        else:
            logger.warning(file_msg)
        return
    if use_spinner or stdout_is_tty():
        _log_file_only(logger, logging.INFO, file_msg)
        print_success(msg, force_color=force_color)
    else:
        logger.info(file_msg)


class _ProgressAdapter:
    """
    Progress bar adapter with rich-first rendering and tqdm fallback.
    """
    def __init__(self, total: int, desc: str, *, force_animate: bool = False) -> None:
        self.total = int(max(0, total))
        self.desc = str(desc)
        self._animate = bool(force_animate or should_animate_status(self.desc))
        self._backend = "none"
        self._progress = None
        self._task_id = None
        self._tqdm = None
        self._start_ts = time.monotonic()
        self._finished = False
        self._done = 0
        self._tick = 0
        self._memory_text = ""
        self._memory_until_tick = 0
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None

        if self._animate and rich_progress_available():
            try:
                self._progress = build_rich_progress(
                    description_template="[green]{task.description:<8}",
                    show_remaining=False,
                    show_percentage=False,
                    field_templates=["{task.fields[metric]}"],
                    bar_width=_GWAS_PROGRESS_BAR_WIDTH,
                    finished_text=" ",
                    transient=True,
                )
                if self._progress is None:
                    raise RuntimeError("rich progress unavailable")
                self._progress.start()
                self._task_id = self._progress.add_task(
                    self.desc,
                    total=self.total,
                    metric=_format_progress_metric(0, self.total, None),
                )
                self._backend = "rich"
            except Exception:
                self._progress = None
                self._task_id = None

        if self._animate and self._backend == "none" and _HAS_TQDM and stdout_is_tty():
            self._tqdm = tqdm(
                total=self.total,
                desc=self.desc,
                ascii=False,
                leave=False,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            self._backend = "tqdm"

        if self._animate and self._backend in {"rich", "tqdm"}:
            self._hb_thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._hb_thread.start()

    def _heartbeat(self) -> None:
        while not self._hb_stop.wait(0.2):
            if self._backend == "rich" and self._progress is not None and self._task_id is not None:
                try:
                    self._progress.update(self._task_id, metric=self._metric_text())
                except Exception:
                    return
            elif self._backend == "tqdm" and self._tqdm is not None:
                try:
                    self._tqdm.refresh()
                except Exception:
                    return

    def _metric_text(self) -> str:
        show_mem = bool(
            self._memory_text
            and (self._tick <= int(self._memory_until_tick))
        )
        return _format_progress_metric(
            self._done,
            self.total,
            self._memory_text if show_mem else None,
        )

    def update(self, n: int) -> None:
        step = int(max(0, n))
        if step == 0:
            return
        self._done += step
        self._tick += 1
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                advance=step,
                metric=self._metric_text(),
            )
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.update(step)
            self._tqdm.set_postfix_str(self._metric_text())

    def set_postfix(self, **kwargs: object) -> None:
        if len(kwargs) == 0:
            return
        if len(kwargs) == 1 and "memory" in kwargs:
            self._memory_text = str(kwargs["memory"]).replace(" ", "")
            self._memory_until_tick = self._tick + 5
            if self._backend == "rich" and self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, metric=self._metric_text())
            elif self._backend == "tqdm" and self._tqdm is not None:
                self._tqdm.set_postfix_str(self._metric_text())
            return
        text = " ".join([f"{k} ~ {v}" for k, v in kwargs.items()])
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, metric=text)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_postfix_str(text)

    def set_desc(self, desc: str) -> None:
        self.desc = str(desc)
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=self.desc)
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.set_description_str(self.desc)

    def finish(self) -> None:
        self._done = self.total
        self._tick += 1
        if self._backend == "rich" and self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self.total,
                metric=self._metric_text(),
            )
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.n = self._tqdm.total
            self._tqdm.set_postfix_str(self._metric_text())
            self._tqdm.refresh()
        self._finished = True

    def close(
        self,
        done_text: str = "Finished",
        show_done: bool = True,
        success_style: bool = True,
    ) -> None:
        elapsed = format_elapsed(time.monotonic() - self._start_ts)
        self._hb_stop.set()
        if self._hb_thread is not None:
            try:
                self._hb_thread.join(timeout=0.5)
            except Exception:
                pass
            self._hb_thread = None
        if self._backend == "rich" and self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        elif self._backend == "tqdm" and self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None
        if self._finished and show_done:
            msg = f"{self.desc} ...{done_text} [{elapsed}]"
            if success_style and self._animate:
                print_success(msg, force_color=True)
            else:
                print(msg, flush=True)


def _start_indeterminate_progress_bar(
    desc: str,
    *,
    enabled: bool,
    interval_s: float = 0.2,
    total_ticks: int = 80,
    cap_ratio: float = 0.95,
) -> Optional[tuple[_ProgressAdapter, threading.Event, threading.Thread]]:
    if not bool(enabled):
        return None
    total = max(20, int(total_ticks))
    cap = max(1, min(total - 1, int(total * float(cap_ratio))))
    pbar = _ProgressAdapter(total=total, desc=str(desc), force_animate=True)
    done = {"n": 0}
    stop_evt = threading.Event()

    def _runner() -> None:
        wait_s = max(0.05, float(interval_s))
        while not stop_evt.wait(wait_s):
            if done["n"] < cap:
                pbar.update(1)
                done["n"] += 1

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    return pbar, stop_evt, th


def _stop_indeterminate_progress_bar(
    handle: Optional[tuple[_ProgressAdapter, threading.Event, threading.Thread]],
    *,
    success: bool,
) -> None:
    if handle is None:
        return
    pbar, stop_evt, th = handle
    stop_evt.set()
    try:
        th.join(timeout=1.0)
    except Exception:
        pass
    if bool(success):
        try:
            pbar.finish()
            # Give terminal renderer one short cycle to paint 100% before closing.
            time.sleep(0.08)
        except Exception:
            pass
    pbar.close(show_done=False)


def fastplot(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    xlabel: str = "",
    outpdf: str = "fastplot.pdf",
) -> None:
    """
    Generate diagnostic plots for GWAS results: phenotype histogram, Manhattan, and QQ.
    """
    mpl.rcParams["font.size"] = 12
    results = gwasresult.astype({"pos": "int64"})
    fig = plt.figure(figsize=(16, 4), dpi=300)
    try:
        layout = [["A", "B", "B", "C"]]
        axes:dict[str,plt.Axes] = fig.subplot_mosaic(mosaic=layout)

        gwasplot = GWASPLOT(results)
        scatter_size = 8.0

        # A: phenotype distribution
        pheno = np.asarray(phenosub, dtype="float64").reshape(-1)
        pheno = pheno[np.isfinite(pheno)]
        n_samples = int(pheno.size)
        label_base = str(xlabel).strip() if str(xlabel).strip() else "phenotype"

        if n_samples > 0:
            counts, edges, _ = axes["A"].hist(
                pheno,
                bins=15,
                color="black",
                edgecolor="none",
                alpha=1.0,
            )
            # Overlay seaborn-like KDE curve, scaled to histogram "Count" axis.
            if counts.size > 1 and np.unique(pheno).size > 1:
                try:
                    kde = gaussian_kde(pheno)
                    x_grid = np.linspace(float(np.min(pheno)), float(np.max(pheno)), 256)
                    y_density = kde(x_grid)
                    bin_width = float(np.mean(np.diff(edges)))
                    y_count = y_density * float(n_samples) * bin_width
                    axes["A"].plot(x_grid, y_count, color="#B3B3B3", linewidth=1.6)
                except Exception:
                    pass
        else:
            axes["A"].text(
                0.5,
                0.5,
                "No valid phenotype values",
                ha="center",
                va="center",
                transform=axes["A"].transAxes,
            )

        x_label_text = f"{label_base} (n={n_samples})"
        if _contains_cjk(x_label_text) and not _ensure_cjk_font():
            # No CJK font found on this host; fallback to ASCII to avoid glyph warnings.
            x_label_text = f"phenotype (n={n_samples})"
        axes["A"].set_xlabel(x_label_text)
        axes["A"].set_ylabel("Count")

        # B: Manhattan plot
        gwasplot.manhattan(
            -np.log10(1 / results.shape[0]),
            ax=axes["B"],
            rasterized=True,
            s=scatter_size,
        )
        manh_yticks = apply_integer_yticks(axes["B"])
        snp_n = int(results.shape[0])
        if snp_n >= 1_000_000:
            snp_val = snp_n / 1_000_000.0
            snp_suffix = "M"
        else:
            snp_val = snp_n / 1_000.0
            snp_suffix = "K"
        snp_text = f"{snp_val:.3f}".rstrip("0").rstrip(".")
        axes["B"].set_xlabel(f"Chromosome (SNP={snp_text}{snp_suffix})")

        # C: QQ plot
        manh_ymin, manh_ymax = axes["B"].get_ylim()
        gwasplot.qq(
            ax=axes["C"],
            scatter_size=scatter_size,
            axis_min=float(manh_ymin),
        )

        # Align QQ with Manhattan:
        # - QQ ylim follows Manhattan ylim
        # - QQ xlim minimum equals Manhattan ylim minimum
        qq_xmin, qq_xmax = axes["C"].get_xlim()
        axes["C"].set_ylim(manh_ymin, manh_ymax)
        qq_left = float(manh_ymin)
        if np.isfinite(qq_xmin) and np.isfinite(qq_xmax) and qq_xmax > qq_xmin:
            x_right = max(float(qq_xmax), qq_left + 1e-9)
            x_span = max(1e-9, x_right - qq_left)
            x_pad = max(1e-9, 0.02 * x_span)
            x_upper = x_right + x_pad
            if x_upper <= qq_left:
                x_upper = qq_left + 1.0
            axes["C"].set_xlim(qq_left, x_upper)
        else:
            axes["C"].set_xlim(qq_left, qq_left + 1.0)
        apply_integer_yticks(axes["C"], ticks=manh_yticks)

        fig.tight_layout()
        fig.savefig(outpdf, transparent=False, facecolor="white")
    finally:
        plt.close(fig)


def _is_writable_dir(path: str) -> bool:
    d = os.path.abspath(path or ".")
    return os.path.isdir(d) and os.access(d, os.W_OK | os.X_OK)


def _warn_gwas_cache_fallback_once(
    key: str,
    msg: str,
    logger: Union[logging.Logger, None] = None,
) -> None:
    if key in _WARNED_GWAS_CACHE_FALLBACK_KEYS:
        return
    _WARNED_GWAS_CACHE_FALLBACK_KEYS.add(key)
    if logger is not None:
        logger.warning(msg)
    else:
        warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _normalize_cache_warning_message(msg: str) -> str:
    s = str(msg).strip()
    prefix = "No write permission for genotype-side cache directory:"
    marker = ". Falling back to cache directory from JANUSX_CACHE_DIR:"
    if s.startswith(prefix) and marker in s:
        try:
            left, right = s.split(marker, 1)
            src = left[len(prefix):].strip()
            dst = right.strip()
            src_abs = str(safe_resolve(src))
            dst_abs = str(safe_resolve(dst))
            return f"{prefix} {src_abs}{marker} {dst_abs}"
        except Exception:
            return s
    return s


def _dedupe_cache_warning_messages(msgs: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in msgs:
        s = _normalize_cache_warning_message(str(m))
        if s == "" or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _is_cache_warning_message(msg: str) -> bool:
    s = str(msg).lower()
    if s == "":
        return False
    return (
        "cache" in s
        or "no write permission for genotype-side cache directory" in s
        or "janusx_cache_dir" in s
    )


def _resolve_gwas_cache_dir(
    genofile: str,
    *,
    cache_dir: Union[str, None] = None,
    logger: Union[logging.Logger, None] = None,
    emit_warning: bool = True,
    warning_collector: Union[list[str], None] = None,
) -> str:
    geno_dir = os.path.abspath(os.path.dirname(str(genofile)) or ".")
    if _is_writable_dir(geno_dir):
        return geno_dir

    preferred = (str(cache_dir).strip() if cache_dir is not None else "")
    if preferred == "":
        preferred = os.environ.get("JANUSX_CACHE_DIR", "").strip()
    if preferred:
        fallback_dir = str(safe_resolve(preferred))
        try:
            os.makedirs(fallback_dir, mode=0o755, exist_ok=True)
        except Exception:
            pass
        if _is_writable_dir(fallback_dir):
            if bool(emit_warning):
                msg = (
                    "No write permission for genotype-side cache directory: "
                    f"{geno_dir}. Falling back to cache directory from "
                    f"JANUSX_CACHE_DIR: {fallback_dir}"
                )
                if warning_collector is not None:
                    warning_collector.append(msg)
                else:
                    _warn_gwas_cache_fallback_once(
                        f"{geno_dir}->{fallback_dir}",
                        msg,
                        logger=logger,
                    )
            return fallback_dir

    return geno_dir


def genotype_cache_prefix(
    genofile: str,
    *,
    snps_only: bool = True,
    cache_dir: Union[str, None] = None,
    logger: Union[logging.Logger, None] = None,
    warning_collector: Union[list[str], None] = None,
) -> str:
    """
    Construct a cache prefix for GWAS GRM/Q caches.
    Prefer genotype directory; if not writable, fallback to JANUSX_CACHE_DIR.
    """
    base = os.path.basename(genofile)
    low = base.lower()
    if low.endswith(".vcf.gz"):
        base = base[: -len(".vcf.gz")]
    else:
        for ext in (".vcf", ".txt", ".tsv", ".csv", ".npy"):
            if low.endswith(ext):
                base = base[: -len(ext)]
                break
    cache_root = _resolve_gwas_cache_dir(
        genofile,
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    return os.path.join(cache_root, f"~{base}").replace("\\", "/")


def _fmt_cache_num(v: Union[float, int]) -> str:
    x = float(v)
    s = f"{x:.6g}"
    if "e" in s or "E" in s:
        s = f"{x:.12f}".rstrip("0").rstrip(".")
    if s in {"", "-0"}:
        s = "0"
    return s


def _param_tag(
    *,
    maf: float,
    geno: float,
    snps_only: bool,
) -> str:
    tag = f".maf{_fmt_cache_num(maf)}.geno{_fmt_cache_num(geno)}.snp{1 if bool(snps_only) else 0}"
    return tag


def _gwas_cache_prefix_with_params(
    genofile: str,
    *,
    maf: float,
    geno: float,
    snps_only: bool,
    cache_dir: Union[str, None] = None,
    logger: Union[logging.Logger, None] = None,
    warning_collector: Union[list[str], None] = None,
) -> str:
    base = genotype_cache_prefix(
        genofile,
        snps_only=bool(snps_only),
        cache_dir=cache_dir,
        logger=logger,
        warning_collector=warning_collector,
    )
    return f"{base}{_param_tag(maf=float(maf), geno=float(geno), snps_only=bool(snps_only))}"


def _grm_cache_paths(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
) -> tuple[str, str]:
    grm_path = f"{cache_prefix_with_params}.grm{mgrm}.npy"
    id_path = f"{grm_path}.id"
    return grm_path, id_path


def _pca_cache_path(
    cache_prefix_with_params: str,
    *,
    mgrm: str,
    qdim: Union[int, str],
) -> str:
    return f"{cache_prefix_with_params}.grm{mgrm}.pc{qdim}.txt"


@contextmanager
def _cache_lock(lock_key: str, timeout_s: float = 7200.0, poll_s: float = 0.2):
    lock_path = f"{lock_key}.lock"
    start = time.monotonic()
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > timeout_s:
                    os.remove(lock_path)
                    continue
            except Exception:
                pass
            if (time.monotonic() - start) > timeout_s:
                raise TimeoutError(f"Timeout waiting cache lock: {lock_path}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass
        except Exception:
            pass

def latest_genotype_mtime(genofile: str) -> Union[float, None]:
    """
    Return the latest modification time of genotype input files.

    - PLINK prefix: max mtime of .bed/.bim/.fam (if all exist)
    - FILE prefix : max mtime of matrix file, prefix.id, and optional site metadata
    - File input  : mtime of the file path itself
    """
    bed = f"{genofile}.bed"
    bim = f"{genofile}.bim"
    fam = f"{genofile}.fam"
    if all(os.path.isfile(p) for p in (bed, bim, fam)):
        return max(os.path.getmtime(bed), os.path.getmtime(bim), os.path.getmtime(fam))

    file_prefix, matrix_path = _resolve_file_input_matrix(genofile)
    if file_prefix and matrix_path:
        mtimes = [os.path.getmtime(matrix_path)]
        for cand in _file_input_sidecars(file_prefix):
            if os.path.isfile(cand):
                mtimes.append(os.path.getmtime(cand))
        return max(mtimes)

    if os.path.isfile(genofile):
        return os.path.getmtime(genofile)
    return None


def _resolve_file_input_matrix(genofile: str) -> tuple[Optional[str], Optional[str]]:
    path = str(safe_expanduser(str(genofile)))
    low = path.lower()
    if low.endswith(".npy"):
        return path[: -len(".npy")], path
    for ext in (".txt", ".tsv", ".csv"):
        if low.endswith(ext):
            return path[: -len(ext)], path
    for ext in (".npy", ".txt", ".tsv", ".csv"):
        cand = f"{path}{ext}"
        if os.path.isfile(cand):
            return path, cand
    return None, None


def _file_input_sidecars(prefix: str) -> list[str]:
    return [
        f"{prefix}.id",
        f"{prefix}.site",
        f"{prefix}.site.tsv",
        f"{prefix}.site.txt",
        f"{prefix}.site.csv",
        f"{prefix}.bim",
    ]


def _normalize_cov_inputs(cov_arg: Union[str, list[str], None]) -> list[str]:
    if cov_arg is None:
        return []
    if isinstance(cov_arg, str):
        raw = [cov_arg]
    else:
        raw = [str(x) for x in cov_arg]
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def _parse_cov_site_token(token: str) -> Union[tuple[str, int], None]:
    """
    Parse site-style covariate token:
      - chr:pos
      - chr:start:end  (single-site only, requires start == end)
    Also accepts full-width colon '：'.
    """
    t = str(token).strip().replace("：", ":")
    parts = [p.strip() for p in t.split(":")]
    if len(parts) not in (2, 3):
        return None

    chrom = parts[0]
    if chrom == "":
        return None

    try:
        start = int(float(parts[1]))
    except Exception:
        return None
    if start <= 0:
        raise ValueError(f"Invalid site position in --cov: {token}")

    if len(parts) == 3:
        try:
            end = int(float(parts[2]))
        except Exception:
            return None
        if end <= 0:
            raise ValueError(f"Invalid site position in --cov: {token}")
        if end != start:
            raise ValueError(
                f"--cov site token must specify a single site (start=end), got: {token}"
            )

    return chrom, start


def _split_cov_sources(cov_inputs: list[str]) -> tuple[list[str], list[tuple[str, int]]]:
    cov_files: list[str] = []
    cov_sites: list[tuple[str, int]] = []
    for item in cov_inputs:
        parsed = _parse_cov_site_token(item)
        if parsed is None:
            cov_files.append(item)
        else:
            cov_sites.append(parsed)
    return cov_files, cov_sites


def _format_grm_display(grm_opt: object) -> str:
    s = str(grm_opt if grm_opt is not None else "").strip()
    if s in {"1", "2"}:
        return s
    if s == "":
        return ""
    return _basename_only(s)


def _format_qcov_display(qcov_opt: object) -> str:
    s = str(qcov_opt if qcov_opt is not None else "").strip()
    if s == "":
        return ""
    try:
        return str(_parse_qcov_dim(qcov_opt))
    except Exception:
        return s


def _parse_qcov_dim(qcov_opt: object) -> int:
    s = str(qcov_opt if qcov_opt is not None else "").strip()
    if s == "":
        raise ValueError(
            "Invalid -q/--qcov: empty value. Use an integer PC dimension (>=0)."
        )
    try:
        q = int(s)
    except Exception:
        raise ValueError(
            "External Q matrix via -q/--qcov is no longer supported. "
            "Use -q <int> for PCA dimension, and pass external covariate files via -c."
        ) from None
    if q < 0:
        raise ValueError(f"Invalid -q/--qcov: {q}. Q/PC dimension must be >= 0.")
    return int(q)


def _format_cov_display(cov_inputs: Union[list[str], None]) -> str:
    if cov_inputs is None:
        return ""
    parts: list[str] = []
    for item in cov_inputs:
        token = str(item).strip()
        if token == "":
            continue
        parsed = _parse_cov_site_token(token)
        if parsed is not None:
            chrom, pos = parsed
            parts.append(f"{chrom}:{int(pos)}")
        else:
            parts.append(_basename_only(token))
    return ";".join(parts)


def _canon_site_key(chrom: str, pos: int) -> tuple[str, int]:
    c = str(chrom).strip().lower()
    if c.startswith("chr"):
        c = c[3:]
    return c, int(pos)


def _read_cov_file_flexible(
    path: str,
    sample_ids: Union[np.ndarray, None],
    logger,
    label: str = "Covariate",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read covariate file.

    Supported format:
      - first column: sample ID
      - remaining columns: numeric covariates

    Numeric-only covariate matrices are not supported.
    """
    try:
        df = pd.read_csv(
            path, sep=None, engine="python", header=None,
            dtype=str, keep_default_na=False
        )
    except Exception:
        df = pd.read_csv(
            path, sep=r"\s+", header=None,
            dtype=str, keep_default_na=False
        )

    if df.empty:
        raise ValueError(f"{label} file is empty: {path}")

    if df.shape[1] < 2:
        raise ValueError(
            f"{label} file must contain at least 2 columns: sample_id + covariate(s): {path}"
        )

    sid = None if sample_ids is None else np.asarray(sample_ids, dtype=str)
    col0 = df.iloc[:, 0].astype(str).str.strip().to_numpy()
    df_use = df
    ids = col0

    if sid is not None:
        sid_set = set(sid)
        need = max(1, int(0.9 * len(sid)))
        overlap = len(set(col0) & sid_set)
        if overlap < need and df.shape[0] > 1:
            # Header-tolerant fallback for small tables:
            # if dropping the first row makes ID overlap sufficient, treat first row as header.
            col0_tail = df.iloc[1:, 0].astype(str).str.strip().to_numpy()
            overlap_tail = len(set(col0_tail) & sid_set)
            if overlap_tail >= need:
                df_use = df.iloc[1:, :].reset_index(drop=True)
                ids = df_use.iloc[:, 0].astype(str).str.strip().to_numpy()
                overlap = overlap_tail

        if overlap < need:
            raise ValueError(
                f"{label} file must include sample IDs in the first column "
                f"(numeric-only matrix is not supported): {path}"
            )

    data = df_use.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return np.asarray(ids, dtype=str), data


def _load_site_covariates(
    genofile: str,
    site_specs: list[tuple[str, int]],
    sample_ids: np.ndarray,
    chunk_size: int,
    logger,
    use_spinner: bool = False,
    snps_only: bool = True,
) -> np.ndarray:
    """
    Load additive genotype values for requested SNP sites as covariates.
    Returns matrix shape (n_samples, n_sites).
    """
    if len(site_specs) == 0:
        return np.zeros((len(sample_ids), 0), dtype="float32")

    sample_ids = np.asarray(sample_ids, dtype=str)
    unique_sites: list[tuple[str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    for c, p in site_specs:
        k = _canon_site_key(c, p)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        unique_sites.append((c, p))
    need_keys: set[tuple[str, int]] = {_canon_site_key(c, p) for c, p in unique_sites}

    picked_rows: list[np.ndarray] = []
    picked_sites: list[tuple[str, int]] = []
    query_chunk = max(1, min(int(max(1, chunk_size)), max(1, len(site_specs))))
    for geno_chunk, site_chunk in load_genotype_chunks(
        genofile,
        chunk_size=query_chunk,
        maf=0.0,
        missing_rate=1.0,
        impute=True,
        model="add",
        snps_only=bool(snps_only),
        snp_sites=unique_sites,
        sample_ids=sample_ids.tolist(),
    ):
        if geno_chunk.shape[0] == 0:
            continue
        for i, s in enumerate(site_chunk):
            key = _canon_site_key(str(s.chrom), int(s.pos))
            if key not in need_keys:
                continue
            picked_rows.append(np.asarray(geno_chunk[i], dtype="float32"))
            picked_sites.append((str(s.chrom), int(s.pos)))
            need_keys.remove(key)
        if len(need_keys) == 0:
            break

    missing_tokens = []
    for c, p in unique_sites:
        key = _canon_site_key(c, p)
        if key in need_keys:
            missing_tokens.append(f"{c}:{p}")
    if missing_tokens:
        show = ", ".join(missing_tokens[:10])
        if len(missing_tokens) > 10:
            show += f", ... ({len(missing_tokens)} missing)"
        raise ValueError(f"Some --cov SNP site(s) were not found in genotype: {show}")

    idx_map: dict[tuple[str, int], list[int]] = {}
    for i, (c, p) in enumerate(picked_sites):
        k = _canon_site_key(c, p)
        idx_map.setdefault(k, []).append(i)

    ordered_rows: list[np.ndarray] = []
    for c, p in site_specs:
        k = _canon_site_key(c, p)
        pool = idx_map.get(k, [])
        if len(pool) == 0:
            raise ValueError(f"--cov site lookup failed for {c}:{p}")
        ordered_rows.append(picked_rows[pool[0]])

    cov = np.vstack(ordered_rows).astype("float32", copy=False).T
    return cov


def _load_covariates_for_models(
    cov_inputs: Union[str, list[str], None],
    genofile: str,
    sample_ids: np.ndarray,
    chunk_size: int,
    logger,
    context: str,
    use_spinner: bool = False,
    snps_only: bool = True,
) -> tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
    inputs = _normalize_cov_inputs(cov_inputs)
    if len(inputs) == 0:
        _rich_success(
            logger,
            f"Loading covariates ({context}) ...Skipped (none)",
            use_spinner=use_spinner,
            log_message=f"No covariate input provided for {context}; skipping covariates.",
        )
        return None, None

    cov_files, cov_sites = _split_cov_sources(inputs)
    parts: list[tuple[np.ndarray, np.ndarray, str]] = []

    for path in cov_files:
        if not os.path.isfile(path):
            logger.warning(f"Covariate file not found: {path}; skipped.")
            continue
        src = _basename_only(path)
        with CliStatus(f"Loading covariate from {src}...", enabled=bool(use_spinner)) as task:
            try:
                ids_i, cov_i = _read_cov_file_flexible(path, sample_ids, logger, label="Covariate")
                if cov_i.ndim == 1:
                    cov_i = cov_i.reshape(-1, 1)
            except Exception:
                task.fail(f"Loading covariate from {src} ...Failed")
                raise
            task.complete(
                f"Loading covariate from {src} (n={cov_i.shape[0]}, ncov={cov_i.shape[1]}) ...Finished"
            )
        parts.append((np.asarray(ids_i, dtype=str), np.asarray(cov_i, dtype="float32"), src))

    if len(cov_sites) > 0:
        for chrom, pos in cov_sites:
            token = f"{chrom}:{pos}"
            with CliStatus(f"Loading covariate from {token}...", enabled=bool(use_spinner)) as task:
                try:
                    cov_site = _load_site_covariates(
                        genofile=genofile,
                        site_specs=[(chrom, pos)],
                        sample_ids=np.asarray(sample_ids, dtype=str),
                        chunk_size=chunk_size,
                        logger=logger,
                        use_spinner=use_spinner,
                        snps_only=bool(snps_only),
                    )
                except Exception:
                    task.fail(f"Loading covariate from {token} ...Failed")
                    raise
                task.complete(
                    f"Loading covariate from {token} (n={cov_site.shape[0]}, ncov={cov_site.shape[1]}) ...Finished"
                )
            parts.append((np.asarray(sample_ids, dtype=str), cov_site, token))

    if len(parts) == 0:
        logger.warning(f"All covariate inputs are empty/unavailable for {context}; skipping.")
        return None, None

    common = set(parts[0][0].astype(str))
    for ids_i, _cov_i, _name in parts[1:]:
        common &= set(ids_i.astype(str))

    ordered_ids = [sid for sid in np.asarray(sample_ids, dtype=str) if sid in common]
    if len(ordered_ids) == 0:
        raise ValueError(f"No overlapping samples between genotype and covariates ({context}).")

    mats: list[np.ndarray] = []
    for ids_i, cov_i, _name in parts:
        idx_map = {sid: i for i, sid in enumerate(ids_i.astype(str))}
        take = [idx_map[sid] for sid in ordered_ids]
        sub = cov_i[take]
        mats.append(sub.astype("float32", copy=False))

    cov_all = np.concatenate(mats, axis=1).astype("float32", copy=False)
    cov_ids = np.asarray(ordered_ids, dtype=str)
    return cov_all, cov_ids


def load_phenotype(
    phenofile: str,
    ncol: Union[list[int] , None],
    logger,
    id_col: int = 0,
    use_spinner: bool = False,
) -> pd.DataFrame:
    """
    Load and preprocess phenotype table.

    Assumptions
    -----------
      - By default, the first column contains sample IDs.
      - If needed, set id_col=1 to use the second column as IDs (PLINK FID/IID).
    - Duplicated IDs are averaged.
    """
    def _sniff_sep(path: str) -> str:
        """
        Fast delimiter sniffing for phenotype tables.
        Returns one of: 'tab', 'comma', 'whitespace'.
        """
        sample = ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                for _ in range(16):
                    line = fh.readline()
                    if not line:
                        break
                    s = line.strip()
                    if s != "":
                        sample = s
                        break
        except Exception:
            sample = ""

        if "\t" in sample:
            return "tab"
        if "," in sample:
            return "comma"
        return "whitespace"

    def _candidate_orders(kind: str) -> list[str]:
        if kind == "tab":
            return ["tab", "comma", "whitespace"]
        if kind == "comma":
            return ["comma", "tab", "whitespace"]
        return ["whitespace", "tab", "comma"]

    ncol_requested: Union[list[int], None] = None
    if ncol is not None:
        ncol_requested = [int(i) for i in ncol]

    # If phenotype columns are explicitly requested, read only ID + target cols.
    # ncol indices are relative to phenotype columns (after removing ID/FID).
    usecols: Union[list[int], None] = None
    if ncol_requested is not None and len(ncol_requested) > 0 and int(id_col) in (0, 1):
        offset = 1 if int(id_col) == 0 else 2
        try:
            wanted = [int(id_col)] + [int(i) + offset for i in ncol_requested]
            usecols = sorted(set(wanted))
        except Exception:
            usecols = None

    sniffed = _sniff_sep(phenofile)
    read_err: Optional[Exception] = None
    df = None
    mixed_type_warned = False
    for mode in _candidate_orders(sniffed):
        try:
            kwargs: dict[str, object] = {"header": None}
            if usecols is not None:
                kwargs["usecols"] = usecols
            # Keep dtype inference in one pass and avoid chunked mixed-type warnings.
            kwargs["low_memory"] = False
            if mode == "tab":
                kwargs["sep"] = "\t"
                kwargs["engine"] = "c"
            elif mode == "comma":
                kwargs["sep"] = ","
                kwargs["engine"] = "c"
            else:
                kwargs["delim_whitespace"] = True
                kwargs["engine"] = "c"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                df_try = pd.read_csv(phenofile, **kwargs)
            if len(caught) > 0:
                for w in caught:
                    if "DtypeWarning" in str(type(getattr(w, "message", w))):
                        mixed_type_warned = True
                        break
            if df_try.shape[1] <= int(id_col):
                continue
            data_cols = int(df_try.shape[1]) - 1 - (1 if int(id_col) == 1 else 0)
            if data_cols <= 0:
                continue
            df = df_try
            break
        except Exception as ex:
            read_err = ex
            continue

    if df is None:
        if read_err is not None:
            raise read_err
        raise ValueError("Failed to read phenotype file.")
    if mixed_type_warned:
        logger.warning(
            "Phenotype file has mixed-type columns; JanusX will coerce phenotype "
            "values to numeric and set non-numeric cells to NaN. "
            "If this is unintended, clean the phenotype file or select columns via -n."
        )

    if df.empty:
        raise ValueError("Phenotype file is empty.")
    if id_col >= df.shape[1]:
        raise ValueError(f"Phenotype file has no column {id_col + 1} for sample IDs.")

    # Detect header-like first row (non-numeric phenotype columns).
    header_like = False
    header_names = None
    if df.shape[0] > 1 and df.shape[1] > 1:
        row0 = pd.to_numeric(df.iloc[0, 1:], errors="coerce")
        row1 = pd.to_numeric(df.iloc[1, 1:], errors="coerce")
        if row0.isna().all() and row1.notna().any():
            header_like = True
            header_names = df.iloc[0, 1:].astype(str).tolist()
            df = df.iloc[1:, :].reset_index(drop=True)

    ids = df.iloc[:, id_col].astype(str)
    data = df.drop(columns=[id_col])
    # If using IID (column 2), drop FID (column 1) as well.
    if id_col == 1 and data.shape[1] >= 2:
        data = data.drop(columns=[0])
    if header_like and header_names is not None and len(header_names) == data.shape[1]:
        data.columns = header_names

    data = data.apply(pd.to_numeric, errors="coerce")
    pheno = data
    pheno.index = ids
    pheno = pheno.groupby(pheno.index).mean()
    selected_ncol: list[int] = list(range(pheno.shape[1]))

    if pheno.shape[1] <= 0:
        msg = (
            "No phenotype data found. Please check the phenotype file format.\n"
            f"{pheno.head()}"
        )
        logger.error(msg)
        raise ValueError(msg)

    if ncol is not None:
        requested_ncol = [int(i) for i in ncol]
        valid_ncol: list[int]
        invalid_ncol: list[int]
        ncol_take: list[int]

        # If usecols pre-filtering is enabled, requested_ncol are global phenotype
        # indices; map them back to local positions in the reduced pheno table.
        if usecols is not None and ncol_requested is not None and int(id_col) in (0, 1):
            offset = 1 if int(id_col) == 0 else 2
            selected_file_cols = [int(c) for c in usecols if int(c) != int(id_col)]
            if int(id_col) == 1:
                selected_file_cols = [c for c in selected_file_cols if c != 0]
            selected_global_ncol = [int(c) - offset for c in selected_file_cols]
            global_to_local = {g: i for i, g in enumerate(selected_global_ncol)}
            valid_ncol = [i for i in requested_ncol if i in global_to_local]
            invalid_ncol = [i for i in requested_ncol if i not in global_to_local]
            ncol_take = [int(global_to_local[i]) for i in valid_ncol]
        else:
            valid_ncol = [i for i in requested_ncol if 0 <= int(i) < int(pheno.shape[1])]
            invalid_ncol = [i for i in requested_ncol if i not in valid_ncol]
            ncol_take = [int(i) for i in valid_ncol]

        if len(requested_ncol) == 0:
            msg = (
                "No phenotype column index was provided for -n/--n. "
                "Use zero-based indices, e.g. -n 0 -n 3."
            )
            logger.error(msg)
            raise ValueError(msg)
        if len(ncol_take) == 0:
            max_idx = int(pheno.shape[1]) - 1
            msg = (
                "Phenotype column index out of range. "
                f"requested={requested_ncol}, valid=[0..{max_idx}]"
            )
            logger.error(msg)
            raise ValueError(msg)
        if len(invalid_ncol) > 0:
            max_idx = int(pheno.shape[1]) - 1
            logger.warning(
                "Ignoring out-of-range phenotype indices: "
                f"{invalid_ncol}. valid=[0..{max_idx}]"
            )
        ncol = [int(i) for i in valid_ncol]
        selected_ncol = [int(i) for i in valid_ncol]
        _log_info(
            logger,
            "Phenotypes to be analyzed: " + "\t".join(map(str, pheno.columns[ncol_take])),
            use_spinner=use_spinner,
        )
        pheno = pheno.iloc[:, ncol_take]
    else:
        selected_ncol = list(range(pheno.shape[1]))

    # Preserve phenotype file column mapping for downstream history recording.
    pheno.attrs["selected_ncol"] = [int(i) for i in selected_ncol]

    return pheno


# ======================================================================
# Low-memory LMM/LM: streaming GRM + PCA with caching
# ======================================================================

def _cache_prefix_tilde(
    genofile: str,
    *,
    snps_only: bool = True,
    maf: float = 0.02,
    max_missing_rate: float = 0.05,
    het_threshold: float = 0.01,
) -> str:
    """
    Cache prefix used by gfreader for VCF/TXT temporary converted files.
    """
    return _gwas_cache_prefix_with_params(
        genofile,
        maf=float(maf),
        geno=float(max_missing_rate),
        snps_only=bool(snps_only),
    )


def _detect_cache_need(
    genofile: str,
    *,
    snps_only: bool = True,
    maf: float = 0.02,
    max_missing_rate: float = 0.05,
    het_threshold: float = 0.01,
) -> tuple[bool, str, list[str]]:
    """
    Detect whether genotype cache build is expected before inspect/load.
    """
    low = str(genofile).lower()
    if low.endswith(".vcf.gz") or low.endswith(".vcf"):
        cprefix = _cache_prefix_tilde(
            genofile,
            snps_only=snps_only,
            maf=float(maf),
            max_missing_rate=float(max_missing_rate),
            het_threshold=float(het_threshold),
        )
        targets = [f"{cprefix}.bed", f"{cprefix}.bim", f"{cprefix}.fam"]
        all_exist = all(os.path.isfile(p) for p in targets)
        stale = False
        if all_exist and os.path.isfile(genofile):
            src_mtime = os.path.getmtime(genofile)
            cache_mtime = min(os.path.getmtime(p) for p in targets)
            stale = cache_mtime < src_mtime
        return (not all_exist) or stale, "vcf", targets

    file_prefix, matrix_path = _resolve_file_input_matrix(genofile)
    if file_prefix and matrix_path:
        if matrix_path.lower().endswith(".npy"):
            return False, "", []
        cprefix = genotype_cache_prefix(genofile, snps_only=True)
        targets = [f"{cprefix}.npy"]
        stale = False
        if os.path.isfile(targets[0]) and os.path.isfile(matrix_path):
            stale = os.path.getmtime(targets[0]) < os.path.getmtime(matrix_path)
        return (not os.path.isfile(targets[0])) or stale, "txt", targets

    return False, "", []


def _format_cache_size(nbytes: int) -> str:
    x = float(max(0, int(nbytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while x >= 1024.0 and idx < len(units) - 1:
        x /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(x)}{units[idx]}"
    return f"{x:.1f}{units[idx]}"


def _load_phenotype_with_status(
    phenofile: str,
    ncol: Union[list[int], None],
    logger: logging.Logger,
    *,
    id_col: int = 0,
    use_spinner: bool = False,
) -> pd.DataFrame:
    """
    Load phenotype with rich/plain CLI status.
    """
    src = _basename_only(phenofile)
    if not bool(use_spinner):
        logger.info(f"Loading phenotype from {src}... [{format_elapsed(0.0)}]")
    with CliStatus(f"Loading phenotype from {src}...", enabled=bool(use_spinner)) as task:
        try:
            pheno = load_phenotype(
                phenofile,
                ncol,
                logger,
                id_col=id_col,
                use_spinner=use_spinner,
            )
        except Exception:
            task.fail(f"Loading phenotype from {src} ...Failed")
            raise
        task.complete(
            f"Loading phenotype from {src} (n={pheno.shape[0]}, npheno={pheno.shape[1]})"
        )
    return pheno


def _inspect_genotype_file_with_warnings(
    genofile: str,
    snps_only: bool = True,
    maf_threshold: float = 0.02,
    max_missing_rate: float = 0.05,
    het_threshold: float = 0.01,
) -> tuple[np.ndarray, int, list[str]]:
    """
    Run inspect_genotype_file and capture warning messages.
    Designed for subprocess execution to avoid GIL stalls during cache build.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ids0, ns0 = inspect_genotype_file(
            genofile,
            snps_only=bool(snps_only),
            maf=float(maf_threshold),
            missing_rate=float(max_missing_rate),
            het=float(het_threshold),
        )
    warn_msgs: list[str] = []
    for w in caught:
        try:
            msg = str(w.message).strip()
        except Exception:
            msg = ""
        if msg:
            warn_msgs.append(msg)
    return np.asarray(ids0, dtype=str), int(ns0), warn_msgs


def _inspect_genotype_with_status(
    genofile: str,
    logger: logging.Logger,
    *,
    use_spinner: bool = False,
    snps_only: bool = True,
    maf_threshold: float = 0.02,
    max_missing_rate: float = 0.05,
    het_threshold: float = 0.01,
    warning_collector: Union[list[str], None] = None,
) -> tuple[np.ndarray, int]:
    """
    Inspect genotype metadata with optional cache-status spinner.
    """
    src = _basename_only(genofile)
    need_cache, cache_kind, cache_targets = _detect_cache_need(
        genofile,
        snps_only=bool(snps_only),
        maf=float(maf_threshold),
        max_missing_rate=float(max_missing_rate),
        het_threshold=float(het_threshold),
    )
    status_enabled = bool(use_spinner)
    plain_progress = (not status_enabled) and (cache_kind in {"vcf", "txt"})

    # For direct PLINK prefixes (no cache-target metadata), inspect directly.
    # For VCF/TXT sources, always run the threaded+monitored path below so that
    # cache rebuilds triggered inside inspect_genotype_file() are still visible.
    if cache_kind == "":
        with CliStatus(f"Loading genotype from {src}...", enabled=status_enabled) as task:
            try:
                ids, n_snps = inspect_genotype_file(
                    genofile,
                    snps_only=bool(snps_only),
                    maf=float(maf_threshold),
                    missing_rate=float(max_missing_rate),
                    het=float(het_threshold),
                )
            except Exception:
                task.fail(f"Loading genotype from {src} ...Failed")
                raise
            task.complete(f"Loading genotype from {src} (n={len(ids)}, nSNP={n_snps})")
        return np.asarray(ids, dtype=str), int(n_snps)

    with CliStatus(f"Loading genotype from {src}...", enabled=status_enabled) as task:
        out: dict[str, tuple[np.ndarray, int]] = {}
        err: dict[str, Exception] = {}
        warn_msgs: list[str] = []
        last_plain_msg = ""
        last_plain_emit = 0.0
        plain_t0 = time.monotonic()
        last_wait_beat = plain_t0
        if plain_progress:
            logger.info(f"Loading genotype from {src}... [{format_elapsed(0.0)}]")
            last_plain_msg = f"Loading genotype from {src}..."
            last_plain_emit = plain_t0

        def _emit_plain(msg: str, *, allow_same_after: float = 5.0) -> None:
            nonlocal last_plain_msg, last_plain_emit
            if not plain_progress:
                return
            m = str(msg)
            now = time.monotonic()
            elapsed = format_elapsed(now - plain_t0)
            changed = (m != last_plain_msg)
            # Emit on change with a small throttle; also emit heartbeat when unchanged.
            if (
                (changed and (now - last_plain_emit >= 0.5))
                or ((not changed) and (now - last_plain_emit >= float(max(0.5, allow_same_after))))
            ):
                logger.info(f"{m} [{elapsed}]")
                last_plain_msg = m
                last_plain_emit = now

        # Use subprocess first so UI updates are not blocked by GIL while Rust builds cache.
        fut = None
        executor = None
        done_evt: Union[threading.Event, None] = None
        t: Union[threading.Thread, None] = None
        use_subproc = cache_kind in {"vcf", "txt"}
        if use_subproc:
            try:
                mp_ctx = mp.get_context("spawn")
                executor = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp_ctx)
                fut = executor.submit(
                    _inspect_genotype_file_with_warnings,
                    genofile,
                    bool(snps_only),
                    float(maf_threshold),
                    float(max_missing_rate),
                    float(het_threshold),
                )
            except Exception:
                use_subproc = False

        if not use_subproc:
            done_evt = threading.Event()

            def _worker() -> None:
                try:
                    ids0, ns0, warns0 = _inspect_genotype_file_with_warnings(
                        genofile,
                        bool(snps_only),
                        float(maf_threshold),
                        float(max_missing_rate),
                        float(het_threshold),
                    )
                    out["value"] = (np.asarray(ids0, dtype=str), int(ns0))
                    warn_msgs.extend(warns0)
                except Exception as ex:
                    err["value"] = ex
                finally:
                    done_evt.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

        bim_path = cache_targets[1] if (cache_kind == "vcf" and len(cache_targets) >= 2) else ""
        bed_path = cache_targets[0] if (cache_kind == "vcf" and len(cache_targets) >= 1) else ""
        npy_path = cache_targets[0] if (cache_kind == "txt" and len(cache_targets) >= 1) else ""
        bim_fp = None
        bim_dev_ino: tuple[int, int] = (-1, -1)
        bim_seen = 0
        bim_count = 0
        last_count = -1
        last_size = -1
        while True:
            if use_subproc:
                if fut is not None and fut.done():
                    break
                time.sleep(0.25)
            else:
                if done_evt is not None and done_evt.wait(timeout=0.25):
                    break
            if bim_path and os.path.isfile(bim_path):
                bim_progressed = False
                try:
                    st = os.stat(bim_path)
                    dev_ino = (int(st.st_dev), int(st.st_ino))
                    if (
                        bim_fp is None
                        or dev_ino != bim_dev_ino
                        or int(st.st_size) < int(bim_seen)
                    ):
                        if bim_fp is not None:
                            try:
                                bim_fp.close()
                            except Exception:
                                pass
                        bim_fp = open(bim_path, "rb")
                        bim_dev_ino = dev_ino
                        bim_seen = 0
                        bim_count = 0
                    if int(st.st_size) > int(bim_seen):
                        bim_fp.seek(int(bim_seen))
                        chunk = bim_fp.read(int(st.st_size) - int(bim_seen))
                        bim_seen = int(st.st_size)
                        bim_count += int(chunk.count(b"\n"))
                        bim_progressed = True
                        if bim_count != last_count:
                            last_count = bim_count
                            msg = f"Loading genotype from {src}... SNP={bim_count}"
                            task.desc = msg
                            _emit_plain(msg)
                except Exception:
                    pass
                if (not bim_progressed) and bed_path and os.path.isfile(bed_path):
                    # .bim may not flush frequently; keep progress alive via .bed growth.
                    try:
                        bsz = int(os.path.getsize(bed_path))
                        if bsz != last_size:
                            last_size = bsz
                            msg = f"Loading genotype from {src}... cache={_format_cache_size(bsz)}"
                            task.desc = msg
                            _emit_plain(msg)
                    except Exception:
                        pass
            elif bed_path and os.path.isfile(bed_path):
                try:
                    sz = int(os.path.getsize(bed_path))
                    if sz != last_size:
                        last_size = sz
                        msg = f"Loading genotype from {src}... cache={_format_cache_size(sz)}"
                        task.desc = msg
                        _emit_plain(msg)
                except Exception:
                    pass
            elif npy_path and os.path.isfile(npy_path):
                try:
                    sz = int(os.path.getsize(npy_path))
                    if sz != last_size:
                        last_size = sz
                        msg = f"Loading genotype from {src}... cache={_format_cache_size(sz)}"
                        task.desc = msg
                        _emit_plain(msg)
                except Exception:
                    pass
            now = time.monotonic()
            if now - last_wait_beat >= 3.0:
                dots = "." * (1 + (int((now - plain_t0) // 3) % 3))
                wait_msg = f"Loading genotype from {src}... waiting{dots}"
                task.desc = wait_msg
                _emit_plain(wait_msg, allow_same_after=3.0)
                last_wait_beat = now

        if use_subproc:
            try:
                if fut is not None:
                    ids0, ns0, warns0 = fut.result()
                    out["value"] = (np.asarray(ids0, dtype=str), int(ns0))
                    warn_msgs.extend(list(warns0))
            except Exception as ex:
                err["value"] = ex
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)
        elif t is not None:
            t.join()
        if bim_fp is not None:
            try:
                bim_fp.close()
            except Exception:
                pass
        if "value" in err:
            task.fail(f"Loading genotype from {src} ...Failed")
            raise err["value"]

        ids, n_snps = out["value"]
        if warning_collector is not None:
            warning_collector.extend(warn_msgs)
        else:
            for wmsg in warn_msgs:
                logger.warning(wmsg)
        task.desc = f"Loading genotype from {src}... SNP={n_snps}"
        task.complete(f"Loading genotype from {src} (n={len(ids)}, nSNP={n_snps})")
        _log_info(logger, f"Cached genotype sites: {n_snps}", use_spinner=use_spinner)
        return ids, n_snps

def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    method: int,
    mmap_window_mb: Union[int , None],
    threads: int,
    logger,
    use_spinner: bool = False,
    snps_only: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Build GRM in a streaming fashion using rust2py.gfreader.load_genotype_chunks.
    """
    _log_info(logger, f"Building GRM (streaming), method={method}", use_spinner=use_spinner)
    pbar = _ProgressAdapter(total=n_snps, desc="GRM (streaming)", force_animate=True)
    process = psutil.Process()

    prefetch_depth = 2
    tuned_chunk_size = auto_stream_grm_chunk_size(
        n_samples=n_samples,
        requested_chunk_size=chunk_size,
        threads=threads,
        prefetch_depth=prefetch_depth,
    )
    if tuned_chunk_size != int(chunk_size):
        _log_file_only(
            logger,
            logging.INFO,
            (
                "Auto-tuned GRM chunk size: "
                f"{chunk_size} -> {tuned_chunk_size} "
                f"(n_samples={n_samples}, threads={threads})"
            ),
        )
    mem_tick_span = max(1, 10 * int(tuned_chunk_size))

    chunk_iter = load_genotype_chunks(
        genofile,
        tuned_chunk_size,
        maf_threshold,
        max_missing_rate,
        model="add",
        snps_only=bool(snps_only),
        mmap_window_mb=mmap_window_mb,
    )

    def _on_grm_chunk(added_snps: int, total_eff: int) -> None:
        pbar.update(int(added_snps))
        if total_eff % mem_tick_span == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f}GB")

    try:
        grm, grm_stats = build_streaming_grm_from_chunks(
            prefetch_iter(chunk_iter, in_flight=prefetch_depth),
            n_samples=n_samples,
            method=method,
            accumulate="gemm",
            on_chunk=_on_grm_chunk,
        )
        # force bar to 100% even if SNPs were filtered in Rust
        pbar.finish()
    finally:
        # Always stop progress renderer on Ctrl+C / exceptions.
        pbar.close(show_done=False)

    _log_info(logger, "GRM construction finished.", use_spinner=use_spinner)
    return grm, int(grm_stats.eff_m)

def load_or_build_grm_with_cache(
    genofile: str,
    cache_prefix: str,
    mgrm: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    chunk_size: int,
    threads: int,
    mmap_limit: bool,
    logger:logging.Logger,
    use_spinner: bool = False,
    ids_preloaded: Union[np.ndarray, None] = None,
    n_snps_preloaded: Union[int, None] = None,
    snps_only: bool = True,
) -> tuple[np.ndarray, int, Union[np.ndarray, None]]:
    """
    Load or build a GRM with caching for streaming LMM/LM runs.
    """
    if ids_preloaded is not None and n_snps_preloaded is not None:
        ids = np.asarray(ids_preloaded, dtype=str)
        n_snps = int(n_snps_preloaded)
    else:
        ids0, n_snps0 = inspect_genotype_file(
            genofile,
            snps_only=bool(snps_only),
            maf=float(maf_threshold),
            missing_rate=float(max_missing_rate),
            het=float(het_threshold),
        )
        ids = np.asarray(ids0, dtype=str)
        n_snps = int(n_snps0)
    n_samples = int(len(ids))
    method_is_builtin = mgrm in ["1", "2"]

    grm_ids = None
    if method_is_builtin:
        grm_path, id_path = _grm_cache_paths(cache_prefix, mgrm=str(mgrm))
        with _cache_lock(grm_path):
            cache_is_stale = False
            if os.path.exists(grm_path):
                g_mtime = latest_genotype_mtime(genofile)
                k_mtime = os.path.getmtime(grm_path)
                if g_mtime is not None and g_mtime > k_mtime:
                    cache_is_stale = True
                    logger.warning(
                        "Genotype input is newer than cached GRM; rebuilding GRM cache."
                    )
            if cache_is_stale:
                for p in (grm_path, id_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

            if os.path.exists(grm_path) and (not cache_is_stale):
                src = _basename_only(grm_path)
                with CliStatus(f"Loading GRM from {src}...", enabled=bool(use_spinner)) as task:
                    try:
                        grm = np.load(grm_path, mmap_mode='r')
                        grm = grm.reshape(n_samples, n_samples)
                        grm_ids = _read_id_file(
                            id_path,
                            logger,
                            "GRM",
                            use_spinner=use_spinner,
                            show_status=False,
                        )
                        if grm_ids is not None and len(grm_ids) != n_samples:
                            raise ValueError(
                                f"GRM ID count ({len(grm_ids)}) does not match GRM shape ({n_samples})."
                            )
                        if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
                            raise ValueError(f"GRM must be square; got shape={grm.shape}")
                        eff_m = n_snps  # approximate; exact effective M not critical here
                    except Exception:
                        task.fail(f"Loading GRM from {src} ...Failed")
                        raise
                    task.complete(f"Loading GRM from {src} (n={grm.shape[0]})")
            else:
                method_int = int(mgrm)
                grm_calc_t0 = time.monotonic()
                grm, eff_m = build_grm_streaming(
                    genofile=genofile,
                    n_samples=n_samples,
                    n_snps=n_snps,
                    maf_threshold=maf_threshold,
                    max_missing_rate=max_missing_rate,
                    chunk_size=chunk_size,
                    method=method_int,
                    mmap_window_mb=auto_mmap_window_mb(
                        genofile, n_samples, n_snps, chunk_size
                    ) if mmap_limit else None,
                    threads=threads,
                    logger=logger,
                    use_spinner=use_spinner,
                    snps_only=bool(snps_only),
                )
                grm_msg = (
                    f"Calculating GRM from genotype (n={int(n_samples)}) "
                    f"...Finished [{format_elapsed(time.monotonic() - grm_calc_t0)}]"
                )
                _log_file_only(logger, logging.INFO, grm_msg)
                print_success(grm_msg, force_color=bool(use_spinner))
                tmp_grm = f"{grm_path}.tmp.{os.getpid()}.npy"
                np.save(tmp_grm, grm)
                os.replace(tmp_grm, grm_path)
                tmp_id = f"{id_path}.tmp.{os.getpid()}"
                pd.Series(ids).to_csv(tmp_id, sep="\t", index=False, header=False)
                os.replace(tmp_id, id_path)
                grm_ids = ids
                grm = np.load(grm_path, mmap_mode='r')
                if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
                    raise ValueError(f"GRM must be square; got shape={grm.shape}")
                _log_file_only(logger, logging.INFO, f"Cached GRM written to {grm_path}")
    else:
        if not os.path.isfile(mgrm):
            msg = f"GRM file not found: {mgrm}"
            logger.error(msg)
            raise ValueError(msg)
        src = _basename_only(mgrm)
        with CliStatus(f"Loading GRM from {src}...", enabled=bool(use_spinner)) as task:
            try:
                if mgrm.endswith('.npy'):
                    grm = np.load(mgrm,mmap_mode='r')
                else:
                    grm = np.genfromtxt(mgrm, dtype="float32")
                grm_ids = _read_id_file(
                    f"{mgrm}.id",
                    logger,
                    "GRM",
                    use_spinner=use_spinner,
                    show_status=False,
                )
                if grm_ids is None:
                    if grm.size != n_samples * n_samples:
                        msg = f"GRM size mismatch: expected {n_samples*n_samples}, got {grm.size}"
                        logger.error(msg)
                        raise ValueError(msg)
                    grm = grm.reshape(n_samples, n_samples)
                else:
                    if grm.size != len(grm_ids) * len(grm_ids):
                        msg = (
                            f"GRM size mismatch: expected {len(grm_ids)*len(grm_ids)}, "
                            f"got {grm.size}"
                        )
                        logger.error(msg)
                        raise ValueError(msg)
                    grm = grm.reshape(len(grm_ids), len(grm_ids))
                if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
                    raise ValueError(f"GRM must be square; got shape={grm.shape}")
                eff_m = n_snps
            except Exception:
                task.fail(f"Loading GRM from {src} ...Failed")
                raise
            task.complete(f"Loading GRM from {src} (n={grm.shape[0]})")

    _log_info(logger, f"GRM shape: {grm.shape}", use_spinner=use_spinner)
    return grm, eff_m, grm_ids


def build_pcs_from_grm(
    grm: np.ndarray,
    dim: int,
    logger: logging.Logger,
    *,
    use_spinner: bool = False,
) -> np.ndarray:
    """
    Compute leading principal components from GRM.
    """
    if use_spinner:
        with CliStatus(f"Computing top {dim} PCs from GRM...", enabled=True) as task:
            try:
                _eigval, eigvec = np.linalg.eigh(grm)
                pcs = eigvec[:, -dim:]
            except Exception:
                task.fail(f"Computing top {dim} PCs from GRM ...Failed")
                raise
            task.complete(
                f"Computing top {dim} PCs from GRM (n={pcs.shape[0]}, nPC={pcs.shape[1]})"
            )
        _log_file_only(logger, logging.INFO, f"Computing top {dim} PCs from GRM...")
        _log_file_only(logger, logging.INFO, "PC computation finished.")
        return pcs

    logger.info(f"Computing top {dim} PCs from GRM...")
    _eigval, eigvec = np.linalg.eigh(grm)
    pcs = eigvec[:, -dim:]
    logger.info("PC computation finished.")
    return pcs


def load_or_build_q_with_cache(
    genofile: str,
    grm: Union[np.ndarray, None],
    n_samples: int,
    cache_prefix: str,
    pcdim: str,
    mgrm: str,
    ids: np.ndarray,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    snps_only: bool,
    logger,
    use_spinner: bool = False,
) -> tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Load or build Q matrix (PCs) with caching for streaming LMM/LM.
    Note: external Q file via -q is no longer supported; pass external
    covariate matrices via -c.
    """
    n = int(n_samples)
    qdim = _parse_qcov_dim(pcdim)
    if qdim >= n and qdim != 0:
        raise ValueError(
            f"Q/PC dimension out of range: {qdim}. valid=[0..{max(0, n-1)}]"
        )

    q_ids = None
    if qdim > 0:
        dim = int(qdim)
        q_path = _pca_cache_path(cache_prefix, mgrm=str(mgrm), qdim=int(dim))
        with _cache_lock(q_path):
            cache_ready = os.path.isfile(q_path)
            if cache_ready:
                g_mtime = latest_genotype_mtime(genofile)
                q_mtime = os.path.getmtime(q_path)
                if g_mtime is not None and g_mtime > q_mtime:
                    cache_ready = False
                    logger.info("Genotype input is newer than cached PCA; rebuilding cache.")
            if cache_ready:
                src = _basename_only(q_path)
                with CliStatus(f"Loading Q matrix from {src}...", enabled=bool(use_spinner)) as task:
                    try:
                        qmatrix = np.loadtxt(q_path, dtype="float32", delimiter="\t")
                        if qmatrix.ndim == 1:
                            qmatrix = qmatrix.reshape(-1, 1)
                        if qmatrix.shape != (n, int(dim)):
                            raise ValueError(
                                f"PCA cache shape mismatch: expected ({n},{dim}), got {qmatrix.shape}"
                            )
                        q_ids = ids
                    except Exception:
                        task.fail(f"Loading Q matrix from {src} ...Failed")
                        raise
                    task.complete(
                        f"Loading Q matrix from {src} (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]}) ...Finished"
                    )
            else:
                if grm is None:
                    raise ValueError(
                        "Q cache not found and GRM is unavailable; cannot generate PCA covariates."
                    )
                pc_calc_t0 = time.monotonic()
                eigval, eigvec = np.linalg.eigh(grm)
                qmatrix = np.asarray(eigvec[:, -dim:], dtype="float32")
                pc_msg = (
                    f"Calculating PCs from GRM (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]}) "
                    f"...Finished [{format_elapsed(time.monotonic() - pc_calc_t0)}]"
                )
                _log_file_only(logger, logging.INFO, pc_msg)
                print_success(pc_msg, force_color=bool(use_spinner))
                tmp_q = f"{q_path}.tmp.{os.getpid()}"
                np.savetxt(tmp_q, qmatrix, fmt="%.8g", delimiter="\t")
                os.replace(tmp_q, q_path)
                q_ids = ids
                _log_file_only(
                    logger,
                    logging.INFO,
                    f"Cached PCA written to {q_path}",
                )
    elif qdim == 0:
        pc_warn_msg = "PC dimension set to 0; using empty Q matrix."
        if use_spinner or stdout_is_tty():
            _log_file_only(logger, logging.WARNING, pc_warn_msg)
            print_warning(pc_warn_msg, force_color=bool(use_spinner))
        else:
            logger.warning(pc_warn_msg)
        qmatrix = np.zeros((n, 0), dtype="float32")
        q_ids = ids
    else:
        raise ValueError(
            "External Q matrix via -q/--qcov is no longer supported. "
            "Use -q <int> and provide external covariates via -c."
        )

    _log_info(logger, f"Q matrix shape: {qmatrix.shape}", use_spinner=use_spinner)
    return qmatrix, q_ids


def _load_covariate_for_streaming(
    cov_inputs: Union[str, list[str], None],
    genofile: str,
    sample_ids: np.ndarray,
    chunk_size: int,
    logger,
    use_spinner: bool = False,
    snps_only: bool = True,
) -> tuple[Union[np.ndarray , None], Union[np.ndarray, None]]:
    """
    Backward-compatible wrapper for streaming covariate loading.
    """
    return _load_covariates_for_models(
        cov_inputs=cov_inputs,
        genofile=genofile,
        sample_ids=np.asarray(sample_ids, dtype=str),
        chunk_size=int(chunk_size),
        logger=logger,
        context="streaming",
        use_spinner=use_spinner,
        snps_only=bool(snps_only),
    )


def prepare_streaming_context(
    genofile: str,
    phenofile: str,
    pheno_cols: Union[list[int] , None],
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    mgrm: str,
    pcdim: str,
    cov_inputs: Union[str, list[str], None],
    threads: int,
    mmap_limit: bool,
    require_kinship: bool,
    logger,
    use_spinner: bool = False,
    snps_only: bool = True,
):
    """
    Prepare all shared resources for streaming LMM/LM once:
      - phenotype
      - genotype metadata (ids, n_snps)
      - GRM + Q (cached)
      - covariates (optional)
    """
    pheno = _load_phenotype_with_status(
        phenofile,
        pheno_cols,
        logger,
        id_col=0,
        use_spinner=use_spinner,
    )

    deferred_cache_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ids, n_snps = _inspect_genotype_with_status(
            genofile,
            logger,
            use_spinner=use_spinner,
            snps_only=bool(snps_only),
            maf_threshold=float(maf_threshold),
            max_missing_rate=float(max_missing_rate),
            het_threshold=float(het_threshold),
            warning_collector=deferred_cache_warnings,
        )
        for w in caught:
            try:
                msg = str(w.message).strip()
            except Exception:
                msg = ""
            if _is_cache_warning_message(msg):
                deferred_cache_warnings.append(msg)
    n_samples = len(ids)
    _log_info(
        logger,
        f"Genotype meta: {n_samples} samples, {n_snps} SNPs.",
        use_spinner=use_spinner,
    )

    cache_prefix = _gwas_cache_prefix_with_params(
        genofile,
        maf=float(maf_threshold),
        geno=float(max_missing_rate),
        snps_only=bool(snps_only),
        logger=logger,
        warning_collector=deferred_cache_warnings,
    )
    _log_info(
        logger,
        f"Cache prefix: {cache_prefix}",
        use_spinner=use_spinner,
    )
    stream_genofile = str(genofile)
    genofile_low = str(genofile).lower()
    if genofile_low.endswith(".vcf") or genofile_low.endswith(".vcf.gz"):
        if all(os.path.isfile(f"{cache_prefix}.{ext}") for ext in ("bed", "bim", "fam")):
            # Reuse the parameterized VCF cache prefix across streaming steps.
            # This avoids creating extra VCF->BED caches with other default filter tags.
            stream_genofile = str(cache_prefix)

    need_generate_q = False
    if pcdim in np.arange(1, n_samples).astype(str):
        qdim = int(pcdim)
        q_path = _pca_cache_path(cache_prefix, mgrm=str(mgrm), qdim=int(qdim))
        if not os.path.isfile(q_path):
            need_generate_q = True
        else:
            g_mtime = latest_genotype_mtime(genofile)
            q_mtime = os.path.getmtime(q_path)
            if g_mtime is not None and g_mtime > q_mtime:
                need_generate_q = True
            else:
                try:
                    q_probe = np.loadtxt(q_path, dtype="float32", delimiter="\t")
                    if q_probe.ndim == 1:
                        q_probe = q_probe.reshape(-1, 1)
                    if q_probe.shape != (n_samples, int(qdim)):
                        need_generate_q = True
                except Exception:
                    need_generate_q = True
    need_grm = bool(require_kinship or need_generate_q)

    grm: Union[np.ndarray, None] = None
    eff_m = n_snps
    grm_ids = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if need_grm:
            # GRM stream...
            grm, eff_m, grm_ids = load_or_build_grm_with_cache(
                genofile=stream_genofile,
                cache_prefix=cache_prefix,
                mgrm=mgrm,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                het_threshold=het_threshold,
                chunk_size=chunk_size,
                threads=threads,
                mmap_limit=mmap_limit,
                logger=logger,
                use_spinner=use_spinner,
                ids_preloaded=ids,
                n_snps_preloaded=n_snps,
                snps_only=bool(snps_only),
            )
        else:
            _rich_success(
                logger,
                "GRM not required (no LMM/fastLMM and q=0).",
                use_spinner=use_spinner,
            )

        # PCA stream...
        qmatrix, q_ids = load_or_build_q_with_cache(
            genofile=genofile,
            grm=grm,
            n_samples=n_samples,
            cache_prefix=cache_prefix,
            pcdim=pcdim,
            mgrm=mgrm,
            ids=ids,
            maf_threshold=maf_threshold,
            max_missing_rate=max_missing_rate,
            het_threshold=het_threshold,
            snps_only=bool(snps_only),
            logger=logger,
            use_spinner=use_spinner,
        )

        cov_all, cov_ids = _load_covariate_for_streaming(
            cov_inputs,
            stream_genofile,
            ids,
            chunk_size,
            logger,
            use_spinner=use_spinner,
            snps_only=bool(snps_only),
        )
        for w in caught:
            try:
                msg = str(w.message).strip()
            except Exception:
                msg = ""
            if _is_cache_warning_message(msg):
                deferred_cache_warnings.append(msg)

    # -----------------------------------------
    # Align all data sources to shared IDs
    # -----------------------------------------
    geno_ids = ids.astype(str)
    pheno_ids = pheno.index.astype(str).to_numpy()

    # If optional ID files are missing, inherit genotype order
    if grm is not None:
        if grm_ids is None:
            if grm.shape[0] != n_samples:
                raise ValueError(
                    f"GRM size mismatch: {grm.shape[0]} != genotype samples {n_samples} "
                    "and no GRM ID file was provided."
                )
            logger.warning("GRM IDs not provided; assuming genotype order.")
            grm_ids = ids
        else:
            grm_ids = np.asarray(grm_ids, dtype=str)
    if q_ids is None:
        if qmatrix.shape[0] != n_samples:
            raise ValueError(
                f"Q matrix size mismatch: {qmatrix.shape[0]} != genotype samples {n_samples} "
                "and no Q ID file was provided."
            )
        logger.warning("Q IDs not provided; assuming genotype order.")
        q_ids = ids
    else:
        q_ids = np.asarray(q_ids, dtype=str)
    if cov_ids is None and cov_all is not None:
        if cov_all.shape[0] != n_samples:
            raise ValueError(
                f"Covariate size mismatch: {cov_all.shape[0]} != genotype samples {n_samples} "
                "and no covariate ID file was provided."
            )
        logger.warning("Covariate IDs not provided; assuming genotype order.")
        cov_ids = ids
    elif cov_ids is not None:
        cov_ids = np.asarray(cov_ids, dtype=str)

    common = set(geno_ids) & set(pheno_ids)
    if grm_ids is not None:
        common &= set(grm_ids.astype(str))
    if q_ids is not None:
        common &= set(q_ids.astype(str))
    if cov_ids is not None:
        common &= set(cov_ids.astype(str))

    common_ids = [i for i in geno_ids if i in common]
    if len(common_ids) == 0:
        # Try using IID (second column) for PLINK-style phenotype files
        try:
            pheno_alt = load_phenotype(
                phenofile,
                pheno_cols,
                logger,
                id_col=1,
                use_spinner=use_spinner,
            )
            pheno_ids_alt = pheno_alt.index.astype(str).to_numpy()
            common_alt = set(geno_ids) & set(pheno_ids_alt)
            if grm_ids is not None:
                common_alt &= set(grm_ids.astype(str))
            if q_ids is not None:
                common_alt &= set(q_ids.astype(str))
            if cov_ids is not None:
                common_alt &= set(cov_ids.astype(str))
            if len(common_alt) > 0:
                logger.warning("Using phenotype column 2 (IID) as sample IDs.")
                pheno = pheno_alt
                pheno_ids = pheno_ids_alt
                common = common_alt
        except Exception as e:
            logger.warning(f"Failed to parse phenotype with IID column: {e}")

    common_ids = [i for i in geno_ids if i in common]
    if len(common_ids) == 0:
        logger.error("No overlapping samples across genotype/phenotype/GRM/Q/cov.")
        logger.error(f"Genotype IDs (first 5): {list(geno_ids[:5])}")
        logger.error(f"Phenotype IDs (first 5): {list(pheno_ids[:5])}")
        if grm_ids is not None:
            logger.error(f"GRM IDs (first 5): {list(grm_ids[:5])}")
        if q_ids is not None:
            logger.error(f"Q IDs (first 5): {list(q_ids[:5])}")
        if cov_ids is not None:
            logger.error(f"Covariate IDs (first 5): {list(cov_ids[:5])}")
        raise ValueError("No overlapping samples across genotype/phenotype/GRM/Q/cov.")

    for wmsg in _dedupe_cache_warning_messages(deferred_cache_warnings):
        logger.warning(wmsg)
    grm_n: Union[int, str] = "NA" if grm_ids is None else int(len(grm_ids))
    q_has_pc = bool(np.asarray(qmatrix).ndim == 2 and int(qmatrix.shape[1]) > 0)
    q_n: Union[int, str] = "NA" if (q_ids is None or (not q_has_pc)) else int(len(q_ids))
    cov_n: Union[int, str] = "NA" if cov_ids is None else int(len(cov_ids))
    _emit_plain_info_line(
        logger,
        (
            f"geno={len(geno_ids)}, pheno={len(pheno_ids)}, "
            f"grm={grm_n}, q={q_n}, cov={cov_n} -> {len(common_ids)}"
        ),
        use_spinner=use_spinner,
    )

    # index maps
    geno_index = {sid: i for i, sid in enumerate(geno_ids)}
    if grm_ids is not None:
        grm_index = {sid: i for i, sid in enumerate(grm_ids.astype(str))}
    else:
        grm_index = geno_index
    if q_ids is not None:
        q_index = {sid: i for i, sid in enumerate(q_ids.astype(str))}
    else:
        q_index = geno_index
    if cov_ids is not None:
        cov_index = {sid: i for i, sid in enumerate(cov_ids.astype(str))}
    else:
        cov_index = geno_index

    # reorder/trim
    ids = np.array(common_ids)
    pheno = pheno.loc[ids]

    if grm is not None:
        grm_idx = [grm_index[sid] for sid in ids]
        grm = grm[np.ix_(grm_idx, grm_idx)]

    q_idx = [q_index[sid] for sid in ids]
    qmatrix = qmatrix[q_idx]

    if cov_all is not None:
        cov_idx = [cov_index[sid] for sid in ids]
        cov_all = cov_all[cov_idx]

    return pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m, stream_genofile


def _as_plink_prefix(path_or_prefix: str) -> Union[str, None]:
    p = str(path_or_prefix).strip()
    if p == "":
        return None
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        p = p[:-4]
    if all(os.path.isfile(f"{p}.{ext}") for ext in ("bed", "bim", "fam")):
        return p
    return None


def _read_bim_sites(prefix: str) -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    bim_path = f"{prefix}.bim"
    with open(bim_path, "r", encoding="utf-8", errors="ignore") as fh:
        for ln, line in enumerate(fh, start=1):
            s = line.strip()
            if s == "":
                continue
            toks = s.split()
            if len(toks) < 6:
                raise ValueError(f"Malformed BIM line at {bim_path}:{ln}")
            chrom = str(toks[0])
            try:
                pos = int(float(toks[3]))
            except Exception as e:
                raise ValueError(f"Invalid BIM POS at {bim_path}:{ln}") from e
            a0 = str(toks[4])
            a1 = str(toks[5])
            out.append((chrom, pos, a0, a1))
    return out


def _resolve_lrlmm_rank(rank_opt: Union[int, None], n_samples: int) -> int:
    n = max(1, int(n_samples))
    if rank_opt is None or int(rank_opt) <= 0:
        return max(1, int(math.sqrt(n)))
    return max(1, min(int(rank_opt), n))


def _resolve_stream_scan_chunk_size(
    chunk_size: int,
    n_snps_hint: int,
    *,
    use_spinner: bool,
) -> int:
    """
    Resolve effective streaming scan chunk size.

    Respect user-provided chunk size strictly.
    """
    _ = int(max(1, int(n_snps_hint)))
    _ = bool(use_spinner)
    return max(1, int(chunk_size))


def _status_stage_thread_budget(threads: int) -> int:
    """
    Reserve one logical core for UI/status refresh when possible.
    """
    t = max(1, int(threads))
    return max(1, t - 1) if t > 1 else 1


@contextmanager
def _native_thread_limit_ctx(limit: int):
    """
    Limit BLAS/OpenMP threadpools during long linear algebra stages.
    """
    lim = max(1, int(limit))
    if _threadpool_limits is None:
        yield
        return
    with _threadpool_limits(limits=lim):
        yield


def _calibrate_lrlmm_spectrum(
    eigvals: np.ndarray,
    n_samples: int,
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Default LowRankLMM spectrum calibration (always enabled):
      1) low-rank + diagonal closure: K ~= Ur*Lr*Ur^T + tau*I
      2) trace matching to full-rank scale (target trace ~= n_samples)

    Returns
    -------
    s_cal : np.ndarray[float32]
        Calibrated leading eigenvalues (length r).
    tau_cal : float
        Diagonal closure term after trace calibration.
    scale : float
        Multiplicative scale used in trace matching.
    raw_trace : float
        Sum of raw leading eigenvalues before calibration.
    target_trace : float
        Target trace used for matching.
    """
    n = max(1, int(n_samples))
    target_trace = float(n)

    s = np.asarray(eigvals, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return (
            np.asarray([], dtype=np.float32),
            0.0,
            1.0,
            0.0,
            target_trace,
        )
    s[~np.isfinite(s)] = 0.0
    s = np.maximum(s, 0.0)

    raw_trace = float(np.sum(s))
    tau = 0.0
    if raw_trace < target_trace:
        tau = (target_trace - raw_trace) / float(n)

    trace_with_tau = raw_trace + tau * float(n)
    scale = 1.0
    if np.isfinite(trace_with_tau) and trace_with_tau > 0.0:
        scale = target_trace / trace_with_tau
        if (not np.isfinite(scale)) or scale <= 0.0:
            scale = 1.0

    s_cal = np.ascontiguousarray((s * scale).astype(np.float32, copy=False))
    tau_cal = float(tau * scale)
    return s_cal, tau_cal, float(scale), raw_trace, target_trace


def run_lrlmm_packed(
    *,
    rank_opt: Union[int, None],
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    chunk_size: int,
    mmap_limit: bool,
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray, None],
    plot: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    emit_trait_header: bool = True,
    prepared_cache: Union[dict[str, object], None] = None,
) -> None:
    if not hasattr(jxrs, "rsvd_packed_subset"):
        raise RuntimeError(
            "Rust extension is missing rsvd_packed_subset. Rebuild/install JanusX extension first."
        )
    if not hasattr(jxrs, "fastlmm_assoc_packed_f32"):
        raise RuntimeError(
            "Rust extension is missing fastlmm_assoc_packed_f32. Rebuild/install JanusX extension first."
        )

    prefix = _as_plink_prefix(genofile)
    if prefix is None:
        cache_guess = _gwas_cache_prefix_with_params(
            genofile,
            maf=float(maf_threshold),
            geno=float(max_missing_rate),
            snps_only=bool(snps_only),
        )
        prefix = _as_plink_prefix(cache_guess)
    if prefix is None:
        raise ValueError(
            "LRLMM requires PLINK BED input (prefix with .bed/.bim/.fam). "
            f"Resolved genotype is not PLINK: {genofile}"
        )

    if genetic_model.lower() != "add":
        raise ValueError("LRLMM currently supports additive coding only (--model add).")

    use_prepared = isinstance(prepared_cache, dict)
    # Keep live progress feedback for long RSVD/GWAS stages (including prepared/full mode).
    show_internal_status = bool(use_spinner)
    if use_prepared:
        pheno_obj = prepared_cache.get("pheno")
        if not isinstance(pheno_obj, pd.DataFrame):
            raise ValueError("prepared_cache['pheno'] must be a pandas DataFrame.")
        pheno = pheno_obj

        famid_obj = prepared_cache.get("famid")
        if famid_obj is None:
            raise ValueError("prepared_cache['famid'] is missing.")
        ids = np.asarray(famid_obj, dtype=str)
        sample_map = np.arange(ids.shape[0], dtype=np.int64)

        q_obj = prepared_cache.get("qmatrix")
        if q_obj is None:
            raise ValueError("prepared_cache['qmatrix'] is missing.")
        qmatrix = np.asarray(q_obj, dtype="float32")
        cov_all = None

        packed_obj = prepared_cache.get("packed_ctx")
        if not isinstance(packed_obj, dict):
            raise ValueError(
                "prepared_cache['packed_ctx'] is missing. "
                "LowRankLMM requires packed BED genotype."
            )
        packed = np.ascontiguousarray(np.asarray(packed_obj["packed"], dtype=np.uint8))
        miss = np.ascontiguousarray(
            np.asarray(packed_obj["missing_rate"], dtype=np.float32).reshape(-1)
        )
        maf = np.ascontiguousarray(
            np.asarray(packed_obj["maf"], dtype=np.float32).reshape(-1)
        )
        packed_n = int(packed_obj["n_samples"])
        if packed_n != int(ids.shape[0]):
            raise ValueError(
                f"prepared packed n_samples={packed_n} does not match famid={ids.shape[0]}"
            )
        if packed.shape[0] != maf.shape[0] or packed.shape[0] != miss.shape[0]:
            raise ValueError("Prepared packed BED arrays have inconsistent SNP dimensions.")

        ref_alt_obj = prepared_cache.get("ref_alt")
        if isinstance(ref_alt_obj, pd.DataFrame) and all(
            c in ref_alt_obj.columns for c in ("chrom", "pos", "allele0", "allele1")
        ):
            ref_alt_df = ref_alt_obj.reset_index(drop=True)
            if int(ref_alt_df.shape[0]) != int(packed.shape[0]):
                raise ValueError(
                    "prepared ref_alt row count does not match packed SNP count."
                )
            sites_all = list(
                zip(
                    ref_alt_df["chrom"].astype(str).tolist(),
                    pd.to_numeric(ref_alt_df["pos"], errors="coerce").fillna(0).astype(int).tolist(),
                    ref_alt_df["allele0"].astype(str).tolist(),
                    ref_alt_df["allele1"].astype(str).tolist(),
                )
            )
        else:
            sites_all = _read_bim_sites(prefix)
            if len(sites_all) != int(packed.shape[0]):
                raise ValueError(
                    f"BIM row count mismatch with packed BED: bim={len(sites_all)}, packed={packed.shape[0]}"
                )
    else:
        full_ids, _ = inspect_genotype_file(
            prefix,
            snps_only=False,
            maf=0.0,
            missing_rate=1.0,
            het=0.0,
        )
        full_ids = np.asarray(full_ids, dtype=str)
        id_to_idx = {sid: i for i, sid in enumerate(full_ids)}
        try:
            sample_map = np.asarray(
                [id_to_idx[str(sid)] for sid in np.asarray(ids, dtype=str)],
                dtype=np.int64,
            )
        except KeyError as e:
            raise ValueError(
                "Some aligned sample IDs are not present in packed BED sample order."
            ) from e

        packed_t0 = time.monotonic()
        with CliStatus("Loading packed BED genotype...", enabled=bool(use_spinner)) as task:
            try:
                packed_raw, miss_raw, maf_raw, _std_raw, packed_n = load_bed_2bit_packed(prefix)
            except Exception:
                task.fail("Loading packed BED genotype ...Failed")
                raise
            task.complete("Loading packed BED genotype ...Finished")
        packed_t = max(time.monotonic() - packed_t0, 0.0)
        _log_model_line(
            logger,
            "LowRankLMM",
            f"Loaded packed BED [{format_elapsed(packed_t)}]",
            use_spinner=bool(use_spinner),
        )

        packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8))
        miss = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1))
        maf = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1))
        packed_n = int(packed_n)
        if packed_n <= 0:
            raise ValueError("Packed BED reported invalid sample count.")
        if packed.shape[0] != maf.shape[0] or packed.shape[0] != miss.shape[0]:
            raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")

        sites_all = _read_bim_sites(prefix)
        if len(sites_all) != int(packed.shape[0]):
            raise ValueError(
                f"BIM row count mismatch with packed BED: bim={len(sites_all)}, packed={packed.shape[0]}"
            )

        keep = (maf >= float(maf_threshold)) & (miss <= float(max_missing_rate))
        if bool(snps_only):
            snp_mask = np.asarray(
                [
                    (len(str(a0)) == 1 and len(str(a1)) == 1)
                    for (_c, _p, a0, a1) in sites_all
                ],
                dtype=bool,
            )
            keep &= snp_mask
        if not np.any(keep):
            raise ValueError("No SNP remains after LRLMM packed-bed filtering.")
        if not np.all(keep):
            packed = np.ascontiguousarray(packed[keep], dtype=np.uint8)
            miss = np.ascontiguousarray(miss[keep], dtype=np.float32)
            maf = np.ascontiguousarray(maf[keep], dtype=np.float32)
            sites_all = [s for s, k in zip(sites_all, keep) if bool(k)]

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    trait_iter = list(pheno.columns) if trait_names is None else [t for t in trait_names if t in pheno.columns]
    multi_trait_mode = len(trait_iter) > 1

    for trait_idx, pname in enumerate(trait_iter):
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        pheno_sub = pheno[pname].dropna()
        sameidx = np.isin(ids, pheno_sub.index)
        n_idv = int(np.sum(sameidx))
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")
            continue

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                str(pname),
                int(n_idv),
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        trait_ids = np.asarray(ids[sameidx], dtype=str)
        y_vec = np.ascontiguousarray(pheno_sub.loc[trait_ids].values, dtype=np.float64)
        x_cov = qmatrix[sameidx]
        if cov_all is not None:
            x_cov = np.concatenate([x_cov, cov_all[sameidx]], axis=1)
        x_cov = np.ascontiguousarray(x_cov, dtype=np.float64)
        x_arg = x_cov if int(x_cov.shape[1]) > 0 else None
        sample_idx_trait = np.ascontiguousarray(sample_map[sameidx], dtype=np.int64)

        rank_trait = _resolve_lrlmm_rank(rank_opt, int(n_idv))
        rsvd_t0 = time.monotonic()
        rsvd_threads = _status_stage_thread_budget(int(threads))
        if hasattr(jxrs, "admx_set_threads"):
            try:
                jxrs.admx_set_threads(int(rsvd_threads))
            except Exception:
                pass
        with CliStatus(
            f"Running LowRankLMM Random-SVD (rank={rank_trait})...",
            enabled=show_internal_status,
            use_process=True,
        ) as task:
            try:
                s_raw, u_sub_raw, maf_trait_raw, row_flip_raw = jxrs.rsvd_packed_subset(
                    packed,
                    int(packed_n),
                    int(rank_trait),
                    sample_idx_trait,
                    42,
                    5,
                    1e-1,
                )
            except Exception:
                task.fail("LowRankLMM Random-SVD ...Failed")
                raise
        rsvd_secs = max(time.monotonic() - rsvd_t0, 0.0)

        s_trait = np.ascontiguousarray(np.asarray(s_raw, dtype=np.float32).reshape(-1))
        u_sub = np.ascontiguousarray(np.asarray(u_sub_raw, dtype=np.float32))
        maf_trait = np.ascontiguousarray(np.asarray(maf_trait_raw, dtype=np.float32).reshape(-1))
        row_flip_trait = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1)
        )
        if u_sub.ndim != 2 or u_sub.shape[0] != int(n_idv) or u_sub.shape[1] != int(s_trait.shape[0]):
            raise ValueError(
                f"Trait RSVD output shape mismatch: u={u_sub.shape}, s={s_trait.shape}, n={n_idv}"
            )
        if maf_trait.shape[0] != packed.shape[0] or row_flip_trait.shape[0] != packed.shape[0]:
            raise ValueError(
                "Trait RSVD row statistics length mismatch with packed SNP rows."
            )
        s_trait, tau_trait, trace_scale, raw_trace, target_trace = _calibrate_lrlmm_spectrum(
            s_trait,
            int(n_idv),
        )
        _log_file_only(
            logger,
            logging.INFO,
            (
                f"LowRankLMM spectrum calibration ({pname}): "
                f"trace_raw={raw_trace:.6g}, trace_target={target_trace:.6g}, "
                f"tau={tau_trait:.6g}, scale={trace_scale:.6g}"
            ),
        )
        u_trait = np.zeros((int(packed_n), int(s_trait.shape[0])), dtype=np.float32)
        u_trait[sample_idx_trait, :] = u_sub
        _log_model_line(
            logger,
            "LowRankLMM",
            f"RSVD rank={int(s_trait.shape[0])} [{format_elapsed(rsvd_secs)}]",
            use_spinner=bool(use_spinner),
        )

        gwas_t0 = time.monotonic()
        gwas_total = int(len(sites_all))
        gwas_last_done = 0
        gwas_pbar: Optional[_ProgressAdapter] = None
        if show_internal_status:
            gwas_pbar = _ProgressAdapter(
                total=max(1, gwas_total),
                desc="LrLMM",
                force_animate=True,
            )

        def _lrlmm_progress(done: int, total: int) -> None:
            nonlocal gwas_last_done, gwas_total, gwas_pbar
            if gwas_pbar is None:
                return
            try:
                d = int(done)
                t = int(total)
            except Exception:
                return
            if t > 0 and t != gwas_total and gwas_last_done == 0:
                try:
                    gwas_pbar.close(show_done=False)
                except Exception:
                    pass
                gwas_total = int(max(1, t))
                gwas_pbar = _ProgressAdapter(
                    total=gwas_total,
                    desc="LrLMM",
                    force_animate=True,
                )
                gwas_last_done = 0
            d = int(max(0, min(d, max(1, gwas_total))))
            step = int(max(0, d - gwas_last_done))
            if step > 0 and gwas_pbar is not None:
                gwas_pbar.update(step)
                gwas_last_done = d

        gwas_ok = False
        try:
            progress_kwargs: dict[str, object] = {}
            if gwas_pbar is not None:
                progress_kwargs = {
                    "progress_callback": _lrlmm_progress,
                    "progress_every": int(max(1, int(chunk_size))),
                }
            try:
                lbd, _ml0, _reml0, res_raw = jxrs.fastlmm_assoc_packed_f32(
                    packed,
                    int(packed_n),
                    row_flip_trait,
                    maf_trait,
                    u_trait,
                    s_trait,
                    y_vec,
                    x_arg,
                    sample_idx_trait,
                    -5.0,
                    5.0,
                    50,
                    1e-2,
                    float(tau_trait),
                    int(threads),
                    "add",
                    **progress_kwargs,
                )
            except TypeError as e:
                # Backward compatibility with older compiled extensions
                # that do not expose `progress_*` and/or `tau`.
                emsg = str(e).lower()
                if ("argument" not in emsg) and ("positional" not in emsg) and ("keyword" not in emsg):
                    raise
                try:
                    lbd, _ml0, _reml0, res_raw = jxrs.fastlmm_assoc_packed_f32(
                        packed,
                        int(packed_n),
                        row_flip_trait,
                        maf_trait,
                        u_trait,
                        s_trait,
                        y_vec,
                        x_arg,
                        sample_idx_trait,
                        -5.0,
                        5.0,
                        50,
                        1e-2,
                        float(tau_trait),
                        int(threads),
                        "add",
                    )
                except TypeError as e2:
                    emsg2 = str(e2).lower()
                    if ("argument" not in emsg2) and ("positional" not in emsg2) and ("keyword" not in emsg2):
                        raise
                    lbd, _ml0, _reml0, res_raw = jxrs.fastlmm_assoc_packed_f32(
                        packed,
                        int(packed_n),
                        row_flip_trait,
                        maf_trait,
                        u_trait,
                        s_trait,
                        y_vec,
                        x_arg,
                        sample_idx_trait,
                        -5.0,
                        5.0,
                        50,
                        1e-2,
                        int(threads),
                        "add",
                    )
            gwas_ok = True
        finally:
            if gwas_pbar is not None:
                try:
                    if gwas_ok:
                        if int(gwas_last_done) < int(max(1, gwas_total)):
                            gwas_pbar.update(int(max(1, gwas_total)) - int(gwas_last_done))
                        gwas_pbar.finish()
                        time.sleep(0.05)
                except Exception:
                    pass
                gwas_pbar.close(show_done=False)
        gwas_secs = max(time.monotonic() - gwas_t0, 0.0)

        res = np.ascontiguousarray(np.asarray(res_raw, dtype=np.float64))
        if res.ndim != 2 or res.shape[0] != len(sites_all) or res.shape[1] < 3:
            raise ValueError(
                f"Unexpected LowRankLMM result shape: {res.shape}, expected ({len(sites_all)}, >=3)"
            )

        pve = None
        try:
            vg = float(
                (np.sum(np.asarray(s_trait, dtype=np.float64)) + float(tau_trait) * float(n_idv))
                / float(max(1, n_idv))
            )
            lbd_v = float(lbd)
            if np.isfinite(vg) and np.isfinite(lbd_v) and (vg + lbd_v) > 0:
                pve = vg / (vg + lbd_v)
        except Exception:
            pve = None

        _log_model_line(
            logger,
            "LowRankLMM",
            f"rank={int(s_trait.shape[0])}; lambda={float(lbd):.4g}",
            use_spinner=bool(use_spinner),
        )

        chrom = [str(c) for (c, _p, _a0, _a1) in sites_all]
        pos = [int(p) for (_c, p, _a0, _a1) in sites_all]
        a0 = [str(v) for (_c, _p, v, _a1) in sites_all]
        a1 = [str(v) for (_c, _p, _a0, v) in sites_all]

        res_df = pd.DataFrame(
            {
                "chrom": chrom,
                "pos": pos,
                "allele0": a0,
                "allele1": a1,
                "maf": np.asarray(maf_trait, dtype=np.float32),
                "beta": np.asarray(res[:, 0], dtype=np.float64),
                "se": np.asarray(res[:, 1], dtype=np.float64),
                "pwald": np.asarray(res[:, 2], dtype=np.float64),
            }
        )
        if res.shape[1] > 3:
            res_df["plrt"] = np.asarray(res[:, 3], dtype=np.float64)

        gm_tag = str(genetic_model).lower()
        if gm_tag == "add":
            out_tsv = f"{outprefix}.{pname}.lrlmm.tsv"
            out_svg = f"{outprefix}.{pname}.lrlmm.svg"
        else:
            out_tsv = f"{outprefix}.{pname}.{gm_tag}.lrlmm.tsv"
            out_svg = f"{outprefix}.{pname}.{gm_tag}.lrlmm.svg"
        viz_secs = 0.0
        if plot:
            viz_t0 = time.time()
            fastplot(
                res_df[["chrom", "pos", "pwald"]],
                y_vec,
                xlabel=str(pname),
                outpdf=out_svg,
            )
            viz_secs = max(time.time() - viz_t0, 0.0)

        cast_map: dict[str, object] = {"pwald": "object", "pos": int}
        if "plrt" in res_df.columns:
            cast_map["plrt"] = "object"
        res_df = res_df.astype(cast_map)
        res_df.loc[:, "pwald"] = res_df["pwald"].map(lambda x: f"{x:.4e}")
        if "plrt" in res_df.columns:
            res_df.loc[:, "plrt"] = res_df["plrt"].map(lambda x: f"{x:.4e}")
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        saved_paths.append(str(out_tsv).replace("//", "/"))

        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        t1 = time.time()
        wall = max(t1 - t0, 1e-12)
        cpu_used = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)
        _log_model_line(
            logger,
            "LowRankLMM",
            f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )
        _log_model_line(
            logger,
            "LowRankLMM",
            f"Results saved to {str(out_tsv).replace('//', '/')}",
            use_spinner=bool(use_spinner),
        )

        eff_snp = int(len(sites_all))
        eff_snp_by_trait[pname] = eff_snp
        compute_secs = rsvd_secs + gwas_secs
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": "LrLMM",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": (float(pve) if pve is not None else None),
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(compute_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv).replace("//", "/"),
            }
        )

        done_times = [format_elapsed(rsvd_secs), format_elapsed(gwas_secs)]
        if plot:
            done_times.append(format_elapsed(viz_secs))
        _rich_success(
            logger,
            f"LowRankLMM ...Finished [{'/'.join(done_times)}]",
            use_spinner=bool(use_spinner),
        )
        if pve is not None and np.isfinite(float(pve)):
            _emit_plain_info_line(
                logger,
                f"pve={float(pve):.3f} (from LrLMM)",
                use_spinner=bool(use_spinner),
            )
        if multi_trait_mode:
            logger.info("")


def run_chunked_gwas_lmm_lm(
    model_name: str,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    mmap_limit: bool,
    grm: Union[np.ndarray, None],
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray , None],
    eff_m: int,
    plot: bool,
    threads: int,
    logger:logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    show_npve_line: bool = False,
    emit_trait_header: bool = True,
) -> None:
    """
    Run LMM/FastLMM/LM GWAS using a streaming pipeline.

    Important: This function assumes pheno/ids/grm/q/cov have already been prepared
    once (no repeated "Loading phenotype" / "Loading GRM/Q" logs).
    """
    model_map = {
        "lmm": LMM,
        "lm": LM,
        "fastlmm": FastLMM,
    }
    model_key = model_name.lower()
    ModelCls = model_map[model_key]
    base_model_label = {
        "lmm": "LMM",
        "lm": "LM",
        "fastlmm": "FastLMM",
    }[model_key]
    # Keep output file suffixes consistent and lowercase.
    base_model_tag = base_model_label.lower()

    def _apply_genetic_model(geno_chunk: np.ndarray, model: str) -> np.ndarray:
        m = model.lower()
        if m == "add":
            return geno_chunk
        if m == "dom":
            return (
                np.isclose(geno_chunk, 1.0, atol=1e-6)
                | np.isclose(geno_chunk, 2.0, atol=1e-6)
            ).astype(np.float32, copy=False)
        if m == "rec":
            return np.isclose(geno_chunk, 2.0, atol=1e-6).astype(np.float32, copy=False)
        if m == "het":
            return np.isclose(geno_chunk, 1.0, atol=1e-6).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported genetic model: {model}")

    def _transform_allele_labels(
        allele0_list: list[str], allele1_list: list[str], model: str
    ) -> tuple[list[str], list[str]]:
        m = model.lower()
        if m == "add":
            return allele0_list, allele1_list
        out0: list[str] = []
        out1: list[str] = []
        for a0, a1 in zip(allele0_list, allele1_list):
            hom0 = f"{a0}{a0}"
            het = f"{a0}{a1}"
            hom1 = f"{a1}{a1}"
            if m == "dom":
                out0.append(hom0)
                out1.append(f"{het}/{hom1}")
            elif m == "rec":
                out0.append(f"{het}/{hom0}")
                out1.append(hom1)
            elif m == "het":
                out0.append(f"{hom0}/{hom1}")
                out1.append(het)
            else:
                raise ValueError(f"Unsupported genetic model: {model}")
        return out0, out1

    def _heter_keep_mask(geno_chunk: np.ndarray, het: float) -> np.ndarray:
        valid = geno_chunk >= 0
        non_missing = np.sum(valid, axis=1)
        keep = non_missing > 0
        if not np.any(keep):
            return keep
        het_count = np.sum(np.isclose(geno_chunk, 1.0, atol=1e-6) & valid, axis=1)
        het_rate = np.zeros(geno_chunk.shape[0], dtype=np.float32)
        idx = non_missing > 0
        het_rate[idx] = het_count[idx] / non_missing[idx]
        keep &= (het_rate >= het) & (het_rate <= (1.0 - het))
        return keep

    process = psutil.Process()
    n_cores = detect_effective_threads()

    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    trait_iter = list(pheno.columns) if trait_names is None else [t for t in trait_names if t in pheno.columns]
    multi_trait_mode = len(trait_iter) > 1
    for trait_idx, pname in enumerate(trait_iter):
        cpu_t0 = process.cpu_times()
        rss0 = process.memory_info().rss
        t0 = time.time()
        peak_rss = rss0
        evd_secs = 0.0

        pheno_sub = pheno[pname].dropna()
        sameidx = np.isin(ids, pheno_sub.index)
        n_idv = int(np.sum(sameidx))
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = 0
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                pname,
                n_idv,
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        trait_ids = np.asarray(ids[sameidx], dtype=str)
        y_vec = pheno_sub.loc[trait_ids].values
        # Build covariate matrix X_cov for this trait
        X_cov = qmatrix[sameidx]
        if cov_all is not None:
            X_cov = np.concatenate([X_cov, cov_all[sameidx]], axis=1)

        model_chunk_size = _resolve_stream_scan_chunk_size(
            int(chunk_size),
            int(n_snps),
            use_spinner=bool(use_spinner),
        )

        header_pve: Optional[float] = None
        init_log_message: Optional[str] = None
        effective_model_key = str(model_key)
        effective_model_label = str(base_model_label)
        effective_model_tag = str(base_model_tag)
        if model_key in ("lmm", "fastlmm"):
            if grm is None:
                raise ValueError("LMM/fastLMM requires GRM, but GRM was not prepared.")
            Ksub = grm[np.ix_(sameidx, sameidx)]
            evd_t0 = time.monotonic()
            evd_label = "LMM" if model_key == "lmm" else "FastLMM"
            evd_desc = f"{evd_label} Eigen-Decomposition"
            with CliStatus(
                f"Running {evd_desc}...",
                enabled=bool(use_spinner),
                use_process=True,
            ) as task:
                try:
                    stage_threads = _status_stage_thread_budget(int(threads))
                    with _native_thread_limit_ctx(stage_threads):
                        mod = ModelCls(y=y_vec, X=X_cov, kinship=Ksub)
                except Exception:
                    task.fail(f"{evd_desc} ...Failed")
                    raise
            try:
                pve_tmp = float(mod.pve)
                if np.isfinite(pve_tmp):
                    header_pve = pve_tmp
            except Exception:
                header_pve = None
            evd_secs = time.monotonic() - evd_t0
            evd_elapsed = format_elapsed(evd_secs)
            if model_key == "fastlmm" and _fastlmm_pve_is_degenerate(header_pve):
                logger.warning(
                    f"Warning: FastLMM fallback to LMM for trait {pname}: "
                    f"PVE(null)={float(header_pve):.3f} outside "
                    f"[{_FASTLMM_PVE_LOW:.2f}, {_FASTLMM_PVE_HIGH:.2f}]."
                )
                effective_model_key = "lmm"
                effective_model_label = "LMM"
                effective_model_tag = "lmm"
            init_log_message = f"PVE(null) ~ {mod.pve:.3f}; eigen-decomposition [{evd_elapsed}]"
        else:
            mod = ModelCls(y=y_vec, X=X_cov)
            init_log_message = "streaming scan initialized"
        if init_log_message is not None:
            _log_model_line(
                logger,
                effective_model_label,
                init_log_message,
                use_spinner=bool(use_spinner),
            )

        done_snps = 0
        has_results = False
        out_tsv = f"{outprefix}.{pname}.{genetic_model}.{effective_model_tag}.tsv"
        tmp_tsv = f"{out_tsv}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        wrote_header = False
        mmap_window_mb = (
            auto_mmap_window_mb(genofile, len(ids), n_snps, model_chunk_size)
            if mmap_limit else None
        )

        # Always pass trait-specific sample IDs to the reader to keep column
        # dimension consistent across BED/VCF/TXT backends.
        sample_sub = trait_ids
        expected_n = int(sample_sub.shape[0])
        scan_threads = int(threads)
        if scan_threads <= 0:
            scan_threads = int(n_cores)
        if effective_model_key == "fastlmm":
            # FastLMM is often decode/write bound; prefer one in-flight chunk
            # so the kernel can use all configured threads.
            max_inflight = 1
            workers = 1
            threads_per_worker = max(1, scan_threads)
            prefetch_depth = 2 if scan_threads >= 2 else 1
        else:
            max_inflight = 2 if scan_threads >= 2 else 1
            workers = max(1, max_inflight)
            threads_per_worker = max(1, scan_threads // workers)
            prefetch_depth = 1

        process.cpu_percent(interval=None)
        scan_t0 = time.time()
        pbar_total = int(eff_snp_by_trait.get(pname, n_snps))
        pbar_desc = f"{effective_model_label}"
        pbar: Optional[_ProgressAdapter] = None
        scan_warmup_task: Optional[CliStatus] = None
        scan_warmup_active = False
        if bool(use_spinner):
            scan_warmup_task = CliStatus(
                "Waiting for LMM GWAS",
                enabled=True,
                use_process=True,
            )
            scan_warmup_task.__enter__()
            scan_warmup_active = True

        inflight: dict[
            int,
            tuple[cf.Future, int, list[tuple[str, int, str, str]], np.ndarray],
        ] = {}
        chunk_seq = 0
        interrupted = False

        class _ChunkBatchWriter:
            """
            Lightweight batch writer for GWAS TSV rows.
            Buffers multiple chunks and flushes in one file append call.
            """

            def __init__(self, path: str, *, batch_chunks: int = 8) -> None:
                self.path = str(path)
                self.batch_chunks = max(1, int(batch_chunks))
                self._parts: list[str] = []
                self._has_plrt: Optional[bool] = None
                self._wrote_header = False
                self.rows_written = 0

            def append(self, text: str, *, has_plrt: bool, rows: int) -> None:
                n_rows = int(rows)
                if n_rows <= 0:
                    return
                hp = bool(has_plrt)
                if self._has_plrt is None:
                    self._has_plrt = hp
                elif self._has_plrt != hp:
                    raise ValueError(
                        "Inconsistent result columns across chunks while writing GWAS TSV."
                    )
                self._parts.append(str(text))
                self.rows_written += n_rows
                if len(self._parts) >= self.batch_chunks:
                    self.flush()

            def flush(self) -> None:
                if len(self._parts) == 0:
                    return
                mode = "a" if self._wrote_header else "w"
                with open(self.path, mode, encoding="utf-8", newline="") as fh:
                    if not self._wrote_header:
                        header = "chrom\tpos\tallele0\tallele1\tmaf\tbeta\tse\tpwald"
                        if bool(self._has_plrt):
                            header += "\tplrt"
                        fh.write(header + "\n")
                        self._wrote_header = True
                    fh.write("".join(self._parts))
                self._parts.clear()

            @property
            def wrote_header(self) -> bool:
                return bool(self._wrote_header)

        writer = _ChunkBatchWriter(str(tmp_tsv), batch_chunks=8)

        def _format_chunk_tsv_text(
            results: np.ndarray,
            info_chunk: list[tuple[str, int, str, str]],
            maf_chunk: np.ndarray,
        ) -> tuple[str, bool, int]:
            if len(info_chunk) == 0:
                return "", bool(np.asarray(results).shape[1] > 3), 0

            chroms, poss, allele0, allele1 = zip(*info_chunk)
            allele0_list = list(allele0)
            allele1_list = list(allele1)
            if genetic_model != "add":
                allele0_list, allele1_list = _transform_allele_labels(
                    allele0_list, allele1_list, genetic_model
                )
            res = np.asarray(results, dtype=np.float64)
            cols = [
                np.asarray(chroms, dtype=object),
                np.asarray(poss, dtype=np.int64).astype(str),
                np.asarray(allele0_list, dtype=object),
                np.asarray(allele1_list, dtype=object),
                np.char.mod("%.4f", np.asarray(maf_chunk, dtype=np.float64)),
                np.char.mod("%.4f", res[:, 0]),
                np.char.mod("%.4f", res[:, 1]),
                np.char.mod("%.4e", res[:, 2]),
            ]
            has_plrt = bool(res.shape[1] > 3)
            if has_plrt:
                cols.append(np.char.mod("%.4e", res[:, 3]))
            rows = np.column_stack(cols)
            text = "\n".join("\t".join(map(str, row)) for row in rows)
            if text:
                text += "\n"
            return text, has_plrt, int(rows.shape[0])

        def _drain_completed(*, wait_for_one: bool) -> None:
            nonlocal done_snps, peak_rss, pbar, scan_warmup_active, scan_warmup_task
            if len(inflight) == 0:
                return
            futures = [x[0] for x in inflight.values()]
            if wait_for_one:
                done_set, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            else:
                done_set = {f for f in futures if f.done()}
                if len(done_set) == 0:
                    return

            done_seq = sorted(
                [
                    seq
                    for seq, (fut, _m, _info, _maf) in inflight.items()
                    if fut in done_set
                ]
            )
            for seq in done_seq:
                fut, m_chunk, info_chunk, maf_chunk = inflight.pop(seq)
                results = fut.result()
                chunk_text, has_plrt, n_rows = _format_chunk_tsv_text(
                    results,
                    info_chunk,
                    maf_chunk,
                )
                writer.append(chunk_text, has_plrt=has_plrt, rows=n_rows)
                done_snps += int(m_chunk)
                if pbar is None:
                    if scan_warmup_active and scan_warmup_task is not None:
                        try:
                            scan_warmup_task.__exit__(None, None, None)
                        except Exception:
                            pass
                        scan_warmup_active = False
                    pbar = _ProgressAdapter(
                        total=pbar_total,
                        desc=pbar_desc,
                        force_animate=True,
                    )
                pbar.update(int(m_chunk))

                mem_info = process.memory_info()
                peak_rss = max(peak_rss, mem_info.rss)
                if done_snps % (10 * model_chunk_size) == 0:
                    mem_gb = mem_info.rss / 1024**3
                    if pbar is not None:
                        pbar.set_postfix(memory=f"{mem_gb:.2f}GB")

        ex = cf.ThreadPoolExecutor(max_workers=workers)
        try:
            chunk_iter = load_genotype_chunks(
                genofile,
                model_chunk_size,
                maf_threshold,
                max_missing_rate,
                model=genetic_model,
                het=het_threshold,
                snps_only=bool(snps_only),
                sample_ids=sample_sub,
                mmap_window_mb=mmap_window_mb,
            )
            for genosub, sites in prefetch_iter(chunk_iter, in_flight=prefetch_depth):
                genosub: np.ndarray
                if genosub.shape[1] != expected_n:
                    # Backward-compatible fallback: if backend ignored sample_ids and
                    # returned columns for full aligned IDs, apply sameidx here.
                    if genosub.shape[1] == int(sameidx.shape[0]):
                        genosub = genosub[:, sameidx]
                    else:
                        raise ValueError(
                            f"Genotype sample dimension mismatch for trait {pname}: "
                            f"chunk has {genosub.shape[1]} columns, "
                            f"expected {expected_n} (or {sameidx.shape[0]} full aligned IDs)."
                        )
                if genetic_model != "add":
                    keep_mask = _heter_keep_mask(genosub, het_threshold)
                    if not np.any(keep_mask):
                        continue
                    genosub = genosub[keep_mask]
                    sites = [s for s, k in zip(sites, keep_mask) if k]
                m_chunk = genosub.shape[0]
                if m_chunk == 0:
                    continue

                info_chunk = [
                    (str(s.chrom), int(s.pos), str(s.ref_allele), str(s.alt_allele))
                    for s in sites
                ]
                if len(info_chunk) == 0:
                    continue

                geno_model = _apply_genetic_model(genosub, genetic_model)
                if genetic_model == "add":
                    maf_chunk = np.mean(genosub, axis=1)
                    maf_chunk = (maf_chunk / 2).astype(np.float32, copy=False)
                else:
                    maf_chunk = np.mean(geno_model, axis=1).astype(np.float32, copy=False)
                # In-place centering avoids one extra (m_chunk, n_samples) allocation.
                geno_center = np.asarray(geno_model, dtype=np.float32, order="C")
                geno_center -= np.mean(
                    geno_center, axis=1, dtype=np.float32, keepdims=True
                )
                fut = ex.submit(mod.gwas, geno_center, threads=threads_per_worker)
                inflight[chunk_seq] = (fut, int(m_chunk), info_chunk, maf_chunk)
                chunk_seq += 1

                if len(inflight) >= max_inflight:
                    _drain_completed(wait_for_one=True)
                else:
                    _drain_completed(wait_for_one=False)

            while len(inflight) > 0:
                _drain_completed(wait_for_one=True)
            writer.flush()
            wrote_header = bool(writer.wrote_header)
            has_results = int(writer.rows_written) > 0
            if pbar is not None:
                pbar.finish()
        except KeyboardInterrupt:
            interrupted = True
            for fut, _m, _info, _maf in inflight.values():
                try:
                    fut.cancel()
                except Exception:
                    pass
            raise
        finally:
            # Always stop renderer first, then tear down workers.
            if scan_warmup_active and scan_warmup_task is not None:
                try:
                    scan_warmup_task.__exit__(None, None, None)
                except Exception:
                    pass
                scan_warmup_active = False
            if pbar is not None:
                pbar.close(show_done=False)
            ex.shutdown(wait=False, cancel_futures=True)
            if interrupted and os.path.exists(tmp_tsv):
                try:
                    os.remove(tmp_tsv)
                except Exception:
                    pass

        cpu_t1 = process.cpu_times()
        t1 = time.time()
        scan_secs = max(t1 - scan_t0, 0.0)

        wall = t1 - t0
        user_cpu = cpu_t1.user - cpu_t0.user
        sys_cpu = cpu_t1.system - cpu_t0.system
        total_cpu = user_cpu + sys_cpu

        avg_cpu_pct = 100.0 * total_cpu / wall / (n_cores or 1) if wall > 0 else 0.0
        peak_rss_gb = peak_rss / 1024**3
        _log_model_line(
            logger,
            effective_model_label,
            f"avg CPU ~ {avg_cpu_pct:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )

        if not has_results:
            logger.info(f"{effective_model_label}: no SNPs passed filters for trait {pname}.")
            if pname not in eff_snp_by_trait:
                eff_snp_by_trait[pname] = int(done_snps)
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": effective_model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(header_pve) if header_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": 0.0,
                    "result_file": "",
                }
            )
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        os.replace(tmp_tsv, out_tsv)
        saved_paths.append(str(out_tsv).replace("//", "/"))
        _log_model_line(
            logger,
            effective_model_label,
            f"Results saved to {str(out_tsv).replace('//', '/')}",
            use_spinner=bool(use_spinner),
        )
        viz_secs = 0.0
        if plot:
            viz_t0 = time.time()
            plot_df = pd.read_csv(
                out_tsv,
                sep="\t",
                usecols=["chrom", "pos", "pwald"],
                dtype={"chrom": str, "pos": "int64"},
            )
            plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
            fastplot(
                plot_df,
                y_vec,
                xlabel=str(pname),
                outpdf=f"{outprefix}.{pname}.{genetic_model}.{effective_model_tag}.svg",
            )
            viz_secs = max(time.time() - viz_t0, 0.0)
        if pname not in eff_snp_by_trait:
            eff_snp_by_trait[pname] = int(done_snps)
        summary_rows.append(
            {
                "phenotype": str(pname),
                "model": effective_model_label,
                "nidv": int(n_idv),
                "eff_snp": int(done_snps),
                "pve": (float(header_pve) if header_pve is not None else None),
                "avg_cpu": float(avg_cpu_pct),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(evd_secs + scan_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv).replace("//", "/"),
            }
        )
        time_parts: list[str] = []
        if evd_secs > 0:
            time_parts.append(format_elapsed(evd_secs))
        time_parts.append(format_elapsed(scan_secs))
        if plot:
            time_parts.append(format_elapsed(viz_secs))
        done_msg = f"{effective_model_label} ...Finished [{'/'.join(time_parts)}]"
        _rich_success(logger, done_msg, use_spinner=use_spinner)
        if header_pve is not None and np.isfinite(float(header_pve)):
            _emit_plain_info_line(
                logger,
                f"pve={float(header_pve):.3f} (from {effective_model_label})",
                use_spinner=bool(use_spinner),
            )
        if multi_trait_mode:
            logger.info("")  # ensure blank line between traits


def run_chunked_gwas_streaming_shared(
    model_names: list[str],
    trait_name: str,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    genetic_model: str,
    het_threshold: float,
    chunk_size: int,
    mmap_limit: bool,
    grm: Union[np.ndarray, None],
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray, None],
    plot: bool,
    threads: int,
    logger: logging.Logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    eff_snp_by_trait: Union[dict[str, int], None] = None,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
) -> None:
    """
    Shared-chunk streaming GWAS for multiple models on one trait.

    Decode/filter each chunk once, then run all selected streaming models on the
    same chunk before moving to the next chunk.
    """
    model_order = [str(m).lower() for m in model_names]
    model_map = {"lmm": LMM, "lm": LM, "fastlmm": FastLMM}
    model_order = [m for m in model_order if m in model_map]
    if len(model_order) == 0:
        return

    process = psutil.Process()
    n_cores = detect_effective_threads()

    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pname = str(trait_name)
    pheno_sub = pheno[pname].dropna()
    sameidx = np.isin(ids, pheno_sub.index)
    n_idv = int(np.sum(sameidx))
    if n_idv == 0:
        logger.warning(f"{pname}: no overlapping samples, skipped.")
        if pname not in eff_snp_by_trait:
            eff_snp_by_trait[pname] = 0
        return

    trait_ids = np.asarray(ids[sameidx], dtype=str)
    y_vec = pheno_sub.loc[trait_ids].values
    X_cov = qmatrix[sameidx]
    if cov_all is not None:
        X_cov = np.concatenate([X_cov, cov_all[sameidx]], axis=1)
    _emit_trait_header(
        logger,
        pname,
        n_idv,
        pve=None,
        use_spinner=bool(use_spinner),
        width=60,
    )

    def _apply_genetic_model(geno_chunk: np.ndarray, model: str) -> np.ndarray:
        m = model.lower()
        if m == "add":
            return geno_chunk
        if m == "dom":
            return (
                np.isclose(geno_chunk, 1.0, atol=1e-6)
                | np.isclose(geno_chunk, 2.0, atol=1e-6)
            ).astype(np.float32, copy=False)
        if m == "rec":
            return np.isclose(geno_chunk, 2.0, atol=1e-6).astype(np.float32, copy=False)
        if m == "het":
            return np.isclose(geno_chunk, 1.0, atol=1e-6).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported genetic model: {model}")

    def _transform_allele_labels(
        allele0_list: list[str], allele1_list: list[str], model: str
    ) -> tuple[list[str], list[str]]:
        m = model.lower()
        if m == "add":
            return allele0_list, allele1_list
        out0: list[str] = []
        out1: list[str] = []
        for a0, a1 in zip(allele0_list, allele1_list):
            hom0 = f"{a0}{a0}"
            het = f"{a0}{a1}"
            hom1 = f"{a1}{a1}"
            if m == "dom":
                out0.append(hom0)
                out1.append(f"{het}/{hom1}")
            elif m == "rec":
                out0.append(f"{het}/{hom0}")
                out1.append(hom1)
            elif m == "het":
                out0.append(f"{hom0}/{hom1}")
                out1.append(het)
            else:
                raise ValueError(f"Unsupported genetic model: {model}")
        return out0, out1

    def _heter_keep_mask(geno_chunk: np.ndarray, het: float) -> np.ndarray:
        valid = geno_chunk >= 0
        non_missing = np.sum(valid, axis=1)
        keep = non_missing > 0
        if not np.any(keep):
            return keep
        het_count = np.sum(np.isclose(geno_chunk, 1.0, atol=1e-6) & valid, axis=1)
        het_rate = np.zeros(geno_chunk.shape[0], dtype=np.float32)
        idx = non_missing > 0
        het_rate[idx] = het_count[idx] / non_missing[idx]
        keep &= (het_rate >= het) & (het_rate <= (1.0 - het))
        return keep

    class _ChunkBatchWriter:
        """
        Lightweight batch writer for GWAS TSV rows.
        Buffers multiple chunks and flushes in one file append call.
        """

        def __init__(self, path: str, *, batch_chunks: int = 8) -> None:
            self.path = str(path)
            self.batch_chunks = max(1, int(batch_chunks))
            self._parts: list[str] = []
            self._has_plrt: Optional[bool] = None
            self._wrote_header = False
            self.rows_written = 0

        def append(self, text: str, *, has_plrt: bool, rows: int) -> None:
            n_rows = int(rows)
            if n_rows <= 0:
                return
            hp = bool(has_plrt)
            if self._has_plrt is None:
                self._has_plrt = hp
            elif self._has_plrt != hp:
                raise ValueError(
                    "Inconsistent result columns across chunks while writing GWAS TSV."
                )
            self._parts.append(str(text))
            self.rows_written += n_rows
            if len(self._parts) >= self.batch_chunks:
                self.flush()

        def flush(self) -> None:
            if len(self._parts) == 0:
                return
            mode = "a" if self._wrote_header else "w"
            with open(self.path, mode, encoding="utf-8", newline="") as fh:
                if not self._wrote_header:
                    header = "chrom\tpos\tallele0\tallele1\tmaf\tbeta\tse\tpwald"
                    if bool(self._has_plrt):
                        header += "\tplrt"
                    fh.write(header + "\n")
                    self._wrote_header = True
                fh.write("".join(self._parts))
            self._parts.clear()

        @property
        def wrote_header(self) -> bool:
            return bool(self._wrote_header)

    def _format_chunk_tsv_text(
        results: np.ndarray,
        info_chunk: list[tuple[str, int, str, str]],
        maf_chunk: np.ndarray,
    ) -> tuple[str, bool]:
        if len(info_chunk) == 0:
            return "", bool(np.asarray(results).shape[1] > 3)

        chroms, poss, allele0, allele1 = zip(*info_chunk)
        allele0_list = list(allele0)
        allele1_list = list(allele1)
        if genetic_model != "add":
            allele0_list, allele1_list = _transform_allele_labels(
                allele0_list, allele1_list, genetic_model
            )

        res = np.asarray(results, dtype=np.float64)
        cols = [
            np.asarray(chroms, dtype=object),
            np.asarray(poss, dtype=np.int64).astype(str),
            np.asarray(allele0_list, dtype=object),
            np.asarray(allele1_list, dtype=object),
            np.char.mod("%.4f", np.asarray(maf_chunk, dtype=np.float64)),
            np.char.mod("%.4f", res[:, 0]),
            np.char.mod("%.4f", res[:, 1]),
            np.char.mod("%.4e", res[:, 2]),
        ]
        has_plrt = bool(res.shape[1] > 3)
        if has_plrt:
            cols.append(np.char.mod("%.4e", res[:, 3]))

        out = np.column_stack(cols)
        buf = io.StringIO()
        np.savetxt(buf, out, fmt="%s", delimiter="\t")
        return buf.getvalue(), has_plrt

    model_label_map = {"lmm": "LMM", "lm": "LM", "fastlmm": "FastLMM"}
    share_evd_lmm_fast = ("lmm" in model_order) and ("fastlmm" in model_order)
    shared_lmm_model: Optional[LMM] = None
    shared_ksub: Optional[np.ndarray] = None
    model_ctxs: list[dict[str, object]] = []
    for mkey in model_order:
        ModelCls = model_map[mkey]
        model_label = model_label_map[mkey]
        model_tag = model_label.lower()
        effective_model_key = str(mkey)
        effective_model_label = str(model_label)
        effective_model_tag = str(model_tag)
        ctx: dict[str, object] = {
            "model_key": mkey,
            "model_label": model_label,
            "model_tag": model_tag,
            "mod": None,
            "evd_secs": 0.0,
            "scan_secs": 0.0,
            "cpu_used": 0.0,
            "peak_rss": int(process.memory_info().rss),
            "done_snps": 0,
            "wrote_header": False,
            "has_results": False,
            "tick": 0,
            "memory_text": "",
            "memory_until_tick": 0,
            "pbar": None,
            "task_id": None,
            "tmp_tsv": f"{outprefix}.{pname}.{genetic_model}.{model_tag}.tsv.tmp.{os.getpid()}.{uuid.uuid4().hex}",
            "out_tsv": f"{outprefix}.{pname}.{genetic_model}.{model_tag}.tsv",
            "writer": None,
            "init_log": None,
        }
        ctx["writer"] = _ChunkBatchWriter(str(ctx["tmp_tsv"]), batch_chunks=8)

        cpu_before = process.cpu_times()
        init_t0 = time.monotonic()
        if mkey in {"lmm", "fastlmm"}:
            if grm is None:
                raise ValueError("LMM/fastLMM requires GRM, but GRM was not prepared.")
            if (
                mkey == "fastlmm"
                and share_evd_lmm_fast
                and shared_lmm_model is not None
            ):
                mod = FastLMM.from_lmm(shared_lmm_model)
                ctx["evd_secs"] = 0.0
                ctx["init_log"] = (
                    f"PVE(null) ~ {mod.pve:.3f}; reusing shared eigen-decomposition from LMM"
                )
            else:
                if shared_ksub is None:
                    shared_ksub = grm[np.ix_(sameidx, sameidx)]
                evd_desc = f"{model_label} Eigen-Decomposition"
                with CliStatus(
                    f"Running {evd_desc}...",
                    enabled=bool(use_spinner),
                    use_process=True,
                ) as task:
                    try:
                        stage_threads = _status_stage_thread_budget(int(threads))
                        with _native_thread_limit_ctx(stage_threads):
                            mod = ModelCls(y=y_vec, X=X_cov, kinship=shared_ksub)
                    except Exception:
                        task.fail(f"{evd_desc} ...Failed")
                        raise
                evd_secs = max(time.monotonic() - init_t0, 0.0)
                evd_elapsed = format_elapsed(evd_secs)
                ctx["evd_secs"] = float(evd_secs)
                ctx["init_log"] = f"PVE(null) ~ {mod.pve:.3f}; eigen-decomposition [{evd_elapsed}]"
                if mkey == "lmm" and share_evd_lmm_fast:
                    shared_lmm_model = mod
        else:
            mod = ModelCls(y=y_vec, X=X_cov)
            ctx["init_log"] = "streaming scan initialized"

        pve_now: Optional[float] = None
        if mkey in {"lmm", "fastlmm"}:
            try:
                pve_tmp = float(getattr(mod, "pve"))
                if np.isfinite(pve_tmp):
                    pve_now = pve_tmp
            except Exception:
                pve_now = None

        if mkey == "fastlmm" and _fastlmm_pve_is_degenerate(pve_now):
            if share_evd_lmm_fast and shared_lmm_model is not None:
                logger.warning(
                    f"Warning: FastLMM skipped for trait {pname}: "
                    f"PVE(null)={float(pve_now):.3f} outside "
                    f"[{_FASTLMM_PVE_LOW:.2f}, {_FASTLMM_PVE_HIGH:.2f}]."
                )
                continue
            logger.warning(
                f"Warning: FastLMM fallback to LMM for trait {pname}: "
                f"PVE(null)={float(pve_now):.3f} outside "
                f"[{_FASTLMM_PVE_LOW:.2f}, {_FASTLMM_PVE_HIGH:.2f}]."
            )
            effective_model_key = "lmm"
            effective_model_label = "LMM"
            effective_model_tag = "lmm"

        cpu_after = process.cpu_times()
        ctx["cpu_used"] = float(
            (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
        )
        ctx["model_key"] = effective_model_key
        ctx["model_label"] = effective_model_label
        ctx["model_tag"] = effective_model_tag
        if effective_model_tag != model_tag:
            ctx["tmp_tsv"] = (
                f"{outprefix}.{pname}.{genetic_model}.{effective_model_tag}.tsv.tmp."
                f"{os.getpid()}.{uuid.uuid4().hex}"
            )
            ctx["out_tsv"] = f"{outprefix}.{pname}.{genetic_model}.{effective_model_tag}.tsv"
            ctx["writer"] = _ChunkBatchWriter(str(ctx["tmp_tsv"]), batch_chunks=8)
        ctx["mod"] = mod
        model_ctxs.append(ctx)

    pbar_total = int(eff_snp_by_trait.get(pname, n_snps))
    use_rich_multi = bool(
        use_spinner
        and should_animate_status("Loading shared GWAS model progress...")
        and rich_progress_available()
    )
    rich_progress = None
    progress_started = False
    shared_warmup_task: Optional[CliStatus] = None
    shared_warmup_active = False
    if bool(use_spinner):
        shared_warmup_task = CliStatus(
            "Waiting for LMM GWAS",
            enabled=True,
            use_process=True,
        )
        shared_warmup_task.__enter__()
        shared_warmup_active = True

    def _ensure_shared_progress_started() -> None:
        nonlocal rich_progress, use_rich_multi, progress_started
        nonlocal shared_warmup_task, shared_warmup_active
        if progress_started:
            return
        if shared_warmup_active and shared_warmup_task is not None:
            try:
                shared_warmup_task.__exit__(None, None, None)
            except Exception:
                pass
            shared_warmup_active = False
        if use_rich_multi:
            rich_progress = build_rich_progress(
                description_template="[green]{task.description:<8}",
                show_remaining=False,
                show_percentage=False,
                field_templates=["{task.fields[metric]}"],
                bar_width=_GWAS_PROGRESS_BAR_WIDTH,
                finished_text=" ",
                transient=True,
            )
            if rich_progress is not None:
                rich_progress.start()
                for ctx in model_ctxs:
                    tid = rich_progress.add_task(
                        str(ctx["model_label"]),
                        total=pbar_total,
                        metric=_format_progress_metric(0, pbar_total, None),
                    )
                    ctx["task_id"] = int(tid)
            else:
                use_rich_multi = False

        if not use_rich_multi:
            for ctx in model_ctxs:
                if ctx.get("pbar") is None:
                    ctx["pbar"] = _ProgressAdapter(
                        total=pbar_total,
                        desc=str(ctx["model_label"]),
                        force_animate=True,
                    )
        progress_started = True

    def _metric_text(ctx: dict[str, object]) -> str:
        mem_text = ""
        if (
            str(ctx.get("memory_text", "")).strip() != ""
            and int(ctx.get("tick", 0)) <= int(ctx.get("memory_until_tick", 0))
        ):
            mem_text = str(ctx.get("memory_text", "")).strip()
        return _format_progress_metric(
            int(ctx.get("done_snps", 0)),
            int(pbar_total),
            mem_text if mem_text else None,
        )

    def _advance_ctx(ctx: dict[str, object], m_chunk: int, mem_text: Union[str, None]) -> None:
        done = int(ctx.get("done_snps", 0)) + int(m_chunk)
        tick = int(ctx.get("tick", 0)) + 1
        ctx["done_snps"] = done
        ctx["tick"] = tick
        if mem_text is not None:
            ctx["memory_text"] = str(mem_text).replace(" ", "")
            ctx["memory_until_tick"] = tick + 5
        metric = _metric_text(ctx)
        _ensure_shared_progress_started()
        if use_rich_multi and rich_progress is not None:
            rich_progress.update(int(ctx["task_id"]), advance=int(m_chunk), metric=metric)
        else:
            pbar_obj = ctx.get("pbar")
            if pbar_obj is not None:
                pbar_obj.update(int(m_chunk))
                if mem_text is not None:
                    pbar_obj.set_postfix(memory=str(mem_text))

    model_chunk_size = _resolve_stream_scan_chunk_size(
        int(chunk_size),
        int(n_snps),
        use_spinner=bool(use_spinner),
    )
    mmap_window_mb = (
        auto_mmap_window_mb(genofile, len(ids), n_snps, model_chunk_size)
        if mmap_limit else None
    )
    sample_sub = trait_ids
    expected_n = int(sample_sub.shape[0])

    scan_threads = int(threads)
    if scan_threads <= 0:
        scan_threads = int(n_cores)
    threads_per_model = max(1, scan_threads)
    prefetch_depth = 2 if scan_threads >= 2 else 1

    interrupted = False
    try:
        chunk_iter = load_genotype_chunks(
            genofile,
            model_chunk_size,
            maf_threshold,
            max_missing_rate,
            model=genetic_model,
            het=het_threshold,
            snps_only=bool(snps_only),
            sample_ids=sample_sub,
            mmap_window_mb=mmap_window_mb,
        )
        for genosub, sites in prefetch_iter(chunk_iter, in_flight=prefetch_depth):
            genosub: np.ndarray
            if genosub.shape[1] != expected_n:
                if genosub.shape[1] == int(sameidx.shape[0]):
                    genosub = genosub[:, sameidx]
                else:
                    raise ValueError(
                        f"Genotype sample dimension mismatch for trait {pname}: "
                        f"chunk has {genosub.shape[1]} columns, "
                        f"expected {expected_n} (or {sameidx.shape[0]} full aligned IDs)."
                    )
            if genetic_model != "add":
                keep_mask = _heter_keep_mask(genosub, het_threshold)
                if not np.any(keep_mask):
                    continue
                genosub = genosub[keep_mask]
                sites = [s for s, k in zip(sites, keep_mask) if k]
            m_chunk = int(genosub.shape[0])
            if m_chunk == 0:
                continue

            info_chunk = [
                (str(s.chrom), int(s.pos), str(s.ref_allele), str(s.alt_allele))
                for s in sites
            ]
            if len(info_chunk) == 0:
                continue

            geno_model = _apply_genetic_model(genosub, genetic_model)
            if genetic_model == "add":
                maf_chunk = (np.mean(genosub, axis=1) / 2).astype(np.float32, copy=False)
            else:
                maf_chunk = np.mean(geno_model, axis=1).astype(np.float32, copy=False)
            # In-place centering avoids one extra (m_chunk, n_samples) allocation.
            geno_center = np.asarray(geno_model, dtype=np.float32, order="C")
            geno_center -= np.mean(
                geno_center, axis=1, dtype=np.float32, keepdims=True
            )

            for ctx in model_ctxs:
                cpu_before = process.cpu_times()
                t0 = time.monotonic()
                results = ctx["mod"].gwas(geno_center, threads=threads_per_model)
                elapsed = max(time.monotonic() - t0, 0.0)
                cpu_after = process.cpu_times()
                ctx["scan_secs"] = float(ctx["scan_secs"]) + float(elapsed)
                ctx["cpu_used"] = float(ctx["cpu_used"]) + float(
                    (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system)
                )

                chunk_text, has_plrt = _format_chunk_tsv_text(results, info_chunk, maf_chunk)
                writer = ctx.get("writer")
                if writer is None:
                    raise RuntimeError("Internal error: shared GWAS writer not initialized.")
                writer.append(chunk_text, has_plrt=bool(has_plrt), rows=m_chunk)
                ctx["has_results"] = True

                mem_info = process.memory_info()
                ctx["peak_rss"] = max(int(ctx["peak_rss"]), int(mem_info.rss))
                done_next = int(ctx["done_snps"]) + m_chunk
                mem_text = None
                if done_next % (10 * chunk_size) == 0:
                    mem_text = f"{mem_info.rss / 1024**3:.2f}GB"
                _advance_ctx(ctx, m_chunk, mem_text)

        for ctx in model_ctxs:
            writer = ctx.get("writer")
            if writer is not None:
                writer.flush()
                ctx["wrote_header"] = bool(writer.wrote_header)
                ctx["has_results"] = int(writer.rows_written) > 0

        first_done = 0
        if progress_started:
            for ctx in model_ctxs:
                first_done = first_done or int(ctx.get("done_snps", 0))
                if use_rich_multi and rich_progress is not None:
                    rich_progress.update(
                        int(ctx["task_id"]),
                        completed=pbar_total,
                        metric=_metric_text(ctx),
                    )
                else:
                    pbar_obj = ctx.get("pbar")
                    if pbar_obj is not None:
                        pbar_obj.finish()
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        # Always stop any active progress renderer on Ctrl+C / errors.
        if shared_warmup_active and shared_warmup_task is not None:
            try:
                shared_warmup_task.__exit__(None, None, None)
            except Exception:
                pass
            shared_warmup_active = False
        if rich_progress is not None:
            try:
                rich_progress.stop()
            except Exception:
                pass
        for ctx in model_ctxs:
            pbar_obj = ctx.get("pbar")
            if pbar_obj is not None:
                try:
                    pbar_obj.close(show_done=False)
                except Exception:
                    pass
            if interrupted:
                tmp_tsv = str(ctx.get("tmp_tsv", ""))
                if tmp_tsv and os.path.exists(tmp_tsv):
                    try:
                        os.remove(tmp_tsv)
                    except Exception:
                        pass

    first_done = 0
    for ctx in model_ctxs:
        first_done = first_done or int(ctx.get("done_snps", 0))

    if pname not in eff_snp_by_trait:
        eff_snp_by_trait[pname] = int(first_done)

    for ctx in model_ctxs:
        model_label = str(ctx["model_label"])
        done_snps = int(ctx["done_snps"])
        evd_secs = float(ctx["evd_secs"])
        scan_secs = float(ctx["scan_secs"])
        cpu_used = float(ctx["cpu_used"])
        peak_rss_gb = float(int(ctx["peak_rss"]) / 1024**3)
        denom = max(evd_secs + scan_secs, 1e-9)
        avg_cpu_pct = 100.0 * cpu_used / denom / max(1, int(n_cores))
        init_log = str(ctx.get("init_log", "") or "").strip()
        if init_log != "":
            _log_model_line(
                logger,
                model_label,
                init_log,
                use_spinner=bool(use_spinner),
            )
        _log_model_line(
            logger,
            model_label,
            f"avg CPU ~ {avg_cpu_pct:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
            use_spinner=bool(use_spinner),
        )

        has_results = bool(ctx["has_results"])
        tmp_tsv = str(ctx["tmp_tsv"])
        out_tsv = str(ctx["out_tsv"])
        viz_secs = 0.0
        ctx_pve: Optional[float] = None
        if str(ctx.get("model_key", "")) in {"lmm", "fastlmm"}:
            mod_obj = ctx.get("mod")
            if mod_obj is not None and hasattr(mod_obj, "pve"):
                try:
                    pve_tmp = float(getattr(mod_obj, "pve"))
                    if np.isfinite(pve_tmp):
                        ctx_pve = pve_tmp
                except Exception:
                    ctx_pve = None
        if not has_results:
            logger.info(f"{model_label}: no SNPs passed filters for trait {pname}.")
            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(ctx_pve) if ctx_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": 0.0,
                    "result_file": "",
                }
            )
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
        if has_results:
            os.replace(tmp_tsv, out_tsv)
            saved_paths.append(str(out_tsv).replace("//", "/"))
            _log_model_line(
                logger,
                model_label,
                f"Results saved to {str(out_tsv).replace('//', '/')}",
                use_spinner=bool(use_spinner),
            )
            if plot:
                viz_t0 = time.time()
                plot_df = pd.read_csv(
                    out_tsv,
                    sep="\t",
                    usecols=["chrom", "pos", "pwald"],
                    dtype={"chrom": str, "pos": "int64"},
                )
                plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
                fastplot(
                    plot_df,
                    y_vec,
                    xlabel=str(pname),
                    outpdf=f"{outprefix}.{pname}.{genetic_model}.{str(ctx['model_tag'])}.svg",
                )
                viz_secs = max(time.time() - viz_t0, 0.0)

            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(done_snps),
                    "pve": (float(ctx_pve) if ctx_pve is not None else None),
                    "avg_cpu": float(avg_cpu_pct),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(evd_secs + scan_secs),
                    "viz_time_s": float(viz_secs),
                    "result_file": str(out_tsv).replace("//", "/"),
                }
            )

        time_parts: list[str] = []
        if evd_secs > 0:
            time_parts.append(format_elapsed(evd_secs))
        time_parts.append(format_elapsed(scan_secs))
        if plot and has_results:
            time_parts.append(format_elapsed(viz_secs))
        done_msg = f"{model_label} ...Finished [{'/'.join(time_parts)}]"
        _rich_success(logger, done_msg, use_spinner=use_spinner)

    trait_pve_val: Optional[float] = None
    trait_pve_src: Optional[str] = None
    for prefer_key in ("lmm", "fastlmm"):
        for ctx in model_ctxs:
            if str(ctx.get("model_key", "")) != prefer_key:
                continue
            mod_obj = ctx.get("mod")
            if mod_obj is None or (not hasattr(mod_obj, "pve")):
                continue
            try:
                pve_tmp = float(getattr(mod_obj, "pve"))
            except Exception:
                continue
            if np.isfinite(pve_tmp):
                trait_pve_val = float(pve_tmp)
                trait_pve_src = str(ctx.get("model_label", "")).strip() or (
                    "LMM" if prefer_key == "lmm" else "FastLMM"
                )
                break
        if trait_pve_val is not None:
            break
    if trait_pve_val is not None and trait_pve_src is not None:
        _emit_plain_info_line(
            logger,
            f"pve={trait_pve_val:.3f} (from {trait_pve_src})",
            use_spinner=bool(use_spinner),
        )


# ======================================================================
# High-memory FarmCPU: full genotype + QK
# ======================================================================

def prepare_qk_and_filter(
    geno: np.ndarray,
    ref_alt: pd.DataFrame,
    maf_threshold: float,
    max_missing_rate: float,
    logger,
):
    """
    Filter SNPs and impute missing values using QK, then update ref_alt.
    """
    logger.info(
        "* Filtering SNPs (MAF < "
        f"{maf_threshold} or missing rate > {max_missing_rate}; mode imputation)..."
    )
    logger.info("  Tip: if available, use pre-imputed genotypes from BEAGLE/IMPUTE2.")
    qkmodel = QK(geno, maff=maf_threshold, missf=max_missing_rate)
    geno_filt = qkmodel.M

    ref_alt_filt = ref_alt.loc[qkmodel.SNPretain].copy()
    # Swap REF/ALT for extremely rare alleles
    ref_alt_filt.iloc[qkmodel.maftmark, [0, 1]] = ref_alt_filt.iloc[
        qkmodel.maftmark, [1, 0]
    ]
    ref_alt_filt["maf"] = qkmodel.maf
    logger.info("Filtering and imputation finished.")
    return geno_filt, ref_alt_filt, qkmodel


def build_qmatrix_farmcpu(
    genofile: str,
    gfile_prefix: str,
    geno: Union[np.ndarray, None],
    qdim: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    cov_inputs: Union[str, list[str], None],
    chunk_size: int,
    logger,
    sample_ids: Union[np.ndarray, None] = None,
    use_spinner: bool = False,
    quiet_terminal: bool = False,
    snps_only: bool = True,
    n_snps_hint: Union[int, None] = None,
    threads: int = 1,
    mmap_limit: bool = False,
) -> np.ndarray:
    """
    Build or load Q matrix for FarmCPU (PCs + optional covariates).
    Note: external Q file via -q is no longer supported; pass external
    covariate matrices via -c.
    """
    def _farm_log(msg: str) -> None:
        if bool(quiet_terminal):
            _log_file_only(logger, logging.INFO, str(msg))
        else:
            logger.info(str(msg))

    sid_arr = None if sample_ids is None else np.asarray(sample_ids, dtype=str)
    if sid_arr is not None:
        n = int(sid_arr.shape[0])
    elif geno is not None:
        n = int(geno.shape[1])
    else:
        raise ValueError(
            "FarmCPU Q-matrix build requires either sample_ids or dense geno matrix."
        )

    def _load_or_build_grm_cache_for_pca() -> np.ndarray:
        grm_path, id_path = _grm_cache_paths(gfile_prefix, mgrm="1")
        with _cache_lock(grm_path):
            cache_ready = os.path.exists(grm_path)
            if cache_ready:
                g_mtime = latest_genotype_mtime(genofile)
                k_mtime = os.path.getmtime(grm_path)
                if g_mtime is not None and g_mtime > k_mtime:
                    cache_ready = False
            if cache_ready:
                _farm_log(f"* Loading GRM cache for FarmCPU PCA from {grm_path}...")
                grm = np.load(grm_path, mmap_mode="r")
                if grm.size != n * n:
                    _farm_log(
                        f"GRM cache shape mismatch ({grm.size} elements) for sample size {n}; rebuilding."
                    )
                    cache_ready = False
                else:
                    grm = grm.reshape(n, n)
                    if sid_arr is not None and os.path.exists(id_path):
                        grm_ids = _read_id_file(id_path, logger, "GRM", use_spinner=use_spinner)
                        if grm_ids is None or len(grm_ids) != n:
                            _farm_log("GRM cache IDs are invalid; rebuilding GRM cache.")
                            cache_ready = False
                        else:
                            grm_ids = np.asarray(grm_ids, dtype=str)
                            sid = sid_arr
                            if np.array_equal(grm_ids, sid):
                                return np.asarray(grm, dtype="float32")
                            index = {s: i for i, s in enumerate(grm_ids)}
                            missing = [s for s in sid if s not in index]
                            if missing:
                                _farm_log(
                                    f"GRM cache missing {len(missing)} sample IDs; rebuilding GRM cache."
                                )
                                cache_ready = False
                            else:
                                ord_idx = [index[s] for s in sid]
                                grm = grm[np.ix_(ord_idx, ord_idx)]
                                return np.asarray(grm, dtype="float32")
                    else:
                        if sid_arr is not None and not os.path.exists(id_path):
                            _farm_log("GRM cache ID file not found; assuming genotype sample order.")
                        return np.asarray(grm, dtype="float32")
            if not cache_ready:
                for p in (grm_path, id_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                _farm_log("* Building GRM cache for FarmCPU PCA...")
                if geno is not None:
                    grm = GRM(geno).astype("float32")
                else:
                    if n_snps_hint is not None:
                        n_snps_eff = int(n_snps_hint)
                    else:
                        _ids0, n_snps0 = inspect_genotype_file(
                            genofile,
                            snps_only=bool(snps_only),
                            maf=float(maf_threshold),
                            missing_rate=float(max_missing_rate),
                            het=float(het_threshold),
                        )
                        n_snps_eff = int(n_snps0)
                    grm, _eff_m = build_grm_streaming(
                        genofile=genofile,
                        n_samples=n,
                        n_snps=n_snps_eff,
                        maf_threshold=float(maf_threshold),
                        max_missing_rate=float(max_missing_rate),
                        chunk_size=int(chunk_size),
                        method=1,
                        mmap_window_mb=auto_mmap_window_mb(
                            genofile, n, n_snps_eff, int(chunk_size)
                        ) if bool(mmap_limit) else None,
                        threads=max(1, int(threads)),
                        logger=logger,
                        use_spinner=bool(use_spinner),
                        snps_only=bool(snps_only),
                    )
                    grm = np.asarray(grm, dtype="float32")
                tmp_grm = f"{grm_path}.tmp.{os.getpid()}.npy"
                np.save(tmp_grm, grm)
                os.replace(tmp_grm, grm_path)
                if sid_arr is not None:
                    tmp_id = f"{id_path}.tmp.{os.getpid()}"
                    pd.Series(sid_arr).to_csv(
                        tmp_id, sep="\t", index=False, header=False
                    )
                    os.replace(tmp_id, id_path)
                _farm_log(f"Cached GRM written to {grm_path}")
                return grm

    q_int = _parse_qcov_dim(qdim)
    if q_int >= n and q_int != 0:
        raise ValueError(
            f"Q/PC dimension out of range for FarmCPU: {q_int}. "
            f"valid=[0..{max(0, n-1)}]"
        )

    if q_int == 0:
        qmatrix = np.zeros((n, 0), dtype="float32")
        _emit_warning_line(
            logger,
            "PC dimension set to 0; using empty Q matrix.",
            use_spinner=bool(use_spinner),
        )
    else:
        q_path = _pca_cache_path(gfile_prefix, mgrm="1", qdim=int(q_int))
        with _cache_lock(q_path):
            cache_ready = os.path.isfile(q_path)
            if cache_ready:
                g_mtime = latest_genotype_mtime(genofile)
                q_mtime = os.path.getmtime(q_path)
                if g_mtime is not None and g_mtime > q_mtime:
                    cache_ready = False
            if cache_ready:
                q_src = _basename_only(q_path)
                with CliStatus(
                    f"Loading Q matrix from {q_src}...",
                    enabled=bool(use_spinner),
                ) as task:
                    qmatrix = np.loadtxt(q_path, dtype="float32", delimiter="\t")
                    if qmatrix.ndim == 1:
                        qmatrix = qmatrix.reshape(-1, 1)
                    if qmatrix.shape != (n, int(q_int)):
                        raise ValueError(
                            f"PCA cache shape mismatch: expected ({n},{q_int}), got {qmatrix.shape}"
                        )
                    task.complete(
                        f"Loading Q matrix from {q_src} (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]}) ...Finished"
                    )
            else:
                _farm_log(f"* PCA dimension for FarmCPU Q matrix: {q_int}")
                grm = _load_or_build_grm_cache_for_pca()
                eigval, eigvec = np.linalg.eigh(grm)
                qmatrix = np.asarray(eigvec[:, -q_int:], dtype="float32")
                tmp_q = f"{q_path}.tmp.{os.getpid()}"
                np.savetxt(tmp_q, qmatrix, fmt="%.8g", delimiter="\t")
                os.replace(tmp_q, q_path)
                _farm_log(f"Cached PCA written to {q_path}")

    if cov_inputs:
        if sid_arr is None:
            raise ValueError("FarmCPU covariate loading requires sample IDs.")
        cov_arr, cov_ids = _load_covariates_for_models(
            cov_inputs=cov_inputs,
            genofile=genofile,
            sample_ids=sid_arr,
            chunk_size=int(chunk_size),
            logger=logger,
            context="FarmCPU",
            use_spinner=bool(use_spinner),
            snps_only=bool(snps_only),
        )
        if cov_arr is not None:
            if cov_ids is None:
                raise ValueError("Internal error: covariate IDs are missing for FarmCPU.")
            sid = sid_arr
            if cov_arr.shape[0] != sid.shape[0]:
                raise ValueError(
                    f"FarmCPU covariate rows ({cov_arr.shape[0]}) do not match sample count ({sid.shape[0]})."
                )
            if not np.array_equal(np.asarray(cov_ids, dtype=str), sid):
                raise ValueError(
                    "FarmCPU covariate sample order does not match genotype sample order after alignment."
                )
            qmatrix = np.concatenate([qmatrix, cov_arr], axis=1)
    else:
        _emit_warning_line(
            logger,
            "Loading covariates (streaming) ...Skipped (none)",
            use_spinner=bool(use_spinner),
        )

    _farm_log(f"Q matrix (FarmCPU) shape: {qmatrix.shape}")
    return qmatrix


def run_farmcpu_fullmem(
    args,
    gfile: str,
    prefix: str,
    logger: logging.Logger,
    pheno_preloaded: Union[pd.DataFrame , None] = None,
    ids_preloaded: Union[np.ndarray, None] = None,
    n_snps_preloaded: Union[int, None] = None,
    qmatrix_preloaded: Union[np.ndarray, None] = None,
    cov_preloaded: Union[np.ndarray, None] = None,
    use_spinner: bool = False,
    context_prepared: bool = False,
    summary_rows: Union[list[dict[str, object]], None] = None,
    saved_paths: Union[list[str], None] = None,
    trait_names: Union[list[str], None] = None,
    farmcpu_cache: Union[dict[str, object], None] = None,
    prepare_only: bool = False,
    emit_trait_header: bool = True,
) -> dict[str, object]:
    """
    Run FarmCPU in high-memory mode (full genotype + QK + PCA).

    If pheno_preloaded is provided from a non-streaming path, it may be reused.
    When called after streaming GWAS context preparation, FarmCPU must use full
    genotype IDs and therefore reload phenotype/Q/cov on the full ID space.
    """
    phenofile = args.pheno
    outfolder = args.out
    qdim = args.qcov
    cov = args.cov
    snps_only = bool(getattr(args, "snps_only", False))

    def _load_bim_ref_alt_filtered(
        prefix: str,
        keep_mask: np.ndarray,
        *,
        snps_only_mode: bool,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        bim_path = f"{prefix}.bim"
        src = _basename_only(bim_path)
        keep = np.ascontiguousarray(np.asarray(keep_mask, dtype=np.bool_).reshape(-1))
        n_total = int(keep.shape[0])
        if n_total == 0:
            return (
                pd.DataFrame(columns=["chrom", "pos", "allele0", "allele1"]),
                keep,
            )

        n_pre_keep = int(np.sum(keep))
        if n_pre_keep == 0:
            return (
                pd.DataFrame(columns=["chrom", "pos", "allele0", "allele1"]),
                keep,
            )

        # Stream BIM and keep only SNPs that survive numeric filters, optionally
        # applying SNP-only filtering in the same pass.
        chrom = np.empty(n_pre_keep, dtype=object)
        pos = np.empty(n_pre_keep, dtype=np.int64)
        allele0 = np.empty(n_pre_keep, dtype=object)
        allele1 = np.empty(n_pre_keep, dtype=object)

        pbar = _ProgressAdapter(total=n_total, desc=f"Loading site metadata ({src})")
        lines = 0
        done = 0
        out = 0
        try:
            with open(bim_path, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    if lines >= n_total:
                        raise ValueError(
                            f"BIM has more rows than genotype matrix: bim>{n_total} ({src})"
                        )
                    idx = lines
                    lines += 1

                    if keep[idx]:
                        parts = raw.strip().split()
                        if len(parts) < 6:
                            raise ValueError(
                                f"Invalid BIM row at line {idx + 1}: expected >=6 columns."
                            )
                        a0 = str(parts[4])
                        a1 = str(parts[5])
                        if snps_only_mode and (len(a0) != 1 or len(a1) != 1):
                            keep[idx] = False
                        else:
                            chrom[out] = str(parts[0])
                            try:
                                pval = int(parts[3])
                            except Exception:
                                try:
                                    pval = int(float(parts[3]))
                                except Exception:
                                    pval = 0
                            pos[out] = int(pval)
                            allele0[out] = a0
                            allele1[out] = a1
                            out += 1

                    if lines - done >= 200_000:
                        step = lines - done
                        pbar.update(step)
                        done = lines

            if lines != n_total:
                raise ValueError(
                    f"BIM row count mismatch: bim={lines}, genotype={n_total} ({src})"
                )
            if done < n_total:
                pbar.update(n_total - done)
            pbar.finish()
        finally:
            pbar.close(success_style=True, show_done=True)

        ref_alt_df = pd.DataFrame(
            {
                "chrom": chrom[:out],
                "pos": pos[:out].astype(int, copy=False),
                "allele0": allele0[:out],
                "allele1": allele1[:out],
            }
        )
        return ref_alt_df, keep

    if farmcpu_cache is None:
        t_loading = time.time()
        # If FarmCPU is invoked after streaming models, pheno_preloaded/ids_preloaded
        # are usually intersection-aligned. FarmCPU must operate on full genotype IDs.
        reuse_preloaded_pheno = bool(pheno_preloaded is not None) and (not bool(context_prepared))
        pheno = pheno_preloaded if reuse_preloaded_pheno else None
        if pheno is None:
            pheno = _load_phenotype_with_status(
                phenofile,
                args.ncol,
                logger,
                id_col=0,
                use_spinner=use_spinner,
            )
        else:
            if not bool(context_prepared):
                with CliStatus("Loading phenotype from dataframe...", enabled=bool(use_spinner)) as task:
                    task.complete(
                        f"Loading phenotype from dataframe (n={pheno.shape[0]}, npheno={pheno.shape[1]})"
                    )

        # Always inspect full genotype metadata for FarmCPU to avoid reusing
        # streaming-intersection IDs (which can be smaller than packed BED n).
        famid, n_snps = _inspect_genotype_with_status(
            gfile,
            logger,
            use_spinner=use_spinner,
            snps_only=bool(snps_only),
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            het_threshold=float(args.het),
        )
        famid = np.asarray(famid, dtype=str)
        geno = None
        packed_ctx: Union[dict[str, object], None] = None

        packed_prefix = _as_plink_prefix(gfile)
        if packed_prefix is None:
            cache_guess = _gwas_cache_prefix_with_params(
                gfile,
                maf=float(args.maf),
                geno=float(args.geno),
                snps_only=bool(snps_only),
                logger=logger,
            )
            packed_prefix = _as_plink_prefix(cache_guess)

        can_use_packed = (
            bool(getattr(args, "model", "add") == "add")
            and packed_prefix is not None
        )
        if can_use_packed:
            packed_load_t0 = time.monotonic()
            packed_status = CliStatus(
                "Loading genotype (Full)...",
                enabled=bool(use_spinner),
            )
            with packed_status as task:
                try:
                    packed_raw, miss_raw, maf_raw, _std_raw, packed_n = load_bed_2bit_packed(
                        str(packed_prefix)
                    )
                except Exception:
                    task.fail("Loading genotype (Full) ...Failed")
                    raise
            _log_file_only(
                logger,
                logging.INFO,
                f"Loading genotype (Full, {int(n_snps)} SNPs) "
                f"[{format_elapsed(time.monotonic() - packed_load_t0)}]",
            )
            if int(packed_n) != int(famid.shape[0]):
                raise ValueError(
                    f"Packed sample size mismatch: packed n={packed_n}, expected {famid.shape[0]}"
                )

            miss_arr = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1))
            maf_arr = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1))
            keep = np.ones(maf_arr.shape[0], dtype=np.bool_)
            maf_thr = float(args.maf)
            if maf_thr > 0.0:
                keep &= (maf_arr >= maf_thr) & (maf_arr <= (1.0 - maf_thr))
            miss_thr = float(args.geno)
            if miss_thr < 1.0:
                keep &= miss_arr <= miss_thr

            ref_alt, keep_final = _load_bim_ref_alt_filtered(
                str(packed_prefix),
                keep,
                snps_only_mode=bool(snps_only),
            )
            if not np.any(keep_final):
                raise ValueError("After filtering, number of SNPs is zero for FarmCPU.")

            if np.all(keep_final):
                packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8))
                miss_arr = np.ascontiguousarray(miss_arr, dtype=np.float32)
                maf_arr = np.ascontiguousarray(maf_arr, dtype=np.float32)
            else:
                packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8)[keep_final])
                miss_arr = np.ascontiguousarray(miss_arr[keep_final], dtype=np.float32)
                maf_arr = np.ascontiguousarray(maf_arr[keep_final], dtype=np.float32)

            loaded_snps = int(ref_alt.shape[0])
            packed_ctx = {
                "packed": packed,
                "missing_rate": miss_arr,
                "maf": maf_arr,
                "n_samples": int(packed_n),
            }
        else:
            geno_chunks = []
            site_rows = []
            full_load_t0 = time.monotonic()
            pbar = _ProgressAdapter(total=n_snps, desc="Loading genotype (Full)")
            try:
                for chunk, sites in load_genotype_chunks(
                    gfile,
                    chunk_size=args.chunksize,
                    maf=args.maf,
                    missing_rate=args.geno,
                    impute=True,
                    snps_only=bool(snps_only),
                    sample_ids=famid.tolist(),
                ):
                    if chunk.shape[0] == 0:
                        continue
                    geno_chunks.append(np.asarray(chunk, dtype="float32"))
                    site_rows.extend(
                        [(s.chrom, int(s.pos), s.ref_allele, s.alt_allele) for s in sites]
                    )
                    pbar.update(chunk.shape[0])
                loaded_snps = int(sum(c.shape[0] for c in geno_chunks))
                pbar.set_desc(f"Loading genotype (Full, {loaded_snps} SNPs)")
                pbar.finish()
            finally:
                # Ensure spinner/progress stops on Ctrl+C.
                pbar.close(success_style=False, show_done=False)
            _log_file_only(
                logger,
                logging.INFO,
                f"Loading genotype (Full, {loaded_snps} SNPs) "
                f"[{format_elapsed(time.monotonic() - full_load_t0)}]",
            )

            if len(geno_chunks) == 0:
                msg = "After filtering, number of SNPs is zero for FarmCPU."
                logger.error(msg)
                raise ValueError(msg)
            geno = np.concatenate(geno_chunks, axis=0)
            ref_alt = pd.DataFrame(site_rows, columns=["chrom", "pos", "allele0", "allele1"])
            ref_alt["pos"] = pd.to_numeric(ref_alt["pos"], errors="coerce").fillna(0).astype(int)

        t_loaded = time.time() - t_loading
        if (not bool(context_prepared)) and (not bool(prepare_only)):
            ns_loaded = int(ref_alt.shape[0])
            _rich_success(
                logger,
                f"FarmCPU input ready (n={len(famid)}, nSNP={ns_loaded}) [{format_elapsed(t_loaded)}]",
                use_spinner=use_spinner,
                log_message=f"Genotype and phenotype loaded in {t_loaded:.2f} seconds",
            )
        else:
            _log_file_only(
                logger,
                logging.INFO,
                f"Genotype and phenotype loaded in {t_loaded:.2f} seconds",
            )
        if ref_alt.shape[0] == 0:
            msg = "After filtering, number of SNPs is zero for FarmCPU."
            logger.error(msg)
            raise ValueError(msg)

        # Build FarmCPU Q/cov on full genotype IDs; do not reuse streaming-aligned
        # preloaded Q/cov because they may be trimmed by GRM/Q/cov intersections.
        gfile_prefix = _gwas_cache_prefix_with_params(
            gfile,
            maf=float(args.maf),
            geno=float(args.geno),
            snps_only=bool(snps_only),
            logger=logger,
        )
        qmatrix = build_qmatrix_farmcpu(
            genofile=gfile,
            gfile_prefix=gfile_prefix,
            geno=geno,
            qdim=qdim,
            maf_threshold=float(args.maf),
            max_missing_rate=float(args.geno),
            het_threshold=float(args.het),
            cov_inputs=cov,
            chunk_size=args.chunksize,
            logger=logger,
            sample_ids=famid.astype(str),
            use_spinner=use_spinner,
            quiet_terminal=bool(context_prepared or prepare_only),
            snps_only=bool(snps_only),
            n_snps_hint=int(ref_alt.shape[0]),
            threads=int(args.thread),
            mmap_limit=bool(args.mmap_limit),
        )

        cov_n: Union[int, str] = "NA"
        if len(_normalize_cov_inputs(cov)) > 0:
            cov_n = int(famid.shape[0])
        geno_n = int(famid.shape[0]) if geno is None else int(geno.shape[1])
        pheno_ids_set = set(np.asarray(pheno.index, dtype=str))
        common_n = int(sum(1 for sid in np.asarray(famid, dtype=str) if sid in pheno_ids_set))
        q_n: Union[int, str] = (
            int(qmatrix.shape[0]) if (np.asarray(qmatrix).ndim == 2 and int(qmatrix.shape[1]) > 0) else "NA"
        )
        _emit_plain_info_line(
            logger,
            f"geno={geno_n}, pheno={pheno.shape[0]}, q={q_n}, cov={cov_n} -> {common_n}",
            use_spinner=bool(use_spinner),
        )
        farmcpu_cache = {
            "pheno": pheno,
            "famid": famid,
            "geno": geno,
            "packed_ctx": packed_ctx,
            "ref_alt": ref_alt,
            "qmatrix": qmatrix,
        }
    else:
        pheno = farmcpu_cache["pheno"]  # type: ignore[assignment]
        famid = np.asarray(farmcpu_cache["famid"], dtype=str)
        geno_obj = farmcpu_cache.get("geno")
        geno = (
            None
            if geno_obj is None
            else np.ascontiguousarray(np.asarray(geno_obj, dtype="float32"))
        )
        packed_obj = farmcpu_cache.get("packed_ctx")
        packed_ctx = None
        if isinstance(packed_obj, dict):
            packed_ctx = {
                "packed": np.ascontiguousarray(
                    np.asarray(packed_obj["packed"], dtype=np.uint8)
                ),
                "missing_rate": np.ascontiguousarray(
                    np.asarray(packed_obj["missing_rate"], dtype=np.float32)
                ),
                "maf": np.ascontiguousarray(
                    np.asarray(packed_obj["maf"], dtype=np.float32)
                ),
                "n_samples": int(packed_obj["n_samples"]),
            }
        ref_alt = farmcpu_cache["ref_alt"]  # type: ignore[assignment]
        qmatrix = np.asarray(farmcpu_cache["qmatrix"], dtype="float32")

    if bool(prepare_only):
        return farmcpu_cache if farmcpu_cache is not None else {}

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    trait_iter = list(pheno.columns) if trait_names is None else [t for t in trait_names if t in pheno.columns]
    multi_trait_mode = len(trait_iter) > 1
    for trait_idx, phename in enumerate(trait_iter):
        p = pheno[phename].dropna()
        famidretain = np.isin(famid, p.index)
        if np.sum(famidretain) == 0:
            logger.warning(f"{phename}: no overlapping samples, skipped.")
            if multi_trait_mode:
                logger.info("")  # single blank line between traits
            continue

        p_sub = p.loc[famid[famidretain]].values.reshape(-1, 1)
        keep_idx = np.flatnonzero(famidretain).astype(np.int64, copy=False)
        q_sub = qmatrix[famidretain]
        n_idv = int(np.sum(famidretain))
        if packed_ctx is None:
            if geno is None:
                raise ValueError("FarmCPU genotype payload is missing.")
            m_input = np.ascontiguousarray(geno[:, famidretain], dtype=np.float32)
            sample_idx_arg = None
            maf = (m_input.mean(axis=1) / 2.0).astype(np.float32, copy=False)
            eff_snp = int(m_input.shape[0])
        else:
            m_input = packed_ctx
            sample_idx_arg = keep_idx
            maf = np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1)
            eff_snp = int(maf.shape[0])

        if bool(emit_trait_header):
            _emit_trait_header(
                logger,
                phename,
                n_idv,
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        cpu_t0 = process.cpu_times()
        t0 = time.time()
        gwas_t0 = time.time()
        peak_rss = process.memory_info().rss
        farm_iter = 20
        farm_label = f"FarmCPU-{phename} (n={n_idv})"
        farm_pbar = _ProgressAdapter(
            total=farm_iter,
            desc=farm_label,
            force_animate=bool(use_spinner),
        )
        farm_state = {"done": 0}

        def _farmcpu_progress(done: int, total: int) -> None:
            nonlocal peak_rss
            target = int(max(0, min(int(total), int(done))))
            delta = target - int(farm_state["done"])
            if delta > 0:
                farm_pbar.update(delta)
                farm_state["done"] = target
            try:
                peak_rss = max(peak_rss, process.memory_info().rss)
            except Exception:
                pass

        try:
            farm_out = farmcpu(
                y=p_sub,
                M=m_input,
                X=q_sub,
                chrlist=ref_alt["chrom"].values,
                poslist=ref_alt["pos"].values,
                iter=farm_iter,
                threads=args.thread,
                sample_indices=sample_idx_arg,
                progress_cb=_farmcpu_progress,
                return_info=True,
            )
            farm_pbar.finish()
        finally:
            # Ensure spinner/progress stops on Ctrl+C.
            farm_pbar.close(show_done=False)
        if isinstance(farm_out, tuple):
            res, _farm_info = farm_out
            n_pseudo_qtn = int(_farm_info.get("n_pseudo_qtn", 0))
        else:
            res = farm_out
            n_pseudo_qtn = 0
        gwas_secs = max(time.time() - gwas_t0, 0.0)
        res_df = pd.DataFrame(res, columns=["beta", "se", "pwald"])
        res_df['maf'] = maf
        res_df = pd.concat([ref_alt.reset_index(drop=True), res_df], axis=1)
        res_df = res_df[['chrom','pos','allele0','allele1','maf','beta','se','pwald']]

        viz_secs = 0.0
        if args.plot:
            viz_t0 = time.time()
            fastplot(
                res_df,
                p_sub,
                xlabel=phename,
                outpdf=f"{outfolder}/{prefix}.{phename}.farmcpu.svg",
            )
            viz_secs = max(time.time() - viz_t0, 0.0)

        peak_rss = max(peak_rss, process.memory_info().rss)
        cpu_t1 = process.cpu_times()
        wall = max(time.time() - t0, 1e-9)
        cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
        avg_cpu = 100.0 * cpu_used / (wall * max(1, n_cores))
        peak_rss_gb = peak_rss / (1024 ** 3)
        out_tsv = f"{outfolder}/{prefix}.{phename}.farmcpu.tsv"
        _log_model_line(
            logger,
            "FarmCPU",
            f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, "
            f"peak RSS ~ {peak_rss_gb:.2f} G, pseudoQTNs ~ {n_pseudo_qtn}",
            use_spinner=bool(use_spinner),
        )
        summary_rows.append(
            {
                "phenotype": str(phename),
                "model": "Farm",
                "nidv": int(n_idv),
                "eff_snp": int(eff_snp),
                "pve": None,
                "avg_cpu": float(avg_cpu),
                "peak_rss_gb": float(peak_rss_gb),
                "gwas_time_s": float(gwas_secs),
                "viz_time_s": float(viz_secs),
                "result_file": str(out_tsv).replace("//", "/"),
            }
        )

        res_df = res_df.astype({"pwald": "object","pos":int})
        res_df.loc[:, "pwald"] = res_df["pwald"].map(lambda x: f"{x:.4e}")
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        saved_paths.append(str(out_tsv).replace("//", "/"))
        _log_model_line(
            logger,
            "FarmCPU",
            f"Results saved to {str(out_tsv).replace('//', '/')}",
            use_spinner=bool(use_spinner),
        )
        farm_times = [format_elapsed(gwas_secs)]
        if args.plot:
            farm_times.append(format_elapsed(viz_secs))
        farm_done_msg = f"FarmCPU ...Found {n_pseudo_qtn} QTNs [{'/'.join(farm_times)}]"
        if (not bool(emit_trait_header)) and multi_trait_mode:
            farm_done_msg = f"FarmCPU({phename}) ...Found {n_pseudo_qtn} QTNs [{'/'.join(farm_times)}]"
        _rich_success(logger, farm_done_msg, use_spinner=use_spinner)
        if multi_trait_mode:
            logger.info("")
    return farmcpu_cache


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = CliArgumentParser(
        prog="jx gwas",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx gwas -vcf example.vcf.gz -p pheno.tsv -lmm",
            "jx gwas -hmp example.hmp.gz -p pheno.tsv -lmm",
            "jx gwas -bfile example_prefix -p pheno.tsv -lm",
        ]),
    )

    required_group = parser.add_argument_group("Required arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=False)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp", type=str,
        help="Input genotype file in HMP format (.hmp or .hmp.gz).",
    )
    geno_group.add_argument(
        "-file", "--file", type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format "
             "(prefix for .bed, .bim, .fam).",
    )

    required_group.add_argument(
        "-p", "--pheno", type=str, required=False,
        help="Phenotype file (tab-delimited, sample IDs in the first column).",
    )

    models_group = parser.add_argument_group("Model Arguments")
    models_group.add_argument(
        "-lmm", "--lmm", action="store_true", default=False,
        help="Run the linear mixed model (streaming, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-fastlmm", "--fastlmm", action="store_true", default=False,
        help="Run the linear mixed model with fixed lambda estimated in null model (streaming, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-lrlmm", "--lrlmm", nargs="?", const=-1, default=100, type=int, metavar="RANK",
        help=(
            "Run low-rank LMM (packed BED + RSVD + Rust FaST GWAS). "
            "Optional RANK; default is %(default)s."
        ),
    )
    models_group.add_argument(
        "-farmcpu", "--farmcpu", action="store_true", default=False,
        help="Run FarmCPU (full genotype in memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-lm", "--lm", action="store_true", default=False,
        help="Run the linear model (streaming, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-model", "--model", type=str, choices=["add", "dom", "rec", "het"], default="add",
        help="Genetic effect coding model for streaming LM/LMM/FastLMM/LRLMM (default: %(default)s).",
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-n", "--n", action="extend", nargs="+", metavar="COL",
        default=None, type=str, dest="ncol",
        help=(
            "Phenotype column(s), zero-based index (excluding sample ID), "
            "comma list (e.g. 0,2), or numeric range (e.g. 0:2). "
            "Repeat this flag for multiple traits."
        ),
    )
    optional_group.add_argument(
        "--ncol", action="extend", nargs="+", metavar="COL",
        default=None, type=str, dest="ncol", help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-k", "--grm", type=str, default="1",
        help="GRM option: 1 (centering), 2 (standardization), "
             "or a path to a precomputed GRM file (default: %(default)s).",
    )
    optional_group.add_argument(
        "-q", "--qcov", type=str, default="0",
        help=(
            "Number of principal components for Q matrix (integer >= 0). "
            "For external covariates, use -c <file> (default: %(default)s)."
        ),
    )
    optional_group.add_argument(
        "-c", "--cov", action="append", type=str, default=None,
        help=(
            "Additional covariate input (repeatable). Each -c accepts either: "
            "(1) covariate file path, or "
            "    covariate file format: first column sample ID, remaining columns numeric covariates; or "
            "(2) single-site token chr:pos / chr:start:end (start must equal end, "
            "supports full-width colon). "
            "Examples: -c cov.tsv -c 1:1000 -c 1:1000:1000."
        ),
    )
    optional_group.add_argument(
        "-snps-only", "--snps-only", action="store_true", default=False,
        help="Exclude non-SNP variants.",
    )
    optional_group.add_argument(
        "-maf", "--maf", type=float, default=0.02,
        help="Exclude variants with minor allele frequency lower than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-geno", "--geno", type=float, default=0.05,
        help="Exclude variants with missing call frequencies greater than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-het", "--het", type=float, default=0.02,
        help="Heterozygosity filter threshold for non-additive models. "
             "Sites with het rate outside [het, 1-het] are removed (default: %(default)s).",
    )
    optional_group.add_argument(
        "-chunksize", "--chunksize", type=int, default=10_000,
        help="Number of SNPs per chunk for streaming LMM/LM and RSVD mmap sizing "
             "(affects GRM and GWAS; default: %(default)s).",
    )
    optional_group.add_argument(
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=detect_effective_threads(),
        help="Number of CPU threads (default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: %(default)s).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix for output files (default: %(default)s).",
    )

    args, extras = parser.parse_known_args()
    has_genotype = bool(args.vcf or args.hmp or args.file or args.bfile)
    has_pheno = bool(args.pheno)
    if (not has_pheno) and (not has_genotype):
        parser.error(
            "the following arguments are required: -p/--pheno & "
            "(-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
        )
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if not has_genotype:
        parser.error(
            "the following arguments are required: "
            "(-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
        )
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))
    try:
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))
    try:
        args.qcov = str(_parse_qcov_dim(args.qcov))
    except ValueError as e:
        parser.error(str(e))
    return args


def main(log: bool = True):
    t_start = time.time()
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_status = "done"
    run_error = ""
    args = parse_args()
    # Plotting is always enabled for GWAS CLI.
    args.plot = True
    args.cov = _normalize_cov_inputs(args.cov)
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False

    if args.thread <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)
    args.model = args.model.lower()
    if not (0.0 <= args.het <= 0.5):
        raise ValueError("--het must be within [0, 0.5].")

    gfile, prefix = determine_genotype_source(args)

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    os.makedirs(args.out, 0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)
    outprefix = f"{args.out}/{prefix}".replace("\\", "/").replace("//", "/")
    log_path = f"{outprefix}.gwas.log"
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    gwas_summary_rows: list[dict[str, object]] = []
    saved_result_paths: list[str] = []
    trait_order: list[str] = []

    if log:
        model_tokens: list[str] = []
        if args.lmm:
            model_tokens.append("LMM")
        if args.fastlmm:
            model_tokens.append("fastLMM")
        if args.lrlmm is not None:
            if int(args.lrlmm) > 0:
                model_tokens.append(f"LRLMM(rank={int(args.lrlmm)})")
            else:
                model_tokens.append("LRLMM(rank=sqrt(n))")
        if args.lm:
            model_tokens.append("LM")
        if args.farmcpu:
            model_tokens.append("FarmCPU")
        cfg_rows: list[tuple[str, object]] = [
            ("Genotype file", gfile),
            ("Phenotype file", args.pheno),
            ("Phenotype cols", args.ncol if args.ncol is not None else "All"),
            ("Mmap limit", args.mmap_limit),
            ("Models", " ".join(model_tokens) if len(model_tokens) > 0 else "None"),
            ("Genetic model", args.model),
            ("SNPs only", args.snps_only),
            ("GRM option", args.grm),
            ("Q option", args.qcov),
            ("MAF threshold", args.maf),
            ("Miss threshold", args.geno),
            ("Chunk size", args.chunksize),
        ]
        if args.lrlmm is not None:
            cfg_rows.append(
                ("LRLMM rank", int(args.lrlmm) if int(args.lrlmm) > 0 else "sqrt(n)")
            )
        if args.model != "add":
            cfg_rows.append(("Het filter", f"{args.het} (keep [{args.het}, {1.0 - args.het}])"))
        if args.cov:
            cfg_rows.append(("Covariates", "; ".join(args.cov)))
        emit_cli_configuration(
            logger,
            app_title="JanusX - GWAS",
            config_title="GWAS CONFIG",
            host=socket.gethostname(),
            sections=[("General", cfg_rows)],
            footer_rows=[
                ("Threads", f"{args.thread} ({detected_threads} available)"),
                ("Output prefix", outprefix),
            ],
            line_max_chars=60,
        )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if args.grm not in ["1", "2"]:
        checks.append(ensure_file_exists(logger, args.grm, "GRM file"))
    if args.cov:
        for cov_item in args.cov:
            try:
                site_token = _parse_cov_site_token(cov_item)
            except Exception as e:
                logger.error(str(e))
                raise SystemExit(1)
            if site_token is None:
                checks.append(ensure_file_exists(logger, cov_item, "Covariate file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if not (args.lm or args.lmm or args.fastlmm or (args.lrlmm is not None) or args.farmcpu):
        logger.error(
            "No model selected. Use -lm, -lmm, -fastlmm, -lrlmm, and/or -farmcpu."
        )
        raise SystemExit(1)
    if (args.lrlmm is not None) and args.model != "add":
        logger.error("LRLMM currently supports additive coding only (--model add).")
        raise SystemExit(1)
    if args.farmcpu and args.model != "add":
        logger.warning(
            "Warning: --model/--het currently apply to streaming LM/LMM/FastLMM/LRLMM; "
            "FarmCPU keeps additive coding."
        )

    try:

        # --- prepare streaming context once if needed ---
        pheno = None
        ids = None
        n_snps = None
        grm = None
        qmatrix = None
        cov_all = None
        eff_m = None
        genofile_stream = gfile
        eff_snp_by_trait: dict[str, int] = {}

        stream_selected = bool(args.lmm or args.lm or args.fastlmm)
        if stream_selected:
            _section(logger, "Streaming task")
            _log_file_only(
                logger,
                logging.INFO,
                "Prepare streaming context (phenotype/genotype meta/GRM/Q/cov)",
            )
            pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m, genofile_stream = prepare_streaming_context(
                genofile=gfile,
                phenofile=args.pheno,
                pheno_cols=args.ncol,
                maf_threshold=args.maf,
                max_missing_rate=args.geno,
                genetic_model=args.model,
                het_threshold=args.het,
                chunk_size=args.chunksize,
                mgrm=args.grm,
                pcdim=args.qcov,
                cov_inputs=args.cov,
                threads=args.thread,
                mmap_limit=args.mmap_limit,
                require_kinship=(args.lmm or args.fastlmm),
                logger=logger,
                use_spinner=use_spinner,
                snps_only=bool(args.snps_only),
            )
            _phase_split(logger)
        else:
            if args.farmcpu and (args.lrlmm is None):
                pheno = _load_phenotype_with_status(
                    args.pheno,
                    args.ncol,
                    logger,
                    id_col=0,
                    use_spinner=use_spinner,
                )

        stream_models: list[str] = []
        if args.lmm:
            stream_models.append("lmm")
        if args.fastlmm:
            stream_models.append("fastlmm")
        if args.lm:
            stream_models.append("lm")
        has_lrlmm = bool(args.lrlmm is not None)
        has_farmcpu = bool(args.farmcpu)

        trait_order = list(pheno.columns) if pheno is not None else []
        trait_col_map: dict[str, int] = {}
        selected_ncol = []
        if pheno is not None:
            selected_ncol = pheno.attrs.get("selected_ncol", [])
        if isinstance(selected_ncol, list) and len(selected_ncol) == len(trait_order):
            for trait_name, col_idx in zip(trait_order, selected_ncol):
                trait_col_map[str(trait_name)] = int(col_idx)
        else:
            for idx0, trait_name in enumerate(trait_order):
                trait_col_map[str(trait_name)] = int(idx0)

        # -------------------------------
        # 1) Streaming models first
        # -------------------------------
        if len(stream_models) > 0:
            for trait_idx, pname in enumerate(trait_order):
                if len(stream_models) == 1:
                    model_key = stream_models[0]
                    pve_line_model = (
                        model_key if model_key in {"lmm", "fastlmm"} else None
                    )
                    run_chunked_gwas_lmm_lm(
                        model_name=model_key,
                        genofile=genofile_stream,
                        pheno=pheno,
                        ids=ids,
                        n_snps=n_snps,
                        outprefix=outprefix,
                        maf_threshold=args.maf,
                        max_missing_rate=args.geno,
                        genetic_model=args.model,
                        het_threshold=args.het,
                        chunk_size=args.chunksize,
                        mmap_limit=args.mmap_limit,
                        grm=grm,
                        qmatrix=qmatrix,
                        cov_all=cov_all,
                        eff_m=eff_m,
                        plot=args.plot,
                        threads=args.thread,
                        logger=logger,
                        use_spinner=use_spinner,
                        snps_only=bool(args.snps_only),
                        eff_snp_by_trait=eff_snp_by_trait,
                        summary_rows=gwas_summary_rows,
                        saved_paths=saved_result_paths,
                        trait_names=[str(pname)],
                        show_npve_line=True if pve_line_model is None else (model_key == pve_line_model),
                        emit_trait_header=True,
                    )
                else:
                    run_chunked_gwas_streaming_shared(
                        model_names=stream_models,
                        trait_name=str(pname),
                        genofile=genofile_stream,
                        pheno=pheno,
                        ids=ids,
                        n_snps=n_snps,
                        outprefix=outprefix,
                        maf_threshold=args.maf,
                        max_missing_rate=args.geno,
                        genetic_model=args.model,
                        het_threshold=args.het,
                        chunk_size=args.chunksize,
                        mmap_limit=args.mmap_limit,
                        grm=grm,
                        qmatrix=qmatrix,
                        cov_all=cov_all,
                        plot=args.plot,
                        threads=args.thread,
                        logger=logger,
                        use_spinner=use_spinner,
                        snps_only=bool(args.snps_only),
                        eff_snp_by_trait=eff_snp_by_trait,
                        summary_rows=gwas_summary_rows,
                        saved_paths=saved_result_paths,
                    )

                if trait_idx < len(trait_order) - 1:
                    logger.info("")

        # -------------------------------
        # 2) Full genotype task (LowRankLMM / FarmCPU)
        # -------------------------------
        if has_lrlmm or has_farmcpu:
            if len(stream_models) > 0:
                logger.info("")
            _section(logger, "Full genotype task")
            context_prepared = bool(pheno is not None and ids is not None and n_snps is not None)

            shared_full_cache: Union[dict[str, object], None] = None
            if has_lrlmm:
                shared_full_cache = run_farmcpu_fullmem(
                    args=args,
                    gfile=gfile,
                    prefix=prefix,
                    logger=logger,
                    pheno_preloaded=pheno,
                    ids_preloaded=ids,
                    n_snps_preloaded=n_snps,
                    qmatrix_preloaded=qmatrix,
                    cov_preloaded=cov_all,
                    use_spinner=use_spinner,
                    context_prepared=context_prepared,
                    summary_rows=gwas_summary_rows,
                    saved_paths=saved_result_paths,
                    trait_names=[str(t) for t in trait_order] if len(trait_order) > 0 else None,
                    farmcpu_cache=None,
                    prepare_only=True,
                    emit_trait_header=True,
                )
                if len(trait_order) == 0 and isinstance(shared_full_cache, dict):
                    pheno_cache = shared_full_cache.get("pheno")
                    if isinstance(pheno_cache, pd.DataFrame):
                        trait_order = [str(t) for t in list(pheno_cache.columns)]
                        trait_col_map = {str(t): int(i) for i, t in enumerate(trait_order)}

                _phase_split(logger)
                run_lrlmm_packed(
                    rank_opt=args.lrlmm,
                    genofile=gfile,
                    pheno=pheno if pheno is not None else pd.DataFrame(),
                    ids=ids if ids is not None else np.asarray([], dtype=str),
                    outprefix=outprefix,
                    maf_threshold=args.maf,
                    max_missing_rate=args.geno,
                    genetic_model=args.model,
                    chunk_size=args.chunksize,
                    mmap_limit=args.mmap_limit,
                    qmatrix=qmatrix if qmatrix is not None else np.zeros((0, 0), dtype=np.float32),
                    cov_all=cov_all,
                    plot=args.plot,
                    threads=args.thread,
                    logger=logger,
                    use_spinner=use_spinner,
                    snps_only=bool(args.snps_only),
                    eff_snp_by_trait=eff_snp_by_trait,
                    summary_rows=gwas_summary_rows,
                    saved_paths=saved_result_paths,
                    trait_names=[str(t) for t in trait_order] if len(trait_order) > 0 else None,
                    emit_trait_header=True,
                    prepared_cache=shared_full_cache,
                )

            if has_farmcpu:
                _ = run_farmcpu_fullmem(
                    args=args,
                    gfile=gfile,
                    prefix=prefix,
                    logger=logger,
                    pheno_preloaded=pheno,
                    ids_preloaded=ids,
                    n_snps_preloaded=n_snps,
                    qmatrix_preloaded=qmatrix,
                    cov_preloaded=cov_all,
                    use_spinner=use_spinner,
                    context_prepared=context_prepared,
                    summary_rows=gwas_summary_rows,
                    saved_paths=saved_result_paths,
                    trait_names=[str(t) for t in trait_order] if len(trait_order) > 0 else None,
                    farmcpu_cache=shared_full_cache,
                    emit_trait_header=(not has_lrlmm),
                )

        if len(gwas_summary_rows) > 0:
            for row in gwas_summary_rows:
                pnm = str(row.get("phenotype", ""))
                row["pheno_col_idx"] = int(trait_col_map.get(pnm, -1))
            _emit_gwas_summary(logger, gwas_summary_rows)
        ordered_result_paths = _ordered_saved_result_paths(
            gwas_summary_rows,
            saved_result_paths,
        )
        if len(ordered_result_paths) > 0:
            logger.info("")
            saved_body = "\n".join([f"  {p}" for p in ordered_result_paths])
            _rich_success(logger, f"Results saved:\n{saved_body}")
        # _rich_success(logger, f"  {str(log_path).replace('//', '/')}")

    except KeyboardInterrupt:
        run_status = "failed"
        logger.info("Interrupted by user (Ctrl+C).")
        # On Windows, worker threads/native kernels can continue briefly after
        # KeyboardInterrupt. Force-exit the Python process to prevent lingering
        # high CPU/RSS background jobs after Ctrl+C.
        if os.name == "nt":
            try:
                logger.info("Terminating all GWAS workers immediately on Windows.")
            except Exception:
                pass
            try:
                logging.shutdown()
            finally:
                os._exit(130)
    except Exception as e:
        run_status = "failed"
        run_error = str(e)
        logger.exception(f"Error in JanusX GWAS pipeline: {e}")

    if run_status == "done":
        try:
            genofile_kind = (
                "bfile" if bool(args.bfile)
                else ("vcf" if bool(args.vcf)
                      else ("hmp" if bool(args.hmp) else "tsv"))
            )
            args_data = {
                "models": {
                    "lmm": bool(args.lmm),
                    "fastlmm": bool(args.fastlmm),
                    "lrlmm": bool(args.lrlmm is not None),
                    "lm": bool(args.lm),
                    "farmcpu": bool(args.farmcpu),
                },
                "model": str(args.model),
                "lrlmm_rank": (None if args.lrlmm is None else int(args.lrlmm)),
                "snps_only": bool(args.snps_only),
                "maf": float(args.maf),
                "geno": float(args.geno),
                "het": float(args.het),
                "chunksize": int(args.chunksize),
                "thread": int(args.thread),
                "ncol": (list(args.ncol) if args.ncol is not None else None),
                "cov": (list(args.cov) if args.cov is not None else []),
                "grm": str(args.grm),
                "qcov": str(args.qcov),
                "grm_display": _format_grm_display(args.grm),
                "qcov_display": _format_qcov_display(args.qcov),
                "cov_display": _format_cov_display(args.cov),
                "traits": [str(x) for x in trait_order],
                "trait_col_map": {str(k): int(v) for k, v in trait_col_map.items()},
            }
            record_gwas_run(
                run_id=run_id,
                status=run_status,
                genofile=str(gfile),
                genofile_kind=genofile_kind,
                phenofile=str(args.pheno),
                outprefix=str(outprefix),
                log_file=str(log_path),
                result_files=[str(x) for x in ordered_result_paths],
                summary_rows=[dict(x) for x in _ordered_gwas_summary_rows(gwas_summary_rows)],
                args_data=args_data,
                error_text="",
                created_at=run_created_at,
            )
        except Exception as e:
            logger.warning(f"Failed to write GWAS history DB: {e}")

    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    _rich_success(logger, endinfo)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
