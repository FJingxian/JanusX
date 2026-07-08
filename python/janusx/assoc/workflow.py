# -*- coding: utf-8 -*-
"""
JanusX: High-Performance GWAS Command-Line Interface

Design overview
---------------
Models:
  - LMM     : streaming, low-memory implementation (slim.LMM)
  - LMM2    : exact LMM scan with Wald + per-SNP ML/PLRT output
  - LM      : streaming, low-memory implementation (slim.LM)
  - SparseLMM: sparse-GRM + sparse-Cholesky mixed-model route
  - ALGWAS  : packed/full-rust adaptive-lasso GWAS route
  - FarmCPU : Rust packed-BED controller with stage1/stage2 scan pipeline

Execution mode (automatic)
--------------------------
  - No explicit "low-memory" flag is required.
  - Default: LM/LMM/FvLMM run in memmap BED mode.
  - Packed BED preload is reserved for FarmCPU and ALGWAS.
  - LM/LMM/FvLMM/SparseLMM stay on memmap / streaming BED routes.
  - FarmCPU runs on the packed BED controller and no longer maintains a Python fallback.

Caching
-------
  - Genotype cache uses a single base cache prefix (no JSON cache metadata):
      * genotype cache (VCF/HMP): ~{geno_prefix}.bed/.bim/.fam
  - GRM and PCA caches for streaming LMM/LM runs remain parameterized:
      * GRM cache:      ~{geno_prefix}.maf{maf}.geno{geno}.cGRM.npy / sGRM.npy (+ .id)
      * PCA cache:      ~{geno_prefix}.maf{maf}.geno{geno}.cGRM.pc{q}.txt / sGRM.pc{q}.txt
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
import math
import time
import socket
import argparse
import logging
import re
import shlex
import sys
import threading
import warnings
import multiprocessing as mp
import concurrent.futures as cf
import textwrap
from shutil import get_terminal_size
from datetime import datetime
from typing import Any, Iterator, Union, Optional, Callable
import uuid
from contextlib import contextmanager

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

import numpy as np
import pandas as pd
try:
    from scipy.special import erfc as _sp_erfc
except Exception:
    _sp_erfc = None
import psutil
import janusx as jx_pkg
from janusx.gfreader import (
    load_genotype_chunks,
    inspect_genotype_file,
    prepare_cli_input_cache,
)
from janusx.pyBLUP.QK2 import QK
from janusx import janusx as jxrs
from janusx.script._common.log import setup_logging
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.cli_args import (
    add_common_covariate_file_or_site_arg,
    add_common_genotype_source_args,
    add_common_grm_option_arg,
    add_common_memory_arg,
    add_common_out_arg,
    add_common_pheno_arg,
    add_common_prefix_arg,
    add_common_snps_only_arg,
    add_common_thread_arg,
    add_common_trait_selector_args,
    add_common_variant_filter_args,
    parse_trait_selector_specs,
)
from janusx.script._common.genoio import (
    determine_genotype_source_force_kind_from_args,
    genotype_load_status_done,
    genotype_load_status_fail,
    genotype_load_status_open,
    genotype_load_status_progress,
    packed_preload_failure_state,
    packed_preload_is_disabled,
    packed_preload_is_ready,
)
from janusx.script._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    safe_expanduser,
    safe_resolve,
)
from janusx.script._common.progress import (
    CliStatus,
    is_skip_status_text,
    print_success,
    print_warning,
    format_elapsed,
    log_success,
    should_animate_status,
    stdout_is_tty,
)
from janusx.script._common.progress import build_rich_progress, rich_progress_available
from janusx.script._common.memory import (
    bed_block_target_env as _common_bed_block_target_env,
    decode_memory_gb_to_mb as _common_decode_memory_gb_to_mb,
    normalize_decode_memory_gb as _common_normalize_decode_memory_gb,
    resolve_decode_block_rows as _common_resolve_decode_block_rows,
    resolve_decode_mmap_window_mb as _common_resolve_decode_mmap_window_mb,
)
from janusx.script._common.threads import (
    apply_outer_thread_cap,
    detect_effective_threads,
    detect_thread_budget_info,
    format_affinity_cpu_summary,
    format_requested_thread_usage,
    format_thread_budget_summary,
    maybe_warn_non_openblas,
    runtime_thread_stage,
    require_openblas_by_default,
)

DEFAULT_BED_MEMORY_GB = 1.0
_GWAS_AUTO_MEM_MIN_GB = 0.125
_GWAS_AUTO_MEM_LM_TARGET_MB = 192.0
_GWAS_AUTO_MEM_GRM_BLOCK_ROWS = 4096
_GWAS_AUTO_MEM_COPY_DECODE_BLOCK_ROWS = 8192
_GWAS_AUTO_MEM_PACKED_BLOCK_ROWS = 4096
_GWAS_WORKING_BUFFERS_LM = 1
_GWAS_WORKING_BUFFERS_GRM = 2
_GWAS_WORKING_BUFFERS_PACKED = 2
_GWAS_WORKING_BUFFERS_COPY = 3
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.genoio import (
    basename_only as _basename_only,
    determine_genotype_source_from_args as determine_genotype_source,
    read_id_file as _read_id_file,
)
from janusx.script._common.grmstable import (
    save_grm_npy_blocked,
)
from janusx.script._common.grmio import (
    grm_load_status_done,
    grm_load_status_fail,
    grm_load_status_open,
    grm_text_materialized_message,
    load_or_materialize_square_grm_cache,
)


def _resolve_pyblup_assoc_symbol(name: str):
    from janusx.pyBLUP import assoc as _assoc

    return getattr(_assoc, str(name))


class _LazyPyBlupAssocSymbol:
    def __init__(self, name: str):
        self._name = str(name)

    def _resolve(self):
        return _resolve_pyblup_assoc_symbol(self._name)

    def __call__(self, *args, **kwargs):
        return self._resolve()(*args, **kwargs)

    def __getattr__(self, attr: str):
        return getattr(self._resolve(), attr)

    def __repr__(self) -> str:
        return f"<LazyPyBlupAssocSymbol {self._name}>"


LM = _LazyPyBlupAssocSymbol("LM")
LMM = _LazyPyBlupAssocSymbol("LMM")
LMM2 = _LazyPyBlupAssocSymbol("LMM2")
FvLMM = _LazyPyBlupAssocSymbol("FvLMM")
farmcpu = _LazyPyBlupAssocSymbol("farmcpu")

from janusx.assoc.workflow_ui import (
    _format_progress_metric,
    _log_file_only,
    _log_info,
    _emit_plain_info_line,
    _emit_warning_line,
    _log_model_line,
    _rich_success,
    _ProgressAdapter,
    _progress_callback_step,
    _start_indeterminate_progress_bar,
    _stop_indeterminate_progress_bar,
    fastplot,
    _run_fastplot_with_status,
    _run_fastplot_from_tsv_with_status,
    _run_result_write_with_status,
)
from janusx.assoc.workflow_cache import (
    _dedupe_cache_warning_messages,
    _gwas_cache_prefix_with_params,
    _grm_cache_paths,
    _grm_cache_paths_legacy,
    _is_cache_warning_message,
    _pca_cache_path,
    _pca_cache_path_legacy,
    _cache_lock,
    genotype_cache_prefix,
)


def _write_pca_cache_with_ids(path: str, sample_ids: np.ndarray, qmatrix: np.ndarray) -> None:
    sid = np.asarray(sample_ids, dtype=str).reshape(-1)
    q = np.asarray(qmatrix, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if q.shape[0] != sid.shape[0]:
        raise ValueError(
            f"PCA cache write shape mismatch: ids={sid.shape[0]}, rows={q.shape[0]}"
        )
    df = pd.DataFrame(q, columns=[f"PC{i + 1}" for i in range(int(q.shape[1]))])
    df.insert(0, "sample_id", sid)
    df.to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
        float_format="%.6f",
    )


def _load_pca_cache_with_ids(
    path: str,
    *,
    expected_rows: Union[int, None],
    expected_dim: int,
    expected_ids: Union[np.ndarray, None] = None,
) -> tuple[np.ndarray, np.ndarray]:
    tab = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        keep_default_na=False,
    )
    expected_cols = int(expected_dim) + 1
    if tab.shape[1] != expected_cols:
        raise ValueError(
            f"PCA cache column mismatch: expected {expected_cols} columns (ID + {int(expected_dim)} PCs), got {tab.shape[1]} ({path})"
        )
    ids = tab.iloc[:, 0].astype(str).to_numpy(dtype=str)
    qmatrix = tab.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    if qmatrix.ndim == 1:
        qmatrix = qmatrix.reshape(-1, 1)
    row_n = int(ids.shape[0])
    if row_n == 0:
        raise ValueError(f"PCA cache is empty: {path}")
    if qmatrix.shape != (row_n, int(expected_dim)):
        raise ValueError(
            f"PCA cache shape mismatch: expected (*,{int(expected_dim)}) with ID column, got ids={ids.shape[0]}, q={qmatrix.shape}"
        )
    if len(set(ids.tolist())) != row_n:
        raise ValueError(f"PCA cache contains duplicated sample IDs: {path}")
    if expected_ids is None and expected_rows is not None and row_n != int(expected_rows):
        raise ValueError(
            f"PCA cache row-count mismatch: expected {int(expected_rows)}, got {row_n} ({path})"
        )
    if expected_ids is not None:
        expect = set(np.asarray(expected_ids, dtype=str).reshape(-1).tolist())
        overlap = sum(1 for sid in ids.tolist() if sid in expect)
        if overlap <= 0:
            raise ValueError(
                f"PCA cache has no overlapping sample IDs with current genotype: {path}"
            )
    if not np.all(np.isfinite(qmatrix)):
        raise ValueError(f"PCA cache contains non-finite values: {path}")
    return ids, qmatrix

_SECTION_WIDTH = 80
_REPORT_RULE = "=" * _SECTION_WIDTH
_REPORT_SUBRULE = "-" * _SECTION_WIDTH
_GWAS_PROGRESS_BAR_WIDTH = 30
_GWAS_PCA_GRM_EIGH_SAMPLE_THRESHOLD = 15_000


def _format_farmcpu_threshold_display(v: object) -> str:
    if v is None:
        return "auto(1/nsnp)"
    try:
        fv = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(fv) or fv <= 0:
        return "auto(1/nsnp)"
    return f"{fv:g}"

# ======================================================================
# Basic utilities
# ======================================================================

def _env_truthy(name: str, default: str = "0") -> bool:
    raw = str(os.environ.get(name, default)).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _gwas_plot_enabled() -> bool:
    return _env_truthy("JX_GWAS_PLOT", "1")


def _gwas_logger_verbose(logger: Optional[logging.Logger]) -> bool:
    try:
        return bool(getattr(logger, "_janusx_gwas_verbose"))
    except Exception:
        return False


def _gwas_terminal_rich(logger: Optional[logging.Logger]) -> bool:
    try:
        return bool(getattr(logger, "_janusx_gwas_terminal_rich"))
    except Exception:
        return False


def _gwas_report_logger(logger: logging.Logger) -> logging.Logger:
    try:
        rep = getattr(logger, "_janusx_gwas_report_logger")
        if isinstance(rep, logging.Logger):
            return rep
    except Exception:
        pass
    return logger


def _gwas_stream_logger(logger: logging.Logger) -> Optional[logging.Logger]:
    try:
        stream_logger = getattr(logger, "_janusx_gwas_stream_logger")
        if isinstance(stream_logger, logging.Logger):
            return stream_logger
    except Exception:
        pass
    return None


def _format_cli_finished_timestamp(ts: Optional[float] = None) -> str:
    lt = time.localtime(time.time() if ts is None else float(ts))
    return (
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour:02d}:{lt.tm_min:02d}:{lt.tm_sec:02d}"
    )


def _gwas_terminal_rule_width(
    default: int = 60,
    *,
    logger: Optional[logging.Logger] = None,
) -> int:
    if logger is not None:
        try:
            w = int(getattr(logger, "_janusx_cli_config_panel_width"))
            if w > 0:
                return w
        except Exception:
            pass
    if not stdout_is_tty():
        return int(max(20, default))
    try:
        cols = int(get_terminal_size((int(default), 20)).columns)
    except Exception:
        cols = int(default)
    return int(max(40, min(120, cols)))


def _gwas_terminal_config_line_max_chars(default: int = 60) -> int:
    if not stdout_is_tty():
        return int(max(20, default))
    try:
        cols = int(get_terminal_size((int(default) + 12, 20)).columns)
    except Exception:
        cols = int(default)
    # Leave room for panel borders/padding while still expanding on wider terminals.
    return int(max(40, min(120, cols - 10)))


def _gwas_invocation_command(argv: Optional[list[str]] = None) -> str:
    tokens = [str(x) for x in (sys.argv[1:] if argv is None else argv)]
    prog_raw = str(sys.argv[0]).strip() if len(sys.argv) > 0 else ""
    if prog_raw == "":
        prog_raw = "jx gwas"
    try:
        prog_parts = shlex.split(prog_raw)
    except Exception:
        prog_parts = [prog_raw]
    if len(prog_parts) == 0:
        prog_parts = ["jx", "gwas"]
    return shlex.join([str(x) for x in (prog_parts + tokens)])


def _emit_report_banner(logger: logging.Logger, title: str) -> None:
    logger.info(_REPORT_RULE)
    logger.info(str(title))
    logger.info(_REPORT_RULE)
    logger.info("")


def _emit_report_block(logger: logging.Logger, title: str) -> None:
    logger.info(f"[ {str(title)} ]")


def _emit_report_major_section(logger: logging.Logger, title: str) -> None:
    logger.info("")
    logger.info(_REPORT_RULE)
    logger.info(f"[ {str(title)} ]")
    logger.info(_REPORT_SUBRULE)


def _emit_report_kv(logger: logging.Logger, key: str, value: object, *, key_width: int = 16) -> None:
    logger.info(f"  {str(key):<{int(key_width)}} : {str(value)}")


def _normalize_gwas_model_name(model_name: object) -> str:
    return str(model_name or "").strip()


def _format_gwas_models_executed(args) -> str:
    models: list[str] = []
    if bool(args.lm):
        models.append("LM")
    if bool(getattr(args, "lm2", None) is not None):
        lm2_selector = list(getattr(args, "lm2_cov_cols", []) or [])
        lm2_missing_external_cov = not bool(getattr(args, "cov", None))
        lm2_will_fallback = bool(getattr(args, "_lm2_fallback_to_lm", False)) or (
            len(lm2_selector) == 0
        ) or lm2_missing_external_cov
        if lm2_will_fallback:
            models.append("LM2->LM")
        else:
            models.append("LM2")
    if bool(args.lmm):
        models.append("LMM")
    if bool(getattr(args, "lmm2", False)):
        models.append("LMM2")
    if bool(args.fvlmm):
        models.append("FvLMM")
    if bool(args.splmm):
        models.append("SparseLMM")
    if bool(args.farmcpu):
        models.append("FarmCPU")
    if bool(args.algwas):
        models.append("ALGWAS")
    return ", ".join(models) if len(models) > 0 else "None"


def _parse_lm2_covariate_selector(
    value: object,
    *,
    label: str = "-lm2/--lm2",
) -> list[int]:
    raw = str(value).strip() if value is not None else ""
    if raw == "" or raw == "__SELF__":
        return []

    picks: list[int] = []
    seen: set[int] = set()

    def _push(idx0: int) -> None:
        if idx0 < 0:
            raise ValueError(
                f"{label}: covariate column indices must be >= 0, got {idx0}."
            )
        if idx0 not in seen:
            seen.add(idx0)
            picks.append(idx0)

    for token in [str(x).strip() for x in raw.split(",") if str(x).strip() != ""]:
        if ":" in token:
            left, right = token.split(":", 1)
            left = left.strip()
            right = right.strip()
            if left == "" and right == "":
                raise ValueError(
                    f"{label}: invalid empty covariate range '{token}'."
                )
            if left == "":
                start = 0
            else:
                if not left.lstrip("+-").isdigit():
                    raise ValueError(
                        f"{label}: invalid covariate range start '{left}'."
                    )
                start = int(left)
            if right == "":
                raise ValueError(
                    f"{label}: open-ended covariate range '{token}' is not supported; provide an explicit end."
                )
            if not right.lstrip("+-").isdigit():
                raise ValueError(
                    f"{label}: invalid covariate range end '{right}'."
                )
            end = int(right)
            step = 1 if end >= start else -1
            for idx1 in range(start, end + step, step):
                _push(idx1)
            continue
        if not token.lstrip("+-").isdigit():
            raise ValueError(
                f"{label}: invalid covariate selector '{token}'. Use 0-based indices/ranges like 0, 0:3, :2, 0,3."
            )
        _push(int(token))
    return picks


def _resolve_lm2_covariate_indices(
    cov_all,
    selector_zero_based: list[int],
    *,
    label: str = "-lm2/--lm2",
) -> np.ndarray:
    arr = np.asarray(cov_all, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("LM2 requires a 2D merged covariate matrix.")
    ncov = int(arr.shape[1])
    if ncov <= 0:
        raise ValueError("LM2 requires at least one covariate column from -c.")
    if len(selector_zero_based) == 0:
        return np.asarray([], dtype=np.int64)
    bad = [int(i) for i in selector_zero_based if int(i) < 0 or int(i) >= ncov]
    if bad:
        raise ValueError(
            f"{label}: covariate column index out of range: {bad[0]}. valid=[0..{ncov - 1}]"
        )
    return np.asarray(selector_zero_based, dtype=np.int64)


def _format_gwas_packed_route_status(
    *,
    farmcpu_auto_fast: bool,
    algwas_auto_fast: bool,
) -> str:
    auto_tags: list[str] = []
    if bool(farmcpu_auto_fast):
        auto_tags.append("FarmCPU Auto-selected")
    if bool(algwas_auto_fast):
        auto_tags.append("ALGWAS Auto-selected")
    if len(auto_tags) > 0:
        return f"Enabled ({'; '.join(auto_tags)})"
    return "Disabled"


def _format_gwas_grm_status(grm, grm_arg: object) -> str:
    if grm is None:
        return "Not loaded"
    try:
        arr = np.asarray(grm)
        if arr.ndim == 2:
            shape_text = f"{int(arr.shape[0])}x{int(arr.shape[1])}"
        else:
            shape_text = str(tuple(int(x) for x in arr.shape))
    except Exception:
        shape_text = "Loaded"
    grm_opt = str(grm_arg).strip()
    if grm_opt == "1":
        src = "Centered GRM"
    elif grm_opt == "2":
        src = "Standardized GRM"
    else:
        src = "Reused"
    return f"{shape_text} ({src})"


def _format_gwas_q_status(qmatrix) -> str:
    if qmatrix is None:
        return "Not loaded"
    try:
        q_arr = np.asarray(qmatrix)
        if q_arr.ndim != 2:
            return "Invalid"
        qdim = int(q_arr.shape[1])
        if qdim <= 0:
            return "Dimension 0 (Empty)"
        return f"Dimension {qdim}"
    except Exception:
        return "Unknown"


def _format_gwas_cov_status(cov_all) -> str:
    if cov_all is None:
        return "None"
    try:
        arr = np.asarray(cov_all)
        if arr.ndim != 2 or int(arr.shape[1]) <= 0:
            return "None"
        return f"{int(arr.shape[1])} columns"
    except Exception:
        return "Loaded"


def _format_gwas_sparse_status(
    sparse_grm_path: Optional[str],
    *,
    sparse_requested: bool,
    sparse_cutoff: Optional[float],
    sparse_source: object,
) -> str:
    if not bool(sparse_requested):
        return "Not requested"
    if sparse_grm_path is None or str(sparse_grm_path).strip() == "":
        cutoff_text = "NA" if sparse_cutoff is None else f"{float(sparse_cutoff):g}"
        return f"Requested (cutoff={cutoff_text})"
    source = str(sparse_source).strip()
    src_label = "Reused" if source not in {"", "1", "2"} else "Prepared"
    if src_label == "Reused":
        cutoff_text = " | cutoff=precomputed"
    else:
        cutoff_text = "" if sparse_cutoff is None else f" | cutoff={float(sparse_cutoff):g}"
    return f"{src_label} ({_display_path(str(sparse_grm_path))}{cutoff_text})"


def _format_gwas_thread_plan(
    args,
    fvlmm_scan_spec: Optional[dict[str, int]],
) -> str:
    scan_stage = _resolve_fvlmm_scan_stage_mode()
    if fvlmm_scan_spec is None:
        blas_threads = int(args.thread)
        rayon_threads = int(args.thread)
    else:
        blas_threads = int(fvlmm_scan_spec.get("blas_threads", args.thread))
        rayon_threads = int(fvlmm_scan_spec.get("rayon_threads", args.thread))
    return (
        f"FvLMM_scan_stage={scan_stage} | "
        f"BLAS={blas_threads} | Rayon={rayon_threads}"
    )


def _section(logger:logging.Logger, title: str) -> None:
    """Emit a formatted log section header with a leading blank line."""
    rule = "=" * _gwas_terminal_rule_width(_SECTION_WIDTH, logger=logger)
    logger.info("")
    logger.info(rule)
    logger.info(title)
    logger.info(rule)


def _phase_split(logger: logging.Logger) -> None:
    """Visual separator between loading stage and compute stage."""
    logger.info("-" * _gwas_terminal_rule_width(_SECTION_WIDTH, logger=logger))


def _set_pending_gwas_overlap_line(
    logger: Optional[logging.Logger],
    line: Optional[str],
) -> None:
    if logger is None:
        return
    try:
        setattr(
            logger,
            "_janusx_gwas_pending_overlap_line",
            (None if line is None else str(line)),
        )
    except Exception:
        pass


def _flush_pending_gwas_overlap_line(
    logger: Optional[logging.Logger],
    *,
    use_spinner: bool,
) -> None:
    if logger is None:
        return
    try:
        line = getattr(logger, "_janusx_gwas_pending_overlap_line", None)
    except Exception:
        line = None
    if not line:
        return
    line_text = str(line)
    stream_logger = _gwas_stream_logger(logger)
    if bool(use_spinner) and _gwas_terminal_rich(logger) and stream_logger is not None:
        _log_file_only(logger, logging.INFO, line_text)
        stream_logger.info(line_text)
    else:
        _emit_plain_info_line(logger, line_text, use_spinner=bool(use_spinner))
    _set_pending_gwas_overlap_line(logger, None)


def _replace_file_with_retry(
    src: str,
    dst: str,
    *,
    attempts: int = 40,
    base_sleep_s: float = 0.05,
) -> None:
    """
    Atomic replace with short retries for transient Windows file locks.

    On Windows, antivirus/indexers or delayed file-handle release can trigger
    PermissionError(winerror=32/5) for a short period.
    """
    n = max(1, int(attempts))
    for i in range(n):
        try:
            os.replace(src, dst)
            return
        except PermissionError as ex:
            winerr = int(getattr(ex, "winerror", 0) or 0)
            if os.name == "nt" and winerr in {5, 32} and (i + 1) < n:
                time.sleep(float(base_sleep_s) * float(min(i + 1, 10)))
                continue
            raise


def _gwas_result_tmp_path(out_tsv: str) -> str:
    return f"{str(out_tsv)}.tmp.{os.getpid()}.{uuid.uuid4().hex}"


def _cleanup_gwas_result_tmp(tmp_tsv: str) -> None:
    path = str(tmp_tsv).strip()
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _mixed_model_switch_to_lm_decision(
    *,
    y_vec: np.ndarray,
    x_cov: np.ndarray,
    lmm_ml0: Optional[float],
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """
    Decide whether a mixed-model null fit should collapse to LM.

    This compares the null-model log-likelihood against the plain LM null
    model under the boundary test H0: Va = 0.
    """
    ml0 = None
    try:
        if lmm_ml0 is not None:
            ml0_tmp = float(lmm_ml0)
            if np.isfinite(ml0_tmp):
                ml0 = ml0_tmp
    except Exception:
        ml0 = None

    if ml0 is None:
        raise RuntimeError(
            "Mixed-model -> LM switch decision requires finite null ML (ml0)."
        )
    if not hasattr(jxrs, "gwas_lmm_lm_null_lrt_decision"):
        raise RuntimeError(
            "Rust symbol gwas_lmm_lm_null_lrt_decision is unavailable."
        )
    y_arr = np.ascontiguousarray(np.asarray(y_vec, dtype=np.float64).reshape(-1))
    x_arr = np.ascontiguousarray(np.asarray(x_cov, dtype=np.float64))
    switch_to_lm, lrt_stat, pval, _lm_ml0 = jxrs.gwas_lmm_lm_null_lrt_decision(
        y_arr,
        x_arr,
        float(ml0),
        float(alpha),
        True,
    )
    return bool(switch_to_lm), float(lrt_stat), float(pval)


def _chi2_sf_df1_vec(stat: np.ndarray) -> np.ndarray:
    """
    Survival function for Chi-square(df=1), vectorized.
    """
    x = np.asarray(stat, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(x) & (x >= 0.0)
    if not np.any(ok):
        return out
    z = np.sqrt(0.5 * x[ok])
    if _sp_erfc is not None:
        p = np.asarray(_sp_erfc(z), dtype=np.float64)
    else:
        p = np.fromiter((math.erfc(float(v)) for v in np.asarray(z, dtype=np.float64)), dtype=np.float64)
    out[ok] = np.clip(p, np.finfo(np.float64).tiny, 1.0)
    return out

def _gwas_use_packed_grm_build() -> bool:
    """
    Whether GRM construction should prefer packed BED + Rust CBLAS route.

    Default is True. In current GWAS policy, packed GRM reuse is reserved for
    packed-only model routes; standard LM/LMM/FvLMM/SparseLMM scans
    stay on memmap/streaming backends.
    """
    raw = str(os.environ.get("JX_GWAS_USE_PACKED_GRM", "")).strip().lower()
    if raw == "":
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return raw in {"1", "true", "yes", "y", "on"}


def _emit_trait_header(
    logger: logging.Logger,
    trait_name: str,
    n_idv: int,
    *,
    pve: Optional[float] = None,
    use_spinner: bool = False,
    width: int = 60,
) -> None:
    trait = str(trait_name)
    pve_val: Optional[float] = None
    if pve is not None:
        try:
            pve_tmp = float(pve)
            if np.isfinite(pve_tmp):
                pve_val = pve_tmp
        except Exception:
            pve_val = None
    report_logger = _gwas_report_logger(logger)
    _emit_report_major_section(report_logger, f"Task Execution: {trait}")
    sample_text = f"{int(n_idv)}"
    if pve_val is not None:
        sample_text = f"{sample_text} | Null PVE(pheno-scale)={pve_val:.3f}"
    _emit_report_kv(report_logger, "Trait Samples", sample_text)
    context_rows = getattr(logger, "_janusx_gwas_task_context_rows", None)
    if isinstance(context_rows, list):
        for item in context_rows:
            if not (isinstance(item, tuple) and len(item) == 2):
                continue
            _emit_report_kv(report_logger, str(item[0]), item[1])

    if _gwas_terminal_rich(logger):
        stream_logger = _gwas_stream_logger(logger)
        if stream_logger is not None:
            if pve_val is None:
                n_line = f"(n={int(n_idv)})"
            else:
                n_line = f"(n={int(n_idv)}; pve(pheno)={pve_val:.3f})"
            prefix = "* "
            full = f"{prefix}{trait} {n_line}"
            if len(full) <= int(width):
                lines = [full]
            else:
                if (len(prefix) + len(trait)) <= int(width):
                    lines = [f"{prefix}{trait}", n_line]
                else:
                    trait_lines = textwrap.wrap(
                        trait,
                        width=max(10, int(width) - len(prefix)),
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                    if len(trait_lines) > 0:
                        trait_lines[0] = f"{prefix}{trait_lines[0]}"
                    else:
                        trait_lines = [prefix.strip()]
                    lines = trait_lines + [n_line]
            for ln in lines:
                stream_logger.info(ln)


def _emit_gwas_summary_legacy(
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
        "pve(ph)",
        "mem(G)",
        "Ctime(s)",
        "Vtime(s)",
    ]

    out_rows: list[list[str]] = []
    for r in _ordered_gwas_summary_rows(rows):
        model_text = _normalize_gwas_model_name(r.get("model", ""))
        pve_raw = r.get("pve", None)
        pve_text = "NA"
        try:
            if pve_raw is not None:
                pve_val = float(pve_raw)
                if np.isfinite(pve_val):
                    pve_text = f"{pve_val:.3f}"
        except Exception:
            pve_text = "NA"
        peak_rss = r.get("peak_rss_gb", None)
        peak_footprint = r.get("peak_footprint_gb", None)
        mem_gb = peak_footprint if peak_footprint is not None else peak_rss
        out_rows.append(
            [
                str(r.get("phenotype", "")),
                model_text,
                f"{int(r.get('nidv', 0))}",
                f"{int(r.get('eff_snp', 0))}",
                pve_text,
                (
                    f"{float(mem_gb):.2f}"
                    if mem_gb is not None and np.isfinite(float(mem_gb))
                    else "NA"
                ),
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


def _emit_gwas_summary(
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> None:
    if len(rows) == 0:
        return

    report_logger = _gwas_report_logger(logger)
    _emit_report_major_section(report_logger, "Summary Report")
    headers = [
        "Trait",
        "Model",
        "N_Indv",
        "N_SNP",
        "PVE(ph)",
        "Mem(GB)",
        "C-Time(s)",
        "V-Time(s)",
    ]

    out_rows: list[list[str]] = []
    for r in _ordered_gwas_summary_rows(rows):
        model_text = _normalize_gwas_model_name(r.get("model", ""))
        pve_raw = r.get("pve", None)
        pve_text = "NA"
        try:
            if pve_raw is not None:
                pve_val = float(pve_raw)
                if np.isfinite(pve_val):
                    pve_text = f"{pve_val:.3f}"
        except Exception:
            pve_text = "NA"
        peak_rss = r.get("peak_rss_gb", None)
        peak_footprint = r.get("peak_footprint_gb", None)
        mem_gb = peak_footprint if peak_footprint is not None else peak_rss
        out_rows.append(
            [
                str(r.get("phenotype", "")),
                model_text,
                f"{int(r.get('nidv', 0))}",
                f"{int(r.get('eff_snp', 0))}",
                pve_text,
                (
                    f"{float(mem_gb):.2f}"
                    if mem_gb is not None and np.isfinite(float(mem_gb))
                    else "NA"
                ),
                f"{float(r.get('gwas_time_s', 0.0)):.1f}",
                f"{float(r.get('viz_time_s', 0.0)):.1f}",
            ]
        )

    widths = [len(h) for h in headers]
    for row in out_rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(v))

    header_line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    report_logger.info(header_line)
    for row in out_rows:
        report_logger.info("  ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    report_logger.info(_REPORT_SUBRULE)

    if _gwas_terminal_rich(logger):
        stream_logger = _gwas_stream_logger(logger)
        if stream_logger is not None:
            _emit_gwas_summary_legacy(stream_logger, rows)


def _build_gwas_trait_summary_lines(
    rows: list[dict[str, object]],
) -> list[str]:
    if len(rows) == 0:
        return []
    headers = ["Model", "N_Indv", "N_SNP", "PVE(ph)", "time(s)", "Mem(GB)"]
    out_rows: list[list[str]] = []
    for r in sorted(list(rows), key=lambda x: _gwas_model_sort_key(x.get("model", ""))):
        model_text = _normalize_gwas_model_name(r.get("model", ""))
        pve_raw = r.get("pve", None)
        pve_text = "NA"
        try:
            if pve_raw is not None:
                pve_val = float(pve_raw)
                if np.isfinite(pve_val):
                    pve_text = f"{pve_val:.3f}"
        except Exception:
            pve_text = "NA"
        peak_rss = r.get("peak_rss_gb", None)
        peak_footprint = r.get("peak_footprint_gb", None)
        mem_gb = peak_footprint if peak_footprint is not None else peak_rss
        total_time = float(r.get("gwas_time_s", 0.0)) + float(r.get("viz_time_s", 0.0))
        out_rows.append(
            [
                str(model_text),
                f"{int(r.get('nidv', 0))}",
                f"{int(r.get('eff_snp', 0))}",
                str(pve_text),
                f"{float(total_time):.1f}",
                (
                    f"{float(mem_gb):.2f}"
                    if mem_gb is not None and np.isfinite(float(mem_gb))
                    else "NA"
                ),
            ]
        )
    widths = [len(h) for h in headers]
    for row in out_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    header_line = " ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    rule = "-" * max(len(header_line), 60)
    lines = [rule, header_line, rule]
    for row in out_rows:
        lines.append(" ".join(row[i].ljust(widths[i]) for i in range(len(row))))
    return lines


def _emit_gwas_trait_summary(
    logger: logging.Logger,
    rows: list[dict[str, object]],
) -> None:
    lines = _build_gwas_trait_summary_lines(rows)
    if len(lines) == 0:
        return
    targets: list[logging.Logger] = []
    report_logger = _gwas_report_logger(logger)
    targets.append(report_logger)
    if _gwas_terminal_rich(logger):
        stream_logger = _gwas_stream_logger(logger)
        if isinstance(stream_logger, logging.Logger) and id(stream_logger) != id(report_logger):
            targets.append(stream_logger)
    for target in targets:
        for line in lines:
            target.info(line)


def _align_pheno_to_sample_order(
    pheno: pd.DataFrame,
    sample_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Align phenotype rows to genotype sample order once, then reuse for fast slicing.
    """
    ids = np.asarray(sample_ids, dtype=str).reshape(-1)
    sid_index = pd.Index(ids, dtype=str)
    if pheno.index.equals(sid_index):
        return pheno, ids
    return pheno.reindex(sid_index), ids


def _looks_sample_header_token(token: object) -> bool:
    text = str(token).strip().lower()
    if text == "":
        return False
    norm = "".join(ch for ch in text if ch.isalnum())
    return norm in {
        "sampleid",
        "sample",
        "id",
        "iid",
        "fid",
        "taxa",
        "accession",
        "line",
    }


def _trait_values_and_mask(
    pheno_aligned: pd.DataFrame,
    trait_name: object,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return trait values in aligned sample order and a non-missing mask.
    """
    col_key: object
    if trait_name in pheno_aligned.columns:
        col_key = trait_name
    else:
        tkey = str(trait_name)
        if tkey in pheno_aligned.columns:
            col_key = tkey
        else:
            matched = [c for c in pheno_aligned.columns if str(c) == tkey]
            if len(matched) == 1:
                col_key = matched[0]
            elif len(matched) > 1:
                raise KeyError(
                    f"Ambiguous trait selector '{tkey}': multiple columns share this string form."
                )
            else:
                raise KeyError(
                    f"Trait '{tkey}' not found in phenotype columns: "
                    + ", ".join(str(c) for c in list(pheno_aligned.columns)[:10])
                    + (" ..." if len(pheno_aligned.columns) > 10 else "")
                )
    col = pheno_aligned[col_key]
    if pd.api.types.is_numeric_dtype(col.dtype):
        y_full = col.to_numpy(dtype=np.float64, copy=False)
    else:
        y_full = pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    keep = ~np.isnan(y_full)
    return y_full, keep


def _resolve_trait_iter(
    pheno_aligned: pd.DataFrame,
    trait_names: Optional[list[object]],
) -> list[object]:
    """
    Resolve requested trait selectors to actual phenotype column keys.

    Rust task dispatch serializes trait names as strings, while phenotype
    tables loaded through some `-n/--n` paths may still carry integer column
    labels. Accept either form and preserve the real column key for downstream
    access.
    """
    cols = list(pheno_aligned.columns)
    if trait_names is None:
        return cols

    requested = list(trait_names)
    resolved: list[object] = []
    missing: list[str] = []
    by_str: dict[str, list[object]] = {}
    for col in cols:
        by_str.setdefault(str(col), []).append(col)

    for trait_name in requested:
        if trait_name in pheno_aligned.columns:
            resolved.append(trait_name)
            continue
        tkey = str(trait_name)
        if tkey in pheno_aligned.columns:
            resolved.append(tkey)
            continue
        matched = by_str.get(tkey, [])
        if len(matched) == 1:
            resolved.append(matched[0])
            continue
        if len(matched) > 1:
            raise KeyError(
                f"Ambiguous trait selector '{tkey}': multiple columns share this string form."
            )
        missing.append(tkey)

    if len(requested) > 0 and len(resolved) == 0:
        raise KeyError(
            "Requested traits not found in phenotype columns: "
            + ", ".join(missing[:10])
            + (
                " ..."
                if len(missing) > 10
                else ""
            )
        )
    return resolved


def _safe_trait_file_label(label: object) -> str:
    """
    Convert phenotype labels to filesystem-safe filename tokens.

    Keep original trait labels for logs and summaries, but avoid path
    separators and reserved filename characters in output files.
    """
    text = str(label).strip()
    if text == "":
        return "trait"
    text = re.sub(r'[<>:"/\\\\|?*]+', "_", text)
    text = text.rstrip(". ").strip()
    return text if text != "" else "trait"


def _gwas_model_sort_key(model_name: object) -> tuple[int, str]:
    model = str(model_name or "").strip()
    order = {
        "LM": 0,
        "LMM": 1,
        "LMM2": 2,
        "FvLMM": 3,
        "SparseLMM": 4,
        "Farm": 5,
        "FarmCPU": 5,
        "ALGWAS": 6,
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


def _terminal_saved_result_paths(paths: list[str]) -> list[str]:
    hidden_suffixes = (
        ".algwas.stage1",
        ".algwas.qtn",
        ".farmcpu.qtn",
    )
    out: list[str] = []
    for path in paths:
        p = str(path).strip()
        if p == "":
            continue
        if any(p.endswith(suf) for suf in hidden_suffixes):
            continue
        out.append(p)
    return out


def _display_path(path: str) -> str:
    """
    Render path separators according to current OS style for terminal output.
    Windows: backslash, Unix-like: slash.
    """
    p = str(path).strip()
    if p == "":
        return p
    if os.name == "nt":
        out = p.replace("/", "\\")
    else:
        out = p.replace("\\", "/")
    try:
        out = os.path.normpath(out)
    except Exception:
        pass
    return out


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
    if low.endswith(".bin"):
        return path[: -len(".bin")], path
    for ext in (".txt", ".tsv", ".csv"):
        if low.endswith(ext):
            return path[: -len(ext)], path
    for ext in (".npy", ".bin", ".txt", ".tsv", ".csv"):
        cand = f"{path}{ext}"
        if os.path.isfile(cand):
            return path, cand
    return None, None


def _file_input_sidecars(prefix: str) -> list[str]:
    return [
        f"{prefix}.bin.id",
        f"{prefix}.id",
        f"{prefix}.bin.site",
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
    if s == "1":
        return "cGRM"
    if s == "2":
        return "sGRM"
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
    ncol: Union[list[object], None],
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

    requested_specs: Union[list[object], None] = None
    ncol_requested: Union[list[int], None] = None
    if ncol is not None:
        requested_specs = list(ncol)
        if all(isinstance(i, int) for i in requested_specs):
            ncol_requested = [int(i) for i in requested_specs]

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
    value_col_idx = [i for i in range(int(df.shape[1])) if i != int(id_col)]
    if int(id_col) == 1:
        value_col_idx = [i for i in value_col_idx if i != 0]
    if df.shape[0] > 1 and len(value_col_idx) > 0:
        row0 = pd.to_numeric(df.iloc[0, value_col_idx], errors="coerce")
        probe_stop = min(int(df.shape[0]), 33)
        probe_rows = df.iloc[1:probe_stop, value_col_idx].apply(pd.to_numeric, errors="coerce")
        probe_has_numeric = bool(probe_rows.notna().to_numpy().any())
        id_header_like = _looks_sample_header_token(df.iloc[0, int(id_col)])
        if id_header_like or (row0.isna().all() and probe_has_numeric):
            header_like = True
            header_names = df.iloc[0, value_col_idx].astype(str).tolist()
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
    # Keep first-seen sample order while averaging duplicated IDs.
    pheno = pheno.groupby(pheno.index, sort=False).mean(numeric_only=True)
    selected_ncol: list[int] = list(range(pheno.shape[1]))

    if pheno.shape[1] <= 0:
        msg = (
            "No phenotype data found. Please check the phenotype file format.\n"
            f"{pheno.head()}"
        )
        logger.error(msg)
        raise ValueError(msg)

    if ncol is not None:
        if requested_specs is None:
            requested_specs = list(ncol)

        if len(requested_specs) == 0:
            msg = (
                "No phenotype selector was provided for -n/--n. "
                "Use zero-based indices, ranges, or column names, e.g. "
                "-n 0 -n 3 or -n TraitA."
            )
            logger.error(msg)
            raise ValueError(msg)

        ncol_take: list[int] = []
        selected_ncol = []
        invalid_specs: list[str] = []
        seen_local: set[int] = set()

        if usecols is not None and ncol_requested is not None and int(id_col) in (0, 1):
            offset = 1 if int(id_col) == 0 else 2
            selected_file_cols = [int(c) for c in usecols if int(c) != int(id_col)]
            if int(id_col) == 1:
                selected_file_cols = [c for c in selected_file_cols if c != 0]
            selected_global_ncol = [int(c) - offset for c in selected_file_cols]
            global_to_local = {g: i for i, g in enumerate(selected_global_ncol)}
            for spec in requested_specs:
                idx = int(spec)
                if idx not in global_to_local:
                    invalid_specs.append(str(spec))
                    continue
                local_idx = int(global_to_local[idx])
                if local_idx in seen_local:
                    continue
                seen_local.add(local_idx)
                ncol_take.append(local_idx)
                selected_ncol.append(idx)
        else:
            cols = list(pheno.columns)
            by_str: dict[str, list[int]] = {}
            for i, col in enumerate(cols):
                by_str.setdefault(str(col), []).append(i)

            def _resolve_name_to_index(name: object) -> Optional[int]:
                exact = [i for i, col in enumerate(cols) if col == name]
                if len(exact) == 1:
                    return int(exact[0])
                if len(exact) > 1:
                    raise ValueError(
                        f"Ambiguous phenotype selector '{name}': multiple columns match exactly."
                    )
                matched = by_str.get(str(name), [])
                if len(matched) == 1:
                    return int(matched[0])
                if len(matched) > 1:
                    raise ValueError(
                        f"Ambiguous phenotype selector '{name}': multiple columns share this string form."
                    )
                return None

            for spec in requested_specs:
                local_idx: Optional[int] = None
                global_idx: Optional[int] = None
                if isinstance(spec, int):
                    idx = int(spec)
                    if 0 <= idx < int(pheno.shape[1]):
                        local_idx = idx
                        global_idx = idx
                    else:
                        fallback_idx = _resolve_name_to_index(str(spec))
                        if fallback_idx is not None:
                            local_idx = int(fallback_idx)
                            global_idx = int(fallback_idx)
                else:
                    resolved_idx = _resolve_name_to_index(spec)
                    if resolved_idx is not None:
                        local_idx = int(resolved_idx)
                        global_idx = int(resolved_idx)

                if local_idx is None or global_idx is None:
                    invalid_specs.append(str(spec))
                    continue
                if local_idx in seen_local:
                    continue
                seen_local.add(local_idx)
                ncol_take.append(local_idx)
                selected_ncol.append(global_idx)

        if len(ncol_take) == 0:
            max_idx = int(pheno.shape[1]) - 1
            preview = ", ".join(str(c) for c in list(pheno.columns)[:10])
            msg = (
                "Requested phenotype selector(s) not found. "
                f"requested={list(requested_specs)}, valid_index=[0..{max_idx}], "
                f"columns={preview}"
                + (" ..." if len(pheno.columns) > 10 else "")
            )
            logger.error(msg)
            raise ValueError(msg)
        if len(invalid_specs) > 0:
            max_idx = int(pheno.shape[1]) - 1
            logger.warning(
                "Ignoring unknown phenotype selectors: "
                f"{invalid_specs}. valid_index=[0..{max_idx}]"
            )
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
    Cache prefix used by gfreader for VCF/HMP/TXT temporary converted files.
    """
    base = genotype_cache_prefix(genofile, snps_only=bool(snps_only))
    return f"{base}.snp{1 if bool(snps_only) else 0}"


def _detect_cache_need(
    genofile: str,
    *,
    snps_only: bool = True,
    maf: float = 0.02,
    max_missing_rate: float = 0.05,
    het_threshold: float = 0.01,
    force_kind: Optional[str] = None,
) -> tuple[bool, str, list[str]]:
    """
    Detect whether genotype cache build is expected before inspect/load.
    """
    low = str(genofile).lower()
    kind_forced = str(force_kind or "").strip().lower()
    if kind_forced == "vcf" or low.endswith(".vcf.gz") or low.endswith(".vcf"):
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

    if kind_forced == "hmp" or low.endswith(".hmp.gz") or low.endswith(".hmp"):
        # HMP caching currently uses the non-parameterized PLINK cache prefix (~base)
        # from gfreader's source-cache path.
        cprefix = f"{genotype_cache_prefix(genofile, snps_only=bool(snps_only))}.snp{1 if bool(snps_only) else 0}"
        targets = [f"{cprefix}.bed", f"{cprefix}.bim", f"{cprefix}.fam"]
        all_exist = all(os.path.isfile(p) for p in targets)
        stale = False
        if all_exist and os.path.isfile(genofile):
            src_mtime = os.path.getmtime(genofile)
            cache_mtime = min(os.path.getmtime(p) for p in targets)
            stale = cache_mtime < src_mtime
        return (not all_exist) or stale, "hmp", targets

    file_prefix, matrix_path = _resolve_file_input_matrix(genofile)
    if file_prefix and matrix_path:
        if matrix_path.lower().endswith(".npy"):
            return False, "", []
        cprefix = genotype_cache_prefix(genofile, snps_only=bool(snps_only))
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
    ncol: Union[list[object], None],
    logger: logging.Logger,
    *,
    id_col: int = 0,
    use_spinner: bool = False,
    emit_complete: bool = True,
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
        if bool(emit_complete):
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
    force_kind: Optional[str] = None,
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
            force_kind=force_kind,
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


def _abort_cli_input_error(logger: logging.Logger, ex: Exception) -> None:
    msg = str(ex).strip()
    if msg:
        for line in msg.splitlines():
            line_text = str(line).strip()
            if line_text:
                logger.error(line_text)
    else:
        logger.error(type(ex).__name__)
    raise SystemExit(1) from None


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
    force_kind: Optional[str] = None,
    emit_complete: bool = True,
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
        force_kind=force_kind,
    )
    status_enabled = bool(use_spinner)
    plain_progress = (not status_enabled) and (cache_kind in {"vcf", "hmp", "txt"})

    # For direct PLINK prefixes (no cache-target metadata), inspect directly.
    # For VCF/TXT sources, always run the threaded+monitored path below so that
    # cache rebuilds triggered inside inspect_genotype_file() are still visible.
    if cache_kind == "":
        # PLINK metadata inspect can still take noticeable time on very large
        # .bim/.fam inputs; use process-backed spinner so animation does not
        # freeze even when Rust/Python call path holds the GIL.
        with CliStatus(
            genotype_load_status_open(src),
            enabled=status_enabled,
            use_process=True,
        ) as task:
            try:
                ids, n_snps = inspect_genotype_file(
                    genofile,
                    snps_only=bool(snps_only),
                    maf=float(maf_threshold),
                    missing_rate=float(max_missing_rate),
                    het=float(het_threshold),
                    force_kind=force_kind,
                )
            except ValueError as ex:
                task.fail(genotype_load_status_fail(src))
                _abort_cli_input_error(logger, ex)
            except Exception:
                task.fail(genotype_load_status_fail(src))
                raise
            if bool(emit_complete):
                task.complete(
                    genotype_load_status_done(
                        src,
                        n_samples=len(ids),
                        n_snps=int(n_snps),
                    )
                )
        return np.asarray(ids, dtype=str), int(n_snps)

    with CliStatus(genotype_load_status_open(src), enabled=status_enabled) as task:
        out: dict[str, tuple[np.ndarray, int]] = {}
        err: dict[str, Exception] = {}
        warn_msgs: list[str] = []
        last_plain_msg = ""
        last_plain_emit = 0.0
        plain_t0 = time.monotonic()
        last_wait_beat = plain_t0
        if plain_progress:
            logger.info(f"{genotype_load_status_open(src)} [{format_elapsed(0.0)}]")
            last_plain_msg = genotype_load_status_open(src)
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
        use_subproc = cache_kind in {"vcf", "hmp", "txt"}
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
                    force_kind,
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
                        force_kind,
                    )
                    out["value"] = (np.asarray(ids0, dtype=str), int(ns0))
                    warn_msgs.extend(warns0)
                except Exception as ex:
                    err["value"] = ex
                finally:
                    done_evt.set()

            t = threading.Thread(target=_worker, daemon=True)
            t.start()

        bim_path = (
            cache_targets[1]
            if (cache_kind in {"vcf", "hmp"} and len(cache_targets) >= 2)
            else ""
        )
        bed_path = (
            cache_targets[0]
            if (cache_kind in {"vcf", "hmp"} and len(cache_targets) >= 1)
            else ""
        )
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
                            msg = genotype_load_status_progress(src, f"SNP={bim_count}")
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
                            msg = genotype_load_status_progress(src, f"cache={_format_cache_size(bsz)}")
                            task.desc = msg
                            _emit_plain(msg)
                    except Exception:
                        pass
            elif bed_path and os.path.isfile(bed_path):
                try:
                    sz = int(os.path.getsize(bed_path))
                    if sz != last_size:
                        last_size = sz
                        msg = genotype_load_status_progress(src, f"cache={_format_cache_size(sz)}")
                        task.desc = msg
                        _emit_plain(msg)
                except Exception:
                    pass
            elif npy_path and os.path.isfile(npy_path):
                try:
                    sz = int(os.path.getsize(npy_path))
                    if sz != last_size:
                        last_size = sz
                        msg = genotype_load_status_progress(src, f"cache={_format_cache_size(sz)}")
                        task.desc = msg
                        _emit_plain(msg)
                except Exception:
                    pass
            now = time.monotonic()
            if now - last_wait_beat >= 3.0:
                dots = "." * (1 + (int((now - plain_t0) // 3) % 3))
                wait_msg = genotype_load_status_progress(src, f"waiting{dots}")
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
            task.fail(genotype_load_status_fail(src))
            if isinstance(err["value"], ValueError):
                _abort_cli_input_error(logger, err["value"])
            raise err["value"]

        ids, n_snps = out["value"]
        if warning_collector is not None:
            warning_collector.extend(warn_msgs)
        else:
            for wmsg in warn_msgs:
                logger.warning(wmsg)
        task.desc = genotype_load_status_progress(src, f"SNP={n_snps}")
        if bool(emit_complete):
            task.complete(
                genotype_load_status_done(
                    src,
                    n_samples=len(ids),
                    n_snps=int(n_snps),
                )
            )
        if _gwas_logger_verbose(logger):
            _log_info(logger, f"Genotype sites cached: {n_snps}", use_spinner=use_spinner)
        return ids, n_snps


def _resolve_rust_grm_backend_plan(
    genofile: str,
    *,
    allow_packed_full_load: bool,
    preloaded_packed: Union[dict[str, object], None],
    cache_write_path: Union[str, None],
) -> tuple[str, str, bool]:
    packed_prefix = _as_plink_prefix(genofile)
    if packed_prefix is None:
        raise RuntimeError(
            "Rust-only GWAS GRM build requires PLINK BED input/prefix."
        )

    cache_target = str(cache_write_path).strip() if cache_write_path is not None else ""
    have_stream = hasattr(jxrs, "grm_stream_bed_f32")
    have_stream_to_npy = hasattr(jxrs, "grm_stream_bed_f32_to_npy")
    have_packed_bed = hasattr(jxrs, "grm_packed_bed_f32")
    have_packed_preloaded = hasattr(jxrs, "grm_packed_f32")

    pre = preloaded_packed if packed_preload_is_ready(preloaded_packed) else None
    preload_matches = bool(
        isinstance(pre, dict)
        and str(pre.get("prefix", "")) == str(packed_prefix)
        and isinstance(pre.get("packed_ctx"), dict)
        and have_packed_preloaded
    )
    if preload_matches:
        return (
            "packed-preloaded",
            "reusing preloaded packed payload for slice-friendly downstream workflow",
            False,
        )

    if have_stream:
        return (
            "memmap-bed",
            "full-sample GRM build without packed reuse requirement",
            bool(cache_target != "" and have_stream_to_npy),
        )

    if have_packed_bed:
        if bool(allow_packed_full_load):
            reason = "memmap GRM kernel unavailable; packed full-load is allowed"
        else:
            reason = "memmap GRM kernel unavailable; fallback to packed full-load"
        return ("packed-bed", reason, False)

    raise RuntimeError(
        "No Rust BED GRM kernel is available. Rebuild JanusX extension to export "
        "`grm_stream_bed_f32` or `grm_packed_bed_f32`."
    )


def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    method: int,
    mmap_window_mb: Union[int , None],
    memory_mb: float,
    threads: int,
    logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    allow_packed_full_load: bool = False,
    preloaded_packed: Union[dict[str, object], None] = None,
    cache_write_path: Union[str, None] = None,
) -> tuple[np.ndarray, int]:
    """
    Build GRM in Rust-only mode from PLINK BED input.

    Route policy:
    - default full-sample GRM builds prefer Rust memmap-BED
    - when a compatible packed preload already exists, reuse it
    - if memmap is unavailable, fallback to packed BED full-load
    """
    packed_prefix = _as_plink_prefix(genofile)
    backend_kind, backend_reason, stream_direct_cache = _resolve_rust_grm_backend_plan(
        genofile,
        allow_packed_full_load=bool(allow_packed_full_load),
        preloaded_packed=preloaded_packed,
        cache_write_path=cache_write_path,
    )

    if _gwas_logger_verbose(logger):
        _log_info(
            logger,
            (
                f"Building GRM (Rust {backend_kind} single-entry), method={method}. "
                f"route={backend_reason}"
            ),
            use_spinner=use_spinner,
        )
    pbar = _ProgressAdapter(
        total=n_snps,
        desc=("GRM (rust-memmap)" if backend_kind == "memmap-bed" else "GRM (rust-bed)"),
        force_animate=True,
        logger=logger,
    )
    last_done = 0

    def _on_packed_progress(done: int, _total: int) -> None:
        nonlocal last_done
        d = int(done)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d

    try:
        grm_raw = None
        grm: Union[np.ndarray, None] = None
        eff_m = 0
        cache_target = str(cache_write_path).strip() if cache_write_path is not None else ""
        with _bed_block_target_env(memory_mb, buffers=_GWAS_WORKING_BUFFERS_GRM):
            if backend_kind == "packed-preloaded":
                pre = preloaded_packed if packed_preload_is_ready(preloaded_packed) else None
                if not isinstance(pre, dict) or str(pre.get("prefix", "")) != str(packed_prefix):
                    raise RuntimeError(
                        "Packed preload route selected, but compatible preloaded payload is unavailable."
                    )
                packed_ctx_obj = pre.get("packed_ctx")
                if not isinstance(packed_ctx_obj, dict):
                    raise RuntimeError("Packed preload route selected without a valid packed_ctx.")
                packed_pre = np.ascontiguousarray(
                    np.asarray(packed_ctx_obj["packed"], dtype=np.uint8),
                    dtype=np.uint8,
                )
                maf_pre = np.ascontiguousarray(
                    np.asarray(packed_ctx_obj["maf"], dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
                row_flip_pre = np.ascontiguousarray(
                    np.asarray(packed_ctx_obj["row_flip"], dtype=np.bool_).reshape(-1),
                    dtype=np.bool_,
                )
                packed_n_pre = int(packed_ctx_obj["n_samples"])
                if packed_n_pre != int(n_samples):
                    raise RuntimeError(
                        "Preloaded packed sample count mismatch: "
                        f"packed={packed_n_pre}, expected={int(n_samples)}"
                    )
                if int(packed_pre.shape[0]) != int(maf_pre.shape[0]) or int(
                    packed_pre.shape[0]
                ) != int(row_flip_pre.shape[0]):
                    raise RuntimeError(
                        "Preloaded packed metadata mismatch: "
                        f"packed_rows={packed_pre.shape[0]}, maf={maf_pre.shape[0]}, row_flip={row_flip_pre.shape[0]}"
                    )
                _log_file_only(
                    logger,
                    logging.INFO,
                    "Packed GRM: reusing preloaded packed payload (skip BED repack).",
                )
                grm_raw = jxrs.grm_packed_f32(
                    packed_pre,
                    int(packed_n_pre),
                    row_flip_pre,
                    maf_pre,
                    sample_indices=None,
                    method=int(method),
                    block_cols=max(1, int(chunk_size)),
                    threads=max(1, int(threads)),
                    progress_callback=_on_packed_progress,
                    progress_every=max(1, int(chunk_size)),
                )
                eff_m = int(packed_pre.shape[0])
            elif backend_kind == "packed-bed":
                grm_raw, eff_m_raw, packed_n_raw = jxrs.grm_packed_bed_f32(
                    str(packed_prefix),
                    method=int(method),
                    maf_threshold=float(maf_threshold),
                    max_missing_rate=float(max_missing_rate),
                    block_cols=max(1, int(chunk_size)),
                    threads=max(1, int(threads)),
                    progress_callback=_on_packed_progress,
                    progress_every=max(1, int(chunk_size)),
                )
                packed_n = int(packed_n_raw)
                if packed_n != int(n_samples):
                    raise RuntimeError(
                        f"Packed sample count mismatch: packed={packed_n}, expected={int(n_samples)}"
                    )
                eff_m = int(eff_m_raw)
            else:
                two_stage_raw = str(os.environ.get("JX_GRM_STREAM_TWO_STAGE", "")).strip().lower()
                if two_stage_raw in {"1", "true", "yes", "on"}:
                    _log_file_only(
                        logger,
                        logging.INFO,
                        "GRM memmap-bed note: JX_GRM_STREAM_TWO_STAGE=1 is enabled; "
                        "memmap entry may internally switch to packed prebuild mode.",
                    )
                if stream_direct_cache:
                    eff_m_raw, stream_n_raw = jxrs.grm_stream_bed_f32_to_npy(
                        str(packed_prefix),
                        str(cache_target),
                        method=int(method),
                        maf_threshold=float(maf_threshold),
                        max_missing_rate=float(max_missing_rate),
                        block_cols=max(1, int(chunk_size)),
                        threads=max(1, int(threads)),
                        progress_callback=_on_packed_progress,
                        progress_every=max(1, int(chunk_size)),
                        mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                    )
                    stream_n = int(stream_n_raw)
                    if stream_n != int(n_samples):
                        raise RuntimeError(
                            f"Memmap sample count mismatch: memmap={stream_n}, expected={int(n_samples)}"
                        )
                    eff_m = int(eff_m_raw)
                    # Keep cache writer output on disk only; caller will atomically
                    # move tmp -> final path and then reopen the final cache.
                    # This avoids holding a memmap handle on tmp cache files, which
                    # can block os.replace on Windows (WinError 32).
                    grm = np.empty((0, 0), dtype=np.float32)
                else:
                    grm_raw, eff_m_raw, stream_n_raw = jxrs.grm_stream_bed_f32(
                        str(packed_prefix),
                        method=int(method),
                        maf_threshold=float(maf_threshold),
                        max_missing_rate=float(max_missing_rate),
                        block_cols=max(1, int(chunk_size)),
                        threads=max(1, int(threads)),
                        progress_callback=_on_packed_progress,
                        progress_every=max(1, int(chunk_size)),
                        mmap_window_mb=(int(mmap_window_mb) if mmap_window_mb is not None else None),
                    )
                    stream_n = int(stream_n_raw)
                    if stream_n != int(n_samples):
                        raise RuntimeError(
                            f"Memmap sample count mismatch: memmap={stream_n}, expected={int(n_samples)}"
                        )
                    eff_m = int(eff_m_raw)

        if grm is None:
            grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float32))
        pbar.finish()
    finally:
        pbar.close(show_done=False)

    if _gwas_logger_verbose(logger):
        _log_info(logger, "GRM construction finished.", use_spinner=use_spinner)
    return grm, int(eff_m)

def load_or_build_grm_with_cache(
    genofile: str,
    cache_prefix: str,
    mgrm: str,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    chunk_size: int,
    memory_mb: float,
    threads: int,
    logger:logging.Logger,
    use_spinner: bool = False,
    ids_preloaded: Union[np.ndarray, None] = None,
    n_snps_preloaded: Union[int, None] = None,
    snps_only: bool = True,
    allow_packed_full_load: bool = False,
    preloaded_packed: Union[dict[str, object], None] = None,
) -> tuple[np.ndarray, int, Union[np.ndarray, None], Union[str, None]]:
    """
    Load or build a GRM with caching for streaming LMM/LM runs.
    """
    genofile_for_grm = str(genofile)
    if _as_plink_prefix(genofile_for_grm) is None:
        cached_candidate = ""
        try:
            delim = "," if str(genofile_for_grm).lower().endswith(".csv") else None
            cached_candidate = str(
                prepare_cli_input_cache(
                    str(genofile_for_grm),
                    snps_only=bool(snps_only),
                    delimiter=delim,
                    prefer_plink_for_txt=True,
                    threads=int(args.thread),
                )
            )
        except Exception as ex:
            _log_file_only(
                logger,
                logging.WARNING,
                "GRM cache materialization to PLINK BED failed; "
                f"input={genofile_for_grm}, reason={ex}",
            )
        cached_prefix = _as_plink_prefix(cached_candidate) if cached_candidate != "" else None
        if cached_prefix is not None:
            genofile_for_grm = str(cached_prefix)
            _log_file_only(
                logger,
                logging.INFO,
                f"GRM input switched to PLINK cache prefix: {genofile_for_grm}",
            )

    if ids_preloaded is not None and n_snps_preloaded is not None:
        ids = np.asarray(ids_preloaded, dtype=str)
        n_snps = int(n_snps_preloaded)
    else:
        ids0, n_snps0 = inspect_genotype_file(
            genofile_for_grm,
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
    grm_cache_path: Union[str, None] = None
    if method_is_builtin:
        grm_path, id_path = _grm_cache_paths(cache_prefix, mgrm=str(mgrm))
        grm_cache_path = str(grm_path)
        legacy_grm_path, legacy_id_path = _grm_cache_paths_legacy(
            cache_prefix, mgrm=str(mgrm)
        )
        with _cache_lock(grm_path):
            if (not os.path.exists(grm_path)) and os.path.exists(legacy_grm_path):
                try:
                    _replace_file_with_retry(legacy_grm_path, grm_path)
                    if os.path.exists(legacy_id_path) and (not os.path.exists(id_path)):
                        _replace_file_with_retry(legacy_id_path, id_path)
                    _log_file_only(
                        logger,
                        logging.INFO,
                        f"Migrated legacy GRM cache name: {legacy_grm_path} -> {grm_path}",
                    )
                except Exception:
                    pass
            cache_is_stale = False
            if os.path.exists(grm_path):
                g_mtime = latest_genotype_mtime(genofile)
                k_mtime = os.path.getmtime(grm_path)
                if g_mtime is not None and g_mtime > k_mtime:
                    cache_is_stale = True
                    logger.warning(
                        "Genotype input is newer than cached GRM; rebuilding GRM cache."
                    )
                elif str(grm_path).lower().endswith(".npy"):
                    try:
                        grm_probe = np.load(grm_path, mmap_mode='r')
                        if np.dtype(grm_probe.dtype) != np.dtype(np.float32):
                            cache_is_stale = True
                            logger.warning(
                                "Cached GRM dtype is not float32; rebuilding GRM cache."
                            )
                    except Exception:
                        cache_is_stale = True
                        logger.warning(
                            "Cached GRM could not be validated; rebuilding GRM cache."
                        )
            if cache_is_stale:
                for p in (grm_path, id_path, legacy_grm_path, legacy_id_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

            grm_cache_source: str | None = None
            cached_grm = None
            if not cache_is_stale:
                cached_grm, grm_cache_source = load_or_materialize_square_grm_cache(
                    grm_path,
                    expected_n=int(n_samples),
                    mmap_mode="r",
                    expected_dtype=np.float32,
                    npy_dtype=np.float32,
                    cache_id_path=id_path,
                )

            if cached_grm is not None:
                with CliStatus(grm_load_status_open(grm_path), enabled=bool(use_spinner)) as task:
                    try:
                        grm = cached_grm
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
                        task.fail(grm_load_status_fail(grm_path))
                        raise
                    task.complete(grm_load_status_done(grm_path, int(grm.shape[0])))
                if str(grm_cache_source) == "txt->npy":
                    _log_file_only(
                        logger,
                        logging.INFO,
                        grm_text_materialized_message(grm_path),
                    )
            else:
                method_int = int(mgrm)
                grm_calc_t0 = time.monotonic()
                tmp_grm = f"{grm_path}.tmp.{os.getpid()}.npy"
                _, _, stream_direct_cache = _resolve_rust_grm_backend_plan(
                    genofile_for_grm,
                    allow_packed_full_load=bool(allow_packed_full_load),
                    preloaded_packed=preloaded_packed,
                    cache_write_path=tmp_grm,
                )
                grm, eff_m = build_grm_streaming(
                    genofile=genofile_for_grm,
                    n_samples=n_samples,
                    n_snps=n_snps,
                    maf_threshold=maf_threshold,
                    max_missing_rate=max_missing_rate,
                    chunk_size=chunk_size,
                    method=method_int,
                    mmap_window_mb=_common_resolve_decode_mmap_window_mb(
                        genofile_for_grm,
                        n_samples,
                        n_snps,
                        float(memory_mb),
                        needs_copy=False,
                        buffers=_GWAS_WORKING_BUFFERS_GRM,
                    ),
                    memory_mb=float(memory_mb),
                    threads=threads,
                    logger=logger,
                    use_spinner=use_spinner,
                    snps_only=bool(snps_only),
                    allow_packed_full_load=bool(allow_packed_full_load),
                    preloaded_packed=preloaded_packed,
                    cache_write_path=(tmp_grm if stream_direct_cache else None),
                )
                grm_msg = (
                    f"Calculating GRM from genotype (n={int(n_samples)}) "
                    f"...Finished [{format_elapsed(time.monotonic() - grm_calc_t0)}]"
                )
                _log_file_only(logger, logging.INFO, grm_msg)
                print_success(grm_msg, force_color=bool(use_spinner))
                if not os.path.exists(tmp_grm):
                    if stream_direct_cache:
                        raise RuntimeError(
                            "Rust GRM stream-to-NPY cache writer did not produce the "
                            f"expected tmp file: {tmp_grm}"
                        )
                    save_grm_npy_blocked(
                        tmp_grm,
                        grm,
                        dtype=np.float32,
                    )
                _replace_file_with_retry(tmp_grm, grm_path)
                tmp_id = f"{id_path}.tmp.{os.getpid()}"
                pd.Series(ids).to_csv(tmp_id, sep="\t", index=False, header=False)
                _replace_file_with_retry(tmp_id, id_path)
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
        with CliStatus(grm_load_status_open(mgrm), enabled=bool(use_spinner)) as task:
            try:
                if mgrm.endswith('.npy'):
                    grm = np.load(mgrm,mmap_mode='r')
                else:
                    grm = np.genfromtxt(mgrm, dtype="float64")
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
                task.fail(grm_load_status_fail(mgrm))
                raise
            task.complete(grm_load_status_done(mgrm, int(grm.shape[0])))

    if _gwas_logger_verbose(logger):
        _log_info(logger, f"GRM shape: {grm.shape}", use_spinner=use_spinner)
    return grm, eff_m, grm_ids, grm_cache_path


def build_pcs_from_grm(
    grm: np.ndarray,
    dim: int,
    logger: logging.Logger,
    *,
    threads: int = 1,
    use_spinner: bool = False,
) -> np.ndarray:
    """
    Compute leading principal components from GRM.

    GWAS policy: enforce Rust eigh backend for PCA construction.
    """
    with CliStatus(
        f"Calculating PCs via GRM + eigh (n={int(grm.shape[0])}, nPC={int(dim)})...",
        enabled=bool(use_spinner),
        use_process=True,
    ) as task:
        try:
            _eigval, eigvec, _evd_backend, _evd_secs = _gwas_eigh_from_grm(
                grm,
                threads=int(threads),
                logger=logger,
                stage_label="GWAS-PCA",
                require_rust=True,
            )
            pcs = eigvec[:, -dim:]
        except Exception:
            task.fail("Calculating PCs via GRM + eigh ...Failed")
            raise
        task.complete(
            f"Calculating PCs via GRM + eigh (n={pcs.shape[0]}, nPC={pcs.shape[1]})"
        )
    return pcs


def build_pcs_from_genotype_rsvd(
    genofile: str,
    dim: int,
    logger: logging.Logger,
    *,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    memory_mb: float,
    snps_only: bool,
    threads: int = 1,
    use_spinner: bool = False,
    preloaded_packed: Union[dict[str, object], None] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Compute leading PCs directly from genotype input via Rust RSVD.

    Preferred route:
      1) reuse packed preload when GWAS is already in packed mode
      2) otherwise use memmap/stream RSVD on the resolved genotype source
    """
    dim_use = max(1, int(dim))
    threads_use = max(1, int(threads))
    if hasattr(jxrs, "admx_set_threads"):
        try:
            jxrs.admx_set_threads(int(threads_use))
        except Exception:
            pass

    if bool(packed_preload_is_ready(preloaded_packed)) and hasattr(jxrs, "rsvd_packed_subset"):
        packed_ctx = preloaded_packed.get("packed_ctx") if isinstance(preloaded_packed, dict) else None
        full_ids_obj = preloaded_packed.get("full_ids") if isinstance(preloaded_packed, dict) else None
        if isinstance(packed_ctx, dict) and full_ids_obj is not None:
            full_ids = np.asarray(full_ids_obj, dtype=str)
            packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
            packed_n = int(packed_ctx["n_samples"])
            if packed_n != int(full_ids.shape[0]):
                raise ValueError(
                    f"Packed preload sample size mismatch for RSVD PCA: packed={packed_n}, ids={full_ids.shape[0]}"
                )
            sample_idx = np.arange(packed_n, dtype=np.int64)
            with runtime_thread_stage(rayon_threads=int(threads_use)):
                with CliStatus(
                    f"Calculating PCs via RSVD packed preload (n={packed_n}, nPC={dim_use})...",
                    enabled=bool(use_spinner),
                    use_process=True,
                ) as task:
                    try:
                        _eval_raw, evec_raw, _maf_raw, _flip_raw = jxrs.rsvd_packed_subset(
                            packed,
                            int(packed_n),
                            int(dim_use),
                            sample_idx,
                            42,
                            3,
                            1e-1,
                        )
                    except Exception:
                        task.fail("Calculating PCs via RSVD packed preload ...Failed")
                        raise
                    qmatrix = np.ascontiguousarray(np.asarray(evec_raw, dtype="float32"), dtype="float32")
                    if qmatrix.ndim == 1:
                        qmatrix = qmatrix.reshape(-1, 1)
                    task.complete(
                        f"Calculating PCs via RSVD packed preload (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]})"
                    )
            return qmatrix, full_ids, "rsvd_packed_preload"

    if not hasattr(jxrs, "admx_rsvd_stream_sample"):
        raise RuntimeError(
            "GWAS Q/PC build requires Rust RSVD backend admx_rsvd_stream_sample, but the symbol is unavailable."
        )

    rsvd_input = str(genofile).strip()
    if _as_plink_prefix(rsvd_input) is None:
        delim = "," if rsvd_input.lower().endswith(".csv") else None
        rsvd_input = str(
            prepare_cli_input_cache(
                rsvd_input,
                snps_only=bool(snps_only),
                delimiter=delim,
                prefer_plink_for_txt=True,
                threads=int(args.thread),
            )
        )
    sample_ids, algo_snp = inspect_genotype_file(
        rsvd_input,
        snps_only=bool(snps_only),
        maf=float(maf_threshold),
        missing_rate=float(max_missing_rate),
    )
    samples = np.asarray(sample_ids, dtype=str)
    mmap_window_mb = _common_resolve_decode_mmap_window_mb(
        rsvd_input,
        int(samples.shape[0]),
        int(algo_snp),
        float(memory_mb),
        needs_copy=False,
    )
    with runtime_thread_stage(rayon_threads=int(threads_use)):
        with CliStatus(
            f"Calculating PCs via RSVD stream (n={samples.shape[0]}, nPC={dim_use})...",
            enabled=bool(use_spinner),
            use_process=True,
        ) as task:
            try:
                with _bed_block_target_env(float(memory_mb)):
                    _eval_raw, evec_raw, _total_variance = jxrs.admx_rsvd_stream_sample(
                        str(rsvd_input),
                        int(dim_use),
                        42,
                        3,
                        1e-1,
                        bool(snps_only),
                        float(maf_threshold),
                        float(max_missing_rate),
                        None,
                        int(mmap_window_mb),
                    )
            except Exception:
                task.fail("Calculating PCs via RSVD stream ...Failed")
                raise
            qmatrix = np.ascontiguousarray(np.asarray(evec_raw, dtype="float32"), dtype="float32")
            if qmatrix.ndim == 1:
                qmatrix = qmatrix.reshape(-1, 1)
            task.complete(
                f"Calculating PCs via RSVD stream (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]})"
            )
    return qmatrix, samples, "rsvd_stream"


def _gwas_pca_backend_display_name(backend: object) -> str:
    key = str(backend or "").strip().lower()
    if key in {"rsvd_stream", "rsvd_packed_preload"}:
        return "RSVD"
    if key in {"grm_eigh", "grm_eigh_fallback"}:
        return "GRM + eigh"
    if key == "":
        return "PCA"
    return str(backend)


def _gwas_qcov_prefers_grm_route(qcov_opt: object, n_samples: int) -> bool:
    try:
        qdim = int(_parse_qcov_dim(qcov_opt))
    except Exception:
        return False
    if int(qdim) <= 0:
        return False
    return bool(int(n_samples) <= int(_GWAS_PCA_GRM_EIGH_SAMPLE_THRESHOLD))


def load_or_build_q_with_cache(
    genofile: str,
    grm: Union[np.ndarray, None],
    grm_ids: Union[np.ndarray, None],
    n_samples: int,
    n_snps_preloaded: Union[int, None],
    cache_prefix: str,
    pcdim: str,
    mgrm: str,
    ids: np.ndarray,
    maf_threshold: float,
    max_missing_rate: float,
    het_threshold: float,
    chunk_size: int,
    memory_mb: float,
    snps_only: bool,
    threads: int,
    logger,
    use_spinner: bool = False,
    preloaded_packed: Union[dict[str, object], None] = None,
    allow_packed_full_load: bool = False,
) -> tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Load or build Q matrix (PCs) with caching for streaming LMM/LM.
    Note: external Q file via -q is no longer supported; pass external
    covariate matrices via -c.

    GWAS policy:
      - n <= 15k: prefer GRM + eigh
      - n > 15k: prefer Rust RSVD on genotype input, fallback to GRM eigh
        only when RSVD is unavailable or fails.
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
        legacy_q_path = _pca_cache_path_legacy(cache_prefix, mgrm=str(mgrm), qdim=int(dim))
        grm_local = grm
        grm_ids_local = (
            None
            if grm_ids is None
            else np.asarray(grm_ids, dtype=str).reshape(-1)
        )

        def _ensure_grm_for_q_build() -> tuple[np.ndarray, Union[np.ndarray, None]]:
            nonlocal grm_local, grm_ids_local
            if grm_local is not None:
                return grm_local, grm_ids_local
            grm_local, _eff_m_local, grm_ids_loaded, _grm_cache_path_local = load_or_build_grm_with_cache(
                genofile=str(genofile),
                cache_prefix=str(cache_prefix),
                mgrm=str(mgrm),
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                chunk_size=int(chunk_size),
                memory_mb=float(memory_mb),
                threads=int(threads),
                logger=logger,
                use_spinner=bool(use_spinner),
                ids_preloaded=np.asarray(ids, dtype=str),
                n_snps_preloaded=(
                    None if n_snps_preloaded is None else int(n_snps_preloaded)
                ),
                snps_only=bool(snps_only),
                allow_packed_full_load=bool(allow_packed_full_load),
                preloaded_packed=preloaded_packed,
            )
            grm_ids_local = (
                None
                if grm_ids_loaded is None
                else np.asarray(grm_ids_loaded, dtype=str).reshape(-1)
            )
            return grm_local, grm_ids_local

        with _cache_lock(q_path):
            if (not os.path.isfile(q_path)) and os.path.isfile(legacy_q_path):
                _log_file_only(
                    logger,
                    logging.INFO,
                    f"Ignoring legacy PCA cache and rebuilding in new format: {legacy_q_path}",
                )
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
                        q_ids, qmatrix = _load_pca_cache_with_ids(
                            q_path,
                            expected_rows=int(n),
                            expected_dim=int(dim),
                            expected_ids=np.asarray(ids, dtype=str),
                        )
                    except Exception as ex:
                        task.fail(f"Loading Q matrix from {src} ...Failed")
                        cache_ready = False
                        _emit_warning_line(
                            logger,
                            f"PCA cache format/content invalid; rebuilding cache. path={src}, reason={ex}",
                            use_spinner=bool(use_spinner),
                        )
                    if not cache_ready:
                        pass
                    else:
                        task.complete(
                            f"Loading Q matrix from {src} (n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]}) ...Finished"
                        )
            if not cache_ready:
                pc_calc_t0 = time.monotonic()
                pc_backend = "unknown"
                prefer_grm_eigh = bool(int(n) <= int(_GWAS_PCA_GRM_EIGH_SAMPLE_THRESHOLD))
                if prefer_grm_eigh:
                    grm_for_q, grm_ids_for_q = _ensure_grm_for_q_build()
                    qmatrix = build_pcs_from_grm(
                        grm_for_q,
                        int(dim),
                        logger=logger,
                        threads=int(threads),
                        use_spinner=bool(use_spinner),
                    )
                    qmatrix = np.asarray(qmatrix, dtype="float32")
                    q_ids = (
                        np.asarray(grm_ids_for_q, dtype=str)
                        if grm_ids_for_q is not None
                        else np.asarray(ids, dtype=str)
                    )
                    pc_backend = "grm_eigh"
                else:
                    try:
                        qmatrix, q_ids, pc_backend = build_pcs_from_genotype_rsvd(
                            genofile=str(genofile),
                            dim=int(dim),
                            logger=logger,
                            maf_threshold=float(maf_threshold),
                            max_missing_rate=float(max_missing_rate),
                            chunk_size=int(chunk_size),
                            memory_mb=float(memory_mb),
                            snps_only=bool(snps_only),
                            threads=int(threads),
                            use_spinner=bool(use_spinner),
                            preloaded_packed=preloaded_packed,
                        )
                    except Exception as rsvd_ex:
                        _emit_warning_line(
                            logger,
                            f"RSVD PCA build failed; falling back to GRM eigh. reason={rsvd_ex}",
                            use_spinner=bool(use_spinner),
                        )
                        grm_for_q, grm_ids_for_q = _ensure_grm_for_q_build()
                        _eigval, eigvec, _evd_backend, _evd_secs = _gwas_eigh_from_grm(
                            grm_for_q,
                            threads=int(threads),
                            logger=logger,
                            stage_label="GWAS-Q-build",
                            require_rust=True,
                        )
                        qmatrix = np.asarray(eigvec[:, -dim:], dtype="float32")
                        q_ids = (
                            np.asarray(grm_ids_for_q, dtype=str)
                            if grm_ids_for_q is not None
                            else np.asarray(ids, dtype=str)
                        )
                        pc_backend = "grm_eigh_fallback"
                if qmatrix.shape != (n, int(dim)):
                    raise ValueError(
                        f"PCA build shape mismatch: expected ({n},{dim}), got {qmatrix.shape}"
                    )
                pc_msg = (
                    f"Calculating PCs via {_gwas_pca_backend_display_name(pc_backend)} "
                    f"(n={qmatrix.shape[0]}, nPC={qmatrix.shape[1]}) "
                    f"...Finished [{format_elapsed(time.monotonic() - pc_calc_t0)}]"
                )
                _log_file_only(logger, logging.INFO, pc_msg)
                tmp_q = f"{q_path}.tmp.{os.getpid()}"
                _write_pca_cache_with_ids(tmp_q, np.asarray(q_ids, dtype=str), qmatrix)
                _replace_file_with_retry(tmp_q, q_path)
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

    if _gwas_logger_verbose(logger):
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
    memory_mb: float,
    mgrm: str,
    pcdim: str,
    cov_inputs: Union[str, list[str], None],
    threads: int,
    require_kinship: bool,
    logger,
    use_spinner: bool = False,
    snps_only: bool = True,
    allow_packed_grm: bool = False,
    preload_packed_context: bool = False,
    require_bed_stream: bool = False,
    post_grm_hook: Optional[Callable[[str, Optional[str]], None]] = None,
    force_kind: Optional[str] = None,
    preinspected_ids: Optional[np.ndarray] = None,
    preinspected_n_snps: Optional[int] = None,
    preinspected_genotype_elapsed_secs: Optional[float] = None,
    preinspected_genotype_src: Optional[str] = None,
    preinspect_warnings: Optional[list[str]] = None,
    working_buffers: int = 1,
    prewarm_global_scanmeta: bool = False,
    scanmeta_outprefix: Optional[str] = None,
):
    """
    Prepare all shared resources for streaming LMM/LM once:
      - phenotype
      - genotype metadata (ids, n_snps)
      - GRM + Q (cached)
      - covariates (optional)
    """
    delay_pheno_complete = bool(prewarm_global_scanmeta)
    pheno_src = _basename_only(phenofile)
    pheno_t0 = time.monotonic() if delay_pheno_complete else None
    pheno = _load_phenotype_with_status(
        phenofile,
        pheno_cols,
        logger,
        id_col=0,
        use_spinner=use_spinner,
        emit_complete=(not delay_pheno_complete),
    )

    deferred_cache_warnings: list[str] = list(preinspect_warnings or [])
    delay_genotype_complete = bool(prewarm_global_scanmeta)
    genotype_src = str(preinspected_genotype_src).strip() or _basename_only(genofile)
    genotype_elapsed_secs = (
        float(preinspected_genotype_elapsed_secs)
        if preinspected_genotype_elapsed_secs is not None
        else 0.0
    )
    if (preinspected_ids is not None) and (preinspected_n_snps is not None):
        ids = np.asarray(preinspected_ids, dtype=str)
        n_snps = int(preinspected_n_snps)
    else:
        genotype_t0 = time.monotonic()
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
                force_kind=force_kind,
                emit_complete=(not delay_genotype_complete),
            )
            for w in caught:
                try:
                    msg = str(w.message).strip()
                except Exception:
                    msg = ""
                if _is_cache_warning_message(msg):
                    deferred_cache_warnings.append(msg)
        genotype_elapsed_secs = max(time.monotonic() - genotype_t0, 0.0)
    n_samples = len(ids)
    qcov_prefers_grm_local = _gwas_qcov_prefers_grm_route(pcdim, int(n_samples))
    working_buffers_effective = int(max(1, int(working_buffers)))
    if bool(qcov_prefers_grm_local):
        working_buffers_effective = max(
            int(working_buffers_effective),
            int(_GWAS_WORKING_BUFFERS_GRM),
        )
    chunk_size = _resolve_bed_block_rows_from_memory(
        float(memory_mb),
        int(n_samples),
        int(n_snps),
        streaming=True,
        working_buffers=int(working_buffers_effective),
    )
    if _gwas_logger_verbose(logger):
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
    if _gwas_logger_verbose(logger):
        _log_info(
            logger,
            f"Cache prefix: {cache_prefix}",
            use_spinner=use_spinner,
        )
    stream_genofile = str(genofile)
    preloaded_packed: Union[dict[str, object], None] = None
    genofile_low = str(genofile).lower()
    cache_candidates: list[str] = []
    if genofile_low.endswith(".vcf") or genofile_low.endswith(".vcf.gz"):
        # VCF caches are parameterized by maf/missing/snps_only in gfreader.
        cache_candidates.append(str(cache_prefix))
    elif str(force_kind or "").strip().lower() == "hmp" or genofile_low.endswith(".hmp") or genofile_low.endswith(".hmp.gz"):
        # HMP source-cache path currently uses non-parameterized '~base' prefix.
        cache_candidates.append(
            genotype_cache_prefix(
                genofile,
                snps_only=bool(snps_only),
                logger=logger,
                warning_collector=deferred_cache_warnings,
            )
        )
        # Keep parameterized candidate for forward compatibility.
        cache_candidates.append(str(cache_prefix))
    for cp in cache_candidates:
        if all(os.path.isfile(f"{cp}.{ext}") for ext in ("bed", "bim", "fam")):
            stream_genofile = str(cp)
            break

    need_bed_cache_now = bool(
        require_kinship
        or allow_packed_grm
        or preload_packed_context
        or require_bed_stream
    )
    if need_bed_cache_now and (_as_plink_prefix(stream_genofile) is None):
        try:
            delim = "," if str(stream_genofile).lower().endswith(".csv") else None
            cached_cli = str(
                prepare_cli_input_cache(
                    str(stream_genofile),
                    snps_only=bool(snps_only),
                    delimiter=delim,
                    prefer_plink_for_txt=True,
                    force_kind=force_kind,
                    threads=int(max(1, int(threads))),
                )
            )
            cached_prefix = _as_plink_prefix(cached_cli)
            if cached_prefix is not None:
                stream_genofile = str(cached_prefix)
                _log_file_only(
                    logger,
                    logging.INFO,
                    f"Streaming genotype switched to PLINK cache: {stream_genofile}",
                )
        except Exception as ex:
            _emit_warning_line(
                logger,
                "Genotype BED cache materialization unavailable; "
                f"continuing with source input. reason={ex}",
                use_spinner=bool(use_spinner),
            )
        if bool(require_bed_stream) and (_as_plink_prefix(stream_genofile) is None):
            raise RuntimeError(
                "Streaming GWAS requires PLINK BED input/prefix or successful BED cache "
                "materialization from the source genotype file."
            )

    # Preload packed BED once right after genotype meta is known when a
    # packed-only model route (currently FarmCPU / ALGWAS) needs it.
    if bool(preload_packed_context):
        try:
            prefix0, full_ids0, packed_ctx0, sites0 = _prepare_packed_bed_once_for_gwas(
                genofile=stream_genofile,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                use_spinner=bool(use_spinner),
                preloaded_packed=None,
                load_site_meta=False,
            )
            preloaded_packed = {
                "prefix": str(prefix0),
                "full_ids": np.asarray(full_ids0, dtype=str),
                "packed_ctx": packed_ctx0,
            }
            # When packed preload is ready, switch streaming source to the packed prefix.
            stream_genofile = str(prefix0)
        except Exception as ex:
            _emit_warning_line(
                logger,
                f"Packed preload unavailable; fallback to on-demand packed load. reason={ex}",
                use_spinner=bool(use_spinner),
            )
            preloaded_packed = packed_preload_failure_state(stream_genofile, ex)

    prewarm_scanmeta_secs = 0.0
    if bool(prewarm_global_scanmeta):
        bed_prefix_global = _as_plink_prefix(stream_genofile)
        if bed_prefix_global is not None:
            scanmeta_t0 = time.monotonic()
            _ = _gwas_logic_meta_global_cached(
                str(bed_prefix_global),
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                outprefix=scanmeta_outprefix,
                logger=logger,
                use_spinner=False,
                emit_status=False,
            )
            prewarm_scanmeta_secs = max(time.monotonic() - scanmeta_t0, 0.0)
    if delay_genotype_complete:
        log_success(
            logger,
            (
                f"{genotype_load_status_done(genotype_src, n_samples=len(ids), n_snps=int(n_snps))} "
                f"[{format_elapsed(float(genotype_elapsed_secs) + float(prewarm_scanmeta_secs))}]"
            ),
            force_color=bool(use_spinner),
        )
    if delay_pheno_complete:
        log_success(
            logger,
            (
                f"Loading phenotype from {pheno_src} "
                f"(n={pheno.shape[0]}, npheno={pheno.shape[1]}) "
                f"[{format_elapsed(None if pheno_t0 is None else (time.monotonic() - pheno_t0))}]"
            ),
            force_color=bool(use_spinner),
        )

    need_grm = bool(require_kinship)

    grm: Union[np.ndarray, None] = None
    eff_m = n_snps
    grm_ids = None
    grm_cache_path: Union[str, None] = None
    post_grm_hook_done = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if need_grm:
            # GRM stream...
            grm, eff_m, grm_ids, grm_cache_path = load_or_build_grm_with_cache(
                genofile=stream_genofile,
                cache_prefix=cache_prefix,
                mgrm=mgrm,
                maf_threshold=maf_threshold,
                max_missing_rate=max_missing_rate,
                het_threshold=het_threshold,
                chunk_size=chunk_size,
                memory_mb=float(memory_mb),
                threads=threads,
                logger=logger,
                use_spinner=use_spinner,
                ids_preloaded=ids,
                n_snps_preloaded=n_snps,
                snps_only=bool(snps_only),
                allow_packed_full_load=bool(allow_packed_grm),
                preloaded_packed=preloaded_packed,
            )
            if post_grm_hook is not None:
                post_grm_hook(str(stream_genofile), grm_cache_path)
                post_grm_hook_done = True
        else:
            # Keep terminal output concise for non-kinship runs (LM/FarmCPU-only).
            pass

        if (not post_grm_hook_done) and (post_grm_hook is not None):
            post_grm_hook(str(stream_genofile), grm_cache_path)
            post_grm_hook_done = True

        # PCA stream...
        qmatrix, q_ids = load_or_build_q_with_cache(
            genofile=stream_genofile,
            grm=grm,
            grm_ids=grm_ids,
            n_samples=n_samples,
            n_snps_preloaded=int(n_snps),
            cache_prefix=cache_prefix,
            pcdim=pcdim,
            mgrm=mgrm,
            ids=ids,
            maf_threshold=maf_threshold,
            max_missing_rate=max_missing_rate,
            het_threshold=het_threshold,
            chunk_size=chunk_size,
            memory_mb=float(memory_mb),
            snps_only=bool(snps_only),
            threads=int(threads),
            logger=logger,
            use_spinner=use_spinner,
            preloaded_packed=preloaded_packed,
            allow_packed_full_load=bool(allow_packed_grm),
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

    # For VCF/HMP inputs, cache BED may be generated during GRM/Q stage.
    # Re-check once so downstream full-rust packed routes can use it.
    if _as_plink_prefix(stream_genofile) is None and len(cache_candidates) > 0:
        for cp in cache_candidates:
            if all(os.path.isfile(f"{cp}.{ext}") for ext in ("bed", "bim", "fam")):
                stream_genofile = str(cp)
                _log_file_only(
                    logger,
                    logging.INFO,
                    f"Streaming genotype switched to packed cache: {stream_genofile}",
                )
                break

    if (
        (not packed_preload_is_ready(preloaded_packed))
        and (not packed_preload_is_disabled(preloaded_packed))
        and bool(preload_packed_context)
    ):
        try:
            prefix1, full_ids1, packed_ctx1, sites1 = _prepare_packed_bed_once_for_gwas(
                genofile=stream_genofile,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                use_spinner=bool(use_spinner),
                preloaded_packed=None,
                load_site_meta=False,
            )
            preloaded_packed = {
                "prefix": str(prefix1),
                "full_ids": np.asarray(full_ids1, dtype=str),
                "packed_ctx": packed_ctx1,
            }
            stream_genofile = str(prefix1)
        except Exception as ex:
            preloaded_packed = packed_preload_failure_state(stream_genofile, ex)

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
    q_has_pc = bool(np.asarray(qmatrix).ndim == 2 and int(qmatrix.shape[1]) > 0)
    q_n: Union[int, str] = "NA" if (q_ids is None or (not q_has_pc)) else int(len(q_ids))
    cov_n: Union[int, str] = "NA" if cov_ids is None else int(len(cov_ids))
    _set_pending_gwas_overlap_line(
        logger,
        (
            f"Sample overlap: geno={len(geno_ids)}, pheno={len(pheno_ids)}, "
            f"q={q_n}, cov={cov_n}, common={len(common_ids)}"
        ),
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
        grm_idx = np.ascontiguousarray([grm_index[sid] for sid in ids], dtype=np.int64)
        if not _is_full_identity_index(grm_idx, int(np.shape(grm)[0])):
            grm = _wrap_square_matrix_with_base_subset(grm, grm_idx)

    q_idx = [q_index[sid] for sid in ids]
    qmatrix = qmatrix[q_idx]

    if cov_all is not None:
        cov_idx = [cov_index[sid] for sid in ids]
        cov_all = cov_all[cov_idx]

    return pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m, stream_genofile, preloaded_packed


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


def _is_full_identity_index(idx: object, size: int) -> bool:
    try:
        n = int(size)
    except Exception:
        return False
    if n < 0:
        return False
    try:
        arr = np.asarray(idx)
    except Exception:
        arr = None
    if arr is not None and arr.ndim == 1:
        if int(arr.shape[0]) != n:
            return False
        try:
            arr_i = np.asarray(arr, dtype=np.int64, order="C")
        except Exception:
            arr_i = None
        if arr_i is not None:
            return bool(np.array_equal(arr_i, np.arange(n, dtype=np.int64)))
    try:
        if len(idx) != n:
            return False
        for i, v in enumerate(idx):
            if int(v) != i:
                return False
        return True
    except Exception:
        return False


class _AlignedSquareMatrix(np.ndarray):
    _jx_square_base_subset_idx: Optional[np.ndarray]

    def __array_finalize__(self, obj) -> None:
        self._jx_square_base_subset_idx = getattr(
            obj, "_jx_square_base_subset_idx", None
        )


def _normalize_square_subset_index(
    idx: object,
    size: int,
    label: str,
) -> np.ndarray:
    arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    if int(arr.shape[0]) == 0:
        raise RuntimeError(f"{label} must not be empty.")
    if np.any(arr < 0):
        bad = int(arr[np.flatnonzero(arr < 0)[0]])
        raise RuntimeError(f"{label} contains negative index: {bad}")
    if np.any(arr >= int(size)):
        bad = int(arr[np.flatnonzero(arr >= int(size))[0]])
        raise RuntimeError(
            f"{label} contains out-of-range index: {bad} for size={int(size)}"
        )
    return np.ascontiguousarray(arr, dtype=np.int64)


def _wrap_square_matrix_with_base_subset(
    grm: np.ndarray,
    base_subset_idx: object,
) -> np.ndarray:
    g_shape = np.shape(grm)
    n = int(g_shape[0]) if len(g_shape) >= 1 else 0
    idx_arr = _normalize_square_subset_index(
        base_subset_idx,
        n,
        "aligned GRM base subset index",
    )
    if _is_full_identity_index(idx_arr, n):
        return grm
    grm_view = np.asarray(grm).view(_AlignedSquareMatrix)
    grm_view._jx_square_base_subset_idx = idx_arr
    return grm_view


def _square_matrix_base_subset_idx(grm: np.ndarray) -> Optional[np.ndarray]:
    idx = getattr(grm, "_jx_square_base_subset_idx", None)
    if idx is None:
        return None
    try:
        arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if int(arr.shape[0]) == 0:
        return None
    return np.ascontiguousarray(arr, dtype=np.int64)


def _subset_square_matrix_identity_aware(grm: np.ndarray, idx: object) -> np.ndarray:
    g_shape = np.shape(grm)
    n = int(g_shape[0]) if len(g_shape) >= 1 else 0
    if _is_full_identity_index(idx, n):
        return grm
    idx_arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    return grm[np.ix_(idx_arr, idx_arr)]


def _site_tuple_parts(
    site: tuple[object, ...],
) -> tuple[str, int, str, str, str]:
    if len(site) >= 5:
        chrom, pos, a0, a1, snp = site[:5]
        snp_name = _normalize_snp_name(snp, chrom, pos)
    elif len(site) == 4:
        chrom, pos, a0, a1 = site
        snp_name = _fallback_snp_name(chrom, pos)
    else:
        raise ValueError(f"GWAS site tuple must have length 4 or 5, got {len(site)}")
    return (
        str(chrom),
        _coerce_site_pos(pos),
        str(a0),
        str(a1),
        str(snp_name),
    )


def _split_gwas_sites(
    sites: list[tuple[object, ...]],
) -> tuple[list[str], list[int], list[str], list[str], list[str]]:
    chrom: list[str] = []
    pos: list[int] = []
    allele0: list[str] = []
    allele1: list[str] = []
    snp: list[str] = []
    for site in sites:
        c, p, a0, a1, sid = _site_tuple_parts(site)
        chrom.append(c)
        pos.append(int(p))
        allele0.append(a0)
        allele1.append(a1)
        snp.append(sid)
    return chrom, pos, allele0, allele1, snp


def _read_bim_sites(prefix: str) -> list[tuple[str, int, str, str, str]]:
    out: list[tuple[str, int, str, str, str]] = []
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
            snp = _normalize_snp_name(toks[1], chrom, pos)
            a0 = str(toks[4])
            a1 = str(toks[5])
            out.append((chrom, pos, a0, a1, snp))
    return out


class _BimSiteColumns:
    __slots__ = ("chrom", "pos", "allele0", "allele1", "snp")

    def __init__(
        self,
        *,
        chrom: list[str],
        pos: list[int],
        allele0: list[str],
        allele1: list[str],
        snp: list[str],
    ) -> None:
        n = int(len(chrom))
        if (
            int(len(pos)) != n
            or int(len(allele0)) != n
            or int(len(allele1)) != n
            or int(len(snp)) != n
        ):
            raise ValueError("BIM metadata column length mismatch.")
        self.chrom = chrom
        self.pos = pos
        self.allele0 = allele0
        self.allele1 = allele1
        self.snp = snp

    @classmethod
    def from_sites(cls, sites: list[tuple[object, ...]]) -> "_BimSiteColumns":
        chrom, pos, allele0, allele1, snp = _split_gwas_sites(list(sites))
        return cls(
            chrom=[str(v) for v in chrom],
            pos=[int(v) for v in pos],
            allele0=[str(v) for v in allele0],
            allele1=[str(v) for v in allele1],
            snp=[str(v) for v in snp],
        )

    def __len__(self) -> int:
        return int(len(self.snp))

    def __iter__(self) -> Iterator[tuple[str, int, str, str, str]]:
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, key: object) -> object:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        idx = int(key)
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError("BIM metadata index out of range")
        return (
            str(self.chrom[idx]),
            int(self.pos[idx]),
            str(self.allele0[idx]),
            str(self.allele1[idx]),
            str(self.snp[idx]),
        )

    def columns(self) -> tuple[list[str], list[int], list[str], list[str], list[str]]:
        return self.chrom, self.pos, self.allele0, self.allele1, self.snp

    def take(self, idx: object) -> "_BimSiteColumns":
        idx_arr = np.asarray(idx, dtype=np.int64).reshape(-1)
        return _BimSiteColumns(
            chrom=[self.chrom[int(i)] for i in idx_arr],
            pos=[int(self.pos[int(i)]) for i in idx_arr],
            allele0=[self.allele0[int(i)] for i in idx_arr],
            allele1=[self.allele1[int(i)] for i in idx_arr],
            snp=[self.snp[int(i)] for i in idx_arr],
        )

    def bool_mask(self, mask: object) -> "_BimSiteColumns":
        mask_arr = np.asarray(mask, dtype=np.bool_).reshape(-1)
        if int(mask_arr.shape[0]) != len(self):
            raise ValueError(
                f"BIM metadata mask length mismatch: got {mask_arr.shape[0]}, expected {len(self)}"
            )
        keep_idx = np.flatnonzero(mask_arr).astype(np.int64, copy=False)
        return self.take(keep_idx)

    def simple_snp_mask(self) -> np.ndarray:
        return np.asarray(
            [
                (len(str(a0)) == 1 and len(str(a1)) == 1)
                for a0, a1 in zip(self.allele0, self.allele1)
            ],
            dtype=np.bool_,
        )

    def to_ref_alt_dict(self) -> dict[str, object]:
        return {
            "chrom": list(self.chrom),
            "pos": list(self.pos),
            "snp": list(self.snp),
            "allele0": list(self.allele0),
            "allele1": list(self.allele1),
        }


def _coerce_bim_site_columns(obj: object) -> Optional[_BimSiteColumns]:
    if isinstance(obj, _BimSiteColumns):
        return obj
    if isinstance(obj, list):
        return _BimSiteColumns.from_sites(obj)
    if isinstance(obj, dict):
        chrom_obj = obj.get("chrom")
        pos_obj = obj.get("pos")
        snp_obj = obj.get("snp")
        allele0_obj = obj.get("allele0")
        allele1_obj = obj.get("allele1")
        if chrom_obj is None or pos_obj is None or snp_obj is None or allele0_obj is None or allele1_obj is None:
            return None
        chrom = [str(v) for v in list(chrom_obj)]
        pos = [int(v) for v in list(pos_obj)]
        snp = [str(v) for v in list(snp_obj)]
        allele0 = [str(v) for v in list(allele0_obj)]
        allele1 = [str(v) for v in list(allele1_obj)]
        return _BimSiteColumns(
            chrom=chrom,
            pos=pos,
            allele0=allele0,
            allele1=allele1,
            snp=snp,
        )
    return None


def _read_bim_site_columns(
    prefix: str,
    row_indices: Union[np.ndarray, list[int], None] = None,
) -> _BimSiteColumns:
    row_idx_arr = None
    if row_indices is not None:
        row_idx_arr = np.ascontiguousarray(
            np.asarray(row_indices, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
    if hasattr(jxrs, "load_bim_columns"):
        chrom, pos, snp, allele0, allele1 = jxrs.load_bim_columns(
            str(prefix),
            row_idx_arr,
        )
        return _BimSiteColumns(
            chrom=[str(v) for v in list(chrom)],
            pos=[int(v) for v in list(pos)],
            allele0=[str(v) for v in list(allele0)],
            allele1=[str(v) for v in list(allele1)],
            snp=[str(v) for v in list(snp)],
        )
    sites_all = _read_bim_sites(str(prefix))
    if row_idx_arr is not None:
        sites_all = [sites_all[int(i)] for i in row_idx_arr.tolist()]
    return _BimSiteColumns.from_sites(sites_all)


def _coerce_site_pos(value: object) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return 0


def _fallback_snp_name(chrom: object, pos: object) -> str:
    return f"{str(chrom).strip()}_{_coerce_site_pos(pos)}"


def _normalize_snp_name(raw_name: object, chrom: object, pos: object) -> str:
    name = str(raw_name).strip()
    if name == "" or name == "." or name.lower() == "nan":
        return _fallback_snp_name(chrom, pos)
    return name


def _make_deferred_bim_metadata(
    prefix: str,
    row_indices: Union[np.ndarray, list[int], None] = None,
    *,
    n_markers: Union[int, None] = None,
) -> dict[str, object]:
    row_idx_arr = None
    if row_indices is not None:
        row_idx_arr = np.ascontiguousarray(
            np.asarray(row_indices, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
    if n_markers is None:
        n_markers_use = int(row_idx_arr.shape[0]) if row_idx_arr is not None else 0
    else:
        n_markers_use = int(max(0, int(n_markers)))
    return {
        "_kind": "deferred_bim",
        "bed_prefix": str(prefix),
        "row_indices": row_idx_arr,
        "n_markers": n_markers_use,
    }


def _bim_metadata_len(meta_obj: object) -> int:
    if isinstance(meta_obj, _BimSiteColumns):
        return int(len(meta_obj))
    if isinstance(meta_obj, pd.DataFrame):
        return int(meta_obj.shape[0])
    if isinstance(meta_obj, dict):
        if str(meta_obj.get("_kind", "")) == "deferred_bim":
            return int(max(0, int(meta_obj.get("n_markers", 0))))
        for key in ("snp", "chrom", "pos", "allele0", "allele1"):
            if key in meta_obj:
                try:
                    return int(len(meta_obj[key]))  # type: ignore[index]
                except Exception:
                    continue
    if isinstance(meta_obj, list):
        return int(len(meta_obj))
    return 0


def _resolve_bim_site_columns_meta(
    meta_obj: object,
    *,
    prefix: Union[str, None] = None,
    row_indices: Union[np.ndarray, list[int], None] = None,
    expected_rows: Union[int, None] = None,
) -> _BimSiteColumns:
    sites = _coerce_bim_site_columns(meta_obj)
    if sites is None and isinstance(meta_obj, dict) and str(meta_obj.get("_kind", "")) == "deferred_bim":
        prefix = str(meta_obj.get("bed_prefix", prefix or "") or "").strip()
        row_idx_obj = meta_obj.get("row_indices", row_indices)
        row_indices = row_idx_obj
    if sites is None:
        prefix_use = str(prefix or "").strip()
        if prefix_use == "":
            raise ValueError("BIM metadata resolution requires a prefix or concrete metadata columns.")
        sites = _read_bim_site_columns(prefix_use, row_indices)
    if expected_rows is not None and int(len(sites)) != int(expected_rows):
        raise ValueError(
            f"BIM metadata length mismatch: loaded={len(sites)}, expected={int(expected_rows)}"
        )
    return sites


def _resolve_ref_alt_columns(
    ref_alt_obj: object,
    *,
    expected_rows: int,
    prefix: Union[str, None] = None,
    row_indices: Union[np.ndarray, list[int], None] = None,
) -> tuple[list[str], list[int], list[str], list[str], list[str]]:
    if isinstance(ref_alt_obj, pd.DataFrame):
        ref_alt_df = ref_alt_obj.reset_index(drop=True)
        cols_need = {"chrom", "pos", "snp", "allele0", "allele1"}
        if not cols_need.issubset(set(ref_alt_df.columns)):
            raise ValueError(
                "ref_alt dataframe missing required columns: chrom/pos/snp/allele0/allele1."
            )
        if int(ref_alt_df.shape[0]) != int(expected_rows):
            raise ValueError(
                f"ref_alt dataframe rows={ref_alt_df.shape[0]} != expected markers={int(expected_rows)}."
            )
        return (
            ref_alt_df["chrom"].astype(str).tolist(),
            (
                pd.to_numeric(ref_alt_df["pos"], errors="coerce")
                .fillna(0)
                .astype(np.int64)
                .tolist()
            ),
            ref_alt_df["snp"].astype(str).tolist(),
            ref_alt_df["allele0"].astype(str).tolist(),
            ref_alt_df["allele1"].astype(str).tolist(),
        )

    sites = _resolve_bim_site_columns_meta(
        ref_alt_obj,
        prefix=prefix,
        row_indices=row_indices,
        expected_rows=int(expected_rows),
    )
    chrom_col, pos_col, allele0_col, allele1_col, snp_raw = sites.columns()
    snp_col = [
        _normalize_snp_name(raw_name, chrom_name, pos_name)
        for raw_name, chrom_name, pos_name in zip(snp_raw, chrom_col, pos_col)
    ]
    return (
        [str(x) for x in chrom_col],
        [int(x) for x in pos_col],
        snp_col,
        [str(x) for x in allele0_col],
        [str(x) for x in allele1_col],
    )


def _resolve_gwas_snp_bim_path(genofile: str) -> Union[str, None]:
    path = str(safe_expanduser(str(genofile))).strip()
    if path == "":
        return None
    if path.lower().endswith(".bim") and os.path.isfile(path):
        return path

    plink_prefix = _as_plink_prefix(path)
    if plink_prefix is not None:
        bim_path = f"{plink_prefix}.bim"
        if os.path.isfile(bim_path):
            return bim_path

    file_prefix, _matrix_path = _resolve_file_input_matrix(path)
    if file_prefix:
        bim_path = f"{file_prefix}.bim"
        if os.path.isfile(bim_path):
            return bim_path

    return None


def _iter_bim_named_sites(bim_path: str) -> Iterator[tuple[str, int, str]]:
    with open(bim_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if line == "":
                continue
            toks = line.split()
            if len(toks) < 4:
                raise ValueError(f"Malformed BIM line at {bim_path}:{line_no}")
            chrom = str(toks[0]).strip()
            pos = _coerce_site_pos(toks[3])
            raw_name = toks[1] if len(toks) > 1 else "."
            yield chrom, pos, _normalize_snp_name(raw_name, chrom, pos)


def _augment_gwas_tsv_with_snp_names(
    out_tsv: str,
    genofile: str,
    *,
    logger: Union[logging.Logger, None] = None,
) -> bool:
    """
    Ensure GWAS TSV contains an explicit `snp` column.

    Naming rules:
      1) Prefer BIM SNP ID when available
      2) If BIM ID is blank / "." / NaN, fallback to "chr_pos"
      3) If no BIM exists (e.g. txt/npy with only .site), fallback to "chr_pos"
    """
    path = str(out_tsv)
    if not os.path.isfile(path):
        return False

    bim_path = _resolve_gwas_snp_bim_path(genofile)
    tmp_path = f"{path}.snp.{os.getpid()}.{uuid.uuid4().hex}.tmp"

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fin:
            header_raw = fin.readline()
            if header_raw == "":
                return False

            header = header_raw.rstrip("\r\n")
            cols = header.split("\t")
            if "snp" in cols:
                return False
            if ("chrom" not in cols) or ("pos" not in cols):
                return False

            chrom_idx = cols.index("chrom")
            pos_idx = cols.index("pos")
            insert_idx = pos_idx + 1

            bim_iter = _iter_bim_named_sites(bim_path) if bim_path is not None else None
            bim_cur = next(bim_iter, None) if bim_iter is not None else None

            with open(tmp_path, "w", encoding="utf-8", newline="") as fout:
                out_cols = cols[:insert_idx] + ["snp"] + cols[insert_idx:]
                fout.write("\t".join(out_cols) + "\n")

                for raw in fin:
                    line = raw.rstrip("\r\n")
                    if line == "":
                        fout.write("\n")
                        continue

                    parts = line.split("\t")
                    if len(parts) <= max(chrom_idx, pos_idx):
                        fout.write(line + "\n")
                        continue

                    chrom = str(parts[chrom_idx]).strip()
                    pos = _coerce_site_pos(parts[pos_idx])
                    snp_name = _fallback_snp_name(chrom, pos)

                    while bim_cur is not None:
                        bim_chrom, bim_pos, bim_name = bim_cur
                        if bim_chrom == chrom and int(bim_pos) == int(pos):
                            snp_name = str(bim_name)
                            bim_cur = next(bim_iter, None) if bim_iter is not None else None
                            break
                        bim_cur = next(bim_iter, None) if bim_iter is not None else None

                    out_parts = parts[:insert_idx] + [snp_name] + parts[insert_idx:]
                    fout.write("\t".join(out_parts) + "\n")

        _replace_file_with_retry(tmp_path, path)
        return True
    except Exception as ex:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        if logger is not None:
            _log_file_only(
                logger,
                logging.WARNING,
                f"Unable to add SNP names to {_display_path(path)}: {ex}",
            )
        return False


def _finalize_gwas_result_tsv(
    tmp_tsv: str,
    out_tsv: str,
    genofile: str,
    *,
    logger: Union[logging.Logger, None] = None,
) -> None:
    _replace_file_with_retry(str(tmp_tsv), str(out_tsv))


def _normalize_bed_memory_gb(memory_gb: Union[int, float, None]) -> float | None:
    if memory_gb is None:
        return None
    return float(
        _common_normalize_decode_memory_gb(
            memory_gb,
            default_gb=float(DEFAULT_BED_MEMORY_GB),
        )
    )


def _bed_memory_gb_to_mb(memory_gb: Union[int, float, None]) -> float:
    return float(
        _common_decode_memory_gb_to_mb(
            memory_gb,
            default_gb=float(DEFAULT_BED_MEMORY_GB),
        )
    )


def _memory_gb_for_target_decode_shape(
    row_width: int,
    target_rows: int,
    *,
    elem_bytes: int = 4,
    buffers: int = 1,
    min_gb: float = _GWAS_AUTO_MEM_MIN_GB,
) -> float:
    width = int(max(1, int(row_width)))
    rows = int(max(1, int(target_rows)))
    bytes_need = (
        width
        * rows
        * int(max(1, int(elem_bytes)))
        * int(max(1, int(buffers)))
    )
    gb_need = float(bytes_need) / float(1024 ** 3)
    return float(max(float(min_gb), gb_need))


def _gwas_model_working_buffers(model_key: str) -> int:
    key = str(model_key).strip().lower()
    if key in {"lmm", "lmm2", "fvlmm"}:
        return int(_GWAS_WORKING_BUFFERS_COPY)
    if key in {"splmm", "algwas", "farmcpu"}:
        return int(_GWAS_WORKING_BUFFERS_PACKED)
    if key in {"lm", "lm2"}:
        return int(_GWAS_WORKING_BUFFERS_LM)
    return 1


def _gwas_requested_working_buffers(
    *,
    requested_models: list[str] | set[str],
    qcov_needs_grm: bool,
) -> int:
    models = {
        str(x).strip().lower()
        for x in requested_models
        if str(x).strip() != ""
    }
    buffers = 1
    for model_key in models:
        buffers = max(buffers, _gwas_model_working_buffers(str(model_key)))
    if bool(qcov_needs_grm) or len(models & {"lmm", "lmm2", "fvlmm"}) > 0:
        buffers = max(buffers, int(_GWAS_WORKING_BUFFERS_GRM))
    return int(max(1, buffers))


def _format_gwas_memory_cfg(
    memory_gb: Union[int, float, None],
    *,
    auto_requested: bool,
) -> str:
    if memory_gb is None:
        return "auto (route-aware)"
    suffix = " (auto)" if bool(auto_requested) else ""
    return f"{float(memory_gb):.2f} GB{suffix}"


def _resolve_gwas_auto_decode_memory_gb(
    *,
    n_samples_total: int,
    n_markers_total: int,
    requested_models: list[str],
    qcov_needs_grm: bool,
) -> tuple[float, str]:
    n_total = int(max(1, int(n_samples_total)))
    m_total = int(max(1, int(n_markers_total)))
    models = {str(x).strip().lower() for x in requested_models if str(x).strip() != ""}
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()

    def _push(memory_gb: float, reason: str) -> None:
        reason_txt = str(reason).strip()
        if reason_txt == "" or reason_txt in seen:
            return
        seen.add(reason_txt)
        candidates.append((float(memory_gb), reason_txt))

    if len(models & {"lm", "lm2"}) > 0:
        lm_target_bytes = int(float(_GWAS_AUTO_MEM_LM_TARGET_MB) * 1024.0 * 1024.0)
        lm_rows = int(
            min(
                m_total,
                max(1, lm_target_bytes // max(1, 4 * n_total)),
            )
        )
        _push(
            _memory_gb_for_target_decode_shape(
                n_total,
                lm_rows,
                elem_bytes=4,
                buffers=_GWAS_WORKING_BUFFERS_LM,
            ),
            f"LM/LM2 target≈{float(_GWAS_AUTO_MEM_LM_TARGET_MB):g}MB rows={int(lm_rows)}",
        )

    if len(models & {"lmm", "lmm2", "fvlmm"}) > 0:
        copy_rows = int(min(m_total, int(_GWAS_AUTO_MEM_COPY_DECODE_BLOCK_ROWS)))
        _push(
            _memory_gb_for_target_decode_shape(
                n_total,
                copy_rows,
                elem_bytes=4,
                buffers=_GWAS_WORKING_BUFFERS_COPY,
            ),
            (
                f"LMM/FvLMM copy-aware decode block_rows={int(copy_rows)} "
                f"x{int(_GWAS_WORKING_BUFFERS_COPY)}"
            ),
        )

    packed_models = sorted(models & {"splmm", "algwas", "farmcpu"})
    if len(packed_models) > 0:
        packed_rows = int(min(m_total, int(_GWAS_AUTO_MEM_PACKED_BLOCK_ROWS)))
        packed_label = "/".join(packed_models)
        _push(
            _memory_gb_for_target_decode_shape(
                n_total,
                packed_rows,
                elem_bytes=4,
                buffers=_GWAS_WORKING_BUFFERS_PACKED,
            ),
            (
                f"{packed_label} block_rows={int(packed_rows)} "
                f"x{int(_GWAS_WORKING_BUFFERS_PACKED)}"
            ),
        )

    if bool(qcov_needs_grm) or len(models & {"lmm", "lmm2", "fvlmm"}) > 0:
        grm_rows = int(min(m_total, int(_GWAS_AUTO_MEM_GRM_BLOCK_ROWS)))
        _push(
            _memory_gb_for_target_decode_shape(
                n_total,
                grm_rows,
                elem_bytes=4,
                buffers=_GWAS_WORKING_BUFFERS_GRM,
            ),
            (
                f"GRM/PCA block_rows={int(grm_rows)} "
                f"x{int(_GWAS_WORKING_BUFFERS_GRM)}"
            ),
        )

    if len(candidates) == 0:
        return (float(DEFAULT_BED_MEMORY_GB), "fallback default=1GB")
    picked_gb, _picked_reason = max(candidates, key=lambda item: float(item[0]))
    return (float(picked_gb), "; ".join(reason for _gb, reason in candidates))


def _resolve_bed_block_rows_from_memory(
    memory_mb: Union[int, float],
    n_samples: int,
    n_snps_hint: int,
    *,
    streaming: bool,
    working_buffers: int = 1,
) -> int:
    _ = streaming
    return int(
        _common_resolve_decode_block_rows(
            int(n_samples),
            float(memory_mb),
            max_rows=max(1, int(n_snps_hint)),
            buffers=int(max(1, int(working_buffers))),
        )
    )


@contextmanager
def _bed_block_target_env(
    memory_mb: Union[int, float, None],
    *,
    needs_copy: bool = False,
    buffers: int = 1,
):
    with _common_bed_block_target_env(
        memory_mb,
        needs_copy=bool(needs_copy),
        buffers=int(max(1, int(buffers))),
    ):
        yield


def _resolve_stream_scan_chunk_size(
    chunk_size: int,
    n_snps_hint: int,
    *,
    use_spinner: bool,
    n_samples_hint: Union[int, None] = None,
    model_keys: Union[str, list[str], tuple[str, ...], None] = None,
    user_specified: bool = True,
) -> int:
    """
    Resolve effective streaming scan chunk size.

    Policy
    ------
    - Respect user-provided chunk size strictly.
    - For LM streaming with small sample size, auto-grow chunk size when
      user did not pass --memory, so per-chunk compute is large enough to
      amortize Python orchestration and file-write overhead.
    """
    if isinstance(model_keys, str):
        models = {str(model_keys).lower()}
    elif model_keys is None:
        models = set()
    else:
        models = {str(x).lower() for x in model_keys}

    _ = bool(use_spinner)
    base = max(1, int(chunk_size))
    n_snps = int(max(1, int(n_snps_hint)))
    resolved = base

    if (not bool(user_specified)) and ("lm" in models):
        n_samples = int(max(1, int(n_samples_hint or 0)))
        if n_samples > 0:
            # Heuristic target block size in memory for LM scan matrix (float32):
            # rows * n_samples * 4 bytes ~= target_bytes.
            target_mb = 192
            target_bytes = int(target_mb * 1024 * 1024)
            auto_rows = int(target_bytes // max(1, (4 * n_samples)))
            auto_rows = max(base, auto_rows)
            # Hard cap to bound transient memory while still improving throughput.
            auto_rows = min(auto_rows, min(250_000, n_snps))
            if auto_rows >= 10_000:
                auto_rows = int((auto_rows // 1_000) * 1_000)
            resolved = max(base, auto_rows)

    return min(n_snps, max(1, int(resolved)))


def _status_stage_thread_budget(threads: int) -> int:
    """
    Reserve one logical core for UI/status refresh when possible.
    """
    t = max(1, int(threads))
    return max(1, t - 1) if t > 1 else 1


@contextmanager
def _gwas_evd_stage_ctx(threads: int):
    """
    EVD / heavy dense linear algebra stage.
    """
    spec = _gwas_evd_stage_thread_plan(threads)
    with runtime_thread_stage(
        blas_threads=int(spec["blas_threads"]),
        rayon_threads=int(spec["rayon_threads"]),
    ):
        yield


def _gwas_evd_stage_thread_plan(threads: int) -> dict[str, int]:
    t = max(1, int(threads))
    return {"blas_threads": int(t), "rayon_threads": int(t)}


@contextmanager
def _gwas_scan_stage_ctx(threads: int):
    """
    GWAS scan stage (Rust/Rayon kernels):
    - Rayon uses full `-t`
    - BLAS pinned to 1 to avoid oversubscription.
    """
    spec = _gwas_scan_stage_thread_plan(threads)
    with runtime_thread_stage(
        blas_threads=int(spec["blas_threads"]),
        rayon_threads=int(spec["rayon_threads"]),
    ):
        yield


def _gwas_scan_stage_thread_plan(threads: int) -> dict[str, int]:
    t = max(1, int(threads))
    return {"blas_threads": 1, "rayon_threads": int(t)}


def _resolve_fvlmm_scan_stage_mode() -> str:
    raw = str(os.environ.get("JX_FVLMM_SCAN_STAGE", "")).strip().lower()
    if raw in {"blas-rayon", "blas_t_rayon_1", "dedicated", "blas_only"}:
        return "blas_t_rayon_1"
    if raw in {"full", "both", "legacy", "blas_t_rayon_t"}:
        return "full"
    if raw in {"generic", "scan", "blas_1_rayon_t"}:
        return "generic"
    # Keep legacy behavior as the initial default so we can benchmark apples-to-apples
    # and only flip the default after validating on real workloads.
    return "full"


@contextmanager
def _gwas_fvlmm_scan_stage_ctx(threads: int):
    """
    FvLMM scan stage.

    Modes:
    - `full` (legacy default): BLAS=t, Rayon=t
    - `blas_t_rayon_1`: BLAS=t, Rayon=1
    - `generic`: reuse the generic GWAS scan stage (BLAS=1, Rayon=t)
    """
    spec = _gwas_fvlmm_scan_stage_thread_plan(threads)
    with runtime_thread_stage(
        blas_threads=int(spec["blas_threads"]),
        rayon_threads=int(spec["rayon_threads"]),
    ):
        yield


def _gwas_fvlmm_scan_stage_thread_plan(threads: int) -> dict[str, int]:
    t = max(1, int(threads))
    mode = _resolve_fvlmm_scan_stage_mode()
    if mode == "blas_t_rayon_1":
        return {"blas_threads": int(t), "rayon_threads": 1}
    if mode == "generic":
        return _gwas_scan_stage_thread_plan(t)
    return {"blas_threads": int(t), "rayon_threads": int(t)}


def _resolve_gwas_eigh_driver(n_samples: int) -> str:
    raw = str(os.environ.get("JX_GWAS_EIGH_DRIVER", "")).strip().lower()
    if raw in {"dsyevd", "dsyevr", "auto"}:
        if raw == "auto" and int(n_samples) >= 32768:
            return "dsyevr"
        return raw
    if int(n_samples) >= 32768:
        return "dsyevr"
    accel_try_dsyevr = str(os.environ.get("JX_GWAS_EIGH_ACCEL_DSYEVR", "")).strip().lower()
    if accel_try_dsyevr in {"1", "true", "yes", "on"}:
        backend = "unknown"
        try:
            probe = getattr(jxrs, "rust_sgemm_backend", None)
            if callable(probe):
                backend = str(probe()).strip().lower()
        except Exception:
            backend = "unknown"
        if backend == "accelerate":
            try:
                switch_n = int(str(os.environ.get("JX_GWAS_EIGH_ACCEL_SWITCH_N", "2500")).strip())
            except Exception:
                switch_n = 2500
            if int(n_samples) >= max(1, int(switch_n)):
                return "dsyevr"
    return "auto"


def _resolve_gwas_eigh_impl(*, require_rust: bool) -> str:
    if bool(require_rust):
        return "rust"
    raw = str(os.environ.get("JX_GWAS_EIGH_IMPL", "")).strip().lower()
    if raw in {"rust", "scipy"}:
        return raw
    accel_try_scipy = str(os.environ.get("JX_GWAS_EIGH_ACCEL_SCIPY", "")).strip().lower()
    if accel_try_scipy in {"1", "true", "yes", "on"}:
        backend = "unknown"
        try:
            probe = getattr(jxrs, "rust_sgemm_backend", None)
            if callable(probe):
                backend = str(probe()).strip().lower()
        except Exception:
            backend = "unknown"
        if backend == "accelerate":
            return "scipy"
    return "rust"


def _gwas_eigh_from_grm(
    grm: np.ndarray,
    *,
    threads: int,
    logger: Union[logging.Logger, None] = None,
    stage_label: str = "GWAS",
    require_rust: bool = False,
    diag_ridge: float = 0.0,
    subset_idx: Optional[object] = None,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
    Eigen-decomposition for symmetric GRM with Rust-first backend.

    Returns
    -------
    eigvals, eigvecs, backend_label, elapsed_seconds
    """
    if not hasattr(jxrs, "rust_eigh_from_array_f64"):
        raise RuntimeError(
            f"{stage_label} requires rust_eigh_from_array_f64, but the symbol is unavailable."
        )
    impl_req = _resolve_gwas_eigh_impl(require_rust=bool(require_rust))
    t = max(1, int(threads))
    ridge = float(diag_ridge)

    def _matrix_file_hint_from_obj(obj) -> Optional[str]:
        cur = obj
        seen: set[int] = set()
        for _ in range(8):
            if cur is None:
                break
            oid = id(cur)
            if oid in seen:
                break
            seen.add(oid)
            try:
                _fn = getattr(cur, "filename", None)
                if _fn is not None:
                    cand = os.fspath(_fn)
                    if isinstance(cand, bytes):
                        cand = cand.decode("utf-8", errors="ignore")
                    cand = str(cand).strip()
                    low = cand.lower()
                    if cand and os.path.isfile(cand) and low.endswith((".npy", ".txt", ".tsv", ".csv")):
                        return cand
            except Exception:
                pass
            try:
                cur = getattr(cur, "base", None)
            except Exception:
                break
        return None

    matrix_file_hint: Optional[str] = None
    if impl_req == "rust" and hasattr(jxrs, "rust_eigh_from_matrix_file_f64"):
        matrix_file_hint = _matrix_file_hint_from_obj(grm)

    g_shape = np.shape(grm)
    if len(g_shape) != 2 or int(g_shape[0]) != int(g_shape[1]) or int(g_shape[0]) == 0:
        raise RuntimeError(f"{stage_label} expects non-empty square GRM, got shape={g_shape}")
    n_full = int(g_shape[0])
    aligned_base_idx_arr = _square_matrix_base_subset_idx(grm)
    aligned_view_n = n_full
    if aligned_base_idx_arr is not None:
        aligned_base_idx_arr = _normalize_square_subset_index(
            aligned_base_idx_arr,
            n_full,
            f"{stage_label} aligned GRM index",
        )
        if _is_full_identity_index(aligned_base_idx_arr, n_full):
            aligned_base_idx_arr = None
        else:
            aligned_view_n = int(aligned_base_idx_arr.shape[0])

    subset_idx_arr: Optional[np.ndarray] = None
    n_eigh = n_full
    if subset_idx is not None:
        subset_idx_arr = _normalize_square_subset_index(
            subset_idx,
            aligned_view_n,
            f"{stage_label} subset_idx",
        )
    effective_subset_idx_arr: Optional[np.ndarray] = None
    if subset_idx_arr is not None:
        if aligned_base_idx_arr is not None:
            if _is_full_identity_index(subset_idx_arr, aligned_view_n):
                effective_subset_idx_arr = aligned_base_idx_arr
            else:
                effective_subset_idx_arr = np.ascontiguousarray(
                    aligned_base_idx_arr[subset_idx_arr],
                    dtype=np.int64,
                )
        elif not _is_full_identity_index(subset_idx_arr, n_full):
            effective_subset_idx_arr = subset_idx_arr
    elif aligned_base_idx_arr is not None:
        effective_subset_idx_arr = aligned_base_idx_arr
    if effective_subset_idx_arr is not None:
        effective_subset_idx_arr = _normalize_square_subset_index(
            effective_subset_idx_arr,
            n_full,
            f"{stage_label} effective GRM subset",
        )
        if _is_full_identity_index(effective_subset_idx_arr, n_full):
            effective_subset_idx_arr = None
    if effective_subset_idx_arr is not None:
        n_eigh = int(effective_subset_idx_arr.shape[0])
    driver_req = _resolve_gwas_eigh_driver(n_eigh)

    can_use_subset_file = bool(
        impl_req == "rust"
        and matrix_file_hint is not None
        and (effective_subset_idx_arr is not None)
        and hasattr(jxrs, "rust_eigh_from_matrix_file_subset_f64")
    )

    g0: Optional[np.ndarray] = None
    g: Optional[np.ndarray] = None
    if (
        impl_req == "scipy"
        or matrix_file_hint is None
        or ((effective_subset_idx_arr is not None) and (not can_use_subset_file))
    ):
        g_src = (
            _subset_square_matrix_identity_aware(grm, effective_subset_idx_arr)
            if (effective_subset_idx_arr is not None)
            else grm
        )
        g0 = np.asarray(g_src, dtype=np.float64)
        if g0.ndim != 2 or g0.shape[0] != g0.shape[1] or g0.shape[0] == 0:
            raise RuntimeError(f"{stage_label} expects non-empty square GRM, got shape={g0.shape}")
        g = np.ascontiguousarray(g0)
        if ridge != 0.0:
            g.flat[:: g.shape[0] + 1] += ridge

    def _run_eigh_rust(
        driver_name: str,
        mat: Optional[np.ndarray],
        matrix_path: Optional[str] = None,
        subset_idx_local: Optional[np.ndarray] = None,
    ):
        with _gwas_evd_stage_ctx(t):
            if matrix_path is not None and hasattr(jxrs, "rust_eigh_from_matrix_file_f64"):
                if (
                    subset_idx_local is not None
                    and int(np.asarray(subset_idx_local).shape[0]) > 0
                    and hasattr(jxrs, "rust_eigh_from_matrix_file_subset_f64")
                ):
                    return jxrs.rust_eigh_from_matrix_file_subset_f64(
                        str(matrix_path),
                        np.ascontiguousarray(np.asarray(subset_idx_local, dtype=np.int64).reshape(-1), dtype=np.int64),
                        threads=int(t),
                        driver=str(driver_name),
                        jobz="V",
                        require_lapack=False,
                        diag_shift=float(ridge),
                    )
                return jxrs.rust_eigh_from_matrix_file_f64(
                    str(matrix_path),
                    threads=int(t),
                    driver=str(driver_name),
                    jobz="V",
                    require_lapack=False,
                    diag_shift=float(ridge),
                )
            if mat is None:
                raise RuntimeError("Rust eigh requires either matrix array or matrix file path.")
            if hasattr(jxrs, "rust_eigh_from_array_f64_inplace") and bool(mat.flags.c_contiguous) and bool(mat.flags.writeable):
                try:
                    return jxrs.rust_eigh_from_array_f64_inplace(
                        mat,
                        threads=int(t),
                        driver=str(driver_name),
                        jobz="V",
                        require_lapack=False,
                    )
                except Exception as ex_inplace:
                    mat_retry = np.ascontiguousarray(mat, dtype=np.float64)
                    try:
                        return jxrs.rust_eigh_from_array_f64(
                            mat_retry,
                            threads=int(t),
                            driver=str(driver_name),
                            jobz="V",
                            require_lapack=False,
                        )
                    except Exception as ex_copy:
                        raise RuntimeError(
                            f"Rust eigh failed after inplace retry "
                            f"(inplace={ex_inplace}; copied={ex_copy})"
                        ) from ex_copy
            return jxrs.rust_eigh_from_array_f64(
                mat,
                threads=int(t),
                driver=str(driver_name),
                jobz="V",
                require_lapack=False,
            )

    def _run_eigh_scipy(driver_name: str, mat: np.ndarray):
        try:
            from scipy import linalg as _sp_linalg
        except Exception as ex:
            raise RuntimeError(
                "SciPy eigh backend requested but scipy.linalg is unavailable."
            ) from ex
        drv = str(driver_name).strip().lower()
        scipy_driver: Optional[str]
        if drv in {"dsyevd", "syevd", "evd"}:
            scipy_driver = "evd"
        elif drv in {"dsyevr", "syevr", "evr"}:
            scipy_driver = "evr"
        else:
            scipy_driver = None
        kwargs = dict(lower=False, check_finite=False, overwrite_a=True)
        if scipy_driver is not None:
            kwargs["driver"] = scipy_driver
        with _gwas_evd_stage_ctx(t):
            t0 = time.monotonic()
            eval_raw, evec_raw = _sp_linalg.eigh(mat, **kwargs)
            elapsed = max(time.monotonic() - t0, 0.0)
        return (
            np.asarray(eval_raw, dtype=np.float64),
            np.asarray(evec_raw, dtype=np.float64),
            f"scipy_{scipy_driver or 'auto'}",
            float(elapsed),
        )

    def _run_eigh(
        driver_name: str,
        mat: Optional[np.ndarray],
        matrix_path: Optional[str] = None,
        subset_idx_local: Optional[np.ndarray] = None,
    ):
        if impl_req == "scipy":
            if mat is None:
                raise RuntimeError("SciPy eigh requires an in-memory GRM array.")
            eval_arr, evec_arr, backend_label, elapsed = _run_eigh_scipy(driver_name, mat)
            meta = {
                "stage_blas_threads": int(t),
                "rust_blas_before": None,
                "rust_blas_in_stage": None,
                "rust_blas_after": None,
            }
            return eval_arr, evec_arr, backend_label, elapsed, meta
        (
            eval_raw,
            evec_raw,
            _blas_backend,
            evd_backend,
            _n,
            _tb,
            _ti,
            _ta,
            _lapack_used,
            elapsed,
        ) = _run_eigh_rust(
            driver_name,
            mat,
            matrix_path=matrix_path,
            subset_idx_local=subset_idx_local,
        )
        meta = {
            "stage_blas_threads": int(t),
            "rust_blas_before": None if int(_tb) <= 0 else int(_tb),
            "rust_blas_in_stage": None if int(_ti) <= 0 else int(_ti),
            "rust_blas_after": None if int(_ta) <= 0 else int(_ta),
        }
        return (
            np.asarray(eval_raw, dtype=np.float64),
            np.asarray(evec_raw, dtype=np.float64) if evec_raw is not None else None,
            str(evd_backend),
            float(elapsed),
            meta,
        )

    try:
        if impl_req == "rust" and matrix_file_hint is not None:
            eval_raw, evec_raw, evd_backend, elapsed, evd_meta = _run_eigh(
                driver_req,
                None,
                matrix_path=matrix_file_hint,
                subset_idx_local=effective_subset_idx_arr,
            )
        else:
            eval_raw, evec_raw, evd_backend, elapsed, evd_meta = _run_eigh(driver_req, g)
    except Exception as ex:
        if str(driver_req).lower() == "dsyevr":
            try:
                if impl_req == "rust" and matrix_file_hint is not None:
                    eval_raw, evec_raw, evd_backend, elapsed, evd_meta = _run_eigh(
                        "dsyevd",
                        None,
                        matrix_path=matrix_file_hint,
                        subset_idx_local=effective_subset_idx_arr,
                    )
                else:
                    if g0 is None:
                        raise RuntimeError("missing GRM array for dsyevd fallback")
                    g_retry = np.ascontiguousarray(np.asarray(g0, dtype=np.float64))
                    eval_raw, evec_raw, evd_backend, elapsed, evd_meta = _run_eigh("dsyevd", g_retry)
                driver_req = "dsyevd(fallback)"
            except Exception as ex2:
                raise RuntimeError(
                    f"{stage_label} eigh failed (impl={impl_req}, driver=dsyevd fallback): {ex2}"
                ) from ex2
        else:
            raise RuntimeError(
                f"{stage_label} eigh failed (impl={impl_req}, driver={driver_req}): {ex}"
            ) from ex

    if evec_raw is None:
        raise RuntimeError(f"{stage_label} eigh returned no eigenvectors for jobz=V.")
    eigvals = np.asarray(eval_raw, dtype=np.float64).reshape(-1)
    eigvecs = np.asarray(evec_raw, dtype=np.float64)
    backend = str(evd_backend)
    jx_pkg.maybe_emit_macos_eigh_fallback_hint(
        evd_backend=backend,
        logger=logger,
    )
    if logger is not None and _gwas_logger_verbose(logger):
        subset_desc = (
            f"{n_eigh}/{n_full}" if effective_subset_idx_arr is not None else f"full/{n_full}"
        )
        _log_file_only(
            logger,
            logging.INFO,
            f"{stage_label} eigh thread cap: "
            f"BLAS={int(evd_meta.get('stage_blas_threads') or t)} "
            f"rust_blas(before={evd_meta.get('rust_blas_before')},"
            f"in_stage={evd_meta.get('rust_blas_in_stage')},"
            f"after={evd_meta.get('rust_blas_after')}) "
            f"driver={driver_req} subset={subset_desc}",
        )
        _log_file_only(
            logger,
            logging.INFO,
            f"{stage_label} eigh impl={impl_req} backend={backend} driver={driver_req} "
            f"elapsed={float(elapsed):.3f}s",
        )
    return eigvals, eigvecs, backend, float(elapsed)


def _prepare_packed_bed_once_for_gwas(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _prepare_packed_bed_once_for_gwas as _impl
    return _impl(*args, **kwargs)

def _splmm_parse_sparse_cutoff(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _splmm_parse_sparse_cutoff as _impl
    return _impl(*args, **kwargs)

def _ensure_splmm_sparse_grm(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _ensure_splmm_sparse_grm as _impl
    return _impl(*args, **kwargs)

def _splmm_normalize_sparse_grm_path(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _splmm_normalize_sparse_grm_path as _impl
    return _impl(*args, **kwargs)

def _splmm_sparse_out_prefix_for_gwas(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _splmm_sparse_out_prefix_for_gwas as _impl
    return _impl(*args, **kwargs)

def prepare_memmap_filtered_bed_for_gwas(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import prepare_memmap_filtered_bed_for_gwas as _impl
    return _impl(*args, **kwargs)

def run_fvlmm_packed_fullrank(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import run_fvlmm_packed_fullrank as _impl
    return _impl(*args, **kwargs)

def run_lm_packed_fullrank(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import run_lm_packed_fullrank as _impl
    return _impl(*args, **kwargs)

def run_lm_stream_bed_single_entry(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import run_lm_stream_bed_single_entry as _impl
    return _impl(*args, **kwargs)

def run_lmm_packed_fullrank(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import run_lmm_packed_fullrank as _impl
    return _impl(*args, **kwargs)

def run_algwas_packed_fullrank(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import run_algwas_packed_fullrank as _impl
    return _impl(*args, **kwargs)

def run_splmm_windowed_fullrank(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import run_splmm_windowed_fullrank as _impl
    return _impl(*args, **kwargs)

def _splmm_bed_logic_meta_selected_cached(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _splmm_bed_logic_meta_selected_cached as _impl
    return _impl(*args, **kwargs)

def _gwas_logic_meta_selected_cached(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _gwas_logic_meta_selected_cached as _impl
    return _impl(*args, **kwargs)

def _gwas_logic_meta_from_packed_ctx(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _gwas_logic_meta_from_packed_ctx as _impl
    return _impl(*args, **kwargs)

def _gwas_logic_meta_global_cached(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _gwas_logic_meta_global_cached as _impl
    return _impl(*args, **kwargs)

def _plink_fam_sample_ids(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _plink_fam_sample_ids as _impl
    return _impl(*args, **kwargs)

def run_splmm_packed_fullrank(*args, **kwargs):
    return run_splmm_windowed_fullrank(*args, **kwargs)

def run_chunked_gwas_lmm_lm(*args, **kwargs):
    from janusx.assoc.workflow_model_stream import run_chunked_gwas_lmm_lm as _impl
    return _impl(*args, **kwargs)

def run_chunked_gwas_streaming_shared(*args, **kwargs):
    from janusx.assoc.workflow_model_stream import run_chunked_gwas_streaming_shared as _impl
    return _impl(*args, **kwargs)

def prepare_qk_and_filter(*args, **kwargs):
    from janusx.assoc.workflow_model_farmcpu import prepare_qk_and_filter as _impl
    return _impl(*args, **kwargs)


def build_qmatrix_farmcpu(*args, **kwargs):
    from janusx.assoc.workflow_model_farmcpu import build_qmatrix_farmcpu as _impl
    return _impl(*args, **kwargs)


def run_farmcpu_fullmem(*args, **kwargs):
    from janusx.assoc.workflow_model_farmcpu import run_farmcpu_fullmem as _impl
    return _impl(*args, **kwargs)


def _run_file_dense_fast_once(
    *,
    args,
    gfile: str,
    prefix: str,
    outprefix: str,
    logger: logging.Logger,
    use_spinner: bool,
    stream_models: list[str],
    has_farmcpu: bool,
    summary_rows: list[dict[str, object]],
    saved_paths: list[str],
) -> tuple[pd.DataFrame, np.ndarray, int]:
    from janusx.pyBLUP.QK2 import GRM as _calc_grm

    def _extract_ref_alt(
        ref_alt_obj: object,
        n_markers: int,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
        chrom_col, pos_col, snp_col, allele0_col, allele1_col = _resolve_ref_alt_columns(
            ref_alt_obj,
            expected_rows=int(n_markers),
            prefix=str(prefix),
        )
        return (
            np.asarray(chrom_col, dtype=object),
            np.asarray(pos_col, dtype=np.int64),
            snp_col,
            allele0_col,
            allele1_col,
        )

    def _heter_keep_mask(geno_add: np.ndarray, het_threshold: float) -> np.ndarray:
        geno = np.asarray(geno_add, dtype=np.float32)
        m = int(geno.shape[0])
        if m == 0:
            return np.zeros((0,), dtype=bool)
        valid = np.isfinite(geno) & (geno >= 0.0)
        non_missing = np.sum(valid, axis=1)
        keep = non_missing > 0
        if not np.any(keep):
            return keep
        het_count = np.sum(np.isclose(geno, 1.0, atol=1e-6) & valid, axis=1)
        het_rate = np.zeros((m,), dtype=np.float32)
        idx = non_missing > 0
        het_rate[idx] = het_count[idx] / non_missing[idx]
        keep &= het_rate <= float(het_threshold)
        return keep

    def _apply_genetic_model(geno_add: np.ndarray, model: str) -> np.ndarray:
        m = str(model).lower()
        g = np.asarray(geno_add, dtype=np.float32)
        if m == "add":
            return g
        if m == "dom":
            return (
                np.isclose(g, 1.0, atol=1e-6) | np.isclose(g, 2.0, atol=1e-6)
            ).astype(np.float32, copy=False)
        if m == "rec":
            return np.isclose(g, 2.0, atol=1e-6).astype(np.float32, copy=False)
        if m == "het":
            return np.isclose(g, 1.0, atol=1e-6).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported genetic model for dense FILE fast route: {model}")

    def _transform_alleles(
        allele0_list: list[str],
        allele1_list: list[str],
        model: str,
    ) -> tuple[list[str], list[str]]:
        m = str(model).lower()
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
                raise ValueError(f"Unsupported genetic model for dense FILE packed route: {model}")
        return out0, out1

    process = psutil.Process()
    n_cores = detect_effective_threads()

    if _gwas_logger_verbose(logger):
        _log_file_only(
            logger,
            logging.INFO,
            "FILE packed route: preparing shared reusable genotype/FarmCPU cache once for LM/LMM/FvLMM/FarmCPU.",
        )
    farmcpu_cache_runtime = run_farmcpu_fullmem(
        args=args,
        gfile=str(gfile),
        prefix=str(prefix),
        logger=logger,
        pheno_preloaded=None,
        ids_preloaded=None,
        n_snps_preloaded=None,
        qmatrix_preloaded=None,
        cov_preloaded=None,
        use_spinner=use_spinner,
        context_prepared=False,
        summary_rows=summary_rows,
        saved_paths=saved_paths,
        trait_names=None,
        farmcpu_cache=None,
        prepare_only=True,
        emit_trait_header=False,
        preloaded_packed=None,
        emit_file_dense_warning=bool(has_farmcpu),
    )
    if not isinstance(farmcpu_cache_runtime, dict):
        raise RuntimeError("FarmCPU prepare_only did not return a reusable cache dict.")

    pheno_obj = farmcpu_cache_runtime.get("pheno")
    if not isinstance(pheno_obj, pd.DataFrame):
        raise RuntimeError("FarmCPU cache missing phenotype dataframe in FILE packed route.")
    pheno = pheno_obj
    ids = np.asarray(farmcpu_cache_runtime.get("famid"), dtype=str).reshape(-1)
    if int(ids.shape[0]) == 0:
        raise RuntimeError("FarmCPU cache has zero genotype samples in FILE packed route.")

    geno_obj = farmcpu_cache_runtime.get("geno")
    if geno_obj is None:
        raise RuntimeError(
            "FILE packed route expected a reusable genotype matrix cache entry, got None."
        )
    geno = np.ascontiguousarray(np.asarray(geno_obj, dtype=np.float32))
    if geno.ndim != 2:
        raise RuntimeError(
            f"Dense FILE packed genotype must be 2D, got ndim={geno.ndim}."
        )
    if int(geno.shape[1]) != int(ids.shape[0]):
        raise RuntimeError(
            "Dense FILE fast genotype sample count mismatch: "
            f"geno_n={geno.shape[1]}, ids_n={ids.shape[0]}."
        )
    n_snps = int(geno.shape[0])

    q_obj = farmcpu_cache_runtime.get("qmatrix")
    if q_obj is None:
        qmatrix = np.zeros((int(ids.shape[0]), 0), dtype=np.float32)
    else:
        qmatrix = np.ascontiguousarray(np.asarray(q_obj, dtype=np.float32))
    if qmatrix.ndim != 2 or int(qmatrix.shape[0]) != int(ids.shape[0]):
        raise RuntimeError(
            "Dense FILE fast Q matrix shape mismatch: "
            f"q={qmatrix.shape}, ids_n={ids.shape[0]}."
        )

    ref_alt_obj = farmcpu_cache_runtime.get("ref_alt")
    chrom_all, pos_all, snp_all, allele0_all, allele1_all = _extract_ref_alt(ref_alt_obj, n_snps)
    setattr(
        logger,
        "_janusx_gwas_task_context_rows",
        [
            ("Data Loaded", f"{int(ids.shape[0])} Samples | {int(n_snps)} SNPs"),
            ("Genotype Cache", _display_path(str(gfile))),
            ("Q Matrix", _format_gwas_q_status(qmatrix)),
            ("Covariates", "None"),
        ],
    )

    model_sequence = [m for m in stream_models if m in {"lm", "lm2", "lmm", "lmm2", "fvlmm"}]
    gm_tag = "add"
    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = list(pheno_aligned.columns)
    multi_trait_mode = len(trait_iter) > 1

    if len(model_sequence) == 0 and has_farmcpu:
        for trait_idx, pname in enumerate(trait_iter):
            trait_summary_start = len(summary_rows)
            farmcpu_cache_runtime = run_farmcpu_fullmem(
                args=args,
                gfile=str(gfile),
                prefix=str(prefix),
                logger=logger,
                pheno_preloaded=pheno_aligned,
                ids_preloaded=ids,
                n_snps_preloaded=n_snps,
                qmatrix_preloaded=qmatrix,
                cov_preloaded=None,
                use_spinner=use_spinner,
                context_prepared=True,
                summary_rows=summary_rows,
                saved_paths=saved_paths,
                trait_names=[str(pname)],
                farmcpu_cache=farmcpu_cache_runtime,
                prepare_only=False,
                emit_trait_header=True,
                preloaded_packed=None,
            )
            _emit_gwas_trait_summary(logger, summary_rows[trait_summary_start:])
            if multi_trait_mode and trait_idx != (len(trait_iter) - 1):
                logger.info("")
        _ = farmcpu_cache_runtime
        return pheno_aligned, ids, n_snps

    for trait_idx, pname in enumerate(trait_iter):
        trait_summary_start = len(summary_rows)
        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_idv = int(keep_idx.shape[0])
        if n_idv == 0:
            logger.warning(f"{pname}: no overlapping samples, skipped.")
            if multi_trait_mode and trait_idx != (len(trait_iter) - 1):
                logger.info("")
            continue

        if len(model_sequence) > 0:
            _emit_trait_header(
                logger,
                str(pname),
                n_idv,
                pve=None,
                use_spinner=bool(use_spinner),
                width=60,
            )

        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        x_cov = np.ascontiguousarray(qmatrix[keep_idx], dtype=np.float64)
        x_arg = x_cov if int(x_cov.shape[1]) > 0 else None

        geno_add_sub = np.ascontiguousarray(geno[:, keep_idx], dtype=np.float32)
        if gm_tag == "add":
            keep_site = np.ones((int(geno_add_sub.shape[0]),), dtype=bool)
        else:
            keep_site = _heter_keep_mask(geno_add_sub, float(args.het))

        if not np.any(keep_site):
            for mkey in model_sequence:
                if mkey == "fvlmm":
                    model_label = "FvLMM"
                elif mkey == "lmm2":
                    model_label = "LMM2"
                else:
                    model_label = str(mkey).upper()
                summary_rows.append(
                    {
                        "phenotype": str(pname),
                        "model": model_label,
                        "nidv": int(n_idv),
                        "eff_snp": 0,
                        "pve": None,
                        "avg_cpu": 0.0,
                        "peak_rss_gb": float(process.memory_info().rss / 1024**3),
                        "gwas_time_s": 0.0,
                        "viz_time_s": 0.0,
                        "result_file": "",
                    }
                )
                logger.info(f"{model_label}: no SNPs passed filters for trait {pname}.")
            if has_farmcpu:
                farmcpu_cache_runtime = run_farmcpu_fullmem(
                    args=args,
                    gfile=str(gfile),
                    prefix=str(prefix),
                    logger=logger,
                    pheno_preloaded=pheno_aligned,
                    ids_preloaded=ids,
                    n_snps_preloaded=n_snps,
                    qmatrix_preloaded=qmatrix,
                    cov_preloaded=None,
                    use_spinner=use_spinner,
                    context_prepared=True,
                    summary_rows=summary_rows,
                    saved_paths=saved_paths,
                    trait_names=[str(pname)],
                    farmcpu_cache=farmcpu_cache_runtime,
                    prepare_only=False,
                    emit_trait_header=False,
                    preloaded_packed=None,
                )
            _emit_gwas_trait_summary(logger, summary_rows[trait_summary_start:])
            if multi_trait_mode and trait_idx != (len(trait_iter) - 1):
                logger.info("")
            continue

        geno_add_use = np.ascontiguousarray(geno_add_sub[keep_site], dtype=np.float32)
        geno_model = np.ascontiguousarray(
            _apply_genetic_model(geno_add_use, gm_tag),
            dtype=np.float32,
        )
        p = np.asarray(np.mean(geno_add_use, axis=1, dtype=np.float64) / 2.0, dtype=np.float64)
        p = np.clip(p, 0.0, 1.0)
        af = np.ascontiguousarray(p, dtype=np.float32)

        chrom_keep = np.asarray(chrom_all[keep_site], dtype=object)
        pos_keep = np.asarray(pos_all[keep_site], dtype=np.int64)
        idx_keep = np.flatnonzero(keep_site).astype(np.int64, copy=False)
        snp_keep = [snp_all[int(i)] for i in idx_keep]
        allele0_keep = [allele0_all[int(i)] for i in idx_keep]
        allele1_keep = [allele1_all[int(i)] for i in idx_keep]
        allele0_use, allele1_use = _transform_alleles(allele0_keep, allele1_keep, gm_tag)

        kinship_sub: Union[np.ndarray, None] = None
        lmm_model_obj: Optional[Any] = None
        lmm2_model_obj: Optional[Any] = None
        fvlmm_model_obj: Optional[Any] = None

        for mkey in model_sequence:
            if mkey == "fvlmm":
                model_label = "FvLMM"
            elif mkey == "lmm2":
                model_label = "LMM2"
            else:
                model_label = str(mkey).upper()
            model_tag = str(mkey).lower()
            pname_tag = _safe_trait_file_label(pname)
            if gm_tag == "add":
                out_tsv = f"{outprefix}.{pname_tag}.{model_tag}.tsv"
                out_svg = f"{outprefix}.{pname_tag}.{model_tag}.svg"
            else:
                out_tsv = f"{outprefix}.{pname_tag}.{gm_tag}.{model_tag}.tsv"
                out_svg = f"{outprefix}.{pname_tag}.{gm_tag}.{model_tag}.svg"

            cpu_t0 = process.cpu_times()
            wall_t0 = time.monotonic()
            peak_rss = int(process.memory_info().rss)
            pve_now: Union[float, None] = None
            init_secs = 0.0

            if mkey == "lm":
                init_t0 = time.monotonic()
                mod = LM(y=y_vec, X=x_arg)
                init_secs = max(time.monotonic() - init_t0, 0.0)
                scan_t0 = time.monotonic()
                res_raw = mod.gwas(geno_model, threads=int(max(1, args.thread)))
                scan_secs = max(time.monotonic() - scan_t0, 0.0)
            else:
                if kinship_sub is None:
                    kin_t0 = time.monotonic()
                    with CliStatus(
                        "Calculating GRM from dense FILE genotype...",
                        enabled=bool(use_spinner),
                        use_process=True,
                    ) as task:
                        try:
                            kinship_sub = np.asarray(
                                _calc_grm(
                                    geno_model,
                                    log=False,
                                    chunksize=max(1, int(args.chunksize)),
                                ),
                                dtype=np.float64,
                            )
                        except Exception:
                            task.fail("Calculating GRM from dense FILE genotype ...Failed")
                            raise
                    kin_secs = max(time.monotonic() - kin_t0, 0.0)
                    _log_model_line(
                        logger,
                        model_label,
                        f"trait GRM prepared [{format_elapsed(kin_secs)}]",
                        use_spinner=bool(use_spinner),
                    )

                if mkey == "lmm":
                    if lmm_model_obj is None:
                        init_t0 = time.monotonic()
                        lmm_model_obj = LMM(
                            y=y_vec,
                            X=x_arg,
                            kinship=np.array(kinship_sub, copy=True),
                        )
                        init_secs = max(time.monotonic() - init_t0, 0.0)
                    mod = lmm_model_obj
                elif mkey == "lmm2":
                    if lmm2_model_obj is None:
                        if lmm_model_obj is not None:
                            lmm2_model_obj = LMM2.from_lmm(lmm_model_obj)
                            init_secs = 0.0
                        else:
                            init_t0 = time.monotonic()
                            lmm2_model_obj = LMM2(
                                y=y_vec,
                                X=x_arg,
                                kinship=np.array(kinship_sub, copy=True),
                            )
                            init_secs = max(time.monotonic() - init_t0, 0.0)
                    mod = lmm2_model_obj
                else:
                    if fvlmm_model_obj is None:
                        if lmm_model_obj is not None:
                            fvlmm_model_obj = FvLMM.from_lmm(lmm_model_obj)
                            init_secs = 0.0
                        elif lmm2_model_obj is not None:
                            fvlmm_model_obj = FvLMM.from_lmm(lmm2_model_obj)
                            init_secs = 0.0
                        else:
                            init_t0 = time.monotonic()
                            fvlmm_model_obj = FvLMM(
                                y=y_vec,
                                X=x_arg,
                                kinship=np.array(kinship_sub, copy=True),
                            )
                            init_secs = max(time.monotonic() - init_t0, 0.0)
                    mod = fvlmm_model_obj

                scan_t0 = time.monotonic()
                res_raw = mod.gwas(geno_model, threads=int(max(1, args.thread)))
                scan_secs = max(time.monotonic() - scan_t0, 0.0)
                try:
                    pve_tmp = float(getattr(mod, "pve"))
                    if np.isfinite(pve_tmp):
                        pve_now = pve_tmp
                except Exception:
                    pve_now = None

            peak_rss = max(peak_rss, int(process.memory_info().rss))
            cpu_t1 = process.cpu_times()
            wall_total = max(time.monotonic() - wall_t0, 1e-12)
            cpu_used = float(
                (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
            )
            avg_cpu = 100.0 * cpu_used / (wall_total * max(1, int(n_cores)))
            peak_rss_gb = float(peak_rss / 1024**3)

            res = np.ascontiguousarray(np.asarray(res_raw, dtype=np.float64))
            if res.ndim != 2 or res.shape[0] != int(geno_model.shape[0]) or res.shape[1] < 3:
                raise RuntimeError(
                    f"Dense FILE fast {model_label} returned invalid shape {res.shape}."
                )
            res_df = pd.DataFrame(
                {
                    "chrom": chrom_keep.astype(str),
                    "pos": pos_keep.astype(np.int64),
                    "snp": snp_keep,
                    "allele0": allele0_use,
                    "allele1": allele1_use,
                    "af": np.asarray(af, dtype=np.float32),
                    "beta": np.asarray(res[:, 0], dtype=np.float64),
                    "se": np.asarray(res[:, 1], dtype=np.float64),
                    "pwald": np.asarray(res[:, 2], dtype=np.float64),
                }
            )
            if res.shape[1] == 6:
                res_df["lambda"] = np.asarray(res[:, 3], dtype=np.float64)
                res_df["ml"] = np.asarray(res[:, 4], dtype=np.float64)
                res_df["plrt"] = np.asarray(res[:, 5], dtype=np.float64)

            viz_secs = 0.0

            def _write_dense_result() -> None:
                cast_map: dict[str, object] = {"pos": int, "pwald": "object"}
                if "ml" in res_df.columns:
                    cast_map["ml"] = "object"
                if "plrt" in res_df.columns:
                    cast_map["plrt"] = "object"
                out_df = res_df.astype(cast_map)
                out_df.loc[:, "pwald"] = out_df["pwald"].map(lambda x: f"{x:.4e}")
                if "ml" in out_df.columns:
                    out_df.loc[:, "ml"] = out_df["ml"].map(lambda x: f"{x:.6e}")
                if "plrt" in out_df.columns:
                    out_df.loc[:, "plrt"] = out_df["plrt"].map(lambda x: f"{x:.4e}")
                out_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=False)

            _run_result_write_with_status(
                _write_dense_result,
                use_spinner=False,
                emit_done_line=False,
            )
            saved_paths.append(str(out_tsv))
            _log_model_line(
                logger,
                model_label,
                f"Results saved to {_display_path(str(out_tsv))}",
                use_spinner=bool(use_spinner),
            )

            if bool(args.plot):
                viz_secs = _run_fastplot_with_status(
                    res_df[["chrom", "pos", "pwald"]],
                    y_vec,
                    xlabel=str(pname),
                    outpdf=out_svg,
                    use_spinner=bool(use_spinner),
                    emit_done_line=False,
                )

            summary_rows.append(
                {
                    "phenotype": str(pname),
                    "model": model_label,
                    "nidv": int(n_idv),
                    "eff_snp": int(geno_model.shape[0]),
                    "pve": (float(pve_now) if pve_now is not None else None),
                    "avg_cpu": float(avg_cpu),
                    "peak_rss_gb": float(peak_rss_gb),
                    "gwas_time_s": float(init_secs + scan_secs),
                    "viz_time_s": float(viz_secs),
                    "result_file": str(out_tsv),
                }
            )

            _log_model_line(
                logger,
                model_label,
                f"avg CPU ~ {avg_cpu:.1f}% of {n_cores} c, peak RSS ~ {peak_rss_gb:.2f} G",
                use_spinner=bool(use_spinner),
            )
            done_times: list[str] = []
            if init_secs > 0.0:
                done_times.append(format_elapsed(init_secs))
                done_times.append(format_elapsed(scan_secs))
            if bool(args.plot):
                done_times.append(format_elapsed(viz_secs))
            if (
                pve_now is not None
                and np.isfinite(float(pve_now))
                and str(model_label).lower() in {"lmm", "lmm2", "fvlmm"}
            ):
                done_msg = (
                    f"{model_label} ...pve {float(pve_now):.3f} "
                    f"[{'/'.join(done_times)}]"
                )
            else:
                done_msg = f"{model_label} ...Finished [{'/'.join(done_times)}]"
            _rich_success(
                logger,
                done_msg,
                use_spinner=bool(use_spinner),
            )

        if has_farmcpu:
            farmcpu_cache_runtime = run_farmcpu_fullmem(
                args=args,
                gfile=str(gfile),
                prefix=str(prefix),
                logger=logger,
                pheno_preloaded=pheno_aligned,
                ids_preloaded=ids,
                n_snps_preloaded=n_snps,
                qmatrix_preloaded=qmatrix,
                cov_preloaded=None,
                use_spinner=use_spinner,
                context_prepared=True,
                summary_rows=summary_rows,
                saved_paths=saved_paths,
                trait_names=[str(pname)],
                farmcpu_cache=farmcpu_cache_runtime,
                prepare_only=False,
                emit_trait_header=False,
                preloaded_packed=None,
            )
        _emit_gwas_trait_summary(logger, summary_rows[trait_summary_start:])
        if multi_trait_mode and trait_idx != (len(trait_iter) - 1):
            logger.info("")

    return pheno_aligned, ids, n_snps


def _parse_float_csv(raw: str, *, label: str) -> list[float]:
    vals: list[float] = []
    for part in str(raw).split(","):
        s = part.strip()
        if not s:
            continue
        try:
            v = float(s)
        except Exception as e:
            raise ValueError(f"{label}: invalid numeric value '{s}'.") from e
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"{label}: values must be finite and > 0.")
        vals.append(float(v))
    if len(vals) == 0:
        raise ValueError(f"{label}: at least one value is required.")
    return vals


def _dev_help_requested(argv: Optional[list[str]] = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return ("-dev" in tokens) or ("--dev" in tokens)


def _option_present(argv: Optional[list[str]], *flags: str) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    flag_set = {str(f).strip() for f in flags if str(f).strip()}
    for tok in tokens:
        t = str(tok).strip()
        if t in flag_set:
            return True
        for f in flag_set:
            if t.startswith(f + "="):
                return True
    return False


def parse_args(argv: Optional[list[str]] = None):
    show_dev_help = _dev_help_requested(argv)
    parser = CliArgumentParser(
        prog="jx gwas",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx gwas -vcf example.vcf.gz -p pheno.tsv -lmm",
            "jx gwas -hmp example.hmp.gz -p pheno.tsv -lmm",
            "jx gwas -bfile example_prefix -p pheno.tsv -lm",
            "jx gwas -h -dev",
        ]),
    )
    parser.add_argument(
        "-dev", "--dev", action="store_true", default=False, help=argparse.SUPPRESS
    )

    required_group = parser.add_argument_group("Required arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=False)
    add_common_genotype_source_args(geno_group, include_file=True)

    add_common_pheno_arg(
        required_group,
        required=False,
        help_text="Phenotype file (tab-delimited, sample IDs in the first column).",
    )

    models_group = parser.add_argument_group("Model Arguments")
    models_group.add_argument(
        "-lm", "--lm", action="store_true", default=False,
        help="Run the linear model (memmap, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-lm2", "--lm2",
        nargs="?",
        const="__SELF__",
        default=None,
        metavar="COVCOL",
        help=(
            "Run LM with SNP-by-covariate interaction terms from selected merged -c covariate columns. "
            "Column selectors are 0-based and accept items like 0, 0:3, :2, 0,3. "
            "If no interaction columns are specified, LM2 falls back to the LM fast path."
        ),
    )
    models_group.add_argument(
        "-lmm", "--lmm", action="store_true", default=False,
        help="Run the linear mixed model (memmap, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-lmm2", "--lmm2", action="store_true", default=False,
        help=(
            "Run exact LMM with Wald beta/se/pwald plus per-SNP ML/plrt output "
            "(roughly ~2x scan cost versus -lmm; default: %(default)s)."
        ),
    )
    models_group.add_argument(
        "-fvlmm", "--fvlmm", action="store_true", default=False,
        help="Run the fixed-variance LMM spectral scan using the null-model lambda for the whole GWAS (default: %(default)s).",
    )
    models_group.add_argument(
        "-splmm", "--splmm",
        dest="splmm",
        nargs="?",
        const="__SELF__",
        default=None,
        metavar="CUTOFF",
        help=(
            "Run SparseLMM (additive-only) with the GRAMMAR-gamma scan approximation. "
            "SparseLMM builds or reuses sparse GRM from the main genotype input, "
            "estimates variance components, then scans with the faster "
            "GRAMMAR-gamma denominator. "
            "Optional CUTOFF sets the sparse kinship cutoff; negative disables "
            "off-diagonal thresholding (example: -splmm 0.001; default: 0.05)."
        ),
    )
    models_group.add_argument(
        "-splmm-exact", "--splmm-exact",
        dest="splmm_exact",
        nargs="?",
        const="__SELF__",
        default=None,
        metavar="CUTOFF",
        help=(
            "Run SparseLMM with the exact sparse-Cholesky g'Pg scan denominator "
            "for every SNP. "
            "Optional CUTOFF sets the sparse kinship cutoff; negative disables "
            "off-diagonal thresholding."
        ),
    )
    models_group.add_argument(
        "-splmm-approx", "--splmm-approx",
        dest="splmm_exact",
        nargs="?",
        const="__SELF__",
        default=argparse.SUPPRESS,
        metavar="CUTOFF",
        help=argparse.SUPPRESS,
    )
    models_group.add_argument(
        "-farmcpu", "--farmcpu",
        dest="farmcpu_raw",
        action="store_true",
        default=False,
        help="Run FarmCPU (classic FEM/REM/SUPER route; default: %(default)s).",
    )
    models_group.add_argument(
        "-frgwas", "--frgwas",
        dest="farmcpu",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    models_group.add_argument(
        "-algwas", "--algwas", action="store_true", default=False,
        help=argparse.SUPPRESS,
    )

    optional_group = parser.add_argument_group("Optional Arguments")
    add_common_trait_selector_args(optional_group, dest="ncol")
    add_common_grm_option_arg(optional_group, default="1", dest="grm")
    optional_group.add_argument(
        "-spk", "--grm-sparse", type=str, default="1", dest="grm_sparse",
        help="Sparse GRM option for SparseLMM (-splmm / -splmm-exact): "
             "1 (centering), 2 (standardization), "
             "or a path to a precomputed Sparse GRM file "
             "(.spgrm/.jxgrm or GCTA/fastGWA prefix/.grm.sp; default: %(default)s).",
    )
    optional_group.add_argument(
        "-q", "--qcov", type=str, default="0",
        help=(
            "Number of principal components for Q matrix (integer >= 0). "
            "For external covariates, use -c <file> (default: %(default)s)."
        ),
    )
    add_common_covariate_file_or_site_arg(optional_group, dest="cov", default=None)
    qtn_group = optional_group.add_mutually_exclusive_group(required=False)
    qtn_group.add_argument(
        "-qvcf", "--qtn-vcf", type=str, default=None,
        help=(
            "Optional QTN-search genotype VCF for FarmCPU/ALGWAS stage1. "
            "Ignored by other models."
        ),
    )
    qtn_group.add_argument(
        "-qhmp", "--qtn-hmp", type=str, default=None,
        help=(
            "Optional QTN-search genotype HMP for FarmCPU/ALGWAS stage1. "
            "Ignored by other models; parsed as HMP regardless of suffix."
        ),
    )
    qtn_group.add_argument(
        "-qbfile", "--qtn-bfile", type=str, default=None,
        help=(
            "Optional QTN-search PLINK BED prefix for FarmCPU/ALGWAS stage1. "
            "Ignored by other models."
        ),
    )
    qtn_group.add_argument(
        "-qfile", "--qtn-file", type=str, default=None,
        help=(
            "Optional QTN-search numeric FILE matrix for FarmCPU/ALGWAS stage1. "
            "Ignored by other models."
        ),
    )
    add_common_snps_only_arg(
        optional_group,
        dest="snps_only",
        default=False,
        include_legacy_only_snps_aliases=True,
    )
    add_common_variant_filter_args(
        optional_group,
        include_maf=True,
        include_geno=True,
        include_het=True,
        maf_default=0.02,
        geno_default=0.05,
        het_default=0.0,
    )
    optional_group.add_argument(
        "--farmcpu-iter", type=int, default=30,
        help=(
            "FarmCPU max iterations (default: %(default)s)."
            if show_dev_help else argparse.SUPPRESS
        ),
    )
    optional_group.add_argument(
        "--farmcpu-threshold", type=float, default=None,
        help=(
            "FarmCPU stage1 threshold. If unset, defaults to 1 / tested_SNP_count."
            if show_dev_help else argparse.SUPPRESS
        ),
    )
    optional_group.add_argument(
        "--farmcpu-qtn-bound", type=int, default=None,
        help=(
            "Optional FarmCPU QTNbound override (default: auto)."
            if show_dev_help else argparse.SUPPRESS
        ),
    )
    optional_group.add_argument(
        "--farmcpu-nbin", type=int, default=5,
        help=(
            "FarmCPU nbin denominator for candidate grid (default: %(default)s)."
            if show_dev_help else argparse.SUPPRESS
        ),
    )
    optional_group.add_argument(
        "--farmcpu-bin-size", type=str, default="500000,5000000,50000000",
        help=(
            "FarmCPU szbin CSV (default: %(default)s)."
            if show_dev_help else argparse.SUPPRESS
        ),
    )
    add_common_memory_arg(
        optional_group,
        default=None,
        help_text=(
            "Working memory budget in GB for BED GWAS kernels. "
            "When omitted, GWAS chooses a route-aware default from loaded sample/marker counts; "
            "explicit -mem keeps the requested fixed budget."
        ),
        dest="memory",
        include_hidden_legacy_single_dash_alias=True,
    )
    optional_group.add_argument(
        "-force-model", "--force-model", action="store_true", default=False,
        help=(
            "Force the requested GWAS model(s) and disable automatic "
            "fixed-variance/mixed-model fallback switching."
        ),
    )
    optional_group.add_argument(
        "-strict-train", "--strict-train",
        "-strict-trait", "--strict-trait",
        dest="strict_train",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-global", "--global",
        dest="global_stats",
        action="store_true",
        default=False,
        help=(
            "Reuse a single full-sample GWAS row-stat pass in memory across traits/folds "
            "instead of recomputing row statistics on each training subset. "
            "Default is strict-train."
            if show_dev_help else argparse.SUPPRESS
        ),
    )
    add_common_thread_arg(optional_group, default_threads=detect_effective_threads())
    add_common_out_arg(optional_group, default=".")
    add_common_prefix_arg(optional_group, default=None)
    optional_group.add_argument(
        "-v", "--verbose", action="store_true", default=False,
        help="Show advanced diagnostics and configuration details at the end of the report.",
    )

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if any(str(tok) in {"-fast", "--fast"} for tok in raw_argv):
        parser.error("unrecognized arguments: -fast/--fast (removed; use model-specific routes)")
    if any(str(tok) in {"-fastlmm", "--fastlmm"} for tok in raw_argv):
        parser.error(
            "`-fastlmm/--fastlmm` was removed in JanusX 1.0.26. "
            "Use `-fvlmm` for fixed-lambda scans or `-lmm`/`-lmm2` for exact LMM."
        )

    args, extras = parser.parse_known_args(argv)
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
        args.ncol = parse_trait_selector_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))
    try:
        args.lm2_cov_cols = _parse_lm2_covariate_selector(getattr(args, "lm2", None))
    except ValueError as e:
        parser.error(str(e))
    try:
        args.qcov = str(_parse_qcov_dim(args.qcov))
    except ValueError as e:
        parser.error(str(e))
    if bool(getattr(args, "farmcpu", False)) and bool(getattr(args, "farmcpu_raw", False)):
        parser.error("Only one of -farmcpu / -frgwas may be specified.")
    if bool(getattr(args, "farmcpu_raw", False)):
        args.farmcpu = True
    if int(args.farmcpu_iter) < 1:
        parser.error("--farmcpu-iter must be >= 1.")
    if args.farmcpu_threshold is not None and (
        (not np.isfinite(float(args.farmcpu_threshold))) or float(args.farmcpu_threshold) <= 0
    ):
        parser.error("--farmcpu-threshold must be a finite value > 0.")
    if int(args.farmcpu_nbin) < 1:
        parser.error("--farmcpu-nbin must be >= 1.")
    if args.farmcpu_qtn_bound is not None and int(args.farmcpu_qtn_bound) < 1:
        parser.error("--farmcpu-qtn-bound must be >= 1.")
    # Normalise SparseLMM mode flags: exactly one of -splmm / -splmm-exact.
    _splmm_modes = [
        (args.splmm, "approx"),  # -splmm → GRAMMAR-gamma scan
        (getattr(args, "splmm_exact", None), "exact"),
    ]
    _active_splmm_modes = [(val, mode) for val, mode in _splmm_modes if val is not None]
    if len(_active_splmm_modes) > 1:
        parser.error("Only one of -splmm / -splmm-exact may be specified.")
    if _active_splmm_modes:
        args.splmm, args._splmm_denom_mode = _active_splmm_modes[0]
        args._splmm_null_objective_mode = (
            "raw"
            if str(args._splmm_denom_mode) == "exact"
            else "fastgwa"
        )
    else:
        args._splmm_denom_mode = None
        args._splmm_null_objective_mode = None

    try:
        args.farmcpu_bin_size = _parse_float_csv(
            args.farmcpu_bin_size,
            label="--farmcpu-bin-size",
        )
    except ValueError as e:
        parser.error(str(e))
    args._memory_user_set = bool(_option_present(argv, "-mem", "--memory", "-memory"))
    try:
        args.memory = _normalize_bed_memory_gb(getattr(args, "memory", None))
    except ValueError as e:
        parser.error(str(e))
    args._memory_auto_requested = bool(args.memory is None)
    args._memory_mb = (
        None if args.memory is None else _bed_memory_gb_to_mb(args.memory)
    )
    args._chunksize_user_set = bool(args._memory_user_set)
    args.mmap_limit = False
    args.chunksize = 10_000
    args.model = "add"
    return args


def _qtn_source_from_args(args) -> Optional[tuple[str, str, Optional[str]]]:
    if getattr(args, "qtn_vcf", None):
        return "vcf", str(args.qtn_vcf), "vcf"
    if getattr(args, "qtn_hmp", None):
        return "hmp", str(args.qtn_hmp), "hmp"
    if getattr(args, "qtn_bfile", None):
        return "bfile", str(args.qtn_bfile), "plink"
    if getattr(args, "qtn_file", None):
        return "file", str(args.qtn_file), "file"
    return None


def _prepare_qtn_packed_preload(
    args,
    *,
    logger: logging.Logger,
    use_spinner: bool,
) -> Optional[dict[str, object]]:
    src = _qtn_source_from_args(args)
    if src is None:
        return None
    if not (bool(getattr(args, "farmcpu", False)) or bool(getattr(args, "algwas", False))):
        return None
    kind, path, force_kind = src
    if kind == "bfile":
        prefix = _as_plink_prefix(path)
        if prefix is None:
            _abort_cli_input_error(
                logger,
                ValueError(f"QTN PLINK prefix is invalid or incomplete: {path}"),
            )
    else:
        try:
            prefix = _as_plink_prefix(
                prepare_cli_input_cache(
                    path,
                    snps_only=bool(getattr(args, "snps_only", False)),
                    delimiter=("," if str(path).lower().endswith(".csv") else None),
                    prefer_plink_for_txt=True,
                    force_kind=force_kind,
                    threads=int(args.thread),
                )
            )
        except ValueError as ex:
            _abort_cli_input_error(logger, ex)
        if prefix is None:
            _abort_cli_input_error(
                logger,
                ValueError(f"Unable to materialize QTN input as PLINK BED: {path}"),
            )
    with CliStatus(
        "Loading QTN genotype (Full packed)...",
        enabled=bool(use_spinner),
        use_process=True,
    ) as task:
        try:
            scan_meta = _gwas_logic_meta_global_cached(
                str(prefix),
                maf_threshold=float(getattr(args, "maf", 0.02)),
                max_missing_rate=float(getattr(args, "geno", 0.05)),
                het_threshold=float(getattr(args, "het", 0.0)),
                snps_only=bool(getattr(args, "snps_only", False)),
                outprefix=(
                    str(getattr(args, "out", ""))
                    if str(getattr(args, "out", "")).strip() != ""
                    else None
                ),
                logger=logger,
                use_spinner=False,
                emit_status=False,
            )
            _prefix_loaded, full_ids, packed_ctx, _sites_meta = _prepare_packed_bed_once_for_gwas(
                genofile=str(prefix),
                maf_threshold=float(getattr(args, "maf", 0.02)),
                max_missing_rate=float(getattr(args, "geno", 0.05)),
                het_threshold=float(getattr(args, "het", 0.0)),
                snps_only=bool(getattr(args, "snps_only", False)),
                use_spinner=False,
                preloaded_packed=None,
                load_site_meta=False,
                status_label="Loading QTN genotype (Full packed)",
                trait_prepared_meta=scan_meta,
                emit_status=False,
            )
        except Exception:
            task.fail("Loading QTN genotype (Full packed) ...Failed")
            raise
        row_idx = packed_ctx.get("row_indices", packed_ctx.get("active_row_idx"))
        if row_idx is None:
            site_keep = packed_ctx.get("site_keep")
            row_idx = np.flatnonzero(np.asarray(site_keep, dtype=np.bool_).reshape(-1)) if site_keep is not None else []
        n_active = int(np.asarray(row_idx, dtype=np.int64).reshape(-1).shape[0])
        task.complete(
            f"Loading QTN genotype (Full packed) ...Finished (n={len(full_ids)}, nSNP={n_active})"
        )
    _log_file_only(
        logger,
        logging.INFO,
        f"QTN stage1 genotype source: {path} -> {prefix}",
    )
    return {
        "kind": kind,
        "source": path,
        "prefix": str(prefix),
        "full_ids": np.asarray(full_ids, dtype=str),
        "packed_ctx": packed_ctx,
    }


def _run_gwas_pipeline(
    args=None,
    *,
    argv: Optional[list[str]] = None,
    log: bool = True,
    return_result: bool = False,
):
    t_start = time.time()
    use_spinner = bool(stdout_is_tty())
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_status = "done"
    run_error = ""
    if args is None:
        args = parse_args(argv)
    # Plotting is enabled by default for GWAS CLI, but benchmarks can disable
    # it to keep visualization out of end-to-end runtime comparisons.
    args.plot = _gwas_plot_enabled()
    args.cov = _normalize_cov_inputs(args.cov)
    args._lm2_cov_idx = None
    args._lm2_fallback_to_lm = False
    thread_budget = detect_thread_budget_info()
    detected_threads = int(thread_budget["effective_threads"])
    requested_threads = int(args.thread)
    thread_capped = False

    if args.thread <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)
    if not (0.0 <= args.het <= 0.5):
        raise ValueError("--het must be within [0, 0.5].")

    gfile, prefix = determine_genotype_source(args)

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    os.makedirs(args.out, 0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)
    outprefix = os.path.join(args.out, prefix)
    log_path = f"{outprefix}.gwas.log"
    logger = setup_logging(log_path)
    setattr(logger, "_janusx_gwas_verbose", bool(getattr(args, "verbose", False)))
    setattr(logger, "_janusx_gwas_progress_enabled", False)
    _file_only_logger = logging.getLogger("janusx.gwas.file_only")
    _file_only_logger.handlers.clear()
    _file_only_logger.setLevel(logging.INFO)
    _file_only_logger.propagate = False
    for _h in list(logger.handlers):
        if isinstance(_h, logging.FileHandler):
            _file_only_logger.addHandler(_h)
    _stream_only_logger = logging.getLogger("janusx.gwas.stream_only")
    _stream_only_logger.handlers.clear()
    _stream_only_logger.setLevel(logging.INFO)
    _stream_only_logger.propagate = False
    for _h in list(logger.handlers):
        if not isinstance(_h, logging.FileHandler):
            _stream_only_logger.addHandler(_h)
    terminal_rich = bool(use_spinner)
    report_logger = _file_only_logger if terminal_rich else logger
    terminal_logger = _stream_only_logger if terminal_rich else logger
    for _lg in {logger, _file_only_logger, _stream_only_logger, report_logger, terminal_logger}:
        try:
            setattr(_lg, "_janusx_gwas_verbose", bool(getattr(args, "verbose", False)))
            setattr(_lg, "_janusx_gwas_progress_enabled", False)
            setattr(_lg, "_janusx_gwas_terminal_rich", terminal_rich)
            setattr(_lg, "_janusx_gwas_report_logger", report_logger)
            setattr(_lg, "_janusx_gwas_stream_logger", terminal_logger if terminal_rich else None)
        except Exception:
            continue
    _prev_algwas_xtx_cache_log = os.environ.get("JX_ALGWAS_XTX_CACHE_LOG")
    if bool(getattr(args, "verbose", False)):
        os.environ["JX_ALGWAS_XTX_CACHE_LOG"] = "1"
    else:
        os.environ.pop("JX_ALGWAS_XTX_CACHE_LOG", None)

    advanced_notes: list[str] = []
    preconfig_terminal_successes: list[str] = []
    preconfig_successes_flushed = False

    def _append_advanced_note(msg: str) -> None:
        text = str(msg).strip()
        if text != "":
            advanced_notes.append(text)

    def _queue_preconfig_success(message: str) -> None:
        text = str(message).strip()
        if text != "":
            preconfig_terminal_successes.append(text)

    def _flush_preconfig_successes() -> None:
        nonlocal preconfig_successes_flushed
        if len(preconfig_terminal_successes) <= 0:
            return
        pending = list(preconfig_terminal_successes)
        preconfig_terminal_successes.clear()
        for msg in pending:
            _rich_success(logger, msg, use_spinner=bool(terminal_rich))
        preconfig_successes_flushed = True

    fvlmm_scan_spec: dict[str, int] | None = None
    if bool(getattr(args, "fvlmm", False)):
        fvlmm_scan_spec = _gwas_fvlmm_scan_stage_thread_plan(int(args.thread))
    _append_advanced_note(f"Thread detect: {format_thread_budget_summary(thread_budget)}")
    _append_advanced_note(format_affinity_cpu_summary(thread_budget))
    _append_advanced_note(
        "Thread plan: "
        f"requested={requested_threads}, using={int(args.thread)}, "
        f"fvlmm_scan_stage={_resolve_fvlmm_scan_stage_mode()}"
    )
    if fvlmm_scan_spec is not None:
        _fb = int(fvlmm_scan_spec["blas_threads"])
        _fr = int(fvlmm_scan_spec["rayon_threads"])
        if _fb == _fr:
            _append_advanced_note(f"FvLMM scan threads: BLAS={_fb}, Rayon={_fr}")
        else:
            _append_advanced_note(f"FvLMM scan BLAS threads: {_fb}")
            _append_advanced_note(f"FvLMM scan Rayon threads: {_fr}")
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds local effective={detected_threads}; "
            f"using {int(args.thread)}. "
            f"Scheduler total={thread_budget.get('scheduler_total_threads')} "
            f"({thread_budget.get('scheduler_total_source') or 'NA'})."
        )
    apply_outer_thread_cap(int(args.thread))
    # maybe_warn_non_openblas(
    #     logger=logger,
    #     strict=require_openblas_by_default(),
    # )
    gwas_summary_rows: list[dict[str, object]] = []
    saved_result_paths: list[str] = []
    ordered_result_paths: list[str] = []
    trait_order: list[str] = []
    advanced_config_rows: list[tuple[str, object]] = []
    _, _file_matrix_path_cli = _resolve_file_input_matrix(gfile)
    input_is_file_matrix = bool(_file_matrix_path_cli is not None)
    qtn_input_requested = _qtn_source_from_args(args) is not None
    requested_stream_models: list[str] = []
    if args.lm:
        requested_stream_models.append("lm")
    if getattr(args, "lm2", None) is not None:
        requested_stream_models.append("lm2")
    if args.lmm:
        requested_stream_models.append("lmm")
    if getattr(args, "lmm2", False):
        requested_stream_models.append("lmm2")
    if args.fvlmm:
        requested_stream_models.append("fvlmm")
    if args.splmm:
        requested_stream_models.append("splmm")
    if args.algwas:
        requested_stream_models.append("algwas")
    requested_memory_models = list(requested_stream_models)
    if bool(args.farmcpu):
        requested_memory_models.append("farmcpu")
    qcov_requested = str(getattr(args, "qcov", "0")).strip() not in {"", "0"}
    qcov_needs_grm = False
    standard_stream_models_requested = bool(
        args.lm
        or (getattr(args, "lm2", None) is not None)
        or args.lmm
        or getattr(args, "lmm2", False)
        or args.fvlmm
        or args.splmm
    )
    farmcpu_auto_fast = bool(args.farmcpu)
    algwas_auto_fast = bool(args.algwas)
    packed_model_routes_enabled = bool(args.farmcpu or args.algwas)
    packed_model_startup_preload = False
    gwas_row_stat_mode = "global" if bool(getattr(args, "global_stats", False)) else "strict-train"
    defer_genotype_success_until_scanmeta = bool(
        (standard_stream_models_requested or farmcpu_auto_fast or algwas_auto_fast)
        and gwas_row_stat_mode == "global"
        and str(args.model).lower() == "add"
    )
    fast_mode = bool(
        farmcpu_auto_fast or algwas_auto_fast
    )
    # FILE-input fast policy (2026-05): prefer Rust routes by materializing
    # PLINK cache then using packed/stream scans, instead of dense Python path.
    file_fast_rust_mode = bool(input_is_file_matrix and fast_mode)
    file_fast_dense_mode = False
    preinspect_ids: Optional[np.ndarray] = None
    preinspect_n_snps: Optional[int] = None
    preinspect_elapsed_secs: Optional[float] = None
    preinspect_src_for_merge: Optional[str] = None
    preinspect_warnings: list[str] = []
    if not (
        args.lm
        or (getattr(args, "lm2", None) is not None)
        or args.lmm
        or getattr(args, "lmm2", False)
        or args.fvlmm
        or args.splmm
        or args.algwas
        or args.farmcpu
    ):
        logger.error(
            "No model selected. Use -lm, -lm2, -lmm, -lmm2, -fvlmm, -splmm, -farmcpu, and/or -algwas."
        )
        raise SystemExit(1)
    if args.memory is None:
        inspect_force_kind = determine_genotype_source_force_kind_from_args(args)
        preinspect_src = _basename_only(gfile)
        preinspect_t0 = time.monotonic()
        with CliStatus(genotype_load_status_open(preinspect_src), enabled=False, use_process=True) as task:
            try:
                preinspect_ids_raw, preinspect_n_snps, preinspect_warn_msgs = _inspect_genotype_file_with_warnings(
                    gfile,
                    snps_only=bool(args.snps_only),
                    maf_threshold=float(args.maf),
                    max_missing_rate=float(args.geno),
                    het_threshold=float(args.het),
                    force_kind=inspect_force_kind,
                )
            except ValueError as ex:
                task.fail(genotype_load_status_fail(preinspect_src))
                _abort_cli_input_error(logger, ex)
            except Exception:
                task.fail(genotype_load_status_fail(preinspect_src))
                raise
        preinspect_ids = np.asarray(preinspect_ids_raw, dtype=str)
        preinspect_n_snps = int(preinspect_n_snps)
        preinspect_warnings.extend(preinspect_warn_msgs)
        preinspect_elapsed_secs = max(time.monotonic() - preinspect_t0, 0.0)
        preinspect_src_for_merge = str(preinspect_src)
        qcov_needs_grm = _gwas_qcov_prefers_grm_route(args.qcov, int(len(preinspect_ids)))
        if not bool(defer_genotype_success_until_scanmeta):
            _queue_preconfig_success(
                f"{genotype_load_status_done(preinspect_src, n_samples=len(preinspect_ids), n_snps=int(preinspect_n_snps))} "
                f"[{format_elapsed(preinspect_elapsed_secs)}]"
            )
        auto_memory_gb, auto_memory_reason = _resolve_gwas_auto_decode_memory_gb(
            n_samples_total=int(len(preinspect_ids)),
            n_markers_total=int(preinspect_n_snps),
            requested_models=list(requested_memory_models),
            qcov_needs_grm=bool(qcov_needs_grm),
        )
        args.memory = float(auto_memory_gb)
        args._memory_mb = _bed_memory_gb_to_mb(args.memory)
        args._memory_auto_reason = str(auto_memory_reason)
        auto_memory_msg = (
            "GWAS decode memory auto: "
            f"{float(args.memory):.2f} GB "
            f"(reason: {str(auto_memory_reason).strip() or 'route-aware default'}). "
            "Override with -mem/--memory to keep a fixed working-memory budget."
        )
        if _gwas_logger_verbose(logger):
            logger.info(auto_memory_msg)
        else:
            _log_file_only(logger, logging.INFO, auto_memory_msg)
    elif preinspect_ids is not None:
        qcov_needs_grm = _gwas_qcov_prefers_grm_route(args.qcov, int(len(preinspect_ids)))

    if log:
        packed_auto_mode = bool(fast_mode)
        memory_cfg = _format_gwas_memory_cfg(
            args.memory,
            auto_requested=bool(getattr(args, "_memory_auto_requested", False)),
        )
        if file_fast_rust_mode:
            bed_backend_policy = (
                "mixed (memmap default; packed only for FarmCPU/ALGWAS; FILE->BED cache)"
            )
        else:
            bed_backend_policy = (
                "mixed (memmap default; packed only for FarmCPU/ALGWAS)"
                if packed_auto_mode
                else "memmap (default)"
            )
        cfg_rows: list[tuple[str, object]] = [
            ("Genotype file", gfile),
            ("Phenotype file", args.pheno),
            ("Phenotype cols", args.ncol if args.ncol is not None else "All"),
            ("BED backend", bed_backend_policy),
            ("Packed auto route", packed_auto_mode),
            ("Models", _format_gwas_models_executed(args)),
            ("SNPs only", args.snps_only),
            ("GRM option", args.grm),
            ("Q option", args.qcov),
            ("MAF threshold", args.maf),
            ("Miss threshold", args.geno),
            ("Memory", memory_cfg),
        ]
        if qtn_input_requested and (args.farmcpu or args.algwas):
            _qtn_kind, _qtn_path, _ = _qtn_source_from_args(args) or ("", "", None)
            cfg_rows.append(("QTN stage1 input", f"{_qtn_kind}:{_qtn_path}"))
        if float(args.het) > 0.0:
            cfg_rows.append(("Het filter", f"{float(args.het):g} (keep [{float(args.het):g}, {1.0 - float(args.het):g}])"))
        if args.farmcpu:
            cfg_rows.extend(
                [
                    ("FarmCPU iter", int(args.farmcpu_iter)),
                    ("FarmCPU threshold", _format_farmcpu_threshold_display(args.farmcpu_threshold)),
                    ("FarmCPU nbin", int(args.farmcpu_nbin)),
                    ("FarmCPU QTNbound", "auto" if args.farmcpu_qtn_bound is None else int(args.farmcpu_qtn_bound)),
                    ("FarmCPU szbin", ",".join(f"{float(x):g}" for x in args.farmcpu_bin_size)),
                    (
                        "FarmCPU stage1",
                        "raw REM/SUPER" if bool(args.farmcpu_raw) else "unified FEM/REM + staged r^2 merge",
                    ),
                ]
            )
        if args.cov:
            cfg_rows.append(("Covariates", "; ".join(str(x) for x in args.cov)))
        if args.splmm:
            cfg_rows.append(("SparseLMM sparse", True))
        footer_rows: list[tuple[str, object]] = [
            (
                "Threads",
                format_requested_thread_usage(
                    requested_threads=int(requested_threads),
                    using_threads=int(args.thread),
                    detected_threads=int(detected_threads),
                ),
            ),
            ("Output prefix", outprefix),
        ]
        if terminal_rich:
            emit_cli_configuration(
                terminal_logger,
                app_title="JanusX - GWAS",
                config_title="GWAS CONFIG",
                host=socket.gethostname(),
                sections=[("General", cfg_rows)],
                footer_rows=footer_rows,
                line_max_chars=_gwas_terminal_config_line_max_chars(60),
            )
        if not bool(terminal_rich):
            _flush_preconfig_successes()

        _emit_report_banner(report_logger, "JanusX - GWAS Pipeline Analysis")
        _emit_report_block(report_logger, "System Context")
        _emit_report_kv(report_logger, "Host", socket.gethostname())
        _emit_report_kv(
            report_logger,
            "Threads",
            f"Requested={requested_threads} | Using={int(args.thread)} (Local Effective: {detected_threads})",
        )
        _emit_report_kv(
            report_logger,
            "Thread Plan",
            _format_gwas_thread_plan(args, fvlmm_scan_spec),
        )
        report_logger.info("")
        _emit_report_block(report_logger, "Configuration")
        _emit_report_kv(report_logger, "Genotype File", _display_path(str(gfile)))
        _emit_report_kv(report_logger, "Phenotype File", _display_path(str(args.pheno)))
        _emit_report_kv(
            report_logger,
            "Phenotype Cols",
            args.ncol if args.ncol is not None else "All",
        )
        _emit_report_kv(report_logger, "Models Executed", _format_gwas_models_executed(args))
        _emit_report_kv(
            report_logger,
            "Thresholds",
            f"MAF={float(args.maf):g} | Miss={float(args.geno):g}",
        )
        _emit_report_kv(report_logger, "Decode Block", memory_cfg)
        _emit_report_kv(report_logger, "Output Prefix", _display_path(str(outprefix)))
        _emit_report_kv(
            report_logger,
            "Packed Route",
            _format_gwas_packed_route_status(
                farmcpu_auto_fast=bool(farmcpu_auto_fast),
                algwas_auto_fast=bool(algwas_auto_fast),
            ),
        )
        report_logger.info("")
        _emit_report_block(report_logger, "Command")
        cmd_text = _gwas_invocation_command(argv)
        report_logger.info(f"  {cmd_text}")
        advanced_config_rows: list[tuple[str, object]] = [
            ("BED Backend", bed_backend_policy),
            ("GRM Option", args.grm),
            ("Q Option", args.qcov),
            ("Memory", memory_cfg),
            ("SNPs Only", bool(args.snps_only)),
            ("Force Model", bool(args.force_model)),
        ]
        if float(args.het) > 0.0:
            advanced_config_rows.append(
                ("Het Filter", f"{float(args.het):g} (keep [{float(args.het):g}, {1.0 - float(args.het):g}])")
            )
        if args.cov:
            advanced_config_rows.append(("Covariates", "; ".join(str(x) for x in args.cov)))
        if args.splmm:
            splmm_mode = str(getattr(args, "_splmm_denom_mode", "exact") or "exact")
            splmm_mode_display = {"approx": "grammar_gamma", "exact": "exact"}.get(
                splmm_mode,
                splmm_mode,
            )
            advanced_config_rows.append(
                ("SparseLMM Mode", splmm_mode_display)
            )
            advanced_config_rows.append(
                ("SparseLMM Null Objective", str(getattr(args, "_splmm_null_objective_mode", "fastgwa") or "fastgwa"))
            )
            advanced_config_rows.append(("Sparse GRM Input", str(getattr(args, "grm_sparse", "1"))))
        if args.farmcpu:
            advanced_config_rows.extend(
                [
                    ("FarmCPU Iter", int(args.farmcpu_iter)),
                    ("FarmCPU Threshold", _format_farmcpu_threshold_display(args.farmcpu_threshold)),
                    ("FarmCPU NBin", int(args.farmcpu_nbin)),
                    ("FarmCPU QTNbound", "auto" if args.farmcpu_qtn_bound is None else int(args.farmcpu_qtn_bound)),
                    ("FarmCPU SzBin", ",".join(f"{float(x):g}" for x in args.farmcpu_bin_size)),
                    (
                        "FarmCPU Stage1",
                        "raw REM/SUPER" if bool(args.farmcpu_raw) else "unified FEM/REM + staged r^2 merge",
                    ),
                ]
            )
    if not bool(terminal_rich):
        _flush_preconfig_successes()

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

    maf_threshold_scan = float(args.maf)
    max_missing_rate_scan = float(args.geno)
    het_threshold_scan = float(args.het)

    input_is_bed = _as_plink_prefix(gfile) is not None
    memmap_mode = bool(standard_stream_models_requested or (not bool(fast_mode)))
    mmap_limit_effective = bool(memmap_mode)
    _append_advanced_note(
        "FILE packed backend policy: Rust packed/stream via PLINK cache."
        if file_fast_rust_mode
        else (
            "BED backend policy: mixed; memmap default, packed reserved for FarmCPU/ALGWAS."
            if fast_mode
            else "BED backend policy: memmap (default)."
        )
    )
    if memmap_mode:
        _append_advanced_note(
            "Default memmap route enabled: stream-aligned windowed mmap will be used on BED/cache input "
            "(no temporary mmapbed.* PLINK cache will be created)."
        )
    if (not input_is_bed) and memmap_mode:
        _append_advanced_note(
            "Input is not direct PLINK BED; memmap route will apply after source conversion/cache to BED."
        )
    if bool(args.splmm) and (not bool(fast_mode)):
        _append_advanced_note(
            "SparseLMM default backend: BED memmap/streaming metadata."
        )
    # GRM route policy:
    # - default full-sample GRM builds prefer memmap BED
    # - packed reuse is disabled for standard GWAS model routes
    # - packed full-load remains the fallback when memmap is unavailable
    allow_packed_grm_reuse = False
    force_bed_stream_scan = bool(
        (
            args.lm
            or (getattr(args, "lm2", None) is not None)
            or args.lmm
            or getattr(args, "lmm2", False)
            or args.fvlmm
        )
        and input_is_file_matrix
        and (not bool(fast_mode))
    )
    allow_packed_grm_effective = bool(
        (not bool(file_fast_dense_mode))
        and (allow_packed_grm_reuse or force_bed_stream_scan)
    )
    if force_bed_stream_scan and bool(args.file) and (not bool(fast_mode)):
        _append_advanced_note(
            "FILE matrix input for LM/LMM/FvLMM: materializing PLINK BED cache for Rust streaming scan."
        )
    if (not bool(fast_mode)) and bool(args.lmm or getattr(args, "lmm2", False) or args.fvlmm or qcov_needs_grm):
        _append_advanced_note(
            "Streaming GWAS: GRM uses memmap path by default (packed single-entry disabled)."
        )
    elif bool(args.lmm or getattr(args, "lmm2", False) or args.fvlmm or qcov_needs_grm):
        _append_advanced_note(
            "GWAS GRM auto route: memmap stays primary for full-sample builds; packed is "
            "reused only when a compatible preloaded packed payload is already available."
        )
    elif bool(qcov_requested):
        _append_advanced_note(
            "GWAS PCA auto route: large-sample Q build uses RSVD directly from genotype; GRM is not preloaded."
        )
    if bool(args.farmcpu):
        _append_advanced_note("FarmCPU selected: enabling packed auto route.")
    _append_advanced_note(
        f"GWAS scan row-stat mode: {gwas_row_stat_mode} "
        f"({'per-trait recompute without disk cache' if gwas_row_stat_mode == 'strict-train' else 'single in-memory full-sample reuse'})."
    )
    os.environ["JX_BED_BLOCK_TARGET_MB"] = f"{float(args._memory_mb):.6g}"
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
        qtn_preloaded_packed: Optional[dict[str, object]] = None

        stream_models: list[str] = list(requested_stream_models)
        has_farmcpu = bool(args.farmcpu)
        farmcpu_handled_in_trait_loop = False
        preloaded_packed: Union[dict[str, object], None] = None
        prepared_splmm_sparse_jxgrm_path: Union[str, None] = None
        prepared_splmm_sparse_cutoff: Union[float, None] = None
        splmm_post_grm_hook: Optional[Callable[[str, Optional[str]], None]] = None

        if file_fast_dense_mode:
            if (not bool(preconfig_successes_flushed)) and len(preconfig_terminal_successes) > 0:
                _flush_preconfig_successes()
            pheno, ids, n_snps = _run_file_dense_fast_once(
                args=args,
                gfile=str(gfile),
                prefix=str(prefix),
                outprefix=str(outprefix),
                logger=logger,
                use_spinner=bool(use_spinner),
                stream_models=list(stream_models),
                has_farmcpu=bool(has_farmcpu),
                summary_rows=gwas_summary_rows,
                saved_paths=saved_result_paths,
            )
            # Dense FILE packed route has already run requested models.
            stream_models = []
            has_farmcpu = False
            farmcpu_handled_in_trait_loop = True
        else:
            stream_selected = bool(len(stream_models) > 0)
            shared_context_needed = bool(stream_selected or fast_mode)
            if args.splmm:
                prepared_splmm_sparse_cutoff, ignored_splmm_arg = _splmm_parse_sparse_cutoff(
                    args.splmm
                )
                splmm_cutoff_explicit = False
                splmm_raw = str(args.splmm).strip() if args.splmm is not None else ""
                if splmm_raw not in {"", "__SELF__"}:
                    try:
                        float(splmm_raw)
                    except Exception:
                        splmm_cutoff_explicit = False
                    else:
                        splmm_cutoff_explicit = True
                if ignored_splmm_arg is not None:
                    _emit_warning_line(
                        logger,
                        (
                            "SparseLMM only accepts an optional numeric sparse cutoff via "
                            "-splmm / -splmm-exact; "
                            f"ignoring non-numeric argument: {ignored_splmm_arg}"
                        ),
                        use_spinner=bool(use_spinner),
                    )

                # Determine sparse GRM: method (1=centering, 2=standardization) or
                # precomputed path.
                spk_val = str(getattr(args, "grm_sparse", "1")).strip()
                prepared_splmm_sparse_method: int = 1
                spk_is_path = False
                if spk_val in ("1", "2"):
                    prepared_splmm_sparse_method = int(spk_val)
                else:
                    spk_path = _splmm_normalize_sparse_grm_path(spk_val)
                    if os.path.exists(spk_path):
                        prepared_splmm_sparse_jxgrm_path = spk_path
                        spk_is_path = True
                    else:
                        raise ValueError(
                            f"--grm-sparse path does not exist: {spk_val} "
                            f"(normalized candidate: {spk_path})"
                        )

                # Always use the post-GRM hook so the loading message appears
                # at the right time (after genotype/GRM, before trait scanning).
                if spk_is_path:
                    if splmm_cutoff_explicit:
                        _emit_warning_line(
                            logger,
                            (
                                "SparseLMM received both `-spk <path>` and a SparseLMM cutoff "
                                "(`-splmm` / `-splmm-exact`); "
                                "the cutoff is ignored when a precomputed sparse GRM path is supplied."
                            ),
                            use_spinner=bool(use_spinner),
                        )
                    def _prepare_splmm_sparse_after_grm(
                        _stream_genofile_ready: str,
                        _loaded_dense_grm_path: Optional[str],
                    ) -> None:
                        src = os.path.basename(str(prepared_splmm_sparse_jxgrm_path))
                        with CliStatus(
                            f"Loading sparse GRM from {src}...",
                            enabled=bool(use_spinner),
                        ) as task:
                            task.complete(f"Loading sparse GRM from {src}...")

                    splmm_post_grm_hook = _prepare_splmm_sparse_after_grm
                else:
                    def _prepare_splmm_sparse_after_grm(
                        stream_genofile_ready: str,
                        loaded_dense_grm_path: Optional[str],
                    ) -> None:
                        nonlocal prepared_splmm_sparse_jxgrm_path
                        splmm_sparse_prefix = _as_plink_prefix(stream_genofile_ready)
                        if splmm_sparse_prefix is None:
                            return
                        dense_grm_path = None
                        sparse_out_prefix = str(splmm_sparse_prefix)
                        if loaded_dense_grm_path is not None and str(loaded_dense_grm_path).strip() != "":
                            dense_grm_path = str(loaded_dense_grm_path)
                        elif str(getattr(args, "grm", "1")).strip() not in {"", "1", "2"}:
                            dense_grm_path = str(getattr(args, "grm"))
                        sparse_out_prefix = _splmm_sparse_out_prefix_for_gwas(
                            str(splmm_sparse_prefix),
                            None,
                            outprefix=str(outprefix),
                            dense_grm_path=dense_grm_path,
                            logger=logger,
                        )
                        prepared_splmm_sparse_jxgrm_path = _ensure_splmm_sparse_grm(
                            str(splmm_sparse_prefix),
                            sample_indices=None,
                            out_prefix=str(sparse_out_prefix),
                            dense_grm_path=dense_grm_path,
                            cutoff=float(prepared_splmm_sparse_cutoff),
                            maf_threshold=float(maf_threshold_scan),
                            max_missing_rate=float(max_missing_rate_scan),
                            het_threshold=float(het_threshold_scan),
                            snps_only=bool(args.snps_only),
                            threads=int(args.thread),
                            logger=logger,
                            use_spinner=bool(use_spinner),
                            method=prepared_splmm_sparse_method,
                        )

                    splmm_post_grm_hook = _prepare_splmm_sparse_after_grm
            post_grm_hook: Optional[Callable[[str, Optional[str]], None]] = splmm_post_grm_hook
            qtn_post_grm_requested = bool(qtn_input_requested and (args.farmcpu or args.algwas))
            if qtn_post_grm_requested:
                base_post_grm_hook = post_grm_hook

                def _prepare_qtn_after_grm(
                    _stream_genofile_ready: str,
                    _loaded_dense_grm_path: Optional[str],
                ) -> None:
                    nonlocal qtn_preloaded_packed
                    if qtn_preloaded_packed is None:
                        qtn_preloaded_packed = _prepare_qtn_packed_preload(
                            args,
                            logger=logger,
                            use_spinner=use_spinner,
                        )

                if base_post_grm_hook is None:
                    post_grm_hook = _prepare_qtn_after_grm
                else:
                    def _prepare_splmm_then_qtn_after_grm(
                        stream_genofile_ready: str,
                        loaded_dense_grm_path: Optional[str],
                    ) -> None:
                        base_post_grm_hook(stream_genofile_ready, loaded_dense_grm_path)
                        _prepare_qtn_after_grm(stream_genofile_ready, loaded_dense_grm_path)

                    post_grm_hook = _prepare_splmm_then_qtn_after_grm
            if shared_context_needed:
                if terminal_rich:
                    _section(terminal_logger, "GWAS task")
                if (not bool(preconfig_successes_flushed)) and len(preconfig_terminal_successes) > 0:
                    _flush_preconfig_successes()
                _append_advanced_note("Prepare shared context (phenotype/genotype meta/GRM/Q/cov)")
                pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m, genofile_stream, preloaded_packed = prepare_streaming_context(
                    genofile=genofile_stream,
                    phenofile=args.pheno,
                    pheno_cols=args.ncol,
                    maf_threshold=maf_threshold_scan,
                    max_missing_rate=max_missing_rate_scan,
                    genetic_model=args.model,
                    het_threshold=het_threshold_scan,
                    memory_mb=float(args._memory_mb),
                    mgrm=args.grm,
                    pcdim=args.qcov,
                    cov_inputs=args.cov,
                    threads=args.thread,
                    require_kinship=(args.lmm or getattr(args, "lmm2", False) or args.fvlmm),
                    logger=logger,
                    use_spinner=use_spinner,
                    snps_only=bool(args.snps_only),
                    allow_packed_grm=bool(allow_packed_grm_effective),
                    preload_packed_context=bool(packed_model_startup_preload),
                    require_bed_stream=bool(
                        stream_selected or qtn_input_requested or packed_model_routes_enabled
                    ),
                    post_grm_hook=post_grm_hook,
                    force_kind=determine_genotype_source_force_kind_from_args(args),
                    preinspected_ids=preinspect_ids,
                    preinspected_n_snps=preinspect_n_snps,
                    preinspected_genotype_elapsed_secs=preinspect_elapsed_secs,
                    preinspected_genotype_src=preinspect_src_for_merge,
                    preinspect_warnings=preinspect_warnings,
                    working_buffers=_gwas_requested_working_buffers(
                        requested_models=list(requested_memory_models),
                        qcov_needs_grm=bool(qcov_needs_grm),
                    ),
                    prewarm_global_scanmeta=bool(
                        gwas_row_stat_mode == "global"
                        and str(args.model).lower() == "add"
                    ),
                    scanmeta_outprefix=str(outprefix),
                )
                if getattr(args, "lm2", None) is not None:
                    lm2_selector = list(getattr(args, "lm2_cov_cols", []) or [])
                    if cov_all is None:
                        args._lm2_fallback_to_lm = True
                        args._lm2_cov_idx = None
                        if "lm2" in stream_models:
                            if "lm" in stream_models:
                                stream_models = [m for m in stream_models if m != "lm2"]
                            else:
                                stream_models = ["lm" if m == "lm2" else m for m in stream_models]
                        _emit_warning_line(
                            logger,
                            "LM2 received no external covariates from -c; falling back to LM.",
                            use_spinner=bool(use_spinner),
                        )
                    elif len(lm2_selector) == 0:
                        args._lm2_fallback_to_lm = True
                        args._lm2_cov_idx = None
                        if "lm2" in stream_models:
                            if "lm" in stream_models:
                                stream_models = [m for m in stream_models if m != "lm2"]
                            else:
                                stream_models = ["lm" if m == "lm2" else m for m in stream_models]
                        _emit_warning_line(
                            logger,
                            "LM2 received no explicit interaction columns; falling back to LM.",
                            use_spinner=bool(use_spinner),
                        )
                    else:
                        args._lm2_cov_idx = _resolve_lm2_covariate_indices(
                            cov_all,
                            lm2_selector,
                        )
                farmcpu_genofile = str(genofile_stream)
                args.chunksize = _resolve_bed_block_rows_from_memory(
                    float(args._memory_mb),
                    int(len(ids)),
                    int(n_snps),
                    streaming=True,
                    working_buffers=_gwas_requested_working_buffers(
                        requested_models=list(requested_memory_models),
                        qcov_needs_grm=bool(qcov_needs_grm),
                    ),
                )
                if (
                    bool(packed_model_startup_preload)
                    and (not packed_preload_is_ready(preloaded_packed))
                    and (not packed_preload_is_disabled(preloaded_packed))
                ):
                    try:
                        prefix0, full_ids0, packed_ctx0, sites0 = _prepare_packed_bed_once_for_gwas(
                            genofile=genofile_stream,
                            maf_threshold=float(maf_threshold_scan),
                            max_missing_rate=float(max_missing_rate_scan),
                            het_threshold=float(het_threshold_scan),
                            snps_only=bool(args.snps_only),
                            use_spinner=bool(use_spinner),
                            preloaded_packed=None,
                            load_site_meta=False,
                        )
                        preloaded_packed = {
                            "prefix": str(prefix0),
                            "full_ids": np.asarray(full_ids0, dtype=str),
                            "packed_ctx": packed_ctx0,
                        }
                    except Exception as ex:
                        logger.warning(
                            f"Packed preload unavailable; falling back to on-demand packed load. reason={ex}"
                        )
                        preloaded_packed = packed_preload_failure_state(genofile_stream, ex)
                _flush_pending_gwas_overlap_line(
                    logger,
                    use_spinner=bool(use_spinner),
                )
                if terminal_rich and len(stream_models) > 0:
                    _phase_split(terminal_logger)
                logger_task_rows: list[tuple[str, object]] = [
                    ("Data Loaded", f"{int(len(ids))} Samples | {int(n_snps)} SNPs"),
                    ("Genotype Cache", _display_path(str(genofile_stream))),
                    ("GWAS Stats", str(gwas_row_stat_mode)),
                    ("GRM Status", _format_gwas_grm_status(grm, args.grm)),
                    ("Q Matrix", _format_gwas_q_status(qmatrix)),
                    ("Covariates", _format_gwas_cov_status(cov_all)),
                ]
                if bool(args.splmm):
                    logger_task_rows.append(
                        (
                            "Sparse GRM",
                            _format_gwas_sparse_status(
                                prepared_splmm_sparse_jxgrm_path,
                                sparse_requested=bool(args.splmm),
                                sparse_cutoff=prepared_splmm_sparse_cutoff,
                                sparse_source=getattr(args, "grm_sparse", "1"),
                            ),
                        )
                    )
                setattr(logger, "_janusx_gwas_task_context_rows", logger_task_rows)
            else:
                if args.farmcpu:
                    pheno = _load_phenotype_with_status(
                        args.pheno,
                        args.ncol,
                        logger,
                        id_col=0,
                        use_spinner=use_spinner,
                    )
                    setattr(
                        logger,
                        "_janusx_gwas_task_context_rows",
                        [("Phenotype Rows", int(pheno.shape[0]))],
                    )

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
        farmcpu_genofile = str(genofile_stream)

        # -------------------------------
        # 1) Rust-only model routes
        # -------------------------------
        if len(stream_models) > 0:
            rust_only_model_routes = True
            if rust_only_model_routes:
                # v3: model routes execute via the trait/model task plan (Rust route planner
                # + thin Python executor) so single-model paths no longer bypass the unified route.
                use_trait_grouped_fast = True
                if use_trait_grouped_fast:
                    farmcpu_cache_prefill: Union[dict[str, object], None] = None
                    farmcpu_cache_runtime: Union[dict[str, object], None] = None
                    if has_farmcpu:
                        context_prepared = bool(pheno is not None and ids is not None and n_snps is not None)
                        if (
                            packed_preload_is_ready(preloaded_packed)
                            and pheno is not None
                            and ids is not None
                            and qmatrix is not None
                        ):
                            try:
                                packed_ctx_prefill = preloaded_packed.get("packed_ctx")
                                if isinstance(packed_ctx_prefill, dict):
                                    row_idx_prefill = packed_ctx_prefill.get("row_indices")
                                    if row_idx_prefill is None:
                                        row_idx_prefill = packed_ctx_prefill.get("active_row_idx")
                                    if row_idx_prefill is None:
                                        site_keep_prefill = packed_ctx_prefill.get("site_keep")
                                        if site_keep_prefill is not None:
                                            site_keep_arr = np.asarray(site_keep_prefill, dtype=np.bool_).reshape(-1)
                                            row_idx_prefill = np.flatnonzero(site_keep_arr).astype(np.int64, copy=False)
                                    row_idx_prefill_arr = None
                                    if row_idx_prefill is not None:
                                        row_idx_prefill_arr = np.asarray(row_idx_prefill, dtype=np.int64).reshape(-1)
                                    full_ids_prefill = np.asarray(
                                        preloaded_packed.get("full_ids", ids),
                                        dtype=str,
                                    )
                                    packed_idx_map: Union[np.ndarray, None] = None
                                    try:
                                        id_to_full = {sid: i for i, sid in enumerate(full_ids_prefill)}
                                        packed_idx_map = np.asarray(
                                            [id_to_full[str(sid)] for sid in np.asarray(ids, dtype=str)],
                                            dtype=np.int64,
                                        )
                                    except Exception:
                                        packed_idx_map = None
                                    q_prefill = (
                                        np.asarray(qmatrix, dtype="float32")
                                        if qmatrix is not None
                                        else np.zeros((int(np.asarray(ids).shape[0]), 0), dtype="float32")
                                    )
                                    farmcpu_cache_prefill = {
                                        "pheno": pheno,
                                        "famid": np.asarray(ids, dtype=str),
                                        "geno": None,
                                        "packed_ctx": packed_ctx_prefill,
                                        "packed_full_ids": full_ids_prefill,
                                        "ref_alt": _make_deferred_bim_metadata(
                                            str(preloaded_packed.get("prefix", farmcpu_genofile)),
                                            row_idx_prefill_arr,
                                            n_markers=(
                                                int(row_idx_prefill_arr.shape[0])
                                                if row_idx_prefill_arr is not None
                                                else 0
                                            ),
                                        ),
                                        "qmatrix": q_prefill,
                                        "packed_sample_idx": packed_idx_map,
                                    }
                            except Exception:
                                farmcpu_cache_prefill = None
                        farmcpu_cache_runtime = run_farmcpu_fullmem(
                            args=args,
                            gfile=farmcpu_genofile,
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
                            farmcpu_cache=farmcpu_cache_prefill,
                            prepare_only=True,
                            emit_trait_header=False,
                            preloaded_packed=preloaded_packed,
                            qtn_preloaded_packed=qtn_preloaded_packed,
                        )
                    require_dispatch_v2 = bool(args.algwas)
                    task_plan = None
                    rust_plan_errors: list[str] = []
                    if hasattr(jxrs, "gwas_trait_model_dispatch_v2"):
                        try:
                            task_plan = jxrs.gwas_trait_model_dispatch_v2(
                                [str(m) for m in stream_models],
                                [str(t) for t in trait_order],
                                bool(has_farmcpu),
                            )
                        except Exception as ex:
                            rust_plan_errors.append(f"gwas_trait_model_dispatch_v2: {ex}")
                    if (task_plan is None) and require_dispatch_v2:
                        err_text = (
                            "; ".join(rust_plan_errors)
                            if len(rust_plan_errors) > 0
                            else "gwas_trait_model_dispatch_v2 unavailable"
                        )
                        raise RuntimeError(
                            "ALGWAS requires Rust GWAS dispatcher v2; "
                            f"unable to build task plan ({err_text})."
                        )
                    if task_plan is None and hasattr(jxrs, "gwas_trait_model_schedule"):
                        try:
                            task_plan = jxrs.gwas_trait_model_schedule(
                                [str(m) for m in stream_models],
                                [str(t) for t in trait_order],
                                bool(has_farmcpu),
                            )
                        except Exception as ex:
                            rust_plan_errors.append(f"gwas_trait_model_schedule: {ex}")
                    plan_models: set[str] = set()
                    if task_plan is not None:
                        try:
                            for task_item in task_plan:
                                if hasattr(task_item, "get"):
                                    mk_plan = str(task_item.get("model", "")).lower().strip()
                                    if mk_plan != "":
                                        plan_models.add(mk_plan)
                        except Exception as ex:
                            rust_plan_errors.append(f"task-plan-parse: {ex}")
                            task_plan = None
                    if task_plan is not None:
                        requested_models = {
                            str(m).lower().strip() for m in stream_models if str(m).strip() != ""
                        }
                        if ("lm2" in requested_models) and ("lm2" not in plan_models):
                            rust_plan_errors.append(
                                "rust task planner returned no lm2 tasks; falling back to Python task plan"
                            )
                            task_plan = None
                        elif (len(requested_models) > 0) and (len(plan_models) == 0):
                            rust_plan_errors.append(
                                "rust task planner returned an empty task plan; falling back to Python task plan"
                            )
                            task_plan = None
                    if task_plan is None:
                        task_plan = []
                        for ti, trait_name_use in enumerate(trait_order):
                            for mi, model_name_use in enumerate(stream_models):
                                task_plan.append(
                                    {
                                        "model": str(model_name_use),
                                        "trait": str(trait_name_use),
                                        "emit_trait_header": bool(mi == 0),
                                        "emit_blank_after": bool(mi == (len(stream_models) - 1) and ti < (len(trait_order) - 1)),
                                    }
                                )
                        if len(task_plan) == 0:
                            err_text = "; ".join(rust_plan_errors) if len(rust_plan_errors) > 0 else "no rust scheduler symbols"
                            raise RuntimeError(
                                "Rust-only GWAS mode requires Rust task scheduler/dispatcher; "
                                f"unable to build model task plan ({err_text})."
                            )
                    def _run_route_algwas_packed(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        run_algwas_packed_fullrank(
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            outprefix=outprefix,
                            maf_threshold=maf_threshold_scan,
                            max_missing_rate=max_missing_rate_scan,
                            genetic_model=args.model,
                            het_threshold=het_threshold_scan,
                            chunk_size=args.chunksize,
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
                            trait_names=trait_one,
                            emit_trait_header=bool(emit_trait_header_model),
                            preloaded_packed=preloaded_packed,
                            qtn_preloaded_packed=qtn_preloaded_packed,
                            trait_prepared_meta=trait_prepared_meta,
                        )

                    def _run_route_splmm_windowed(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        preloaded_packed: Union[dict[str, object], None] = None,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        run_splmm_windowed_fullrank(
                            genofile=genofile_stream,
                            splmm_source=args.splmm,
                            splmm_sparse_cutoff=prepared_splmm_sparse_cutoff,
                            splmm_sparse_jxgrm_path=prepared_splmm_sparse_jxgrm_path,
                            splmm_sparse_method=prepared_splmm_sparse_method,
                            pheno=pheno,
                            ids=ids,
                            outprefix=outprefix,
                            maf_threshold=maf_threshold_scan,
                            max_missing_rate=max_missing_rate_scan,
                            genetic_model=args.model,
                            het_threshold=het_threshold_scan,
                            chunk_size=args.chunksize,
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
                            trait_names=trait_one,
                            emit_trait_header=bool(emit_trait_header_model),
                            preloaded_packed=preloaded_packed,
                            trait_prepared_meta=trait_prepared_meta,
                            force_model=bool(args.force_model),
                            scan_mode=str(args._splmm_denom_mode or "exact"),
                            null_objective_mode=str(args._splmm_null_objective_mode or "fastgwa"),
                        )

                    def _run_route_fvlmm_stream(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        run_chunked_gwas_lmm_lm(
                            model_name="fvlmm",
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            n_snps=n_snps,
                            outprefix=outprefix,
                            maf_threshold=maf_threshold_scan,
                            max_missing_rate=max_missing_rate_scan,
                            genetic_model=args.model,
                            het_threshold=het_threshold_scan,
                            chunk_size=args.chunksize,
                            mmap_limit=mmap_limit_effective,
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
                            trait_names=trait_one,
                            emit_trait_header=bool(emit_trait_header_model),
                            chunk_size_user_set=bool(
                                getattr(args, "_chunksize_user_set", True)
                            ),
                            force_model=bool(args.force_model),
                            trait_prepared_meta=trait_prepared_meta,
                        )

                    def _run_route_lm_stream(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        run_chunked_gwas_lmm_lm(
                            model_name="lm",
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            n_snps=n_snps,
                            outprefix=outprefix,
                            maf_threshold=maf_threshold_scan,
                            max_missing_rate=max_missing_rate_scan,
                            genetic_model=args.model,
                            het_threshold=het_threshold_scan,
                            chunk_size=args.chunksize,
                            mmap_limit=mmap_limit_effective,
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
                            trait_names=trait_one,
                            emit_trait_header=bool(emit_trait_header_model),
                            chunk_size_user_set=bool(
                                getattr(args, "_chunksize_user_set", True)
                            ),
                            force_model=bool(args.force_model),
                            trait_prepared_meta=trait_prepared_meta,
                        )

                    def _run_route_lm2_stream(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        run_chunked_gwas_lmm_lm(
                            model_name="lm2",
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            n_snps=n_snps,
                            outprefix=outprefix,
                            maf_threshold=maf_threshold_scan,
                            max_missing_rate=max_missing_rate_scan,
                            genetic_model=args.model,
                            het_threshold=het_threshold_scan,
                            chunk_size=args.chunksize,
                            mmap_limit=mmap_limit_effective,
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
                            trait_names=trait_one,
                            emit_trait_header=bool(emit_trait_header_model),
                            chunk_size_user_set=bool(
                                getattr(args, "_chunksize_user_set", True)
                            ),
                            force_model=bool(args.force_model),
                            lm2_covariate_indices=getattr(args, "_lm2_cov_idx", None),
                            trait_prepared_meta=trait_prepared_meta,
                        )

                    def _run_route_lmm_stream(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        run_chunked_gwas_lmm_lm(
                            model_name="lmm",
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            n_snps=n_snps,
                            outprefix=outprefix,
                            maf_threshold=maf_threshold_scan,
                            max_missing_rate=max_missing_rate_scan,
                            genetic_model=args.model,
                            het_threshold=het_threshold_scan,
                            chunk_size=args.chunksize,
                            mmap_limit=mmap_limit_effective,
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
                            trait_names=trait_one,
                            emit_trait_header=bool(emit_trait_header_model),
                            chunk_size_user_set=bool(
                                getattr(args, "_chunksize_user_set", True)
                            ),
                            force_model=bool(args.force_model),
                            trait_prepared_meta=trait_prepared_meta,
                        )

                    def _run_route_lmm2_stream(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        run_chunked_gwas_lmm_lm(
                            model_name="lmm2",
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            n_snps=n_snps,
                            outprefix=outprefix,
                            maf_threshold=maf_threshold_scan,
                            max_missing_rate=max_missing_rate_scan,
                            genetic_model=args.model,
                            het_threshold=het_threshold_scan,
                            chunk_size=args.chunksize,
                            mmap_limit=mmap_limit_effective,
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
                            trait_names=trait_one,
                            emit_trait_header=bool(emit_trait_header_model),
                            chunk_size_user_set=bool(
                                getattr(args, "_chunksize_user_set", True)
                            ),
                            force_model=bool(args.force_model),
                            trait_prepared_meta=trait_prepared_meta,
                        )

                    def _run_route_farmcpu(
                        trait_one: list[str],
                        emit_trait_header_model: bool,
                        trait_prepared_meta: Union[dict[str, object], None] = None,
                    ) -> None:
                        nonlocal farmcpu_cache_runtime, farmcpu_handled_in_trait_loop
                        farmcpu_cache_runtime = run_farmcpu_fullmem(
                            args=args,
                            gfile=farmcpu_genofile,
                            prefix=prefix,
                            logger=logger,
                            pheno_preloaded=pheno,
                            ids_preloaded=ids,
                            n_snps_preloaded=n_snps,
                            qmatrix_preloaded=qmatrix,
                            cov_preloaded=cov_all,
                            use_spinner=use_spinner,
                            context_prepared=True,
                            summary_rows=gwas_summary_rows,
                            saved_paths=saved_result_paths,
                            trait_names=trait_one,
                            farmcpu_cache=farmcpu_cache_runtime,
                            emit_trait_header=bool(emit_trait_header_model),
                            preloaded_packed=preloaded_packed,
                            qtn_preloaded_packed=qtn_preloaded_packed,
                            trait_prepared_meta=trait_prepared_meta,
                        )
                        farmcpu_handled_in_trait_loop = True

                    route_handlers = {
                        "splmm": _run_route_splmm_windowed,
                        "algwas": _run_route_algwas_packed,
                        "lm_stream": _run_route_lm_stream,
                        "lm2_stream": _run_route_lm2_stream,
                        "lmm_stream": _run_route_lmm_stream,
                        "lmm2_stream": _run_route_lmm2_stream,
                        "fvlmm_stream": _run_route_fvlmm_stream,
                        "farmcpu": _run_route_farmcpu,
                    }
                    route_aliases = {
                        "lm_stream": "lm_stream",
                        "lm_memmap": "lm_stream",
                        "lm_rust": "lm_stream",
                        "lm2_stream": "lm2_stream",
                        "lm2_memmap": "lm2_stream",
                        "lm2_rust": "lm2_stream",
                        "lmm_stream": "lmm_stream",
                        "lmm_memmap": "lmm_stream",
                        "lmm_rust": "lmm_stream",
                        "lmm2_stream": "lmm2_stream",
                        "lmm2_memmap": "lmm2_stream",
                        "lmm2_rust": "lmm2_stream",
                        "fvlmm_memmap": "fvlmm_stream",
                        "fvlmm_rust": "fvlmm_stream",
                        "farm": "farmcpu",
                    }
                    stream_group_routes = {"lm_stream", "lm2_stream", "lmm_stream", "lmm2_stream", "fvlmm_stream"}
                    meta_capable_routes = stream_group_routes | {"splmm", "algwas", "farmcpu"}
                    trait_shared_meta_cache: dict[str, dict[str, object]] = {}
                    global_trait_meta_shared: Union[dict[str, object], None] = None
                    normalized_tasks: list[dict[str, object]] = []
                    for task_item in task_plan:
                        mk = str(task_item.get("model", "")).lower().strip()
                        route = str(task_item.get("route", "")).lower().strip()
                        if not route:
                            if mk == "lm":
                                route = "lm_stream"
                            elif mk == "lm2":
                                route = "lm2_stream"
                            elif mk == "lmm":
                                route = "lmm_stream"
                            elif mk == "lmm2":
                                route = "lmm2_stream"
                            elif mk == "fvlmm":
                                route = "fvlmm_stream"
                            elif mk == "splmm":
                                route = "splmm"
                            elif mk == "algwas":
                                route = "algwas"
                            elif mk == "farmcpu":
                                route = "farmcpu"
                            else:
                                route = "unknown"
                        route = route_aliases.get(route, route)
                        normalized_tasks.append(
                            {
                                "model": mk,
                                "route": route,
                                "trait": str(task_item.get("trait", "")),
                                "emit_trait_header": bool(
                                    task_item.get("emit_trait_header", False)
                                ),
                                "emit_blank_after": bool(
                                    task_item.get("emit_blank_after", False)
                                ),
                            }
                        )
                    prefix_meta = (
                        _as_plink_prefix(genofile_stream)
                        if (
                            genofile_stream is not None
                            and pheno is not None
                            and ids is not None
                            and str(args.model).lower() == "add"
                        )
                        else None
                    )
                    ids_arr_meta = (
                        np.asarray(ids, dtype=str).reshape(-1)
                        if ids is not None
                        else None
                    )
                    full_ids_meta = (
                        _plink_fam_sample_ids(str(prefix_meta))
                        if prefix_meta is not None
                        else None
                    )
                    id_to_full_meta = (
                        {sid: i for i, sid in enumerate(full_ids_meta)}
                        if full_ids_meta is not None
                        else None
                    )
                    gwas_meta_window_mb = int(
                        max(
                            1,
                            float(os.environ.get("JX_BED_BLOCK_TARGET_MB", "512")),
                        )
                    )

                    def _trait_meta_needed_later(start_idx: int, trait_name: str) -> bool:
                        for idx in range(int(start_idx), len(normalized_tasks)):
                            future_trait = str(normalized_tasks[idx].get("trait", ""))
                            if future_trait != trait_name:
                                continue
                            future_route = str(
                                normalized_tasks[idx].get("route", "")
                            ).lower().strip()
                            if future_route in meta_capable_routes:
                                return True
                        return False

                    def _build_trait_sample_indices(trait_name: str) -> Union[np.ndarray, None]:
                        if pheno is None or ids_arr_meta is None or id_to_full_meta is None:
                            return None
                        _, sameidx_meta = _trait_values_and_mask(pheno, trait_name)
                        keep_idx_meta = np.flatnonzero(sameidx_meta).astype(np.int64, copy=False)
                        if int(keep_idx_meta.shape[0]) == 0:
                            return None
                        trait_ids_meta = np.asarray(ids_arr_meta[keep_idx_meta], dtype=str)
                        try:
                            return np.ascontiguousarray(
                                np.asarray(
                                    [id_to_full_meta[str(sid)] for sid in trait_ids_meta],
                                    dtype=np.int64,
                                ),
                                dtype=np.int64,
                            )
                        except KeyError:
                            return None

                    def _trait_prepared_meta_needs_build(
                        trait_name: str,
                        route_name: str,
                    ) -> tuple[bool, Union[np.ndarray, None]]:
                        route_key = str(route_name).lower().strip()
                        if (
                            route_key not in meta_capable_routes
                            or prefix_meta is None
                            or str(args.model).lower() != "add"
                            or gwas_row_stat_mode != "strict-train"
                        ):
                            return False, None
                        cached_meta = trait_shared_meta_cache.get(trait_name)
                        if cached_meta is not None:
                            return False, None
                        sample_idx_meta = _build_trait_sample_indices(trait_name)
                        if sample_idx_meta is None:
                            return False, None
                        return True, sample_idx_meta

                    def _resolve_trait_prepared_meta(
                        trait_name: str,
                        route_name: str,
                        sample_idx_meta_prefetched: Union[np.ndarray, None] = None,
                    ) -> Union[dict[str, object], None]:
                        nonlocal global_trait_meta_shared
                        route_key = str(route_name).lower().strip()
                        if (
                            route_key not in meta_capable_routes
                            or prefix_meta is None
                            or str(args.model).lower() != "add"
                        ):
                            return None
                        if gwas_row_stat_mode != "strict-train":
                            if global_trait_meta_shared is not None:
                                return global_trait_meta_shared
                            global_trait_meta_shared = _gwas_logic_meta_global_cached(
                                str(prefix_meta),
                                maf_threshold=float(maf_threshold_scan),
                                max_missing_rate=float(max_missing_rate_scan),
                                het_threshold=float(het_threshold_scan),
                                snps_only=bool(args.snps_only),
                                outprefix=str(outprefix),
                                logger=logger,
                                use_spinner=False,
                            )
                            return global_trait_meta_shared

                        cached_meta = trait_shared_meta_cache.get(trait_name)
                        if cached_meta is not None:
                            return cached_meta
                        sample_idx_meta = sample_idx_meta_prefetched
                        if sample_idx_meta is None:
                            sample_idx_meta = _build_trait_sample_indices(trait_name)
                        if sample_idx_meta is None:
                            return None
                        with CliStatus(
                            "Computing trait-subset row statistics...",
                            enabled=bool(use_spinner),
                            use_process=True,
                        ) as task:
                            try:
                                cached_meta = _gwas_logic_meta_selected_cached(
                                    str(prefix_meta),
                                    sample_indices=sample_idx_meta,
                                    maf_threshold=float(maf_threshold_scan),
                                    max_missing_rate=float(max_missing_rate_scan),
                                    het_threshold=float(het_threshold_scan),
                                    snps_only=bool(args.snps_only),
                                    mmap_window_mb=int(gwas_meta_window_mb),
                                    outprefix=str(outprefix),
                                    logger=logger,
                                    threads=int(args.thread),
                                    use_cache=False,
                                )
                            except Exception:
                                task.fail("Computing trait-subset row statistics ...Failed")
                                raise
                            task.complete("Computing trait-subset row statistics ...Finished")
                        trait_shared_meta_cache[trait_name] = cached_meta
                        return cached_meta

                    task_idx = 0
                    current_trait_summary_name = ""
                    current_trait_summary_start = len(gwas_summary_rows)
                    while task_idx < len(normalized_tasks):
                        task_item = normalized_tasks[task_idx]
                        mk = str(task_item.get("model", "")).lower().strip()
                        route = str(task_item.get("route", "")).lower().strip()
                        trait_name_use = str(task_item.get("trait", ""))
                        if trait_name_use != current_trait_summary_name:
                            current_trait_summary_name = str(trait_name_use)
                            current_trait_summary_start = len(gwas_summary_rows)
                        emit_trait_header_model = bool(
                            task_item.get("emit_trait_header", False)
                        )
                        emit_blank_after = bool(task_item.get("emit_blank_after", False))
                        prefetched_trait_meta_indices: Union[np.ndarray, None] = None
                        if bool(emit_trait_header_model):
                            meta_build_needed, prefetched_trait_meta_indices = (
                                _trait_prepared_meta_needs_build(
                                    trait_name_use,
                                    route,
                                )
                            )
                            if meta_build_needed:
                                _emit_trait_header(
                                    logger,
                                    trait_name_use,
                                    int(prefetched_trait_meta_indices.shape[0]),
                                    pve=None,
                                    use_spinner=bool(use_spinner),
                                    width=60,
                                )
                                emit_trait_header_model = False
                        trait_meta_shared = _resolve_trait_prepared_meta(
                            trait_name_use,
                            route,
                            sample_idx_meta_prefetched=prefetched_trait_meta_indices,
                        )

                        if route in stream_group_routes and mk != "lm2":
                            group_end = task_idx + 1
                            grouped_models: list[str] = [mk]
                            while group_end < len(normalized_tasks):
                                next_item = normalized_tasks[group_end]
                                next_route = str(next_item.get("route", "")).lower().strip()
                                next_trait = str(next_item.get("trait", ""))
                                next_model = str(next_item.get("model", "")).lower().strip()
                                if (
                                    next_route not in stream_group_routes
                                    or next_trait != trait_name_use
                                    or next_model == "lm2"
                                ):
                                    break
                                grouped_models.append(next_model)
                                group_end += 1
                            if len(grouped_models) >= 2:
                                run_chunked_gwas_streaming_shared(
                                    model_names=grouped_models,
                                    trait_name=trait_name_use,
                                    genofile=genofile_stream,
                                    pheno=pheno,
                                    ids=ids,
                                    n_snps=n_snps,
                                    outprefix=outprefix,
                                    maf_threshold=maf_threshold_scan,
                                    max_missing_rate=max_missing_rate_scan,
                                    genetic_model=args.model,
                                    het_threshold=het_threshold_scan,
                                    chunk_size=args.chunksize,
                                    mmap_limit=mmap_limit_effective,
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
                                    chunk_size_user_set=bool(
                                        getattr(args, "_chunksize_user_set", True)
                                    ),
                                    force_model=bool(args.force_model),
                                    emit_trait_header=bool(emit_trait_header_model),
                                    trait_prepared_meta=trait_meta_shared,
                                )
                                next_idx = group_end
                                trait_done = (
                                    next_idx >= len(normalized_tasks)
                                    or str(normalized_tasks[next_idx].get("trait", "")) != trait_name_use
                                )
                                if trait_done:
                                    _emit_gwas_trait_summary(
                                        logger,
                                        gwas_summary_rows[current_trait_summary_start:],
                                    )
                                if bool(
                                    normalized_tasks[group_end - 1].get(
                                        "emit_blank_after", False
                                    )
                                ):
                                    logger.info("")
                                if not _trait_meta_needed_later(
                                    group_end, trait_name_use
                                ) and gwas_row_stat_mode == "strict-train":
                                    trait_shared_meta_cache.pop(trait_name_use, None)
                                task_idx = group_end
                                continue

                        trait_one = [trait_name_use]
                        route_handler = route_handlers.get(route)
                        if route_handler is not None:
                            if route == "splmm":
                                route_handler(
                                    trait_one,
                                    bool(emit_trait_header_model),
                                    preloaded_packed=None,
                                    trait_prepared_meta=trait_meta_shared,
                                )
                            elif route in stream_group_routes:
                                route_handler(
                                    trait_one,
                                    bool(emit_trait_header_model),
                                    trait_prepared_meta=trait_meta_shared,
                                )
                            elif route in {"algwas", "farmcpu"}:
                                route_handler(
                                    trait_one,
                                    bool(emit_trait_header_model),
                                    trait_prepared_meta=trait_meta_shared,
                                )
                            else:
                                route_handler(trait_one, bool(emit_trait_header_model))
                        else:
                            raise RuntimeError(
                                "Rust-only GWAS mode received unsupported task route: "
                                f"route={route}, model={mk}"
                            )
                        next_idx = task_idx + 1
                        trait_done = (
                            next_idx >= len(normalized_tasks)
                            or str(normalized_tasks[next_idx].get("trait", "")) != trait_name_use
                        )
                        if trait_done:
                            _emit_gwas_trait_summary(
                                logger,
                                gwas_summary_rows[current_trait_summary_start:],
                            )
                        if emit_blank_after:
                            logger.info("")
                        if not _trait_meta_needed_later(
                            task_idx + 1, trait_name_use
                        ) and gwas_row_stat_mode == "strict-train":
                            trait_shared_meta_cache.pop(trait_name_use, None)
                        task_idx += 1
            else:
                raise RuntimeError(
                    "Rust-only GWAS mode cannot enter Python streaming fallback routes."
                )

        # FarmCPU (same model flow; no separate task section banner)
        if has_farmcpu and (not farmcpu_handled_in_trait_loop):
            context_prepared = bool(pheno is not None and ids is not None and n_snps is not None)
            emit_farmcpu_trait_header = len(stream_models) == 0
            trait_names_full = [str(t) for t in trait_order] if len(trait_order) > 0 else None
            farmcpu_cache_prefill: Union[dict[str, object], None] = None
            def _force_farmcpu_trait_prepared_deferred_load(
                cache_obj: Union[dict[str, object], None],
            ) -> Union[dict[str, object], None]:
                if not isinstance(cache_obj, dict):
                    return cache_obj
                cache_obj["packed_ctx"] = None
                cache_obj["ref_alt"] = None
                cache_obj["_defer_packed_trait_load"] = True
                return cache_obj
            if (
                bool(fast_mode)
                and packed_preload_is_ready(preloaded_packed)
                and pheno is not None
                and ids is not None
                and qmatrix is not None
            ):
                try:
                    packed_ctx_prefill = preloaded_packed.get("packed_ctx")
                    if isinstance(packed_ctx_prefill, dict):
                        row_idx_prefill = packed_ctx_prefill.get("row_indices")
                        if row_idx_prefill is None:
                            row_idx_prefill = packed_ctx_prefill.get("active_row_idx")
                        if row_idx_prefill is None:
                            site_keep_prefill = packed_ctx_prefill.get("site_keep")
                            if site_keep_prefill is not None:
                                site_keep_arr = np.asarray(site_keep_prefill, dtype=np.bool_).reshape(-1)
                                row_idx_prefill = np.flatnonzero(site_keep_arr).astype(np.int64, copy=False)
                        row_idx_prefill_arr = None
                        if row_idx_prefill is not None:
                            row_idx_prefill_arr = np.asarray(row_idx_prefill, dtype=np.int64).reshape(-1)
                        full_ids_prefill = np.asarray(
                            preloaded_packed.get("full_ids", ids),
                            dtype=str,
                        )
                        packed_idx_map: Union[np.ndarray, None] = None
                        try:
                            id_to_full = {sid: i for i, sid in enumerate(full_ids_prefill)}
                            packed_idx_map = np.asarray(
                                [id_to_full[str(sid)] for sid in np.asarray(ids, dtype=str)],
                                dtype=np.int64,
                            )
                        except Exception:
                            packed_idx_map = None
                        q_prefill = (
                            np.asarray(qmatrix, dtype="float32")
                            if qmatrix is not None
                            else np.zeros((int(np.asarray(ids).shape[0]), 0), dtype="float32")
                        )
                        farmcpu_cache_prefill = {
                            "pheno": pheno,
                            "famid": np.asarray(ids, dtype=str),
                            "geno": None,
                            "packed_ctx": packed_ctx_prefill,
                            "packed_full_ids": full_ids_prefill,
                            "ref_alt": _make_deferred_bim_metadata(
                                str(preloaded_packed.get("prefix", farmcpu_genofile)),
                                row_idx_prefill_arr,
                                n_markers=(
                                    int(row_idx_prefill_arr.shape[0])
                                    if row_idx_prefill_arr is not None
                                    else 0
                                ),
                            ),
                            "qmatrix": q_prefill,
                            "packed_sample_idx": packed_idx_map,
                        }
                except Exception:
                    farmcpu_cache_prefill = None
            farmcpu_cache_runtime = run_farmcpu_fullmem(
                args=args,
                gfile=farmcpu_genofile,
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
                trait_names=trait_names_full,
                farmcpu_cache=farmcpu_cache_prefill,
                prepare_only=True,
                emit_trait_header=False,
                preloaded_packed=preloaded_packed,
                qtn_preloaded_packed=qtn_preloaded_packed,
            )
            _flush_pending_gwas_overlap_line(
                logger,
                use_spinner=bool(use_spinner),
            )
            if terminal_rich:
                _phase_split(terminal_logger)
            trait_names_seq = (
                [str(t) for t in trait_names_full]
                if trait_names_full is not None
                else [str(t) for t in trait_order]
            )
            strict_trait_meta_ready = False
            strict_prefix_meta = None
            strict_ids_arr_meta = None
            strict_id_to_full_meta = None
            global_trait_meta_single: Union[dict[str, object], None] = None
            if (
                gwas_row_stat_mode == "strict-train"
                and str(args.model).lower() == "add"
                and pheno is not None
                and ids is not None
            ):
                strict_prefix_meta = _as_plink_prefix(genofile_stream)
                if strict_prefix_meta is not None:
                    try:
                        strict_ids_arr_meta = np.asarray(ids, dtype=str).reshape(-1)
                        strict_full_ids_meta = _plink_fam_sample_ids(
                            str(strict_prefix_meta)
                        )
                        strict_id_to_full_meta = {
                            sid: i for i, sid in enumerate(strict_full_ids_meta)
                        }
                        strict_trait_meta_ready = True
                    except Exception:
                        strict_trait_meta_ready = False
                        strict_prefix_meta = None
                        strict_ids_arr_meta = None
                        strict_id_to_full_meta = None
            elif (
                gwas_row_stat_mode == "global"
                and str(args.model).lower() == "add"
            ):
                global_prefix_meta = _as_plink_prefix(genofile_stream)
                if global_prefix_meta is not None:
                    global_trait_meta_single = _gwas_logic_meta_global_cached(
                        str(global_prefix_meta),
                        maf_threshold=float(maf_threshold_scan),
                        max_missing_rate=float(max_missing_rate_scan),
                        het_threshold=float(het_threshold_scan),
                        snps_only=bool(args.snps_only),
                        outprefix=str(outprefix),
                        logger=logger,
                        use_spinner=False,
                    )
            if bool(strict_trait_meta_ready) and len(trait_names_seq) > 0:
                gwas_meta_window_mb = int(
                    max(
                        1,
                        float(os.environ.get("JX_BED_BLOCK_TARGET_MB", "512")),
                    )
                )
                for trait_idx_use, trait_name_use in enumerate(trait_names_seq):
                    trait_meta_single: Union[dict[str, object], None] = None
                    emit_trait_header_now = bool(emit_farmcpu_trait_header)
                    trait_summary_start = len(gwas_summary_rows)
                    sample_idx_meta = None
                    try:
                        _, sameidx_meta = _trait_values_and_mask(pheno, trait_name_use)
                        keep_idx_meta = np.flatnonzero(sameidx_meta).astype(
                            np.int64,
                            copy=False,
                        )
                        if int(keep_idx_meta.shape[0]) > 0:
                            trait_ids_meta = np.asarray(
                                strict_ids_arr_meta[keep_idx_meta],
                                dtype=str,
                            )
                            sample_idx_meta = np.ascontiguousarray(
                                np.asarray(
                                    [
                                        strict_id_to_full_meta[str(sid)]
                                        for sid in trait_ids_meta
                                    ],
                                    dtype=np.int64,
                                ),
                                dtype=np.int64,
                            )
                    except Exception:
                        sample_idx_meta = None
                    if sample_idx_meta is not None and int(sample_idx_meta.shape[0]) > 0:
                        if emit_trait_header_now:
                            _emit_trait_header(
                                logger,
                                trait_name_use,
                                int(sample_idx_meta.shape[0]),
                                pve=None,
                                use_spinner=bool(use_spinner),
                                width=60,
                            )
                            emit_trait_header_now = False
                        with CliStatus(
                            "Computing trait-subset row statistics...",
                            enabled=bool(use_spinner),
                            use_process=True,
                        ) as task:
                            try:
                                trait_meta_single = _gwas_logic_meta_selected_cached(
                                    str(strict_prefix_meta),
                                    sample_indices=sample_idx_meta,
                                    maf_threshold=float(maf_threshold_scan),
                                    max_missing_rate=float(max_missing_rate_scan),
                                    het_threshold=float(het_threshold_scan),
                                    snps_only=bool(args.snps_only),
                                    mmap_window_mb=int(gwas_meta_window_mb),
                                    outprefix=str(outprefix),
                                    logger=logger,
                                    threads=int(args.thread),
                                    use_cache=False,
                                )
                            except Exception:
                                task.fail(
                                    "Computing trait-subset row statistics ...Failed"
                                )
                                raise
                            task.complete(
                                "Computing trait-subset row statistics ...Finished"
                            )
                    farmcpu_cache_runtime = _force_farmcpu_trait_prepared_deferred_load(
                        farmcpu_cache_runtime,
                    )
                    farmcpu_cache_runtime = run_farmcpu_fullmem(
                        args=args,
                        gfile=farmcpu_genofile,
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
                        trait_names=[str(trait_name_use)],
                        farmcpu_cache=farmcpu_cache_runtime,
                        emit_trait_header=bool(emit_trait_header_now),
                        preloaded_packed=preloaded_packed,
                        qtn_preloaded_packed=qtn_preloaded_packed,
                        trait_prepared_meta=trait_meta_single,
                    )
                    _emit_gwas_trait_summary(
                        logger,
                        gwas_summary_rows[trait_summary_start:],
                    )
                    if trait_idx_use < (len(trait_names_seq) - 1):
                        logger.info("")
            else:
                if global_trait_meta_single is not None:
                    farmcpu_cache_runtime = _force_farmcpu_trait_prepared_deferred_load(
                        farmcpu_cache_runtime,
                    )
                trait_names_seq = [] if trait_names_full is None else [str(t) for t in trait_names_full]
                for trait_idx_use, trait_name_use in enumerate(trait_names_seq):
                    trait_summary_start = len(gwas_summary_rows)
                    farmcpu_cache_runtime = run_farmcpu_fullmem(
                        args=args,
                        gfile=farmcpu_genofile,
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
                        trait_names=[str(trait_name_use)],
                        farmcpu_cache=farmcpu_cache_runtime,
                        emit_trait_header=bool(emit_farmcpu_trait_header),
                        preloaded_packed=preloaded_packed,
                        qtn_preloaded_packed=qtn_preloaded_packed,
                        trait_prepared_meta=global_trait_meta_single,
                    )
                    _emit_gwas_trait_summary(
                        logger,
                        gwas_summary_rows[trait_summary_start:],
                    )
                    if trait_idx_use < (len(trait_names_seq) - 1):
                        logger.info("")

        if len(gwas_summary_rows) > 0:
            for row in gwas_summary_rows:
                pnm = str(row.get("phenotype", ""))
                row["pheno_col_idx"] = int(trait_col_map.get(pnm, -1))
        ordered_result_paths = _ordered_saved_result_paths(
            gwas_summary_rows,
            saved_result_paths,
        )
        if len(ordered_result_paths) > 0:
            report_logger.info("")
            _emit_report_block(report_logger, "Outputs Generated")
            for p in ordered_result_paths:
                report_logger.info(f"  * {_display_path(p)}")
            if terminal_rich:
                terminal_result_paths = _terminal_saved_result_paths(ordered_result_paths)
                if len(terminal_result_paths) > 0:
                    saved_body = "\n".join([f"  {_display_path(p)}" for p in terminal_result_paths])
                    _rich_success(terminal_logger, f"Results saved:\n{saved_body}")
        if bool(getattr(args, "verbose", False)):
            report_logger.info("")
            _emit_report_block(report_logger, "Advanced Parameters")
            for key, value in advanced_config_rows:
                _emit_report_kv(report_logger, str(key), value)
            for note in advanced_notes:
                report_logger.info(f"  Note             : {note}")

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
    finally:
        if _prev_algwas_xtx_cache_log is None:
            os.environ.pop("JX_ALGWAS_XTX_CACHE_LOG", None)
        else:
            os.environ["JX_ALGWAS_XTX_CACHE_LOG"] = _prev_algwas_xtx_cache_log

    if terminal_rich:
        lt = time.localtime()
        endinfo = (
            f"\nFinished. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
            f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
            f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
        )
        _rich_success(terminal_logger, endinfo)

    report_logger.info("")
    _emit_report_kv(report_logger, "Total Wall Time", f"{max(time.time() - t_start, 0.0):.2f}s")
    _emit_report_kv(report_logger, "Finished At", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report_logger.info(_REPORT_RULE)

    if bool(return_result):
        return {
            "status": str(run_status),
            "error": str(run_error),
            "genofile": str(gfile),
            "outprefix": str(outprefix),
            "log_file": str(log_path),
            "summary_rows": [dict(x) for x in _ordered_gwas_summary_rows(gwas_summary_rows)],
            "result_files": [str(x) for x in ordered_result_paths],
            "elapsed_sec": float(max(time.time() - t_start, 0.0)),
            "traits": [str(x) for x in trait_order],
        }


def main(
    argv: Optional[list[str]] = None,
    log: bool = True,
    return_result: bool = False,
):
    args = parse_args(argv)
    from janusx.assoc.runner import run_gwas_args
    return run_gwas_args(args, log=log, return_result=return_result)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
