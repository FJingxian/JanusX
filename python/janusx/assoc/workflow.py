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
  - Default: LM/LMM/FastLMM run in streaming mode.
  - With `-fast` (or when `-farmcpu` is selected), packed BED is loaded once
    and reused for full-rust packed routes when available.
  - FarmCPU always runs on the full in-memory genotype matrix.

Caching
-------
  - Genotype cache uses a single base cache prefix (no JSON cache metadata):
      * genotype cache (VCF/HMP): ~{geno_prefix}.bed/.bim/.fam
  - GRM and PCA caches for streaming LMM/LM runs remain parameterized:
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
from janusx.gfreader import (
    load_genotype_chunks,
    inspect_genotype_file,
    auto_mmap_window_mb,
)
from janusx.pyBLUP.QK2 import QK
from janusx.pyBLUP.assoc import LMM, LM, FastLMM, farmcpu
from janusx import janusx as jxrs
from janusx.script._common.log import setup_logging
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_plink_prefix_exists,
    safe_expanduser,
    safe_resolve,
)
from janusx.script._common.status import (
    CliStatus,
    is_skip_status_text,
    print_success,
    print_warning,
    format_elapsed,
    should_animate_status,
    stdout_is_tty,
)
from janusx.script._common.progress import build_rich_progress, rich_progress_available
from janusx.script._common.threads import (
    apply_outer_thread_cap,
    detect_effective_threads,
    maybe_warn_non_openblas,
    runtime_thread_stage,
    require_openblas_by_default,
)
from janusx.script._common.gwas_history import record_gwas_run
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.genoio import (
    basename_only as _basename_only,
    determine_genotype_source_from_args as determine_genotype_source,
    read_id_file as _read_id_file,
)
from janusx.script._common.packedctx import prepare_packed_ctx_from_plink

from janusx.script._common.colspec import parse_zero_based_index_specs
from janusx.assoc.workflow_ui import (
    _format_progress_metric,
    _log_file_only,
    _log_info,
    _emit_plain_info_line,
    _emit_warning_line,
    _log_model_line,
    _rich_success,
    _ProgressAdapter,
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
    _is_cache_warning_message,
    _pca_cache_path,
    _cache_lock,
    genotype_cache_prefix,
)

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


def _plrt_from_beta_se(
    beta: np.ndarray,
    se: np.ndarray,
    *,
    n_obs: int,
    df: int,
) -> np.ndarray:
    """
    Compute 1-df LM LRT p-values from beta/se without re-scanning genotype.
    """
    b = np.asarray(beta, dtype=np.float64).reshape(-1)
    s = np.asarray(se, dtype=np.float64).reshape(-1)
    out = np.full(b.shape, np.nan, dtype=np.float64)
    n = int(n_obs)
    d = int(df)
    if n <= 0 or d <= 0:
        return out
    ok = np.isfinite(b) & np.isfinite(s) & (s > 0.0)
    if not np.any(ok):
        return out
    t2 = np.square(b[ok] / s[ok])
    stat = float(n) * np.log1p(t2 / float(d))
    out[ok] = _chi2_sf_df1_vec(stat)
    return out


def _gwas_use_packed_fullrust_routes() -> bool:
    """
    Whether optional packed single-entry full-rust routes are enabled.

    Currently used for LMM/FastLMM packed route gating. LM packed route is
    enabled automatically for PLINK BED additive scans when available.
    """
    raw = str(os.environ.get("JX_GWAS_USE_PACKED_FULLRUST", "")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _gwas_use_packed_fastlmm_scan() -> bool:
    """
    Whether to use packed full-rust FastLMM scan in `-fast` mode.

    Default is True so `-fast` follows the packed path. Set
    `JX_GWAS_USE_PACKED_FASTLMM_SCAN=0` to force streaming prepared-chunk scan.
    """
    raw = str(os.environ.get("JX_GWAS_USE_PACKED_FASTLMM_SCAN", "")).strip().lower()
    if raw == "":
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return raw in {"1", "true", "yes", "y", "on"}


def _gwas_use_packed_grm_build() -> bool:
    """
    Whether GRM construction should prefer packed BED + Rust CBLAS route.

    Default is True. In current GWAS policy, this toggle is mainly applied
    in fast/farmcpu mode; non-fast mode prefers streaming GRM.
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

    if low.endswith(".hmp.gz") or low.endswith(".hmp"):
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
    plain_progress = (not status_enabled) and (cache_kind in {"vcf", "hmp", "txt"})

    # For direct PLINK prefixes (no cache-target metadata), inspect directly.
    # For VCF/TXT sources, always run the threaded+monitored path below so that
    # cache rebuilds triggered inside inspect_genotype_file() are still visible.
    if cache_kind == "":
        # PLINK metadata inspect can still take noticeable time on very large
        # .bim/.fam inputs; use process-backed spinner so animation does not
        # freeze even when Rust/Python call path holds the GIL.
        with CliStatus(
            f"Loading genotype from {src}...",
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
    allow_packed_full_load: bool = False,
    preloaded_packed: Union[dict[str, object], None] = None,
    cache_write_path: Union[str, None] = None,
) -> tuple[np.ndarray, int]:
    """
    Build GRM in Rust-only mode from PLINK BED input.

    Route policy:
    - non-fast/non-farmcpu: Rust streaming BED single-entry (`grm_stream_bed_f32`)
    - fast/farmcpu: packed BED single-entry (`grm_packed_bed_f32` / preloaded packed reuse)
    """
    packed_prefix = _as_plink_prefix(genofile)
    if packed_prefix is None:
        raise RuntimeError(
            "Rust-only GWAS GRM build requires PLINK BED input/prefix."
        )
    use_packed_kernel = bool(allow_packed_full_load)
    if use_packed_kernel:
        if not hasattr(jxrs, "grm_packed_bed_f32"):
            raise RuntimeError(
                "Rust symbol grm_packed_bed_f32 is unavailable. "
                "Rebuild janusx extension for Rust-only GWAS mode."
            )
    else:
        if not hasattr(jxrs, "grm_stream_bed_f32"):
            raise RuntimeError(
                "Rust symbol grm_stream_bed_f32 is unavailable. "
                "Rebuild janusx extension for Rust-only GWAS mode."
            )

    _log_info(
        logger,
        (
            f"Building GRM (Rust {'packed-bed' if use_packed_kernel else 'stream-bed'} single-entry), "
            f"method={method}"
        ),
        use_spinner=use_spinner,
    )
    pbar = _ProgressAdapter(
        total=n_snps,
        desc=("GRM (rust-bed)" if use_packed_kernel else "GRM (rust-stream)"),
        force_animate=True,
    )
    process = psutil.Process()
    mem_tick_span = max(1, 10 * int(chunk_size))
    last_done = 0
    next_mem_tick = mem_tick_span
    last_mem_ts = 0.0

    def _on_packed_progress(done: int, _total: int) -> None:
        nonlocal last_done, next_mem_tick, last_mem_ts
        d = int(done)
        if d > last_done:
            pbar.update(d - last_done)
            last_done = d
        now = time.monotonic()
        if d >= next_mem_tick or (d >= int(n_snps) and (now - last_mem_ts) >= 0.2):
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f}GB")
            last_mem_ts = now
            while next_mem_tick <= d:
                next_mem_tick += mem_tick_span

    try:
        grm_raw = None
        grm: Union[np.ndarray, None] = None
        eff_m = 0
        cache_target = str(cache_write_path).strip() if cache_write_path is not None else ""
        if use_packed_kernel:
            pre = preloaded_packed if isinstance(preloaded_packed, dict) else None
            used_preloaded = False
            if (
                pre is not None
                and str(pre.get("prefix", "")) == str(packed_prefix)
                and hasattr(jxrs, "grm_packed_f32")
            ):
                packed_ctx_obj = pre.get("packed_ctx")
                if isinstance(packed_ctx_obj, dict):
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
                    used_preloaded = True

            if not used_preloaded:
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
                    "GRM stream note: JX_GRM_STREAM_TWO_STAGE=1 is enabled; "
                    "stream entry may internally switch to packed prebuild mode.",
                )
            if cache_target != "" and hasattr(jxrs, "grm_stream_bed_f32_to_npy"):
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
                        f"Stream sample count mismatch: stream={stream_n}, expected={int(n_samples)}"
                    )
                eff_m = int(eff_m_raw)
                grm = np.load(str(cache_target), mmap_mode="r")
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
                        f"Stream sample count mismatch: stream={stream_n}, expected={int(n_samples)}"
                    )
                eff_m = int(eff_m_raw)

        if grm is None:
            grm = np.ascontiguousarray(np.asarray(grm_raw, dtype=np.float32))
        mem = process.memory_info().rss / 1024**3
        pbar.set_postfix(memory=f"{mem:.2f}GB")
        pbar.finish()
    finally:
        pbar.close(show_done=False)

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
    threads: int,
    mmap_limit: bool,
    logger:logging.Logger,
    use_spinner: bool = False,
    ids_preloaded: Union[np.ndarray, None] = None,
    n_snps_preloaded: Union[int, None] = None,
    snps_only: bool = True,
    allow_packed_full_load: bool = False,
    preloaded_packed: Union[dict[str, object], None] = None,
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
                tmp_grm = f"{grm_path}.tmp.{os.getpid()}.npy"
                stream_direct_cache = bool(
                    (not bool(allow_packed_full_load))
                    and hasattr(jxrs, "grm_stream_bed_f32_to_npy")
                )
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
    threads: int = 1,
    use_spinner: bool = False,
) -> np.ndarray:
    """
    Compute leading principal components from GRM.
    """
    if use_spinner:
        with CliStatus(f"Computing top {dim} PCs from GRM...", enabled=True) as task:
            try:
                _eigval, eigvec, _evd_backend, _evd_secs = _gwas_eigh_from_grm(
                    grm,
                    threads=int(threads),
                    logger=logger,
                    stage_label="GWAS-PCA",
                )
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
    _eigval, eigvec, _evd_backend, _evd_secs = _gwas_eigh_from_grm(
        grm,
        threads=int(threads),
        logger=logger,
        stage_label="GWAS-PCA",
    )
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
    threads: int,
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
                eigval, eigvec, _evd_backend, _evd_secs = _gwas_eigh_from_grm(
                    grm,
                    threads=int(threads),
                    logger=logger,
                    stage_label="GWAS-Q-build",
                )
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
    allow_packed_grm: bool = False,
    preload_packed_context: bool = False,
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
    preloaded_packed: Union[dict[str, object], None] = None
    genofile_low = str(genofile).lower()
    cache_candidates: list[str] = []
    if genofile_low.endswith(".vcf") or genofile_low.endswith(".vcf.gz"):
        # VCF caches are parameterized by maf/missing/snps_only in gfreader.
        cache_candidates.append(str(cache_prefix))
    elif genofile_low.endswith(".hmp") or genofile_low.endswith(".hmp.gz"):
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

    # In fast/farmcpu mode, preload packed BED once right after genotype meta
    # is known, then reuse for GRM and downstream GWAS/FarmCPU stages.
    if bool(allow_packed_grm) and bool(preload_packed_context):
        try:
            prefix0, full_ids0, packed_ctx0, sites0 = _prepare_packed_bed_once_for_gwas(
                genofile=stream_genofile,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                use_spinner=bool(use_spinner),
                preloaded_packed=None,
            )
            preloaded_packed = {
                "prefix": str(prefix0),
                "full_ids": np.asarray(full_ids0, dtype=str),
                "packed_ctx": packed_ctx0,
                "sites_all": sites0,
            }
            # When packed preload is ready, switch streaming source to the packed prefix.
            stream_genofile = str(prefix0)
        except Exception as ex:
            _emit_warning_line(
                logger,
                f"Fast mode packed preload unavailable; fallback to on-demand packed load. reason={ex}",
                use_spinner=bool(use_spinner),
            )
            preloaded_packed = None

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
                allow_packed_full_load=bool(allow_packed_grm),
                preloaded_packed=preloaded_packed,
            )
        else:
            # Keep terminal output concise for non-kinship runs (LM/FarmCPU-only).
            pass

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
            threads=int(threads),
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

    if preloaded_packed is None and bool(allow_packed_grm) and bool(preload_packed_context):
        try:
            prefix1, full_ids1, packed_ctx1, sites1 = _prepare_packed_bed_once_for_gwas(
                genofile=stream_genofile,
                maf_threshold=float(maf_threshold),
                max_missing_rate=float(max_missing_rate),
                het_threshold=float(het_threshold),
                snps_only=bool(snps_only),
                use_spinner=bool(use_spinner),
                preloaded_packed=None,
            )
            preloaded_packed = {
                "prefix": str(prefix1),
                "full_ids": np.asarray(full_ids1, dtype=str),
                "packed_ctx": packed_ctx1,
                "sites_all": sites1,
            }
            stream_genofile = str(prefix1)
        except Exception:
            preloaded_packed = None

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
      user did not pass --chunksize, so per-chunk compute is large enough to
      amortize Python orchestration and file-write overhead.
    """
    _ = bool(use_spinner)
    base = max(1, int(chunk_size))
    n_snps = int(max(1, int(n_snps_hint)))
    if bool(user_specified):
        return base

    if isinstance(model_keys, str):
        models = {str(model_keys).lower()}
    elif model_keys is None:
        models = set()
    else:
        models = {str(x).lower() for x in model_keys}
    if "lm" not in models:
        return base

    n_samples = int(max(1, int(n_samples_hint or 0)))
    if n_samples <= 0:
        return base

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
    return max(base, auto_rows)


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

    In outer-cap mode, keep both BLAS and Rayon at full `-t`.
    """
    t = max(1, int(threads))
    with runtime_thread_stage(blas_threads=t, rayon_threads=t):
        yield


@contextmanager
def _gwas_scan_stage_ctx(threads: int):
    """
    GWAS scan stage (Rust/Rayon kernels):
    - Rayon uses full `-t`
    - BLAS pinned to 1 to avoid oversubscription.
    """
    t = max(1, int(threads))
    with runtime_thread_stage(blas_threads=1, rayon_threads=t):
        yield


def _resolve_gwas_eigh_driver(n_samples: int) -> str:
    raw = str(os.environ.get("JX_GWAS_EIGH_DRIVER", "")).strip().lower()
    if raw in {"dsyevd", "dsyevr", "auto"}:
        return raw
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
) -> tuple[np.ndarray, np.ndarray, str, float]:
    """
    Eigen-decomposition for symmetric GRM with Rust-first backend.

    Returns
    -------
    eigvals, eigvecs, backend_label, elapsed_seconds
    """
    g0 = np.asarray(grm, dtype=np.float64)
    if g0.ndim != 2 or g0.shape[0] != g0.shape[1] or g0.shape[0] == 0:
        raise RuntimeError(f"{stage_label} expects non-empty square GRM, got shape={g0.shape}")
    g = np.ascontiguousarray(g0)
    t = max(1, int(threads))

    if not hasattr(jxrs, "rust_eigh_from_array_f64"):
        raise RuntimeError(
            f"{stage_label} requires rust_eigh_from_array_f64, but the symbol is unavailable."
        )
    impl_req = _resolve_gwas_eigh_impl(require_rust=bool(require_rust))
    driver_req = _resolve_gwas_eigh_driver(int(g.shape[0]))

    def _run_eigh_rust(driver_name: str, mat: np.ndarray):
        with _gwas_evd_stage_ctx(t):
            if hasattr(jxrs, "rust_eigh_from_array_f64_inplace") and bool(mat.flags.c_contiguous) and bool(mat.flags.writeable):
                return jxrs.rust_eigh_from_array_f64_inplace(
                    mat,
                    threads=int(t),
                    driver=str(driver_name),
                    jobz="V",
                    require_lapack=False,
                )
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

    def _run_eigh(driver_name: str, mat: np.ndarray):
        if impl_req == "scipy":
            return _run_eigh_scipy(driver_name, mat)
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
        ) = _run_eigh_rust(driver_name, mat)
        return (
            np.asarray(eval_raw, dtype=np.float64),
            np.asarray(evec_raw, dtype=np.float64) if evec_raw is not None else None,
            str(evd_backend),
            float(elapsed),
        )

    try:
        eval_raw, evec_raw, evd_backend, elapsed = _run_eigh(driver_req, g)
    except Exception as ex:
        if str(driver_req).lower() == "dsyevr":
            try:
                g_retry = np.ascontiguousarray(np.asarray(g0, dtype=np.float64))
                eval_raw, evec_raw, evd_backend, elapsed = _run_eigh("dsyevd", g_retry)
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
    if logger is not None:
        _log_file_only(
            logger,
            logging.INFO,
            f"{stage_label} eigh impl={impl_req} backend={backend} driver={driver_req} "
            f"elapsed={float(elapsed):.3f}s",
        )
    return eigvals, eigvecs, backend, float(elapsed)


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
        packed_t0 = time.monotonic()
        with CliStatus("Loading packed BED genotype...", enabled=bool(use_spinner)) as task:
            try:
                full_ids, packed_ctx_raw = prepare_packed_ctx_from_plink(
                    str(prefix),
                    maf=float(maf_threshold),
                    missing_rate=float(max_missing_rate),
                    snps_only=False,
                )
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

        packed = np.ascontiguousarray(np.asarray(packed_ctx_raw["packed"], dtype=np.uint8))
        miss = np.ascontiguousarray(
            np.asarray(packed_ctx_raw["missing_rate"], dtype=np.float32).reshape(-1)
        )
        maf = np.ascontiguousarray(np.asarray(packed_ctx_raw["maf"], dtype=np.float32).reshape(-1))
        packed_n = int(packed_ctx_raw["n_samples"])
        if packed_n <= 0:
            raise ValueError("Packed BED reported invalid sample count.")
        if packed.shape[0] != maf.shape[0] or packed.shape[0] != miss.shape[0]:
            raise ValueError("Packed BED arrays have inconsistent SNP dimensions.")

        sites_all_full = _read_bim_sites(prefix)
        keep_numeric = np.ascontiguousarray(
            np.asarray(packed_ctx_raw["site_keep"], dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if len(sites_all_full) != int(keep_numeric.shape[0]):
            raise ValueError(
                "BIM row count mismatch with packed BED filtering mask: "
                f"bim={len(sites_all_full)}, mask={keep_numeric.shape[0]}"
            )
        keep_final = np.ascontiguousarray(keep_numeric, dtype=np.bool_)
        if bool(snps_only):
            snp_mask = np.asarray(
                [
                    (len(str(a0)) == 1 and len(str(a1)) == 1)
                    for (_c, _p, a0, a1) in sites_all_full
                ],
                dtype=bool,
            )
            keep_final &= snp_mask
        if not np.any(keep_final):
            raise ValueError("No SNP remains after LRLMM packed-bed filtering.")
        if not np.array_equal(keep_final, keep_numeric):
            kept_numeric_idx = np.flatnonzero(keep_numeric).astype(np.int64, copy=False)
            keep_local = np.ascontiguousarray(keep_final[kept_numeric_idx], dtype=np.bool_)
            packed = np.ascontiguousarray(packed[keep_local], dtype=np.uint8)
            miss = np.ascontiguousarray(miss[keep_local], dtype=np.float32)
            maf = np.ascontiguousarray(maf[keep_local], dtype=np.float32)
        sites_all = [s for s, k in zip(sites_all_full, keep_final) if bool(k)]

    process = psutil.Process()
    n_cores = detect_effective_threads()
    if eff_snp_by_trait is None:
        eff_snp_by_trait = {}
    if summary_rows is None:
        summary_rows = []
    if saved_paths is None:
        saved_paths = []

    pheno_aligned, ids = _align_pheno_to_sample_order(pheno, ids)
    trait_iter = list(pheno_aligned.columns) if trait_names is None else [t for t in trait_names if t in pheno_aligned.columns]
    multi_trait_mode = len(trait_iter) > 1

    for trait_idx, pname in enumerate(trait_iter):
        cpu_t0 = process.cpu_times()
        t0 = time.time()
        peak_rss = process.memory_info().rss

        y_full, sameidx = _trait_values_and_mask(pheno_aligned, str(pname))
        keep_idx = np.flatnonzero(sameidx).astype(np.int64, copy=False)
        n_idv = int(keep_idx.shape[0])
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

        trait_ids = np.asarray(ids[keep_idx], dtype=str)
        y_vec = np.ascontiguousarray(y_full[keep_idx], dtype=np.float64)
        x_cov = qmatrix[keep_idx]
        if cov_all is not None:
            x_cov = np.concatenate([x_cov, cov_all[keep_idx]], axis=1)
        x_cov = np.ascontiguousarray(x_cov, dtype=np.float64)
        x_arg = x_cov if int(x_cov.shape[1]) > 0 else None
        sample_idx_trait = np.ascontiguousarray(sample_map[keep_idx], dtype=np.int64)
        fixed_lbd: Optional[float] = None
        fixed_ml0: Optional[float] = None

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
            viz_secs = _run_fastplot_with_status(
                res_df[["chrom", "pos", "pwald"]],
                y_vec,
                xlabel=str(pname),
                outpdf=out_svg,
                use_spinner=bool(use_spinner),
            )

        cast_map: dict[str, object] = {"pwald": "object", "pos": int}
        if "plrt" in res_df.columns:
            cast_map["plrt"] = "object"
        def _write_lrlmm() -> None:
            nonlocal res_df
            res_df = res_df.astype(cast_map)
            res_df.loc[:, "pwald"] = res_df["pwald"].map(lambda x: f"{x:.4e}")
            if "plrt" in res_df.columns:
                res_df.loc[:, "plrt"] = res_df["plrt"].map(lambda x: f"{x:.4e}")
            res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        _run_result_write_with_status(
            _write_lrlmm,
            use_spinner=bool(use_spinner),
            emit_done_line=False,
        )
        saved_paths.append(str(out_tsv))

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
            f"Results saved to {_display_path(str(out_tsv))}",
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
                "result_file": str(out_tsv),
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
        if multi_trait_mode:
            logger.info("")


def _prepare_packed_bed_once_for_gwas(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import _prepare_packed_bed_once_for_gwas as _impl
    return _impl(*args, **kwargs)

def run_fastlmm_packed_fullrank(*args, **kwargs):
    from janusx.assoc.workflow_model_packed import run_fastlmm_packed_fullrank as _impl
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
        "-farmcpu", "--farmcpu", action="store_true", default=False,
        help="Run FarmCPU (full genotype in memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-lm", "--lm", action="store_true", default=False,
        help="Run the linear model (streaming, low-memory; default: %(default)s).",
    )
    models_group.add_argument(
        "-model", "--model", type=str, choices=["add", "dom", "rec", "het"], default="add",
        help="Genetic effect coding model for streaming LM/LMM/FastLMM (default: %(default)s).",
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
        "-snps-only", "--snps-only", "-only-snps", "--only-snps", action="store_true", default=False,
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
        "--farmcpu-iter", type=int, default=20,
        help=(
            "FarmCPU max iterations (default: %(default)s)."
            if show_dev_help else argparse.SUPPRESS
        ),
    )
    optional_group.add_argument(
        "--farmcpu-threshold", type=float, default=0.05,
        help=(
            "FarmCPU global threshold (effective per-marker threshold is threshold/m; default: %(default)s)."
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
        "-fast", "--fast", action="store_true", default=False,
        help=(
            "Use packed full-rust paths when available (loads BED-packed genotype once "
            "and reuses it across LM/LMM/FastLMM/FarmCPU)."
        ),
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
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))
    try:
        args.qcov = str(_parse_qcov_dim(args.qcov))
    except ValueError as e:
        parser.error(str(e))
    if int(args.farmcpu_iter) < 1:
        parser.error("--farmcpu-iter must be >= 1.")
    if not np.isfinite(float(args.farmcpu_threshold)) or float(args.farmcpu_threshold) <= 0:
        parser.error("--farmcpu-threshold must be a finite value > 0.")
    if int(args.farmcpu_nbin) < 1:
        parser.error("--farmcpu-nbin must be >= 1.")
    if args.farmcpu_qtn_bound is not None and int(args.farmcpu_qtn_bound) < 1:
        parser.error("--farmcpu-qtn-bound must be >= 1.")
    try:
        args.farmcpu_bin_size = _parse_float_csv(
            args.farmcpu_bin_size,
            label="--farmcpu-bin-size",
        )
    except ValueError as e:
        parser.error(str(e))
    args._chunksize_user_set = bool(_option_present(argv, "-chunksize", "--chunksize"))
    return args


def _run_gwas_pipeline(
    args=None,
    *,
    argv: Optional[list[str]] = None,
    log: bool = True,
    return_result: bool = False,
):
    t_start = time.time()
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_status = "done"
    run_error = ""
    if args is None:
        args = parse_args(argv)
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
    outprefix = os.path.join(args.out, prefix)
    log_path = f"{outprefix}.gwas.log"
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    apply_outer_thread_cap(int(args.thread))
    maybe_warn_non_openblas(
        logger=logger,
        strict=require_openblas_by_default(),
    )
    gwas_summary_rows: list[dict[str, object]] = []
    saved_result_paths: list[str] = []
    ordered_result_paths: list[str] = []
    trait_order: list[str] = []

    if log:
        model_tokens: list[str] = []
        if args.lm:
            model_tokens.append("LM")
        if args.fastlmm:
            model_tokens.append("fastLMM")
        if args.lmm:
            model_tokens.append("LMM")
        if args.farmcpu:
            model_tokens.append("FarmCPU")
        cfg_rows: list[tuple[str, object]] = [
            ("Genotype file", gfile),
            ("Phenotype file", args.pheno),
            ("Phenotype cols", args.ncol if args.ncol is not None else "All"),
            ("Mmap limit", args.mmap_limit),
            ("Fast mode", bool(getattr(args, "fast", False) or bool(args.farmcpu))),
            ("Models", " ".join(model_tokens) if len(model_tokens) > 0 else "None"),
            ("Genetic model", args.model),
            ("SNPs only", args.snps_only),
            ("GRM option", args.grm),
            ("Q option", args.qcov),
            ("MAF threshold", args.maf),
            ("Miss threshold", args.geno),
            ("Chunk size", args.chunksize),
        ]
        if args.model != "add":
            cfg_rows.append(("Het filter", f"{args.het} (keep [{args.het}, {1.0 - args.het}])"))
        if args.farmcpu:
            cfg_rows.extend(
                [
                    ("FarmCPU iter", int(args.farmcpu_iter)),
                    ("FarmCPU threshold", float(args.farmcpu_threshold)),
                    ("FarmCPU nbin", int(args.farmcpu_nbin)),
                    ("FarmCPU QTNbound", "auto" if args.farmcpu_qtn_bound is None else int(args.farmcpu_qtn_bound)),
                    ("FarmCPU szbin", ",".join(f"{float(x):g}" for x in args.farmcpu_bin_size)),
                ]
            )
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

    if not (args.lm or args.lmm or args.fastlmm or args.farmcpu):
        logger.error(
            "No model selected. Use -lm, -lmm, -fastlmm, and/or -farmcpu."
        )
        raise SystemExit(1)
    if args.farmcpu and args.model != "add":
        logger.warning(
            "Warning: --model/--het currently apply to streaming LM/LMM/FastLMM; "
            "FarmCPU keeps additive coding."
        )
    fast_mode = bool(getattr(args, "fast", False) or bool(args.farmcpu))
    qcov_needs_grm = str(getattr(args, "qcov", "0")).strip() not in {"", "0"}
    # GRM route policy:
    # - non-fast GWAS: prefer streaming GRM path (Rust reader + streaming accumulation)
    # - fast/farmcpu GWAS: allow packed single-entry kernel when enabled
    prefer_packed_grm = bool(
        fast_mode
        and (args.lmm or args.fastlmm or qcov_needs_grm)
        and _gwas_use_packed_grm_build()
    )
    if (not bool(fast_mode)) and bool(args.lmm or args.fastlmm or qcov_needs_grm):
        _log_file_only(
            logger,
            logging.INFO,
            "Non-fast GWAS: GRM uses streaming path by default (packed single-entry disabled).",
        )
    if bool(args.farmcpu) and (not bool(getattr(args, "fast", False))):
        _log_file_only(
            logger,
            logging.INFO,
            "FarmCPU selected: enabling fast mode automatically.",
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
        shared_context_needed = bool(stream_selected or fast_mode)
        preloaded_packed: Union[dict[str, object], None] = None
        if shared_context_needed:
            _section(logger, "Streaming task")
            _log_file_only(
                logger,
                logging.INFO,
                "Prepare shared context (phenotype/genotype meta/GRM/Q/cov)",
            )
            pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m, genofile_stream, preloaded_packed = prepare_streaming_context(
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
                allow_packed_grm=bool(prefer_packed_grm),
                preload_packed_context=bool(fast_mode),
            )
            if bool(fast_mode) and (not isinstance(preloaded_packed, dict)):
                try:
                    prefix0, full_ids0, packed_ctx0, sites0 = _prepare_packed_bed_once_for_gwas(
                        genofile=genofile_stream,
                        maf_threshold=float(args.maf),
                        max_missing_rate=float(args.geno),
                        het_threshold=float(args.het),
                        snps_only=bool(args.snps_only),
                        use_spinner=bool(use_spinner),
                        preloaded_packed=None,
                    )
                    preloaded_packed = {
                        "prefix": str(prefix0),
                        "full_ids": np.asarray(full_ids0, dtype=str),
                        "packed_ctx": packed_ctx0,
                        "sites_all": sites0,
                    }
                except Exception as ex:
                    logger.warning(
                        f"Fast mode packed preload unavailable; falling back to on-demand packed load. reason={ex}"
                    )
                    preloaded_packed = None
            _phase_split(logger)
        else:
            if args.farmcpu:
                pheno = _load_phenotype_with_status(
                    args.pheno,
                    args.ncol,
                    logger,
                    id_col=0,
                    use_spinner=use_spinner,
                )

        stream_models: list[str] = []
        if args.lm:
            stream_models.append("lm")
        if args.lmm:
            stream_models.append("lmm")
        if args.fastlmm:
            stream_models.append("fastlmm")
        has_farmcpu = bool(args.farmcpu)
        farmcpu_handled_in_trait_loop = False

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
            if bool(fast_mode):
                # v3: fast mode always executes via trait/model task plan (Rust route planner
                # + thin Python executor) so single-model paths no longer bypass the unified route.
                use_trait_grouped_fast = True
                if use_trait_grouped_fast:
                    farmcpu_cache_prefill: Union[dict[str, object], None] = None
                    farmcpu_cache_runtime: Union[dict[str, object], None] = None
                    if has_farmcpu:
                        context_prepared = bool(pheno is not None and ids is not None and n_snps is not None)
                        if (
                            isinstance(preloaded_packed, dict)
                            and pheno is not None
                            and ids is not None
                            and qmatrix is not None
                        ):
                            try:
                                sites_prefill = preloaded_packed.get("sites_all")
                                if isinstance(sites_prefill, list):
                                    chrom_prefill: list[str] = []
                                    pos_prefill: list[int] = []
                                    allele0_prefill: list[str] = []
                                    allele1_prefill: list[str] = []
                                    for (c, p, a0, a1) in sites_prefill:
                                        chrom_prefill.append(str(c))
                                        try:
                                            pos_prefill.append(int(p))
                                        except Exception:
                                            try:
                                                pos_prefill.append(int(float(p)))
                                            except Exception:
                                                pos_prefill.append(0)
                                        allele0_prefill.append(str(a0))
                                        allele1_prefill.append(str(a1))
                                    ref_alt_prefill = {
                                        "chrom": chrom_prefill,
                                        "pos": pos_prefill,
                                        "allele0": allele0_prefill,
                                        "allele1": allele1_prefill,
                                    }
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
                                        "packed_ctx": preloaded_packed.get("packed_ctx"),
                                        "ref_alt": ref_alt_prefill,
                                        "qmatrix": q_prefill,
                                        "packed_sample_idx": packed_idx_map,
                                    }
                            except Exception:
                                farmcpu_cache_prefill = None
                        farmcpu_cache_runtime = run_farmcpu_fullmem(
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
                            farmcpu_cache=farmcpu_cache_prefill,
                            prepare_only=True,
                            emit_trait_header=False,
                            preloaded_packed=preloaded_packed,
                        )
                    use_packed_fastlmm_scan = bool(_gwas_use_packed_fastlmm_scan())
                    task_plan = None
                    if hasattr(jxrs, "gwas_trait_model_dispatch_v2"):
                        try:
                            task_plan = jxrs.gwas_trait_model_dispatch_v2(
                                [str(m) for m in stream_models],
                                [str(t) for t in trait_order],
                                bool(has_farmcpu),
                                bool(use_packed_fastlmm_scan),
                            )
                        except Exception as ex:
                            _emit_warning_line(
                                logger,
                                f"Rust v2 task dispatcher unavailable; fallback to v1/Python planner. reason={ex}",
                                use_spinner=bool(use_spinner),
                            )
                            task_plan = None
                    if task_plan is None and hasattr(jxrs, "gwas_trait_model_schedule"):
                        try:
                            task_plan = jxrs.gwas_trait_model_schedule(
                                [str(m) for m in stream_models],
                                [str(t) for t in trait_order],
                                bool(has_farmcpu),
                            )
                        except Exception as ex:
                            _emit_warning_line(
                                logger,
                                f"Rust task scheduler unavailable; fallback to Python planner. reason={ex}",
                                use_spinner=bool(use_spinner),
                            )
                            task_plan = None
                    if task_plan is None:
                        task_plan = []
                        trait_total = int(len(trait_order))
                        for trait_idx, trait_name in enumerate(trait_order):
                            ordered_models = [str(m).lower().strip() for m in stream_models]
                            if bool(has_farmcpu):
                                ordered_models.append("farmcpu")
                            model_last = int(len(ordered_models)) - 1
                            for idx_model, mk in enumerate(ordered_models):
                                route = "unknown"
                                if mk == "lm":
                                    route = "lm_packed"
                                elif mk == "lmm":
                                    route = "lmm_packed"
                                elif mk == "fastlmm":
                                    route = (
                                        "fastlmm_packed"
                                        if bool(use_packed_fastlmm_scan)
                                        else "fastlmm_stream"
                                    )
                                elif mk == "farmcpu":
                                    route = "farmcpu"
                                task_plan.append(
                                    {
                                        "model": str(mk),
                                        "route": str(route),
                                        "trait": str(trait_name),
                                        "emit_trait_header": bool(idx_model == 0),
                                        "emit_blank_after": bool(
                                            (idx_model == model_last)
                                            and ((trait_idx + 1) < trait_total)
                                        ),
                                    }
                                )
                    def _run_route_lm_packed(
                        trait_one: list[str], emit_trait_header_model: bool
                    ) -> None:
                        run_lm_packed_fullrank(
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            outprefix=outprefix,
                            maf_threshold=args.maf,
                            max_missing_rate=args.geno,
                            genetic_model=args.model,
                            het_threshold=args.het,
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
                        )

                    def _run_route_lmm_packed(
                        trait_one: list[str], emit_trait_header_model: bool
                    ) -> None:
                        run_lmm_packed_fullrank(
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            grm=grm,
                            outprefix=outprefix,
                            maf_threshold=args.maf,
                            max_missing_rate=args.geno,
                            genetic_model=args.model,
                            het_threshold=args.het,
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
                        )

                    def _run_route_fastlmm_packed(
                        trait_one: list[str], emit_trait_header_model: bool
                    ) -> None:
                        run_fastlmm_packed_fullrank(
                            genofile=genofile_stream,
                            pheno=pheno,
                            ids=ids,
                            grm=grm,
                            outprefix=outprefix,
                            maf_threshold=args.maf,
                            max_missing_rate=args.geno,
                            genetic_model=args.model,
                            het_threshold=args.het,
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
                        )

                    def _run_route_fastlmm_stream(
                        trait_one: list[str], emit_trait_header_model: bool
                    ) -> None:
                        _log_file_only(
                            logger,
                            logging.INFO,
                            "Fast mode: FastLMM scan uses streaming prepared-chunk route "
                            "(set JX_GWAS_USE_PACKED_FASTLMM_SCAN=1 or unset it to use packed scan).",
                        )
                        run_chunked_gwas_lmm_lm(
                            model_name="fastlmm",
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
                            trait_names=trait_one,
                            show_npve_line=True,
                            emit_trait_header=bool(emit_trait_header_model),
                            chunk_size_user_set=bool(getattr(args, "_chunksize_user_set", False)),
                            prefer_packed_fullrust=False,
                        )

                    def _run_route_farmcpu(
                        trait_one: list[str], emit_trait_header_model: bool
                    ) -> None:
                        nonlocal farmcpu_cache_runtime, farmcpu_handled_in_trait_loop
                        farmcpu_cache_runtime = run_farmcpu_fullmem(
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
                            context_prepared=True,
                            summary_rows=gwas_summary_rows,
                            saved_paths=saved_result_paths,
                            trait_names=trait_one,
                            farmcpu_cache=farmcpu_cache_runtime,
                            emit_trait_header=False,
                            preloaded_packed=preloaded_packed,
                        )
                        farmcpu_handled_in_trait_loop = True

                    route_handlers = {
                        "lm_packed": _run_route_lm_packed,
                        "lmm_packed": _run_route_lmm_packed,
                        "fastlmm_packed": _run_route_fastlmm_packed,
                        "fastlmm_stream": _run_route_fastlmm_stream,
                        "farmcpu": _run_route_farmcpu,
                    }

                    for task_item in task_plan:
                        mk = str(task_item.get("model", "")).lower().strip()
                        route = str(task_item.get("route", "")).lower().strip()
                        if not route:
                            if mk == "lm":
                                route = "lm_packed"
                            elif mk == "lmm":
                                route = "lmm_packed"
                            elif mk == "fastlmm":
                                route = (
                                    "fastlmm_packed"
                                    if bool(use_packed_fastlmm_scan)
                                    else "fastlmm_stream"
                                )
                            elif mk == "farmcpu":
                                route = "farmcpu"
                            else:
                                route = "unknown"
                        trait_name_use = str(task_item.get("trait", ""))
                        emit_trait_header_model = bool(
                            task_item.get("emit_trait_header", False)
                        )
                        emit_blank_after = bool(task_item.get("emit_blank_after", False))
                        trait_one = [trait_name_use]
                        route_handler = route_handlers.get(route)
                        if route_handler is not None:
                            route_handler(trait_one, bool(emit_trait_header_model))
                        else:
                            logger.warning(
                                f"Unknown task route in fast mode: route={route}, model={mk}"
                            )
                        if emit_blank_after:
                            logger.info("")
            else:
                # If some models can use packed full-rust single-entry routes, run
                # them separately first to avoid being merged into shared streaming.
                if len(stream_models) > 1:
                    packed_fullrust_enabled = False
                    packed_prefix_ok = _as_plink_prefix(genofile_stream) is not None

                    def _can_fullrust_model(mkey: str) -> bool:
                        mk = str(mkey).lower().strip()
                        if mk == "lmm":
                            if not packed_fullrust_enabled:
                                return False
                            if not packed_prefix_ok:
                                return False
                            return (
                                str(args.model).lower() == "add"
                                and (grm is not None)
                                and hasattr(jxrs, "lmm_reml_assoc_packed_f32")
                            )
                        if mk == "lm":
                            return False
                        return False

                    for model_sep in ("lmm", "lm"):
                        if (model_sep not in stream_models) or (not _can_fullrust_model(model_sep)):
                            continue
                        run_chunked_gwas_lmm_lm(
                            model_name=model_sep,
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
                            trait_names=[str(t) for t in trait_order] if len(trait_order) > 0 else None,
                            show_npve_line=True,
                            emit_trait_header=True,
                            chunk_size_user_set=bool(getattr(args, "_chunksize_user_set", False)),
                        )
                        stream_models = [m for m in stream_models if str(m).lower() != model_sep]
                        if len(stream_models) > 0:
                            logger.info("")

                if len(stream_models) == 1:
                    model_key = stream_models[0]
                    pve_line_model = model_key if model_key in {"lmm", "fastlmm"} else None
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
                        trait_names=[str(t) for t in trait_order] if len(trait_order) > 0 else None,
                        show_npve_line=True if pve_line_model is None else (model_key == pve_line_model),
                        emit_trait_header=True,
                        chunk_size_user_set=bool(getattr(args, "_chunksize_user_set", False)),
                    )
                elif len(stream_models) > 1:
                    for trait_idx, pname in enumerate(trait_order):
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
                            chunk_size_user_set=bool(getattr(args, "_chunksize_user_set", False)),
                        )
                        if trait_idx < len(trait_order) - 1:
                            logger.info("")

        # FarmCPU (same model flow; no separate task section banner)
        if has_farmcpu and (not farmcpu_handled_in_trait_loop):
            context_prepared = bool(pheno is not None and ids is not None and n_snps is not None)
            emit_farmcpu_trait_header = len(stream_models) == 0
            trait_names_full = [str(t) for t in trait_order] if len(trait_order) > 0 else None
            farmcpu_cache_prefill: Union[dict[str, object], None] = None
            if (
                bool(fast_mode)
                and isinstance(preloaded_packed, dict)
                and pheno is not None
                and ids is not None
                and qmatrix is not None
            ):
                try:
                    sites_prefill = preloaded_packed.get("sites_all")
                    if isinstance(sites_prefill, list):
                        chrom_prefill: list[str] = []
                        pos_prefill: list[int] = []
                        allele0_prefill: list[str] = []
                        allele1_prefill: list[str] = []
                        for (c, p, a0, a1) in sites_prefill:
                            chrom_prefill.append(str(c))
                            try:
                                pos_prefill.append(int(p))
                            except Exception:
                                try:
                                    pos_prefill.append(int(float(p)))
                                except Exception:
                                    pos_prefill.append(0)
                            allele0_prefill.append(str(a0))
                            allele1_prefill.append(str(a1))
                        ref_alt_prefill = {
                            "chrom": chrom_prefill,
                            "pos": pos_prefill,
                            "allele0": allele0_prefill,
                            "allele1": allele1_prefill,
                        }
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
                            "packed_ctx": preloaded_packed.get("packed_ctx"),
                            "ref_alt": ref_alt_prefill,
                            "qmatrix": q_prefill,
                            "packed_sample_idx": packed_idx_map,
                        }
                except Exception:
                    farmcpu_cache_prefill = None
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
                trait_names=trait_names_full,
                farmcpu_cache=farmcpu_cache_prefill,
                emit_trait_header=bool(emit_farmcpu_trait_header),
                preloaded_packed=preloaded_packed,
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
            saved_body = "\n".join([f"  {_display_path(p)}" for p in ordered_result_paths])
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
                    "lm": bool(args.lm),
                    "farmcpu": bool(args.farmcpu),
                },
                "model": str(args.model),
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
