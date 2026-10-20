# -*- coding: utf-8 -*-
"""
JanusX: Post-GWAS Visualization and Annotation

Examples
--------
  # Basic usage with default column names (#CHROM, POS, p)
  -gwasfile result.assoc.txt

  # Specify alternative column names
  -gwasfile result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald"

  # Specify output path and format
  -gwasfile result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald" \
    --out test --fmt pdf
  # Results will be saved as:
  #   test/result.assoc.manh.pdf
  #   test/result.assoc.qq.pdf

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import logging
import os
import gzip
import tempfile
from dataclasses import dataclass
from pathlib import Path
from ._common.cli_args import (
    add_common_memory_arg,
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from ._common.log import setup_logging
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    ensure_file_input_site_metadata_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.progress import (
    CliStatus,
    log_success,
    print_success,
    print_failure,
    print_warning,
    format_elapsed,
    should_animate_status,
    stdout_is_tty,
    warn_deprecated_alias_usage,
)
from ._common.progress import build_rich_progress, rich_progress_available
from ._common.genocache import configure_genotype_cache_from_out
from ._common.outprefix import apply_output_prefix_compat
from ._common.config_render import emit_cli_configuration
from ._common.threads import (
    apply_blas_thread_env,
    detect_effective_threads,
    format_requested_thread_usage,
    maybe_warn_non_openblas,
    require_openblas_by_default,
    runtime_thread_stage,
)

# Ensure matplotlib uses a non-interactive backend.
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl
mpl.use("Agg")
from janusx.bioplotkit import (
    GWASPLOT,
    LDblock,
    apply_integer_xticks,
    apply_integer_yticks,
    resolve_manhattan_chr_gap,
)
from janusx.bioplotkit.geneplot import draw_gene_structure_records
from janusx.gfreader import load_genotype_chunks, prepare_cli_input_cache

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.markers import MarkerStyle
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd
import numpy as np
from scipy.stats import beta
import argparse
import difflib
import inspect
import heapq
import re
import shlex
import time
import socket
import sys
import zipfile
import colorsys
import concurrent.futures as cf
import multiprocessing as mp
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from functools import lru_cache
from typing import Any, Iterable, Optional, Sequence, Tuple
from urllib.parse import unquote
from janusx import janusx as jxrs
from janusx.assoc.null_model_sidecar import (
    FineMapSkip,
    GwasNullModelSidecarV1,
    SidecarFormatError,
    _dependency_search_roots,
    _resolve_dependency_path,
    discover_matching_sidecar,
    hash_ordered_sample_ids,
    serialize_sidecar_block,
    validate_sidecar_dependencies,
)
from janusx.gtools.reader import GFFQuery, bedreader, readanno, _gff_prefetched_attr_colname
import warnings
from ._common.cjk import contains_cjk as _contains_cjk, ensure_cjk_font as _ensure_cjk_font
from matplotlib import font_manager as mpl_font_manager

_LEAD_SNP_INFO_COLS = ["allele0", "allele1", "af", "maf", "beta", "se"]
_INTERACTION_PLOT_INFO_COLS = ["snp", "padj", "row_role"]
_INTERACTION_SIG_PADJ_DEFAULT = 0.05
_QQ_FIXED_RATIO = 5.0 / 4.0
_QQ_FAST_MAX_POINTS = 120_000
_QQ_BAND_MAX_POINTS = 20_000
_QQ_BAND_COLOR = "grey"
_CONFIG_LINE_MAX_CHARS = 60
_CONFIG_OVERFLOW_MARK = "***"
_ANNO_DESC_KEY = "description"
_POSTGWAS_GFF_BATCH_ANNOTATION_MIN_SITES = 64
_DEFAULT_SINGLE_MARKER = "o"
_DEFAULT_MERGE_MARKERS = ("1", "2", "3", "4", "*", "+", "x")
_DEFAULT_SCATTER_SIZE = 8.0
_DEFAULT_MERGE_ALPHA = 0.45
_DEFAULT_CIRCLE_SIZE_IN = 8.5
_DEFAULT_CIRCLE_TRACK_RATIO = 0.5
_DEFAULT_CIRCLE_INTERVAL = 1.0
_DEFAULT_CIRCLE_LW = 1.0
_POSTGWAS_LEGEND_SIZE_SCALE = 1.35
_POSTGWAS_LEGEND_SIZE_MIN_BONUS = 2.0
_POSTGWAS_LD_LINK_Y = 0.0
_POSTGWAS_RASTERIZE_THRESHOLD = 50_000
_PANEL_WIDTH_IN = 8.0
_PANEL_LEFT_IN = 0.95
_PANEL_RIGHT_IN = 0.20
_PANEL_TOP_IN = 0.20
_PANEL_BOTTOM_IN = 0.65
_PANEL_LEGEND_RIGHT_IN = 2.25
_PANEL_STACK_VSPACE_IN = 0.10
_POSTGWAS_PDF_BACKEND_SENTINEL = object()
_POSTGWAS_PREFERRED_PDF_BACKEND: object = _POSTGWAS_PDF_BACKEND_SENTINEL
_POSTGWAS_DEFAULT_FONT_SIZE = 9.0
_POSTGWAS_MIN_FONT_SCALE = 0.80
_POSTGWAS_FONT_FILE_EXTENSIONS = frozenset({".ttf", ".otf", ".ttc", ".otc"})
_POSTGWAS_SHARED_GFF_KEY: Optional[tuple[str, int, int]] = None
_POSTGWAS_SHARED_GFF_QUERY: Optional[GFFQuery] = None
_POSTGWAS_SHARED_GFF_ANNOTATION_CTX: Optional[dict[str, object]] = None
_POSTGWAS_SHARED_GFF_RUST_KEY: Optional[tuple[str, int, int]] = None
_POSTGWAS_SHARED_GFF_RUST_INDEX: Optional[object] = None
_POSTGWAS_GENERIC_FONT_FAMILIES = {
    "serif": "serif",
    "sans": "sans-serif",
    "sansserif": "sans-serif",
    "monospace": "monospace",
    "mono": "monospace",
    "cursive": "cursive",
    "fantasy": "fantasy",
}
_ANNOTATION_APPEND_BASE_COLS = ("start", "end", "nsnps", "MeanR2", "desc", "broaden", "LDclump")

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False

def _postgwas_status_enabled(args) -> bool:
    return not bool(getattr(args, "_postgwas_worker_mute_stream", False))


def _postgwas_invocation_command(argv: Optional[list[str]] = None) -> str:
    tokens = [str(x) for x in (sys.argv[1:] if argv is None else argv)]
    prog_raw = str(sys.argv[0]).strip() if len(sys.argv) > 0 else ""
    if prog_raw == "":
        prog_raw = "jx postgwas"
    try:
        prog_parts = shlex.split(prog_raw)
    except Exception:
        prog_parts = [prog_raw]
    if len(prog_parts) == 0:
        prog_parts = ["jx", "postgwas"]
    return shlex.join([str(x) for x in (prog_parts + tokens)])


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _postgwas_annotation_suffix(path: Optional[str]) -> str:
    if path is None:
        return ""
    text = str(path).strip()
    if text == "":
        return ""
    return text.replace(".gz", "").split(".")[-1].lower()


def _resolve_postgwas_annotation_kind(
    *,
    gff: Optional[str] = None,
    bed: Optional[str] = None,
    anno_file: Optional[str] = None,
) -> str:
    gff_text = str(gff or "").strip()
    if gff_text != "":
        return "gff"
    bed_text = str(bed or "").strip()
    if bed_text != "":
        return "bed"
    suffix = _postgwas_annotation_suffix(anno_file)
    if suffix in {"gff", "gff3"}:
        return "gff"
    if suffix != "":
        return "bed"
    return ""


def _postgwas_annotation_is_gff(
    path: Optional[str],
    *,
    annotation_kind: Optional[str] = None,
) -> bool:
    kind = str(annotation_kind or "").strip().lower()
    if kind == "gff":
        return True
    if kind == "bed":
        return False
    return _postgwas_annotation_suffix(path) in {"gff", "gff3"}


def _postgwas_gff_cache_key(gff_path: str) -> tuple[str, int, int]:
    real = os.path.realpath(str(gff_path))
    st = os.stat(real)
    return (real, int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))


def _postgwas_has_rust_gff_index() -> bool:
    return hasattr(jxrs, "GffAnnotationIndex")


def _postgwas_rust_gff_cache_path(gff_path: str) -> str:
    real, size, mtime_ns = _postgwas_gff_cache_key(gff_path)
    cache_root = os.path.join(os.path.expanduser("~"), ".janusx", "cache", "gff")
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(real))
    return os.path.join(
        cache_root,
        f"{safe_base}.{int(size)}.{int(mtime_ns)}.jxgff.bin",
    )


@lru_cache(maxsize=None)
def _postgwas_attr_pair_regex() -> re.Pattern[str]:
    return re.compile(r"(?:^|;|\s)([^;=\s]+)=([^;]*?)(?=;|\s+[^\s;=]+=|$)")


def _postgwas_clean_attr_text(value: object) -> str:
    text = unquote(str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    if text == "" or text.lower() == "nan":
        return "NA"
    return text


def _postgwas_normalize_gff_ref_text(value: object, *, split_multi: bool) -> str:
    text = _postgwas_clean_attr_text(value)
    if text == "NA":
        return "NA"
    if not bool(split_multi):
        return re.sub(r"^[^:]*:", "", text)
    out: list[str] = []
    seen: set[str] = set()
    for token in str(text).split(","):
        norm = re.sub(r"^[^:]*:", "", str(token).strip())
        if norm == "" or norm.lower() == "nan" or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return ",".join(out) if len(out) > 0 else "NA"


def _postgwas_extract_attr_subset(attr_text: object, *, include_gene_meta: bool) -> dict[str, str]:
    need = {"ID", "Parent"}
    if bool(include_gene_meta):
        need.update({_ANNO_DESC_KEY, "Name"})
    out = {key: "NA" for key in need}
    text = str(attr_text).strip()
    if text == "":
        return out
    missing = set(need)

    # Fast path for standard semicolon-delimited GFF attributes.
    if ";" in text:
        for token in text.split(";"):
            if len(missing) == 0:
                break
            if "=" not in token:
                continue
            key, raw_value = token.split("=", 1)
            key = str(key).strip()
            if key not in missing:
                continue
            if key == "ID":
                out[key] = _postgwas_normalize_gff_ref_text(raw_value, split_multi=False)
            elif key == "Parent":
                out[key] = _postgwas_normalize_gff_ref_text(raw_value, split_multi=True)
            else:
                out[key] = _postgwas_clean_attr_text(raw_value)
            missing.discard(key)

    # Fallback for malformed space-delimited payloads or any missing keys.
    if len(missing) > 0:
        for match in _postgwas_attr_pair_regex().finditer(text):
            key = str(match.group(1)).strip()
            if key not in missing:
                continue
            raw_value = match.group(2)
            if key == "ID":
                out[key] = _postgwas_normalize_gff_ref_text(raw_value, split_multi=False)
            elif key == "Parent":
                out[key] = _postgwas_normalize_gff_ref_text(raw_value, split_multi=True)
            else:
                out[key] = _postgwas_clean_attr_text(raw_value)
            missing.discard(key)
            if len(missing) == 0:
                break
    return out


def _postgwas_read_minimal_gff(gff_path: str) -> pd.DataFrame:
    try:
        opener = gzip.open if str(gff_path).lower().endswith(".gz") else open
        chroms: list[str] = []
        features: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        strands: list[str] = []
        attr_ids: list[str] = []
        attr_parents: list[str] = []
        attr_descs: list[str] = []
        attr_names: list[str] = []

        skipped_too_short = 0
        with opener(gff_path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                line = line.rstrip("\r\n")
                if line == "":
                    continue
                parts = line.split("\t")
                if len(parts) < 9:
                    parts = re.split(r"\s+", line, maxsplit=8)
                if len(parts) < 9:
                    skipped_too_short += 1
                    continue
                if len(parts) > 9:
                    attr_head = parts[8].replace(";", " ")
                    attr_tail = " ".join(parts[9:]).strip()
                    merged_attr = f"{attr_head} {attr_tail}".strip() if attr_tail else attr_head
                    parts = parts[:8] + [re.sub(r"\s+", " ", merged_attr).strip()]

                try:
                    start_int = int(parts[3])
                    end_int = int(parts[4])
                except Exception:
                    continue

                feature = str(parts[2]).strip()
                if feature == "":
                    continue
                strand = str(parts[6]).strip()
                attr_text = str(parts[8]).strip()
                feature_lc = feature.lower()

                chroms.append(_normalize_chr(parts[0]))
                features.append(feature)
                starts.append(start_int)
                ends.append(end_int)
                strands.append(strand)
                attr_map = _postgwas_extract_attr_subset(
                    attr_text,
                    include_gene_meta=(feature_lc == "gene"),
                )
                attr_ids.append(attr_map.get("ID", "NA"))
                attr_parents.append(attr_map.get("Parent", "NA"))
                if feature_lc == "gene":
                    attr_descs.append(attr_map.get(_ANNO_DESC_KEY, "NA"))
                    attr_names.append(attr_map.get("Name", "NA"))
                else:
                    attr_descs.append("NA")
                    attr_names.append("NA")
    except Exception:
        # Fallback to the more permissive reader for malformed GFF variants.
        fallback_q = GFFQuery.from_file(gff_path, copy_df=False)
        fallback_gff = fallback_q.gff
        gff = fallback_gff.loc[
            :,
            [
                x
                for x in ("feature", "start", "end", "strand", "attributes", "chrom_norm")
                if x in fallback_gff.columns
            ],
        ].copy()
        return gff

    if len(starts) == 0:
        raise ValueError(f"No valid GFF rows found in file: {gff_path}")

    if skipped_too_short > 0:
        warnings.warn(
            f"Warning: shared GFF parser skipped malformed rows in {gff_path} "
            f"(<9 columns: {int(skipped_too_short)}).",
            RuntimeWarning,
            stacklevel=2,
        )

    gff = pd.DataFrame(
        {
            "feature": pd.Categorical(features),
            "start": np.asarray(starts, dtype=np.int64),
            "end": np.asarray(ends, dtype=np.int64),
            "strand": pd.Categorical(strands),
            "chrom_norm": pd.Categorical(chroms),
            _gff_prefetched_attr_colname("ID"): pd.Categorical(attr_ids),
            _gff_prefetched_attr_colname("Parent"): pd.Categorical(attr_parents),
            _gff_prefetched_attr_colname(_ANNO_DESC_KEY): pd.Categorical(attr_descs),
            _gff_prefetched_attr_colname("Name"): pd.Categorical(attr_names),
        }
    )
    gff = gff.sort_values(["chrom_norm", "start", "end"]).reset_index(drop=True)
    return gff


def _postgwas_get_shared_gff_query(gff_path: str) -> GFFQuery:
    global _POSTGWAS_SHARED_GFF_KEY
    global _POSTGWAS_SHARED_GFF_QUERY
    global _POSTGWAS_SHARED_GFF_ANNOTATION_CTX

    key = _postgwas_gff_cache_key(gff_path)
    if _POSTGWAS_SHARED_GFF_QUERY is None or _POSTGWAS_SHARED_GFF_KEY != key:
        _POSTGWAS_SHARED_GFF_QUERY = GFFQuery(
            _postgwas_read_minimal_gff(gff_path),
            copy_df=False,
        )
        _POSTGWAS_SHARED_GFF_KEY = key
        _POSTGWAS_SHARED_GFF_ANNOTATION_CTX = None
    return _POSTGWAS_SHARED_GFF_QUERY


def _postgwas_get_shared_gff_rust_index(gff_path: str) -> Optional[object]:
    global _POSTGWAS_SHARED_GFF_RUST_KEY
    global _POSTGWAS_SHARED_GFF_RUST_INDEX

    if not _postgwas_has_rust_gff_index():
        return None
    key = _postgwas_gff_cache_key(gff_path)
    if _POSTGWAS_SHARED_GFF_RUST_INDEX is None or _POSTGWAS_SHARED_GFF_RUST_KEY != key:
        _POSTGWAS_SHARED_GFF_RUST_INDEX = jxrs.GffAnnotationIndex.from_gff(
            str(gff_path),
            _postgwas_rust_gff_cache_path(gff_path),
        )
        _POSTGWAS_SHARED_GFF_RUST_KEY = key
    return _POSTGWAS_SHARED_GFF_RUST_INDEX


def _postgwas_get_shared_gff_annotation_context(gff_path: str) -> dict[str, object]:
    global _POSTGWAS_SHARED_GFF_ANNOTATION_CTX
    gff_query = _postgwas_get_shared_gff_query(gff_path)
    if _POSTGWAS_SHARED_GFF_ANNOTATION_CTX is None:
        _POSTGWAS_SHARED_GFF_ANNOTATION_CTX = _build_postgwas_gff_annotation_context(gff_query)
    return _POSTGWAS_SHARED_GFF_ANNOTATION_CTX


def _postgwas_get_gff_query(
    anno_file: Optional[str],
    *,
    use_shared: bool,
    current: Optional[GFFQuery] = None,
) -> Optional[GFFQuery]:
    if current is not None:
        return current
    if anno_file is None:
        return None
    if bool(use_shared):
        return _postgwas_get_shared_gff_query(str(anno_file))
    return GFFQuery.from_file(str(anno_file))


def _postgwas_get_gff_rust_index(
    anno_file: Optional[str],
    *,
    use_shared: bool,
    current: Optional[object] = None,
) -> Optional[object]:
    if current is not None:
        return current
    if anno_file is None or not _postgwas_has_rust_gff_index():
        return None
    if bool(use_shared):
        return _postgwas_get_shared_gff_rust_index(str(anno_file))
    return jxrs.GffAnnotationIndex.from_gff(
        str(anno_file),
        _postgwas_rust_gff_cache_path(str(anno_file)),
    )


def _postgwas_get_gff_annotation_context(
    anno_file: Optional[str],
    *,
    use_shared: bool,
    gff_query: Optional[GFFQuery] = None,
) -> Optional[dict[str, object]]:
    if anno_file is None:
        return None
    if bool(use_shared):
        return _postgwas_get_shared_gff_annotation_context(str(anno_file))
    query = _postgwas_get_gff_query(
        anno_file,
        use_shared=False,
        current=gff_query,
    )
    if query is None:
        return None
    return _build_postgwas_gff_annotation_context(query)


def _allow_windows_postgwas_process_pool() -> bool:
    """
    Windows matplotlib multi-process plotting has proven unstable in some
    environments (observed fail-fast 0xC0000409 crashes during ggval/postgwas).
    Keep a conservative serial default and allow explicit opt-in for further
    diagnostics.
    """
    if os.name != "nt":
        return True
    return _env_truthy("JANUSX_POSTGWAS_WINDOWS_PROCESS_POOL", False)


class _PostgwasFilePrefixFormatter(logging.Formatter):
    """Prefix warning/error records in worker append-mode log handlers."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if record.levelno >= logging.ERROR:
            return msg if msg.startswith("Error: ") else f"Error: {msg}"
        if record.levelno == logging.WARNING:
            return msg if msg.startswith("Warning: ") else f"Warning: {msg}"
        return msg


def _select_postgwas_mp_context():
    """
    Choose a multiprocessing start method for postgwas workers.

    Prefer spawn/forkserver to avoid Python 3.12+ warnings and potential
    deadlocks when forking from a multi-threaded parent after matplotlib /
    native runtimes have already been initialized. Allow manual override via
    JANUSX_POSTGWAS_MP_START_METHOD or the broader JANUSX_MP_START_METHOD.
    """
    try:
        methods = [str(m).strip().lower() for m in mp.get_all_start_methods()]
    except Exception:
        methods = []
    if len(methods) == 0:
        return None

    env_method = str(os.environ.get("JANUSX_POSTGWAS_MP_START_METHOD", "")).strip().lower()
    if env_method == "":
        env_method = str(os.environ.get("JANUSX_MP_START_METHOD", "")).strip().lower()
    if env_method != "" and env_method in methods:
        return mp.get_context(env_method)

    for name in ("spawn", "forkserver", "fork"):
        if name in methods:
            return mp.get_context(name)
    return None


def _build_postgwas_process_pool(max_workers: int) -> cf.ProcessPoolExecutor:
    kwargs: dict[str, Any] = {"max_workers": max(1, int(max_workers))}
    mp_ctx = _select_postgwas_mp_context()
    if mp_ctx is not None:
        kwargs["mp_context"] = mp_ctx
    return cf.ProcessPoolExecutor(**kwargs)


def _resolve_postgwas_worker_count(requested_threads: int, n_files: int) -> int:
    """
    Decide outer postgwas process count.

    Plotting workers are memory-heavy (pandas + matplotlib + optional LD/gene
    structures). Keep a conservative default cap, but allow explicit override.
    """
    req = max(1, int(requested_threads))
    total = max(1, int(n_files))

    env_raw = str(os.environ.get("JANUSX_POSTGWAS_MAX_WORKERS", "")).strip()
    if env_raw != "":
        try:
            env_cap = max(1, int(env_raw))
        except Exception:
            env_cap = None
        if env_cap is not None:
            return max(1, min(req, total, env_cap))

    default_cap = 4
    return max(1, min(req, total, default_cap))


def _log_postgwas_broken_pool_hint(
    logger: logging.Logger,
    *,
    n_workers: int,
    req_threads: int,
    n_files: int,
) -> None:
    mp_ctx = _select_postgwas_mp_context()
    method = ""
    try:
        if mp_ctx is not None:
            method = str(mp_ctx.get_start_method()).strip()
    except Exception:
        method = ""
    method_text = method if method != "" else "default"
    logger.error(
        "PostGWAS worker process exited unexpectedly. "
        f"Likely causes: native crash in matplotlib/numpy stack, OOM kill, or unsafe fork-style startup. "
        f"Current workers={int(n_workers)} (requested threads={int(req_threads)}, files={int(n_files)}), "
        f"mp_start={method_text}. "
        "Try rerunning with fewer outer workers, for example `-t 1` or "
        "`JANUSX_POSTGWAS_MAX_WORKERS=1`, and optionally force "
        "`JANUSX_POSTGWAS_MP_START_METHOD=spawn`."
    )


def _ensure_postgwas_worker_file_logging(args, logger: logging.Logger) -> logging.Logger:
    """
    Spawned workers do not inherit the parent's file handlers. Reattach an
    append-mode file handler so per-task logs are preserved in the main log.
    """
    log_path = str(getattr(args, "_postgwas_log_path", "")).strip()
    if log_path == "":
        return logger
    target = os.path.abspath(log_path)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        try:
            base = os.path.abspath(str(handler.baseFilename))
        except Exception:
            base = ""
        if base == target:
            return logger
    try:
        file_handler = logging.FileHandler(target, mode="a", encoding="utf-8")
    except Exception:
        return logger
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_PostgwasFilePrefixFormatter())
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logging.captureWarnings(True)
    return logger


def _sanitize_plot_text(text: object) -> str:
    s = str(text)
    if not _contains_cjk(s):
        return s
    if _ensure_cjk_font():
        return s
    # No CJK-capable font: fallback to ASCII to avoid glyph warnings.
    fallback = re.sub(r"[^\x00-\x7F]+", " ", s)
    fallback = re.sub(r"\s+", " ", fallback).strip()
    return fallback if fallback != "" else "NA"


def _strip_postgwas_input_suffix(path: object) -> str:
    name = os.path.basename(str(path).rstrip("/\\"))
    lower = name.lower()
    for ext in (".tsv.gz", ".txt.gz", ".csv.gz", ".tsv", ".txt", ".csv", ".gz"):
        if lower.endswith(ext):
            stem = name[: -len(ext)]
            return stem if stem != "" else name
    stem = os.path.splitext(name)[0]
    return stem if stem != "" else name


def _resolve_postgwas_output_stem(file: str, plot_prefix: object | None) -> str:
    base_stem = _strip_postgwas_input_suffix(file)
    prefix_text = str(plot_prefix).strip() if plot_prefix is not None else ""
    if prefix_text == "":
        return base_stem
    return f"{prefix_text}.{base_stem}"


def _prepare_cjk_plotting() -> None:
    # Try to enable a CJK font. If unavailable, silence glyph warnings and
    # fallback labels to ASCII where possible.
    if not _ensure_cjk_font():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"Glyph .* missing from font\(s\).*",
        )


def _postgwas_font_key(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _postgwas_looks_like_font_path(text: str) -> bool:
    token = str(text).strip()
    if token == "":
        return False
    if os.path.sep in token:
        return True
    if os.path.altsep is not None and os.path.altsep in token:
        return True
    return os.path.splitext(token)[1].lower() in _POSTGWAS_FONT_FILE_EXTENSIONS


@lru_cache(maxsize=1)
def _list_postgwas_font_names() -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in mpl_font_manager.fontManager.ttflist:
        name = str(getattr(entry, "name", "")).strip()
        if name == "":
            continue
        lower = name.lower()
        if lower in seen:
            continue
        seen.add(lower)
        names.append(name)
    names.sort(key=lambda x: x.lower())
    return tuple(names)


def _resolve_postgwas_fontstyle(value: object) -> tuple[str, str, str]:
    text = str(value).strip()
    if text == "":
        raise ValueError("-fontstyle/--fontstyle/--fontstype cannot be empty.")

    norm = _postgwas_font_key(text)
    if norm == "":
        raise ValueError(f"Invalid font selector: {text!r}")

    generic = _POSTGWAS_GENERIC_FONT_FAMILIES.get(norm)
    if generic is not None:
        return generic, generic, "generic"

    font_path = os.path.abspath(os.path.expanduser(text))
    if os.path.isfile(font_path):
        try:
            mpl_font_manager.fontManager.addfont(font_path)
            _list_postgwas_font_names.cache_clear()
            font_name = str(mpl_font_manager.FontProperties(fname=font_path).get_name()).strip()
        except Exception as e:
            raise ValueError(f"Failed to load font file: {text}") from e
        if font_name == "":
            raise ValueError(f"Unable to resolve font name from file: {text}")
        return font_name, font_path, "file"

    if _postgwas_looks_like_font_path(text):
        raise ValueError(f"Font file not found: {text}")

    names = list(_list_postgwas_font_names())
    if len(names) == 0:
        raise ValueError("No matplotlib fonts are available in the current environment.")

    lower_map = {name.lower(): name for name in names}
    if text.lower() in lower_map:
        chosen = lower_map[text.lower()]
        return chosen, chosen, "exact"

    key_map: dict[str, str] = {}
    for name in names:
        key = _postgwas_font_key(name)
        if key != "" and key not in key_map:
            key_map[key] = name
    if norm in key_map:
        chosen = key_map[norm]
        return chosen, chosen, "exact"

    prefix_matches = [
        name for name in names if _postgwas_font_key(name).startswith(norm)
    ]
    if len(prefix_matches) > 0:
        chosen = sorted(
            prefix_matches,
            key=lambda name: (len(_postgwas_font_key(name)), len(name), name.lower()),
        )[0]
        return chosen, chosen, "prefix"

    contains_matches = [
        name for name in names if norm in _postgwas_font_key(name)
    ]
    if len(contains_matches) > 0:
        chosen = sorted(
            contains_matches,
            key=lambda name: (len(_postgwas_font_key(name)), len(name), name.lower()),
        )[0]
        return chosen, chosen, "contains"

    fuzzy_keys = difflib.get_close_matches(
        norm,
        list(key_map.keys()),
        n=5,
        cutoff=0.45,
    )
    if len(fuzzy_keys) > 0:
        chosen = key_map[fuzzy_keys[0]]
        return chosen, chosen, "fuzzy"

    suggestions = difflib.get_close_matches(
        text.lower(),
        list(lower_map.keys()),
        n=5,
        cutoff=0.35,
    )
    if len(suggestions) > 0:
        tips = ", ".join(lower_map[x] for x in suggestions[:5])
        raise ValueError(f"Unknown font: {text}. Close matches: {tips}")
    raise ValueError(f"Unknown font: {text}")


def _postgwas_resolve_fontsize(
    args,
    *,
    manh_ratio: Optional[float] = None,
) -> float:
    manual_size = getattr(args, "fontsize", None)
    if manual_size is not None:
        return float(manual_size)
    base_size = float(
        getattr(args, "_postgwas_base_fontsize", _POSTGWAS_DEFAULT_FONT_SIZE)
    )
    if manh_ratio is None:
        return base_size
    return _scaled_fontsize_for_manhattan(
        manh_ratio,
        width_in=_PANEL_WIDTH_IN,
        base_font=base_size,
        min_scale=_POSTGWAS_MIN_FONT_SCALE,
    )


def _apply_postgwas_matplotlib_style(args) -> None:
    base_fontsize = float(
        getattr(args, "_postgwas_base_fontsize", _POSTGWAS_DEFAULT_FONT_SIZE)
    )
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["font.size"] = base_fontsize
    mpl.rcParams["axes.labelsize"] = base_fontsize
    mpl.rcParams["xtick.labelsize"] = base_fontsize
    mpl.rcParams["ytick.labelsize"] = base_fontsize
    mpl.rcParams["legend.fontsize"] = base_fontsize
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False
    _prepare_cjk_plotting()

    font_family = str(getattr(args, "_postgwas_font_family", "") or "").strip()
    if font_family == "":
        return

    current_sans = mpl.rcParams.get("font.sans-serif", [])
    if not isinstance(current_sans, list):
        current_sans = [str(current_sans)]

    family_list = [font_family]
    if font_family != "sans-serif":
        family_list.append("sans-serif")
    mpl.rcParams["font.family"] = family_list
    if font_family not in _POSTGWAS_GENERIC_FONT_FAMILIES.values():
        mpl.rcParams["font.sans-serif"] = [
            font_family
        ] + [str(x) for x in current_sans if str(x) != font_family]


def _resolve_postgwas_gene_panel_height_in(
    font_size: float,
    *,
    width_in: float = _PANEL_WIDTH_IN,
) -> float:
    base_height = float(width_in) / 20.0
    text_height = max(0.0, float(font_size)) / 72.0
    return max(base_height, text_height * 3.6)


def _apply_postgwas_gene_panel_layout(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    x_align_bounds: Optional[tuple[float, float, float, float]] = None,
) -> None:
    # Gene-only panels have no regular axis decorations and are often very short;
    # use deterministic vertical padding instead of tight_layout to avoid spurious
    # margin warnings when larger fonts or edge labels are present.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)
    if x_align_bounds is None:
        return
    mx0, _my0, mw, _mh = x_align_bounds
    _cx0, cy0, _cw, ch = ax.get_position().bounds
    ax.set_position([float(mx0), cy0, float(mw), ch])
    ax.set_anchor("N")


def _set_postgwas_axis_transparent(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    ax.patch.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.patch.set_visible(False)


def _resolve_postgwas_bridge_panel_height_in(
    font_size: float,
    *,
    width_in: float = _PANEL_WIDTH_IN,
    use_gene_bridge: bool = True,
) -> float:
    if not bool(use_gene_bridge):
        return 0.24
    gene_h_in = _resolve_postgwas_gene_panel_height_in(font_size, width_in=width_in)
    return max(
        min(gene_h_in, float(width_in) / 9.0),
        float(width_in) / 15.0,
    )


def _emit_info_to_file_handlers(logger: logging.Logger, message: str) -> None:
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args=(),
        exc_info=None,
    )
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.handle(record)


def _detach_stream_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """
    Temporarily detach non-file handlers from logger.
    Used by parallel workers to prevent progress-line corruption on TTY.
    """
    removed: list[logging.Handler] = []
    try:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                continue
            try:
                logger.removeHandler(handler)
                removed.append(handler)
            except Exception:
                continue
    except Exception:
        return []
    return removed


def _restore_handlers(logger: logging.Logger, handlers: list[logging.Handler]) -> None:
    if len(handlers) == 0:
        return
    try:
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
    except Exception:
        return


def _parse_ratio(value: object, name: str) -> float:
    """Parse aspect ratio string/number: supports '2', '1.25', '5/4'."""
    if value is None:
        raise ValueError(f"{name} ratio is None.")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{name} ratio is empty.")
    if "/" in text:
        parts = text.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"{name} ratio format error: {value}")
        num = float(parts[0].strip())
        den = float(parts[1].strip())
        if den == 0:
            raise ValueError(f"{name} ratio denominator cannot be zero.")
        ratio = num / den
    else:
        ratio = float(text)
    if ratio <= 0:
        raise ValueError(f"{name} ratio must be > 0.")
    return ratio


def _scaled_fontsize_for_manhattan(
    manh_ratio: Optional[float],
    *,
    width_in: float = 8.0,
    base_font: float = 6.0,
    ref_ratio: float = 2.0,
    min_scale: float = 0.55,
) -> float:
    """
    Scale font size with Manhattan panel height.
    As panel height decreases (ratio increases), font size decreases.
    """
    if manh_ratio is None:
        return float(base_font)
    try:
        ratio = float(manh_ratio)
    except Exception:
        return float(base_font)
    if not np.isfinite(ratio) or ratio <= 0:
        return float(base_font)

    ref_h = float(width_in) / float(ref_ratio)
    now_h = float(width_in) / float(ratio)
    if not (np.isfinite(ref_h) and ref_h > 0 and np.isfinite(now_h) and now_h > 0):
        return float(base_font)
    scale = float(np.sqrt(now_h / ref_h))
    scale = float(np.clip(scale, float(min_scale), 1.0))
    return float(base_font) * scale


def _parse_ldblock_spec(
    value: object,
    name: str,
    logger: logging.Logger,
) -> tuple[float, Optional[tuple[float, float]]]:
    """
    Parse ldblock option payload.
    Supports:
      - ratio only: "2", "5/4"
        -> x-span defaults to 0:1 (full Manhattan width)
      - x-span only (fraction of Manhattan width): "0.2:0.8" or "0.2-0.8"
        -> ratio defaults to 2.0
    """
    if value is None:
        raise ValueError(f"{name} is None.")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"{name} is empty.")

    m = re.match(r"^([0-9]*\.?[0-9]+)\s*(?:-|:)\s*([0-9]*\.?[0-9]+)$", text)
    if m is not None:
        x0 = float(m.group(1))
        x1 = float(m.group(2))
        if x0 < 0 or x1 < 0 or x0 > 1 or x1 > 1:
            raise ValueError(
                f"{name} x-span must be within [0, 1]: {text}."
            )
        if np.isclose(x0, x1):
            raise ValueError(f"{name} x-span start and end cannot be equal: {text}.")
        if x0 > x1:
            logger.warning(
                f"Warning: {name} x-span start > end ({x0} > {x1}); swapped to {x1}-{x0}."
            )
            x0, x1 = x1, x0
        return 2.0, (float(x0), float(x1))

    return _parse_ratio(text, name), (0.0, 1.0)


def _parse_rgb_triplet(token: str) -> str:
    text = token.strip()
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"Invalid RGB tuple token: {token}")
    parts = [p.strip() for p in text[1:-1].split(",")]
    if len(parts) != 3:
        raise ValueError(f"RGB tuple must have 3 values: {token}")
    try:
        rgb = [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"RGB tuple must be integers: {token}") from e
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"RGB tuple values must be in [0, 255]: {token}")
    return mcolors.to_hex([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0])


def _split_palette_tokens(text: str) -> list[str]:
    """
    Split palette string by ';' or ',' while preserving '(R,G,B)' tuples.
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in str(text):
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if depth == 0 and ch in {";", ","}:
            tok = "".join(buf).strip()
            if tok != "":
                out.append(tok)
            buf = []
            continue
        buf.append(ch)
    tok = "".join(buf).strip()
    if tok != "":
        out.append(tok)
    return out


def _parse_custom_palette(text: str) -> list[str]:
    colors: list[str] = []
    for token in _split_palette_tokens(text):
        tok = token.strip()
        if tok == "":
            continue
        if tok.startswith("(") and tok.endswith(")"):
            colors.append(_parse_rgb_triplet(tok))
            continue
        try:
            colors.append(mcolors.to_hex(mcolors.to_rgba(tok)))
        except ValueError as e:
            raise ValueError(
                f"Invalid --palette color token: {tok}. "
                "Use #RRGGBB or (R,G,B)."
            ) from e
    if len(colors) == 0:
        raise ValueError("Invalid --palette: empty color list.")
    return colors


def _expand_single_palette_color(base_color: str) -> list[str]:
    """
    Expand one color to two colors by grayscale(lightness) direction.
    - dark input  -> generate a darker mate
    - light input -> generate a lighter mate
    Returns [light, dark].
    """
    base = mcolors.to_hex(mcolors.to_rgba(base_color))
    gray = _relative_luminance(base)
    # Grayscale threshold for light/dark split.
    # Use stronger contrast so generated light/dark are visually distinct.
    if gray < 0.5:
        # dark input: push a clearly lighter mate + a deeper dark mate
        light = _blend_hex_color(base, "#ffffff", 0.85)
        dark = _blend_hex_color(base, "#000000", 0.35)
    else:
        # light input: push a clearly darker mate + an even lighter mate
        light = _blend_hex_color(base, "#ffffff", 0.35)
        dark = _blend_hex_color(base, "#000000", 0.85)
    return [light, dark]


def _parse_palette_spec(value: object) -> Optional[Tuple[str, Any]]:
    """
    Parse --palette into:
      - ("cmap", "<matplotlib cmap name>")
      - ("list", ["#hex1", "#hex2", ...])
      - None (use default black/grey)
    """
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("Invalid --palette: value is empty.")
    if (";" in text) or ("," in text):
        toks = _split_palette_tokens(text)
        if len(toks) >= 2:
            return ("list", _parse_custom_palette(text))
    try:
        plt.get_cmap(text)
        return ("cmap", text)
    except ValueError:
        # Allow single explicit color token (hex/name/rgb tuple) as shorthand.
        if text.startswith("(") and text.endswith(")"):
            return ("list", [_parse_rgb_triplet(text)])
        try:
            return ("list", [mcolors.to_hex(mcolors.to_rgba(text))])
        except ValueError as e:
            raise ValueError(
                f"Invalid --palette: {text}. "
                "Use a cmap name (e.g. tab10) or ';' / ',' separated colors."
            ) from e


def _blend_hex_color(c1: str, c2: str, ratio_to_c2: float) -> str:
    """Blend c1->c2 with ratio in [0, 1], return hex color."""
    t = float(np.clip(float(ratio_to_c2), 0.0, 1.0))
    rgb1 = np.asarray(mcolors.to_rgb(c1), dtype=float)
    rgb2 = np.asarray(mcolors.to_rgb(c2), dtype=float)
    out = (1.0 - t) * rgb1 + t * rgb2
    return mcolors.to_hex(out)


def _relative_luminance(color: str) -> float:
    """Simple RGB luminance in [0, 1] for light/dark ordering."""
    r, g, b = mcolors.to_rgb(color)
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def _resolve_two_color_style(
    spec: Optional[Tuple[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Return two-color style for QQ:
      - exactly two colors -> use as-is
      - single color       -> auto-expand by grayscale direction
    """
    if spec is None:
        return None

    mode, payload = spec
    colors: list[str]
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 1:
            colors = _expand_single_palette_color(colors[0])
        elif len(colors) != 2:
            return None
    elif mode == "cmap":
        cmap = plt.get_cmap(str(payload))
        if int(getattr(cmap, "N", 256)) != 2:
            return None
        colors = [mcolors.to_hex(cmap(0)), mcolors.to_hex(cmap(1))]
    else:
        return None

    lum0 = _relative_luminance(colors[0])
    lum1 = _relative_luminance(colors[1])
    if lum0 >= lum1:
        light, dark = colors[0], colors[1]
    else:
        light, dark = colors[1], colors[0]

    # Avoid fully-white fill in gene rectangles by nudging it 20% toward dark.
    if _relative_luminance(light) >= 0.98:
        light = _blend_hex_color(light, dark, 0.2)

    ld_cmap = mcolors.LinearSegmentedColormap.from_list(
        "janusx_ld_bicolor",
        [colors[0], colors[1]],
    )
    return {
        "colors": colors,
        "ld_cmap": ld_cmap,
        "gene_block_color": light,
        "gene_line_color": dark,
    }


def _resolve_ldblock_style(
    spec: Optional[Tuple[str, Any]],
) -> Optional[dict[str, Any]]:
    """
    Resolve LD-block colormap + matched gene colors.
    Supports cmap names and custom color lists (2+ colors).
    """
    if spec is None:
        return None

    mode, payload = spec
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 0:
            return None
        if len(colors) == 1:
            colors = _expand_single_palette_color(colors[0])
        ld_cmap = mcolors.LinearSegmentedColormap.from_list(
            "janusx_ld_custom",
            colors,
        )
    elif mode == "cmap":
        cmap = plt.get_cmap(str(payload))
        ld_cmap = cmap
        n_bins = int(getattr(cmap, "N", 256))
        if n_bins <= 32:
            colors = [
                mcolors.to_hex(cmap(i / max(1, n_bins - 1)))
                for i in range(max(2, n_bins))
            ]
        else:
            colors = [mcolors.to_hex(cmap(x)) for x in (0.0, 0.5, 1.0)]
    else:
        return None

    lums = np.asarray([_relative_luminance(c) for c in colors], dtype=float)
    i_light = int(np.argmax(lums))
    i_dark = int(np.argmin(lums))
    light = colors[i_light]
    dark = colors[i_dark]
    if _relative_luminance(light) >= 0.98:
        light = _blend_hex_color(light, dark, 0.2)

    return {
        "colors": colors,
        "ld_cmap": ld_cmap,
        "gene_block_color": light,
        "gene_line_color": dark,
    }


def _resolve_manhattan_colors(spec: Optional[Tuple[str, Any]], n_chr: int) -> Optional[list[str]]:
    if spec is None:
        return None
    mode, payload = spec
    if mode == "list":
        return list(payload)
    cmap_name = str(payload).strip()
    cmap = plt.get_cmap(cmap_name)
    # For discrete palettes (tab10/tab20/Set*, etc.), cycle native bins directly.
    # This avoids adjacent duplicate colors when n_chr > number of bins.
    if getattr(cmap, "N", 256) <= 32:
        n_bins = max(1, int(cmap.N))
        colors = [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_chr)]
    else:
        colors = [mcolors.to_hex(cmap(i / max(1, n_chr - 1))) for i in range(n_chr)]
    if cmap_name.lower() == "tab10":
        return [_desaturate_color(c, 0.80) for c in colors]
    return colors


def _desaturate_color(color: str, sat_scale: float = 0.80) -> str:
    r, g, b = mcolors.to_rgb(color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s2 = float(np.clip(float(s) * float(sat_scale), 0.0, 1.0))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s2, v)
    return mcolors.to_hex((r2, g2, b2))


def _resolve_merge_series_colors(
    spec: Optional[Tuple[str, Any]],
    n_series: int,
) -> list[str]:
    if n_series <= 0:
        return []

    if spec is None:
        cmap_name = "tab10" if n_series <= 10 else "tab20"
        cmap = plt.get_cmap(cmap_name)
        n_bins = max(1, int(getattr(cmap, "N", 10 if n_series <= 10 else 20)))
        colors = [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_series)]
        if cmap_name.lower() == "tab10":
            colors = [_desaturate_color(c, 0.90) for c in colors]
        return colors

    mode, payload = spec
    if mode == "list":
        colors = [mcolors.to_hex(mcolors.to_rgba(c)) for c in list(payload)]
        if len(colors) == 0:
            cmap = plt.get_cmap("tab20")
            n_bins = max(1, int(getattr(cmap, "N", 20)))
            return [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_series)]
        return [colors[i % len(colors)] for i in range(n_series)]

    cmap_name = str(payload).strip()
    cmap = plt.get_cmap(cmap_name)
    if getattr(cmap, "N", 256) <= 32:
        n_bins = max(1, int(cmap.N))
        colors = [mcolors.to_hex(cmap(i % n_bins)) for i in range(n_series)]
    else:
        colors = [mcolors.to_hex(cmap(i / max(1, n_series - 1))) for i in range(n_series)]
    if cmap_name.lower() == "tab10":
        colors = [_desaturate_color(c, 0.90) for c in colors]
    return colors


def _resolve_qq_point_color(spec: Optional[Tuple[str, Any]]) -> str:
    colors = _resolve_merge_series_colors(spec, 1)
    if len(colors) == 0:
        return "black"
    return str(colors[0])


def _split_cli_series_tokens(value: object, *, name: str) -> list[str]:
    if value is None:
        return []
    raw_items = list(value) if isinstance(value, (list, tuple)) else [value]
    out: list[str] = []
    for item in raw_items:
        text = str(item).strip()
        if text == "":
            continue
        parts = [tok.strip() for tok in re.split(r"[;,]", text) if tok.strip() != ""]
        if len(parts) == 0:
            continue
        out.extend(parts)
    if len(out) == 0:
        raise ValueError(f"Invalid {name}: no value provided.")
    return out


def _parse_scatter_size_spec(value: object) -> Optional[list[float]]:
    if value is None:
        return None
    tokens = _split_cli_series_tokens(value, name="--scatter-size")
    out: list[float] = []
    for tok in tokens:
        try:
            num = float(tok)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid --scatter-size token: {tok}") from e
        if (not np.isfinite(num)) or num <= 0.0:
            raise ValueError("--scatter-size values must be finite numbers > 0.")
        out.append(float(num))
    return out


def _parse_alpha_spec(value: object) -> Optional[list[float]]:
    if value is None:
        return None
    tokens = _split_cli_series_tokens(value, name="--alpha")
    out: list[float] = []
    for tok in tokens:
        try:
            num = float(tok)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid --alpha token: {tok}") from e
        if (not np.isfinite(num)) or num < 0.0 or num > 1.0:
            raise ValueError("--alpha values must be within [0, 1].")
        out.append(float(num))
    return out


def _resolve_single_series_value(
    spec: Optional[list[float]],
    default: float,
) -> float:
    if spec is None or len(spec) == 0:
        return float(default)
    return float(spec[0])


def _resolve_merge_series_values(
    spec: Optional[list[float]],
    n_series: int,
    *,
    default: float,
) -> list[float]:
    if n_series <= 0:
        return []
    base = list(spec) if spec is not None and len(spec) > 0 else [float(default)]
    return [float(base[i % len(base)]) for i in range(n_series)]


def _parse_marker_spec(value: object) -> Optional[list[str]]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        raise ValueError("Invalid --marker: value is empty.")
    raw_tokens = [tok.strip() for tok in re.split(r"[;,]", text) if tok.strip() != ""]
    if len(raw_tokens) == 0:
        raise ValueError("Invalid --marker: no marker token found.")
    out: list[str] = []
    for tok in raw_tokens:
        try:
            MarkerStyle(tok)
        except Exception as e:
            raise ValueError(
                f"Invalid --marker token: {tok}. "
                "Examples: o, x, +, *, 1, 2, 3, 4."
            ) from e
        out.append(str(tok))
    return out


def _resolve_single_marker(spec: Optional[list[str]]) -> str:
    if spec is None or len(spec) == 0:
        return str(_DEFAULT_SINGLE_MARKER)
    return str(spec[0])


def _resolve_merge_markers(spec: Optional[list[str]], n_series: int) -> list[str]:
    if n_series <= 0:
        return []
    base = list(spec) if spec is not None and len(spec) > 0 else list(_DEFAULT_MERGE_MARKERS)
    return [str(base[i % len(base)]) for i in range(n_series)]


def _marker_scatter_style(marker: str) -> dict[str, object]:
    try:
        marker_obj = MarkerStyle(str(marker))
        is_filled = bool(marker_obj.is_filled())
    except Exception:
        is_filled = True
    if is_filled:
        return {
            "edgecolors": "none",
            "linewidths": 0.0,
        }
    return {
        "linewidths": 0.8,
    }


def _natural_tokens(text: str) -> tuple[tuple[int, object], ...]:
    tokens: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        if part.isdigit():
            tokens.append((0, int(part)))
        else:
            tokens.append((1, part.lower()))
    # Keep sort tokens hashable so they can be used safely in pandas sort keys.
    return tuple(tokens)


def _chrom_sort_key(label: object) -> tuple[int, object]:
    if pd.isna(label):
        return (3, "")

    if isinstance(label, (int, np.integer)):
        return (0, int(label))
    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        return (0, int(label))

    text = str(label).strip()
    no_chr_prefix = text[3:] if text.lower().startswith("chr") else text
    upper = no_chr_prefix.upper()

    if no_chr_prefix.isdigit():
        return (0, int(no_chr_prefix))

    special_chr = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if upper in special_chr:
        return (1, special_chr[upper])

    return (2, _natural_tokens(no_chr_prefix))


def _manhattan_colors_for_subset(
    spec: Optional[Tuple[str, Any]],
    full_chr_labels: list[object],
    subset_chr_labels: list[object],
) -> list[str]:
    full_order = sorted(pd.unique(pd.Series(full_chr_labels)).tolist(), key=_chrom_sort_key)
    subset_order = sorted(pd.unique(pd.Series(subset_chr_labels)).tolist(), key=_chrom_sort_key)
    if len(full_order) == 0 or len(subset_order) == 0:
        return ["black", "grey"]

    full_colors = _resolve_manhattan_colors(spec, len(full_order))
    if full_colors is None:
        full_colors = [("black" if i % 2 == 0 else "grey") for i in range(len(full_order))]

    color_by_chr = {chrom: full_colors[i % len(full_colors)] for i, chrom in enumerate(full_order)}
    return [color_by_chr[c] for c in subset_order]


def _normalize_chr(value: object) -> str:
    s = str(value).strip()
    if s.lower().startswith("chr"):
        s = s[3:]
    return s


def _postgwas_finemap_normalize_chr(value: object) -> str:
    """Canonicalize chromosome tokens for case-insensitive fine-mapping joins."""
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            value = int(numeric)
    chrom = _normalize_chr(value)
    return chrom if chrom.isdigit() else chrom.upper()


def _postgwas_finemap_locus_label(item: tuple[str, int, int]) -> str:
    """Return the stable label used to identify one fine-mapping locus."""
    chrom, start_bp, end_bp = item
    return f"{_postgwas_finemap_normalize_chr(chrom)}:{int(start_bp)}-{int(end_bp)}"


_POSTGWAS_FINEMAP_DEFAULT_MEMORY_GB = 8.0
_POSTGWAS_FINEMAP_MAX_MEMORY_BYTES = int(_POSTGWAS_FINEMAP_DEFAULT_MEMORY_GB * 1024**3)
_POSTGWAS_FINEMAP_MEMORY_RESERVE_BYTES = 512 * 1024**2
_POSTGWAS_FVLMM_DIAG_RIDGE = 1e-6
# Effective-LD validation tolerances.  Symmetry and unit diagonal use an
# absolute tolerance of 1e-10.  PSD accepts only round-off-scale negative
# eigenvalues: max(1e-10, 1e-8 * max(1, max_abs_eigenvalue)).
_POSTGWAS_FVLMM_LD_SYMMETRY_ATOL = 1e-10
_POSTGWAS_FVLMM_LD_UNIT_DIAGONAL_ATOL = 1e-10
_POSTGWAS_FVLMM_LD_PSD_ATOL = 1e-10
_POSTGWAS_FVLMM_LD_PSD_RTOL = 1e-8
_POSTGWAS_FINEMAP_LDCLUMP_R2 = 0.99
_POSTGWAS_FINEMAP_PURITY_R2_DEFAULT = 0.25
_POSTGWAS_SUSIE_LOCUS_NEUTRAL_COLOR = "#9A9A91"
_POSTGWAS_SUSIE_PIP_HEADROOM_THRESHOLD = 0.98
_POSTGWAS_SUSIE_PIP_Y_MAX_WITH_HEADROOM = 1.05
_POSTGWAS_SUSIE_PIP_FLOOR_THRESHOLD = 0.02
_POSTGWAS_SUSIE_PIP_Y_MIN_WITH_HEADROOM = -0.05
_POSTGWAS_MIXED_MODEL_RESULT_SUFFIXES = (
    (".fvlmm.tsv", "fvlmm"),
    (".lmm.tsv", "lmm"),
    (".lmm2.tsv", "lmm2"),
    (".splmm.tsv", "splmm"),
    (".splmm2.tsv", "splmm2"),
)


@dataclass
class FvLMMFineMapContext:
    """Verified sample, fixed-effect, and null-model state for FvLMM LD."""

    sample_ids: np.ndarray
    sample_indices_in_bfile: np.ndarray
    fixed_effects: np.ndarray
    kinship: np.ndarray
    lambda_null: float
    sidecar: GwasNullModelSidecarV1


class _PostGWASExpectedCompatibilityError(Exception):
    """Explicit source-adapter sentinel for expected compatibility failures."""


def _postgwas_skip_for_expected_source_error(
    exc: BaseException,
    label: str,
    *,
    value_markers: Sequence[str] = (),
    runtime_markers: Sequence[str] = (),
) -> None:
    """Convert only an explicit/recognized source failure into FineMapSkip."""
    if isinstance(exc, _PostGWASExpectedCompatibilityError):
        raise FineMapSkip(f"{label} is incompatible: {exc}") from exc
    if isinstance(exc, (OSError, EOFError, UnicodeError, pd.errors.ParserError)):
        raise FineMapSkip(f"{label} could not be reconstructed: {exc}") from exc
    message = str(exc).lower()
    if type(exc) is ValueError and any(marker in message for marker in value_markers):
        raise FineMapSkip(f"{label} contains malformed user data: {exc}") from exc
    if type(exc) is RuntimeError and any(marker in message for marker in runtime_markers):
        raise FineMapSkip(f"{label} is unavailable in this runtime: {exc}") from exc


def _postgwas_validate_fvlmm_sidecar(
    record: GwasNullModelSidecarV1,
) -> None:
    """Validate metadata needed by the Task 5 reconstruction before I/O."""
    if not isinstance(record, GwasNullModelSidecarV1):
        raise FineMapSkip("FvLMM sidecar metadata has an invalid record type")
    try:
        # Reuse the schema's complete semantic validation, including the
        # fixed-effect/covariate relationship established by Task 3.
        serialize_sidecar_block(record)
    except (SidecarFormatError, TypeError, ValueError) as exc:
        raise FineMapSkip(f"sidecar metadata is incompatible: {exc}") from exc

    if record.model != "fvlmm":
        raise FineMapSkip(
            f"sidecar model {record.model!r} is not compatible with FvLMM reconstruction"
        )
    fixed_columns = tuple(str(column) for column in record.fixed_effect_columns)
    covariate_columns = tuple(str(column) for column in record.covariate_columns)
    qmatrix_columns = tuple(str(column) for column in record.qmatrix_columns)
    if fixed_columns != ("Intercept",) + covariate_columns:
        raise FineMapSkip(
            "sidecar fixed-effect metadata disagrees with its covariate columns"
        )
    if qmatrix_columns and record.qmatrix_file is None:
        raise FineMapSkip("sidecar Q/PC columns have no numerical source")
    if record.qmatrix_file is not None and not qmatrix_columns:
        raise FineMapSkip("sidecar Q/PC source has no column identity")
    try:
        lambda_null = float(record.lambda_null)
    except (TypeError, ValueError) as exc:
        raise FineMapSkip("FvLMM sidecar lambda is invalid") from exc
    if not np.isfinite(lambda_null) or lambda_null <= 0.0:
        raise FineMapSkip("FvLMM sidecar lambda must be finite and positive")
    if str(record.kinship_format).strip().lower() != "dense":
        raise FineMapSkip(
            f"FvLMM sidecar GRM format {record.kinship_format!r} is unsupported"
        )
    try:
        kinship_shape = tuple(int(value) for value in record.kinship_shape)
    except (TypeError, ValueError) as exc:
        raise FineMapSkip("FvLMM sidecar GRM shape is invalid") from exc
    if (
        len(kinship_shape) != 2
        or kinship_shape[0] <= 0
        or kinship_shape[1] <= 0
        or kinship_shape[0] != kinship_shape[1]
    ):
        raise FineMapSkip(
            f"FvLMM sidecar GRM shape must be non-empty square, got {record.kinship_shape!r}"
        )
    if int(record.sample_count) <= 0:
        raise FineMapSkip("FvLMM sidecar sample count must be positive")


def _postgwas_resolve_fvlmm_dependency(
    fingerprint: object,
    label: str,
    dependency_roots: Optional[Sequence[Path]],
) -> Path:
    """Resolve a validated sidecar dependency, including moved bundles."""
    try:
        if dependency_roots is not None:
            return Path(
                _resolve_dependency_path(
                    fingerprint,
                    dependency_roots,
                    label,
                )
            )
        path = Path(str(getattr(fingerprint, "canonical_path"))).expanduser()
        if not path.is_file():
            raise FineMapSkip(f"{label} file not found: {path}")
        return path
    except FineMapSkip:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise FineMapSkip(f"{label} dependency is unavailable: {exc}") from exc


def _postgwas_require_nonempty_source(path: Path, label: str) -> None:
    try:
        if path.stat().st_size <= 0:
            raise FineMapSkip(f"{label} file is empty: {path}")
    except FineMapSkip:
        raise
    except OSError as exc:
        raise FineMapSkip(f"{label} file could not be inspected: {exc}") from exc


def _postgwas_fvlmm_genotype_fam_path(
    record: GwasNullModelSidecarV1,
    genotype_prefix: Optional[str | os.PathLike[str]],
) -> Path:
    if genotype_prefix is not None and str(genotype_prefix).strip() != "":
        prefix = _normalize_plink_prefix(genotype_prefix)
        return Path(f"{prefix}.fam")
    for fingerprint in record.genotype_files:
        if str(getattr(fingerprint, "basename", "")).lower().endswith(".fam"):
            return Path(str(fingerprint.canonical_path)).expanduser()
    prefix = _normalize_plink_prefix(record.genotype_prefix)
    return Path(f"{prefix}.fam")


def _postgwas_load_qmatrix_source(
    path: Path,
    expected_columns: Sequence[object],
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the persisted Q/PC source used by GWAS."""

    try:
        source = np.load(path, allow_pickle=False)
        if not isinstance(source, np.lib.npyio.NpzFile):
            raise ValueError("Q/PC source is not the required NPZ format")
        with source:
            if "sample_ids" not in source or "qmatrix" not in source:
                raise KeyError("NPZ must contain sample_ids and qmatrix")
            sample_ids_raw = np.asarray(source["sample_ids"])
            qmatrix_raw = np.asarray(source["qmatrix"])
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise FineMapSkip(f"Q/PC matrix could not be reconstructed: {exc}") from exc

    if sample_ids_raw.ndim != 1 or qmatrix_raw.ndim != 2:
        raise FineMapSkip("Q/PC matrix source has invalid array dimensions")
    if qmatrix_raw.shape[0] != sample_ids_raw.size:
        raise FineMapSkip("Q/PC matrix source IDs and rows disagree")
    if qmatrix_raw.shape[1] != len(tuple(expected_columns)):
        raise FineMapSkip(
            "Q/PC matrix source column count disagrees with sidecar metadata"
        )
    try:
        numeric = np.issubdtype(qmatrix_raw.dtype, np.number)
    except TypeError as exc:
        raise FineMapSkip("Q/PC matrix source is not numeric") from exc
    if not numeric or not np.all(np.isfinite(qmatrix_raw)):
        raise FineMapSkip("Q/PC matrix source contains non-finite/non-numeric values")
    sample_ids = np.asarray(sample_ids_raw, dtype=str).reshape(-1)
    if sample_ids.size == 0 or len(set(sample_ids.tolist())) != sample_ids.size:
        raise FineMapSkip("Q/PC matrix source contains duplicate or empty IDs")
    qmatrix = np.ascontiguousarray(
        np.asarray(qmatrix_raw, dtype=np.float64), dtype=np.float64
    )
    if not np.all(np.isfinite(qmatrix)):
        raise FineMapSkip("Q/PC matrix source conversion produced non-finite values")
    return sample_ids, qmatrix


def _postgwas_read_npy_storage_header(
    handle: object,
    label: str,
) -> tuple[tuple[int, ...], np.dtype]:
    """Read an NPY shape/dtype header without touching its data payload."""
    try:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        elif version == (3, 0):
            reader = getattr(np.lib.format, "read_array_header_3_0", None)
            if reader is None:
                raise ValueError("NumPy does not provide a v3 header reader")
            shape, _fortran_order, dtype = reader(handle)
        else:
            raise ValueError(f"unsupported NPY header version {version!r}")
    except (OSError, EOFError, ValueError, TypeError) as exc:
        raise FineMapSkip(f"{label} storage header is malformed: {exc}") from exc
    try:
        normalized_shape = tuple(int(dimension) for dimension in shape)
        normalized_dtype = np.dtype(dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FineMapSkip(f"{label} storage header has invalid shape/dtype") from exc
    if any(dimension < 0 for dimension in normalized_shape):
        raise FineMapSkip(f"{label} storage header has a negative dimension")
    return normalized_shape, normalized_dtype


def _postgwas_inspect_qmatrix_storage_header(
    path: Path,
    expected_columns: Sequence[object],
) -> tuple[int, int]:
    """Inspect Q/PC NPZ member headers without loading numeric arrays."""
    if path.suffix.lower() != ".npz":
        raise FineMapSkip("Q/PC source has no safe header-only inspector")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = set(archive.namelist())
            shapes: dict[str, tuple[int, ...]] = {}
            dtypes: dict[str, np.dtype] = {}
            for key in ("sample_ids", "qmatrix"):
                member = f"{key}.npy"
                if member not in members:
                    raise ValueError(f"NPZ is missing {key}")
                with archive.open(member, "r") as member_handle:
                    shapes[key], dtypes[key] = _postgwas_read_npy_storage_header(
                        member_handle,
                        f"Q/PC {key}",
                    )
    except FineMapSkip:
        raise
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise FineMapSkip(f"Q/PC storage header is malformed: {exc}") from exc

    if len(shapes["sample_ids"]) != 1 or len(shapes["qmatrix"]) != 2:
        raise FineMapSkip("Q/PC storage header has invalid array dimensions")
    q_rows, q_columns = shapes["qmatrix"]
    if shapes["sample_ids"][0] != q_rows:
        raise FineMapSkip("Q/PC storage header IDs and rows disagree")
    if q_columns != len(tuple(expected_columns)):
        raise FineMapSkip("Q/PC storage header columns disagree with sidecar metadata")
    if dtypes["sample_ids"].kind not in {"U", "S"}:
        raise FineMapSkip("Q/PC sample ID storage dtype is not a safe string type")
    try:
        if not np.issubdtype(dtypes["qmatrix"], np.number):
            raise ValueError("Q/PC storage dtype is not numeric")
    except TypeError as exc:
        raise FineMapSkip("Q/PC storage dtype is invalid") from exc
    return int(shapes["sample_ids"][0]), int(q_columns)


def _postgwas_inspect_grm_storage_header(path: Path) -> tuple[int, int]:
    """Inspect supported dense GRM storage shape without materializing it."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        try:
            with path.open("rb") as handle:
                shape, dtype = _postgwas_read_npy_storage_header(handle, "GRM")
        except OSError as exc:
            raise FineMapSkip(f"GRM storage header could not be read: {exc}") from exc
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] <= 0:
            raise FineMapSkip(f"GRM storage header is not a non-empty square: {shape!r}")
        try:
            if not np.issubdtype(dtype, np.number):
                raise ValueError("GRM storage dtype is not numeric")
        except TypeError as exc:
            raise FineMapSkip("GRM storage dtype is invalid") from exc
        return int(shape[0]), int(shape[1])
    if suffix not in {".txt", ".tsv", ".csv"}:
        raise FineMapSkip(
            f"GRM format {suffix or '<none>'!r} has no safe header-only inspector"
        )

    row_count = 0
    column_count: Optional[int] = None
    try:
        with path.open("rt", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if line == "":
                    continue
                tokens = line.replace(",", " ").split()
                if column_count is None:
                    column_count = len(tokens)
                if len(tokens) != column_count:
                    raise ValueError(
                        f"GRM text row width mismatch at row {line_number}"
                    )
                for token in tokens:
                    float(token)
                row_count += 1
    except (OSError, UnicodeError, ValueError) as exc:
        raise FineMapSkip(f"GRM text storage header is malformed: {exc}") from exc
    if column_count is None or row_count <= 0 or row_count != column_count:
        raise FineMapSkip(
            f"GRM text storage is not a non-empty square: ({row_count}, {column_count})"
        )
    return int(row_count), int(column_count)


def _postgwas_inspect_id_storage(path: Path, label: str) -> int:
    """Count and validate ID rows without constructing a matrix."""
    count = 0
    seen: set[str] = set()
    try:
        with path.open("rt", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                tokens = raw_line.split()
                if not tokens:
                    continue
                sample_id = str(tokens[0])
                if sample_id in seen:
                    raise ValueError(f"duplicate sample ID {sample_id!r}")
                seen.add(sample_id)
                count += 1
    except (OSError, UnicodeError, ValueError) as exc:
        raise FineMapSkip(f"{label} storage is malformed: {exc}") from exc
    if count <= 0:
        raise FineMapSkip(f"{label} storage is empty")
    return count


def _postgwas_validate_kernel_valid_indices(
    raw_indices: object,
    upper_bound: int,
) -> np.ndarray:
    """Validate raw kernel indices before any narrowing integer cast."""
    try:
        values = np.asarray(raw_indices)
    except (TypeError, ValueError) as exc:
        raise FineMapSkip(f"kernel valid_indices are not an array: {exc}") from exc
    if values.ndim != 1 or values.size == 0:
        raise FineMapSkip("kernel valid_indices must be a non-empty one-dimensional array")
    kind = values.dtype.kind
    int64_limit = 1 << 63
    if kind == "b":
        raise FineMapSkip("kernel valid_indices must contain exact integers, not booleans")
    if kind in "iu":
        if kind == "i" and np.any(values < 0):
            raise FineMapSkip("kernel valid_indices contain a negative index")
        if kind == "u" and np.any(values > np.uint64(int64_limit - 1)):
            raise FineMapSkip("kernel valid_indices exceed int64 range")
        indices = np.asarray(values, dtype=np.int64)
    elif kind == "f":
        if not np.all(np.isfinite(values)):
            raise FineMapSkip("kernel valid_indices must be finite")
        if np.any(values != np.trunc(values)):
            raise FineMapSkip("kernel valid_indices must be exact integers")
        if np.any(values < 0) or np.any(values >= float(int64_limit)):
            raise FineMapSkip("kernel valid_indices are outside int64 range")
        indices = np.asarray(values, dtype=np.int64)
    else:
        converted: list[int] = []
        for value in values.tolist():
            if isinstance(value, (bool, np.bool_)):
                raise FineMapSkip("kernel valid_indices must contain exact integers")
            if isinstance(value, (int, np.integer)):
                integer = int(value)
            elif isinstance(value, (float, np.floating)):
                numeric = float(value)
                if not np.isfinite(numeric) or numeric != np.trunc(numeric):
                    raise FineMapSkip("kernel valid_indices must be finite exact integers")
                if numeric < 0 or numeric >= float(int64_limit):
                    raise FineMapSkip("kernel valid_indices are outside int64 range")
                integer = int(numeric)
            else:
                raise FineMapSkip("kernel valid_indices contain non-numeric values")
            if integer < 0 or integer >= int64_limit:
                raise FineMapSkip("kernel valid_indices are outside int64 range")
            converted.append(integer)
        indices = np.asarray(converted, dtype=np.int64)
    if np.any(indices >= int(upper_bound)):
        raise FineMapSkip("kernel valid_indices contain an out-of-range index")
    if np.unique(indices).size != indices.size:
        raise FineMapSkip("kernel valid_indices contain duplicate indices")
    return np.ascontiguousarray(indices, dtype=np.int64)


def _postgwas_variant_token(value: object, *, allele: bool = False) -> str | None:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text == "" or text.lower() in {".", "nan", "none"}:
        return None
    return text.upper() if allele else text


def _postgwas_variant_allele_pair(
    allele0: str | None,
    allele1: str | None,
) -> frozenset[str] | None:
    if allele0 is None or allele1 is None:
        return None
    return frozenset((allele0, allele1))


def _postgwas_scan_bim_candidates(
    handle: Iterable[str],
    target_sites: set[tuple[str, int]],
) -> tuple[
    dict[
        tuple[str, int],
        list[tuple[int, tuple[str, int, str | None, str | None, str | None]]],
    ],
    int,
]:
    """Scan BIM once and retain identity metadata only for target coordinates."""

    candidates = {site: [] for site in target_sites}
    source_index = 0
    for line_number, line in enumerate(handle, start=1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < 6:
            raise ValueError(f"Malformed BIM row at line {line_number}")
        chrom = _normalize_chr(tokens[0])
        try:
            pos = int(float(tokens[3]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid BIM position at line {line_number}") from exc
        metadata = (
            chrom,
            pos,
            _postgwas_variant_token(tokens[1]),
            _postgwas_variant_token(tokens[4], allele=True),
            _postgwas_variant_token(tokens[5], allele=True),
        )
        site = (chrom, pos)
        if site in candidates:
            candidates[site].append((source_index, metadata))
        source_index += 1
    return candidates, source_index


def _postgwas_index_bim_candidates(
    candidates_by_site: dict[
        tuple[str, int],
        list[tuple[int, tuple[str, int, str | None, str | None, str | None]]],
    ],
) -> dict[
    tuple[str, int],
    tuple[
        list[tuple[int, tuple[str, int, str | None, str | None, str | None]]],
        dict[str, list[tuple[int, tuple[str, int, str | None, str | None, str | None]]]],
        dict[
            frozenset[str],
            list[tuple[int, tuple[str, int, str | None, str | None, str | None]]],
        ],
        dict[
            tuple[str, frozenset[str]],
            list[tuple[int, tuple[str, int, str | None, str | None, str | None]]],
        ],
        bool,
    ],
]:
    """Index retained BIM candidates once per target coordinate."""

    indexed = {}
    for site, candidates in candidates_by_site.items():
        by_id: dict[
            str, list[tuple[int, tuple[str, int, str | None, str | None, str | None]]]
        ] = {}
        by_allele_pair: dict[
            frozenset[str],
            list[tuple[int, tuple[str, int, str | None, str | None, str | None]]],
        ] = {}
        by_id_and_allele_pair: dict[
            tuple[str, frozenset[str]],
            list[tuple[int, tuple[str, int, str | None, str | None, str | None]]],
        ] = {}
        has_named_id = False
        for candidate in candidates:
            metadata = candidate[1]
            if metadata[2] is not None:
                has_named_id = True
                by_id.setdefault(metadata[2], []).append(candidate)
            allele_pair = _postgwas_variant_allele_pair(metadata[3], metadata[4])
            if allele_pair is not None:
                by_allele_pair.setdefault(allele_pair, []).append(candidate)
                if metadata[2] is not None:
                    by_id_and_allele_pair.setdefault(
                        (metadata[2], allele_pair), []
                    ).append(candidate)
        indexed[site] = (
            candidates,
            by_id,
            by_allele_pair,
            by_id_and_allele_pair,
            has_named_id,
        )
    return indexed


def _postgwas_resolve_prepared_bim_rows(
    genotype_prefix: object,
    prepared: pd.DataFrame,
) -> tuple[
    list[int],
    list[tuple[str, int, str | None, str | None, str | None]],
]:
    """Resolve prepared variants from one BIM scan without materializing BIM."""

    prepared_identity: list[tuple[str, int, str | None, str | None, str | None]] = []
    target_sites: set[tuple[str, int]] = set()
    used: set[int] = set()
    has_snp = "snp" in prepared.columns
    has_allele0 = "allele0" in prepared.columns
    has_allele1 = "allele1" in prepared.columns
    for row_number, (_index, row) in enumerate(prepared.iterrows()):
        chrom = _normalize_chr(row["chrom"])
        try:
            pos_float = float(pd.to_numeric(row["pos"], errors="coerce"))
        except (TypeError, ValueError):
            pos_float = float("nan")
        if not np.isfinite(pos_float) or pos_float != np.floor(pos_float):
            raise FineMapSkip(
                f"FvLMM variant identity has an invalid position at row {row_number}"
            )
        pos = int(pos_float)
        prepared_snp = (
            _postgwas_variant_token(row["snp"]) if has_snp else None
        )
        prepared_a0 = (
            _postgwas_variant_token(row["allele0"], allele=True)
            if has_allele0
            else None
        )
        prepared_a1 = (
            _postgwas_variant_token(row["allele1"], allele=True)
            if has_allele1
            else None
        )
        prepared_identity.append(
            (chrom, pos, prepared_snp, prepared_a0, prepared_a1)
        )
        target_sites.add((chrom, pos))

    prefix = _normalize_plink_prefix(genotype_prefix)
    bim_path = Path(f"{prefix}.bim")
    try:
        with bim_path.open("rt", encoding="utf-8", errors="replace") as handle:
            candidates_by_site, source_rows = _postgwas_scan_bim_candidates(
                handle, target_sites
            )
    except (OSError, ValueError) as exc:
        raise FineMapSkip(f"BIM variant identity could not be read: {exc}") from exc
    if source_rows == 0:
        raise FineMapSkip("BIM variant identity is empty")
    indexed_candidates_by_site = _postgwas_index_bim_candidates(candidates_by_site)

    selected_indices: list[int] = []
    selected_metadata: list[tuple[str, int, str | None, str | None, str | None]] = []
    for row_number, (chrom, pos, prepared_snp, prepared_a0, prepared_a1) in enumerate(
        prepared_identity
    ):
        (
            all_candidates,
            candidates_by_id,
            candidates_by_allele_pair,
            candidates_by_id_and_allele_pair,
            has_named_id,
        ) = indexed_candidates_by_site[(chrom, pos)]
        candidates = all_candidates
        if not candidates:
            raise FineMapSkip(
                f"no BIM variant matches prepared row {row_number} at {chrom}:{pos}"
            )
        prepared_pair = _postgwas_variant_allele_pair(prepared_a0, prepared_a1)
        id_matches = candidates_by_id.get(prepared_snp, []) if prepared_snp is not None else []
        if id_matches:
            candidates = id_matches
        elif prepared_snp is not None and has_named_id:
            # A named variant that cannot be found by ID is not proven by a
            # coordinate fallback when BIM contains real IDs.
            raise FineMapSkip(
                f"prepared SNP ID {prepared_snp!r} is not present in BIM; "
                "variant identity cannot be proven"
            )
        if prepared_pair is not None:
            if not (has_allele0 and has_allele1):
                raise FineMapSkip(
                    f"prepared variant row {row_number} has incomplete allele identity"
                )
            allele_matches = candidates_by_allele_pair.get(prepared_pair, [])
            candidates = (
                allele_matches
                if candidates is all_candidates
                else candidates_by_id_and_allele_pair.get(
                    (prepared_snp, prepared_pair), []
                )
            )
        if len(candidates) != 1:
            raise FineMapSkip(
                "variant identity is ambiguous; exact BIM alignment cannot be proven "
                f"for prepared row {row_number} at {chrom}:{pos}"
            )
        bim_index, bim_metadata = candidates[0]
        if bim_index in used:
            raise FineMapSkip("prepared variants resolve to the same BIM row")
        used.add(bim_index)
        selected_indices.append(bim_index)
        selected_metadata.append(bim_metadata)
    return selected_indices, selected_metadata


def _postgwas_site_identity_signature(site: object) -> tuple[str, int, frozenset[str] | None]:
    try:
        chrom = _normalize_chr(getattr(site, "chrom"))
        pos = int(getattr(site, "pos"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise FineMapSkip("genotype loader returned invalid site metadata") from exc
    allele0 = _postgwas_variant_token(
        getattr(site, "ref_allele", getattr(site, "allele0", None)),
        allele=True,
    )
    allele1 = _postgwas_variant_token(
        getattr(site, "alt_allele", getattr(site, "allele1", None)),
        allele=True,
    )
    return chrom, pos, _postgwas_variant_allele_pair(allele0, allele1)


def _postgwas_reconstruct_fvlmm_context(
    record: GwasNullModelSidecarV1,
    *,
    genotype_prefix: Optional[str | os.PathLike[str]] = None,
    dependency_roots: Optional[Sequence[Path]] = None,
    logger: Optional[logging.Logger] = None,
) -> FvLMMFineMapContext:
    """Reconstruct the GWAS FvLMM sample order and fixed-effect design.

    The source loaders used here are the same workflow loaders used by GWAS:
    phenotype parsing/duplicate-ID averaging, covariate parsing, FAM IID
    extraction, and GRM ID-based alignment.  The full GRM is intentionally
    loaded only after the final ordered sample hash has been checked.
    """
    _postgwas_validate_fvlmm_sidecar(record)
    log = logger if isinstance(logger, logging.Logger) else logging.getLogger(__name__)

    from janusx.assoc.workflow import (
        _TraitRef,
        _align_pheno_to_sample_order,
        _read_cov_file_flexible,
        _trait_values_and_mask,
        load_phenotype,
    )
    from janusx.script._common.genoio import read_id_file as _read_geno_id_file
    from janusx.script._common.grmio import (
        load_and_align_grm,
        read_id_file as _read_grm_id_file,
    )

    roots = None
    if dependency_roots is not None:
        roots = tuple(Path(root).expanduser().resolve() for root in dependency_roots)

    fam_path = _postgwas_fvlmm_genotype_fam_path(record, genotype_prefix)
    if not fam_path.is_file():
        raise FineMapSkip(f"PLINK FAM file not found: {fam_path}")
    try:
        fam_ids_raw = _read_geno_id_file(
            str(fam_path),
            log,
            "PLINK FAM",
            show_status=False,
        )
    except (OSError, ValueError) as exc:
        raise FineMapSkip(f"PLINK FAM IDs could not be read: {exc}") from exc
    if fam_ids_raw is None:
        raise FineMapSkip("PLINK FAM IDs are missing")
    fam_ids = np.asarray(fam_ids_raw, dtype=str).reshape(-1)
    if fam_ids.size == 0:
        raise FineMapSkip("PLINK FAM IDs are empty")
    fam_id_list = [str(value) for value in fam_ids.tolist()]
    if len(set(fam_id_list)) != len(fam_id_list):
        raise FineMapSkip("PLINK FAM contains duplicate sample IDs")

    phenotype_path = _postgwas_resolve_fvlmm_dependency(
        record.phenotype_file,
        "phenotype",
        roots,
    )
    _postgwas_require_nonempty_source(phenotype_path, "phenotype")
    try:
        phenotype = load_phenotype(
            str(phenotype_path),
            None,
            log,
            id_col=0,
            use_spinner=False,
        )
    except _PostGWASExpectedCompatibilityError as exc:
        _postgwas_skip_for_expected_source_error(exc, "phenotype")
        raise
    except (OSError, EOFError, UnicodeError, pd.errors.ParserError) as exc:
        _postgwas_skip_for_expected_source_error(exc, "phenotype")
        raise
    except ValueError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "phenotype",
            value_markers=(
                "failed to read phenotype",
                "phenotype file is empty",
                "no phenotype data",
                "requested phenotype selector",
                "no phenotype selector",
            ),
        )
        raise

    qmatrix_ids: Optional[np.ndarray] = None
    qmatrix_values: Optional[np.ndarray] = None
    if record.qmatrix_file is not None:
        qmatrix_path = _postgwas_resolve_fvlmm_dependency(
            record.qmatrix_file,
            "Q/PC matrix",
            roots,
        )
        qmatrix_ids, qmatrix_values = _postgwas_load_qmatrix_source(
            qmatrix_path,
            record.qmatrix_columns,
        )

    covariate_ids: Optional[np.ndarray] = None
    covariate_values: Optional[np.ndarray] = None
    if record.covariate_file is not None:
        covariate_path = _postgwas_resolve_fvlmm_dependency(
            record.covariate_file,
            "covariate",
            roots,
        )
        _postgwas_require_nonempty_source(covariate_path, "covariate")
        try:
            covariate_ids_raw, covariate_values_raw = _read_cov_file_flexible(
                str(covariate_path),
                fam_ids,
                log,
                label="Covariate",
            )
        except _PostGWASExpectedCompatibilityError as exc:
            _postgwas_skip_for_expected_source_error(exc, "covariate")
            raise
        except (OSError, EOFError, UnicodeError, pd.errors.ParserError) as exc:
            _postgwas_skip_for_expected_source_error(exc, "covariate")
            raise
        except ValueError as exc:
            _postgwas_skip_for_expected_source_error(
                exc,
                "covariate",
                value_markers=(
                    "file is empty",
                    "at least 2 columns",
                    "must include sample ids",
                    "numeric-only matrix",
                ),
            )
            raise
        covariate_ids = np.asarray(covariate_ids_raw, dtype=str).reshape(-1)
        covariate_values = np.asarray(covariate_values_raw, dtype=np.float32)
        if covariate_values.ndim == 1:
            covariate_values = covariate_values.reshape(-1, 1)
        if covariate_values.ndim != 2 or covariate_values.shape[0] != covariate_ids.size:
            raise FineMapSkip(
                "covariate ID and value shapes disagree"
            )
        if len(set(str(value) for value in covariate_ids.tolist())) != covariate_ids.size:
            raise FineMapSkip("covariate file contains duplicate sample IDs")
        external_covariate_columns = tuple(
            record.covariate_columns[len(record.qmatrix_columns) :]
        )
        if covariate_values.shape[1] != len(external_covariate_columns):
            raise FineMapSkip(
                "covariate column count disagrees with the FvLMM sidecar metadata"
            )
    elif len(record.covariate_columns) > len(record.qmatrix_columns):
        raise FineMapSkip(
            "FvLMM sidecar declares covariates but has no covariate file"
        )

    grm_id_path = _postgwas_resolve_fvlmm_dependency(
        record.kinship_id_file,
        "GRM ID",
        roots,
    )
    try:
        grm_ids_raw = _read_grm_id_file(str(grm_id_path))
    except (OSError, ValueError) as exc:
        raise FineMapSkip(f"GRM ID file could not be read: {exc}") from exc
    grm_ids = np.asarray(grm_ids_raw, dtype=str).reshape(-1)
    if grm_ids.size == 0:
        raise FineMapSkip("GRM ID file is empty")
    grm_id_list = [str(value) for value in grm_ids.tolist()]
    if len(set(grm_id_list)) != len(grm_id_list):
        raise FineMapSkip("GRM ID file contains duplicate sample IDs")
    expected_grm_n = int(record.kinship_shape[0])
    if grm_ids.size != expected_grm_n:
        raise FineMapSkip(
            "GRM ID count mismatch: "
            f"sidecar shape={record.kinship_shape}, IDs={grm_ids.size}"
        )

    # This mirrors prepare_streaming_context: intersect sources in FAM order,
    # then apply the exact single-trait ~isnan phenotype mask.
    source_sets = [set(fam_id_list), set(str(value) for value in phenotype.index)]
    source_sets.append(set(grm_id_list))
    if qmatrix_ids is not None:
        source_sets.append(set(str(value) for value in qmatrix_ids.tolist()))
    if covariate_ids is not None:
        source_sets.append(set(str(value) for value in covariate_ids.tolist()))
    common = set.intersection(*source_sets)
    common_ids = [sample_id for sample_id in fam_id_list if sample_id in common]
    if len(common_ids) == 0:
        # The GWAS loader retries with PLINK IID (phenotype column 2) only when
        # the first phenotype ID column has no overlap.
        try:
            phenotype_alt = load_phenotype(
                str(phenotype_path),
                None,
                log,
                id_col=1,
                use_spinner=False,
            )
        except _PostGWASExpectedCompatibilityError as exc:
            _postgwas_skip_for_expected_source_error(exc, "alternate phenotype")
            phenotype_alt = None
        except (OSError, EOFError, UnicodeError, pd.errors.ParserError, IndexError):
            phenotype_alt = None
        except ValueError as exc:
            _postgwas_skip_for_expected_source_error(
                exc,
                "alternate phenotype",
                value_markers=(
                    "failed to read phenotype",
                    "phenotype file is empty",
                    "no phenotype data",
                    "requested phenotype selector",
                    "no phenotype selector",
                ),
            )
            phenotype_alt = None
        if phenotype_alt is not None:
            alt_sets = [set(fam_id_list), set(str(value) for value in phenotype_alt.index)]
            alt_sets.append(set(grm_id_list))
            if qmatrix_ids is not None:
                alt_sets.append(set(str(value) for value in qmatrix_ids.tolist()))
            if covariate_ids is not None:
                alt_sets.append(set(str(value) for value in covariate_ids.tolist()))
            alt_common = set.intersection(*alt_sets)
            alt_ids = [sample_id for sample_id in fam_id_list if sample_id in alt_common]
            if len(alt_ids) > 0:
                phenotype = phenotype_alt
                common = alt_common
                common_ids = alt_ids
    if len(common_ids) == 0:
        raise FineMapSkip(
            "no overlapping samples across FAM, phenotype, GRM, and covariates"
        )

    aligned_pheno, ordered_common = _align_pheno_to_sample_order(
        phenotype,
        np.asarray(common_ids, dtype=str),
    )
    trait_selector: object = record.phenotype_trait_column
    raw_trait_index = (
        record.phenotype_trait_source_index
        if record.phenotype_trait_source_index is not None
        else record.phenotype_trait_index
    )
    if raw_trait_index is not None:
        trait_selector = _TraitRef(
            col_idx=int(raw_trait_index),
            label=str(record.phenotype_trait_column),
        )
    try:
        _trait_values, phenotype_keep = _trait_values_and_mask(
            aligned_pheno,
            trait_selector,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise FineMapSkip(
            "phenotype trait metadata disagrees with the FvLMM sidecar"
        ) from exc
    keep_idx = np.flatnonzero(np.asarray(phenotype_keep, dtype=bool)).astype(
        np.int64,
        copy=False,
    )
    sample_ids = np.ascontiguousarray(ordered_common[keep_idx], dtype=str)
    if sample_ids.size == 0:
        raise FineMapSkip("phenotype missingness leaves no FvLMM samples")
    if sample_ids.size != int(record.sample_count):
        raise FineMapSkip(
            "sample order/count disagrees with the FvLMM sidecar: "
            f"reconstructed={sample_ids.size}, sidecar={record.sample_count}"
        )
    actual_hash = hash_ordered_sample_ids([str(value) for value in sample_ids.tolist()])
    if actual_hash.lower() != str(record.sample_order_sha256).lower():
        raise FineMapSkip(
            "sample order hash mismatch between reconstructed GWAS samples and sidecar"
        )

    fam_index = {sample_id: index for index, sample_id in enumerate(fam_id_list)}
    sample_indices = np.ascontiguousarray(
        np.asarray([fam_index[str(value)] for value in sample_ids.tolist()], dtype=np.int64),
        dtype=np.int64,
    )

    fixed_effect_columns_data: list[np.ndarray] = [
        np.ones((sample_ids.size,), dtype=np.float64)
    ]
    if qmatrix_ids is not None and qmatrix_values is not None:
        qmatrix_index = {
            str(sample_id): index
            for index, sample_id in enumerate(qmatrix_ids.tolist())
        }
        try:
            q_take = np.asarray(
                [qmatrix_index[str(value)] for value in sample_ids.tolist()],
                dtype=np.int64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FineMapSkip(
                "Q/PC matrix IDs do not cover reconstructed FvLMM samples"
            ) from exc
        q_subset = np.ascontiguousarray(
            qmatrix_values[q_take],
            dtype=np.float64,
        )
        if q_subset.ndim != 2 or q_subset.shape[1] != len(record.qmatrix_columns):
            raise FineMapSkip("reconstructed Q/PC matrix shape is invalid")
        fixed_effect_columns_data.extend(
            q_subset[:, index] for index in range(q_subset.shape[1])
        )
    if covariate_values is not None:
        covariate_index = {
            str(sample_id): index
            for index, sample_id in enumerate(covariate_ids.tolist())
        }
        cov_take = np.asarray(
            [covariate_index[str(value)] for value in sample_ids.tolist()],
            dtype=np.int64,
        )
        cov_subset = np.ascontiguousarray(
            covariate_values[cov_take],
            dtype=np.float64,
        )
        fixed_effect_columns_data.extend(
            cov_subset[:, index] for index in range(cov_subset.shape[1])
        )
    fixed_effects = np.ascontiguousarray(
        np.column_stack(fixed_effect_columns_data), dtype=np.float64
    )
    if fixed_effects.shape != (sample_ids.size, len(record.fixed_effect_columns)):
        raise FineMapSkip(
            "reconstructed fixed-effect shape disagrees with the FvLMM sidecar"
        )
    if not np.all(np.isfinite(fixed_effects)):
        raise FineMapSkip("reconstructed fixed effects contain non-finite values")

    kinship_path = _postgwas_resolve_fvlmm_dependency(
        record.kinship_file,
        "GRM",
        roots,
    )
    try:
        kinship_raw, _resolved_grm_id_path = load_and_align_grm(
            str(kinship_path),
            [str(value) for value in sample_ids.tolist()],
            grm_id_path=str(grm_id_path),
            label="GRM",
        )
    except FineMapSkip:
        raise
    except _PostGWASExpectedCompatibilityError as exc:
        _postgwas_skip_for_expected_source_error(exc, "GRM")
        raise
    except (OSError, EOFError, UnicodeError, pd.errors.ParserError) as exc:
        _postgwas_skip_for_expected_source_error(exc, "GRM")
        raise
    except ValueError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "GRM",
            value_markers=(
                "shape",
                "square",
                "id",
                "missing target",
                "nan/inf",
                "matrix",
            ),
        )
        raise
    kinship = np.ascontiguousarray(np.asarray(kinship_raw, dtype=np.float64), dtype=np.float64)
    if kinship.shape != (sample_ids.size, sample_ids.size):
        raise FineMapSkip(
            "aligned GRM shape disagrees with reconstructed FvLMM samples: "
            f"{kinship.shape} vs {(sample_ids.size, sample_ids.size)}"
        )
    if not np.all(np.isfinite(kinship)):
        raise FineMapSkip("aligned GRM contains non-finite values")

    return FvLMMFineMapContext(
        sample_ids=sample_ids,
        sample_indices_in_bfile=sample_indices,
        fixed_effects=fixed_effects,
        kinship=kinship,
        lambda_null=float(record.lambda_null),
        sidecar=record,
    )


_POSTGWAS_MEMORY_INT_MAX = sys.maxsize


def _postgwas_memory_count(value: object) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return _POSTGWAS_MEMORY_INT_MAX
    return min(_POSTGWAS_MEMORY_INT_MAX, max(0, integer))


def _postgwas_memory_product(*values: int) -> int:
    result = 1
    for raw_value in values:
        value = _postgwas_memory_count(raw_value)
        if value == 0:
            return 0
        if result > _POSTGWAS_MEMORY_INT_MAX // value:
            return _POSTGWAS_MEMORY_INT_MAX
        result *= value
    return result


def _postgwas_memory_sum(*values: int) -> int:
    result = 0
    for raw_value in values:
        value = _postgwas_memory_count(raw_value)
        if result > _POSTGWAS_MEMORY_INT_MAX - value:
            return _POSTGWAS_MEMORY_INT_MAX
        result += value
    return result


def _postgwas_fvlmm_memory_components(
    *,
    n_variants: int,
    n_samples: int,
    fixed_effect_columns: int | Sequence[object],
    bfile_samples: Optional[int] = None,
    grm_samples: Optional[int] = None,
    qmatrix_rows: Optional[int] = None,
    susie_l: int = 5,
    existing_bytes: int = 0,
    live_gwas_bytes: int = 0,
) -> dict[str, int]:
    """Return a conservative complete peak working-set accounting."""
    m = _postgwas_memory_count(n_variants)
    n = _postgwas_memory_count(n_samples)
    n_bfile = max(n, _postgwas_memory_count(bfile_samples or 0))
    n_grm = max(n, _postgwas_memory_count(grm_samples or 0))
    n_q_source = (
        _postgwas_memory_count(qmatrix_rows)
        if qmatrix_rows is not None
        else n_bfile
    )
    if isinstance(fixed_effect_columns, (int, np.integer)):
        q = _postgwas_memory_count(fixed_effect_columns)
    else:
        q = _postgwas_memory_count(len(fixed_effect_columns))
    l = max(1, min(max(1, _postgwas_memory_count(susie_l)), max(1, m)))
    f64 = 8
    f32 = 4
    mn = _postgwas_memory_product(m, n)
    nn = _postgwas_memory_product(n, n)
    mm = _postgwas_memory_product(m, m)
    qq = _postgwas_memory_product(n, q)
    mq = _postgwas_memory_product(m, q)
    components = {
        "existing_gwas_frame": _postgwas_memory_count(existing_bytes),
        "live_gwas_frame": _postgwas_memory_count(live_gwas_bytes),
        "grm_source_full": _postgwas_memory_product(n_grm, n_grm, f64),
        "grm_subset": _postgwas_memory_product(nn, f64),
        "eigh_input_copy": _postgwas_memory_product(nn, f64),
        "eigensystem": _postgwas_memory_sum(
            _postgwas_memory_product(nn, f64),
            _postgwas_memory_product(n, f64),
        ),
        # The Python list retains all decoded chunks while vstack creates a
        # second full f32 matrix.  Selection then creates a live f64 matrix.
        "regional_genotype_chunks_f32": _postgwas_memory_product(mn, f32),
        "regional_genotype_vstack_f32": _postgwas_memory_product(mn, f32),
        "regional_genotype_f32": _postgwas_memory_product(mn, f32),
        "regional_genotype_f64": _postgwas_memory_product(mn, f64),
        "selected_genotype_f64": _postgwas_memory_product(mn, f64),
        "regional_bed_packed": _postgwas_memory_product(m, (n_bfile + 3) // 4),
        "rotated_genotype": _postgwas_memory_product(mn, f64),
        "residual_genotype": _postgwas_memory_product(mn, f64),
        "fixed_effects_f64": _postgwas_memory_product(qq, f64),
        "fixed_effect_source_f32": _postgwas_memory_product(n_q_source, q, f32),
        "qmatrix_source_f64": _postgwas_memory_product(n_q_source, q, f64),
        "whitened_fixed_effects": _postgwas_memory_product(qq, f64),
        "projected_coefficients": _postgwas_memory_product(mq, f64),
        "projected_gram_ld": _postgwas_memory_product(mm, f64),
        # Rust's PyO3 wrapper first materializes row-major Vec values and then
        # copies each Vec into a nalgebra DMatrix.  u_t is already a Python
        # transpose, so count that matrix plus both Rust copies.
        "u_t_transpose_f64": _postgwas_memory_product(nn, f64),
        "pyo3_vec_genotype": _postgwas_memory_product(mn, f64),
        "pyo3_dmatrix_genotype": _postgwas_memory_product(mn, f64),
        "pyo3_vec_eigvals": _postgwas_memory_product(n, f64),
        "pyo3_dmatrix_eigvals": _postgwas_memory_product(n, f64),
        "pyo3_vec_u_t": _postgwas_memory_product(nn, f64),
        "pyo3_dmatrix_u_t": _postgwas_memory_product(nn, f64),
        "pyo3_vec_fixed_effects": _postgwas_memory_product(qq, f64),
        "pyo3_dmatrix_fixed_effects": _postgwas_memory_product(qq, f64),
        "pyo3_genotype_copy": _postgwas_memory_product(mn, f64),
        "pyo3_eigvals_copy": _postgwas_memory_product(n, f64),
        "pyo3_u_t_copy": _postgwas_memory_product(nn, f64),
        "pyo3_fixed_effect_copy": _postgwas_memory_product(qq, f64),
        "pyo3_ld_output_copy": _postgwas_memory_product(mm, f64),
        "diagnostic_symmetrized_ld": _postgwas_memory_product(mm, f64),
        "diagnostic_eigenvalues": _postgwas_memory_product(m, f64),
        "susie_ld_copy": _postgwas_memory_product(mm, f64),
        "susie_vectors": _postgwas_memory_sum(
            _postgwas_memory_product(l + 4, m, f64),
            _postgwas_memory_product(l, f64),
        ),
        "reserve": _POSTGWAS_FINEMAP_MEMORY_RESERVE_BYTES,
    }
    return components


def _postgwas_estimate_fvlmm_memory_bytes(
    n_variants: int,
    n_samples: int,
    fixed_effect_columns: int | Sequence[object] = 1,
    *,
    bfile_samples: Optional[int] = None,
    grm_samples: Optional[int] = None,
    qmatrix_rows: Optional[int] = None,
    susie_l: int = 5,
    existing_bytes: int = 0,
    live_gwas_bytes: int = 0,
) -> int:
    """Estimate the complete FvLMM effective-LD/SuSiE peak in bytes."""
    return _postgwas_memory_sum(
        *_postgwas_fvlmm_memory_components(
            n_variants=n_variants,
            n_samples=n_samples,
            fixed_effect_columns=fixed_effect_columns,
            bfile_samples=bfile_samples,
            grm_samples=grm_samples,
            qmatrix_rows=qmatrix_rows,
            susie_l=susie_l,
            existing_bytes=existing_bytes,
            live_gwas_bytes=live_gwas_bytes,
        ).values()
    )


def _postgwas_log_fvlmm_stage(
    logger: logging.Logger,
    stage: str,
    started: float,
    *,
    requested_threads: int,
    using_threads: int,
    backend: Optional[str] = None,
) -> None:
    elapsed = max(0.0, time.perf_counter() - started)
    logger.info(
        "FvLMM fine-map stage: stage=%s seconds=%.6f requested_threads=%d "
        "using_threads=%d%s",
        stage,
        elapsed,
        requested_threads,
        using_threads,
        "" if backend is None else f" backend={backend}",
    )


def _postgwas_build_fvlmm_effective_ld(
    *,
    args: argparse.Namespace,
    record: GwasNullModelSidecarV1,
    prepared: pd.DataFrame,
    bed_indices: Optional[np.ndarray],
    logger: logging.Logger,
    live_gwas_bytes: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build exact FvLMM effective LD and retain kernel valid-row indices."""
    requested_threads = max(1, int(getattr(args, "thread", 1)))
    _postgwas_validate_fvlmm_sidecar(record)
    if not isinstance(prepared, pd.DataFrame):
        raise FineMapSkip("FvLMM fine-mapping input rows are not a DataFrame")
    if prepared.empty:
        raise FineMapSkip("FvLMM fine-mapping has no regional GWAS rows")
    required_columns = {"chrom", "pos"}
    missing_columns = sorted(required_columns.difference(prepared.columns))
    if missing_columns:
        raise FineMapSkip(
            "FvLMM fine-mapping rows are missing required columns: "
            + ", ".join(missing_columns)
        )

    requested_prefix = getattr(args, "bfile", None)
    if requested_prefix is None or str(requested_prefix).strip() == "":
        requested_prefix = record.genotype_prefix
    try:
        bfile_samples = _postgwas_count_plink_samples(requested_prefix)
        if int(bfile_samples) <= 0:
            raise ValueError("PLINK FAM contains no samples")
    except (OSError, ValueError, OverflowError) as exc:
        raise FineMapSkip(f"FvLMM memory preflight could not inspect FAM: {exc}") from exc

    dependency_roots = _dependency_search_roots(
        record.result.canonical_path,
        requested_prefix,
    )
    grm_path = _postgwas_resolve_fvlmm_dependency(
        record.kinship_file,
        "GRM",
        dependency_roots,
    )
    actual_grm_shape = _postgwas_inspect_grm_storage_header(grm_path)
    expected_grm_shape = tuple(int(value) for value in record.kinship_shape)
    if actual_grm_shape != expected_grm_shape:
        raise FineMapSkip(
            "GRM storage header shape disagrees with sidecar metadata: "
            f"actual={actual_grm_shape}, sidecar={expected_grm_shape}"
        )
    grm_id_path = _postgwas_resolve_fvlmm_dependency(
        record.kinship_id_file,
        "GRM ID",
        dependency_roots,
    )
    actual_grm_id_count = _postgwas_inspect_id_storage(grm_id_path, "GRM ID")
    if actual_grm_id_count != actual_grm_shape[0]:
        raise FineMapSkip(
            "GRM ID storage count disagrees with the actual GRM header: "
            f"ids={actual_grm_id_count}, matrix={actual_grm_shape[0]}"
        )
    if int(record.sample_count) > int(bfile_samples) or int(record.sample_count) > actual_grm_shape[0]:
        raise FineMapSkip(
            "FvLMM sidecar sample count exceeds verified FAM/GRM dimensions"
        )
    qmatrix_rows: Optional[int] = None
    if record.qmatrix_file is not None:
        qmatrix_path = _postgwas_resolve_fvlmm_dependency(
            record.qmatrix_file,
            "Q/PC matrix",
            dependency_roots,
        )
        qmatrix_rows, _qmatrix_columns = _postgwas_inspect_qmatrix_storage_header(
            qmatrix_path,
            record.qmatrix_columns,
        )
        if qmatrix_rows != int(bfile_samples):
            raise FineMapSkip(
                "Q/PC storage header row count disagrees with verified FAM IDs: "
                f"q_rows={qmatrix_rows}, fam_rows={bfile_samples}"
            )
    existing_bytes = int(
        prepared.memory_usage(deep=True, index=True).sum()
    )
    try:
        susie_l = int(getattr(args, "finemap_l", 5))
        estimated_bytes = _postgwas_estimate_fvlmm_memory_bytes(
            n_variants=len(prepared),
            n_samples=int(record.sample_count),
            fixed_effect_columns=record.fixed_effect_columns,
            bfile_samples=bfile_samples,
            grm_samples=actual_grm_shape[0],
            qmatrix_rows=qmatrix_rows,
            susie_l=susie_l,
            existing_bytes=existing_bytes,
            live_gwas_bytes=live_gwas_bytes,
        )
        memory_limit_bytes = min(
            _POSTGWAS_MEMORY_INT_MAX,
            max(
                0,
                int(
                    getattr(
                        args,
                        "finemap_memory_bytes",
                        _POSTGWAS_FINEMAP_MAX_MEMORY_BYTES,
                    )
                ),
            ),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise FineMapSkip(f"FvLMM memory preflight inputs are invalid: {exc}") from exc
    if estimated_bytes > max(0, memory_limit_bytes):
        raise FineMapSkip(
            "FvLMM memory preflight rejected the locus before loading the full "
            "GRM/genotype: "
            f"estimated_peak={estimated_bytes / float(1024**3):.2f} GiB, "
            f"limit={max(0, memory_limit_bytes) / float(1024**3):.2f} GiB"
        )
    logger.info(
        "FvLMM memory preflight: variants=%d samples=%d fixed_effect_columns=%d "
        "locus_frame=%.2f GiB live_gwas=%.2f GiB estimated_peak=%.2f GiB "
        "limit=%.2f GiB.",
        len(prepared),
        int(record.sample_count),
        len(record.fixed_effect_columns),
        existing_bytes / float(1024**3),
        max(0, int(live_gwas_bytes)) / float(1024**3),
        estimated_bytes / float(1024**3),
        max(0, memory_limit_bytes) / float(1024**3),
    )
    prepared_source = prepared.reset_index(drop=True)

    context_started = time.perf_counter()
    context = _postgwas_reconstruct_fvlmm_context(
        record,
        genotype_prefix=requested_prefix,
        dependency_roots=dependency_roots,
        logger=logger,
    )
    _postgwas_log_fvlmm_stage(
        logger,
        "context",
        context_started,
        requested_threads=requested_threads,
        using_threads=1,
    )

    bim_scan_started = time.perf_counter()
    selected_bim_indices, selected_bim_metadata = _postgwas_resolve_prepared_bim_rows(
        requested_prefix,
        prepared_source,
    )
    _postgwas_log_fvlmm_stage(
        logger,
        "bim_scan",
        bim_scan_started,
        requested_threads=requested_threads,
        using_threads=1,
    )
    genotype_decode_started = time.perf_counter()
    selected_signatures = [
        (
            metadata[0],
            metadata[1],
            _postgwas_variant_allele_pair(metadata[3], metadata[4]),
        )
        for metadata in selected_bim_metadata
    ]
    signature_to_bim: dict[
        tuple[str, int, frozenset[str] | None], int
    ] = {}
    for index, signature in zip(selected_bim_indices, selected_signatures):
        if signature in signature_to_bim:
            raise FineMapSkip(
                "selected BIM rows have indistinguishable coordinate/allele metadata"
            )
        signature_to_bim[signature] = index

    # The effective-LD kernel consumes genotype rows in selected-BIM order.
    # Align the summary rows to that same order before filtering projected
    # diagonals, retaining one explicit map for every later row subset.
    alignment_input = prepared_source.copy()
    for column in ("beta", "se", "z"):
        if column not in alignment_input.columns:
            alignment_input[column] = 0.0
    selected_bim_meta = pd.DataFrame(
        {
            "chrom": [metadata[0] for metadata in selected_bim_metadata],
            "pos": [metadata[1] for metadata in selected_bim_metadata],
            "snp": [metadata[2] for metadata in selected_bim_metadata],
            "allele0": [metadata[3] for metadata in selected_bim_metadata],
            "allele1": [metadata[4] for metadata in selected_bim_metadata],
        }
    )
    selected_site_counts: dict[tuple[str, int], int] = {}
    for chrom, pos in zip(selected_bim_meta["chrom"], selected_bim_meta["pos"]):
        key = (_postgwas_finemap_normalize_chr(chrom), int(pos))
        selected_site_counts[key] = selected_site_counts.get(key, 0) + 1
    selected_ambiguous_sites = {
        key for key, count in selected_site_counts.items() if count > 1
    }
    aligned_input, effective_bed_indices, alignment_counts = _postgwas_align_finemap_locus(
        alignment_input,
        selected_bim_meta,
        ambiguous_bim_sites=selected_ambiguous_sites,
    )
    if aligned_input.empty:
        raise FineMapSkip(
            "FvLMM fine-mapping has no variants after BED identity and allele alignment"
        )
    effective_bed_indices = np.asarray(effective_bed_indices, dtype=np.int64)
    if effective_bed_indices.size != len(aligned_input):
        raise FineMapSkip(
            "FvLMM summary/BED alignment map has inconsistent dimensions"
        )

    filters = dict(record.genotype_filters)
    try:
        maf = float(filters.get("maf", 0.0))
        missing_rate = float(
            filters.get("max_missing_rate", filters.get("missing_rate", 1.0))
        )
        het = float(filters.get("het_threshold", filters.get("het", 1.0)))
    except (TypeError, ValueError) as exc:
        raise FineMapSkip("FvLMM sidecar genotype filters are invalid") from exc
    if not all(np.isfinite(value) for value in (maf, missing_rate, het)):
        raise FineMapSkip("FvLMM sidecar genotype filters must be finite")
    if maf < 0.0 or missing_rate < 0.0 or het < 0.0:
        raise FineMapSkip("FvLMM sidecar genotype filters must be non-negative")
    model = str(filters.get("genetic_model", "add"))
    snps_only = bool(filters.get("snps_only", True))

    genotype_chunks: list[np.ndarray] = []
    returned_sites: list[object] = []
    try:
        loader_signature = inspect.signature(load_genotype_chunks)
    except (TypeError, ValueError) as exc:
        raise FineMapSkip(
            "regional genotype loader signature is unavailable; exact SNP selection cannot be proven"
        ) from exc
    loader_parameters = loader_signature.parameters
    if (
        "snp_indices" not in loader_parameters
        and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in loader_parameters.values()
        )
    ):
        raise FineMapSkip(
            "regional genotype loader does not support exact SNP selection"
        )
    try:
        genotype_iter = load_genotype_chunks(
            str(requested_prefix),
            chunk_size=max(1, min(20_000, len(selected_bim_indices))),
            maf=maf,
            missing_rate=missing_rate,
            impute=True,
            model=model,
            het=het,
            snps_only=snps_only,
            snp_indices=[int(index) for index in selected_bim_indices],
            sample_ids=[str(value) for value in context.sample_ids.tolist()],
        )
        for genotype_chunk, sites in genotype_iter:
            block = np.asarray(genotype_chunk, dtype=np.float32)
            if block.ndim != 2 or block.shape[0] != len(sites):
                raise ValueError(
                    "regional genotype chunk and site metadata have inconsistent shapes"
                )
            genotype_chunks.append(np.ascontiguousarray(block, dtype=np.float32))
            returned_sites.extend(list(sites))
    except _PostGWASExpectedCompatibilityError as exc:
        _postgwas_skip_for_expected_source_error(exc, "regional genotype")
        raise
    except (OSError, EOFError, UnicodeError, pd.errors.ParserError) as exc:
        _postgwas_skip_for_expected_source_error(exc, "regional genotype")
        raise
    except RuntimeError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "regional genotype",
            runtime_markers=(
                "snp selection",
                "snp_indices",
                "does not support exact",
                "selection unavailable",
            ),
        )
        raise
    except TypeError as exc:
        if "snp_indices" in str(exc).lower() and "keyword" in str(exc).lower():
            raise FineMapSkip(
                f"regional genotype loader does not support exact SNP selection: {exc}"
            ) from exc
        raise
    except ValueError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "regional genotype",
            value_markers=(
                "snp_indices is empty",
                "invalid snp_",
                "genotype source",
                "genotype chunk",
                "site metadata",
                "malformed",
            ),
        )
        raise
    if len(genotype_chunks) == 0:
        raise FineMapSkip("regional genotype loader returned no variants")
    if len(returned_sites) != len(selected_bim_indices):
        raise FineMapSkip(
            "regional genotype filters changed the requested variant set; exact "
            "GWAS variant alignment cannot be proven"
        )

    genotype_all = np.vstack(genotype_chunks).astype(np.float32, copy=False)
    row_by_bim_index: dict[int, int] = {}
    for row_index, site in enumerate(returned_sites):
        signature = _postgwas_site_identity_signature(site)
        bim_index = signature_to_bim.get(signature)
        if bim_index is None:
            # A loader without allele metadata cannot distinguish duplicate
            # coordinates.  Coordinate-only fallback is safe only when the
            # selected BIM set has exactly one row at that coordinate.
            coordinate_matches = [
                index
                for index, metadata in zip(selected_bim_indices, selected_bim_metadata)
                if metadata[0] == signature[0] and metadata[1] == signature[1]
            ]
            if signature[2] is None and len(coordinate_matches) == 1:
                bim_index = coordinate_matches[0]
            else:
                raise FineMapSkip(
                    "genotype site metadata cannot prove exact variant identity"
                )
        if bim_index in row_by_bim_index:
            raise FineMapSkip("genotype loader returned duplicate variant identity")
        row_by_bim_index[bim_index] = row_index
    if set(row_by_bim_index) != set(selected_bim_indices):
        raise FineMapSkip(
            "genotype site metadata does not cover the requested BIM variants"
        )
    selected_rows = [row_by_bim_index[index] for index in selected_bim_indices]
    genotypes = np.ascontiguousarray(
        np.asarray(genotype_all[np.asarray(selected_rows, dtype=np.int64)], dtype=np.float64),
        dtype=np.float64,
    )
    if (
        np.any(effective_bed_indices < 0)
        or np.any(effective_bed_indices >= genotypes.shape[0])
        or len(np.unique(effective_bed_indices)) != len(effective_bed_indices)
    ):
        raise FineMapSkip("FvLMM summary/BED alignment map contains invalid indices")
    genotypes = np.ascontiguousarray(genotypes[effective_bed_indices], dtype=np.float64)
    _postgwas_log_fvlmm_stage(
        logger,
        "genotype_decode",
        genotype_decode_started,
        requested_threads=requested_threads,
        using_threads=1,
    )

    from janusx.assoc import workflow as workflow_module
    if not hasattr(workflow_module.jxrs, "rust_eigh_from_array_f64"):
        raise FineMapSkip(
            "FvLMM Rust EVD symbol rust_eigh_from_array_f64 is unavailable"
        )
    grm_evd_started = time.perf_counter()
    try:
        _gwas_eigh_from_grm = workflow_module._gwas_eigh_from_grm
        eigvals, eigvecs, _eigh_backend, _eigh_elapsed = _gwas_eigh_from_grm(
            context.kinship,
            threads=requested_threads,
            logger=logger,
            stage_label="PostGWAS FvLMM",
            require_rust=True,
            diag_ridge=_POSTGWAS_FVLMM_DIAG_RIDGE,
        )
    except _PostGWASExpectedCompatibilityError as exc:
        _postgwas_skip_for_expected_source_error(exc, "FvLMM GRM eigendecomposition")
        raise
    except RuntimeError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "FvLMM GRM eigendecomposition",
            runtime_markers=(
                "rust_eigh_from_array_f64",
                "rust evd symbol",
                "missing rust evd",
                "symbol is unavailable",
            ),
        )
        raise
    except ValueError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "FvLMM GRM eigendecomposition",
            value_markers=(
                "non-empty square grm",
                "grm shape",
                "grm matrix",
            ),
        )
        raise
    eigvals = np.ascontiguousarray(np.asarray(eigvals, dtype=np.float64).reshape(-1))
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    if (
        eigvecs.ndim != 2
        or eigvecs.shape != (context.sample_ids.size, context.sample_ids.size)
        or eigvals.shape != (context.sample_ids.size,)
        or not np.all(np.isfinite(eigvals))
        or not np.all(np.isfinite(eigvecs))
    ):
        raise FineMapSkip("FvLMM GRM eigendecomposition returned invalid arrays")
    u_t = np.ascontiguousarray(eigvecs.T, dtype=np.float64)
    _postgwas_log_fvlmm_stage(
        logger,
        "grm_evd",
        grm_evd_started,
        requested_threads=requested_threads,
        using_threads=requested_threads,
        backend=str(_eigh_backend),
    )

    effective_ld_started = time.perf_counter()
    try:
        kernel_result = jxrs.fvlmm_effective_ld_spectral_f64(
            genotypes,
            eigvals,
            u_t,
            context.fixed_effects,
            context.lambda_null,
            threads=requested_threads,
        )
    except _PostGWASExpectedCompatibilityError as exc:
        _postgwas_skip_for_expected_source_error(exc, "FvLMM effective-LD kernel")
        raise
    except ValueError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "FvLMM effective-LD kernel",
            value_markers=("shape", "finite", "rank", "lambda_null"),
        )
        raise
    if not isinstance(kernel_result, dict):
        raise FineMapSkip("FvLMM effective-LD kernel returned a non-mapping result")
    try:
        locus_r = np.ascontiguousarray(
            np.asarray(kernel_result["r"], dtype=np.float64),
            dtype=np.float64,
        )
        raw_valid_indices = kernel_result["valid_indices"]
    except (KeyError, TypeError, ValueError) as exc:
        raise FineMapSkip("FvLMM effective-LD kernel returned invalid arrays") from exc
    valid_indices = _postgwas_validate_kernel_valid_indices(
        raw_valid_indices,
        int(genotypes.shape[0]),
    )
    try:
        diagnostic_fixed_columns = kernel_result["fixed_effect_columns"]
        diagnostic_rank = kernel_result["fixed_effect_rank"]
        diagnostic_rank_tolerance = float(kernel_result["rank_tolerance"])
        diagnostic_min_projected_diag = float(
            kernel_result["min_projected_diag"]
        )
        diagnostic_backend = str(kernel_result["backend"])
        diagnostic_requested_threads = kernel_result["requested_threads"]
        diagnostic_using_threads = kernel_result["using_threads"]
        diagnostic_timings = [
            float(kernel_result[key])
            for key in (
                "rotation_seconds",
                "svd_seconds",
                "residual_seconds",
                "gram_seconds",
                "total_seconds",
            )
        ]
        if (
            isinstance(diagnostic_fixed_columns, (bool, np.bool_))
            or not isinstance(diagnostic_fixed_columns, (int, np.integer))
            or int(diagnostic_fixed_columns) != len(record.fixed_effect_columns)
            or isinstance(diagnostic_rank, (bool, np.bool_))
            or not isinstance(diagnostic_rank, (int, np.integer))
            or int(diagnostic_rank) < 1
            or int(diagnostic_rank) > int(diagnostic_fixed_columns)
            or int(diagnostic_rank) >= int(context.sample_ids.size)
            or not np.isfinite(diagnostic_rank_tolerance)
            or diagnostic_rank_tolerance <= 0.0
            or not np.isfinite(diagnostic_min_projected_diag)
            or diagnostic_min_projected_diag <= 0.0
            or diagnostic_backend not in {"accelerate", "openblas", "blas", "rust"}
            or isinstance(diagnostic_requested_threads, (bool, np.bool_))
            or not isinstance(diagnostic_requested_threads, (int, np.integer))
            or int(diagnostic_requested_threads) != requested_threads
            or isinstance(diagnostic_using_threads, (bool, np.bool_))
            or not isinstance(diagnostic_using_threads, (int, np.integer))
            or int(diagnostic_using_threads) != requested_threads
            or not all(
                np.isfinite(value) and value >= 0.0
                for value in diagnostic_timings
            )
        ):
            raise ValueError("kernel diagnostics are non-finite or inconsistent")
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise FineMapSkip(
            f"FvLMM effective-LD kernel diagnostics are invalid: {exc}"
        ) from exc
    if (
        locus_r.ndim != 2
        or locus_r.shape[0] != locus_r.shape[1]
        or locus_r.shape[0] != valid_indices.size
        or valid_indices.size == 0
        or np.any(valid_indices < 0)
        or np.any(valid_indices >= genotypes.shape[0])
        or len(np.unique(valid_indices)) != valid_indices.size
        or not np.all(np.isfinite(locus_r))
        or not np.allclose(
            locus_r,
            locus_r.T,
            atol=_POSTGWAS_FVLMM_LD_SYMMETRY_ATOL,
            rtol=_POSTGWAS_FVLMM_LD_SYMMETRY_ATOL,
        )
        or not np.allclose(
            np.diag(locus_r),
            1.0,
            atol=_POSTGWAS_FVLMM_LD_UNIT_DIAGONAL_ATOL,
            rtol=_POSTGWAS_FVLMM_LD_UNIT_DIAGONAL_ATOL,
        )
    ):
        raise FineMapSkip("FvLMM effective-LD kernel returned inconsistent results")
    aligned = aligned_input.iloc[valid_indices].reset_index(drop=True)
    aligned.attrs["_janusx_finemap_alignment_counts"] = dict(alignment_counts)
    if aligned.shape[0] != locus_r.shape[0]:
        raise FineMapSkip("FvLMM effective-LD and GWAS rows are misaligned")

    try:
        symmetric_ld = 0.5 * (locus_r + locus_r.T)
        ld_eigenvalues = np.linalg.eigvalsh(symmetric_ld)
        min_ld_eigenvalue = float(np.min(ld_eigenvalues))
        max_ld_eigenvalue = float(np.max(ld_eigenvalues))
        psd_tolerance = max(
            _POSTGWAS_FVLMM_LD_PSD_ATOL,
            _POSTGWAS_FVLMM_LD_PSD_RTOL
            * max(1.0, float(np.max(np.abs(ld_eigenvalues)))),
        )
        if min_ld_eigenvalue < -psd_tolerance:
            raise ValueError(
                "effective LD is materially indefinite: "
                f"min_eigenvalue={min_ld_eigenvalue:.6g}, "
                f"tolerance={psd_tolerance:.6g}"
            )
        condition_number = (
            float("inf")
            if min_ld_eigenvalue <= psd_tolerance
            else max_ld_eigenvalue / min_ld_eigenvalue
        )
    except (np.linalg.LinAlgError, ValueError, TypeError) as exc:
        raise FineMapSkip(f"FvLMM LD diagnostics failed: {exc}") from exc
    _postgwas_log_fvlmm_stage(
        logger,
        "effective_ld",
        effective_ld_started,
        requested_threads=int(diagnostic_requested_threads),
        using_threads=int(diagnostic_using_threads),
        backend=diagnostic_backend,
    )
    logger.info(
        "FvLMM effective LD: samples=%d lambda=%.12g pve=%.12g "
        "fixed_effect_columns=%s fixed_effect_rank=%s rank_tolerance=%.6g "
        "min_projected_diag=%.6g min_ld_eigenvalue=%.6g condition_number=%.6g.",
        context.sample_ids.size,
        context.lambda_null,
        float(record.pve),
        tuple(record.fixed_effect_columns),
        int(diagnostic_rank),
        diagnostic_rank_tolerance,
        diagnostic_min_projected_diag,
        min_ld_eigenvalue,
        condition_number,
    )
    return locus_r, aligned


def _postgwas_estimate_finemap_memory_bytes(
    n_variants: int,
    n_samples: int,
    existing_bytes: int = 0,
) -> int:
    """Estimate the dense signed-LD peak before any BED decoding starts."""
    m = max(0, int(n_variants))
    n = max(0, int(n_samples))
    existing = max(0, int(existing_bytes))
    bytes_per_snp = (n + 3) // 4
    # Rust's signed-LD path retains one centered decoded f64 row buffer and
    # reuses one f64 BLAS Gram buffer as the returned correlation matrix.
    dense_bytes = (m * n * 8) + (m * m * 8)
    packed_bytes = m * bytes_per_snp
    return (
        existing
        + dense_bytes
        + packed_bytes
        + _POSTGWAS_FINEMAP_MEMORY_RESERVE_BYTES
    )


def _postgwas_check_finemap_memory(
    n_variants: int,
    n_samples: int,
    existing_bytes: int = 0,
    max_bytes: int = _POSTGWAS_FINEMAP_MAX_MEMORY_BYTES,
) -> int:
    """Reject a dense-LD locus that cannot fit within the configured budget."""
    estimated = _postgwas_estimate_finemap_memory_bytes(
        n_variants=n_variants,
        n_samples=n_samples,
        existing_bytes=existing_bytes,
    )
    limit = max(0, int(max_bytes))
    if estimated > limit:
        estimated_gib = estimated / float(1024**3)
        limit_gib = limit / float(1024**3)
        raise MemoryError(
            "Fine-mapping dense signed-LD was refused before BED decoding: "
            f"{int(n_variants)} variants x {int(n_samples)} samples would require "
            f"approximately {estimated_gib:.2f} GiB, exceeding the {limit_gib:.2f} "
            "GiB memory limit. Use a smaller -bimrange or split the locus."
        )
    return estimated


def _postgwas_count_plink_samples(bfile: object) -> int:
    """Count non-empty FAM rows without materializing genotype data."""
    prefix = _normalize_plink_prefix(bfile)
    fam_path = f"{prefix}.fam"
    try:
        with open(fam_path, "rt", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError as exc:
        raise ValueError(f"Unable to read PLINK sample file {fam_path}: {exc}") from exc


def _postgwas_finemap_result_model(
    result_path: str | os.PathLike[str],
) -> Optional[str]:
    """Classify maintained mixed-model result suffixes before LD allocation."""
    name = os.path.basename(os.fspath(result_path)).lower()
    for suffix, model in _POSTGWAS_MIXED_MODEL_RESULT_SUFFIXES:
        if name.endswith(suffix):
            return model
    return None


def _postgwas_select_finemap_ld_route(
    model: Optional[str],
    record: Optional[GwasNullModelSidecarV1] = None,
) -> str:
    """Select the LD implementation before any ordinary dense-LD allocation."""
    route_model = str(model or "").strip().lower()
    if record is not None:
        record_model = str(getattr(record, "model", "")).strip().lower()
        if record_model != "":
            route_model = record_model
    if route_model == "fvlmm":
        return "fvlmm"
    if route_model in {"lmm", "lmm2", "splmm", "splmm2"}:
        raise FineMapSkip(
            f"{route_model} mixed-model fine-mapping is unsupported: "
            "common-null score statistics are unavailable"
        )
    return "raw"


def _postgwas_read_finemap_fam_sample_ids(bfile: object) -> list[str]:
    """Read unique PLINK IID values in their exact FAM order."""
    prefix = _normalize_plink_prefix(bfile)
    fam_path = Path(f"{prefix}.fam")
    sample_ids: list[str] = []
    try:
        with fam_path.open("rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 2 or str(fields[1]).strip() == "":
                    raise ValueError(f"malformed FAM row at {fam_path}:{line_number}")
                sample_ids.append(str(fields[1]).strip())
    except (OSError, ValueError) as exc:
        raise FineMapSkip(f"PLINK FAM sample metadata is unavailable: {exc}") from exc
    if not sample_ids:
        raise FineMapSkip("PLINK FAM sample metadata is empty")
    if len(set(sample_ids)) != len(sample_ids):
        raise FineMapSkip("PLINK FAM sample metadata contains duplicate IIDs")
    return sample_ids


def _postgwas_resolve_finemap_sample_ids(
    args: argparse.Namespace,
    bfile: object,
) -> tuple[list[str], np.ndarray]:
    """Resolve and hash-check the exact sample order used by an ordinary GWAS.

    Ordinary/legacy LM results do not carry the sidecar's reconstructable
    phenotype context.  They are therefore safe only when the caller supplies
    an explicit ordered sample list or indices together with its ordered-ID
    hash.  In particular, FAM order is never silently treated as GWAS order.
    """
    fam_ids = _postgwas_read_finemap_fam_sample_ids(bfile)
    supplied_ids = getattr(args, "finemap_sample_ids", None)
    sample_id_file = getattr(args, "finemap_sample_id_file", None)
    supplied_indices = getattr(args, "finemap_sample_indices", None)
    if supplied_ids is None and sample_id_file is not None:
        try:
            supplied_ids = [
                line.split()[0]
                for line in Path(sample_id_file).read_text(encoding="utf-8").splitlines()
                if line.split()
            ]
        except (OSError, UnicodeError, ValueError) as exc:
            raise FineMapSkip(
                f"LM fine-mapping sample metadata could not be read: {exc}"
            ) from exc
    if supplied_ids is None and supplied_indices is None:
        raise FineMapSkip(
            "LM fine-mapping skipped: verified GWAS sample metadata is unavailable; "
            "legacy LM cannot safely use the full FAM cohort"
        )

    sample_ids: list[str]
    if supplied_ids is not None:
        if isinstance(supplied_ids, (str, bytes)):
            raise FineMapSkip("LM fine-mapping sample IDs must be an ordered sequence")
        try:
            sample_ids = [str(value).strip() for value in list(supplied_ids)]
        except (TypeError, ValueError) as exc:
            raise FineMapSkip("LM fine-mapping sample IDs are invalid") from exc
        if not sample_ids or any(value == "" for value in sample_ids):
            raise FineMapSkip("LM fine-mapping sample IDs are empty")
        if len(set(sample_ids)) != len(sample_ids):
            raise FineMapSkip("LM fine-mapping sample IDs contain duplicates")
    else:
        try:
            raw_indices = list(supplied_indices)
        except (TypeError, ValueError) as exc:
            raise FineMapSkip("LM fine-mapping sample indices are invalid") from exc
        indices: list[int] = []
        for raw_index in raw_indices:
            if isinstance(raw_index, (bool, np.bool_)):
                raise FineMapSkip("LM fine-mapping sample indices must be integers")
            try:
                index = int(raw_index)
            except (TypeError, ValueError, OverflowError) as exc:
                raise FineMapSkip("LM fine-mapping sample indices must be integers") from exc
            if index != raw_index or index < 0 or index >= len(fam_ids):
                raise FineMapSkip("LM fine-mapping sample indices are out of range")
            indices.append(index)
        if not indices or len(set(indices)) != len(indices):
            raise FineMapSkip("LM fine-mapping sample indices are empty or duplicated")
        sample_ids = [fam_ids[index] for index in indices]

    fam_index = {sample_id: index for index, sample_id in enumerate(fam_ids)}
    try:
        sample_indices = np.asarray(
            [fam_index[sample_id] for sample_id in sample_ids], dtype=np.int64
        )
    except KeyError as exc:
        raise FineMapSkip(
            f"LM fine-mapping sample metadata contains IID absent from FAM: {exc.args[0]}"
        ) from exc
    if supplied_indices is not None and supplied_ids is not None:
        try:
            provided_indices = np.asarray(list(supplied_indices), dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FineMapSkip("LM fine-mapping sample indices are invalid") from exc
        if not np.array_equal(provided_indices, sample_indices):
            raise FineMapSkip(
                "LM fine-mapping sample IDs and sample indices disagree"
            )

    expected_hash = getattr(args, "finemap_sample_order_sha256", None)
    if not isinstance(expected_hash, str) or expected_hash.strip() == "":
        raise FineMapSkip(
            "LM fine-mapping skipped: ordered GWAS sample hash is unavailable"
        )
    actual_hash = hash_ordered_sample_ids(sample_ids)
    if actual_hash.lower() != expected_hash.strip().lower():
        raise FineMapSkip(
            "LM fine-mapping sample order hash does not match supplied metadata"
        )
    return sample_ids, np.ascontiguousarray(sample_indices, dtype=np.int64)


def _postgwas_dev_help_requested(argv: Optional[list[str]] = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    return "-dev" in tokens or "--dev" in tokens


def _validate_postgwas_finemap_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    args.finemap_requested = str(getattr(args, "finemap", "") or "").lower() == "susie"
    if not args.finemap_requested:
        return

    if len(list(getattr(args, "gwasfile", []) or [])) != 1:
        parser.error(
            "Fine-mapping requires exactly one GWAS input file from -i/--gwasfile."
        )
    if not getattr(args, "bfile", None):
        parser.error("Fine-mapping requires -bfile/--bfile.")
    if not getattr(args, "bimrange_tuples", None):
        parser.error("Fine-mapping requires at least one -bimrange/--bimrange.")
    if int(args.finemap_l) <= 0:
        parser.error("finemap-L must be > 0.")
    if int(args.finemap_max_iter) <= 0:
        parser.error("finemap-max-iter must be > 0.")
    if not np.isfinite(float(args.finemap_tol)) or float(args.finemap_tol) < 0.0:
        parser.error("finemap-tol must be a finite number >= 0.")
    purity_r2 = float(
        getattr(args, "finemap_purity_r2", _POSTGWAS_FINEMAP_PURITY_R2_DEFAULT)
    )
    if not np.isfinite(purity_r2) or not 0.0 <= purity_r2 <= 1.0:
        parser.error("finemap-purity-r2 must be a finite number in [0, 1].")
    args.finemap_purity_r2 = purity_r2
    if not np.isfinite(float(args.memory)) or float(args.memory) <= 0.0:
        parser.error("-mem must be a finite number > 0 (GB).")
    args.finemap_memory_bytes = int(round(float(args.memory) * 1024**3))


def _postgwas_prepare_finemap_summary(
    df: pd.DataFrame,
    chr_col: str,
    pos_col: str,
    locus: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select one locus and derive finite signed z-scores from beta / se."""
    required = [str(chr_col), str(pos_col), "beta", "se"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            "Fine-mapping summary is missing required column(s): " + ", ".join(missing)
        )

    locus_chrom, start_bp, end_bp = locus
    target_chrom = _postgwas_finemap_normalize_chr(locus_chrom)
    if {
        _POSTGWAS_FINEMAP_INDEX_CHROM_COLUMN,
        _POSTGWAS_FINEMAP_INDEX_POS_COLUMN,
    }.issubset(df.columns):
        chrom_norm = df[_POSTGWAS_FINEMAP_INDEX_CHROM_COLUMN]
        pos_num = df[_POSTGWAS_FINEMAP_INDEX_POS_COLUMN]
    else:
        chrom_norm = df[str(chr_col)].map(_postgwas_finemap_normalize_chr)
        pos_num = pd.to_numeric(df[str(pos_col)], errors="coerce")
    pos_valid = np.isfinite(pos_num.to_numpy(dtype=float, na_value=np.nan))
    pos_integral = np.zeros(len(df), dtype=bool)
    if bool(np.any(pos_valid)):
        finite_pos = pos_num.to_numpy(dtype=float, na_value=np.nan)
        pos_integral[pos_valid] = finite_pos[pos_valid] == np.floor(finite_pos[pos_valid])
    in_locus = (
        (chrom_norm == target_chrom).to_numpy(dtype=bool)
        & pos_valid
        & pos_integral
        & (pos_num.to_numpy(dtype=float, na_value=np.nan) >= int(start_bp))
        & (pos_num.to_numpy(dtype=float, na_value=np.nan) <= int(end_bp))
    )

    out = df.loc[in_locus].copy()
    out["chrom_norm"] = chrom_norm.loc[in_locus].to_numpy(dtype=object)
    out["chrom"] = out["chrom_norm"].to_numpy(dtype=object)
    out["pos"] = pos_num.loc[in_locus].to_numpy(dtype=np.int64)
    out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
    out["se"] = pd.to_numeric(out["se"], errors="coerce")
    beta_values = out["beta"].to_numpy(dtype=float, na_value=np.nan)
    se_values = out["se"].to_numpy(dtype=float, na_value=np.nan)
    valid_summary = np.isfinite(beta_values) & np.isfinite(se_values) & (se_values > 0.0)
    z_values = np.full(len(out), np.nan, dtype=float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.divide(beta_values, se_values, out=z_values, where=valid_summary)
    valid_summary &= np.isfinite(z_values)
    invalid_summary = int(len(out) - int(np.count_nonzero(valid_summary)))
    out = out.loc[valid_summary].copy()
    out["z"] = z_values[valid_summary]
    out["locus"] = _postgwas_finemap_locus_label(locus)
    out = out.reset_index(drop=True)
    return out, {
        "input_rows": int(len(df)),
        "locus_rows": int(np.count_nonzero(in_locus)),
        "invalid_summary": invalid_summary,
        "prepared_rows": int(len(out)),
    }


def _postgwas_finemap_nonempty_text(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text != "" and text.lower() not in {"na", "nan", "none"} else None


def _postgwas_align_finemap_locus(
    gwas_df: pd.DataFrame,
    bed_meta: pd.DataFrame,
    ambiguous_bim_sites: Optional[set[tuple[str, int]]] = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Align one prepared GWAS locus to BED metadata in BED matrix order."""
    prefilter_ambiguous_sites = {
        (_postgwas_finemap_normalize_chr(chrom), int(pos))
        for chrom, pos in (ambiguous_bim_sites or set())
    }
    gwas_chrom_col = "chrom_norm" if "chrom_norm" in gwas_df.columns else "chrom"
    for name, frame, columns in (
        ("GWAS", gwas_df, [gwas_chrom_col, "pos", "beta", "se", "z"]),
        ("BED", bed_meta, ["chrom", "pos", "snp", "allele0", "allele1"]),
    ):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Fine-mapping {name} metadata is missing column(s): {', '.join(missing)}")

    gwas = gwas_df.copy().reset_index(drop=True)
    bed = bed_meta.copy().reset_index(drop=True)
    gwas["_chrom_norm"] = gwas[gwas_chrom_col].map(_postgwas_finemap_normalize_chr)
    bed["_chrom_norm"] = bed["chrom"].map(_postgwas_finemap_normalize_chr)
    gwas["_pos_num"] = pd.to_numeric(gwas["pos"], errors="coerce")
    bed["_pos_num"] = pd.to_numeric(bed["pos"], errors="coerce")

    def _valid_identity(frame: pd.DataFrame) -> pd.Series:
        pos_values = frame["_pos_num"].to_numpy(dtype=float, na_value=np.nan)
        return pd.Series(
            np.isfinite(pos_values) & (pos_values == np.floor(pos_values)), index=frame.index
        )

    gwas = gwas.loc[_valid_identity(gwas)].copy()
    bed = bed.loc[_valid_identity(bed)].copy()
    gwas["_pos_num"] = gwas["_pos_num"].astype(np.int64)
    bed["_pos_num"] = bed["_pos_num"].astype(np.int64)

    gwas_by_site: dict[tuple[str, int], list[int]] = {}
    bed_by_site: dict[tuple[str, int], list[int]] = {}
    for idx, row in gwas.iterrows():
        gwas_by_site.setdefault((str(row["_chrom_norm"]), int(row["_pos_num"])), []).append(int(idx))
    for idx, row in bed.iterrows():
        bed_by_site.setdefault((str(row["_chrom_norm"]), int(row["_pos_num"])), []).append(int(idx))

    pair_for_bed: dict[int, int] = {}
    snp_conflicts = 0
    for site, bed_rows in bed_by_site.items():
        gwas_rows = gwas_by_site.get(site, [])
        if len(gwas_rows) == 1 and len(bed_rows) == 1:
            gwas_snp = (
                _postgwas_finemap_nonempty_text(gwas.at[gwas_rows[0], "snp"])
                if "snp" in gwas.columns
                else None
            )
            bed_snp = (
                _postgwas_finemap_nonempty_text(bed.at[bed_rows[0], "snp"])
                if "snp" in bed.columns
                else None
            )
            if bed_snp is None:
                snp_conflicts += 1
                continue
            if gwas_snp is not None and gwas_snp != bed_snp:
                snp_conflicts += 1
                continue
            if (
                site in prefilter_ambiguous_sites
                and gwas_snp is not None
                and bed_snp is not None
                and gwas_snp != bed_snp
            ):
                continue
            pair_for_bed[bed_rows[0]] = gwas_rows[0]
            continue
        if len(gwas_rows) == 0:
            continue
        gwas_ids: dict[str, list[int]] = {}
        bed_ids: dict[str, list[int]] = {}
        for idx in gwas_rows:
            snp = (
                _postgwas_finemap_nonempty_text(gwas.at[idx, "snp"])
                if "snp" in gwas.columns
                else None
            )
            if snp is not None:
                gwas_ids.setdefault(snp, []).append(idx)
        for idx in bed_rows:
            snp = _postgwas_finemap_nonempty_text(bed.at[idx, "snp"])
            if snp is not None:
                bed_ids.setdefault(snp, []).append(idx)
        for snp, matching_bed_rows in bed_ids.items():
            matching_gwas_rows = gwas_ids.get(snp, [])
            if len(matching_bed_rows) == 1 and len(matching_gwas_rows) == 1:
                pair_for_bed[matching_bed_rows[0]] = matching_gwas_rows[0]

        # Duplicate coordinates are only safe when the remaining unnamed
        # summary rows can be resolved by a unique allele pair.  Coordinate
        # order is not a valid identity because BIM may contain alternate
        # alleles at the same position.  Named summary rows are deliberately
        # excluded from the allele fallback: a mismatched named SNP must not
        # acquire a different canonical label merely because its alleles fit.
        unmatched_gwas = {
            int(idx)
            for idx in gwas_rows
            if int(idx) not in set(pair_for_bed.values())
            and (
                "snp" not in gwas.columns
                or _postgwas_finemap_nonempty_text(gwas.at[idx, "snp"]) is None
            )
        }
        unmatched_bed = {
            int(idx) for idx in bed_rows if int(idx) not in pair_for_bed
        }
        gwas_by_alleles: dict[frozenset[str], list[int]] = {}
        bed_by_alleles: dict[frozenset[str], list[int]] = {}
        for idx in unmatched_gwas:
            gwas_pair = _postgwas_variant_allele_pair(
                _postgwas_variant_token(gwas.at[idx, "allele0"], allele=True)
                if "allele0" in gwas.columns
                else None,
                _postgwas_variant_token(gwas.at[idx, "allele1"], allele=True)
                if "allele1" in gwas.columns
                else None,
            )
            if gwas_pair is not None:
                gwas_by_alleles.setdefault(gwas_pair, []).append(idx)
        for idx in unmatched_bed:
            bed_pair = _postgwas_variant_allele_pair(
                _postgwas_variant_token(bed.at[idx, "allele0"], allele=True),
                _postgwas_variant_token(bed.at[idx, "allele1"], allele=True),
            )
            if bed_pair is not None:
                bed_by_alleles.setdefault(bed_pair, []).append(idx)
        for allele_pair, matching_bed_rows in bed_by_alleles.items():
            matching_gwas_rows = gwas_by_alleles.get(allele_pair, [])
            if len(matching_bed_rows) == 1 and len(matching_gwas_rows) == 1:
                pair_for_bed[matching_bed_rows[0]] = matching_gwas_rows[0]

        if len(gwas_rows) != len(bed_rows) or any(
            bed_idx not in pair_for_bed for bed_idx in bed_rows
        ) or any(
            gwas_idx not in set(pair_for_bed.values()) for gwas_idx in gwas_rows
        ):
            snp_conflicts += 1

    gwas_has_alleles = {"allele0", "allele1"}.issubset(gwas.columns)
    records: list[pd.Series] = []
    retained_bed_indices: list[int] = []
    counters = {
        "bed_matches": 0,
        "allele_flips": 0,
        "allele_conflicts": 0,
        "assumed_direction": 0,
        "unresolved_identities": 0,
    }
    for bed_idx in bed.index:
        gwas_idx = pair_for_bed.get(int(bed_idx))
        if gwas_idx is None:
            continue
        counters["bed_matches"] += 1
        gwas_row = gwas.loc[gwas_idx].copy()
        bed_row = bed.loc[bed_idx]
        flip = False
        if gwas_has_alleles:
            g0 = _postgwas_finemap_nonempty_text(gwas_row["allele0"])
            g1 = _postgwas_finemap_nonempty_text(gwas_row["allele1"])
            b0 = _postgwas_finemap_nonempty_text(bed_row["allele0"])
            b1 = _postgwas_finemap_nonempty_text(bed_row["allele1"])
            if None not in (g0, g1, b0, b1):
                assert g0 is not None and g1 is not None and b0 is not None and b1 is not None
                if g0.upper() == b0.upper() and g1.upper() == b1.upper():
                    pass
                elif g0.upper() == b1.upper() and g1.upper() == b0.upper():
                    flip = True
                else:
                    counters["allele_conflicts"] += 1
                    snp_conflicts += 1
                    continue
            else:
                counters["assumed_direction"] += 1
        else:
            counters["assumed_direction"] += 1

        if flip:
            gwas_row["beta"] = -float(gwas_row["beta"])
            gwas_row["z"] = -float(gwas_row["z"])
            counters["allele_flips"] += 1
        canonical_snp = _postgwas_finemap_nonempty_text(bed_row["snp"])
        if canonical_snp is None:
            snp_conflicts += 1
            continue
        gwas_row["snp"] = canonical_snp
        gwas_row["allele0"] = bed_row["allele0"]
        gwas_row["allele1"] = bed_row["allele1"]
        gwas_row["chrom_norm"] = bed_row["_chrom_norm"]
        gwas_row["pos"] = int(bed_row["_pos_num"])
        records.append(gwas_row.drop(labels=["_chrom_norm", "_pos_num"], errors="ignore"))
        retained_bed_indices.append(int(bed_idx))

    counters["unresolved_identities"] = int(len(gwas) - len(set(pair_for_bed.values())))
    if snp_conflicts > 0:
        raise FineMapSkip(
            "GWAS/BIM SNP identity mismatch or missing canonical BIM SNP metadata"
        )
    if len(records) == 0:
        aligned = gwas.drop(columns=["_chrom_norm", "_pos_num"], errors="ignore").iloc[0:0].copy()
        for allele_col in ("allele0", "allele1"):
            if allele_col not in aligned.columns:
                aligned[allele_col] = pd.Series(dtype=object)
    else:
        aligned = pd.DataFrame(records).reset_index(drop=True)
    return aligned, np.asarray(retained_bed_indices, dtype=np.int64), counters


def _postgwas_canonicalize_raw_finemap_rows(
    prepared: pd.DataFrame,
    bfile: object,
) -> pd.DataFrame:
    """Canonicalize every raw-route summary row against its BIM identity.

    This is deliberately a metadata-only pass.  It runs after the LD route has
    been selected and before raw LD clumping, so folded rows cannot carry
    summary-provided SNP labels into credible-set expansion and no genotype
    matrix is allocated merely to establish variant identity.
    """
    prepared_source = prepared.reset_index(drop=True)
    selected_bim_indices, selected_bim_metadata = _postgwas_resolve_prepared_bim_rows(
        bfile,
        prepared_source,
    )
    selected_bim_meta = pd.DataFrame(
        {
            "chrom": [metadata[0] for metadata in selected_bim_metadata],
            "pos": [metadata[1] for metadata in selected_bim_metadata],
            "snp": [metadata[2] for metadata in selected_bim_metadata],
            "allele0": [metadata[3] for metadata in selected_bim_metadata],
            "allele1": [metadata[4] for metadata in selected_bim_metadata],
        }
    )
    selected_site_counts: dict[tuple[str, int], int] = {}
    for chrom, pos, _snp, _allele0, _allele1 in selected_bim_metadata:
        site = (_postgwas_finemap_normalize_chr(chrom), int(pos))
        selected_site_counts[site] = selected_site_counts.get(site, 0) + 1
    ambiguous_sites = {
        site for site, count in selected_site_counts.items() if count > 1
    }
    aligned, aligned_bim_indices, alignment_counts = _postgwas_align_finemap_locus(
        prepared_source,
        selected_bim_meta,
        ambiguous_bim_sites=ambiguous_sites,
    )
    expected_indices = np.arange(len(prepared_source), dtype=np.int64)
    if len(aligned) != len(prepared_source) or not np.array_equal(
        aligned_bim_indices, expected_indices
    ):
        raise FineMapSkip(
            "raw-route BIM identity alignment changed prepared row order or dropped rows"
        )
    aligned = aligned.reset_index(drop=True)
    aligned.attrs["_janusx_finemap_alignment_counts"] = dict(alignment_counts)
    aligned.attrs["_janusx_finemap_bed_rows"] = int(len(selected_bim_indices))
    aligned.attrs["_janusx_finemap_bim_indices"] = list(selected_bim_indices)
    aligned.attrs["_janusx_finemap_bim_metadata"] = list(selected_bim_metadata)
    return aligned


_POSTGWAS_FINEMAP_BASE_OUTPUT_COLUMNS = [
    "locus",
    "chrom",
    "pos",
    "snp",
    "allele0",
    "allele1",
    "beta",
    "se",
    "posterior_mean",
    "pip",
]

_POSTGWAS_FINEMAP_OUTPUT_COLUMNS = [
    "locus",
    "chrom",
    "pos",
    "snp",
    "allele0",
    "allele1",
    "beta",
    "se",
    "cs_set",
    "tag",
    "rsqr",
    "posterior_mean",
    "pip",
    "coverage",
]

_POSTGWAS_FINEMAP_CS_SUMMARY_COLUMNS = [
    "cs",
    "coverage",
    "representative_snp",
    "representative_pip",
    "n_snps",
    "start",
    "end",
    "min_r2",
]

_POSTGWAS_FINEMAP_CS_OUTPUT_COLUMNS = [
    "locus",
    "cs",
    "coverage",
    "chrom",
    "pos",
    "snp",
    "allele0",
    "allele1",
    "pip",
    "representative_snp",
    "is_representative",
]

_POSTGWAS_FINEMAP_CS_MEMBER_COLUMNS = [
    "locus",
    "cs",
    "coverage",
    "chrom",
    "pos",
    "snp",
    "allele0",
    "allele1",
    "beta",
    "se",
    "pip",
    "posterior_mean",
    "tag",
    "rsqr",
]

_POSTGWAS_FINEMAP_CS_COVERAGE = 0.95
_POSTGWAS_FINEMAP_PRIOR_TOL = 1e-9


def _postgwas_finemap_row_key(row: pd.Series) -> tuple[str, int]:
    """Return the coordinate identity used by fine-mapping row joins."""
    chrom_column = "chrom" if "chrom" in row.index else "chrom_norm"
    return (
        _postgwas_finemap_normalize_chr(row[chrom_column]),
        int(row["pos"]),
    )


def _postgwas_sort_finemap_output_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Place all CS rows first and keep each locus/CS block contiguous."""
    if frame.empty:
        return frame.reset_index(drop=True)
    ordered = frame.copy()
    ordered["_finemap_non_cs_order"] = ordered["cs_set"].isna().astype(np.int8)
    ordered["_finemap_cs_order"] = pd.to_numeric(
        ordered["cs_set"].astype(str).str.extract(r"(\d+)$", expand=False),
        errors="coerce",
    ).fillna(np.iinfo(np.int64).max)
    ordered["_finemap_original_order"] = np.arange(len(ordered), dtype=np.int64)
    ordered = ordered.sort_values(
        [
            "_finemap_non_cs_order",
            "locus",
            "_finemap_cs_order",
            "chrom",
            "pos",
            "snp",
            "_finemap_original_order",
        ],
        kind="mergesort",
    )
    return ordered.drop(
        columns=[
            "_finemap_non_cs_order",
            "_finemap_cs_order",
            "_finemap_original_order",
        ]
    ).reset_index(drop=True)


def _postgwas_build_finemap_base_output(
    prepared_rows: pd.DataFrame,
    fitted_rows: pd.DataFrame,
    pip: object,
    posterior_mean: object,
    *,
    locus: str,
) -> pd.DataFrame:
    """Build one base row for every prepared SNP before CS annotation."""
    prepared = prepared_rows.reset_index(drop=True)
    fitted = fitted_rows.reset_index(drop=True)
    required = {"chrom", "pos", "snp", "allele0", "allele1", "beta", "se"}
    missing = sorted(required.difference(prepared.columns))
    if missing:
        raise ValueError(
            "Fine-mapping prepared rows are missing column(s): " + ", ".join(missing)
        )
    missing = sorted(required.difference(fitted.columns))
    if missing:
        raise ValueError(
            "Fine-mapping fitted rows are missing column(s): " + ", ".join(missing)
        )
    pip_array = np.asarray(pip, dtype=np.float64).reshape(-1)
    posterior_array = np.asarray(posterior_mean, dtype=np.float64).reshape(-1)
    if pip_array.shape != (len(fitted),) or posterior_array.shape != (len(fitted),):
        raise ValueError("Fine-mapping fitted arrays do not match fitted rows.")

    fitted_by_key: dict[tuple[str, int], int] = {}
    for index, row in fitted.iterrows():
        key = _postgwas_finemap_row_key(row)
        if key not in fitted_by_key:
            fitted_by_key[key] = int(index)

    rows: list[dict[str, object]] = []
    for _, row in prepared.iterrows():
        fitted_index = fitted_by_key.get(_postgwas_finemap_row_key(row))
        rows.append(
            {
                "locus": locus,
                "chrom": row["chrom"],
                "pos": int(row["pos"]),
                "snp": row["snp"],
                "allele0": row["allele0"],
                "allele1": row["allele1"],
                "beta": float(row["beta"]),
                "se": float(row["se"]),
                "posterior_mean": (
                    float(posterior_array[fitted_index])
                    if fitted_index is not None
                    else np.nan
                ),
                "pip": (
                    float(pip_array[fitted_index])
                    if fitted_index is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=_POSTGWAS_FINEMAP_BASE_OUTPUT_COLUMNS)


def _postgwas_build_finemap_credible_sets(
    alpha: object,
    prior_variance: object,
    *,
    n_variants: int,
    coverage: float = _POSTGWAS_FINEMAP_CS_COVERAGE,
    prior_tol: float = _POSTGWAS_FINEMAP_PRIOR_TOL,
) -> list[tuple[int, tuple[int, ...], float]]:
    """Build deterministic credible sets from SuSiE alpha rows."""
    alpha_array = np.asarray(alpha, dtype=np.float64)
    if alpha_array.ndim != 2:
        raise ValueError("Fine-mapping alpha must be a two-dimensional array.")

    prior_array = np.asarray(prior_variance, dtype=np.float64)
    if prior_array.ndim != 1:
        raise ValueError("Fine-mapping prior variance must be a one-dimensional array.")

    expected_variants = int(n_variants)
    n_effects, observed_variants = alpha_array.shape
    if observed_variants != expected_variants:
        raise ValueError(
            "Fine-mapping alpha shape does not match n_variants: "
            f"got {observed_variants}, expected {expected_variants}."
        )
    if int(prior_array.shape[0]) != int(n_effects):
        raise ValueError(
            "Fine-mapping prior variance count does not match alpha effects: "
            f"got {int(prior_array.shape[0])}, expected {int(n_effects)}."
        )
    if not np.all(np.isfinite(prior_array)):
        raise ValueError("Fine-mapping prior variance contains non-finite values.")
    if np.any(prior_array < 0.0):
        raise ValueError("Fine-mapping prior variance contains negative values.")

    coverage_value = float(coverage)
    if not np.isfinite(coverage_value) or not 0.0 < coverage_value <= 1.0:
        raise ValueError("Fine-mapping credible-set coverage must be in (0, 1].")
    prior_tolerance = float(prior_tol)
    if not np.isfinite(prior_tolerance) or prior_tolerance < 0.0:
        raise ValueError("Fine-mapping prior tolerance must be finite and nonnegative.")

    credible_sets: list[tuple[int, tuple[int, ...], float]] = []
    selected_sets: set[frozenset[int]] = set()
    for effect_index in range(int(n_effects)):
        if (
            not np.isfinite(prior_array[effect_index])
            or prior_array[effect_index] <= prior_tolerance
        ):
            continue

        row = alpha_array[effect_index]
        if np.any(~np.isfinite(row)) or np.any(row < 0.0):
            raise ValueError(
                "Fine-mapping active alpha row contains non-finite or negative values."
            )

        rank = np.argsort(-row, kind="stable")
        cumulative = np.cumsum(row[rank], dtype=np.float64)
        threshold_positions = np.flatnonzero(cumulative >= coverage_value)
        if threshold_positions.size == 0:
            raise ValueError(
                "Fine-mapping active alpha row does not reach requested coverage."
            )
        stop = int(threshold_positions[0])
        representative_indices = tuple(int(index) for index in rank[: stop + 1])
        selected_key = frozenset(representative_indices)
        if selected_key in selected_sets:
            continue
        selected_sets.add(selected_key)
        credible_sets.append(
            (effect_index, representative_indices, float(cumulative[stop]))
        )

    return credible_sets


def _postgwas_filter_finemap_credible_sets_by_purity(
    credible_sets: Sequence[tuple[int, tuple[int, ...], float]],
    ld_matrix: object,
    *,
    purity_r2_threshold: float = _POSTGWAS_FINEMAP_PURITY_R2_DEFAULT,
) -> tuple[list[tuple[int, tuple[int, ...], float]], list[dict[str, object]]]:
    """Filter SuSiE CSs by minimum pairwise absolute LD among fitted tags.

    ``credible_sets`` contains indices in the same matrix order as
    ``ld_matrix``.  The matrix is the route-selected LD used by SuSiE: raw LD
    for the LM route or projected/effective LD for the FvLMM route.  Clumped
    proxy SNPs are deliberately excluded from this calculation; they are
    restored only after the CS-level purity decision.
    """
    threshold = float(purity_r2_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("Fine-mapping purity r2 threshold must be in [0, 1].")

    matrix = np.asarray(ld_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Fine-mapping purity LD matrix must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Fine-mapping purity LD matrix must be finite.")

    n_variants = int(matrix.shape[0])
    kept: list[tuple[int, tuple[int, ...], float]] = []
    diagnostics: list[dict[str, object]] = []
    for cs_number, (
        effect_index_raw,
        representative_indices_raw,
        coverage_raw,
    ) in enumerate(credible_sets, start=1):
        effect_index = int(effect_index_raw)
        representative_indices = tuple(
            int(index) for index in representative_indices_raw
        )
        if not representative_indices:
            raise ValueError("Fine-mapping credible set cannot be empty.")
        if len(set(representative_indices)) != len(representative_indices):
            raise ValueError(
                "Fine-mapping credible-set tag indices must be unique."
            )
        if any(
            index < 0 or index >= n_variants for index in representative_indices
        ):
            raise ValueError(
                "Fine-mapping credible-set tag index is outside the LD matrix."
            )

        n_tags = len(representative_indices)
        if n_tags == 1:
            min_abs_corr = 1.0
            purity_r2 = 1.0
        else:
            submatrix = matrix[np.ix_(representative_indices, representative_indices)]
            upper = submatrix[np.triu_indices(n_tags, k=1)]
            if upper.size == 0 or not np.all(np.isfinite(upper)):
                raise ValueError(
                    "Fine-mapping credible-set purity submatrix is invalid."
                )
            min_abs_corr = float(np.min(np.abs(upper)))
            purity_r2 = float(np.min(np.square(upper)))

        kept_flag = bool(purity_r2 >= threshold)
        diagnostic = {
            "cs_number": cs_number,
            "effect_index": effect_index,
            "n_tags": n_tags,
            "coverage": float(coverage_raw),
            "min_abs_corr": min_abs_corr,
            "purity_r2": purity_r2,
            "threshold": threshold,
            "kept": kept_flag,
        }
        diagnostics.append(diagnostic)
        if kept_flag:
            kept.append(
                (effect_index, representative_indices, float(coverage_raw))
            )

    return kept, diagnostics


def _postgwas_expand_finemap_credible_sets(
    aligned_representatives: pd.DataFrame,
    prepared_rows: pd.DataFrame,
    fold_groups: dict[tuple[str, int], tuple[int, ...]],
    pip: object,
    credible_sets: object,
    *,
    locus: str,
    cs_numbers: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Restore clumped prepared rows into deterministic credible-set rows."""
    fitted = aligned_representatives.reset_index(drop=True)
    prepared = prepared_rows.reset_index(drop=True)
    pip_array = np.asarray(pip, dtype=np.float64).reshape(-1)
    if pip_array.shape != (len(fitted),):
        raise ValueError(
            "Fine-mapping credible-set expansion PIP length does not match "
            f"aligned representatives: got {pip_array.shape[0]}, expected {len(fitted)}."
        )
    if cs_numbers is not None:
        cs_numbers = tuple(int(number) for number in cs_numbers)
        if len(cs_numbers) != len(credible_sets):
            raise ValueError(
                "Fine-mapping credible-set number count does not match credible sets."
            )
        if any(number <= 0 for number in cs_numbers) or len(set(cs_numbers)) != len(
            cs_numbers
        ):
            raise ValueError(
                "Fine-mapping credible-set numbers must be unique and positive."
            )
    for frame_name, frame, required in (
        (
            "aligned representative",
            fitted,
            ("chrom", "pos", "snp", "allele0", "allele1"),
        ),
        ("prepared",
            prepared,
            ("chrom", "pos", "snp"),
        ),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Fine-mapping {frame_name} metadata is missing column(s): "
                + ", ".join(missing)
            )

    def _row_key(row: pd.Series) -> tuple[str, int]:
        return (
            _postgwas_finemap_normalize_chr(row["chrom"]),
            int(row["pos"]),
        )

    def _snp_value(row: pd.Series) -> object:
        return row["snp"] if "snp" in row.index else ""

    def _metadata_row(
        row: pd.Series,
        *,
        cs_name: str,
        representative_snp: object,
        representative: bool,
        representative_pip: float,
        coverage: float,
    ) -> dict[str, object]:
        return {
            "locus": locus,
            "cs": cs_name,
            "chrom": row["chrom"],
            "pos": int(row["pos"]),
            "snp": _snp_value(row),
            "allele0": row["allele0"] if "allele0" in row.index else "",
            "allele1": row["allele1"] if "allele1" in row.index else "",
            # The restored proxy is not a fitted SuSiE variable.  Keep the
            # tag's PIP only on the representative row; duplicating it on
            # every proxy would inflate posterior mass in the CS table.
            "pip": float(representative_pip) if representative else np.nan,
            "coverage": float(coverage),
            "representative_snp": representative_snp,
            "is_representative": bool(representative),
        }

    rows: list[dict[str, object]] = []
    next_cs_number = 1
    for credible_set_index, (
        _effect_index,
        representative_indices,
        achieved_coverage,
    ) in enumerate(credible_sets):
        cs_rows: list[dict[str, object]] = []
        seen_groups: set[tuple[str, int]] = set()
        for fitted_index_raw in representative_indices:
            fitted_index = int(fitted_index_raw)
            if fitted_index < 0 or fitted_index >= len(fitted):
                raise ValueError(
                    "Fine-mapping credible-set representative index is outside "
                    "the aligned representative table."
                )
            representative_row = fitted.iloc[fitted_index]
            representative_key = _row_key(representative_row)
            group = fold_groups.get(representative_key)
            if representative_key in seen_groups:
                continue
            if not group:
                raise ValueError(
                    "Fine-mapping credible-set representative "
                    f"{representative_key[0]}:{representative_key[1]} has no fold group."
                )
            group_indices = tuple(int(index) for index in group)
            invalid_group_indices = tuple(
                index
                for index in group_indices
                if index < 0 or index >= len(prepared)
            )
            if invalid_group_indices:
                raise ValueError(
                    "Fine-mapping credible-set fold-group member index is outside "
                    "the prepared table."
                )
            matching_representative_indices = tuple(
                index
                for index in group_indices
                if _row_key(prepared.iloc[index]) == representative_key
            )
            if not matching_representative_indices:
                raise ValueError(
                    "Fine-mapping credible-set fold group does not contain "
                    f"representative {representative_key[0]}:{representative_key[1]}."
                )
            if matching_representative_indices[0] != group_indices[0]:
                raise ValueError(
                    "Fine-mapping credible-set representative identity mismatch "
                    f"for {representative_key[0]}:{representative_key[1]}."
                )

            representative_snp = _snp_value(representative_row)
            cs_rows.append(
                _metadata_row(
                    representative_row,
                    cs_name="",
                    representative_snp=representative_snp,
                    representative=True,
                    representative_pip=float(pip_array[fitted_index]),
                    coverage=float(achieved_coverage),
                )
            )
            member_indices = sorted(
                {int(index) for index in group_indices[1:]},
                key=lambda index: (
                    _chrom_sort_key(prepared.iloc[index]["chrom"]),
                    int(prepared.iloc[index]["pos"]),
                    str(_snp_value(prepared.iloc[index])),
                    index,
                ),
            )
            for member_index in member_indices:
                if member_index < 0 or member_index >= len(prepared):
                    raise ValueError(
                        "Fine-mapping credible-set fold-group member index is "
                        "outside the prepared table."
                    )
                member_row = prepared.iloc[member_index]
                cs_rows.append(
                    _metadata_row(
                        member_row,
                        cs_name="",
                        representative_snp=representative_snp,
                        representative=False,
                        representative_pip=float(pip_array[fitted_index]),
                        coverage=float(achieved_coverage),
                    )
                )
            seen_groups.add(representative_key)

        if not cs_rows:
            continue
        cs_number = (
            int(cs_numbers[credible_set_index])
            if cs_numbers is not None
            else next_cs_number
        )
        cs_name = f"CS_{cs_number}"
        for row in cs_rows:
            row["cs"] = cs_name
        rows.extend(cs_rows)
        if cs_numbers is None:
            next_cs_number += 1

    return pd.DataFrame(rows, columns=_POSTGWAS_FINEMAP_CS_OUTPUT_COLUMNS)


def _postgwas_build_finemap_cs_summary(cs_rows: pd.DataFrame) -> pd.DataFrame:
    """Build one compact summary row for every expanded credible set.

    The standalone CS table contains one row per member, including restored
    clumped/folded SNPs.  The PIP table needs one row per CS, so this helper
    chooses the highest-PIP representative and summarizes the expanded member
    coordinates without changing the standalone CS output contract.
    """
    summary_columns = ["locus", *_POSTGWAS_FINEMAP_CS_SUMMARY_COLUMNS]
    if cs_rows.empty:
        return pd.DataFrame(columns=summary_columns)
    required = {
        "locus",
        "cs",
        "coverage",
        "chrom",
        "pos",
        "snp",
        "pip",
        "representative_snp",
        "is_representative",
    }
    missing = sorted(required.difference(cs_rows.columns))
    if missing:
        raise ValueError(
            "Fine-mapping CS summary is missing column(s): " + ", ".join(missing)
        )

    rows: list[dict[str, object]] = []
    grouped = cs_rows.groupby(
        ["locus", "cs"], sort=False, dropna=False, as_index=False
    )
    for (locus, cs_name), group in grouped:
        members = group.reset_index(drop=True)
        representatives = members.loc[
            members["is_representative"].astype(bool)
        ].copy()
        if representatives.empty:
            representatives = members.copy()
        representatives["_pip_numeric"] = pd.to_numeric(
            representatives["pip"], errors="coerce"
        )
        representatives = representatives.sort_values(
            ["_pip_numeric", "pos", "snp"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        representative = representatives.iloc[0]
        representative_snp = str(representative["representative_snp"])
        representative_pip = float(representative["pip"])
        positions = pd.to_numeric(members["pos"], errors="coerce")
        if positions.isna().any():
            raise ValueError(
                f"Fine-mapping CS {locus}/{cs_name} contains a non-numeric position."
            )
        group_sizes = (
            members.groupby("representative_snp", sort=False, dropna=False)
            .size()
            .astype(int)
        )
        min_r2 = ">0.99" if bool((group_sizes > 1).any()) else "1"
        rows.append(
            {
                "locus": locus,
                "cs": cs_name,
                "coverage": float(members["coverage"].iloc[0]),
                "representative_snp": representative_snp,
                "representative_pip": representative_pip,
                "n_snps": int(len(members)),
                "start": int(positions.min()),
                "end": int(positions.max()),
                "min_r2": min_r2,
            }
        )
    return pd.DataFrame(rows, columns=summary_columns)


def _postgwas_merge_finemap_cs_summary(
    pip_rows: pd.DataFrame, cs_summary: pd.DataFrame
) -> pd.DataFrame:
    """Annotate representative SNP rows in the per-SNP PIP table with CS data."""
    output = pip_rows.copy()
    for column in _POSTGWAS_FINEMAP_CS_SUMMARY_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    if cs_summary.empty:
        return output
    required = {"locus", "representative_snp", *_POSTGWAS_FINEMAP_CS_SUMMARY_COLUMNS}
    missing = sorted(required.difference(cs_summary.columns))
    if missing:
        raise ValueError(
            "Fine-mapping CS summary is missing column(s): " + ", ".join(missing)
        )
    summary = cs_summary.loc[:, ["locus", *_POSTGWAS_FINEMAP_CS_SUMMARY_COLUMNS]].copy()
    if summary.duplicated(["locus", "representative_snp"]).any():
        raise ValueError("Fine-mapping CS summary has duplicate representatives.")
    if "snp" not in output.columns:
        raise ValueError("Fine-mapping PIP output is missing the SNP identifier column.")
    summary = summary.rename(columns={"representative_snp": "_cs_representative_snp"})
    merged = output.merge(
        summary,
        how="left",
        left_on=["locus", "snp"],
        right_on=["locus", "_cs_representative_snp"],
        sort=False,
        suffixes=("", "_summary"),
        validate="many_to_one",
    )
    representative_key = "_cs_representative_snp"
    merged["representative_snp"] = merged[representative_key].where(
        merged[representative_key].notna(), merged["representative_snp"]
    )
    for column in _POSTGWAS_FINEMAP_CS_SUMMARY_COLUMNS:
        summary_column = f"{column}_summary"
        if summary_column in merged.columns:
            merged[column] = merged[summary_column]
            merged = merged.drop(columns=[summary_column])
    merged = merged.drop(columns=[representative_key])
    for column in ("n_snps", "start", "end"):
        numeric = pd.to_numeric(merged[column], errors="coerce")
        if bool((numeric.dropna() != np.floor(numeric.dropna())).any()):
            raise ValueError(
                f"Fine-mapping CS summary column {column} contains non-integer values."
            )
        merged[column] = numeric.astype("Int64")
    return merged


def _postgwas_build_finemap_cs_member_annotations(
    cs_rows: pd.DataFrame,
    *,
    locus_r: object,
    fitted_rows: pd.DataFrame,
    prepared_rows: pd.DataFrame,
    pip: object,
    posterior_mean: object,
    fold_r2: Optional[
        dict[tuple[str, int], dict[tuple[str, int], float]]
    ] = None,
) -> pd.DataFrame:
    """Build per-SNP CS annotations, including restored clump members.

    ``cs_rows`` contains the expanded CS membership.  Fitted representatives
    receive their fitted PIP/posterior mean, while restored members retain a
    missing PIP and posterior mean because they were not separate SuSiE
    variables.  Their ``tag`` and ``rsqr`` identify the fitted variable whose
    posterior mass applies to the entire LD clump.  ``rsqr`` is computed
    against the tag in the same LD matrix used by the fit.
    """
    if cs_rows.empty:
        return pd.DataFrame(columns=_POSTGWAS_FINEMAP_CS_MEMBER_COLUMNS)
    matrix = np.asarray(locus_r, dtype=np.float64)
    fitted = fitted_rows.reset_index(drop=True)
    prepared = prepared_rows.reset_index(drop=True)
    pip_array = np.asarray(pip, dtype=np.float64).reshape(-1)
    posterior_array = np.asarray(posterior_mean, dtype=np.float64).reshape(-1)
    fold_r2 = fold_r2 or {}
    if matrix.ndim != 2 or matrix.shape != (len(fitted), len(fitted)):
        raise ValueError("Fine-mapping CS annotation LD matrix has invalid dimensions.")
    if pip_array.shape != (len(fitted),) or posterior_array.shape != (len(fitted),):
        raise ValueError("Fine-mapping CS annotation fit arrays have invalid dimensions.")

    fitted_by_key: dict[tuple[str, int], int] = {}
    for index, row in fitted.iterrows():
        key = _postgwas_finemap_row_key(row)
        if key in fitted_by_key:
            raise ValueError(
                "Fine-mapping fitted rows contain duplicate coordinate identity: "
                f"{key[0]}:{key[1]}"
            )
        fitted_by_key[key] = int(index)
    prepared_by_key: dict[tuple[str, int], pd.Series] = {}
    for _, row in prepared.iterrows():
        key = _postgwas_finemap_row_key(row)
        if key not in prepared_by_key:
            prepared_by_key[key] = row
    rows: list[dict[str, object]] = []
    for (locus, cs_name, representative_snp), group in cs_rows.groupby(
        ["locus", "cs", "representative_snp"], sort=False, dropna=False
    ):
        group = group.reset_index(drop=True)
        representatives = group.loc[group["is_representative"].astype(bool)]
        if representatives.empty:
            representatives = group.iloc[[0]]
        tag_row = representatives.iloc[0]
        tag_snp = str(representative_snp)
        tag_key = _postgwas_finemap_row_key(tag_row)
        tag_index = fitted_by_key.get(tag_key)
        if tag_index is None:
            raise ValueError(
                f"Fine-mapping CS {locus}/{cs_name} tag is absent from fitted rows: {tag_snp}"
            )
        for _, member in group.iterrows():
            member_snp = str(member["snp"])
            member_key = _postgwas_finemap_row_key(member)
            fitted_index = fitted_by_key.get(member_key)
            if fitted_index is None:
                source_row = prepared_by_key.get(member_key)
                if source_row is None:
                    raise ValueError(
                        f"Fine-mapping CS member is absent from fitted and prepared rows: {member_snp}"
                    )
                member_beta = source_row.get("beta", np.nan)
                member_se = source_row.get("se", np.nan)
            else:
                source_row = fitted.iloc[int(fitted_index)]
                member_beta = source_row.get("beta", np.nan)
                member_se = source_row.get("se", np.nan)
            if fitted_index is None:
                # A clumped proxy is not a separate SuSiE variable.  Copying
                # the tag PIP to every proxy duplicates posterior mass and
                # makes the per-SNP PIP column cease to be a probability.
                member_pip = np.nan
                member_posterior = np.nan
                rsqr_value = fold_r2.get(tag_key, {}).get(member_key)
                if rsqr_value is None:
                    raise ValueError(
                        "Fine-mapping CS folded member is missing its tag LD r2: "
                        f"{tag_snp}->{member_snp}"
                    )
                rsqr = float(rsqr_value)
            else:
                member_pip = float(pip_array[int(fitted_index)])
                member_posterior = float(posterior_array[int(fitted_index)])
                rsqr = float(matrix[int(tag_index), int(fitted_index)] ** 2)
            if member_key == tag_key:
                member_pip = float(pip_array[tag_index])
                member_posterior = float(posterior_array[tag_index])
                rsqr = 1.0
            rows.append(
                {
                    "locus": locus,
                    "cs": cs_name,
                    "coverage": float(member["coverage"]),
                    "chrom": member["chrom"],
                    "pos": int(member["pos"]),
                    "snp": member_snp,
                    "allele0": member.get("allele0", ""),
                    "allele1": member.get("allele1", ""),
                    "beta": float(member_beta),
                    "se": float(member_se),
                    "pip": member_pip,
                    "posterior_mean": member_posterior,
                    "tag": tag_snp,
                    "rsqr": rsqr,
                }
            )
    return pd.DataFrame(rows, columns=_POSTGWAS_FINEMAP_CS_MEMBER_COLUMNS)


def _postgwas_merge_finemap_cs_members(
    pip_rows: pd.DataFrame,
    cs_rows: pd.DataFrame,
    prepared_rows: pd.DataFrame,
    *,
    locus_r: object = None,
    fitted_rows: Optional[pd.DataFrame] = None,
    pip: object = None,
    posterior_mean: object = None,
    fold_r2: Optional[
        dict[tuple[str, int], dict[tuple[str, int], float]]
    ] = None,
    rsqr_by_member: Optional[dict[tuple[str, str], float]] = None,
) -> pd.DataFrame:
    """Merge CS members into the requested ordered per-SNP output table."""
    if rsqr_by_member is not None:
        # Small deterministic unit-test adapter; production uses locus_r.
        member_rows = cs_rows.copy()
        member_rows["tag"] = member_rows["representative_snp"].astype(str)
        member_rows["rsqr"] = [
            float(rsqr_by_member[(str(row["locus"]), str(row["snp"]))])
            for _, row in member_rows.iterrows()
        ]
        prepared_by_key = {
            _postgwas_finemap_row_key(row): row
            for _, row in prepared_rows.iterrows()
        }
        member_rows["beta"] = [
            prepared_by_key[_postgwas_finemap_row_key(row)]["beta"]
            for _, row in member_rows.iterrows()
        ]
        member_rows["se"] = [
            prepared_by_key[_postgwas_finemap_row_key(row)]["se"]
            for _, row in member_rows.iterrows()
        ]
        member_rows["posterior_mean"] = np.nan
        member_rows["pip"] = member_rows["pip"].astype(float)
        member_rows.loc[
            ~member_rows["is_representative"].astype(bool), "pip"
        ] = np.nan
    else:
        if fitted_rows is None or pip is None or posterior_mean is None or locus_r is None:
            raise ValueError("Fine-mapping CS member merge requires fitted LD inputs.")
        member_rows = _postgwas_build_finemap_cs_member_annotations(
            cs_rows,
            locus_r=locus_r,
            fitted_rows=fitted_rows,
            prepared_rows=prepared_rows,
            pip=pip,
            posterior_mean=posterior_mean,
            fold_r2=fold_r2,
        )
    base = pip_rows.copy()
    if "locus" not in base.columns or "snp" not in base.columns:
        raise ValueError("Fine-mapping PIP output is missing locus or SNP identifiers.")
    member_rows = member_rows.copy()
    member_rows["_cs_order"] = pd.to_numeric(
        member_rows["cs"].astype(str).str.extract(r"(\d+)$", expand=False),
        errors="coerce",
    )
    member_rows = member_rows.sort_values(
        ["locus", "_cs_order", "pos", "snp"],
        kind="mergesort",
    )
    member_rows["_merge_chrom"] = member_rows["chrom"].map(
        _postgwas_finemap_normalize_chr
    )
    member_rows["_merge_pos"] = pd.to_numeric(
        member_rows["pos"], errors="raise"
    ).astype(np.int64)
    member_rows = member_rows.drop_duplicates(
        ["locus", "_merge_chrom", "_merge_pos"], keep="first"
    )
    cs_keys = {
        (str(row["locus"]), str(row["_merge_chrom"]), int(row["_merge_pos"]))
        for _, row in member_rows.iterrows()
    }
    output_rows: list[dict[str, object]] = []
    for _, member in member_rows.iterrows():
        output_rows.append(
            {
                "locus": member["locus"],
                "chrom": member["chrom"],
                "pos": int(member["pos"]),
                "snp": member["snp"],
                "allele0": member["allele0"],
                "allele1": member["allele1"],
                "beta": member["beta"],
                "se": member["se"],
                "cs_set": member["cs"],
                "tag": member["tag"],
                "rsqr": member["rsqr"],
                "posterior_mean": member["posterior_mean"],
                "pip": member["pip"],
                "coverage": member["coverage"],
            }
        )
    for _, source in base.iterrows():
        key = (
            str(source["locus"]),
            _postgwas_finemap_normalize_chr(source["chrom"]),
            int(source["pos"]),
        )
        if key in cs_keys:
            continue
        output_rows.append(
            {
                "locus": source["locus"],
                "chrom": source["chrom"],
                "pos": int(source["pos"]),
                "snp": source["snp"],
                "allele0": source["allele0"],
                "allele1": source["allele1"],
                "beta": source["beta"],
                "se": source["se"],
                "cs_set": pd.NA,
                "tag": pd.NA,
                "rsqr": pd.NA,
                "posterior_mean": source["posterior_mean"],
                "pip": source["pip"],
                "coverage": pd.NA,
            }
        )
    result = pd.DataFrame(output_rows, columns=_POSTGWAS_FINEMAP_OUTPUT_COLUMNS)
    # The CS/non-CS merge introduces ``pd.NA`` in annotation columns.  Without
    # an explicit numeric conversion pandas can retain an object dtype, in
    # which case ``DataFrame.to_csv(float_format=...)`` does not format the
    # values consistently (notably rsqr and coverage).
    for column in ("beta", "se", "rsqr", "posterior_mean", "pip", "coverage"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return _postgwas_sort_finemap_output_rows(result)


_POSTGWAS_FINEMAP_INDEX_CHROM_COLUMN = "_janusx_finemap_chrom_norm"
_POSTGWAS_FINEMAP_INDEX_POS_COLUMN = "_janusx_finemap_pos_num"


def _postgwas_format_finemap_float(value: float) -> str:
    """Format fine-mapping values without rounding tiny nonzero values to zero."""
    numeric = float(value)
    if numeric != 0.0 and abs(numeric) < 1e-4:
        return f"{numeric:.4e}"
    return f"{numeric:.4f}"


def _postgwas_load_finemap_gwas(
    path: str,
    chr_col: str,
    pos_col: str,
) -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load only fine-map columns and build stable chromosome/position row indices."""
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    required = [str(chr_col), str(pos_col), "beta", "se"]
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(
            "Fine-mapping summary is missing required column(s): " + ", ".join(missing)
        )
    requested = [str(chr_col), str(pos_col), "snp", "allele0", "allele1", "beta", "se"]
    usecols = list(dict.fromkeys(column for column in requested if column in header))
    identity_dtypes = {
        column: "string"
        for column in ("snp", "allele0", "allele1")
        if column in usecols
    }
    gwas = pd.read_csv(path, sep="\t", usecols=usecols, dtype=identity_dtypes)
    gwas[_POSTGWAS_FINEMAP_INDEX_CHROM_COLUMN] = gwas[str(chr_col)].map(
        _postgwas_finemap_normalize_chr
    )
    gwas[_POSTGWAS_FINEMAP_INDEX_POS_COLUMN] = pd.to_numeric(
        gwas[str(pos_col)], errors="coerce"
    )

    chrom_values = gwas[_POSTGWAS_FINEMAP_INDEX_CHROM_COLUMN].to_numpy(dtype=object)
    pos_values = gwas[_POSTGWAS_FINEMAP_INDEX_POS_COLUMN].to_numpy(
        dtype=float, na_value=np.nan
    )
    valid = np.isfinite(pos_values) & (pos_values == np.floor(pos_values))
    rows_by_chrom: dict[str, list[int]] = {}
    for row_index in np.flatnonzero(valid):
        rows_by_chrom.setdefault(str(chrom_values[row_index]), []).append(int(row_index))

    index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, row_indices in rows_by_chrom.items():
        rows = np.asarray(row_indices, dtype=np.int64)
        order = np.argsort(pos_values[rows], kind="stable")
        sorted_rows = rows[order]
        index[chrom] = (sorted_rows, pos_values[sorted_rows].astype(np.int64))
    return gwas, index


def _postgwas_select_finemap_gwas_locus(
    gwas: pd.DataFrame,
    index: dict[str, tuple[np.ndarray, np.ndarray]],
    locus: tuple[str, int, int],
) -> pd.DataFrame:
    target_chrom = _postgwas_finemap_normalize_chr(locus[0])
    indexed = index.get(target_chrom)
    if indexed is None:
        return gwas.iloc[0:0]
    row_indices, positions = indexed
    left = int(np.searchsorted(positions, int(locus[1]), side="left"))
    right = int(np.searchsorted(positions, int(locus[2]), side="right"))
    return gwas.iloc[row_indices[left:right]]


def _postgwas_build_sample_matched_raw_ld(
    *,
    args: argparse.Namespace,
    prepared: pd.DataFrame,
    locus: tuple[str, int, int],
    sample_ids: Sequence[str],
    bed_indices: Optional[np.ndarray],
    selected_bim_indices: list[int],
    selected_bim_metadata: list[
        tuple[str, int, str | None, str | None, str | None]
    ],
    existing_bytes: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build ordinary LD from the verified GWAS sample subset."""
    locus_label = _postgwas_finemap_locus_label(locus)
    verified_sample_ids = [str(value) for value in sample_ids]
    if not verified_sample_ids:
        raise FineMapSkip(
            "ordinary LD requires verified GWAS sample metadata; sample subset is empty"
        )
    if len(selected_bim_indices) != len(prepared) or len(selected_bim_metadata) != len(
        prepared
    ):
        raise FineMapSkip("raw-route BIM identity metadata does not match prepared rows")
    selected_bim_rows = len(selected_bim_indices)
    try:
        memory_limit_bytes = int(
            getattr(args, "finemap_memory_bytes", _POSTGWAS_FINEMAP_MAX_MEMORY_BYTES)
        )
        estimated_memory = _postgwas_check_finemap_memory(
            n_variants=selected_bim_rows,
            n_samples=len(verified_sample_ids),
            existing_bytes=existing_bytes,
            max_bytes=memory_limit_bytes,
        )
    except MemoryError as exc:
        raise FineMapSkip(
            f"SuSiE locus {locus_label} raw-LD memory preflight rejected the locus: {exc}"
        ) from exc
    logger.info(
        "SuSiE locus %s memory preflight: variants=%d samples=%d estimated_peak=%.2f GiB limit=%.2f GiB.",
        locus_label,
        selected_bim_rows,
        len(verified_sample_ids),
        estimated_memory / float(1024**3),
        memory_limit_bytes / float(1024**3),
    )
    selected_bim_meta = pd.DataFrame(
        {
            "chrom": [metadata[0] for metadata in selected_bim_metadata],
            "pos": [metadata[1] for metadata in selected_bim_metadata],
            "snp": [metadata[2] for metadata in selected_bim_metadata],
            "allele0": [metadata[3] for metadata in selected_bim_metadata],
            "allele1": [metadata[4] for metadata in selected_bim_metadata],
        }
    )
    selected_site_counts: dict[tuple[str, int], int] = {}
    for chrom, pos, _snp, _allele0, _allele1 in selected_bim_metadata:
        site = (_postgwas_finemap_normalize_chr(chrom), int(pos))
        selected_site_counts[site] = selected_site_counts.get(site, 0) + 1
    ambiguous_bim_sites = {
        site for site, count in selected_site_counts.items() if count > 1
    }
    genotype_chunks: list[np.ndarray] = []
    returned_sites: list[object] = []
    try:
        loader_signature = inspect.signature(load_genotype_chunks)
    except (TypeError, ValueError) as exc:
        raise FineMapSkip(
            "ordinary LD genotype loader signature is unavailable; exact sample/SNP selection cannot be proven"
        ) from exc
    loader_parameters = loader_signature.parameters
    if "sample_ids" not in loader_parameters:
        raise FineMapSkip(
            "ordinary LD genotype loader does not support exact sample selection"
        )
    if (
        "snp_indices" not in loader_parameters
        and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in loader_parameters.values()
        )
    ):
        raise FineMapSkip(
            "ordinary LD genotype loader does not support exact SNP selection"
        )
    try:
        genotype_iter = load_genotype_chunks(
            str(args.bfile),
            chunk_size=max(1, min(20_000, len(selected_bim_indices))),
            maf=0.0,
            missing_rate=1.0,
            impute=True,
            model="add",
            het=1.0,
            snp_indices=[int(index) for index in selected_bim_indices],
            sample_ids=verified_sample_ids,
        )
        for genotype_chunk, sites in genotype_iter:
            block = np.asarray(genotype_chunk, dtype=np.float32)
            if block.ndim != 2 or block.shape[0] != len(sites):
                raise ValueError(
                    "ordinary LD genotype chunk and site metadata have inconsistent shapes"
                )
            genotype_chunks.append(np.ascontiguousarray(block, dtype=np.float32))
            returned_sites.extend(list(sites))
    except FineMapSkip:
        raise
    except MemoryError as exc:
        raise FineMapSkip(
            f"SuSiE locus {locus_label} raw-LD memory allocation was unavailable: {exc}"
        ) from exc
    except _PostGWASExpectedCompatibilityError as exc:
        _postgwas_skip_for_expected_source_error(exc, "ordinary LD genotype")
        raise
    except (OSError, EOFError, UnicodeError, pd.errors.ParserError) as exc:
        _postgwas_skip_for_expected_source_error(exc, "ordinary LD genotype")
        raise
    except RuntimeError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "ordinary LD genotype",
            runtime_markers=(
                "snp selection",
                "snp_indices",
                "sample selection",
                "sample_ids",
                "does not support exact",
                "selection unavailable",
            ),
        )
        raise
    except TypeError as exc:
        message = str(exc).lower()
        if any(
            marker in message
            for marker in ("snp_indices", "sample_ids", "unexpected keyword")
        ):
            raise FineMapSkip(
                f"ordinary LD genotype loader does not support exact selection: {exc}"
            ) from exc
        raise
    except ValueError as exc:
        _postgwas_skip_for_expected_source_error(
            exc,
            "ordinary LD genotype",
            value_markers=(
                "snp_indices is empty",
                "invalid snp_",
                "genotype source",
                "genotype chunk",
                "site metadata",
                "malformed",
            ),
        )
        raise
    if len(genotype_chunks) == 0:
        raise FineMapSkip("ordinary LD genotype loader returned no variants")
    if len(returned_sites) != len(selected_bim_indices):
        raise FineMapSkip(
            "ordinary LD genotype selection changed the requested variant set"
        )

    genotype_all = np.vstack(genotype_chunks).astype(np.float64, copy=False)
    signature_to_bim: dict[
        tuple[str, int, frozenset[str] | None], int
    ] = {}
    for bim_index, bim_metadata in zip(selected_bim_indices, selected_bim_metadata):
        signature = (
            bim_metadata[0],
            bim_metadata[1],
            _postgwas_variant_allele_pair(
                bim_metadata[3], bim_metadata[4]
            ),
        )
        if signature in signature_to_bim:
            raise FineMapSkip(
                "ordinary LD selected BIM rows have indistinguishable identity"
            )
        signature_to_bim[signature] = bim_index
    row_by_bim_index: dict[int, int] = {}
    for row_index, site in enumerate(returned_sites):
        signature = _postgwas_site_identity_signature(site)
        bim_index = signature_to_bim.get(signature)
        if bim_index is None:
            coordinate_matches = [
                index
                for index, metadata in zip(selected_bim_indices, selected_bim_metadata)
                if metadata[0] == signature[0] and metadata[1] == signature[1]
            ]
            if signature[2] is None and len(coordinate_matches) == 1:
                bim_index = coordinate_matches[0]
            else:
                raise FineMapSkip(
                    "ordinary LD genotype site metadata cannot prove exact variant identity"
                )
        if bim_index in row_by_bim_index:
            raise FineMapSkip("ordinary LD genotype loader returned duplicate identity")
        row_by_bim_index[bim_index] = row_index
    if set(row_by_bim_index) != set(selected_bim_indices):
        raise FineMapSkip(
            "ordinary LD genotype site metadata does not cover selected BIM rows"
        )
    selected_rows = [row_by_bim_index[index] for index in selected_bim_indices]
    genotypes = np.ascontiguousarray(
        genotype_all[np.asarray(selected_rows, dtype=np.int64)], dtype=np.float64
    )
    aligned, aligned_bim_indices, alignment_counts = _postgwas_align_finemap_locus(
        prepared,
        selected_bim_meta,
        ambiguous_bim_sites=ambiguous_bim_sites,
    )
    if aligned.empty:
        empty = prepared.iloc[0:0].copy()
        empty.attrs["_janusx_finemap_alignment_counts"] = dict(alignment_counts)
        empty.attrs["_janusx_finemap_empty_reason"] = (
            "no variants remained after BED identity and allele alignment"
        )
        empty.attrs["_janusx_finemap_bed_rows"] = int(len(selected_bim_indices))
        return np.empty((0, 0), dtype=np.float64), empty
    if (
        np.any(aligned_bim_indices < 0)
        or np.any(aligned_bim_indices >= genotypes.shape[0])
        or len(np.unique(aligned_bim_indices)) != len(aligned_bim_indices)
    ):
        raise FineMapSkip(f"SuSiE locus {locus_label} returned invalid BED indices.")
    if len(aligned) != len(aligned_bim_indices):
        raise FineMapSkip(
            f"SuSiE locus {locus_label} alignment length mismatch: "
            f"{len(aligned)} rows vs {len(aligned_bim_indices)} BED indices."
        )
    genotypes = np.ascontiguousarray(genotypes[aligned_bim_indices], dtype=np.float64)
    if genotypes.shape[0] == 0:
        empty = prepared.iloc[0:0].copy()
        empty.attrs["_janusx_finemap_alignment_counts"] = dict(alignment_counts)
        empty.attrs["_janusx_finemap_empty_reason"] = (
            f"no polymorphic BED variants matched {len(prepared)} prepared GWAS rows"
        )
        empty.attrs["_janusx_finemap_bed_rows"] = int(len(selected_bim_indices))
        return np.empty((0, 0), dtype=np.float64), empty
    if not np.all(np.isfinite(genotypes)):
        raise FineMapSkip("ordinary LD genotypes contain non-finite values")
    centered = genotypes - np.mean(genotypes, axis=1, keepdims=True)
    norms = np.sqrt(np.einsum("ij,ij->i", centered, centered, dtype=np.float64))
    valid = np.isfinite(norms) & (norms > 0.0)
    monomorphic_exclusions = int(np.count_nonzero(~valid))
    if not np.any(valid):
        empty = prepared.iloc[0:0].copy()
        empty.attrs["_janusx_finemap_alignment_counts"] = dict(alignment_counts)
        empty.attrs["_janusx_finemap_empty_reason"] = (
            f"no polymorphic BED variants matched {len(prepared)} prepared GWAS rows "
            f"(monomorphic={monomorphic_exclusions})"
        )
        empty.attrs["_janusx_finemap_bed_rows"] = int(len(selected_bim_indices))
        return np.empty((0, 0), dtype=np.float64), empty
    if not np.all(valid):
        valid_indices = np.flatnonzero(valid).astype(np.int64, copy=False)
        aligned = aligned.iloc[valid_indices].reset_index(drop=True)
        centered = centered[valid_indices]
        norms = norms[valid_indices]
    locus_r = np.asarray(
        (centered @ centered.T) / (norms[:, None] * norms[None, :]),
        dtype=np.float64,
    )
    locus_r = np.clip(0.5 * (locus_r + locus_r.T), -1.0, 1.0)
    np.fill_diagonal(locus_r, 1.0)
    aligned.attrs["_janusx_finemap_alignment_counts"] = dict(alignment_counts)
    aligned.attrs["_janusx_finemap_bed_rows"] = int(len(selected_bim_indices))
    return locus_r, aligned


def _postgwas_cleanup_finemap_paths(paths: tuple[str, ...]) -> None:
    """Best-effort cleanup without replacing an active generation error."""
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            continue


def _postgwas_create_finemap_backup_path(final_path: str) -> str:
    """Create a unique, current-run backup path beside its final output."""
    directory = os.path.dirname(final_path) or "."
    prefix = f"{os.path.basename(final_path)}."
    descriptor, backup_path = tempfile.mkstemp(
        prefix=prefix,
        suffix=".bak",
        dir=directory,
    )
    try:
        os.close(descriptor)
    except OSError:
        _postgwas_cleanup_finemap_paths((backup_path,))
        raise
    return backup_path


def _postgwas_run_susie_finemap_body(
    args: argparse.Namespace, logger: logging.Logger
) -> str:
    """Run SuSiE-RSS serially for each requested PLINK-backed locus."""
    gwas_files = [str(path) for path in list(getattr(args, "gwasfile", []) or [])]
    if len(gwas_files) != 1:
        raise ValueError("Fine-mapping requires exactly one GWAS input file.")
    result_model = _postgwas_finemap_result_model(gwas_files[0])
    sidecar: Optional[GwasNullModelSidecarV1] = None
    if result_model is not None:
        sidecar = discover_matching_sidecar(gwas_files[0])
        validate_sidecar_dependencies(
            sidecar,
            gwas_files[0],
            args.bfile,
            expected_model=result_model,
        )
        if result_model != "fvlmm":
            raise FineMapSkip(
                f"{result_model} mixed-model fine-mapping is unsupported: "
                "common-null score statistics are unavailable"
            )
    route_model = sidecar.model if sidecar is not None else result_model
    ld_route = _postgwas_select_finemap_ld_route(route_model, sidecar)
    raw_sample_ids: Optional[list[str]] = None
    raw_sample_indices: Optional[np.ndarray] = None
    finemap_loci = getattr(args, "finemap_bimrange_tuples", None)
    loci = list(
        finemap_loci
        if finemap_loci is not None
        else (getattr(args, "bimrange_tuples", []) or [])
    )
    if len(loci) == 0:
        raise ValueError("Fine-mapping requires at least one bimrange locus.")
    plot_bimrange_tuples = list(
        getattr(args, "bimrange_tuples", None) or loci
    )
    gwas, gwas_index = _postgwas_load_finemap_gwas(
        gwas_files[0], str(args.chr), str(args.pos)
    )
    _, plot_seg_defs, _ = _filter_df_by_bimranges(
        gwas,
        str(args.chr),
        str(args.pos),
        plot_bimrange_tuples,
        logger,
        gwas_files[0],
    )
    plot_layout = _build_bimrange_layout(
        plot_seg_defs,
        interval_ratio=float(getattr(args, "interval", 0.5)),
    )
    if ld_route == "raw":
        raw_sample_ids, raw_sample_indices = _postgwas_resolve_finemap_sample_ids(
            args,
            args.bfile,
        )
        if raw_sample_indices.shape != (len(raw_sample_ids),):
            raise FineMapSkip(
                "LM fine-mapping sample ID/index mapping has inconsistent dimensions"
            )
        logger.info(
            "LM fine-mapping verified sample subset: count=%d indices=%s order_sha256=%s.",
            len(raw_sample_ids),
            raw_sample_indices.tolist(),
            hash_ordered_sample_ids(raw_sample_ids),
        )
    gwas_memory_bytes = int(gwas.memory_usage(deep=True, index=True).sum())
    plink_sample_count = _postgwas_count_plink_samples(args.bfile)
    clump_sample_count = (
        len(raw_sample_ids) if raw_sample_ids is not None else plink_sample_count
    )
    finemap_memory_limit_bytes = int(
        getattr(args, "finemap_memory_bytes", _POSTGWAS_FINEMAP_MAX_MEMORY_BYTES)
    )
    clump_available_bytes = max(
        0,
        finemap_memory_limit_bytes
        - gwas_memory_bytes
        - _POSTGWAS_FINEMAP_MEMORY_RESERVE_BYTES,
    )
    clump_preload_max_rows = max(
        1,
        clump_available_bytes // max(1, clump_sample_count * 4),
    )

    out_dir = str(getattr(args, "out", ".") or ".")
    output_path, final_paths, temporary_paths = _postgwas_susie_finemap_output_paths(args)
    plot_requested = len(final_paths) > 1
    temporary_path = temporary_paths[0]
    figure_path = final_paths[1] if plot_requested else None
    figure_temporary_path = temporary_paths[1] if plot_requested else None
    os.makedirs(out_dir, mode=0o755, exist_ok=True)
    _postgwas_cleanup_finemap_paths(temporary_paths)

    output_frames: list[pd.DataFrame] = []
    plot_records: list[dict[str, object]] = []
    warned_assumed_direction = False
    gwas_has_allele_columns = {"allele0", "allele1"}.issubset(gwas.columns)

    for locus in loci:
        locus_tuple = (str(locus[0]), int(locus[1]), int(locus[2]))
        locus_label = _postgwas_finemap_locus_label(locus_tuple)
        locus_gwas = _postgwas_select_finemap_gwas_locus(
            gwas, gwas_index, locus_tuple
        )
        prepared, summary_counts = _postgwas_prepare_finemap_summary(
            locus_gwas,
            chr_col=str(args.chr),
            pos_col=str(args.pos),
            locus=locus_tuple,
        )
        if prepared.empty:
            logger.warning(
                "SuSiE locus %s skipped: no finite beta/se rows with se > 0 "
                "(%d GWAS rows considered, %d invalid summary rows).",
                locus_label,
                summary_counts["locus_rows"],
                summary_counts["invalid_summary"],
            )
            continue

        prepared_source = prepared.reset_index(drop=True)
        plot_source = prepared_source.loc[
            :, ["chrom", "pos", "snp", "z"]
        ].reset_index(drop=True)
        effective_source: Optional[pd.DataFrame] = None
        effective_locus_r: Optional[np.ndarray] = None
        if ld_route == "raw":
            # Establish canonical BIM identity for every regional row before
            # raw clumping can form folded groups.  This metadata-only pass is
            # intentionally after route selection and before any raw LD use.
            prepared_source = _postgwas_canonicalize_raw_finemap_rows(
                prepared_source,
                args.bfile,
            )
        else:
            if sidecar is None:
                raise FineMapSkip("FvLMM fine-mapping requires matched sidecar metadata")
            # Build the effective matrix on every verified regional row before
            # any clumping.  The same matrix is then used by both clumping and
            # SuSiE; no raw genotype LD is touched on this route.
            effective_locus_r, effective_source = _postgwas_build_fvlmm_effective_ld(
                args=args,
                record=sidecar,
                prepared=prepared_source,
                bed_indices=None,
                logger=logger,
                live_gwas_bytes=gwas_memory_bytes,
            )
            prepared_source = effective_source.reset_index(drop=True)
        effective_clump_started = time.perf_counter()
        prepared, clump_counts, fold_groups = _postgwas_finemap_ldclump(
            prepared_source,
            genofile=str(args.bfile),
            locus=locus_tuple,
            logger=logger,
            preload_max_rows=clump_preload_max_rows,
            sample_ids=raw_sample_ids,
            ld_matrix=effective_locus_r,
        )
        if ld_route == "fvlmm":
            fvlmm_threads = max(1, int(getattr(args, "thread", 1)))
            _postgwas_log_fvlmm_stage(
                logger,
                "effective_clump",
                effective_clump_started,
                requested_threads=fvlmm_threads,
                using_threads=1,
            )
        logger.info(
            "SuSiE locus %s LDclump: retained=%d/%d clumped=%d groups=%d "
            "(r2>=%.6f, window=%d bp).",
            locus_label,
            clump_counts["retained_rows"],
            clump_counts["input_rows"],
            clump_counts["clumped_rows"],
            clump_counts["groups"],
            float(_POSTGWAS_FINEMAP_LDCLUMP_R2),
            max(1, abs(int(locus_tuple[2]) - int(locus_tuple[1]))),
        )
        if prepared.empty:
            logger.warning(
                "SuSiE locus %s skipped: LDclump retained no lead variants.",
                locus_label,
            )
            continue

        selected_bim_rows = 0
        if ld_route == "fvlmm":
            if effective_locus_r is None:
                raise FineMapSkip("FvLMM effective-LD matrix was not constructed")
            matrix_indices = np.asarray(
                getattr(prepared, "attrs", {}).get(
                    "_janusx_finemap_matrix_indices", []
                ),
                dtype=np.int64,
            )
            if matrix_indices.ndim != 1 or matrix_indices.size != len(prepared):
                raise FineMapSkip(
                    "FvLMM clumping did not return an explicit matrix index map"
                )
            if (
                np.any(matrix_indices < 0)
                or np.any(matrix_indices >= effective_locus_r.shape[0])
                or np.unique(matrix_indices).size != matrix_indices.size
            ):
                raise FineMapSkip("FvLMM clumping returned an invalid matrix index map")
            locus_r = np.ascontiguousarray(
                effective_locus_r[np.ix_(matrix_indices, matrix_indices)],
                dtype=np.float64,
            )
            aligned = prepared.copy()
        else:
            selected_bim_indices = getattr(prepared, "attrs", {}).get(
                "_janusx_finemap_bim_indices"
            )
            selected_bim_metadata = getattr(prepared, "attrs", {}).get(
                "_janusx_finemap_bim_metadata"
            )
            locus_r, aligned = _postgwas_build_sample_matched_raw_ld(
                args=args,
                prepared=prepared,
                locus=locus_tuple,
                sample_ids=raw_sample_ids or [],
                selected_bim_indices=selected_bim_indices,
                selected_bim_metadata=selected_bim_metadata,
                existing_bytes=gwas_memory_bytes,
                logger=logger,
            )
            selected_bim_rows = int(
                getattr(aligned, "attrs", {}).get(
                    "_janusx_finemap_bed_rows", len(aligned)
                )
            )

        alignment_counts = {
            "bed_matches": len(aligned),
            "allele_flips": 0,
            "allele_conflicts": 0,
            "assumed_direction": 0,
            "unresolved_identities": 0,
        }
        raw_alignment_counts = getattr(aligned, "attrs", {}).get(
            "_janusx_finemap_alignment_counts", {}
        )
        if isinstance(raw_alignment_counts, dict):
            alignment_counts.update(
                {
                    key: int(value)
                    for key, value in raw_alignment_counts.items()
                    if key in alignment_counts
                }
            )
        if alignment_counts["assumed_direction"] > 0 and not warned_assumed_direction:
            if gwas_has_allele_columns:
                warning_detail = "some GWAS allele values are missing"
            else:
                warning_detail = "GWAS allele0/allele1 columns are missing"
            logger.warning(
                "Warning: %s; effect direction is assumed to match -bfile.",
                warning_detail,
            )
            warned_assumed_direction = True
        monomorphic_exclusions = max(0, selected_bim_rows - len(aligned))
        empty_reason = getattr(aligned, "attrs", {}).get(
            "_janusx_finemap_empty_reason"
        )
        bed_rows = int(
            getattr(aligned, "attrs", {}).get(
                "_janusx_finemap_bed_rows", len(aligned)
            )
        )
        if len(aligned) == 0 or locus_r.shape[0] == 0:
            logger.warning(
                "SuSiE locus %s skipped: %s.",
                locus_label,
                empty_reason
                or (
                    f"no polymorphic BED variants matched {len(prepared)} prepared "
                    f"GWAS rows (monomorphic={monomorphic_exclusions})"
                ),
            )
            continue
        if not isinstance(aligned, pd.DataFrame):
            raise RuntimeError(
                f"SuSiE locus {locus_label} LD route returned invalid aligned rows."
            )
        if locus_r.ndim != 2 or locus_r.shape != (len(aligned), len(aligned)):
            raise RuntimeError(
                f"SuSiE locus {locus_label} aligned z/LD dimensions do not agree."
            )
        z = np.ascontiguousarray(aligned["z"], dtype=np.float64)
        if locus_r.shape != (len(aligned), len(aligned)) or z.shape != (len(aligned),):
            raise RuntimeError(
                f"SuSiE locus {locus_label} aligned z/LD dimensions do not agree."
            )
        susie_started = time.perf_counter()
        try:
            fit = jxrs.susie_rss_f64(
                z,
                locus_r,
                l=min(int(args.finemap_l), len(aligned)),
                max_iter=int(args.finemap_max_iter),
                tol=float(args.finemap_tol),
                threads=int(args.thread),
            )
        except Exception as exc:
            raise RuntimeError(f"SuSiE locus {locus_label} solver failed: {exc}") from exc
        if ld_route == "fvlmm":
            fvlmm_threads = max(1, int(getattr(args, "thread", 1)))
            _postgwas_log_fvlmm_stage(
                logger,
                "susie",
                susie_started,
                requested_threads=fvlmm_threads,
                using_threads=fvlmm_threads,
            )

        try:
            pip = np.asarray(fit["pip"], dtype=np.float64).reshape(-1)
            posterior_mean = np.asarray(
                fit["posterior_mean"], dtype=np.float64
            ).reshape(-1)
            alpha = np.asarray(fit["alpha"], dtype=np.float64)
            prior_variance = np.asarray(
                fit["prior_variance"], dtype=np.float64
            ).reshape(-1)
        except Exception as exc:
            raise RuntimeError(
                f"SuSiE locus {locus_label} solver returned invalid result arrays."
            ) from exc
        if pip.shape != (len(aligned),) or posterior_mean.shape != (len(aligned),):
            raise RuntimeError(
                f"SuSiE locus {locus_label} solver result length mismatch."
            )
        if (
            alpha.ndim != 2
            or alpha.shape[1] != len(aligned)
            or prior_variance.shape != (alpha.shape[0],)
        ):
            raise RuntimeError(
                f"SuSiE locus {locus_label} solver alpha/prior variance dimensions "
                "do not agree with the fitted representatives."
            )
        logger.info(
            "SuSiE locus %s prior_variances=%s.",
            locus_label,
            prior_variance.tolist(),
        )
        if not np.all(np.isfinite(prior_variance)) or np.any(prior_variance < 0.0):
            raise RuntimeError(
                f"SuSiE locus {locus_label} solver returned invalid prior variance values."
            )
        if not bool(np.all(np.isfinite(pip))) or bool(np.any((pip < 0.0) | (pip > 1.0))):
            raise RuntimeError(
                f"SuSiE locus {locus_label} solver returned PIP outside finite [0, 1]."
            )
        if not bool(np.all(np.isfinite(posterior_mean))):
            raise RuntimeError(
                f"SuSiE locus {locus_label} solver returned non-finite posterior mean."
            )

        try:
            credible_sets = _postgwas_build_finemap_credible_sets(
                alpha,
                prior_variance,
                n_variants=len(aligned),
            )
        except Exception as exc:
            raise RuntimeError(
                f"SuSiE locus {locus_label} credible-set construction failed: {exc}"
            ) from exc
        purity_threshold = float(
            getattr(
                args,
                "finemap_purity_r2",
                _POSTGWAS_FINEMAP_PURITY_R2_DEFAULT,
            )
        )
        try:
            credible_sets, purity_diagnostics = (
                _postgwas_filter_finemap_credible_sets_by_purity(
                    credible_sets,
                    locus_r,
                    purity_r2_threshold=purity_threshold,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                f"SuSiE locus {locus_label} credible-set purity check failed: {exc}"
            ) from exc
        kept_cs_numbers: list[int] = []
        for diagnostic in purity_diagnostics:
            cs_number = int(diagnostic["cs_number"])
            kept = bool(diagnostic["kept"])
            if kept:
                kept_cs_numbers.append(cs_number)
            logger.info(
                "SuSiE locus %s CS_%d purity: tags=%d coverage=%.6f "
                "min_abs_corr=%.6f purity_r2=%.6f threshold=%.6f kept=%s.",
                locus_label,
                cs_number,
                int(diagnostic["n_tags"]),
                float(diagnostic["coverage"]),
                float(diagnostic["min_abs_corr"]),
                float(diagnostic["purity_r2"]),
                purity_threshold,
                kept,
            )

        metadata_columns = ("chrom", "pos", "snp", "allele0", "allele1")
        missing_metadata = [
            column for column in metadata_columns if column not in aligned.columns
        ]
        if missing_metadata:
            raise RuntimeError(
                f"SuSiE locus {locus_label} LD route omitted metadata: "
                + ", ".join(missing_metadata)
            )
        # Both routes return aligned rows in the exact matrix order.  Keeping
        # this single table as the metadata source prevents projected-diagonal
        # filtering from drifting away from Z, PIP, or folded CS members.
        retained_meta = aligned.loc[:, metadata_columns].reset_index(drop=True)
        try:
            cs_locus_output = _postgwas_expand_finemap_credible_sets(
                retained_meta,
                prepared_source,
                fold_groups,
                pip,
                credible_sets,
                locus=locus_label,
                cs_numbers=kept_cs_numbers,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SuSiE locus {locus_label} credible-set expansion failed: {exc}"
            ) from exc
        fitted_output_rows = retained_meta.assign(
            beta=aligned["beta"].to_numpy(dtype=np.float64),
            se=aligned["se"].to_numpy(dtype=np.float64),
        )
        # Start from every prepared GWAS row.  Clumping only determines the
        # variables fitted by SuSiE; it must not silently remove non-CS folded
        # variants from the final per-SNP table.
        locus_base_output = _postgwas_build_finemap_base_output(
            prepared_source,
            fitted_output_rows,
            pip,
            posterior_mean,
            locus=locus_label,
        )
        locus_output = _postgwas_merge_finemap_cs_members(
            locus_base_output,
            cs_locus_output,
            prepared_source,
            locus_r=locus_r,
            fitted_rows=fitted_output_rows,
            pip=pip,
            posterior_mean=posterior_mean,
            fold_r2=getattr(prepared, "attrs", {}).get(
                "_janusx_finemap_fold_r2", {}
            ),
        )
        locus_output = locus_output.reset_index(drop=True)
        if plot_requested:
            plot_records.append(
                _postgwas_build_susie_locus_plot_record(
                    plot_source,
                    locus_output,
                    fitted_output_rows,
                    pip,
                    locus=locus_label,
                    bimrange_tuples=plot_bimrange_tuples,
                    interval_ratio=float(getattr(args, "interval", 0.5)),
                    palette_spec=getattr(args, "palette_spec", None),
                    scatter_size=float(
                        getattr(
                            args,
                            "_postgwas_single_scatter_size",
                            _DEFAULT_SCATTER_SIZE,
                        )
                    ),
                    layout=plot_layout,
                )
            )
        output_frames.append(locus_output)
        logger.info(
            "SuSiE locus %s: GWAS=%d invalid=%d BED=%d monomorphic=%d matched=%d "
            "flips=%d conflicts=%d assumed_direction=%d unresolved=%d iterations=%d "
            "converged=%s output=%d.",
            locus_label,
            summary_counts["locus_rows"],
            summary_counts["invalid_summary"],
            bed_rows,
            monomorphic_exclusions,
            alignment_counts["bed_matches"],
            alignment_counts["allele_flips"],
            alignment_counts["allele_conflicts"],
            alignment_counts["assumed_direction"],
            alignment_counts["unresolved_identities"],
            int(fit.get("n_iter", 0)),
            bool(fit.get("converged", False)),
            len(locus_output),
        )
        del (
            locus_gwas,
            prepared_source,
            plot_source,
            effective_source,
            prepared,
            fold_groups,
            aligned,
            locus_r,
            z,
            fit,
            pip,
            posterior_mean,
            alpha,
            prior_variance,
            retained_meta,
            cs_locus_output,
            locus_base_output,
            locus_output,
            effective_locus_r,
        )

    if len(output_frames) == 0:
        raise RuntimeError("SuSiE fine-mapping produced no successful locus.")

    merged = _postgwas_sort_finemap_output_rows(
        pd.concat(output_frames, ignore_index=True)
    )
    backup_paths: list[Optional[str]] = [None] * len(final_paths)
    had_prior = [os.path.exists(path) for path in final_paths]
    backed_up = [False] * len(final_paths)
    published = [False] * len(final_paths)
    try:
        merged.to_csv(
            temporary_path,
            sep="\t",
            index=False,
            columns=_POSTGWAS_FINEMAP_OUTPUT_COLUMNS,
            float_format=_postgwas_format_finemap_float,
            na_rep="",
            lineterminator="\n",
        )
        if plot_requested:
            _postgwas_plot_susie_locus_records(
                plot_records,
                args,
                str(figure_temporary_path),
                logger=logger,
            )
        for index, final_path in enumerate(final_paths):
            if had_prior[index]:
                backup_path = _postgwas_create_finemap_backup_path(final_path)
                backup_paths[index] = backup_path
                os.replace(final_path, backup_path)
                backed_up[index] = True
        for index, temporary_output in enumerate(temporary_paths):
            os.replace(temporary_output, final_paths[index])
            published[index] = True
    except Exception:
        for index in range(len(final_paths) - 1, -1, -1):
            if backed_up[index]:
                try:
                    os.replace(str(backup_paths[index]), final_paths[index])
                except Exception as rollback_exc:
                    logger.warning(
                        "SuSiE fine-mapping rollback restore failed for backup %s "
                        "to final %s: %s; the original publication error is "
                        "preserved and the backup is retained.",
                        str(backup_paths[index]),
                        final_paths[index],
                        rollback_exc,
                    )
                    continue
                backed_up[index] = False
            elif published[index] and not had_prior[index]:
                _postgwas_cleanup_finemap_paths((final_paths[index],))
        cleanup_backups = tuple(
            path
            for index, path in enumerate(backup_paths)
            if path is not None and not backed_up[index]
        )
        _postgwas_cleanup_finemap_paths(temporary_paths + cleanup_backups)
        raise
    _postgwas_cleanup_finemap_paths(
        tuple(path for path in backup_paths if path is not None)
    )
    logger.info("SuSiE fine-mapping output: %s", format_path_for_display(output_path))
    if plot_requested and figure_path is not None:
        logger.info("SuSiE locus figure: %s", format_path_for_display(figure_path))
    return output_path


def _run_postgwas_susie_finemap(
    args: argparse.Namespace, logger: logging.Logger
) -> Optional[str]:
    """Run fine-mapping, warning-and-skipping expected compatibility failures."""
    _output_path, _final_paths, temporary_paths = _postgwas_susie_finemap_output_paths(args)
    try:
        return _postgwas_run_susie_finemap_body(args, logger)
    except FineMapSkip as exc:
        _postgwas_cleanup_finemap_paths(temporary_paths)
        logger.warning(
            "fine-mapping skipped: %s; no new PIP output was generated.",
            str(exc),
        )
        return None
    except Exception:
        _postgwas_cleanup_finemap_paths(temporary_paths)
        raise


def _parse_bimrange(value: object, logger: logging.Logger) -> tuple[str, int, int]:
    """
    Parse --bimrange.
    Supported formats:
      - chr:start-end
      - chr:start:end
    Numeric interpretation:
      - default: Mb
      - if start/end look like bp-scale integers (>6 digits), they are
        interpreted as bp (and axis labels remain in Mb).
    """
    text = str(value).strip()
    m = re.match(r"^([^:]+):([0-9]*\.?[0-9]+)(?:-|:)([0-9]*\.?[0-9]+)$", text)
    if m is None:
        raise ValueError(
            f"Invalid --bimrange format: {value}. "
            "Use chr:start-end (or chr:start:end)."
        )
    chrom = m.group(1)
    start_raw = m.group(2).strip()
    end_raw = m.group(3).strip()
    start_num = float(start_raw)
    end_num = float(end_raw)

    def _looks_like_bp_integer(token: str) -> bool:
        tok = str(token).strip()
        if "." in tok:
            return False
        tok = tok.lstrip("0")
        if tok == "":
            tok = "0"
        return len(tok) > 6

    use_bp_input = _looks_like_bp_integer(start_raw) or _looks_like_bp_integer(end_raw)
    if use_bp_input:
        logger.warning(
            "Warning: --bimrange looks like bp coordinates (>6 digits); "
            "interpreting start/end as bp and showing axis in Mb."
        )
        start = int(round(start_num))
        end = int(round(end_num))
        if start < 0 or end < 0:
            raise ValueError("Invalid --bimrange: start/end must be >= 0 (bp).")
        if start > end:
            logger.warning(
                f"bimrange start > end ({start} > {end}); swapped to {end}-{start} bp."
            )
            start, end = end, start
    else:
        if start_num < 0 or end_num < 0:
            raise ValueError("Invalid --bimrange: start/end must be >= 0 (Mb).")
        if start_num > end_num:
            logger.warning(
                f"bimrange start > end ({start_num} > {end_num}); swapped to {end_num}-{start_num} Mb."
            )
            start_num, end_num = end_num, start_num
        start = int(round(start_num * 1_000_000))
        end = int(round(end_num * 1_000_000))
    return chrom, start, end


def _parse_ldclump_window_bp(value: object) -> int:
    """
    Parse LD clump window size to bp.

    Supported units:
      - kb / k (default when unit omitted)
      - mb / m
      - bp / b
    """
    text = str(value).strip().lower()
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$", text)
    if m is None:
        raise ValueError(
            f"Invalid --LDclump window: {value}. "
            "Use formats like 500kb, 0.5mb, or 200000bp."
        )
    raw_val = float(m.group(1))
    unit = m.group(2)
    if raw_val <= 0:
        raise ValueError("--LDclump window must be > 0.")
    if unit in {"", "kb", "k"}:
        factor = 1_000.0
    elif unit in {"mb", "m"}:
        factor = 1_000_000.0
    elif unit in {"bp", "b"}:
        factor = 1.0
    else:
        raise ValueError(
            f"Unsupported --LDclump window unit: {unit}. "
            "Use kb/mb/bp."
        )
    out = int(round(raw_val * factor))
    if out <= 0:
        raise ValueError("--LDclump window must be > 0 bp.")
    return out


def _parse_ldclump_spec(value: object) -> tuple[int, float]:
    """
    Parse --LDclump payload: [window, r2].
    """
    if value is None:
        raise ValueError("--LDclump is None.")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "Invalid --LDclump format. Use: --LDclump <window> <r2> "
            "(e.g. --LDclump 500kb 0.8)."
        )
    window_bp = _parse_ldclump_window_bp(value[0])
    try:
        r2_thr = float(value[1])
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid --LDclump r2 threshold: {value[1]}") from e
    if not (0.0 <= r2_thr <= 1.0):
        raise ValueError("--LDclump r2 threshold must be within [0, 1].")
    return window_bp, r2_thr


def _parse_ylim_spec(value: object) -> tuple[Optional[float], Optional[float]]:
    """
    Parse y-range spec for Manhattan plotting.
    Supports:
      - "6" -> (0.0, 6.0)
      - "0:6" / "0-6" / "[0:6]" -> (0.0, 6.0)
      - "2:" / "2-" -> (2.0, None)     # upper auto
      - ":6" / "-6" -> (None, 6.0)     # lower auto
    """
    if value is None:
        raise ValueError("--ylim is None.")
    raw_items = list(value) if isinstance(value, (list, tuple)) else [value]
    tokens = [str(x).strip() for x in raw_items if str(x).strip() != ""]
    if len(tokens) == 0:
        raise ValueError("--ylim is empty.")
    if len(tokens) == 2:
        try:
            lo = float(tokens[0])
            hi = float(tokens[1])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid --ylim format: {' '.join(tokens)}. Use <max>, <min:max>, <min:>, <:max>, or <min> <max>."
            ) from e
    elif len(tokens) > 2:
        raise ValueError(
            "Invalid --ylim format: too many values. Use <max>, <min:max>, <min:>, <:max>, or <min> <max>."
        )
    else:
        text = tokens[0]
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()

        # Range form: <lo:hi>, <lo:>, <:hi> (also accepts '-')
        if (":" in text) or ("-" in text):
            m = re.match(r"^\s*([0-9]*\.?[0-9]*)\s*(?:-|:)\s*([0-9]*\.?[0-9]*)\s*$", text)
            if m is None:
                raise ValueError(
                    f"Invalid --ylim format: {value}. Use <max>, <min:max>, <min:>, <:max>, or <min> <max> "
                    "(e.g. 6, 0:6, 2:, :6, 2 10)."
                )
            lo_txt = m.group(1).strip()
            hi_txt = m.group(2).strip()
            if lo_txt == "" and hi_txt == "":
                raise ValueError("Invalid --ylim: at least one bound must be provided.")
            lo = float(lo_txt) if lo_txt != "" else None
            hi = float(hi_txt) if hi_txt != "" else None
        else:
            # Single number means [0, max]
            try:
                hi = float(text)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid --ylim format: {value}. Use <max>, <min:max>, <min:>, <:max>, or <min> <max>."
                ) from e
            lo = 0.0

    if lo is not None:
        if not np.isfinite(lo):
            raise ValueError("--ylim lower bound must be finite.")
        if lo < 0:
            raise ValueError("--ylim lower bound must be >= 0.")
    if hi is not None:
        if not np.isfinite(hi):
            raise ValueError("--ylim upper bound must be finite.")
        if hi <= 0:
            raise ValueError("--ylim upper bound must be > 0.")
    if lo is not None and hi is not None and hi <= lo:
        raise ValueError("--ylim upper bound must be greater than lower bound.")
    return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)


def _format_ylim_spec_text(
    source: object,
    lo: Optional[float],
    hi: Optional[float],
) -> Optional[str]:
    if source is None:
        return None
    raw_items = list(source) if isinstance(source, (list, tuple)) else [source]
    tokens = [str(x).strip() for x in raw_items if str(x).strip() != ""]
    if len(tokens) == 1:
        text = tokens[0]
        if ":" not in text and "-" not in text and "," not in text:
            return f"{float(hi):g}" if hi is not None else text
    if lo is not None and hi is not None:
        return f"{float(lo):g}:{float(hi):g}"
    if lo is not None:
        return f"{float(lo):g}:"
    if hi is not None:
        return f":{float(hi):g}"
    return None


def _format_float_series(values: list[float]) -> str:
    return ",".join(f"{float(x):g}" for x in values)


def _format_bimrange_tuple(item: tuple[str, int, int]) -> str:
    chrom, start, end = item
    return f"{chrom}:{start / 1_000_000:g}-{end / 1_000_000:g} Mb"


def _merge_overlapping_bimranges(
    bimranges: list[tuple[str, int, int]],
    logger: logging.Logger,
    *,
    warn_overlaps: bool = True,
) -> list[tuple[str, int, int]]:
    """
    Merge overlapping bimranges on the same chromosome.

    Input is assumed to be sorted by chromosome/start/end.
    """
    if len(bimranges) <= 1:
        return bimranges

    merged: list[list[object]] = []
    for chrom, start, end in bimranges:
        chrom = str(chrom)
        start_i = int(start)
        end_i = int(end)
        chrom_norm = _normalize_chr(chrom)
        if len(merged) == 0:
            merged.append([chrom, chrom_norm, start_i, end_i])
            continue

        prev = merged[-1]
        prev_chrom = str(prev[0])
        prev_norm = str(prev[1])
        prev_start = int(prev[2])
        prev_end = int(prev[3])

        if prev_norm == chrom_norm and start_i <= prev_end:
            new_start = min(prev_start, start_i)
            new_end = max(prev_end, end_i)
            if warn_overlaps:
                logger.warning(
                    "Warning: Overlapping bimrange detected; merged "
                    f"{_format_bimrange_tuple((prev_chrom, prev_start, prev_end))} and "
                    f"{_format_bimrange_tuple((chrom, start_i, end_i))} to "
                    f"{_format_bimrange_tuple((prev_chrom, new_start, new_end))}."
                )
            prev[2] = new_start
            prev[3] = new_end
            continue

        merged.append([chrom, chrom_norm, start_i, end_i])

    return [(str(x[0]), int(x[2]), int(x[3])) for x in merged]


def _filter_df_by_bimranges(
    df: pd.DataFrame,
    chr_col: str,
    pos_col: str,
    bimranges: list[tuple[str, int, int]],
    logger: logging.Logger,
    file: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], int]:
    """
    Keep SNPs in one or more bimranges.

    - Multiple ranges are supported.
    - Overlapping ranges should be merged before calling this function.
    """
    n_before = int(df.shape[0])
    if n_before == 0 or len(bimranges) == 0:
        out = df.iloc[0:0].copy()
        out["__seg_id"] = np.array([], dtype=int)
        return out, [], n_before

    chr_norm = df[chr_col].astype(str).map(_normalize_chr)
    pos_num = pd.to_numeric(df[pos_col], errors="coerce")
    chrom_max_map: dict[str, int] = {}
    valid_pos_mask = pos_num.notna() & np.isfinite(pos_num.to_numpy(dtype=float))
    if bool(valid_pos_mask.any()):
        chrom_pos_df = pd.DataFrame(
            {
                "chrom_norm": chr_norm.loc[valid_pos_mask].to_numpy(dtype=object),
                "pos_num": pos_num.loc[valid_pos_mask].astype(np.int64).to_numpy(),
            }
        )
        if chrom_pos_df.shape[0] > 0:
            for chrom_key, max_pos in chrom_pos_df.groupby("chrom_norm")["pos_num"].max().items():
                chrom_max_map[str(chrom_key)] = int(max_pos)
    seg_id = np.full(n_before, -1, dtype=int)
    seg_defs: list[dict[str, object]] = []

    for i, (chrom, start, end) in enumerate(bimranges):
        target_chr = _normalize_chr(chrom)
        mask = (chr_norm == target_chr) & (pos_num >= start) & (pos_num <= end)
        n_hit = int(mask.sum())
        if n_hit == 0:
            logger.warning(
                f"No SNPs in bimrange {_format_bimrange_tuple((chrom, start, end))} for file {file}."
            )
        else:
            logger.info(
                f"Bimrange {_format_bimrange_tuple((chrom, start, end))}: matched {n_hit} SNPs."
            )
        assign = (seg_id < 0) & mask.to_numpy()
        seg_id[assign] = i
        display_end = int(end)
        chrom_max = chrom_max_map.get(str(target_chr))
        if chrom_max is not None:
            display_end = min(int(end), int(chrom_max))
        display_end = max(int(start), int(display_end))
        seg_defs.append(
            {
                "id": i,
                "chrom": str(chrom),
                "chrom_norm": target_chr,
                "start": int(start),
                "end": int(display_end),
                "query_end": int(end),
                "length": float(max(1, int(display_end) - int(start))),
            }
        )

    keep = seg_id >= 0
    out = df.loc[keep].copy()
    out["__seg_id"] = seg_id[keep]
    return out, seg_defs, n_before


def _build_bimrange_layout(
    seg_defs: list[dict[str, object]],
    *,
    interval_ratio: float = 0.5,
) -> list[dict[str, object]]:
    if len(seg_defs) == 0:
        return []
    seg_lengths = [float(s["length"]) for s in seg_defs]
    gap = (
        float(resolve_manhattan_chr_gap(seg_lengths, interval_ratio=float(interval_ratio)))
        if len(seg_defs) > 1
        else 0.0
    )
    offset = 0.0
    layout: list[dict[str, object]] = []
    for s in seg_defs:
        seg = dict(s)
        seg["offset"] = offset
        seg["x_start"] = offset
        seg["x_end"] = offset + float(seg["length"])
        seg["label"] = (
            f"{seg['chrom']}:{int(seg['start']) / 1_000_000:g}-"
            f"{int(seg['end']) / 1_000_000:g}Mb"
        )
        layout.append(seg)
        offset = float(seg["x_end"]) + gap
    return layout


def _apply_segmented_x_to_plotmodel(
    plotmodel: GWASPLOT,
    filtered_df: pd.DataFrame,
    chr_col: str,
    pos_col: str,
    layout: list[dict[str, object]],
) -> None:
    if len(layout) == 0 or filtered_df.shape[0] == 0:
        return

    # (chrom_norm, pos) -> seg_id
    chr_vals = filtered_df[chr_col].astype(str).map(_normalize_chr).to_numpy()
    pos_vals = pd.to_numeric(filtered_df[pos_col], errors="coerce").to_numpy()
    seg_vals = filtered_df["__seg_id"].to_numpy()
    key_to_seg: dict[tuple[str, int], int] = {}
    for c, p, sid in zip(chr_vals, pos_vals, seg_vals):
        if not np.isfinite(p):
            continue
        key = (str(c), int(round(float(p))))
        if key not in key_to_seg:
            key_to_seg[key] = int(sid)

    layout_by_id = {int(seg["id"]): seg for seg in layout}

    # plotmodel uses integer chr IDs; map them back to normalized chr labels.
    id_to_chr_norm = {
        i + 1: _normalize_chr(label) for i, label in enumerate(plotmodel.chr_labels)
    }

    idx_chr = np.asarray(plotmodel.df.index.get_level_values(0), dtype=np.int64)
    idx_pos = np.asarray(plotmodel.df.index.get_level_values(1), dtype=np.int64)
    x_new = np.asarray(plotmodel.df["x"], dtype=float).copy()
    for j, (cid, pos) in enumerate(zip(idx_chr, idx_pos)):
        chr_norm = id_to_chr_norm.get(int(cid))
        if chr_norm is None:
            continue
        sid = key_to_seg.get((chr_norm, int(pos)))
        if sid is None:
            continue
        seg = layout_by_id.get(int(sid))
        if seg is None:
            continue
        start = int(seg["start"])
        offset = float(seg["offset"])
        length = float(seg["length"])
        rel = float(pos - start)
        rel = min(max(rel, 0.0), length)
        x_new[j] = offset + rel

    plotmodel.df["x"] = x_new
    plotmodel._janusx_bim_layout = layout  # type: ignore[attr-defined]


def _apply_bimrange_manhattan_axis(
    ax: plt.Axes,
    chrom_label: object,
    start_bp: int,
    end_bp: int,
) -> None:
    left = float(min(start_bp, end_bp))
    right = float(max(start_bp, end_bp))
    if right <= left:
        return
    ax.set_xlim(left, right)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1_000_000:g}Mb"))
    c = _sanitize_plot_text(chrom_label)
    ax.set_xlabel(c)


def _apply_multi_bimrange_manhattan_axis(
    ax: plt.Axes,
    layout: list[dict[str, object]],
    *,
    label_fontsize: Optional[float] = None,
) -> None:
    if len(layout) == 0:
        return
    left = float(layout[0]["x_start"])
    right = float(layout[-1]["x_end"])
    if not right > left:
        return
    ax.set_xlim(left, right)
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    centers: list[float] = []
    chromosomes: list[str] = []
    for i in range(len(layout) - 1):
        x_end = float(layout[i]["x_end"])
        x_next = float(layout[i + 1]["x_start"])
        xb = 0.5 * (x_end + x_next)
        ax.axvline(x=xb, color="black", linestyle=":", linewidth=0.7, alpha=0.9)
    for segment in layout:
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        segment_ticks = MaxNLocator(nbins=4).tick_values(segment_start, segment_end)
        segment_ticks = np.asarray(segment_ticks, dtype=np.float64)
        segment_ticks = segment_ticks[
            np.isfinite(segment_ticks)
            & (segment_ticks >= segment_start)
            & (segment_ticks <= segment_end)
        ]
        if segment_ticks.size == 0:
            segment_ticks = np.asarray([segment_start, segment_end], dtype=np.float64)
        segment_ticks = np.unique(segment_ticks)
        offset = float(segment["offset"])
        for tick in segment_ticks.tolist():
            tick_positions.append(offset + float(tick) - segment_start)
            tick_labels.append(f"{float(tick) / 1_000_000:g}Mb")
        centers.append(0.5 * (float(segment["x_start"]) + float(segment["x_end"])))
        chromosomes.append(_sanitize_plot_text(segment["chrom"]))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")
    if label_fontsize is not None:
        ax.tick_params(axis="x", labelsize=float(label_fontsize))
    chromosome_transform = ax.get_xaxis_transform()
    chromosome_fontsize = 5.0 if label_fontsize is None else float(label_fontsize)
    for center, chromosome in zip(centers, chromosomes):
        ax.text(
            center,
            -0.18,
            chromosome,
            transform=chromosome_transform,
            ha="center",
            va="top",
            fontsize=chromosome_fontsize,
            clip_on=False,
            zorder=40,
        )


def _show_end_locs_without_xticks(
    ax: plt.Axes,
    *,
    label_fontsize: Optional[float] = None,
) -> None:
    x0, x1 = ax.get_xlim()
    fmt = ax.xaxis.get_major_formatter()

    def _fmt(v: float, i: int) -> str:
        if callable(fmt):
            try:
                s = str(fmt(v, i)).strip()
                if s != "":
                    return s
            except Exception:
                pass
        return f"{v:g}"

    left_lab = getattr(ax, "_janusx_loc_left_label", None)
    right_lab = getattr(ax, "_janusx_loc_right_label", None)
    if left_lab is None:
        left_lab = _fmt(float(x0), 0)
    if right_lab is None:
        right_lab = _fmt(float(x1), 1)
    left_lab = _sanitize_plot_text(left_lab)
    right_lab = _sanitize_plot_text(right_lab)

    ax.set_xticks([])
    ax.tick_params(axis="x", which="both", length=0, labelbottom=False)
    trans = ax.transAxes
    loc_fontsize = 5.0 if label_fontsize is None else float(label_fontsize)
    ax.text(
        0.0,
        -0.06,
        left_lab,
        transform=trans,
        ha="left",
        va="top",
        fontsize=loc_fontsize,
        clip_on=False,
        zorder=40,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.55,
            "pad": 0.2,
        },
    )
    if not np.isclose(float(x0), float(x1)):
        ax.text(
            1.0,
            -0.06,
            right_lab,
            transform=trans,
            ha="right",
            va="top",
            fontsize=loc_fontsize,
            clip_on=False,
            zorder=40,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.2,
            },
        )


def _extract_ld_site_set(
    df: pd.DataFrame,
    chr_col: str,
    pos_col: str,
    p_col: str,
    threshold: float,
    use_all_sites: bool,
    no_logtrans: bool = False,
) -> set[tuple[str, int]]:
    pvals = pd.to_numeric(df[p_col], errors="coerce")
    pos = pd.to_numeric(df[pos_col], errors="coerce")
    mask_valid = pos.notna() & _postgwas_valid_p_mask(
        pvals,
        no_logtrans=no_logtrans,
    )
    if use_all_sites:
        mask = mask_valid
    else:
        mask = mask_valid & _postgwas_significant_p_mask(
            pvals,
            threshold,
            no_logtrans=no_logtrans,
        )
    out: set[tuple[str, int]] = set()
    if not bool(mask.any()):
        return out
    for c, p in zip(df.loc[mask, chr_col], pos.loc[mask]):
        out.add((_normalize_chr(c), int(round(float(p)))))
    return out


def _normalize_plink_prefix(path_or_prefix: object) -> str:
    s = str(path_or_prefix).strip()
    low = s.lower()
    for ext in (".bed", ".bim", ".fam"):
        if low.endswith(ext):
            return s[: -len(ext)]
    return s


def _is_existing_plink_prefix(path_or_prefix: object) -> bool:
    prefix = _normalize_plink_prefix(path_or_prefix)
    if prefix == "":
        return False
    return (
        os.path.isfile(f"{prefix}.bed")
        and os.path.isfile(f"{prefix}.bim")
        and os.path.isfile(f"{prefix}.fam")
    )


def _build_bimrange_lookup(
    bimrange_tuples: list[tuple[str, int, int]],
) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for chrom, start, end in bimrange_tuples:
        ck = _normalize_chr(chrom)
        s = int(min(start, end))
        e = int(max(start, end))
        out.setdefault(ck, []).append((s, e))
    return out


def _site_in_bimrange_lookup(
    chrom_norm: str,
    pos: int,
    bim_lookup: dict[str, list[tuple[int, int]]],
) -> bool:
    spans = bim_lookup.get(str(chrom_norm), [])
    for s, e in spans:
        if int(s) <= int(pos) <= int(e):
            return True
    return False


def _expand_bimranges_for_reader(
    bimrange_tuples: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for chrom, start, end in bimrange_tuples:
        s = int(min(start, end))
        e = int(max(start, end))
        raw = str(chrom).strip()
        norm = _normalize_chr(raw)
        candidates = [raw, norm]
        if norm != "":
            candidates.append(f"chr{norm}")
        for c in candidates:
            cc = str(c).strip()
            if cc == "":
                continue
            key = (cc, s, e)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _ld_r2_from_geno_rows(geno_rows: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(np.asarray(geno_rows, dtype=np.float32))
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    m = int(x.shape[0])
    if m == 1:
        return np.ones((1, 1), dtype=np.float32)

    x64 = np.asarray(x, dtype=np.float64)
    centered = x64 - x64.mean(axis=1, keepdims=True)
    gram = centered @ centered.T
    ss = np.einsum("ij,ij->i", centered, centered, dtype=np.float64, optimize=True)
    den = np.sqrt(np.outer(ss, ss))

    corr = np.zeros_like(gram, dtype=np.float64)
    valid = den > 0.0
    corr[valid] = gram[valid] / den[valid]
    corr = np.clip(corr, -1.0, 1.0)
    r2 = np.square(corr)
    np.fill_diagonal(r2, 1.0)
    r2 = np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(r2.astype(np.float32, copy=False))


def _compute_ld_from_genotype_generic(
    genofile: str,
    bimrange_tuples: list[tuple[str, int, int]],
    *,
    selected_sites: Optional[set[tuple[str, int]]] = None,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    if len(bimrange_tuples) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    bim_lookup = _build_bimrange_lookup(bimrange_tuples)
    wanted: Optional[set[tuple[str, int]]] = None
    if selected_sites is not None:
        picked = {
            (_normalize_chr(c), int(p))
            for c, p in selected_sites
            if _site_in_bimrange_lookup(_normalize_chr(c), int(p), bim_lookup)
        }
        if len(picked) == 0:
            return np.zeros((0, 0), dtype=np.float32), []
        wanted = picked

    reader_ranges = _expand_bimranges_for_reader(bimrange_tuples)
    if len(reader_ranges) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    chunk_size = 20_000
    if wanted is not None:
        chunk_size = max(1, min(20_000, len(wanted)))

    geno_chunks: list[np.ndarray] = []
    keys: list[tuple[str, int]] = []
    for chunk, sites in load_genotype_chunks(
        str(genofile),
        chunk_size=chunk_size,
        maf=0.0,
        missing_rate=1.0,
        impute=True,
        ranges=reader_ranges,
    ):
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue

        keep_idx: list[int] = []
        keep_keys: list[tuple[str, int]] = []
        for ridx, site in enumerate(sites):
            chrom_norm = _normalize_chr(getattr(site, "chrom", ""))
            try:
                pos = int(getattr(site, "pos"))
            except Exception:
                continue
            if not _site_in_bimrange_lookup(chrom_norm, pos, bim_lookup):
                continue
            key = (chrom_norm, pos)
            if wanted is not None and key not in wanted:
                continue
            keep_idx.append(int(ridx))
            keep_keys.append(key)

        if len(keep_idx) == 0:
            continue
        if len(keep_idx) == int(arr.shape[0]):
            geno_chunks.append(arr)
        else:
            idx_arr = np.asarray(keep_idx, dtype=np.int64)
            geno_chunks.append(arr[idx_arr, :])
        keys.extend(keep_keys)

    if len(keys) == 0 or len(geno_chunks) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    x = geno_chunks[0] if len(geno_chunks) == 1 else np.vstack(geno_chunks)
    if len(keys) > 1:
        seen: set[tuple[str, int]] = set()
        keep_rows: list[int] = []
        for i, key in enumerate(keys):
            if key in seen:
                continue
            seen.add(key)
            keep_rows.append(i)
        if len(keep_rows) != len(keys):
            idx_arr = np.asarray(keep_rows, dtype=np.int64)
            x = x[idx_arr, :]
            keys = [keys[i] for i in keep_rows]

    ld_mat = _ld_r2_from_geno_rows(x)
    return ld_mat, keys


def _compute_ld_from_bed_rust(
    genofile: str,
    bimrange_tuples: list[tuple[str, int, int]],
    *,
    selected_sites: Optional[set[tuple[str, int]]] = None,
    threads: int = 0,
    logger: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """
    Compute LD r2 matrix with automatic backend routing.

    Routing:
      1) If an existing PLINK prefix can be resolved and Rust API is available,
         use Rust backend (BLAS-first, bitwise fallback).
      2) Otherwise, fallback to generic genotype reader + NumPy correlation.

    Returns:
      ld_r2 (float32, m x m), ld_keys [(chrom_norm, pos), ...] in matrix order.
    """
    if len(bimrange_tuples) == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    source = str(genofile).strip()
    plink_prefix = _normalize_plink_prefix(source)
    if not _is_existing_plink_prefix(plink_prefix):
        plink_prefix = ""
        source_low = source.lower()
        delim = None
        if source_low.endswith(".csv") or os.path.isfile(f"{source}.csv"):
            delim = ","
        try:
            cached = prepare_cli_input_cache(
                source,
                snps_only=True,
                delimiter=delim,
                prefer_plink_for_txt=True,
                threads=int(threads),
            )
            cached_prefix = _normalize_plink_prefix(cached)
            if _is_existing_plink_prefix(cached_prefix):
                plink_prefix = cached_prefix
                if logger is not None and cached_prefix != source:
                    logger.info(
                        "LD backend: using cached PLINK prefix for Rust path: "
                        f"{format_path_for_display(cached_prefix)}"
                    )
        except Exception as ex:
            if logger is not None:
                logger.warning(
                    "Warning: failed to materialize PLINK cache for LD backend; "
                    f"falling back to generic path. Reason: {ex}"
                )
            plink_prefix = ""

    if plink_prefix != "" and hasattr(jxrs, "bed_ldblock_r2_rust"):
        chrom_ranges = [str(x[0]) for x in bimrange_tuples]
        start_bp = [int(x[1]) for x in bimrange_tuples]
        end_bp = [int(x[2]) for x in bimrange_tuples]

        kwargs: dict[str, object] = {}
        if selected_sites is not None:
            sel = sorted(
                [(_normalize_chr(c), int(p)) for (c, p) in selected_sites],
                key=lambda z: (str(z[0]), int(z[1])),
            )
            kwargs["selected_chrom"] = [str(x[0]) for x in sel]
            kwargs["selected_pos"] = [int(x[1]) for x in sel]

        ld_raw, chr_raw, pos_raw = jxrs.bed_ldblock_r2_rust(
            str(plink_prefix),
            chrom_ranges,
            start_bp,
            end_bp,
            threads=int(max(0, int(threads))),
            **kwargs,
        )
        ld_mat = np.ascontiguousarray(np.asarray(ld_raw, dtype=np.float32))
        ld_keys = [(_normalize_chr(c), int(p)) for c, p in zip(list(chr_raw), list(pos_raw))]
        if ld_mat.ndim != 2:
            return np.zeros((0, 0), dtype=np.float32), []
        if int(ld_mat.shape[0]) != int(ld_mat.shape[1]):
            raise RuntimeError(f"Rust LD matrix is not square: shape={ld_mat.shape}")
        if int(ld_mat.shape[0]) != int(len(ld_keys)):
            raise RuntimeError(
                f"Rust LD key count mismatch: matrix_n={ld_mat.shape[0]}, keys={len(ld_keys)}"
            )
        return ld_mat, ld_keys

    if logger is not None:
        logger.warning(
            "Warning: Rust LD matrix backend requires PLINK BED/BIM/FAM input; "
            "falling back to generic genotype LD computation."
        )
    return _compute_ld_from_genotype_generic(
        source,
        bimrange_tuples,
        selected_sites=selected_sites,
    )


def _format_bimrange_title(item: tuple[str, int, int]) -> str:
    chrom, start, end = item
    # return _sanitize_plot_text(f"{chrom}:{start / 1_000_000:g}-{end / 1_000_000:g}")
    return _sanitize_plot_text(f"")


def _ld_bimrange_spans(
    ld_keys: list[tuple[str, int]],
    bimrange_tuples: list[tuple[str, int, int]],
) -> list[dict[str, object]]:
    """
    Build LD index spans for each bimrange.
    x coordinate in LD panel:
      SNP i center -> x = i + 0.5
    """
    if len(ld_keys) == 0 or len(bimrange_tuples) == 0:
        return []

    spans_meta: list[dict[str, object]] = []
    for sid, (chrom, start, end) in enumerate(bimrange_tuples):
        spans_meta.append(
            {
                "sid": int(sid),
                "chrom_norm": _normalize_chr(chrom),
                "chrom": str(chrom),
                "start": int(start),
                "end": int(end),
                "first": None,
                "last": None,
                "count": 0,
            }
        )

    for idx, (k_chrom, k_pos) in enumerate(ld_keys):
        kk = str(k_chrom)
        pp = int(k_pos)
        hit_sid: Optional[int] = None
        for s in spans_meta:
            if kk != str(s["chrom_norm"]):
                continue
            if int(s["start"]) <= pp <= int(s["end"]):
                hit_sid = int(s["sid"])
                break
        if hit_sid is None:
            continue
        seg = spans_meta[hit_sid]
        if seg["first"] is None:
            seg["first"] = int(idx)
        seg["last"] = int(idx)
        seg["count"] = int(seg["count"]) + 1

    out: list[dict[str, object]] = []
    for s in spans_meta:
        if int(s["count"]) <= 0:
            continue
        first = int(s["first"])
        last = int(s["last"])
        x_start = float(first) + 0.5
        x_end = float(last) + 0.5
        out.append(
            {
                "sid": int(s["sid"]),
                "chrom": str(s["chrom"]),
                "start": int(s["start"]),
                "end": int(s["end"]),
                "count": int(s["count"]),
                "first": int(first),
                "last": int(last),
                "x_start": float(x_start),
                "x_end": float(x_end),
                "x_center": 0.5 * (float(x_start) + float(x_end)),
                "label": _format_bimrange_title(
                    (str(s["chrom"]), int(s["start"]), int(s["end"]))
                ),
            }
        )
    return out


def _project_gene_records_to_ld_spans(
    records: pd.DataFrame,
    bimrange_tuples: list[tuple[str, int, int]],
    ld_spans: list[dict[str, object]],
) -> pd.DataFrame:
    """
    Project gene records to LD-index x space so gene track aligns with LD triangle.
    """
    out_cols = ["feature", "strand", "attribute", "x_start", "x_end"]
    if records.shape[0] == 0 or len(bimrange_tuples) == 0 or len(ld_spans) == 0:
        return pd.DataFrame(columns=out_cols)

    span_by_sid: dict[int, dict[str, object]] = {
        int(s["sid"]): s for s in ld_spans if int(s.get("count", 0)) > 0
    }
    rows: list[dict[str, object]] = []
    for _, row in records.iterrows():
        chrom = _normalize_chr(row["chrom_norm"])
        r_start = int(min(int(row["start"]), int(row["end"])))
        r_end = int(max(int(row["start"]), int(row["end"])))
        feature = str(row["feature"])
        strand = str(row["strand"])
        attr = row["attribute"]
        for sid, (bchrom, bstart, bend) in enumerate(bimrange_tuples):
            if chrom != _normalize_chr(bchrom):
                continue
            ov_start = max(r_start, int(bstart))
            ov_end = min(r_end, int(bend))
            if ov_end < ov_start:
                continue
            span = span_by_sid.get(int(sid))
            if span is None:
                continue

            seg_start_bp = int(bstart)
            seg_end_bp = int(bend)
            seg_len_bp = max(1, int(seg_end_bp - seg_start_bp))
            x0 = float(span["x_start"])
            x1 = float(span["x_end"])
            # Keep one-SNP segments drawable.
            if np.isclose(x0, x1):
                x0 -= 0.45
                x1 += 0.45
            scale = (x1 - x0) / float(seg_len_bp)
            gx0 = x0 + float(ov_start - seg_start_bp) * scale
            gx1 = x0 + float(ov_end - seg_start_bp) * scale
            rows.append(
                {
                    "feature": feature,
                    "strand": strand,
                    "attribute": attr,
                    "x_start": float(min(gx0, gx1)),
                    "x_end": float(max(gx0, gx1)),
                }
            )
    if len(rows) == 0:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows, columns=out_cols)


def _draw_ld_bimrange_titles(
    ax: plt.Axes,
    spans: list[dict[str, object]],
    *,
    enabled: bool,
    font_size: float = _POSTGWAS_DEFAULT_FONT_SIZE,
) -> None:
    if not enabled or len(spans) == 0:
        return
    trans = ax.get_xaxis_transform()
    for i, seg in enumerate(spans):
        ax.text(
            float(seg["x_center"]),
            1.015,
            _sanitize_plot_text(seg["label"]),
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=float(font_size),
            clip_on=False,
            zorder=40,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 0.2,
            },
        )
        if i > 0:
            prev = spans[i - 1]
            xb = 0.5 * (float(prev["x_end"]) + float(seg["x_start"]))
            ax.axvline(
                x=float(xb),
                color="black",
                linestyle=":",
                linewidth=0.7,
                alpha=0.75,
                zorder=5,
            )


def _lead_vs_all_r2(
    geno_block: np.ndarray,
    *,
    row_sums: Optional[np.ndarray] = None,
    row_sumsq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute r^2 between the lead SNP (row 0) and all SNP rows.

    This avoids building a full correlation matrix and is much cheaper for
    LD clumping where only lead-vs-window correlations are needed.
    """
    x = np.asarray(geno_block, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    m, n = x.shape
    if n <= 1:
        out = np.zeros((m,), dtype=np.float32)
        out[0] = 1.0
        return out

    x0 = x[0]  # lead SNP
    n_f = float(n)

    # Pearson correlation via sum products:
    # r = (n*sum(xy)-sum(x)sum(y)) / sqrt((n*sum(x^2)-sum(x)^2)*(n*sum(y^2)-sum(y)^2))
    if row_sums is None:
        sx = np.sum(x, axis=1, dtype=np.float64)
    else:
        sx = np.asarray(row_sums, dtype=np.float64).reshape(-1)
    if row_sumsq is None:
        sx2 = np.einsum("ij,ij->i", x, x, dtype=np.float64, optimize=True)
    else:
        sx2 = np.asarray(row_sumsq, dtype=np.float64).reshape(-1)
    if sx.shape != (m,) or sx2.shape != (m,):
        raise ValueError(
            "Precomputed genotype row statistics must match the candidate block."
        )
    s0 = float(sx[0])
    s02 = float(sx2[0])
    sxx0 = np.einsum("ij,j->i", x, x0, dtype=np.float64, optimize=True)

    num = n_f * sxx0 - sx * s0
    den_left = n_f * sx2 - np.square(sx)
    den_right = max(0.0, n_f * s02 - s0 * s0)
    den = np.sqrt(np.maximum(den_left, 0.0) * den_right)

    corr = np.zeros((m,), dtype=np.float64)
    valid = den > 0.0
    corr[valid] = num[valid] / den[valid]
    corr = np.clip(corr, -1.0, 1.0)
    corr[0] = 1.0

    r2 = np.square(corr)
    r2 = np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0)
    return r2.astype(np.float32, copy=False)


def _clean_anno_token(value: object) -> str:
    if value is None:
        return "NA"
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    # Normalize simple serialized list-like wrappers, e.g. "['geneA']".
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if inner != "" and "," not in inner:
            text = inner
    if len(text) >= 2 and ((text[0] == "'" and text[-1] == "'") or (text[0] == '"' and text[-1] == '"')):
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    if text == "" or text.lower() == "nan":
        return "NA"
    return text


def _merge_anno_value(base: str, value: str) -> str:
    if value == "NA":
        return base
    if base == "NA":
        return value
    existing = base.split("|")
    if value in existing:
        return base
    return f"{base}|{value}"


def _format_gene_annotation_dict(hits: pd.DataFrame) -> str:
    """
    Format annotation hits as:
      gene1:description/additionaldesc;gene2:description/additionaldesc
    """
    if hits is None or hits.shape[0] == 0:
        return "NA"
    out: dict[str, list[str]] = {}
    for _, row in hits.iterrows():
        gene = _clean_anno_token(row.iloc[3] if hits.shape[1] > 3 else "NA")
        desc = _clean_anno_token(row.iloc[4] if hits.shape[1] > 4 else "NA")
        add_desc = _clean_anno_token(row.iloc[5] if hits.shape[1] > 5 else "NA")
        if gene not in out:
            out[gene] = [desc, add_desc]
        else:
            out[gene][0] = _merge_anno_value(out[gene][0], desc)
            out[gene][1] = _merge_anno_value(out[gene][1], add_desc)
    if len(out) == 0:
        return "NA"
    return ";".join(
        [f"{gene}:{vals[0]}/{vals[1]}" for gene, vals in out.items()]
    )


def _anno_token_is_numeric(value: object) -> bool:
    text = _clean_anno_token(value)
    if text == "NA":
        return False
    try:
        float(text)
    except Exception:
        return False
    return True


def _normalize_bed_annotation_label(value: object) -> str:
    text = _clean_anno_token(value)
    if text == "NA":
        return ""
    compact = re.sub(r"[^0-9A-Za-z]+", "", text).lower()
    mapping = {
        "cds": "CDS",
        "gene": "Gene",
        "exon": "Exon",
        "intron": "Intron",
        "fiveprimeutr": "FivePrimeUTR",
        "fiveutr": "FivePrimeUTR",
        "utr5": "FivePrimeUTR",
        "threeprimeutr": "ThreePrimeUTR",
        "threeutr": "ThreePrimeUTR",
        "utr3": "ThreePrimeUTR",
        "upstream": "Upstream2kb",
        "upstream2kb": "Upstream2kb",
        "downstream": "Downstream2kb",
        "downstream2kb": "Downstream2kb",
        "intergenic": "Intergenic",
    }
    return str(mapping.get(compact, ""))


def _resolve_bed_annotation_row_meta(
    row: pd.Series,
    *,
    chrom_norm: str,
    start: int,
    end: int,
) -> tuple[str, str, str, str]:
    extras = [
        _clean_anno_token(row.iloc[idx])
        for idx in range(3, int(len(row)))
    ]
    feature_label = ""
    strand = "."
    named_tokens: list[str] = []
    first_extra_special = False
    for idx, text in enumerate(extras):
        label = _normalize_bed_annotation_label(text)
        is_special = bool(label != "" or text in {"+", "-"} or _anno_token_is_numeric(text))
        if idx == 0:
            first_extra_special = bool(is_special)
        if label != "" and feature_label == "":
            feature_label = str(label)
            continue
        if text in {"+", "-"} and strand == ".":
            strand = str(text)
            continue
        if text == "NA" or _anno_token_is_numeric(text):
            continue
        named_tokens.append(str(text))

    if feature_label == "":
        feature_label = "Gene"

    if first_extra_special:
        if feature_label != "Intergenic" and len(named_tokens) >= 2:
            gene_id = str(named_tokens[0])
            desc = " ".join([str(x) for x in named_tokens[1:]]).strip()
        else:
            gene_id = "NA" if feature_label == "Intergenic" else f"{chrom_norm}:{int(start)}-{int(end)}"
            desc = str(named_tokens[-1]) if len(named_tokens) > 0 else ("NA" if gene_id == "NA" else gene_id)
    else:
        gene_id = str(named_tokens[0]) if len(named_tokens) > 0 else f"{chrom_norm}:{int(start)}-{int(end)}"
        desc = (
            " ".join([str(x) for x in named_tokens[1:]]).strip()
            if len(named_tokens) >= 2
            else str(gene_id)
        )

    gene_id = _clean_anno_token(gene_id)
    desc = _clean_anno_token(desc)
    if feature_label == "Intergenic":
        gene_id = "NA"
    return feature_label, gene_id, desc, str(strand)


def _extract_gff_attr_series(attr_series: pd.Series, key: str) -> pd.Series:
    key_pat = re.escape(str(key))
    pattern = rf"(?:^|;|\s){key_pat}=([^;]*?)(?=;|\s+[^\s;=]+=|$)"
    out = attr_series.astype(str).str.extract(pattern, expand=False).fillna("NA")
    return out.map(_clean_anno_token)


def _get_gff_attr_series(gff: pd.DataFrame, key: str) -> pd.Series:
    col = _gff_prefetched_attr_colname(key)
    if col != "" and col in gff.columns:
        base = pd.Series(gff[col].astype(object), index=gff.index, copy=False)
        return base.map(_clean_anno_token)
    if "attributes" not in gff.columns:
        return pd.Series("NA", index=gff.index, dtype=object)
    return _extract_gff_attr_series(gff["attributes"], key)


def _normalize_gff_entity_id(value: object) -> str:
    text = _clean_anno_token(value)
    if text == "NA":
        return "NA"
    return re.sub(r"^[^:]*:", "", text)


def _split_gff_entity_ids(value: object) -> tuple[str, ...]:
    text = _clean_anno_token(value)
    if text == "NA":
        return tuple()
    out: list[str] = []
    seen: set[str] = set()
    for token in str(text).split(","):
        norm = _normalize_gff_entity_id(token)
        if norm == "NA" or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return tuple(out)


def _build_postgwas_gff_annotation_context(gff_query: GFFQuery) -> dict[str, object]:
    gff = gff_query.gff
    if gff.shape[0] == 0:
        return {
            "gene_meta": {},
            "row_gene_ids": {},
            "feature_index_by_chr": {},
            "gene_index_by_chr": {},
        }

    ids = _get_gff_attr_series(gff, "ID").map(_normalize_gff_entity_id)
    parents = _get_gff_attr_series(gff, "Parent").map(_split_gff_entity_ids)
    descs = _get_gff_attr_series(gff, _ANNO_DESC_KEY)
    names = _get_gff_attr_series(gff, "Name")
    features = gff["feature"].astype(str).str.strip().str.lower()

    gene_meta: dict[str, dict[str, object]] = {}
    id_to_parents: dict[str, tuple[str, ...]] = {}

    for idx in gff.index:
        feature = str(features.loc[idx]).strip().lower()
        feature_id = str(ids.loc[idx]).strip()
        parent_ids = tuple(parents.loc[idx]) if idx in parents.index else tuple()
        if feature_id != "NA":
            if len(parent_ids) > 0:
                id_to_parents[feature_id] = parent_ids
        if feature != "gene" or feature_id == "NA":
            continue
        desc = str(descs.loc[idx]).strip()
        if desc == "NA":
            desc = str(names.loc[idx]).strip()
        gene_meta[feature_id] = {
            "chrom_norm": _normalize_chr(gff.loc[idx, "chrom_norm"]),
            "start": int(gff.loc[idx, "start"]),
            "end": int(gff.loc[idx, "end"]),
            "strand": _clean_anno_token(gff.loc[idx, "strand"]),
            "desc": _clean_anno_token(desc),
        }

    _resolve_gene_ids = _build_postgwas_gene_id_resolver(
        gene_meta=gene_meta,
        id_to_parents=id_to_parents,
    )

    row_gene_ids: dict[int, tuple[str, ...]] = {}
    for idx in gff.index:
        feature = str(features.loc[idx]).strip().lower()
        feature_id = str(ids.loc[idx]).strip()
        candidate_ids: list[str] = []
        if feature == "gene" and feature_id in gene_meta:
            row_gene_ids[int(idx)] = (feature_id,)
            continue
        if feature_id != "NA":
            candidate_ids.append(feature_id)
        candidate_ids.extend(list(parents.loc[idx]) if idx in parents.index else [])
        out: list[str] = []
        seen: set[str] = set()
        for candidate in candidate_ids:
            for gene_id in _resolve_gene_ids(candidate):
                if gene_id in seen or gene_id not in gene_meta:
                    continue
                seen.add(gene_id)
                out.append(gene_id)
        row_gene_ids[int(idx)] = tuple(out)

    feature_index_by_chr: dict[str, dict[str, np.ndarray]] = {}
    for chrom_norm, block in gff.groupby("chrom_norm", sort=False, observed=False):
        block_sorted = block.sort_values(["start", "end"], kind="mergesort")
        block_idx = block_sorted.index.to_numpy(dtype=np.int64, copy=False)
        block_gene_ids = np.empty((int(block_sorted.shape[0]),), dtype=object)
        for arr_idx, row_idx in enumerate(block_idx.tolist()):
            block_gene_ids[int(arr_idx)] = row_gene_ids.get(int(row_idx), tuple())
        feature_index_by_chr[str(chrom_norm)] = {
            "starts": block_sorted["start"].to_numpy(dtype=np.int64, copy=False),
            "ends": block_sorted["end"].to_numpy(dtype=np.int64, copy=False),
            "features": (
                block_sorted["feature"].astype(str).str.strip().str.lower().to_numpy(dtype=object, copy=False)
            ),
            "row_gene_ids": block_gene_ids,
        }

    gene_index_by_chr: dict[str, dict[str, np.ndarray]] = {}
    genes_by_chr: dict[str, list[tuple[int, int, str, str, str]]] = {}
    for gene_id, meta in gene_meta.items():
        chrom_norm = str(meta.get("chrom_norm", "")).strip()
        genes_by_chr.setdefault(chrom_norm, []).append(
            (
                int(meta.get("start", 0)),
                int(meta.get("end", 0)),
                str(gene_id),
                _clean_anno_token(meta.get("strand", ".")),
                _clean_anno_token(meta.get("desc", "NA")),
            )
        )
    for chrom_norm, records in genes_by_chr.items():
        records_sorted = sorted(records, key=lambda x: (int(x[0]), int(x[1]), str(x[2])))
        gene_index_by_chr[str(chrom_norm)] = {
            "starts": np.asarray([int(x[0]) for x in records_sorted], dtype=np.int64),
            "ends": np.asarray([int(x[1]) for x in records_sorted], dtype=np.int64),
            "gene_ids": np.asarray([str(x[2]) for x in records_sorted], dtype=object),
            "strands": np.asarray([str(x[3]) for x in records_sorted], dtype=object),
            "descs": np.asarray([str(x[4]) for x in records_sorted], dtype=object),
        }

    return {
        "gene_meta": gene_meta,
        "row_gene_ids": row_gene_ids,
        "feature_index_by_chr": feature_index_by_chr,
        "gene_index_by_chr": gene_index_by_chr,
    }


def _postgwas_choose_exact_gff_label(features: set[str]) -> str:
    feature_set = {str(x).strip().lower() for x in features if str(x).strip() != ""}
    if "cds" in feature_set:
        return "CDS"
    if "five_prime_utr" in feature_set:
        return "FivePrimeUTR"
    if "three_prime_utr" in feature_set:
        return "ThreePrimeUTR"
    if "exon" in feature_set:
        return "Exon"
    if "intron" in feature_set:
        return "Intron"
    return "Intron"


def _postgwas_choose_exact_bed_label(labels: set[str]) -> str:
    label_set = {str(x).strip().lower() for x in labels if str(x).strip() != ""}
    if "cds" in label_set:
        return "CDS"
    if "fiveprimeutr" in label_set:
        return "FivePrimeUTR"
    if "threeprimeutr" in label_set:
        return "ThreePrimeUTR"
    if "exon" in label_set:
        return "Exon"
    if "intron" in label_set:
        return "Intron"
    if "gene" in label_set:
        return "Gene"
    if "upstream2kb" in label_set:
        return "Upstream2kb"
    if "downstream2kb" in label_set:
        return "Downstream2kb"
    if "intergenic" in label_set:
        return "Intergenic"
    return "Gene"


def _postgwas_annotation_triplet(label: object, gene_id: object, desc: object) -> str:
    return (
        f"{_clean_anno_token(label)};"
        f"{_normalize_gff_entity_id(gene_id)};"
        f"{_clean_anno_token(desc)}"
    )


def _join_postgwas_annotation_entries(entries: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        text = str(entry).strip()
        if text == "" or text in seen:
            continue
        seen.add(text)
        out.append(text)
    if len(out) == 0:
        return "Intergenic;NA;NA"
    return " | ".join(out)


def _postgwas_attr_list_value(values: object, idx: int) -> str:
    if isinstance(values, (list, tuple)) and len(values) > int(idx):
        return _clean_anno_token(values[int(idx)])
    return "NA"


def _build_postgwas_gene_id_resolver(
    *,
    gene_meta: dict[str, dict[str, object]],
    id_to_parents: dict[str, tuple[str, ...]],
):
    @lru_cache(maxsize=None)
    def _resolve_gene_ids(feature_id: str) -> tuple[str, ...]:
        norm_id = _normalize_gff_entity_id(feature_id)
        if norm_id == "NA":
            return tuple()
        if norm_id in gene_meta:
            return (norm_id,)

        out: list[str] = []
        seen_genes: set[str] = set()
        seen_nodes: set[str] = set()
        stack: list[str] = [norm_id]

        while len(stack) > 0:
            node_id = _normalize_gff_entity_id(stack.pop())
            if node_id == "NA":
                continue
            if node_id in gene_meta:
                if node_id not in seen_genes:
                    seen_genes.add(node_id)
                    out.append(node_id)
                continue
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)

            parent_ids = tuple(id_to_parents.get(node_id, tuple()))
            for parent_id in reversed(parent_ids):
                parent_norm = _normalize_gff_entity_id(parent_id)
                if parent_norm == "NA":
                    continue
                if parent_norm in seen_nodes:
                    continue
                stack.append(parent_norm)

        return tuple(out)

    return _resolve_gene_ids


def _build_postgwas_local_gff_context(hit: pd.DataFrame) -> tuple[
    dict[str, dict[str, object]],
    dict[int, tuple[str, ...]],
]:
    if hit is None or hit.shape[0] == 0 or "attribute" not in hit.columns:
        return {}, {}

    features = hit["feature"].astype(str).str.strip().str.lower()
    attrs = hit["attribute"]
    ids = attrs.map(lambda x: _normalize_gff_entity_id(_postgwas_attr_list_value(x, 0)))
    parents = attrs.map(lambda x: _split_gff_entity_ids(_postgwas_attr_list_value(x, 1)))
    descs = attrs.map(lambda x: _clean_anno_token(_postgwas_attr_list_value(x, 2)))
    names = attrs.map(lambda x: _clean_anno_token(_postgwas_attr_list_value(x, 3)))

    gene_meta: dict[str, dict[str, object]] = {}
    id_to_parents: dict[str, tuple[str, ...]] = {}
    for idx in hit.index:
        feature = str(features.loc[idx]).strip().lower()
        feature_id = str(ids.loc[idx]).strip()
        parent_ids = tuple(parents.loc[idx]) if idx in parents.index else tuple()
        if feature_id != "NA" and len(parent_ids) > 0:
            id_to_parents[feature_id] = parent_ids
        if feature != "gene" or feature_id == "NA":
            continue
        desc = str(descs.loc[idx]).strip()
        if desc == "NA":
            desc = str(names.loc[idx]).strip()
        gene_meta[feature_id] = {
            "start": int(hit.loc[idx, "start"]),
            "end": int(hit.loc[idx, "end"]),
            "strand": _clean_anno_token(hit.loc[idx, "strand"]),
            "desc": _clean_anno_token(desc),
        }

    _resolve_gene_ids_local = _build_postgwas_gene_id_resolver(
        gene_meta=gene_meta,
        id_to_parents=id_to_parents,
    )

    row_gene_ids: dict[int, tuple[str, ...]] = {}
    for idx in hit.index:
        feature = str(features.loc[idx]).strip().lower()
        feature_id = str(ids.loc[idx]).strip()
        if feature == "gene" and feature_id in gene_meta:
            row_gene_ids[int(idx)] = (feature_id,)
            continue
        candidate_ids: list[str] = []
        if feature_id != "NA":
            candidate_ids.append(feature_id)
        candidate_ids.extend(list(parents.loc[idx]) if idx in parents.index else [])
        out: list[str] = []
        seen: set[str] = set()
        for candidate in candidate_ids:
            for gene_id in _resolve_gene_ids_local(candidate):
                if gene_id in seen or gene_id not in gene_meta:
                    continue
                seen.add(gene_id)
                out.append(gene_id)
        row_gene_ids[int(idx)] = tuple(out)
    return gene_meta, row_gene_ids


def _postgwas_take_row_values_object(
    gff: pd.DataFrame,
    column: str,
    rowids: np.ndarray,
    *,
    na_value: object = "NA",
) -> np.ndarray:
    rowids_arr = np.asarray(rowids, dtype=np.int64)
    if rowids_arr.size == 0:
        return np.empty((0,), dtype=object)
    series = gff[column]
    values = getattr(series, "_values", None)
    if isinstance(values, pd.Categorical):
        codes = values.codes[rowids_arr]
        cats = np.asarray(values.categories, dtype=object)
        out = np.empty((int(rowids_arr.size),), dtype=object)
        valid = codes >= 0
        if bool(np.any(valid)):
            out[valid] = cats[codes[valid]]
        if bool(np.any(~valid)):
            out[~valid] = na_value
        return out
    arr = series.to_numpy(dtype=object, na_value=na_value)
    out = arr[rowids_arr]
    if na_value is not None and out.size > 0:
        empty_mask = out == ""
        if bool(np.any(empty_mask)):
            out = out.copy()
            out[empty_mask] = na_value
    return out


def _build_postgwas_local_gff_context_arrays(
    *,
    features: np.ndarray,
    feature_ids: np.ndarray,
    parents_text: np.ndarray,
    descs: np.ndarray,
    names: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    strands: np.ndarray,
) -> tuple[dict[str, dict[str, object]], list[tuple[str, ...]]]:
    gene_meta: dict[str, dict[str, object]] = {}
    id_to_parents: dict[str, tuple[str, ...]] = {}
    parent_tuples: list[tuple[str, ...]] = []

    n_rows = int(features.shape[0])
    for idx in range(n_rows):
        feature = str(features[idx]).strip().lower()
        feature_id = _normalize_gff_entity_id(feature_ids[idx])
        parent_ids = _split_gff_entity_ids(parents_text[idx])
        parent_tuples.append(parent_ids)
        if feature_id != "NA" and len(parent_ids) > 0:
            id_to_parents[feature_id] = parent_ids
        if feature != "gene" or feature_id == "NA":
            continue
        desc = _clean_anno_token(descs[idx])
        if desc == "NA":
            desc = _clean_anno_token(names[idx])
        gene_meta[feature_id] = {
            "start": int(starts[idx]),
            "end": int(ends[idx]),
            "strand": _clean_anno_token(strands[idx]),
            "desc": _clean_anno_token(desc),
        }

    resolve_gene_ids = _build_postgwas_gene_id_resolver(
        gene_meta=gene_meta,
        id_to_parents=id_to_parents,
    )

    row_gene_ids: list[tuple[str, ...]] = [tuple() for _ in range(n_rows)]
    for idx in range(n_rows):
        feature = str(features[idx]).strip().lower()
        feature_id = _normalize_gff_entity_id(feature_ids[idx])
        if feature == "gene" and feature_id in gene_meta:
            row_gene_ids[idx] = (feature_id,)
            continue
        candidate_ids: list[str] = []
        if feature_id != "NA":
            candidate_ids.append(feature_id)
        candidate_ids.extend(list(parent_tuples[idx]))
        out: list[str] = []
        seen: set[str] = set()
        for candidate in candidate_ids:
            for gene_id in resolve_gene_ids(candidate):
                if gene_id in seen or gene_id not in gene_meta:
                    continue
                seen.add(gene_id)
                out.append(gene_id)
        row_gene_ids[idx] = tuple(out)
    return gene_meta, row_gene_ids


def _build_postgwas_bed_annotation_context(anno: pd.DataFrame) -> dict[str, object]:
    if anno is None or anno.shape[0] == 0:
        return {
            "gene_meta": {},
            "feature_index_by_chr": {},
            "gene_index_by_chr": {},
        }

    work = anno.copy()
    work[0] = work[0].astype(str).map(_normalize_chr)
    work[1] = pd.to_numeric(work[1], errors="coerce")
    work[2] = pd.to_numeric(work[2], errors="coerce")
    work = work.dropna(subset=[0, 1, 2]).copy()
    if work.shape[0] == 0:
        return {
            "gene_meta": {},
            "feature_index_by_chr": {},
            "gene_index_by_chr": {},
        }
    work[1] = work[1].astype(int)
    work[2] = work[2].astype(int)

    feature_rows_by_chr: dict[str, list[tuple[int, int, str, str, str]]] = {}
    gene_meta: dict[str, dict[str, object]] = {}

    for _, row in work.iterrows():
        chrom_norm = str(row.iloc[0]).strip()
        start = int(row.iloc[1])
        end = int(row.iloc[2])
        feature_label, gene_id, desc, strand = _resolve_bed_annotation_row_meta(
            row,
            chrom_norm=chrom_norm,
            start=start,
            end=end,
        )
        feature_rows_by_chr.setdefault(chrom_norm, []).append(
            (int(start), int(end), str(feature_label), str(gene_id), str(desc))
        )

        if feature_label in {"Upstream2kb", "Downstream2kb", "Intergenic"}:
            continue
        if str(gene_id).strip() in {"", "NA"}:
            continue
        meta = gene_meta.get(str(gene_id))
        if meta is None:
            gene_meta[str(gene_id)] = {
                "chrom_norm": str(chrom_norm),
                "start": int(start),
                "end": int(end),
                "strand": str(strand),
                "desc": str(desc),
            }
        else:
            meta["start"] = int(min(int(meta.get("start", start)), int(start)))
            meta["end"] = int(max(int(meta.get("end", end)), int(end)))
            if _clean_anno_token(meta.get("strand", ".")) not in {"+", "-"} and str(strand) in {"+", "-"}:
                meta["strand"] = str(strand)
            meta["desc"] = _merge_anno_value(
                _clean_anno_token(meta.get("desc", "NA")),
                _clean_anno_token(desc),
            )

    feature_index_by_chr: dict[str, dict[str, np.ndarray]] = {}
    for chrom_norm, records in feature_rows_by_chr.items():
        records_sorted = sorted(records, key=lambda x: (int(x[0]), int(x[1]), str(x[3]), str(x[2]), str(x[4])))
        feature_index_by_chr[str(chrom_norm)] = {
            "starts": np.asarray([int(x[0]) for x in records_sorted], dtype=np.int64),
            "ends": np.asarray([int(x[1]) for x in records_sorted], dtype=np.int64),
            "labels": np.asarray([str(x[2]) for x in records_sorted], dtype=object),
            "gene_ids": np.asarray([str(x[3]) for x in records_sorted], dtype=object),
            "descs": np.asarray([str(x[4]) for x in records_sorted], dtype=object),
        }

    gene_index_by_chr: dict[str, dict[str, np.ndarray]] = {}
    genes_by_chr: dict[str, list[tuple[int, int, str, str, str]]] = {}
    for gene_id, meta in gene_meta.items():
        chrom_norm = str(meta.get("chrom_norm", "")).strip()
        genes_by_chr.setdefault(chrom_norm, []).append(
            (
                int(meta.get("start", 0)),
                int(meta.get("end", 0)),
                str(gene_id),
                _clean_anno_token(meta.get("strand", ".")),
                _clean_anno_token(meta.get("desc", "NA")),
            )
        )
    for chrom_norm, records in genes_by_chr.items():
        records_sorted = sorted(records, key=lambda x: (int(x[0]), int(x[1]), str(x[2])))
        gene_index_by_chr[str(chrom_norm)] = {
            "starts": np.asarray([int(x[0]) for x in records_sorted], dtype=np.int64),
            "ends": np.asarray([int(x[1]) for x in records_sorted], dtype=np.int64),
            "gene_ids": np.asarray([str(x[2]) for x in records_sorted], dtype=object),
            "strands": np.asarray([str(x[3]) for x in records_sorted], dtype=object),
            "descs": np.asarray([str(x[4]) for x in records_sorted], dtype=object),
        }

    return {
        "gene_meta": gene_meta,
        "feature_index_by_chr": feature_index_by_chr,
        "gene_index_by_chr": gene_index_by_chr,
    }


def _format_postgwas_gff_site_desc_direct_slow(
    *,
    chrom: object,
    pos: object,
    gff_query: GFFQuery,
    flank_bp: int = 2_000,
) -> str:
    try:
        pos_int = int(pos)
    except Exception:
        return "Intergenic;NA;NA"

    exact_hit = gff_query.query_range(
        chrom,
        pos_int,
        pos_int,
        features=None,
        attr=("ID", "Parent", _ANNO_DESC_KEY, "Name"),
    )
    gene_meta, row_gene_ids = _build_postgwas_local_gff_context(exact_hit)
    gene_features: dict[str, set[str]] = {}
    gene_order: list[str] = []
    for idx, row in exact_hit.iterrows():
        feature = str(row.get("feature", "")).strip().lower()
        if feature == "":
            continue
        for gene_id in row_gene_ids.get(int(idx), tuple()):
            if gene_id not in gene_meta:
                continue
            if gene_id not in gene_features:
                gene_features[gene_id] = set()
                gene_order.append(gene_id)
            gene_features[gene_id].add(feature)

    entries: list[str] = []
    exact_gene_ids = set(gene_features)
    for gene_id in gene_order:
        meta = gene_meta.get(gene_id, {})
        entries.append(
            _postgwas_annotation_triplet(
                _postgwas_choose_exact_gff_label(gene_features.get(gene_id, set())),
                gene_id,
                meta.get("desc", "NA"),
            )
        )

    nearby_hit = gff_query.query_range(
        chrom,
        max(0, pos_int - max(0, int(flank_bp))),
        pos_int + max(0, int(flank_bp)),
        features=None,
        attr=("ID", _ANNO_DESC_KEY, "Name"),
    )
    nearby_hit = nearby_hit.loc[
        nearby_hit["feature"].astype(str).str.strip().str.lower() == "gene"
    ].copy()
    nearby_records: list[tuple[int, int, str, str]] = []
    for _, row in nearby_hit.iterrows():
        attrs = row.get("attribute", [])
        gene_id = _normalize_gff_entity_id(_postgwas_attr_list_value(attrs, 0))
        if gene_id == "NA" or gene_id in exact_gene_ids:
            continue
        start = int(row["start"])
        end = int(row["end"])
        strand = _clean_anno_token(row.get("strand", "."))
        if pos_int < start:
            dist = int(start - pos_int)
            if dist > int(flank_bp):
                continue
            label = "Upstream2kb" if strand != "-" else "Downstream2kb"
        elif pos_int > end:
            dist = int(pos_int - end)
            if dist > int(flank_bp):
                continue
            label = "Downstream2kb" if strand != "-" else "Upstream2kb"
        else:
            continue
        nearby_records.append((int(dist), int(start), gene_id, str(label)))

    nearby_records.sort(key=lambda x: (int(x[0]), int(x[1]), str(x[2])))
    for _dist, _start, gene_id, label in nearby_records:
        entries.append(_postgwas_annotation_triplet(label, gene_id, "NA"))
    return _join_postgwas_annotation_entries(entries)


def _format_postgwas_gff_site_desc_direct(
    *,
    chrom: object,
    pos: object,
    gff_query: Optional[GFFQuery] = None,
    gff_rust_index: Optional[object] = None,
    flank_bp: int = 2_000,
) -> str:
    try:
        pos_int = int(pos)
    except Exception:
        return "Intergenic;NA;NA"
    if gff_rust_index is not None:
        try:
            return str(
                gff_rust_index.annotate_site_desc(
                    str(chrom),
                    int(pos_int),
                    int(flank_bp),
                )
            )
        except Exception:
            pass
    if gff_query is None:
        return "Intergenic;NA;NA"

    chrom_norm = _normalize_chr(chrom)
    hit = getattr(gff_query, "_chr_index", {}).get(chrom_norm)
    gff = getattr(gff_query, "gff", None)
    required_cols = (
        _gff_prefetched_attr_colname("ID"),
        _gff_prefetched_attr_colname("Parent"),
        _gff_prefetched_attr_colname(_ANNO_DESC_KEY),
        _gff_prefetched_attr_colname("Name"),
        "strand",
    )
    if (
        hit is None
        or gff is None
        or any(str(col) not in gff.columns for col in required_cols)
    ):
        return _format_postgwas_gff_site_desc_direct_slow(
            chrom=chrom,
            pos=pos,
            gff_query=gff_query,
            flank_bp=flank_bp,
        )

    starts = np.asarray(hit.get("starts", np.asarray([], dtype=np.int64)), dtype=np.int64)
    ends = np.asarray(hit.get("ends", np.asarray([], dtype=np.int64)), dtype=np.int64)
    features = np.asarray(hit.get("features", np.asarray([], dtype=object)), dtype=object)
    rowids = np.asarray(hit.get("rowids", np.asarray([], dtype=np.int64)), dtype=np.int64)
    if starts.size == 0 or ends.size == 0 or features.size == 0 or rowids.size == 0:
        return "Intergenic;NA;NA"

    hi_exact = int(np.searchsorted(starts, pos_int, side="right"))
    exact_entries: list[str] = []
    exact_gene_ids: set[str] = set()
    if hi_exact > 0:
        exact_mask = ends[:hi_exact] >= pos_int
        if bool(np.any(exact_mask)):
            exact_rowids = rowids[:hi_exact][exact_mask]
            exact_features = features[:hi_exact][exact_mask]
            exact_starts = starts[:hi_exact][exact_mask]
            exact_ends = ends[:hi_exact][exact_mask]
            exact_feature_ids = _postgwas_take_row_values_object(
                gff,
                _gff_prefetched_attr_colname("ID"),
                exact_rowids,
            )
            exact_parents = _postgwas_take_row_values_object(
                gff,
                _gff_prefetched_attr_colname("Parent"),
                exact_rowids,
            )
            exact_descs = _postgwas_take_row_values_object(
                gff,
                _gff_prefetched_attr_colname(_ANNO_DESC_KEY),
                exact_rowids,
            )
            exact_names = _postgwas_take_row_values_object(
                gff,
                _gff_prefetched_attr_colname("Name"),
                exact_rowids,
            )
            exact_strands = _postgwas_take_row_values_object(
                gff,
                "strand",
                exact_rowids,
                na_value=".",
            )

            gene_meta, row_gene_ids = _build_postgwas_local_gff_context_arrays(
                features=exact_features,
                feature_ids=exact_feature_ids,
                parents_text=exact_parents,
                descs=exact_descs,
                names=exact_names,
                starts=exact_starts,
                ends=exact_ends,
                strands=exact_strands,
            )
            gene_features: dict[str, set[str]] = {}
            gene_order: list[str] = []
            for idx in range(int(exact_features.shape[0])):
                feature = str(exact_features[idx]).strip().lower()
                if feature == "":
                    continue
                for gene_id in row_gene_ids[idx]:
                    if gene_id not in gene_meta:
                        continue
                    if gene_id not in gene_features:
                        gene_features[gene_id] = set()
                        gene_order.append(gene_id)
                    gene_features[gene_id].add(feature)

            exact_gene_ids = set(gene_features)
            for gene_id in gene_order:
                meta = gene_meta.get(gene_id, {})
                exact_entries.append(
                    _postgwas_annotation_triplet(
                        _postgwas_choose_exact_gff_label(gene_features.get(gene_id, set())),
                        gene_id,
                        meta.get("desc", "NA"),
                    )
                )

    flank = max(0, int(flank_bp))
    hi_near = int(np.searchsorted(starts, pos_int + flank, side="right"))
    nearby_records: list[tuple[int, int, str, str]] = []
    if hi_near > 0:
        nearby_mask = (ends[:hi_near] >= pos_int - flank) & (features[:hi_near] == "gene")
        if bool(np.any(nearby_mask)):
            nearby_rowids = rowids[:hi_near][nearby_mask]
            nearby_starts = starts[:hi_near][nearby_mask]
            nearby_ends = ends[:hi_near][nearby_mask]
            nearby_gene_ids = _postgwas_take_row_values_object(
                gff,
                _gff_prefetched_attr_colname("ID"),
                nearby_rowids,
            )
            nearby_strands = _postgwas_take_row_values_object(
                gff,
                "strand",
                nearby_rowids,
                na_value=".",
            )
            for gene_id_raw, start_raw, end_raw, strand_raw in zip(
                nearby_gene_ids.tolist(),
                nearby_starts.tolist(),
                nearby_ends.tolist(),
                nearby_strands.tolist(),
            ):
                gene_id = _normalize_gff_entity_id(gene_id_raw)
                if gene_id == "NA" or gene_id in exact_gene_ids:
                    continue
                start_i = int(start_raw)
                end_i = int(end_raw)
                strand = _clean_anno_token(strand_raw)
                if pos_int < start_i:
                    dist = int(start_i - pos_int)
                    if dist > flank:
                        continue
                    label = "Upstream2kb" if strand != "-" else "Downstream2kb"
                elif pos_int > end_i:
                    dist = int(pos_int - end_i)
                    if dist > flank:
                        continue
                    label = "Downstream2kb" if strand != "-" else "Upstream2kb"
                else:
                    continue
                nearby_records.append((int(dist), int(start_i), gene_id, str(label)))

    nearby_records.sort(key=lambda x: (int(x[0]), int(x[1]), str(x[2])))
    for _dist, _start, gene_id, label in nearby_records:
        exact_entries.append(_postgwas_annotation_triplet(label, gene_id, "NA"))
    return _join_postgwas_annotation_entries(exact_entries)


def _format_postgwas_gff_broaden_direct(
    *,
    chrom: object,
    pos: object,
    gff_query: Optional[GFFQuery] = None,
    gff_rust_index: Optional[object] = None,
    window_bp: int,
) -> str:
    try:
        pos_int = int(pos)
    except Exception:
        return "NA"
    if gff_rust_index is not None:
        try:
            return str(
                gff_rust_index.annotate_site_broaden(
                    str(chrom),
                    int(pos_int),
                    int(window_bp),
                )
            )
        except Exception:
            pass
    if gff_query is None:
        return "NA"
    window = max(0, int(window_bp))
    hit = gff_query.query_range(
        chrom,
        max(0, pos_int - window),
        pos_int + window,
        features=None,
        attr=("ID", _ANNO_DESC_KEY, "Name"),
    )
    hit = hit.loc[
        hit["feature"].astype(str).str.strip().str.lower() == "gene"
    ].copy()
    if hit.shape[0] == 0:
        return "NA"
    out: list[str] = []
    seen: set[str] = set()
    for _, row in hit.iterrows():
        attrs = row.get("attribute", [])
        gene_id = _normalize_gff_entity_id(_postgwas_attr_list_value(attrs, 0))
        if gene_id == "NA" or gene_id in seen:
            continue
        seen.add(gene_id)
        desc = _clean_anno_token(_postgwas_attr_list_value(attrs, 1))
        if desc == "NA":
            desc = _clean_anno_token(_postgwas_attr_list_value(attrs, 2))
        out.append(f"{gene_id}:{desc}")
    if len(out) == 0:
        return "NA"
    return ";".join(out)


def _group_postgwas_sites_by_chr(
    site_index: pd.MultiIndex,
) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for out_idx, key in enumerate(site_index.tolist()):
        if not isinstance(key, tuple) or len(key) < 2:
            continue
        chrom = _normalize_chr(key[0])
        try:
            pos = int(key[1])
        except Exception:
            continue
        out.setdefault(chrom, []).append((int(out_idx), int(pos)))
    return out


def _collect_postgwas_exact_gff_hits_sorted(
    positions_sorted: np.ndarray,
    feature_block: Optional[dict[str, np.ndarray]],
    *,
    gene_meta: dict[str, dict[str, object]],
) -> tuple[list[list[str]], list[dict[str, set[str]]]]:
    n_pos = int(positions_sorted.shape[0])
    empty_orders = [[] for _ in range(n_pos)]
    empty_maps: list[dict[str, set[str]]] = [dict() for _ in range(n_pos)]
    if feature_block is None:
        return empty_orders, empty_maps

    starts = np.asarray(feature_block.get("starts", np.asarray([], dtype=np.int64)), dtype=np.int64)
    ends = np.asarray(feature_block.get("ends", np.asarray([], dtype=np.int64)), dtype=np.int64)
    features = np.asarray(feature_block.get("features", np.asarray([], dtype=object)), dtype=object)
    row_gene_ids = np.asarray(feature_block.get("row_gene_ids", np.asarray([], dtype=object)), dtype=object)
    if starts.size == 0:
        return empty_orders, empty_maps

    active_heap: list[tuple[int, int]] = []
    active_rows: dict[int, None] = {}
    add_ptr = 0
    n_feat = int(starts.shape[0])
    for pos_idx, pos in enumerate(positions_sorted.tolist()):
        pos_int = int(pos)
        while add_ptr < n_feat and int(starts[add_ptr]) <= pos_int:
            active_rows[int(add_ptr)] = None
            heapq.heappush(active_heap, (int(ends[add_ptr]), int(add_ptr)))
            add_ptr += 1
        while len(active_heap) > 0 and int(active_heap[0][0]) < pos_int:
            _end, feat_idx = heapq.heappop(active_heap)
            active_rows.pop(int(feat_idx), None)
        gene_order: list[str] = []
        gene_features: dict[str, set[str]] = {}
        for feat_idx in active_rows.keys():
            feature = str(features[int(feat_idx)]).strip().lower()
            if feature == "":
                continue
            gene_ids = row_gene_ids[int(feat_idx)]
            if gene_ids is None:
                continue
            for gene_id in tuple(gene_ids):
                gene_id_text = str(gene_id).strip()
                if gene_id_text == "" or gene_id_text not in gene_meta:
                    continue
                if gene_id_text not in gene_features:
                    gene_features[gene_id_text] = set()
                    gene_order.append(gene_id_text)
                gene_features[gene_id_text].add(feature)
        empty_orders[pos_idx] = gene_order
        empty_maps[pos_idx] = gene_features
    return empty_orders, empty_maps


def _collect_postgwas_nearby_gene_hits_sorted(
    positions_sorted: np.ndarray,
    gene_block: Optional[dict[str, np.ndarray]],
    *,
    exact_gene_sets: list[set[str]],
    flank_bp: int,
) -> list[list[tuple[int, int, str, str]]]:
    out: list[list[tuple[int, int, str, str]]] = [[] for _ in range(int(positions_sorted.shape[0]))]
    if gene_block is None:
        return out

    starts = np.asarray(gene_block.get("starts", np.asarray([], dtype=np.int64)), dtype=np.int64)
    ends = np.asarray(gene_block.get("ends", np.asarray([], dtype=np.int64)), dtype=np.int64)
    gene_ids = np.asarray(gene_block.get("gene_ids", np.asarray([], dtype=object)), dtype=object)
    strands = np.asarray(gene_block.get("strands", np.asarray([], dtype=object)), dtype=object)
    if starts.size == 0:
        return out

    flank = max(0, int(flank_bp))
    active_heap: list[tuple[int, int]] = []
    active_genes: dict[int, None] = {}
    add_ptr = 0
    n_gene = int(starts.shape[0])
    for pos_idx, pos in enumerate(positions_sorted.tolist()):
        pos_int = int(pos)
        add_limit = pos_int + flank
        while add_ptr < n_gene and int(starts[add_ptr]) <= add_limit:
            active_genes[int(add_ptr)] = None
            heapq.heappush(active_heap, (int(ends[add_ptr]), int(add_ptr)))
            add_ptr += 1
        keep_min_end = pos_int - flank
        while len(active_heap) > 0 and int(active_heap[0][0]) < keep_min_end:
            _end, gene_idx = heapq.heappop(active_heap)
            active_genes.pop(int(gene_idx), None)

        exact_genes = exact_gene_sets[pos_idx]
        records: list[tuple[int, int, str, str]] = []
        for gene_idx in active_genes.keys():
            gene_id = str(gene_ids[int(gene_idx)]).strip()
            if gene_id == "" or gene_id in exact_genes:
                continue
            start = int(starts[int(gene_idx)])
            end = int(ends[int(gene_idx)])
            strand = _clean_anno_token(strands[int(gene_idx)])
            if pos_int < start:
                dist = int(start - pos_int)
                if dist > flank:
                    continue
                label = "Upstream2kb" if strand != "-" else "Downstream2kb"
            elif pos_int > end:
                dist = int(pos_int - end)
                if dist > flank:
                    continue
                label = "Downstream2kb" if strand != "-" else "Upstream2kb"
            else:
                continue
            records.append((int(dist), int(start), gene_id, str(label)))
        records.sort(key=lambda x: (int(x[0]), int(x[1]), str(x[2])))
        out[pos_idx] = records
    return out


def _collect_postgwas_gene_window_hits_sorted(
    positions_sorted: np.ndarray,
    gene_block: Optional[dict[str, np.ndarray]],
    *,
    window_bp: int,
) -> list[list[str]]:
    out: list[list[str]] = [[] for _ in range(int(positions_sorted.shape[0]))]
    if gene_block is None:
        return out

    starts = np.asarray(gene_block.get("starts", np.asarray([], dtype=np.int64)), dtype=np.int64)
    ends = np.asarray(gene_block.get("ends", np.asarray([], dtype=np.int64)), dtype=np.int64)
    gene_ids = np.asarray(gene_block.get("gene_ids", np.asarray([], dtype=object)), dtype=object)
    if starts.size == 0:
        return out

    window = max(0, int(window_bp))
    active_heap: list[tuple[int, int]] = []
    active_genes: dict[int, None] = {}
    add_ptr = 0
    n_gene = int(starts.shape[0])
    for pos_idx, pos in enumerate(positions_sorted.tolist()):
        pos_int = int(pos)
        add_limit = pos_int + window
        while add_ptr < n_gene and int(starts[add_ptr]) <= add_limit:
            active_genes[int(add_ptr)] = None
            heapq.heappush(active_heap, (int(ends[add_ptr]), int(add_ptr)))
            add_ptr += 1
        keep_min_end = pos_int - window
        while len(active_heap) > 0 and int(active_heap[0][0]) < keep_min_end:
            _end, gene_idx = heapq.heappop(active_heap)
            active_genes.pop(int(gene_idx), None)

        seen: set[str] = set()
        gene_list: list[str] = []
        for gene_idx in active_genes.keys():
            gene_id = str(gene_ids[int(gene_idx)]).strip()
            if gene_id == "" or gene_id in seen:
                continue
            seen.add(gene_id)
            gene_list.append(gene_id)
        out[pos_idx] = gene_list
    return out


def _format_postgwas_gene_annotation_from_ids(
    gene_ids: list[str],
    *,
    gene_meta: dict[str, dict[str, object]],
) -> str:
    if gene_ids is None or len(gene_ids) == 0:
        return "NA"
    out: list[str] = []
    seen: set[str] = set()
    for gene_id in gene_ids:
        gene_id_text = str(gene_id).strip()
        if gene_id_text == "" or gene_id_text in seen or gene_id_text not in gene_meta:
            continue
        seen.add(gene_id_text)
        out.append(
            f"{gene_id_text}:{_clean_anno_token(gene_meta[gene_id_text].get('desc', 'NA'))}"
        )
    if len(out) == 0:
        return "NA"
    return ";".join(out)


def _postgwas_site_index_to_rust_query_arrays(
    site_index: pd.MultiIndex,
) -> tuple[list[str], list[int]]:
    chroms: list[str] = []
    poss: list[int] = []
    for key in site_index.tolist():
        if not isinstance(key, tuple) or len(key) < 2:
            chroms.append("")
            poss.append(0)
            continue
        chroms.append(str(key[0]))
        try:
            poss.append(int(key[1]))
        except Exception:
            poss.append(0)
    return chroms, poss


def _format_postgwas_gff_site_desc_many_rust(
    site_index: pd.MultiIndex,
    *,
    gff_rust_index: object,
    flank_bp: int = 2_000,
) -> list[str]:
    if len(site_index) == 0:
        return []
    chroms, poss = _postgwas_site_index_to_rust_query_arrays(site_index)
    return [str(x) for x in gff_rust_index.annotate_many_desc(chroms, poss, int(flank_bp))]


def _format_postgwas_gff_broaden_many_rust(
    site_index: pd.MultiIndex,
    *,
    gff_rust_index: object,
    window_bp: int,
) -> list[str]:
    if len(site_index) == 0:
        return []
    chroms, poss = _postgwas_site_index_to_rust_query_arrays(site_index)
    return [str(x) for x in gff_rust_index.annotate_many_broaden(chroms, poss, int(window_bp))]


def _format_postgwas_gff_site_desc_many(
    site_index: pd.MultiIndex,
    *,
    annotation_ctx: dict[str, object],
    flank_bp: int = 2_000,
) -> list[str]:
    out = ["Intergenic;NA;NA"] * len(site_index)
    if len(site_index) == 0:
        return out

    gene_meta = annotation_ctx.get("gene_meta", {})
    feature_index_by_chr = annotation_ctx.get("feature_index_by_chr", {})
    gene_index_by_chr = annotation_ctx.get("gene_index_by_chr", {})
    if not isinstance(gene_meta, dict):
        return out

    grouped = _group_postgwas_sites_by_chr(site_index)
    for chrom_norm, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: int(x[1]))
        positions_sorted = np.asarray([int(pos) for _, pos in items_sorted], dtype=np.int64)
        feature_block = None
        if isinstance(feature_index_by_chr, dict):
            feature_block = feature_index_by_chr.get(str(chrom_norm))
        gene_block = None
        if isinstance(gene_index_by_chr, dict):
            gene_block = gene_index_by_chr.get(str(chrom_norm))

        exact_orders, exact_maps = _collect_postgwas_exact_gff_hits_sorted(
            positions_sorted,
            feature_block,
            gene_meta=gene_meta,
        )
        exact_gene_sets = [set(order) for order in exact_orders]
        nearby_records = _collect_postgwas_nearby_gene_hits_sorted(
            positions_sorted,
            gene_block,
            exact_gene_sets=exact_gene_sets,
            flank_bp=int(flank_bp),
        )
        for local_idx, (out_idx, _pos) in enumerate(items_sorted):
            entries: list[str] = []
            for gene_id in exact_orders[local_idx]:
                meta = gene_meta.get(gene_id, {})
                entries.append(
                    _postgwas_annotation_triplet(
                        _postgwas_choose_exact_gff_label(
                            exact_maps[local_idx].get(gene_id, set())
                        ),
                        gene_id,
                        meta.get("desc", "NA"),
                    )
                )
            for _dist, _start, gene_id, label in nearby_records[local_idx]:
                entries.append(_postgwas_annotation_triplet(label, gene_id, "NA"))
            out[int(out_idx)] = _join_postgwas_annotation_entries(entries)
    return out


def _format_postgwas_bed_site_desc_many(
    site_index: pd.MultiIndex,
    *,
    annotation_ctx: dict[str, object],
    flank_bp: int = 2_000,
) -> list[str]:
    out = ["Intergenic;NA;NA"] * len(site_index)
    if len(site_index) == 0:
        return out

    feature_index_by_chr = annotation_ctx.get("feature_index_by_chr", {})
    gene_index_by_chr = annotation_ctx.get("gene_index_by_chr", {})
    if not isinstance(feature_index_by_chr, dict) or not isinstance(gene_index_by_chr, dict):
        return out

    grouped = _group_postgwas_sites_by_chr(site_index)
    flank = max(0, int(flank_bp))
    for chrom_norm, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: int(x[1]))
        feature_block = feature_index_by_chr.get(str(chrom_norm))
        gene_block = gene_index_by_chr.get(str(chrom_norm))
        if feature_block is None and gene_block is None:
            continue

        feature_starts = (
            np.asarray(feature_block.get("starts", np.asarray([], dtype=np.int64)), dtype=np.int64)
            if feature_block is not None
            else np.asarray([], dtype=np.int64)
        )
        feature_ends = (
            np.asarray(feature_block.get("ends", np.asarray([], dtype=np.int64)), dtype=np.int64)
            if feature_block is not None
            else np.asarray([], dtype=np.int64)
        )
        feature_labels = (
            np.asarray(feature_block.get("labels", np.asarray([], dtype=object)), dtype=object)
            if feature_block is not None
            else np.asarray([], dtype=object)
        )
        feature_gene_ids = (
            np.asarray(feature_block.get("gene_ids", np.asarray([], dtype=object)), dtype=object)
            if feature_block is not None
            else np.asarray([], dtype=object)
        )
        feature_descs = (
            np.asarray(feature_block.get("descs", np.asarray([], dtype=object)), dtype=object)
            if feature_block is not None
            else np.asarray([], dtype=object)
        )

        gene_starts = (
            np.asarray(gene_block.get("starts", np.asarray([], dtype=np.int64)), dtype=np.int64)
            if gene_block is not None
            else np.asarray([], dtype=np.int64)
        )
        gene_ends = (
            np.asarray(gene_block.get("ends", np.asarray([], dtype=np.int64)), dtype=np.int64)
            if gene_block is not None
            else np.asarray([], dtype=np.int64)
        )
        gene_ids = (
            np.asarray(gene_block.get("gene_ids", np.asarray([], dtype=object)), dtype=object)
            if gene_block is not None
            else np.asarray([], dtype=object)
        )
        gene_strands = (
            np.asarray(gene_block.get("strands", np.asarray([], dtype=object)), dtype=object)
            if gene_block is not None
            else np.asarray([], dtype=object)
        )

        for out_idx, pos in items_sorted:
            pos_int = int(pos)
            entries: list[str] = []
            if feature_starts.size > 0:
                hi_exact = int(np.searchsorted(feature_starts, pos_int, side="right"))
                if hi_exact > 0:
                    exact_mask = feature_ends[:hi_exact] >= pos_int
                    if bool(np.any(exact_mask)):
                        exact_labels = feature_labels[:hi_exact][exact_mask]
                        exact_gene_ids = feature_gene_ids[:hi_exact][exact_mask]
                        exact_descs = feature_descs[:hi_exact][exact_mask]
                        grouped_rows: dict[str, dict[str, object]] = {}
                        grouped_order: list[str] = []
                        for hit_i in range(int(exact_labels.shape[0])):
                            label = _clean_anno_token(exact_labels[hit_i])
                            gene_id = _clean_anno_token(exact_gene_ids[hit_i])
                            desc = _clean_anno_token(exact_descs[hit_i])
                            key = str(gene_id) if gene_id != "NA" else f"__row_{hit_i}"
                            if key not in grouped_rows:
                                grouped_rows[key] = {
                                    "gene_id": gene_id,
                                    "desc": desc,
                                    "labels": set(),
                                }
                                grouped_order.append(key)
                            grouped_rows[key]["labels"].add(str(label).lower())
                            grouped_rows[key]["desc"] = _merge_anno_value(
                                _clean_anno_token(grouped_rows[key]["desc"]),
                                desc,
                            )
                        for key in grouped_order:
                            info = grouped_rows[key]
                            entries.append(
                                _postgwas_annotation_triplet(
                                    _postgwas_choose_exact_bed_label(info["labels"]),
                                    info["gene_id"],
                                    info["desc"],
                                )
                            )

            if len(entries) == 0 and gene_starts.size > 0 and flank > 0:
                hi_near = int(np.searchsorted(gene_starts, pos_int + flank, side="right"))
                if hi_near > 0:
                    nearby_records: list[tuple[int, int, str, str]] = []
                    seen_genes: set[str] = set()
                    for gene_i in range(int(hi_near)):
                        gene_id = _clean_anno_token(gene_ids[gene_i])
                        if gene_id == "NA" or gene_id in seen_genes:
                            continue
                        strand = _clean_anno_token(gene_strands[gene_i])
                        if strand not in {"+", "-"}:
                            continue
                        start_i = int(gene_starts[gene_i])
                        end_i = int(gene_ends[gene_i])
                        if pos_int < start_i:
                            dist = int(start_i - pos_int)
                            if dist > flank:
                                continue
                            label = "Upstream2kb" if strand != "-" else "Downstream2kb"
                        elif pos_int > end_i:
                            dist = int(pos_int - end_i)
                            if dist > flank:
                                continue
                            label = "Downstream2kb" if strand != "-" else "Upstream2kb"
                        else:
                            continue
                        seen_genes.add(gene_id)
                        nearby_records.append((int(dist), int(start_i), str(gene_id), str(label)))
                    nearby_records.sort(key=lambda x: (int(x[0]), int(x[1]), str(x[2])))
                    for _dist, _start, gene_id, label in nearby_records:
                        entries.append(_postgwas_annotation_triplet(label, gene_id, "NA"))

            out[int(out_idx)] = _join_postgwas_annotation_entries(entries)
    return out


def _format_postgwas_bed_broaden_many(
    site_index: pd.MultiIndex,
    *,
    annotation_ctx: dict[str, object],
    window_bp: int,
) -> list[str]:
    return _format_postgwas_gff_broaden_many(
        site_index,
        annotation_ctx=annotation_ctx,
        window_bp=int(window_bp),
    )


def _format_postgwas_gff_broaden_many(
    site_index: pd.MultiIndex,
    *,
    annotation_ctx: dict[str, object],
    window_bp: int,
) -> list[str]:
    out = ["NA"] * len(site_index)
    if len(site_index) == 0:
        return out

    gene_meta = annotation_ctx.get("gene_meta", {})
    gene_index_by_chr = annotation_ctx.get("gene_index_by_chr", {})
    if not isinstance(gene_meta, dict) or not isinstance(gene_index_by_chr, dict):
        return out

    grouped = _group_postgwas_sites_by_chr(site_index)
    for chrom_norm, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: int(x[1]))
        positions_sorted = np.asarray([int(pos) for _, pos in items_sorted], dtype=np.int64)
        gene_block = gene_index_by_chr.get(str(chrom_norm))
        gene_hits = _collect_postgwas_gene_window_hits_sorted(
            positions_sorted,
            gene_block,
            window_bp=int(window_bp),
        )
        for local_idx, (out_idx, _pos) in enumerate(items_sorted):
            out[int(out_idx)] = _format_postgwas_gene_annotation_from_ids(
                gene_hits[local_idx],
                gene_meta=gene_meta,
            )
    return out


def _format_postgwas_gff_site_desc(
    *,
    chrom: object,
    pos: object,
    gff_query: GFFQuery,
    annotation_ctx: dict[str, object],
    flank_bp: int = 2_000,
) -> str:
    _ = gff_query
    one_index = pd.MultiIndex.from_tuples([(chrom, pos)])
    descs = _format_postgwas_gff_site_desc_many(
        one_index,
        annotation_ctx=annotation_ctx,
        flank_bp=int(flank_bp),
    )
    if len(descs) == 0:
        return "Intergenic;NA;NA"
    return str(descs[0])


def _format_clump_sites(sites: list[tuple[str, int]]) -> str:
    if sites is None or len(sites) == 0:
        return ""
    return ";".join([f"{str(chrom)}_{int(pos)}" for chrom, pos in sites])


def _resolve_annotation_append_colnames(original_columns: list[str]) -> dict[str, str]:
    taken = {str(col) for col in original_columns}
    resolved: dict[str, str] = {}
    for base in _ANNOTATION_APPEND_BASE_COLS:
        candidate = str(base)
        if candidate in taken:
            candidate = f"anno_{base}"
            suffix = 2
            while candidate in taken:
                candidate = f"anno_{base}_{suffix}"
                suffix += 1
        resolved[str(base)] = candidate
        taken.add(candidate)
    return resolved


def _prepare_annotation_base_rows(
    df_sig_raw: pd.DataFrame,
    *,
    chr_col: str,
    pos_col: str,
) -> pd.DataFrame:
    if chr_col not in df_sig_raw.columns or pos_col not in df_sig_raw.columns:
        raise KeyError(f"Annotation output requires columns {chr_col!r} and {pos_col!r}.")
    base = df_sig_raw.copy()
    base[chr_col] = base[chr_col].astype(str)
    base[pos_col] = pd.to_numeric(base[pos_col], errors="coerce")
    base = base.dropna(subset=[pos_col]).copy()
    if base.shape[0] == 0:
        return base.set_index([chr_col, pos_col], drop=True)
    base[pos_col] = base[pos_col].astype(int)
    base = base.drop_duplicates(subset=[chr_col, pos_col], keep="first")
    return base.set_index([chr_col, pos_col], drop=True)


def _finalize_annotation_output_df(
    df_filter: pd.DataFrame,
    *,
    chr_col: str,
    pos_col: str,
    original_columns: list[str],
    annotation_col_map: dict[str, str],
) -> pd.DataFrame:
    df_out = df_filter.reset_index()
    if pos_col in df_out.columns:
        df_out[pos_col] = pd.to_numeric(df_out[pos_col], errors="coerce").fillna(0).astype(int)

    start_col = annotation_col_map["start"]
    end_col = annotation_col_map["end"]
    nsnps_col = annotation_col_map["nsnps"]
    meanr2_col = annotation_col_map["MeanR2"]

    if start_col in df_out.columns:
        df_out[start_col] = (
            pd.to_numeric(df_out[start_col], errors="coerce")
            .fillna(df_out[pos_col] if pos_col in df_out.columns else 0)
            .astype(int)
        )
    if end_col in df_out.columns:
        df_out[end_col] = (
            pd.to_numeric(df_out[end_col], errors="coerce")
            .fillna(df_out[pos_col] if pos_col in df_out.columns else 0)
            .astype(int)
        )
    if nsnps_col in df_out.columns:
        df_out[nsnps_col] = pd.to_numeric(df_out[nsnps_col], errors="coerce").fillna(0).astype(int)
    if meanr2_col in df_out.columns:
        df_out[meanr2_col] = (
            pd.to_numeric(df_out[meanr2_col], errors="coerce")
            .fillna(0.0)
            .map(lambda x: f"{float(x):.2f}")
        )

    if chr_col in df_out.columns and pos_col in df_out.columns:
        df_out["_chr_sort_key"] = df_out[chr_col].map(_chrom_sort_key)
        df_out = df_out.sort_values(
            by=["_chr_sort_key", pos_col],
            ascending=[True, True],
            kind="mergesort",
        ).drop(columns=["_chr_sort_key"])

    orig_cols = [str(col) for col in original_columns if str(col) in df_out.columns]
    append_cols = [
        annotation_col_map[base]
        for base in _ANNOTATION_APPEND_BASE_COLS
        if annotation_col_map.get(base) in df_out.columns
    ]
    remain_cols = [c for c in df_out.columns if c not in orig_cols and c not in append_cols]
    return df_out.loc[:, orig_cols + remain_cols + append_cols]


def _load_postgwas_input_table(
    file: str,
    *,
    chr_col: str,
    pos_col: str,
    p_col: str,
    keep_all_columns: bool = False,
) -> tuple[pd.DataFrame, list[object]]:
    try:
        header_cols = pd.read_csv(file, sep="\t", nrows=0).columns.tolist()
    except Exception:
        header_cols = [chr_col, pos_col, p_col]
    resolved_p_col = _postgwas_resolve_input_pvalue_column(header_cols, p_col)
    lead_info_cols = [c for c in _LEAD_SNP_INFO_COLS if c in header_cols]
    interaction_info_cols = [c for c in _INTERACTION_PLOT_INFO_COLS if c in header_cols]
    read_cols = [chr_col, pos_col, resolved_p_col] + lead_info_cols + interaction_info_cols
    read_cols = list(dict.fromkeys(read_cols))
    if bool(keep_all_columns):
        df_all = pd.read_csv(file, sep="\t")
    else:
        df_all = pd.read_csv(file, sep="\t", usecols=read_cols)
    requested_p_col = str(p_col).strip()
    if (
        requested_p_col != ""
        and requested_p_col not in df_all.columns
        and resolved_p_col in df_all.columns
    ):
        df_all[requested_p_col] = df_all[resolved_p_col]
    full_chr_labels = df_all[chr_col].drop_duplicates().tolist()
    return df_all, full_chr_labels


def _postgwas_first_present_column(
    columns: object,
    candidates: list[object],
) -> Optional[str]:
    if columns is None:
        return None
    colset = {str(col) for col in list(columns)}
    for candidate in candidates:
        text = str(candidate).strip()
        if text != "" and text in colset:
            return text
    return None


def _postgwas_resolve_input_pvalue_column(
    columns: object,
    requested_p_col: object,
) -> str:
    requested = str(requested_p_col).strip()
    candidates = [
        requested,
        "pwald",
        "padj",
        "p",
        "P",
        "pvalue",
        "Pvalue",
        "pval",
        "Pval",
        "P_Wald",
        "P_wald",
        "p_wald",
    ]
    resolved = _postgwas_first_present_column(
        columns,
        list(dict.fromkeys([c for c in candidates if str(c).strip() != ""])),
    )
    if resolved is not None:
        return str(resolved)
    col_list = [str(col) for col in list(columns or [])]
    shown = ", ".join(col_list[:20])
    extra = "" if len(col_list) <= 20 else f" ... (+{len(col_list) - 20} more)"
    raise ValueError(
        f"P-value column '{requested or 'NA'}' was not found in the input table. "
        f"Available columns: {shown}{extra}"
    )


def _postgwas_parse_interact_spec(spec_text: object | None) -> dict[str, object]:
    defaults = {
        "snp_col": "snp",
        "chr_col": "chrom",
        "pos_col": "pos",
        "p_col": "pwald",
        "group_tokens": ["|", "&", "*"],
    }
    if spec_text is None:
        return dict(defaults)
    text = str(spec_text).strip()
    if text == "":
        return dict(defaults)
    parts = [str(x).strip() for x in text.split(";")]
    parts = [x for x in parts if x != ""]
    if len(parts) < 4:
        raise ValueError(
            "interact spec must be 'snp;chrom;pos;pvalue;group1;group2;...'"
        )
    out = {
        "snp_col": parts[0],
        "chr_col": parts[1],
        "pos_col": parts[2],
        "p_col": parts[3],
        "group_tokens": [
            re.sub(r"\\(.)", r"\1", str(x))
            for x in (parts[4:] if len(parts) > 4 else list(defaults["group_tokens"]))
        ],
    }
    return out


def _postgwas_load_circle_interact_df(
    interact_path: str,
    *,
    group_col: str,
    chr_col: str,
    pos_col: str,
    p_col: Optional[str],
) -> pd.DataFrame:
    header = pd.read_csv(interact_path, sep="\t", nrows=0)
    present_cols = [str(col) for col in list(header.columns)]
    needed = [str(group_col), str(chr_col), str(pos_col)]
    if p_col is not None:
        needed.append(str(p_col))
    if "row_role" in present_cols:
        needed.append("row_role")
    usecols = [col for col in dict.fromkeys(needed) if col in present_cols]
    return pd.read_csv(interact_path, sep="\t", usecols=usecols)


def _postgwas_build_circle_link_table_from_groups(
    df: pd.DataFrame,
    *,
    group_col: str,
    chr_col: str,
    pos_col: str,
    p_col: Optional[str],
    type_col: Optional[str] = None,
    group_tokens: Optional[list[str]] = None,
    pvalue_is_log10: bool = False,
) -> tuple[Optional[pd.DataFrame], dict[str, Optional[str]]]:
    if df.shape[0] == 0:
        return None, {
            "group_col": group_col,
            "type_col": type_col,
            "pvalue_col": p_col,
            "score_col": None,
        }

    if group_col not in df.columns:
        raise ValueError(f"Interaction group column '{group_col}' was not found in the input table.")
    if chr_col not in df.columns:
        raise ValueError(f"Interaction chromosome column '{chr_col}' was not found in the input table.")
    if pos_col not in df.columns:
        raise ValueError(f"Interaction position column '{pos_col}' was not found in the input table.")
    if p_col is not None and p_col not in df.columns:
        raise ValueError(f"Interaction p-value column '{p_col}' was not found in the input table.")
    if type_col is not None and type_col not in df.columns:
        raise ValueError(f"Interaction type column '{type_col}' was not found in the input table.")

    work = df.copy()
    work[group_col] = work[group_col].astype(str).str.strip()
    work[chr_col] = work[chr_col].astype(str)
    work[pos_col] = pd.to_numeric(work[pos_col], errors="coerce")
    work = work[(work[group_col] != "") & work[pos_col].notna()].copy()
    if work.shape[0] == 0:
        return None, {
            "group_col": group_col,
            "type_col": type_col,
            "pvalue_col": p_col,
            "score_col": None,
        }

    if "row_role" in work.columns:
        row_role = work["row_role"].astype(str).str.strip().str.lower()
        combo_mask = row_role.eq("combo")
        if bool(combo_mask.any()):
            work = work.loc[combo_mask].copy()

    tokens = [str(x) for x in list(group_tokens or []) if str(x) != ""]
    if len(tokens) > 0:
        token_mask = work[group_col].map(lambda text: any(tok in str(text) for tok in tokens))
        if bool(token_mask.any()):
            work = work.loc[token_mask].copy()
    if work.shape[0] == 0:
        return None, {
            "group_col": group_col,
            "type_col": type_col,
            "pvalue_col": p_col,
            "score_col": None,
        }

    records: list[dict[str, object]] = []
    for group_name, grp in work.groupby(group_col, sort=False):
        endpoints = grp[[chr_col, pos_col]].copy()
        endpoints[pos_col] = pd.to_numeric(endpoints[pos_col], errors="coerce")
        endpoints = endpoints.dropna(subset=[pos_col]).copy()
        if endpoints.shape[0] == 0:
            continue
        endpoints[pos_col] = endpoints[pos_col].astype(np.int64)
        endpoints = endpoints.drop_duplicates(subset=[chr_col, pos_col], keep="first").reset_index(drop=True)
        if endpoints.shape[0] < 2:
            continue
        if type_col is not None and type_col in grp.columns:
            type_values = grp[type_col].dropna().astype(str)
            type_value = str(type_values.iloc[0]).strip() if type_values.shape[0] > 0 else str(group_name)
        else:
            type_value = str(group_name)
        if p_col is not None and p_col in grp.columns:
            pvals = pd.to_numeric(grp[p_col], errors="coerce")
            pvals = pvals[np.isfinite(pvals)]
            if pvals.shape[0] > 0:
                pvalue_value = float(pvals.max() if pvalue_is_log10 else pvals.min())
            else:
                pvalue_value = float("nan")
        else:
            pvalue_value = float("nan")

        n_endpoints = int(endpoints.shape[0])
        for i in range(n_endpoints - 1):
            row_i = endpoints.iloc[i]
            for j in range(i + 1, n_endpoints):
                row_j = endpoints.iloc[j]
                records.append(
                    {
                        "chrom1": str(row_i[chr_col]),
                        "pos1": int(row_i[pos_col]),
                        "chrom2": str(row_j[chr_col]),
                        "pos2": int(row_j[pos_col]),
                        "combo_id": str(group_name),
                        "link_type": str(type_value),
                        "link_pvalue": float(pvalue_value),
                    }
                )

    if len(records) == 0:
        return None, {
            "group_col": group_col,
            "type_col": type_col,
            "pvalue_col": p_col,
            "score_col": None,
        }
    return pd.DataFrame.from_records(records), {
        "group_col": group_col,
        "type_col": type_col,
        "pvalue_col": p_col,
        "score_col": None,
    }


def _postgwas_build_circle_link_table(
    df: pd.DataFrame,
    *,
    chr_col: str,
    pos_col: str,
    p_col: str,
    pvalue_is_log10: bool = False,
) -> tuple[Optional[pd.DataFrame], dict[str, Optional[str]]]:
    group_col = _postgwas_first_present_column(
        df.columns,
        ["combo_id", "parent_combo", "combo"],
    )
    resolved_type_col = _postgwas_first_present_column(
        df.columns,
        ["logic", "gate", "type", "interaction_type", group_col] if group_col is not None else ["logic", "gate", "type", "interaction_type"],
    )

    resolved_p_col = _postgwas_first_present_column(
        df.columns,
        [
            "p_combo_joint",
            "combo_pwald_joint",
            "p_combo_marginal",
            "combo_pwald_joint_fdr",
            p_col,
            "pwald",
            "p",
        ],
    )

    if group_col is not None:
        link_df, meta = _postgwas_build_circle_link_table_from_groups(
            df,
            group_col=str(group_col),
            chr_col=chr_col,
            pos_col=pos_col,
            p_col=resolved_p_col,
            type_col=resolved_type_col,
            group_tokens=None,
            pvalue_is_log10=pvalue_is_log10,
        )
        if link_df is not None and link_df.shape[0] > 0:
            return link_df, meta

    snp_col = _postgwas_first_present_column(df.columns, ["snp"])
    if snp_col is not None:
        return _postgwas_build_circle_link_table_from_groups(
            df,
            group_col=str(snp_col),
            chr_col=chr_col,
            pos_col=pos_col,
            p_col=resolved_p_col,
            type_col=str(snp_col),
            group_tokens=["|", "&", "*"],
            pvalue_is_log10=pvalue_is_log10,
        )
    return None, {
        "group_col": None,
        "type_col": None,
        "pvalue_col": resolved_p_col,
        "score_col": None,
    }


def _load_ldclump_genotype_rows(
    genofile: str,
    keys: Sequence[tuple[str, int]],
    *,
    chunk_size: int,
    sample_ids: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Load requested LD-clump rows while consuming every returned chunk.

    ``load_genotype_chunks(..., snp_sites=...)`` may return more than one
    chunk, and a PLINK BIM file may contain repeated coordinates.  The old
    LD-clump fallback used only the first chunk, which silently paired a lead
    SNP with the wrong genotype rows in those cases.  This helper materializes
    the selected rows once, de-duplicates coordinate matches deterministically,
    and restores the caller's requested order.
    """
    requested = [(str(chrom), int(pos)) for chrom, pos in keys]
    if len(requested) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if sample_ids is not None:
        try:
            loader_parameters = inspect.signature(load_genotype_chunks).parameters
        except (TypeError, ValueError) as exc:
            raise FineMapSkip(
                "LD-clump genotype loader signature is unavailable; exact sample "
                "selection cannot be proven"
            ) from exc
        if "sample_ids" not in loader_parameters:
            raise FineMapSkip(
                "LD-clump genotype loader does not support exact sample selection"
            )

    # The fine-mapping path uses normalized chromosome labels, whereas the
    # PLINK BIM can retain a ``chr`` prefix.  Query both spellings so the
    # optimization remains useful for the generic postgwas LDclump path too.
    query_keys: list[tuple[str, int]] = []
    for key in requested:
        for candidate in (key, (_normalize_chr(key[0]), key[1])):
            if candidate not in query_keys:
                query_keys.append(candidate)
    query_set = set(query_keys)
    genotype_chunks: list[np.ndarray] = []
    returned_aliases: list[tuple[tuple[str, int], tuple[str, int]]] = []
    for chunk, sites in load_genotype_chunks(
        genofile,
        chunk_size=max(1, int(chunk_size)),
        maf=0.0,
        missing_rate=1.0,
        impute=True,
        snp_sites=query_keys,
        sample_ids=(list(sample_ids) if sample_ids is not None else None),
    ):
        genotype = np.asarray(chunk, dtype=np.float32)
        if genotype.ndim != 2 or genotype.shape[0] != len(sites):
            raise RuntimeError(
                "LD-clump genotype chunk and site metadata have inconsistent shapes"
            )
        genotype_chunks.append(genotype)
        for site in sites:
            site_chrom = str(getattr(site, "chrom"))
            site_pos = int(getattr(site, "pos"))
            returned_aliases.append(
                ((site_chrom, site_pos), (_normalize_chr(site_chrom), site_pos))
            )

    if len(genotype_chunks) == 0:
        raise RuntimeError("no genotype rows were returned for LD-clump SNPs")

    all_genotypes = np.vstack(genotype_chunks).astype(np.float32, copy=False)
    row_by_key: dict[tuple[str, int], int] = {}
    for row_idx, aliases in enumerate(returned_aliases):
        # Each SiteInfo contributes two aliases, but the genotype row only
        # occurs once in the stacked matrix. Keep the first row for repeated
        # coordinates; the GWAS preparation stage collapses them by priority.
        for key in aliases:
            if key in query_set and key not in row_by_key:
                row_by_key[key] = int(row_idx)

    missing = [key for key in requested if key not in row_by_key]
    if missing:
        raise RuntimeError(
            "genotype rows missing for LD-clump SNPs: "
            + ", ".join(f"{chrom}:{pos}" for chrom, pos in missing[:5])
        )

    order = np.asarray([row_by_key[key] for key in requested], dtype=np.int64)
    return np.ascontiguousarray(all_genotypes[order, :], dtype=np.float32)


def _ldclump_lead_r2_streaming(
    genofile: str,
    snps: Sequence[tuple[str, int]],
    *,
    chunk_rows: int,
    sample_ids: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Compute one lead's r2 values without materializing its whole window."""
    requested = [(str(chrom), int(pos)) for chrom, pos in snps]
    if len(requested) == 0:
        return np.zeros((0,), dtype=np.float64)

    lead = _load_ldclump_genotype_rows(
        genofile,
        [requested[0]],
        chunk_size=1,
        sample_ids=sample_ids,
    )
    if lead.ndim != 2 or lead.shape[0] != 1:
        raise RuntimeError("LD-clump lead genotype row has an invalid shape")

    r2 = np.ones((len(requested),), dtype=np.float64)
    block_rows = max(1, int(chunk_rows))
    for start in range(1, len(requested), block_rows):
        block_keys = requested[start : start + block_rows]
        block = _load_ldclump_genotype_rows(
            genofile,
            block_keys,
            chunk_size=len(block_keys),
            sample_ids=sample_ids,
        )
        if block.ndim != 2 or block.shape[0] != len(block_keys):
            raise RuntimeError("LD-clump candidate genotype block has an invalid shape")
        block_with_lead = np.vstack((lead, block))
        r2[start : start + len(block_keys)] = _lead_vs_all_r2(block_with_lead)[1:]
    return r2


def _postgwas_clump_compatibility_exception(exc: BaseException) -> bool:
    """Recognize source/adapter failures that may safely keep only the lead."""
    if isinstance(exc, _PostGWASExpectedCompatibilityError):
        return True
    if type(exc) in (OSError, EOFError, UnicodeError, pd.errors.ParserError):
        return True
    message = str(exc).lower()
    markers = (
        "no genotype rows",
        "genotype rows missing",
        "genotype chunk",
        "site metadata",
        "snp selection",
        "sample selection",
        "snp_sites",
        "sample_ids",
        "unsupported",
        "unavailable",
        "malformed",
        "truncated",
        "invalid shape",
    )
    if type(exc) in (ValueError, TypeError, RuntimeError):
        return any(marker in message for marker in markers)
    return False


def _ldclump_significant_snps(
    df_sig: pd.DataFrame,
    *,
    chr_col: str,
    pos_col: str,
    p_col: str,
    genofile: str,
    window_bp: int,
    r2_thr: float,
    logger: logging.Logger,
    show_progress: bool = True,
    pvalue_is_log10: bool = False,
    preload_max_rows: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    fold_r2: Optional[
        dict[tuple[str, int], dict[tuple[str, int], float]]
    ] = None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], list[tuple[str, int]]]]:
    """
    LD-clump threshold-passing SNPs and keep lead SNPs only in annotation output.
    """
    if df_sig.shape[0] == 0:
        out_empty = df_sig.set_index([chr_col, pos_col], drop=True)
        out_empty["start"] = pd.Series(dtype=int)
        out_empty["end"] = pd.Series(dtype=int)
        out_empty["nsnps"] = pd.Series(dtype=int)
        out_empty["MeanR2"] = pd.Series(dtype=float)
        out_empty["LDclump"] = pd.Series(dtype=str)
        return out_empty, {}

    work = df_sig[[chr_col, pos_col, p_col]].copy()
    work[chr_col] = work[chr_col].astype(str)
    work[pos_col] = pd.to_numeric(work[pos_col], errors="coerce")
    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    work = work.dropna(subset=[pos_col, p_col]).copy()
    if work.shape[0] == 0:
        out_empty = work.set_index([chr_col, pos_col], drop=True)
        out_empty["start"] = pd.Series(dtype=int)
        out_empty["end"] = pd.Series(dtype=int)
        out_empty["nsnps"] = pd.Series(dtype=int)
        out_empty["MeanR2"] = pd.Series(dtype=float)
        out_empty["LDclump"] = pd.Series(dtype=str)
        return out_empty, {}

    work[pos_col] = work[pos_col].astype(int)
    work = (
        work.sort_values(
            p_col,
            ascending=not bool(pvalue_is_log10),
            kind="mergesort",
        )
        .drop_duplicates(subset=[chr_col, pos_col], keep="first")
        .reset_index(drop=True)
    )
    if work.shape[0] == 0:
        out_empty = work.set_index([chr_col, pos_col], drop=True)
        out_empty["start"] = pd.Series(dtype=int)
        out_empty["end"] = pd.Series(dtype=int)
        out_empty["nsnps"] = pd.Series(dtype=int)
        out_empty["MeanR2"] = pd.Series(dtype=float)
        out_empty["LDclump"] = pd.Series(dtype=str)
        return out_empty, {}

    all_keys = [
        (str(c), int(p))
        for c, p in zip(work[chr_col].tolist(), work[pos_col].tolist())
    ]
    chrom_arr = work[chr_col].to_numpy(dtype=str)
    pos_arr = work[pos_col].to_numpy(dtype=np.int64)

    # Preload all threshold-passing SNP genotypes once, then reuse in memory.
    # This avoids repeated random-access reads for each lead SNP.
    preloaded_geno: Optional[np.ndarray] = None
    preloaded_row_sums: Optional[np.ndarray] = None
    preloaded_row_sumsq: Optional[np.ndarray] = None
    key_to_row: dict[tuple[str, int], int] = {}
    stream_chunk_rows = max(1, min(20_000, len(all_keys)))
    preload_budget_limited = bool(
        preload_max_rows is not None
        and len(all_keys) > max(1, int(preload_max_rows))
    )
    preload_error: Optional[BaseException] = None
    if not preload_budget_limited:
        logger.info(
            f"Preloading genotype rows for LD clump: {len(all_keys)} SNPs..."
        )
        try:
            preloaded_geno = _load_ldclump_genotype_rows(
                genofile,
                all_keys,
                chunk_size=max(1, min(20_000, len(all_keys))),
                sample_ids=sample_ids,
            )
            key_to_row = {k: i for i, k in enumerate(all_keys)}
            # These row statistics are invariant across all lead/window calls.
            # Reusing them avoids a full O(window * n_samples) reduction for every
            # lead and leaves the hot loop to one matrix-vector product.
            preloaded_row_sums = np.sum(preloaded_geno, axis=1, dtype=np.float64)
            preloaded_row_sumsq = np.einsum(
                "ij,ij->i", preloaded_geno, preloaded_geno, dtype=np.float64, optimize=True
            )
        except FineMapSkip:
            raise
        except MemoryError as exc:
            raise FineMapSkip(
                "LD-clump memory allocation was unavailable during preload: "
                f"{exc}"
            ) from exc
        except _PostGWASExpectedCompatibilityError as exc:
            preload_error = exc
        except (OSError, EOFError, UnicodeError, pd.errors.ParserError) as exc:
            preload_error = exc
        except ValueError as exc:
            if not _postgwas_clump_compatibility_exception(exc):
                raise
            preload_error = exc
        except TypeError as exc:
            if not _postgwas_clump_compatibility_exception(exc):
                raise
            preload_error = exc
        except RuntimeError as exc:
            if not _postgwas_clump_compatibility_exception(exc):
                raise
            preload_error = exc

    if preload_budget_limited or preload_error is not None:
        preloaded_geno = None
        preloaded_row_sums = None
        preloaded_row_sumsq = None
        key_to_row = {}
        if preload_max_rows is not None:
            stream_chunk_rows = max(1, min(stream_chunk_rows, int(preload_max_rows)))
        if preload_budget_limited:
            logger.info(
                "LD-clump preload budget limits the candidate matrix to %d rows; "
                "using streaming genotype blocks.",
                stream_chunk_rows,
            )
        else:
            logger.warning(
                "Warning: Failed to preload all LD-clump genotypes; "
                "falling back to per-lead genotype loading. Reason: %s",
                preload_error,
            )

    if not key_to_row:
        key_to_row = {k: i for i, k in enumerate(all_keys)}
    remaining = np.ones((len(all_keys),), dtype=bool)
    key_to_p: dict[tuple[str, int], float] = {
        (str(c), int(p)): float(v)
        for c, p, v in zip(work[chr_col].tolist(), work[pos_col].tolist(), work[p_col].tolist())
    }
    kept_rows: list[tuple[str, int, float, int, int, int, float, str]] = []
    clump_dict: dict[tuple[str, int], list[tuple[str, int]]] = {}

    warn_count = 0
    warn_limit = 5
    animate_progress = bool(show_progress and should_animate_status("LD clumping..."))
    use_rich_progress = bool(animate_progress and rich_progress_available())
    progress = None
    task_id = None
    progress_tqdm = None
    clump_start_ts = time.monotonic()
    clump_success = False
    if use_rich_progress:
        progress = build_rich_progress(
            show_remaining=True,
            finished_text=" ",
            transient=True,
        )

    with (progress if progress is not None else nullcontext()):
        if progress is not None:
            task_id = progress.add_task("LD clumping...", total=int(work.shape[0]))
        elif bool(animate_progress and _HAS_TQDM and stdout_is_tty()):
            progress_tqdm = tqdm(
                total=int(work.shape[0]),
                desc="LD clumping",
                unit="snp",
                leave=False,
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                           "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )

        try:
            for _, row in work.iterrows():
                try:
                    lead_chr = str(row[chr_col])
                    lead_pos = int(row[pos_col])
                    lead_key = (lead_chr, lead_pos)
                    lead_idx = key_to_row.get(lead_key)
                    if lead_idx is None or not bool(remaining[lead_idx]):
                        continue

                    start = int(lead_pos - window_bp)
                    end = int(lead_pos + window_bp)
                    win_mask = (
                        (chrom_arr == lead_chr)
                        & (pos_arr >= start)
                        & (pos_arr <= end)
                    )
                    win_idx = np.flatnonzero(win_mask & remaining)
                    win_idx = win_idx[win_idx != int(lead_idx)]
                    candidate_idx = np.concatenate(
                        (np.asarray([int(lead_idx)], dtype=np.int64), win_idx)
                    )
                    snps = [all_keys[int(i)] for i in candidate_idx]

                    clumped = [lead_key]
                    r2_map: dict[tuple[str, int], float] = {lead_key: 1.0}
                    mean_r2 = 1.0
                    if len(snps) > 1:
                        try:
                            if preloaded_geno is not None:
                                geno_block = preloaded_geno[candidate_idx, :]
                                row_sums = preloaded_row_sums[candidate_idx]
                                row_sumsq = preloaded_row_sumsq[candidate_idx]
                                r2 = _lead_vs_all_r2(
                                    geno_block,
                                    row_sums=row_sums,
                                    row_sumsq=row_sumsq,
                                )
                            else:
                                r2 = _ldclump_lead_r2_streaming(
                                    genofile,
                                    snps,
                                    chunk_rows=stream_chunk_rows,
                                    sample_ids=sample_ids,
                                )
                            if r2 is not None and r2.shape[0] == len(snps):
                                clumped = [
                                    snps[i]
                                    for i, keep in enumerate(r2 >= float(r2_thr))
                                    if bool(keep)
                                ]
                                if lead_key not in clumped:
                                    clumped.insert(0, lead_key)
                                r2_map = {snps[i]: float(r2[i]) for i in range(len(snps))}
                                mean_r2 = float(
                                    np.mean([float(r2_map.get(k, 1.0)) for k in clumped])
                                )
                            else:
                                if warn_count < warn_limit:
                                    logger.warning(
                                        "Warning: LDclump genotype rows do not match requested SNP count; "
                                        f"fallback to keep lead SNP only for {lead_chr}:{lead_pos}."
                                    )
                                warn_count += 1
                        except FineMapSkip:
                            raise
                        except MemoryError as exc:
                            raise FineMapSkip(
                                "LD-clump memory allocation was unavailable during streaming: "
                                f"{exc}"
                            ) from exc
                        except _PostGWASExpectedCompatibilityError as exc:
                            if warn_count < warn_limit:
                                logger.warning(
                                    "Warning: LDclump genotype lookup failed for "
                                    f"{lead_chr}:{lead_pos}; fallback to keep lead SNP only. "
                                    "Reason: %s",
                                    exc,
                                )
                            warn_count += 1
                            clumped = [lead_key]
                        except (OSError, EOFError, UnicodeError, pd.errors.ParserError) as exc:
                            if warn_count < warn_limit:
                                logger.warning(
                                    "Warning: LDclump genotype lookup failed for "
                                    f"{lead_chr}:{lead_pos}; fallback to keep lead SNP only. "
                                    "Reason: %s",
                                    exc,
                                )
                            warn_count += 1
                            clumped = [lead_key]
                        except ValueError as exc:
                            if not _postgwas_clump_compatibility_exception(exc):
                                raise
                            if warn_count < warn_limit:
                                logger.warning(
                                    "Warning: LDclump genotype lookup failed for "
                                    f"{lead_chr}:{lead_pos}; fallback to keep lead SNP only. "
                                    "Reason: %s",
                                    exc,
                                )
                            warn_count += 1
                            clumped = [lead_key]
                        except TypeError as exc:
                            if not _postgwas_clump_compatibility_exception(exc):
                                raise
                            if warn_count < warn_limit:
                                logger.warning(
                                    "Warning: LDclump genotype lookup failed for "
                                    f"{lead_chr}:{lead_pos}; fallback to keep lead SNP only. "
                                    "Reason: %s",
                                    exc,
                                )
                            warn_count += 1
                            clumped = [lead_key]
                        except RuntimeError as exc:
                            if not _postgwas_clump_compatibility_exception(exc):
                                raise
                            if warn_count < warn_limit:
                                logger.warning(
                                    "Warning: LDclump genotype lookup failed for "
                                    f"{lead_chr}:{lead_pos}; fallback to keep lead SNP only. "
                                    "Reason: %s",
                                    exc,
                                )
                            warn_count += 1
                            clumped = [lead_key]

                    clumped_idx = np.asarray(
                        [key_to_row[key] for key in clumped], dtype=np.int64
                    )
                    clumped = sorted(
                        clumped,
                        key=(
                            lambda k: (
                                -float(key_to_p.get(k, -np.inf)),
                                int(k[1]),
                            )
                        )
                        if bool(pvalue_is_log10)
                        else lambda k: (
                            float(key_to_p.get(k, np.inf)),
                            int(k[1]),
                        ),
                    )
                    if fold_r2 is not None:
                        fold_r2[lead_key] = {
                            key: float(r2_map.get(key, 1.0)) for key in clumped
                        }
                    clump_dict[lead_key] = clumped
                    ld_start = int(min([int(x[1]) for x in clumped])) if len(clumped) > 0 else int(lead_pos)
                    ld_end = int(max([int(x[1]) for x in clumped])) if len(clumped) > 0 else int(lead_pos)
                    nsnps = int(len(clumped))
                    kept_rows.append(
                        (
                            lead_chr,
                            lead_pos,
                            float(row[p_col]),
                            ld_start,
                            ld_end,
                            nsnps,
                            mean_r2,
                            _format_clump_sites(clumped),
                        )
                    )
                    remaining[clumped_idx] = False
                finally:
                    if progress is not None and task_id is not None:
                        progress.advance(task_id, 1)
                    if progress_tqdm is not None:
                        progress_tqdm.update(1)
        finally:
            if progress_tqdm is not None:
                progress_tqdm.close()
        clump_success = True
    clump_elapsed = format_elapsed(time.monotonic() - clump_start_ts)
    if bool(show_progress):
        if clump_success:
            print_success(f"LD clumping ...Finished [{clump_elapsed}]", force_color=True)
        else:
            print_failure(f"LD clumping ...Failed [{clump_elapsed}]")

    if warn_count > warn_limit:
        logger.warning(
            f"Warning: LDclump warnings truncated; {warn_count - warn_limit} more similar warnings omitted."
        )

    out_df = pd.DataFrame(
        kept_rows,
        columns=[chr_col, pos_col, p_col, "start", "end", "nsnps", "MeanR2", "LDclump"],
    )
    out_df = out_df.set_index([chr_col, pos_col], drop=True)
    return out_df, clump_dict


def _postgwas_finalize_finemap_ldclump(
    prepared: pd.DataFrame,
    clump_df: pd.DataFrame,
    clump_dict: dict[tuple[str, int], list[tuple[str, int]]],
    *,
    fold_r2: Optional[
        dict[tuple[str, int], dict[tuple[str, int], float]]
    ] = None,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[tuple[str, int], tuple[int, ...]],
]:
    """Restore fold groups and one matrix-order map after clumping."""
    input_rows = int(len(prepared))
    prepared_rows = prepared.reset_index(drop=True)
    if input_rows == 0:
        empty = prepared_rows.copy()
        empty.attrs["_janusx_finemap_matrix_indices"] = np.empty(
            (0,), dtype=np.int64
        )
        return empty, {
            "input_rows": 0,
            "retained_rows": 0,
            "clumped_rows": 0,
            "groups": 0,
        }, {}

    priority = -np.abs(
        pd.to_numeric(prepared_rows["z"], errors="coerce").to_numpy(dtype=float)
    )
    prepared_chroms = [
        _postgwas_finemap_normalize_chr(value)
        for value in prepared_rows["chrom_norm"].tolist()
    ]
    prepared_positions = (
        pd.to_numeric(prepared_rows["pos"], errors="coerce")
        .astype(np.int64)
        .tolist()
    )
    lead_keys = {
        (_postgwas_finemap_normalize_chr(chrom), int(pos))
        for chrom, pos in clump_df.index.tolist()
    }
    # A coordinate can occur more than once in a summary table.  LDclump is
    # coordinate-based here, so retain only the strongest-|z| representative
    # for each retained coordinate; otherwise a duplicate BIM site could
    # re-enter the dense SuSiE matrix after the clump step.
    representative_by_key: dict[tuple[str, int], int] = {}
    rows_by_key: dict[tuple[str, int], list[int]] = {}
    for row_idx, (chrom, pos) in enumerate(zip(prepared_chroms, prepared_positions)):
        key = (chrom, int(pos))
        rows_by_key.setdefault(key, []).append(int(row_idx))
        old_idx = representative_by_key.get(key)
        if old_idx is None or priority[row_idx] < priority[old_idx]:
            representative_by_key[key] = int(row_idx)

    snp_identifiers = (
        [
            _postgwas_finemap_nonempty_text(value) or ""
            for value in prepared_rows["snp"].tolist()
        ]
        if "snp" in prepared_rows.columns
        else [""] * input_rows
    )
    fold_groups: dict[tuple[str, int], tuple[int, ...]] = {}
    for lead_key_raw, clump_keys in clump_dict.items():
        lead_key = (
            _postgwas_finemap_normalize_chr(lead_key_raw[0]),
            int(lead_key_raw[1]),
        )
        normalized_clump_keys = {
            (_postgwas_finemap_normalize_chr(chrom), int(pos))
            for chrom, pos in clump_keys
        }
        if lead_key not in normalized_clump_keys:
            raise ValueError(
                "Fine-mapping LD-clump group does not contain its representative "
                f"coordinate {lead_key[0]}:{lead_key[1]}"
            )
        representative_idx = representative_by_key.get(lead_key)
        if representative_idx is None:
            raise ValueError(
                "Fine-mapping LD-clump representative coordinate is absent from "
                f"prepared rows: {lead_key[0]}:{lead_key[1]}"
            )
        member_indices = sorted(
            {
                row_idx
                for key in normalized_clump_keys
                for row_idx in rows_by_key.get(key, [])
            },
            key=lambda row_idx: (
                prepared_chroms[row_idx],
                int(prepared_positions[row_idx]),
                snp_identifiers[row_idx],
                int(row_idx),
            ),
        )
        if representative_idx not in member_indices:
            raise ValueError(
                "Fine-mapping LD-clump group does not resolve its representative "
                f"row for {lead_key[0]}:{lead_key[1]}"
            )
        fold_groups[lead_key] = tuple(
            [representative_idx]
            + [row_idx for row_idx in member_indices if row_idx != representative_idx]
        )

    keep = np.asarray(
        [
            (str(chrom), int(pos)) in lead_keys
            and representative_by_key.get((str(chrom), int(pos))) == row_idx
            for row_idx, (chrom, pos) in enumerate(
                zip(prepared_chroms, prepared_positions)
            )
        ],
        dtype=bool,
    )
    retained = prepared_rows.loc[keep].copy().reset_index(drop=True)
    retained.attrs.update(getattr(prepared_rows, "attrs", {}))
    matrix_indices = np.flatnonzero(keep).astype(
        np.int64, copy=False
    )
    retained.attrs["_janusx_finemap_matrix_indices"] = matrix_indices
    if fold_r2 is not None:
        normalized_fold_r2: dict[
            tuple[str, int], dict[tuple[str, int], float]
        ] = {}
        for lead_key_raw, member_map in fold_r2.items():
            lead_key = (
                _postgwas_finemap_normalize_chr(lead_key_raw[0]),
                int(lead_key_raw[1]),
            )
            normalized_fold_r2[lead_key] = {
                (
                    _postgwas_finemap_normalize_chr(member_key[0]),
                    int(member_key[1]),
                ): float(value)
                for member_key, value in member_map.items()
            }
        retained.attrs["_janusx_finemap_fold_r2"] = normalized_fold_r2
    bim_indices = prepared_rows.attrs.get("_janusx_finemap_bim_indices")
    bim_metadata = prepared_rows.attrs.get("_janusx_finemap_bim_metadata")
    if bim_indices is not None or bim_metadata is not None:
        if (
            not isinstance(bim_indices, list)
            or not isinstance(bim_metadata, list)
            or len(bim_indices) != input_rows
            or len(bim_metadata) != input_rows
        ):
            raise FineMapSkip("raw-route BIM identity metadata does not match prepared rows")
        retained.attrs["_janusx_finemap_bim_indices"] = [
            int(bim_indices[index]) for index in matrix_indices.tolist()
        ]
        retained.attrs["_janusx_finemap_bim_metadata"] = [
            bim_metadata[index] for index in matrix_indices.tolist()
        ]
    retained_rows = int(len(retained))
    counters = {
        "input_rows": input_rows,
        "retained_rows": retained_rows,
        "clumped_rows": max(0, input_rows - retained_rows),
        "groups": int(len(fold_groups)),
    }
    return retained, counters, fold_groups


def _postgwas_finemap_ldclump_from_matrix(
    prepared: pd.DataFrame,
    *,
    ld_matrix: object,
    locus: tuple[str, int, int],
    logger: logging.Logger,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[tuple[str, int], tuple[int, ...]],
]:
    """Clump against an already-built LD matrix in the exact row order."""
    prepared_rows = prepared.reset_index(drop=True)
    input_rows = int(len(prepared_rows))
    matrix = np.asarray(ld_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (input_rows, input_rows):
        raise FineMapSkip(
            "FvLMM effective-LD clumping matrix dimensions do not match GWAS rows"
        )
    if not np.all(np.isfinite(matrix)):
        raise FineMapSkip("FvLMM effective-LD clumping matrix is non-finite")
    if input_rows == 0:
        return _postgwas_finalize_finemap_ldclump(
            prepared_rows,
            prepared_rows.iloc[0:0].set_index(["chrom_norm", "pos"]),
            {},
        )

    required = {"chrom_norm", "pos", "z"}
    missing = sorted(required.difference(prepared_rows.columns))
    if missing:
        raise FineMapSkip(
            "FvLMM effective-LD clumping rows are missing: " + ", ".join(missing)
        )
    priority = -np.abs(
        pd.to_numeric(prepared_rows["z"], errors="coerce").to_numpy(dtype=float)
    )
    if not np.all(np.isfinite(priority)):
        raise FineMapSkip("FvLMM effective-LD clumping priorities are non-finite")
    prepared_chroms = [
        _postgwas_finemap_normalize_chr(value)
        for value in prepared_rows["chrom_norm"].tolist()
    ]
    prepared_positions = pd.to_numeric(
        prepared_rows["pos"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(prepared_positions)) or not np.all(
        prepared_positions == np.floor(prepared_positions)
    ):
        raise FineMapSkip("FvLMM effective-LD clumping positions are invalid")
    prepared_positions = prepared_positions.astype(np.int64).tolist()

    work = pd.DataFrame(
        {
            "chrom_norm": prepared_chroms,
            "pos": prepared_positions,
            "_finemap_priority": priority,
            "_matrix_index": np.arange(input_rows, dtype=np.int64),
        }
    )
    work = (
        work.sort_values("_finemap_priority", ascending=True, kind="mergesort")
        .drop_duplicates(subset=["chrom_norm", "pos"], keep="first")
        .reset_index(drop=True)
    )
    all_keys = [
        (str(chrom), int(pos))
        for chrom, pos in zip(work["chrom_norm"], work["pos"])
    ]
    chrom_arr = work["chrom_norm"].to_numpy(dtype=str)
    pos_arr = work["pos"].to_numpy(dtype=np.int64)
    matrix_indices = work["_matrix_index"].to_numpy(dtype=np.int64)
    remaining = np.ones((len(work),), dtype=bool)
    key_to_p = {
        (str(chrom), int(pos)): float(value)
        for chrom, pos, value in zip(
            work["chrom_norm"], work["pos"], work["_finemap_priority"]
        )
    }
    kept_rows: list[tuple[str, int, float, int, int, int, float, str]] = []
    clump_dict: dict[tuple[str, int], list[tuple[str, int]]] = {}
    fold_r2: dict[tuple[str, int], dict[tuple[str, int], float]] = {}
    window_bp = max(1, abs(int(locus[2]) - int(locus[1])))
    for work_idx, row in work.iterrows():
        if not bool(remaining[int(work_idx)]):
            continue
        lead_chr = str(row["chrom_norm"])
        lead_pos = int(row["pos"])
        start = lead_pos - window_bp
        end = lead_pos + window_bp
        win_idx = np.flatnonzero(
            remaining
            & (chrom_arr == lead_chr)
            & (pos_arr >= start)
            & (pos_arr <= end)
        )
        lead_matrix_index = int(matrix_indices[int(work_idx)])
        candidate_matrix_indices = matrix_indices[win_idx]
        r2 = np.square(matrix[lead_matrix_index, candidate_matrix_indices])
        if r2.shape != (len(win_idx),) or not np.all(np.isfinite(r2)):
            raise FineMapSkip(
                "FvLMM effective-LD clumping returned invalid lead correlations"
            )
        clumped_idx = win_idx[r2 >= float(_POSTGWAS_FINEMAP_LDCLUMP_R2)]
        if int(work_idx) not in set(int(index) for index in clumped_idx.tolist()):
            clumped_idx = np.concatenate(
                (np.asarray([int(work_idx)], dtype=np.int64), clumped_idx)
            )
        clumped = [all_keys[int(index)] for index in clumped_idx.tolist()]
        r2_map = {
            all_keys[int(index)]: float(value)
            for index, value in zip(win_idx.tolist(), r2.tolist())
        }
        fold_r2[(lead_chr, lead_pos)] = {
            key: float(r2_map.get(key, 1.0)) for key in clumped
        }
        mean_r2 = float(
            np.mean([float(r2_map.get(key, 1.0)) for key in clumped])
        )
        clumped = sorted(
            clumped,
            key=lambda key: (float(key_to_p.get(key, np.inf)), int(key[1])),
        )
        clump_dict[(lead_chr, lead_pos)] = clumped
        ld_start = int(min(int(key[1]) for key in clumped))
        ld_end = int(max(int(key[1]) for key in clumped))
        kept_rows.append(
            (
                lead_chr,
                lead_pos,
                float(row["_finemap_priority"]),
                ld_start,
                ld_end,
                len(clumped),
                mean_r2,
                _format_clump_sites(clumped),
            )
        )
        remaining[clumped_idx] = False

    clump_df = pd.DataFrame(
        kept_rows,
        columns=[
            "chrom_norm",
            "pos",
            "_finemap_priority",
            "start",
            "end",
            "nsnps",
            "MeanR2",
            "LDclump",
        ],
    ).set_index(["chrom_norm", "pos"], drop=True)
    logger.info(
        "FvLMM effective-LD clumping used the already-built matrix: rows=%d retained=%d.",
        input_rows,
        len(kept_rows),
    )
    return _postgwas_finalize_finemap_ldclump(
        prepared_rows,
        clump_df,
        clump_dict,
        fold_r2=fold_r2,
    )


def _postgwas_finemap_ldclump(
    prepared: pd.DataFrame,
    *,
    genofile: str,
    locus: tuple[str, int, int],
    logger: logging.Logger,
    preload_max_rows: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    ld_matrix: object = None,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[tuple[str, int], tuple[int, ...]],
]:
    """Clump fine-mapping rows using the route-selected LD source."""
    if ld_matrix is not None:
        return _postgwas_finemap_ldclump_from_matrix(
            prepared,
            ld_matrix=ld_matrix,
            locus=locus,
            logger=logger,
        )
    if sample_ids is None or len(list(sample_ids)) == 0:
        raise FineMapSkip(
            "ordinary LD clumping requires verified GWAS sample metadata"
        )
    prepared_rows = prepared.reset_index(drop=True)
    input_rows = int(len(prepared_rows))
    if input_rows <= 1:
        if input_rows == 0:
            clump_df = prepared_rows.iloc[0:0].set_index(["chrom_norm", "pos"])
            return _postgwas_finalize_finemap_ldclump(prepared_rows, clump_df, {})
        key = (
            _postgwas_finemap_normalize_chr(prepared_rows.iloc[0]["chrom_norm"]),
            int(prepared_rows.iloc[0]["pos"]),
        )
        clump_df = pd.DataFrame(
            {"_finemap_priority": [float(-abs(float(prepared_rows.iloc[0]["z"])))]},
            index=pd.MultiIndex.from_tuples([key], names=["chrom_norm", "pos"]),
        )
        return _postgwas_finalize_finemap_ldclump(
            prepared_rows,
            clump_df,
            {key: [key]},
            fold_r2={key: {key: 1.0}},
        )

    priority = -np.abs(
        pd.to_numeric(prepared_rows["z"], errors="coerce").to_numpy(dtype=float)
    )
    prepared_chroms = [
        _postgwas_finemap_normalize_chr(value)
        for value in prepared_rows["chrom_norm"].tolist()
    ]
    prepared_positions = (
        pd.to_numeric(prepared_rows["pos"], errors="coerce")
        .astype(np.int64)
        .tolist()
    )
    work = pd.DataFrame(
        {
            "chrom_norm": prepared_chroms,
            "pos": prepared_positions,
            "_finemap_priority": priority,
        }
    )
    window_bp = max(1, abs(int(locus[2]) - int(locus[1])))
    fold_r2: dict[tuple[str, int], dict[tuple[str, int], float]] = {}
    clump_df, clump_dict = _ldclump_significant_snps(
        work,
        chr_col="chrom_norm",
        pos_col="pos",
        p_col="_finemap_priority",
        genofile=str(genofile),
        window_bp=window_bp,
        r2_thr=float(_POSTGWAS_FINEMAP_LDCLUMP_R2),
        logger=logger,
        show_progress=False,
        preload_max_rows=preload_max_rows,
        sample_ids=sample_ids,
        fold_r2=fold_r2,
    )
    return _postgwas_finalize_finemap_ldclump(
        prepared_rows,
        clump_df,
        clump_dict,
        fold_r2=fold_r2,
    )


def _draw_empty_ldblock(
    ax: plt.Axes,
    *,
    n_sites: int = 2,
    text: Optional[str] = None,
    font_size: float = _POSTGWAS_DEFAULT_FONT_SIZE,
) -> None:
    n = max(2, int(n_sites))
    LDblock(np.zeros((n, n), dtype=np.float32), ax=ax, vmin=0, vmax=1, cmap="Greys")
    # Keep LD triangle body filling the whole panel width.
    ax.set_xlim(0.5, float(n) - 0.5)
    ax.margins(x=0.0)
    if text:
        ax.text(
            n / 2.0,
            -n / 2.0,
            text,
            ha="center",
            va="center",
            fontsize=float(font_size),
        )


def _ld_min_height_over_width(n_sites: int) -> float:
    """
    Minimal LD panel height/width ratio that keeps LDblock (aspect=0.5)
    from shrinking panel width for small SNP counts under xlim=0.5..n-0.5.
    """
    n = max(2, int(n_sites))
    return 0.5 * (float(n) / float(n - 1))


def _build_layout_from_bimrange_tuples(
    bimrange_tuples: list[tuple[str, int, int]],
    *,
    interval_ratio: float = 0.5,
) -> list[dict[str, object]]:
    if len(bimrange_tuples) == 0:
        return []
    seg_defs: list[dict[str, object]] = []
    for i, (chrom, start, end) in enumerate(bimrange_tuples):
        seg_defs.append(
            {
                "id": int(i),
                "chrom": str(chrom),
                "chrom_norm": _normalize_chr(chrom),
                "start": int(start),
                "end": int(end),
                "length": float(max(1, int(end) - int(start))),
            }
        )
    return _build_bimrange_layout(seg_defs, interval_ratio=float(interval_ratio))


def _postgwas_project_susie_locus_x(
    chrom: object,
    pos: object,
    bimrange_tuples: Sequence[tuple[str, int, int]],
    layout: Sequence[dict[str, object]],
) -> np.ndarray:
    """Project locus positions into the local-Manhattan ``bimrange`` x-axis."""
    chrom_values = np.asarray([_normalize_chr(value) for value in list(chrom)], dtype=object)
    pos_values = pd.to_numeric(pd.Series(list(pos)), errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    if chrom_values.shape != pos_values.shape:
        raise ValueError("SuSiE locus plot chromosome and position lengths differ.")
    if len(layout) != len(bimrange_tuples):
        raise ValueError("SuSiE locus plot layout does not match bimrange tuples.")
    x_values = np.empty(pos_values.shape[0], dtype=np.float64)
    for row_index, (chrom_value, pos_value) in enumerate(
        zip(chrom_values.tolist(), pos_values.tolist())
    ):
        if not np.isfinite(pos_value) or pos_value != np.floor(pos_value):
            raise ValueError("SuSiE locus plot position is not a finite integer.")
        matches: list[int] = []
        for segment_index, (segment, bimrange) in enumerate(
            zip(layout, bimrange_tuples)
        ):
            segment_chrom = _normalize_chr(segment.get("chrom", bimrange[0]))
            start = int(bimrange[1])
            end = int(bimrange[2])
            if segment_chrom == chrom_value and start <= int(pos_value) <= end:
                matches.append(segment_index)
        if len(matches) == 0:
            raise ValueError(
                "SuSiE locus plot position is outside requested bimrange: "
                f"{chrom_value}:{int(pos_value)}"
            )
        segment = layout[matches[0]]
        start = int(bimrange_tuples[matches[0]][1])
        if len(layout) == 1:
            x_values[row_index] = float(pos_value)
            continue
        length = float(segment.get("length", int(bimrange_tuples[matches[0]][2]) - start))
        relative = min(max(float(pos_value) - float(start), 0.0), max(length, 1.0))
        x_values[row_index] = float(segment["offset"]) + relative
    return x_values


def _postgwas_susie_locus_axis_limits(
    bimrange_tuples: Sequence[tuple[str, int, int]],
    layout: Sequence[dict[str, object]],
) -> tuple[float, float]:
    """Return the exact x limits used by local Manhattan for this layout."""
    if len(layout) == 0 or len(layout) != len(bimrange_tuples):
        raise ValueError("SuSiE locus plot layout/range dimensions differ.")
    if len(layout) == 1:
        return (
            float(bimrange_tuples[0][1]),
            float(layout[0].get("end", bimrange_tuples[0][2])),
        )
    return (float(layout[0]["x_start"]), float(layout[-1]["x_end"]))


def _postgwas_susie_locus_palette(
    cs_names: Sequence[str],
    palette_spec: Optional[Tuple[str, Any]] = None,
) -> dict[str, str]:
    """Return deterministic colors keyed by numeric CS name.

    SuSiE credible sets use the same ordered palette semantics as other
    post-GWAS series.  With no explicit ``-palette``, the native Matplotlib
    ``tab10`` colors are used in the order of the retained CS labels.  A
    filtered-out CS therefore does not consume a palette slot.
    """
    unique = {str(name) for name in cs_names if str(name).strip() != ""}

    def _cs_sort_key(value: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", value)
        return (int(match.group(1)), value) if match else (10**9, value)

    ordered = sorted(unique, key=_cs_sort_key)
    if len(ordered) == 0:
        return {}
    if palette_spec is None:
        cmap = plt.get_cmap("tab10")
        colors = [
            mcolors.to_hex(cmap(index % max(1, int(getattr(cmap, "N", 10)))))
            for index in range(len(ordered))
        ]
    else:
        colors = _resolve_merge_series_colors(palette_spec, len(ordered))
    if len(colors) == 0:
        cmap = plt.get_cmap("tab10")
        colors = [
            mcolors.to_hex(cmap(index % max(1, int(getattr(cmap, "N", 10)))))
            for index in range(len(ordered))
        ]
    mapping: dict[str, str] = {}
    for index, name in enumerate(ordered):
        mapping[name] = str(colors[index % len(colors)])
    return mapping


def _postgwas_susie_finemap_output_paths(
    args: argparse.Namespace,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return PIP output paths and optional plot paths for one fine-map run."""
    out_dir = str(getattr(args, "out", ".") or ".")
    out_stem = str(getattr(args, "prefix", "JanusX") or "JanusX").strip() or "JanusX"
    output_path = os.path.join(out_dir, f"{out_stem}.susie.pip.tsv")
    final_paths: list[str] = [output_path]
    temporary_paths: list[str] = [f"{output_path}.tmp"]
    if getattr(args, "manh_ratio", None) is not None:
        figure_path = os.path.join(
            out_dir,
            f"{out_stem}.susie.locus.{str(getattr(args, 'format', 'png')).lower()}",
        )
        final_paths.append(figure_path)
        temporary_paths.append(_postgwas_finemap_figure_temp_path(figure_path))
    return output_path, tuple(final_paths), tuple(temporary_paths)


def _postgwas_build_susie_locus_plot_record(
    prepared_source: pd.DataFrame,
    locus_output: pd.DataFrame,
    fitted_output_rows: pd.DataFrame,
    pip: object,
    *,
    locus: str,
    bimrange_tuples: Sequence[tuple[str, int, int]],
    interval_ratio: float = 0.5,
    palette_spec: Optional[Tuple[str, Any]] = None,
    scatter_size: float = _DEFAULT_SCATTER_SIZE,
    layout: Optional[Sequence[dict[str, object]]] = None,
) -> dict[str, object]:
    """Build compact PIP plot data without retaining genotype or LD matrices.

    The plot contains every prepared GWAS row. Fitted representatives are Tag
    SNPs and receive the fitted PIP; restored clump members are proxy points
    that inherit their Tag PIP for visualization only. Rows outside a retained
    CS remain neutral background points, retaining their fitted PIP when one
    exists and otherwise using zero.
    """
    prepared = prepared_source.reset_index(drop=True)
    output = locus_output.reset_index(drop=True)
    fitted = fitted_output_rows.reset_index(drop=True)
    required_prepared = {"chrom", "pos", "snp"}
    missing = sorted(required_prepared.difference(prepared.columns))
    if missing:
        raise ValueError("SuSiE locus plot input is missing: " + ", ".join(missing))
    if not {"chrom", "pos", "snp"}.issubset(fitted.columns):
        raise ValueError("SuSiE locus plot fitted rows are missing metadata.")
    pip_values = np.asarray(pip, dtype=np.float64).reshape(-1)
    if pip_values.shape != (len(fitted),) or not np.all(np.isfinite(pip_values)):
        raise ValueError("SuSiE locus plot fitted PIP values are invalid.")
    if np.any((pip_values < 0.0) | (pip_values > 1.0)):
        raise ValueError("SuSiE locus plot fitted PIP values must be in [0, 1].")
    scatter_size_value = float(scatter_size)
    if not np.isfinite(scatter_size_value) or scatter_size_value <= 0.0:
        raise ValueError("SuSiE locus plot scatter size must be finite and > 0.")

    annotation_by_key: dict[tuple[str, int], pd.Series] = {}
    if not output.empty:
        for _, row in output.iterrows():
            key = _postgwas_finemap_row_key(row)
            if key in annotation_by_key:
                raise ValueError(
                    "SuSiE locus plot output contains duplicate coordinates: "
                    f"{key[0]}:{key[1]}"
                )
            annotation_by_key[key] = row

    def _annotation_value(row: pd.Series, name: str) -> object:
        value = row.get(name, pd.NA)
        return value

    def _cs_name(value: object) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        text_value = str(value).strip()
        return text_value if text_value else None

    point_keys = [
        _postgwas_finemap_row_key(row)
        for _, row in prepared.iterrows()
    ]
    point_chrom = prepared["chrom"].tolist()
    point_pos = pd.to_numeric(prepared["pos"], errors="raise").astype(np.int64).tolist()
    if layout is None:
        layout = _build_layout_from_bimrange_tuples(
            [(str(chrom), int(start), int(end)) for chrom, start, end in bimrange_tuples],
            interval_ratio=float(interval_ratio),
        )
    else:
        layout = [dict(segment) for segment in layout]
    point_x = _postgwas_project_susie_locus_x(
        point_chrom, point_pos, bimrange_tuples, layout
    )
    point_cs: list[Optional[str]] = []
    point_roles: list[str] = []
    point_pip = np.zeros(len(prepared), dtype=np.float64)
    point_rsqr = np.full(len(prepared), np.nan, dtype=np.float64)
    fitted_by_key: dict[tuple[str, int], int] = {}
    fitted_by_snp: dict[str, list[tuple[str, int]]] = {}
    for index, (_, row) in enumerate(fitted.iterrows()):
        fitted_key = _postgwas_finemap_row_key(row)
        if fitted_key in fitted_by_key:
            raise ValueError(
                "SuSiE locus plot fitted rows contain duplicate coordinates: "
                f"{fitted_key[0]}:{fitted_key[1]}"
            )
        fitted_by_key[fitted_key] = int(index)
        fitted_by_snp.setdefault(str(row["snp"]), []).append(fitted_key)
    for row_index, key in enumerate(point_keys):
        annotation = annotation_by_key.get(key)
        cs = _cs_name(_annotation_value(annotation, "cs_set")) if annotation is not None else None
        tag_key = None
        tag_snp: Optional[str] = None
        if annotation is not None:
            tag_value = _annotation_value(annotation, "tag")
            if tag_value is not None and not pd.isna(tag_value):
                tag_snp = str(tag_value).strip()
                if tag_snp == "":
                    tag_snp = None
            if cs is not None and key not in fitted_by_key and tag_snp is None:
                raise ValueError(
                    "SuSiE locus plot proxy Tag does not match exactly one fitted Tag: "
                    f"{key[0]}:{key[1]} has no Tag SNP"
                )
            if cs is not None and key not in fitted_by_key and tag_snp is not None:
                matching = fitted_by_snp.get(tag_snp, [])
                if len(matching) > 1:
                    raise ValueError(
                        "SuSiE locus plot tag SNP is ambiguous in fitted rows: "
                        f"{tag_snp}"
                    )
                if len(matching) != 1:
                    raise ValueError(
                        "SuSiE locus plot proxy Tag does not match exactly one fitted Tag: "
                        f"{tag_snp}"
                    )
                tag_key = matching[0]
        is_tag = cs is not None and key in fitted_by_key
        if not is_tag and annotation is not None and tag_key is not None:
            is_tag = cs is not None and key == tag_key
        role = "tag" if is_tag else ("proxy" if cs is not None else "non_cs")
        if role == "proxy" and annotation is not None and tag_key is not None:
            point_pip[row_index] = float(pip_values[fitted_by_key[tag_key]])
        elif key in fitted_by_key:
            point_pip[row_index] = float(pip_values[fitted_by_key[key]])
        elif annotation is not None:
            annotation_pip = pd.to_numeric(
                pd.Series([_annotation_value(annotation, "pip")]), errors="coerce"
            ).to_numpy(dtype=float)[0]
            if np.isfinite(annotation_pip):
                point_pip[row_index] = float(np.clip(annotation_pip, 0.0, 1.0))
        point_roles.append(role)
        point_cs.append(cs)
        if annotation is not None:
            rsqr_value = pd.to_numeric(
                pd.Series([_annotation_value(annotation, "rsqr")]), errors="coerce"
            ).to_numpy(dtype=float)[0]
            if np.isfinite(rsqr_value):
                point_rsqr[row_index] = float(np.clip(rsqr_value, 0.0, 1.0))

    cs_colors = _postgwas_susie_locus_palette(
        [name for name in point_cs if name is not None],
        palette_spec,
    )

    def _color_for(cs: Optional[str]) -> str:
        return cs_colors.get(cs, _POSTGWAS_SUSIE_LOCUS_NEUTRAL_COLOR)

    point_marker = ["D" if role == "tag" else "o" for role in point_roles]
    point_size = np.asarray(
        [
            scatter_size_value if role == "tag" else scatter_size_value / 3.0
            for role in point_roles
        ],
        dtype=np.float64,
    )
    return {
        "locus": str(locus),
        "layout": layout,
        "bimrange_tuples": [tuple(item) for item in bimrange_tuples],
        "cs_colors": cs_colors,
        "points": {
            "x": point_x,
            "pip": point_pip,
            "color": [_color_for(value) for value in point_cs],
            "size": point_size,
            "marker": point_marker,
            "snp": [str(value) for value in prepared["snp"].tolist()],
            "cs": point_cs,
            "role": point_roles,
            "rsqr": point_rsqr,
        },
    }


def _postgwas_plot_susie_locus_records(
    records: Sequence[dict[str, object]],
    args: argparse.Namespace,
    path: str,
    *,
    logger: logging.Logger,
) -> None:
    """Render the single-panel PIP SuSiE locus figure."""
    if len(records) == 0:
        raise ValueError("SuSiE locus plot requires at least one record.")
    _apply_postgwas_matplotlib_style(args)
    first = records[0]
    layout = list(first["layout"])
    ranges = list(first["bimrange_tuples"])
    if len(layout) != len(ranges):
        raise ValueError("SuSiE locus plot layout/range dimensions differ.")
    for record in records[1:]:
        if record["bimrange_tuples"] != first["bimrange_tuples"]:
            raise ValueError("SuSiE locus plot records do not share bimrange coordinates.")

    manh_ratio = getattr(args, "manh_ratio", None)
    if manh_ratio is None:
        raise ValueError("SuSiE locus plotting requires -manh/--manh.")
    manh_ratio = float(manh_ratio)
    if not np.isfinite(manh_ratio) or manh_ratio <= 0.0:
        raise ValueError("SuSiE locus plotting requires a finite positive Manhattan ratio.")
    fig, ax_pip, _width, _height = _create_ratio_panel_figure(
        ratio=manh_ratio,
        dpi=300,
        panel_width_in=float(_PANEL_WIDTH_IN),
        reserve_right_in=1.25,
    )
    all_cs_names = [
        str(cs_name)
        for record in records
        for panel in (record["points"],)
        for cs_name in list(panel["cs"])
        if cs_name is not None and not pd.isna(cs_name)
    ]
    global_cs_colors = _postgwas_susie_locus_palette(
        all_cs_names,
        getattr(args, "palette_spec", None),
    )
    legend_handles: dict[str, Patch] = {}
    for record in records:
        points = record["points"]
        for role, marker in (("non_cs", "o"), ("proxy", "o"), ("tag", "D")):
            indices = np.asarray(
                [index for index, value in enumerate(points["role"]) if value == role],
                dtype=np.int64,
            )
            if indices.size == 0:
                continue
            colors = [
                global_cs_colors.get(str(points["cs"][index]), _POSTGWAS_SUSIE_LOCUS_NEUTRAL_COLOR)
                if points["cs"][index] is not None and not pd.isna(points["cs"][index])
                else _POSTGWAS_SUSIE_LOCUS_NEUTRAL_COLOR
                for index in indices.tolist()
            ]
            ax_pip.scatter(
                np.asarray(points["x"], dtype=float)[indices],
                np.asarray(points["pip"], dtype=float)[indices],
                s=np.asarray(points["size"], dtype=float)[indices],
                c=colors,
                marker=marker,
                alpha=0.92 if role != "non_cs" else 0.75,
                edgecolors="none",
                linewidths=0.0,
                rasterized=False,
                zorder=4 if role == "tag" else (3 if role == "proxy" else 2),
            )
        for cs_name, color in global_cs_colors.items():
            legend_handles.setdefault(
                str(cs_name),
                Patch(facecolor=str(color), edgecolor="none", label=str(cs_name)),
            )

    ax_pip.set_ylabel("PIP")
    ax_pip.set_xlabel("Genomic position")
    pip_values = np.concatenate(
        [
            np.asarray(record["points"]["pip"], dtype=np.float64).reshape(-1)
            for record in records
        ]
    )
    finite_pip = pip_values[np.isfinite(pip_values)]
    pip_y_min = 0.0
    if finite_pip.size > 0 and float(np.min(finite_pip)) <= _POSTGWAS_SUSIE_PIP_FLOOR_THRESHOLD:
        pip_y_min = _POSTGWAS_SUSIE_PIP_Y_MIN_WITH_HEADROOM
    pip_y_max = 1.0
    if finite_pip.size > 0 and float(np.max(finite_pip)) >= _POSTGWAS_SUSIE_PIP_HEADROOM_THRESHOLD:
        pip_y_max = _POSTGWAS_SUSIE_PIP_Y_MAX_WITH_HEADROOM
    ax_pip.set_ylim(pip_y_min, pip_y_max)
    ax_pip.grid(axis="y", color="#D9D9D2", linewidth=0.6, alpha=0.7)
    if legend_handles:
        ax_pip.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            title="Credible set",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
    if len(ranges) == 1:
        chrom, start, end = ranges[0]
        _apply_bimrange_manhattan_axis(
            ax_pip,
            chrom,
            int(start),
            int(layout[0].get("end", end)),
        )
    else:
        _apply_multi_bimrange_manhattan_axis(
            ax_pip,
            layout,
            label_fontsize=float(getattr(args, "_postgwas_base_fontsize", 8.0)),
        )
    axis_limits = _postgwas_susie_locus_axis_limits(ranges, layout)
    ax_pip.set_xlim(axis_limits)
    os.makedirs(os.path.dirname(path) or ".", mode=0o755, exist_ok=True)
    _save_figure_and_close(fig, path)
    logger.info("SuSiE locus figure: %s", format_path_for_display(path))


def _postgwas_finemap_figure_temp_path(path: str) -> str:
    """Keep the requested image extension while making a sibling temp file."""
    stem, extension = os.path.splitext(str(path))
    return f"{stem}.tmp{extension}"


def _load_gene_like_records_from_anno(
    annofile: str,
    bimrange_tuples: list[tuple[str, int, int]],
    logger: logging.Logger,
    annotation_kind: Optional[str] = None,
    gff_query: Optional[GFFQuery] = None,
    gff_rust_index: Optional[object] = None,
) -> pd.DataFrame:
    """
    Load gene-structure-like records from GFF/BED for selected bimranges.

    Output columns:
      chrom_norm, feature, start, end, strand, attribute
    """
    out_cols = ["chrom_norm", "feature", "start", "end", "strand", "attribute"]
    if not annofile or len(bimrange_tuples) == 0:
        return pd.DataFrame(columns=out_cols)

    features = ["gene", "five_prime_UTR", "three_prime_UTR", "CDS"]

    if _postgwas_annotation_is_gff(annofile, annotation_kind=annotation_kind):
        if gff_rust_index is not None:
            try:
                chroms = [str(x[0]) for x in bimrange_tuples]
                starts = [int(x[1]) for x in bimrange_tuples]
                ends = [int(x[2]) for x in bimrange_tuples]
                (
                    out_chroms,
                    out_features,
                    out_starts,
                    out_ends,
                    out_strands,
                    out_ids,
                ) = gff_rust_index.fetch_gene_panel_ranges(
                    chroms,
                    starts,
                    ends,
                )
                if len(out_starts) == 0:
                    return pd.DataFrame(columns=out_cols)
                out = pd.DataFrame(
                    {
                        "chrom_norm": pd.Series(out_chroms, dtype=object).map(_normalize_chr),
                        "feature": pd.Series(out_features, dtype=object),
                        "start": np.asarray(out_starts, dtype=np.int64),
                        "end": np.asarray(out_ends, dtype=np.int64),
                        "strand": pd.Series(out_strands, dtype=object),
                        "attribute": [[str(x)] for x in out_ids],
                    }
                )
                return out.loc[:, out_cols]
            except Exception as e:
                logger.warning(
                    "Warning: Rust GFF gene-panel range query failed; "
                    f"falling back to Python GFFQuery ({e})."
                )
        q = gff_query if gff_query is not None else GFFQuery.from_file(annofile)
        chunks: list[pd.DataFrame] = []
        for chrom, start, end in bimrange_tuples:
            hit = q.query_range(
                chrom=chrom,
                start=int(start),
                end=int(end),
                features=features,
                attr="ID",
            )
            if hit.shape[0] == 0:
                continue
            chunk = hit.loc[:, ["chrom_norm", "feature", "start", "end", "strand", "attribute"]].copy()
            chunks.append(chunk)
        if len(chunks) == 0:
            return pd.DataFrame(columns=out_cols)
        out = pd.concat(chunks, axis=0, ignore_index=True)
        out["start"] = pd.to_numeric(out["start"], errors="coerce").astype("Int64")
        out["end"] = pd.to_numeric(out["end"], errors="coerce").astype("Int64")
        out = out.dropna(subset=["start", "end"]).copy()
        out["start"] = out["start"].astype(int)
        out["end"] = out["end"].astype(int)
        return out[out_cols]

    bed = bedreader(annofile)
    if bed.shape[0] == 0:
        return pd.DataFrame(columns=out_cols)
    s = pd.to_numeric(bed[1], errors="coerce")
    e = pd.to_numeric(bed[2], errors="coerce")
    valid = s.notna() & e.notna()
    if not bool(valid.any()):
        return pd.DataFrame(columns=out_cols)

    bed_v = bed.loc[valid].copy()
    s_v = s.loc[valid].astype(int)
    e_v = e.loc[valid].astype(int)
    starts = np.minimum(s_v.to_numpy(dtype=np.int64), e_v.to_numpy(dtype=np.int64))
    ends = np.maximum(s_v.to_numpy(dtype=np.int64), e_v.to_numpy(dtype=np.int64))
    chroms = bed_v[0].astype(str).map(_normalize_chr).to_numpy(dtype=object)
    if 3 in bed_v.columns:
        names = bed_v[3].astype(str).to_numpy(dtype=object)
    else:
        names = np.array([""] * len(bed_v), dtype=object)

    rows: list[dict[str, object]] = []
    for chrom, start, end, name in zip(chroms, starts, ends, names):
        gene_id = str(name).strip()
        if gene_id == "" or gene_id.lower() == "nan":
            gene_id = f"{chrom}:{int(start)}-{int(end)}"
        attr = [gene_id]
        # BED-like text has no canonical feature segmentation; build gene+CDS proxy.
        # Strand is explicitly kept as '.' per requirement.
        rows.append(
            {
                "chrom_norm": str(chrom),
                "feature": "gene",
                "start": int(start),
                "end": int(end),
                "strand": ".",
                "attribute": attr,
            }
        )
        rows.append(
            {
                "chrom_norm": str(chrom),
                "feature": "CDS",
                "start": int(start),
                "end": int(end),
                "strand": ".",
                "attribute": attr,
            }
        )
    return pd.DataFrame(rows, columns=out_cols)


def _project_gene_records_to_plot_x(
    records: pd.DataFrame,
    bimrange_tuples: list[tuple[str, int, int]],
    layout: list[dict[str, object]],
    *,
    use_segmented_x: bool,
) -> pd.DataFrame:
    """
    Clip records to selected bimranges and project start/end into plot-x coordinates.
    """
    out_cols = ["feature", "strand", "attribute", "x_start", "x_end"]
    if records.shape[0] == 0 or len(bimrange_tuples) == 0:
        return pd.DataFrame(columns=out_cols)

    seg_by_key: dict[tuple[str, int], dict[str, object]] = {}
    if use_segmented_x:
        for seg in layout:
            sid = int(seg["id"])
            seg_by_key[(str(seg["chrom_norm"]), sid)] = seg

    rows: list[dict[str, object]] = []
    for _, row in records.iterrows():
        chrom = _normalize_chr(row["chrom_norm"])
        r_start = int(min(int(row["start"]), int(row["end"])))
        r_end = int(max(int(row["start"]), int(row["end"])))
        feature = str(row["feature"])
        strand = str(row["strand"])
        attr = row["attribute"]

        for sid, (bchrom, bstart, bend) in enumerate(bimrange_tuples):
            bchrom_norm = _normalize_chr(bchrom)
            if chrom != bchrom_norm:
                continue
            ov_start = max(r_start, int(bstart))
            ov_end = min(r_end, int(bend))
            if ov_end < ov_start:
                continue
            if use_segmented_x:
                seg = seg_by_key.get((bchrom_norm, int(sid)))
                if seg is None:
                    continue
                offset = float(seg["offset"])
                seg_start = int(seg["start"])
                x_start = offset + float(ov_start - seg_start)
                x_end = offset + float(ov_end - seg_start)
            else:
                x_start = float(ov_start)
                x_end = float(ov_end)
            rows.append(
                {
                    "feature": feature,
                    "strand": strand,
                    "attribute": attr,
                    "x_start": float(min(x_start, x_end)),
                    "x_end": float(max(x_start, x_end)),
                }
            )
    if len(rows) == 0:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows, columns=out_cols)


def _draw_gene_structure_axis(
    ax: plt.Axes,
    gene_df: pd.DataFrame,
    *,
    arrow_color: str = "black",
    block_color: str = "grey",
    line_width: float = 0.5,
    arrow_step: float = 1_000.0,
    thickness_scale: float = 1.0,
    y_offset: float = 0.0,
    gene_text_size: float = _POSTGWAS_DEFAULT_FONT_SIZE,
) -> None:
    """
    Draw gene/CDS/UTR structure into `ax` using projected x coordinates.
    """
    gene_df_plot = gene_df.copy()
    if "attribute" in gene_df_plot.columns:
        gene_df_plot["attribute"] = gene_df_plot["attribute"].map(_sanitize_plot_text)

    draw_gene_structure_records(
        ax,
        gene_df_plot,
        arrow_color=arrow_color,
        block_color=block_color,
        line_width=line_width,
        arrow_step=arrow_step,
        gene_text_size=float(gene_text_size),
        thickness_scale=thickness_scale,
        y_offset=y_offset,
        unknown_strand_as_plus=False,
        label_bbox={
            "facecolor": "white",
            "alpha": 0.55,
            "edgecolor": "none",
            "pad": 0.2,
        },
    )


def _draw_manh_gene_ld_links(
    fig: plt.Figure,
    ax_gene: plt.Axes,
    ax_manh: plt.Axes,
    ax_ld: plt.Axes,
    pairs: list[tuple[float, float, bool]],
    *,
    gene_route_y: float = -0.08,
    force_line_color: Optional[str] = None,
    nonsig_line_color: str = "grey",
) -> None:
    """
    Draw connectors:
      1) Manhattan bottom -> routing lane below gene structure
      2) gene bottom routing lane -> LD triangle at the same SNP / LD x
    Significant SNP lines are red; non-significant SNP lines use
    `nonsig_line_color`.
    """
    if len(pairs) == 0:
        return
    gx0, gx1 = ax_gene.get_xlim()
    y_manh_ref = float(ax_manh.get_ylim()[0])
    tx0, tx1 = ax_manh.get_xlim()
    dt = float(tx1 - tx0)
    if np.isclose(dt, 0.0):
        return

    edge_margin_n = 0.004
    for x_top, x_ld, is_sig in pairs:
        if not (np.isfinite(x_top) and np.isfinite(x_ld)):
            continue
        # Map Manhattan x into gene-axis data coordinates.
        x_top_n = (float(x_top) - tx0) / dt
        if not np.isfinite(x_top_n):
            continue
        x_top_n = float(np.clip(x_top_n, edge_margin_n, 1.0 - edge_margin_n))
        x_top_g = gx0 + x_top_n * float(gx1 - gx0)
        if force_line_color is not None:
            line_color = str(force_line_color)
        else:
            line_color = "red" if bool(is_sig) else str(nonsig_line_color)
        upper = ConnectionPatch(
            xyA=(float(x_top), y_manh_ref),
            xyB=(x_top_g, float(gene_route_y)),
            coordsA=ax_manh.transData,
            coordsB=ax_gene.transData,
            color=line_color,
            linewidth=0.35,
            alpha=0.9,
            clip_on=False,
            zorder=0.2,
        )
        ax_gene.add_artist(upper)
        # Draw diagonal segment in real cross-axes geometry so mapping
        # remains correct even if LD panel width is narrower than middle panel.
        # Start from a dedicated routing lane below the gene structure so
        # connectors do not cover the gene body/label.
        diag = ConnectionPatch(
            xyA=(x_top_g, float(gene_route_y)),
            xyB=(float(x_ld), float(_POSTGWAS_LD_LINK_Y)),
            coordsA=ax_gene.transData,
            coordsB=ax_ld.transData,
            color=line_color,
            linewidth=0.35,
            alpha=0.9,
            clip_on=False,
            zorder=10.0,
        )
        ax_ld.add_artist(diag)


def _format_input_files(files: list[str]) -> str:
    if len(files) == 0:
        return "0 files"
    if len(files) == 1:
        return str(files[0])
    return f"{len(files)} files"


def _format_bimrange_summary(
    bimranges: Optional[list[tuple[str, int, int]]]
) -> str:
    if bimranges is None or len(bimranges) == 0:
        return "None"
    if len(bimranges) == 1:
        return _format_bimrange_tuple(bimranges[0])
    return f"{_format_bimrange_tuple(bimranges[0])},...({len(bimranges)} ranges)"


def _postgwas_highlight_rows(
    plotmodel: GWASPLOT,
    path: str,
) -> tuple[np.ndarray, dict[tuple[str, int], str], list[object]]:
    """Resolve highlight coordinates against every matching plotted row.

    GARFIELD tables can contain several rows at the same endpoint (for
    example, a singleton and one or more logical-rule endpoints).  Matching
    by a boolean coordinate mask keeps all of those rows visible while using
    one label per requested coordinate.
    """
    raw = pd.read_csv(path, sep="\t", header=None, comment="#")
    if raw.shape[1] < 2:
        raise ValueError(
            "Highlight file must contain at least chromosome and position columns."
        )

    labels: dict[tuple[str, int], str] = {}
    for row in raw.itertuples(index=False, name=None):
        try:
            key = (_normalize_chr(row[0]), int(float(row[1])))
        except (TypeError, ValueError):
            continue
        label = ""
        if len(row) >= 4 and row[3] is not None and not pd.isna(row[3]):
            label = str(row[3]).strip()
        if label == "":
            label = f"{key[0]}_{key[1]}"
        labels.setdefault(key, label)

    chr_by_id = getattr(plotmodel, "_chr_label_by_id", {})
    keep = np.zeros(len(plotmodel.df.index), dtype=bool)
    for i, index in enumerate(plotmodel.df.index):
        try:
            raw_chrom = index[0]
            try:
                chrom = chr_by_id.get(int(raw_chrom), raw_chrom)
            except (TypeError, ValueError):
                chrom = raw_chrom
            key = (_normalize_chr(chrom), int(index[1]))
        except (TypeError, ValueError, IndexError):
            key = ("", -1)
        keep[i] = key in labels

    highlight_index = [index for index, is_keep in zip(plotmodel.df.index, keep) if is_keep]
    return keep, labels, highlight_index


def _overlay_manhattan_threshold_points(
    ax: plt.Axes,
    plotmodel: GWASPLOT,
    *,
    threshold: float,
    draw_threshold_line: bool = True,
    base_size: float,
    marker: str,
    rasterized: bool,
    alpha_override: Optional[float] = None,
    min_logp: float = 0.5,
    max_logp: Optional[float] = None,
    ignore: Optional[list[object]] = None,
) -> None:
    """
    Overlay threshold line and significant points on top of Manhattan base points.
    Significant points are enlarged by 1.5x.
    """
    if not np.isfinite(threshold):
        return

    if ignore is None:
        ignore = []
    ignore_set = set(ignore)

    dfp = plotmodel.df.iloc[plotmodel.minidx, -3:].copy()
    pvals = pd.to_numeric(dfp["y"], errors="coerce")
    no_logtrans = bool(getattr(plotmodel, "pvalue_is_log10", False))
    keep = _postgwas_valid_p_mask(pvals, no_logtrans=no_logtrans)
    if not bool(keep.any()):
        return
    dfp = dfp.loc[keep].copy()
    dfp["ylog"] = _postgwas_plot_logp_values(
        dfp["y"],
        no_logtrans=no_logtrans,
    )
    dfp = dfp[dfp["ylog"] >= float(min_logp)]
    if max_logp is not None:
        dfp = dfp[dfp["ylog"] <= float(max_logp)]
    if dfp.shape[0] == 0:
        return

    thr_log = _postgwas_threshold_to_logp(
        threshold,
        no_logtrans=no_logtrans,
    )
    if not np.isfinite(thr_log):
        return

    sig_mask = dfp["ylog"] >= thr_log
    if len(ignore_set) > 0:
        sig_mask = sig_mask & (~dfp.index.isin(ignore_set))

    if bool(sig_mask.any()):
        ax.scatter(
            dfp.loc[sig_mask, "x"],
            dfp.loc[sig_mask, "ylog"],
            color="red",
            marker=str(marker),
            s=float(base_size) * 1.5,
            alpha=(
                float(alpha_override)
                if alpha_override is not None
                else 0.85
            ),
            rasterized=rasterized,
            zorder=6,
            **_marker_scatter_style(str(marker)),
        )
    if bool(draw_threshold_line):
        ax.axhline(
            y=thr_log,
            linestyle="dashed",
            color="grey",
            linewidth=1.0,
        )


def _postgwas_logic_combo_mask(df: pd.DataFrame) -> np.ndarray:
    if df is None or df.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    combo_mask = np.zeros(df.shape[0], dtype=bool)
    if "row_role" in df.columns:
        row_role = df["row_role"].astype(str).str.strip().str.lower()
        combo_mask |= row_role.eq("combo").to_numpy(dtype=bool, copy=False)
    if "snp" in df.columns:
        snp = df["snp"].astype(str)
        combo_mask |= snp.str.contains(r"[&|*]", regex=True, na=False).to_numpy(
            dtype=bool,
            copy=False,
        )
    return combo_mask


def _overlay_manhattan_interaction_padj_points(
    ax: plt.Axes,
    plotmodel: GWASPLOT,
    *,
    threshold: float,
    draw_threshold_line: bool = True,
    base_size: float,
    marker: str,
    rasterized: bool,
    alpha_override: Optional[float] = None,
    min_logp: float = 0.5,
    max_logp: Optional[float] = None,
    ignore: Optional[list[object]] = None,
    padj_cutoff: float = _INTERACTION_SIG_PADJ_DEFAULT,
) -> bool:
    df_full = plotmodel.df.iloc[plotmodel.minidx].copy()
    if (
        df_full.shape[0] == 0
        or "padj" not in df_full.columns
        or ("snp" not in df_full.columns and "row_role" not in df_full.columns)
    ):
        return False

    combo_mask_full = _postgwas_logic_combo_mask(df_full)
    if combo_mask_full.size == 0 or not bool(np.any(combo_mask_full)):
        return False

    if ignore is None:
        ignore = []
    ignore_set = set(ignore)

    pvals = pd.to_numeric(df_full["y"], errors="coerce")
    no_logtrans = bool(getattr(plotmodel, "pvalue_is_log10", False))
    keep = _postgwas_valid_p_mask(pvals, no_logtrans=no_logtrans)
    if not bool(keep.any()):
        return True
    dfp = df_full.loc[keep].copy()
    dfp["ylog"] = _postgwas_plot_logp_values(
        dfp["y"],
        no_logtrans=no_logtrans,
    )
    dfp = dfp[dfp["ylog"] >= float(min_logp)]
    if max_logp is not None:
        dfp = dfp[dfp["ylog"] <= float(max_logp)]
    if dfp.shape[0] == 0:
        return True

    combo_mask = _postgwas_logic_combo_mask(dfp)
    padj = pd.to_numeric(dfp["padj"], errors="coerce")
    sig_mask = combo_mask & np.isfinite(padj.to_numpy(dtype=float, copy=False))
    sig_mask &= padj.to_numpy(dtype=float, copy=False) <= float(padj_cutoff)
    if len(ignore_set) > 0:
        sig_mask = sig_mask & (~dfp.index.isin(ignore_set))

    if bool(np.any(sig_mask)):
        ax.scatter(
            dfp.loc[sig_mask, "x"],
            dfp.loc[sig_mask, "ylog"],
            color="red",
            marker=str(marker),
            s=float(base_size) * 1.5,
            alpha=(
                float(alpha_override)
                if alpha_override is not None
                else 0.85
            ),
            rasterized=rasterized,
            zorder=6,
            **_marker_scatter_style(str(marker)),
        )

    if bool(draw_threshold_line) and np.isfinite(threshold):
        thr_log = _postgwas_threshold_to_logp(
            threshold,
            no_logtrans=no_logtrans,
        )
        if np.isfinite(thr_log):
            ax.axhline(
                y=thr_log,
                linestyle="dashed",
                color="grey",
                linewidth=1.0,
            )
    return True


def _overlay_postgwas_manhattan_hits(
    ax: plt.Axes,
    plotmodel: GWASPLOT,
    *,
    threshold: Optional[float],
    draw_threshold_line: bool = True,
    base_size: float,
    marker: str,
    rasterized: bool,
    alpha_override: Optional[float] = None,
    min_logp: float = 0.5,
    max_logp: Optional[float] = None,
    ignore: Optional[list[object]] = None,
) -> None:
    # ``threshold`` is also used internally as a default for QQ sampling and
    # threshold-based auxiliary outputs.  Manhattan hit coloring, however,
    # should only happen when the user explicitly requested ``--thr``.
    if threshold is None or not np.isfinite(threshold):
        return

    if _overlay_manhattan_interaction_padj_points(
        ax,
        plotmodel,
        threshold=threshold,
        draw_threshold_line=draw_threshold_line,
        base_size=base_size,
        marker=marker,
        rasterized=rasterized,
        alpha_override=alpha_override,
        min_logp=min_logp,
        max_logp=max_logp,
        ignore=ignore,
    ):
        return
    _overlay_manhattan_threshold_points(
        ax,
        plotmodel,
        threshold=threshold,
        draw_threshold_line=draw_threshold_line,
        base_size=base_size,
        marker=marker,
        rasterized=rasterized,
        alpha_override=alpha_override,
        min_logp=min_logp,
        max_logp=max_logp,
        ignore=ignore,
    )


def _safe_neglog10_p(values: object) -> np.ndarray:
    """
    Backward-compatible safe ``-log10(p)`` transform for p-values.
    """
    return _postgwas_plot_logp_values(values, no_logtrans=False)


def _postgwas_plot_logp_values(
    values: object,
    *,
    no_logtrans: bool = False,
) -> np.ndarray:
    """
    Return values in the plot's ``-log10(p)`` scale.

    With ``no_logtrans=True`` the input is already treated as ``-log10(p)``
    and finite values are returned unchanged. Non-finite values are mapped
    to zero in that mode, matching the neutral ``p=1`` value used by the
    normal transform.
    """
    p = pd.to_numeric(values, errors="coerce")
    if isinstance(p, pd.Series):
        arr = p.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(p, dtype=float)
    arr = np.array(arr, dtype=float, copy=True)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if bool(no_logtrans):
        arr[~np.isfinite(arr)] = 0.0
        return arr
    arr[~np.isfinite(arr)] = 1.0
    arr = np.clip(arr, np.nextafter(0.0, 1.0), 1.0)
    return -np.log10(arr)


def _postgwas_valid_p_mask(
    values: object,
    *,
    no_logtrans: bool = False,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        arr = numeric.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(numeric, dtype=float)
    finite = np.isfinite(arr)
    if bool(no_logtrans):
        return finite & (arr >= 0.0)
    return finite & (arr > 0.0)


def _postgwas_significant_p_mask(
    values: object,
    threshold: object,
    *,
    no_logtrans: bool = False,
) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if isinstance(numeric, pd.Series):
        arr = numeric.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(numeric, dtype=float)
    try:
        thr = float(threshold)
    except (TypeError, ValueError):
        return np.zeros(arr.shape, dtype=bool)
    valid = _postgwas_valid_p_mask(arr, no_logtrans=no_logtrans)
    if not np.isfinite(thr):
        return np.zeros(arr.shape, dtype=bool)
    if bool(no_logtrans):
        return valid & (arr >= thr)
    return valid & (arr <= thr)


def _postgwas_threshold_to_logp(
    threshold: object,
    *,
    no_logtrans: bool = False,
) -> float:
    try:
        value = float(threshold)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    if bool(no_logtrans):
        return value
    if value <= 0.0:
        return float("nan")
    return float(_postgwas_plot_logp_values([value], no_logtrans=False)[0])


def _postgwas_default_threshold(
    n_snp: int,
    *,
    no_logtrans: bool = False,
) -> float:
    if int(n_snp) <= 0:
        return float("nan")
    raw_threshold = 0.05 / float(n_snp)
    if bool(no_logtrans):
        return float(_postgwas_plot_logp_values([raw_threshold], no_logtrans=False)[0])
    return raw_threshold


def _postgwas_manhattan_y_label(
    pvalue_column: object,
    *,
    no_logtrans: bool = False,
) -> Optional[str]:
    """Return the explicit Manhattan y-axis label for precomputed logP."""
    if not bool(no_logtrans):
        return None
    label = str(pvalue_column).strip()
    return label if label != "" else None


def _postgwas_should_draw_threshold_line(
    explicit_threshold: object,
    *,
    no_logtrans: bool = False,
) -> bool:
    """Draw a threshold line only when the user explicitly requests one."""
    return explicit_threshold is not None


def _postgwas_output_format_from_path(path: str) -> str:
    _, ext = os.path.splitext(str(path))
    return ext.lstrip(".").strip().lower()


def _postgwas_should_rasterize_dense_layers(
    output_format: object,
    *,
    n_points: Optional[int] = None,
) -> bool:
    fmt = str(output_format).strip().lower()
    if fmt != "pdf":
        return True
    # Keep small PDF plots vector-friendly, but avoid embedding one vector
    # marker/path per site in large Manhattan or circular plots.
    if n_points is None:
        return False
    try:
        return int(n_points) >= _POSTGWAS_RASTERIZE_THRESHOLD
    except (TypeError, ValueError):
        return False


def _postgwas_savefig_kwargs(output_format: object) -> dict[str, object]:
    fmt = str(output_format).strip().lower()
    return {
        "transparent": False,
        "facecolor": "white",
        "edgecolor": "white",
    }


def _postgwas_resolve_pdf_backend() -> Optional[str]:
    global _POSTGWAS_PREFERRED_PDF_BACKEND
    cached = _POSTGWAS_PREFERRED_PDF_BACKEND
    if cached is not _POSTGWAS_PDF_BACKEND_SENTINEL:
        return cached if isinstance(cached, str) else None
    backend_name: Optional[str]
    try:
        from matplotlib.backends import backend_cairo  # noqa: F401
    except Exception:
        backend_name = None
    else:
        backend_name = "cairo"
    _POSTGWAS_PREFERRED_PDF_BACKEND = backend_name
    return backend_name


def _qq_select_points_with_threshold(
    pvals: np.ndarray,
    *,
    sig_p_threshold: Optional[float],
    max_points: int = _QQ_FAST_MAX_POINTS,
    keep_all: bool = False,
    no_logtrans: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select QQ scatter points with deterministic down-sampling:
    - always keep all points with p <= sig_p_threshold
    - for remaining points, keep an evenly spaced rank grid up to max_points
    """
    p = np.asarray(pvals, dtype=float)
    if bool(no_logtrans):
        p = p[np.isfinite(p) & (p >= 0.0)]
    else:
        p = p[np.isfinite(p) & (p > 0.0)]
    if p.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if bool(no_logtrans):
        p_sorted = np.sort(p, kind="mergesort")[::-1]
    else:
        p = np.clip(p, np.nextafter(0.0, 1.0), 1.0)
        p_sorted = np.sort(p, kind="mergesort")
    n = int(p_sorted.size)

    if sig_p_threshold is None or (not np.isfinite(sig_p_threshold)):
        sig_thr = (
            float(np.log10(max(1, n)))
            if bool(no_logtrans)
            else 1.0 / float(max(1, n))
        )
    else:
        sig_thr = float(sig_p_threshold)
    if not bool(no_logtrans):
        sig_thr = float(np.clip(sig_thr, np.nextafter(0.0, 1.0), 1.0))

    if keep_all or n <= int(max_points):
        draw_idx = np.arange(n, dtype=np.int64)
    else:
        base_idx = np.linspace(0, n - 1, int(max_points), dtype=np.int64)
        if bool(no_logtrans):
            sig_n = int(np.sum(p_sorted >= sig_thr))
        else:
            sig_n = int(np.searchsorted(p_sorted, sig_thr, side="right"))
        if sig_n > 0:
            sig_idx = np.arange(sig_n, dtype=np.int64)
            draw_idx = np.unique(np.concatenate([base_idx, sig_idx]))
        else:
            draw_idx = np.unique(base_idx)

    ranks = draw_idx.astype(float) + 1.0
    exp = -np.log10(ranks / (n + 1.0))
    obs = (
        p_sorted[draw_idx]
        if bool(no_logtrans)
        else -np.log10(p_sorted[draw_idx])
    )
    keep = np.isfinite(exp) & np.isfinite(obs)
    return exp[keep], obs[keep]


def _qq_confidence_band_from_n(
    n_points: int,
    *,
    ci: int = 95,
    max_points: Optional[int] = _QQ_BAND_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_n = int(max(0, n_points))
    if band_n <= 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    if max_points is not None and band_n > int(max_points):
        n_band = max(2, int(max_points))
        ranks = np.unique(
            np.round(np.geomspace(1.0, float(band_n), num=n_band)).astype(np.int64)
        )
        if ranks[0] != 1:
            ranks = np.insert(ranks, 0, 1)
        if ranks[-1] != band_n:
            ranks = np.append(ranks, band_n)
    else:
        ranks = np.arange(1, band_n + 1, dtype=np.int64)
    ranks = np.sort(ranks)[::-1]
    ci_frac = float(ci)
    if ci_frac > 1.0:
        ci_frac /= 100.0
    ci_frac = float(np.clip(ci_frac, 1e-12, 1.0 - 1e-12))
    alpha = 1.0 - ci_frac
    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0
    x_band = -np.log10(ranks.astype(np.float64) / (band_n + 1.0))
    p_upper = beta.ppf(q_hi, ranks, band_n - ranks + 1)
    p_lower = beta.ppf(q_lo, ranks, band_n - ranks + 1)
    lower = -np.log10(p_upper)
    upper = -np.log10(p_lower)
    keep = np.isfinite(x_band) & np.isfinite(lower) & np.isfinite(upper)
    return x_band[keep], lower[keep], upper[keep]


def _resolve_qq_ylim(
    ax: plt.Axes,
    *,
    lower: float,
    upper: Optional[float],
) -> tuple[float, float]:
    lo = float(lower)
    _y0, _y1 = ax.get_ylim()
    if upper is not None and np.isfinite(float(upper)):
        hi = float(upper)
    else:
        hi = float(_y1)
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + max(1.0, abs(lo) * 0.1, 1e-9)
    return lo, hi


def _apply_qq_axes(
    ax: plt.Axes,
    *,
    y_lower: float,
    y_upper: float,
    x_right: float,
    y_ticks: Optional[np.ndarray] = None,
) -> None:
    lo = float(y_lower)
    hi = float(y_upper)
    xr = float(x_right)
    if not np.isfinite(xr) or xr <= lo:
        xr = lo + 1.0
    x_pad = max(1e-9, 0.02 * float(max(1e-9, xr - lo)))
    x_upper = xr + x_pad
    if x_upper <= lo:
        x_upper = lo + 1.0
    ax.set_ylim(lo, hi)
    ax.set_xlim(lo, x_upper)
    ax.margins(x=0.0)
    if y_ticks is not None:
        ticks = np.asarray(y_ticks, dtype=float)
        keep = np.isfinite(ticks) & (ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)
        kept = ticks[keep]
        if kept.size > 0:
            apply_integer_yticks(ax, ticks=kept)
            return
    apply_integer_yticks(ax)


def _create_ratio_panel_figure(
    *,
    ratio: float,
    dpi: int,
    panel_width_in: Optional[float] = None,
    panel_height_in: Optional[float] = None,
    reserve_right_in: float = 0.0,
    left_in: float = _PANEL_LEFT_IN,
    right_in: float = _PANEL_RIGHT_IN,
    top_in: float = _PANEL_TOP_IN,
    bottom_in: float = _PANEL_BOTTOM_IN,
) -> tuple[plt.Figure, plt.Axes, float, float]:
    use_ratio = max(0.2, float(ratio))
    if panel_width_in is None and panel_height_in is None:
        panel_width_in = float(_PANEL_WIDTH_IN)
    if panel_width_in is None:
        plot_h = max(1e-6, float(panel_height_in))
        plot_w = plot_h * use_ratio
    elif panel_height_in is None:
        plot_w = max(1e-6, float(panel_width_in))
        plot_h = plot_w / use_ratio
    else:
        plot_w = max(1e-6, float(panel_width_in))
        plot_h = max(1e-6, float(panel_height_in))
    fig_w = float(left_in) + float(plot_w) + float(right_in) + float(reserve_right_in)
    fig_h = float(bottom_in) + float(plot_h) + float(top_in)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))
    ax = fig.add_axes(
        [
            float(left_in) / float(fig_w),
            float(bottom_in) / float(fig_h),
            float(plot_w) / float(fig_w),
            float(plot_h) / float(fig_h),
        ]
    )
    try:
        ax.set_box_aspect(float(plot_h) / float(plot_w))
    except Exception:
        pass
    return fig, ax, float(plot_w), float(plot_h)


def _create_stacked_panel_figure(
    *,
    panel_width_in: float,
    panel_heights_in: list[float],
    dpi: int,
    reserve_right_in: float = 0.0,
    left_in: float = _PANEL_LEFT_IN,
    right_in: float = _PANEL_RIGHT_IN,
    top_in: float = _PANEL_TOP_IN,
    bottom_in: float = _PANEL_BOTTOM_IN,
    vspace_in: float = _PANEL_STACK_VSPACE_IN,
) -> tuple[plt.Figure, list[plt.Axes], float, list[float]]:
    heights = [max(1e-6, float(h)) for h in panel_heights_in]
    plot_w = max(1e-6, float(panel_width_in))
    gaps_h = max(0, len(heights) - 1) * max(0.0, float(vspace_in))
    fig_w = float(left_in) + plot_w + float(right_in) + float(reserve_right_in)
    fig_h = float(bottom_in) + float(top_in) + float(sum(heights)) + float(gaps_h)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))
    axes: list[plt.Axes] = []
    current_top = fig_h - float(top_in)
    for h_in in heights:
        y0_in = current_top - float(h_in)
        ax = fig.add_axes(
            [
                float(left_in) / float(fig_w),
                float(y0_in) / float(fig_h),
                float(plot_w) / float(fig_w),
                float(h_in) / float(fig_h),
            ]
        )
        axes.append(ax)
        current_top = y0_in - float(vspace_in)
    return fig, axes, float(plot_w), heights


def _save_figure(fig: plt.Figure, path: str) -> None:
    fmt = _postgwas_output_format_from_path(path)
    save_kwargs = _postgwas_savefig_kwargs(fmt)
    if fmt == "pdf":
        backend_name = _postgwas_resolve_pdf_backend()
        if backend_name is not None:
            try:
                fig.savefig(path, backend=backend_name, **save_kwargs)
                return
            except Exception:
                pass
    fig.savefig(path, **save_kwargs)


def _save_figure_and_close(fig: plt.Figure, path: str) -> None:
    _save_figure(fig, path)
    plt.close(fig)


def _add_postgwas_annotation_source_args(
    group: argparse._ArgumentGroup,
) -> None:
    anno_src_group = group.add_mutually_exclusive_group(required=False)
    anno_src_group.add_argument(
        "-gff",
        "--gff",
        type=str,
        default=None,
        help=(
            "Annotation source in GFF/GFF3 format, shared by "
            "--anno output and --ldblock/--ldblock-all gene-structure tracks."
        ),
    )
    anno_src_group.add_argument(
        "-bed",
        "--bed",
        type=str,
        default=None,
        help=(
            "Annotation source in BED-like interval text format, shared by "
            "--anno output and --ldblock/--ldblock-all gene-structure tracks. "
            "Delimiter auto-detects tab/comma/space; suffix is not required."
        ),
    )


def _resolve_postgwas_annotation_file(args) -> Optional[str]:
    gff = str(getattr(args, "gff", "") or "").strip()
    if gff != "":
        return gff
    bed = str(getattr(args, "bed", "") or "").strip()
    if bed != "":
        return bed
    return None


def _resolve_postgwas_anno_cli(raw_anno: object) -> tuple[bool, Optional[float]]:
    if raw_anno is None or raw_anno is False:
        return False, None
    text = str(raw_anno).strip()
    if text == "" or text.lower() in {"on", "true", "yes", "y"}:
        return True, None
    try:
        broaden_kb = float(text)
    except Exception as e:
        raise ValueError(
            "--anno accepts no value or one numeric extension in kb, "
            "e.g. `--anno` or `--anno 50`."
        ) from e
    if (not np.isfinite(broaden_kb)) or float(broaden_kb) < 0.0:
        raise ValueError("--anno extension must be a finite number >= 0.")
    return True, float(broaden_kb)


def GWASplot(file: str, args, logger:logging.Logger) -> None:
    """
    Plot Manhattan/QQ figures and optionally annotate significant hits
    for a single GWAS result file.
    """
    _apply_postgwas_matplotlib_style(args)

    # Silence pandas chained-assignment warnings in this script
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*ChainedAssignmentError.*",
    )

    output_stem = _resolve_postgwas_output_stem(
        file,
        getattr(args, "_postgwas_plot_prefix", None),
    )

    chr_col, pos_col, p_col = args.chr, args.pos, args.pvalue
    no_logtrans = bool(getattr(args, "no_logtrans", False))
    anno_is_gff = _postgwas_annotation_is_gff(
        args.anno_file,
        annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
    )
    use_shared_gff = bool(getattr(args, "_postgwas_use_shared_gff", False))
    gff_query_cache: Optional[GFFQuery] = None
    gff_rust_index_cache: Optional[object] = (
        _postgwas_get_gff_rust_index(
            args.anno_file,
            use_shared=use_shared_gff,
        )
        if anno_is_gff
        else None
    )
    status_enabled = _postgwas_status_enabled(args)
    src = os.path.basename(str(file))
    task_label = os.path.basename(str(output_stem))
    single_marker = str(getattr(args, "_postgwas_single_marker", _DEFAULT_SINGLE_MARKER))
    single_scatter_size = float(
        getattr(args, "_postgwas_single_scatter_size", _DEFAULT_SCATTER_SIZE)
    )
    single_alpha = getattr(args, "_postgwas_single_alpha", None)
    highlight_color = str(getattr(args, "highlight_color", "red"))
    highlight_marker = str(getattr(args, "highlight_marker", "D"))

    if not status_enabled:
        logger.info(f"Loading GWAS results from {src}... [{format_elapsed(0.0)}]")
    with CliStatus(
        f"Loading GWAS results from {src}...",
        enabled=status_enabled,
    ) as load_status:
        try:
            df_all, full_chr_labels = _load_postgwas_input_table(
                str(file),
                chr_col=chr_col,
                pos_col=pos_col,
                p_col=p_col,
                keep_all_columns=bool(args.anno or (args.circle_size is not None)),
            )
            df = df_all
        except Exception:
            load_status.fail(f"Loading GWAS results from {src} ...Failed")
            raise
        load_status.complete(
            f"Loading GWAS results from {src} (nSNP={int(df_all.shape[0])})"
        )
    if status_enabled:
        logger.info(f"* {task_label} (nSNP={int(df_all.shape[0])})")

    bim_layout: list[dict[str, object]] = []
    if args.bimrange_tuples is not None:
        df_sel, seg_defs, n_before = _filter_df_by_bimranges(
            df_all,
            chr_col,
            pos_col,
            args.bimrange_tuples,
            logger,
            file,
        )
        n_after = int(df_sel.shape[0])
        if n_after == 0:
            logger.warning(
                f"No SNPs found in all bimrange settings for file {file}; skipped."
            )
            # Keep LD-only no-genotype flow alive: draw a zero-correlation block.
            if not (
                args.ldblock_ratio is not None
                and args.genofile is None
                and args.manh_ratio is None
                and args.qq_ratio is None
            ):
                return
            df = df_sel
        else:
            df = df_sel
            bim_layout = _build_bimrange_layout(
                seg_defs,
                interval_ratio=float(args.interval),
            )
            logger.info(
                f"Applied {len(args.bimrange_tuples)} bimrange settings: kept {n_after}/{n_before} SNPs."
            )
    # Keep positional index contiguous to avoid iloc out-of-bounds in plot model.
    df = df.reset_index(drop=True)

    # Bonferroni-style default threshold if not provided
    threshold = (
        float(args.thr)
        if args.thr is not None
        else _postgwas_default_threshold(
            int(df.shape[0]),
            no_logtrans=no_logtrans,
        )
    )
    highlight_threshold = float(args.thr) if args.thr is not None else None
    draw_threshold_line = _postgwas_should_draw_threshold_line(
        args.thr,
        no_logtrans=no_logtrans,
    )
    effective_ldblock_ratio = (
        args.ldblock_ratio
        if bool(getattr(args, "_postgwas_single_ldblock_requested", False))
        else None
    )
    if effective_ldblock_ratio is not None and args.bimrange_tuples is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all requires --bimrange; LD block and Manhattan+LD plotting are skipped."
        )
        effective_ldblock_ratio = None

    # ------------------------------------------------------------------
    # 1. Visualization: Manhattan & QQ
    # ------------------------------------------------------------------
    if (
        args.manh_ratio is not None
        or args.circle_size is not None
        or args.qq_ratio is not None
        or effective_ldblock_ratio is not None
    ):
        t_plot = time.time()
        logger.info(f"Visualizing GWAS results from {src}...")
        ld_use_all_sites = bool(args.ldblock_all is not None)

        need_manh_panel = (
            args.manh_ratio is not None
            or args.circle_size is not None
            or effective_ldblock_ratio is not None
        )
        plotmodel = None
        plotmodel_qq = None
        if (need_manh_panel or args.qq_ratio is not None) and df.shape[0] > 0:
            plotmodel = GWASPLOT(
                df,
                chr_col,
                pos_col,
                p_col,
                float(args.interval),
                compression=(not args.disable_compression),
                pvalue_is_log10=no_logtrans,
            )
            if args.bimrange_tuples is not None and len(bim_layout) > 1:
                _apply_segmented_x_to_plotmodel(plotmodel, df, chr_col, pos_col, bim_layout)
            if args.qq_ratio is not None:
                # Keep QQ tied to the same filtered row universe as Manhattan,
                # including both singleton and combo rows from GARFIELD FvLMM tables.
                plotmodel_qq = GWASPLOT(
                    df,
                    chr_col,
                    pos_col,
                    p_col,
                    float(args.interval),
                    compression=False,
                    pvalue_is_log10=no_logtrans,
                )
        width_in = float(_PANEL_WIDTH_IN)
        dpi = 300
        rasterized = _postgwas_should_rasterize_dense_layers(
            args.format,
            n_points=int(df.shape[0]),
        )
        manh_ratio_for_font = float(args.manh_ratio) if args.manh_ratio is not None else 2.0
        manh_fontsize_target = _postgwas_resolve_fontsize(
            args,
            manh_ratio=manh_ratio_for_font,
        )
        gene_panel_h_in = _resolve_postgwas_gene_panel_height_in(
            float(manh_fontsize_target),
            width_in=width_in,
        )
        manh_loc_fontsize = float(manh_fontsize_target)
        manh_ylim = None
        manh_height_in = None
        manh_fontsize = None
        manh_yticks = None
        manh_xlim = None
        manh_axes_bounds = None
        hide_axis_labels = args.manh_ratio is not None and args.manh_ratio > 4.0
        default_manh_min = 0.0
        manh_min_logp = args.ylim_min if args.ylim_min is not None else default_manh_min
        manh_max_logp = args.ylim_max
        manh_ymin = manh_min_logp
        postgwas_job_workers = max(1, int(getattr(args, "_postgwas_job_workers", 1)))
        enable_pure_manhqq_parallel = (
            postgwas_job_workers == 1
            and args.manh_ratio is not None
            and args.qq_ratio is not None
            and effective_ldblock_ratio is None
        )
        pending_manh_fig: Optional[plt.Figure] = None
        pending_manh_path: Optional[str] = None

        plot_colors = None
        ldblock_style = _resolve_ldblock_style(args.ldblock_palette_spec)
        if plotmodel is not None:
            plot_colors = _manhattan_colors_for_subset(
                args.palette_spec,
                full_chr_labels,
                df[chr_col].drop_duplicates().tolist(),
            )
        qq_point_color = _resolve_qq_point_color(args.palette_spec)
        qq_band_color = str(_QQ_BAND_COLOR)
        circle_links_df: Optional[pd.DataFrame] = None
        circle_link_meta: dict[str, Optional[str]] = {
            "group_col": None,
            "type_col": None,
            "pvalue_col": None,
            "score_col": None,
        }
        if args.circle_size is not None and plotmodel is not None:
            interact_path = getattr(args, "_circle_interact_path", None)
            interact_spec = getattr(args, "_circle_interact_spec", None)
            if interact_path is not None:
                spec = dict(interact_spec or _postgwas_parse_interact_spec(None))
                group_col = str(spec["snp_col"])
                interact_chr_col = str(spec["chr_col"])
                interact_pos_col = str(spec["pos_col"])
                interact_p_col = str(spec["p_col"])
                if os.path.abspath(str(interact_path)) == os.path.abspath(str(file)):
                    interact_df = df_all
                else:
                    interact_df = _postgwas_load_circle_interact_df(
                        str(interact_path),
                        group_col=group_col,
                        chr_col=interact_chr_col,
                        pos_col=interact_pos_col,
                        p_col=interact_p_col,
                    )
                circle_links_df, circle_link_meta = _postgwas_build_circle_link_table_from_groups(
                    interact_df,
                    group_col=group_col,
                    chr_col=interact_chr_col,
                    pos_col=interact_pos_col,
                    p_col=interact_p_col,
                    type_col=group_col,
                    group_tokens=[str(x) for x in list(spec.get("group_tokens", []))],
                    pvalue_is_log10=no_logtrans,
                )
            else:
                circle_links_df, circle_link_meta = _postgwas_build_circle_link_table(
                    df,
                    chr_col=chr_col,
                    pos_col=pos_col,
                    p_col=p_col,
                    pvalue_is_log10=no_logtrans,
                )
            if (
                circle_links_df is not None
                and circle_links_df.shape[0] > 0
                and np.isfinite(threshold)
                and "link_pvalue" in circle_links_df.columns
            ):
                before_n = int(circle_links_df.shape[0])
                circle_links_df = circle_links_df.loc[
                    _postgwas_significant_p_mask(
                        circle_links_df["link_pvalue"],
                        threshold,
                        no_logtrans=no_logtrans,
                    )
                ].copy()
                after_n = int(circle_links_df.shape[0])
                logger.info(
                    "Circular Manhattan: "
                    f"kept {after_n}/{before_n} interaction link(s) with "
                    f"{'logP>=thr' if no_logtrans else 'p<=thr'} ({float(threshold):.4g})."
                )
            if circle_links_df is None or circle_links_df.shape[0] == 0:
                logger.info("Circular Manhattan: no interaction links were detected; drawing track only.")
            else:
                link_p_col = circle_link_meta.get("pvalue_col")
                logger.info(
                    "Circular Manhattan: "
                    f"detected {int(circle_links_df.shape[0])} link(s) "
                    f"(group={circle_link_meta.get('group_col') or 'NA'}, "
                    f"pvalue={link_p_col or 'NA'}, "
                    f"source={os.path.basename(str(interact_path)) if interact_path is not None else 'auto'})."
                )

        def _draw_manhattan_axis(
            ax: plt.Axes,
        ) -> tuple[tuple[float, float], np.ndarray, float, tuple[float, float]]:
            if plotmodel is None:
                ax.text(0.5, 0.5, "No SNPs", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.xaxis.label.set_size(manh_fontsize_target)
                ax.yaxis.label.set_size(manh_fontsize_target)
                ax.tick_params(axis="both", labelsize=manh_fontsize_target)
                if hide_axis_labels:
                    ax.set_xlabel("")
                    ax.set_ylabel("")
                return (
                    ax.get_ylim(),
                    np.asarray([], dtype=float),
                    float(manh_fontsize_target),
                    ax.get_xlim(),
                )

            if args.highlight:
                # Match by coordinate, not by unique MultiIndex lookup:
                # GARFIELD may have multiple rows at one endpoint.
                hl_mask, hl_labels, hl_index = _postgwas_highlight_rows(
                    plotmodel,
                    str(args.highlight),
                )
                if not bool(np.any(hl_mask)):
                    logger.warning("Nothing to highlight. Check the BED file.")
                    plotmodel.manhattan(
                        None,
                        ax=ax,
                        color_set=plot_colors,
                        marker=single_marker,
                        min_logp=manh_min_logp,
                        max_logp=manh_max_logp,
                        y_min=manh_ymin,
                        y_label=_postgwas_manhattan_y_label(
                            p_col,
                            no_logtrans=no_logtrans,
                        ),
                        s=single_scatter_size,
                        alpha=(float(single_alpha) if single_alpha is not None else 0.78),
                        rasterized=rasterized,
                    )
                    _overlay_postgwas_manhattan_hits(
                        ax,
                        plotmodel,
                        threshold=highlight_threshold,
                        draw_threshold_line=draw_threshold_line,
                        base_size=single_scatter_size,
                        marker=single_marker,
                        rasterized=rasterized,
                        alpha_override=(float(single_alpha) if single_alpha is not None else None),
                        min_logp=manh_min_logp,
                        max_logp=manh_max_logp,
                    )
                else:
                    hl_rows = plotmodel.df.iloc[np.flatnonzero(hl_mask)].copy()
                    y_hl = _postgwas_plot_logp_values(
                        hl_rows["y"],
                        no_logtrans=no_logtrans,
                    )
                    keep_hl = np.isfinite(y_hl) & (y_hl >= float(manh_min_logp))
                    if manh_max_logp is not None:
                        keep_hl = keep_hl & (y_hl <= float(manh_max_logp))
                    if bool(np.any(keep_hl)):
                        ax.scatter(
                            hl_rows.loc[keep_hl, "x"],
                            y_hl[keep_hl],
                            marker=highlight_marker,
                            color=highlight_color,
                            alpha=(float(single_alpha) if single_alpha is not None else 0.85),
                            zorder=10,
                            s=single_scatter_size,
                            rasterized=rasterized,
                            **_marker_scatter_style(highlight_marker),
                        )

                    # Put one label at the highest plotted row per coordinate,
                    # while preserving every endpoint row in the highlighted layer.
                    chr_by_id = getattr(plotmodel, "_chr_label_by_id", {})
                    hl_keys = [
                        (
                            _normalize_chr(
                                chr_by_id.get(int(index[0]), index[0])
                            ),
                            int(index[1]),
                        )
                        for index in hl_rows.index
                    ]
                    for key, text in hl_labels.items():
                        row_mask = np.asarray(
                            [item == key for item in hl_keys],
                            dtype=bool,
                        ) & keep_hl
                        if not bool(np.any(row_mask)):
                            continue
                        row_pos = np.flatnonzero(row_mask)
                        best_pos = int(row_pos[np.argmax(y_hl[row_pos])])
                        ax.text(
                            float(hl_rows.iloc[best_pos]["x"]),
                            float(y_hl[best_pos]),
                            s=_sanitize_plot_text(text),
                            ha="center",
                            zorder=11,
                        )

                    plotmodel.manhattan(
                        None,
                        ax=ax,
                        color_set=plot_colors,
                        marker=single_marker,
                        min_logp=manh_min_logp,
                        max_logp=manh_max_logp,
                        y_min=manh_ymin,
                        y_label=_postgwas_manhattan_y_label(
                            p_col,
                            no_logtrans=no_logtrans,
                        ),
                        s=single_scatter_size,
                        alpha=(float(single_alpha) if single_alpha is not None else 0.78),
                        ignore=hl_index,
                        rasterized=rasterized,
                    )
                    _overlay_postgwas_manhattan_hits(
                        ax,
                        plotmodel,
                        threshold=highlight_threshold,
                        draw_threshold_line=draw_threshold_line,
                        base_size=single_scatter_size,
                        marker=single_marker,
                        rasterized=rasterized,
                        alpha_override=(float(single_alpha) if single_alpha is not None else None),
                        min_logp=manh_min_logp,
                        max_logp=manh_max_logp,
                        ignore=hl_index,
                    )
            else:
                plotmodel.manhattan(
                    None,
                    ax=ax,
                    color_set=plot_colors,
                    marker=single_marker,
                    min_logp=manh_min_logp,
                    max_logp=manh_max_logp,
                    y_min=manh_ymin,
                    y_label=_postgwas_manhattan_y_label(
                        p_col,
                        no_logtrans=no_logtrans,
                    ),
                    s=single_scatter_size,
                    alpha=(float(single_alpha) if single_alpha is not None else 0.78),
                    rasterized=rasterized,
                )
                _overlay_postgwas_manhattan_hits(
                    ax,
                    plotmodel,
                    threshold=highlight_threshold,
                    draw_threshold_line=draw_threshold_line,
                    base_size=single_scatter_size,
                    marker=single_marker,
                    rasterized=rasterized,
                    alpha_override=(float(single_alpha) if single_alpha is not None else None),
                    min_logp=manh_min_logp,
                    max_logp=manh_max_logp,
                )

            if args.bimrange_tuples is not None:
                if len(bim_layout) > 1:
                    _apply_multi_bimrange_manhattan_axis(
                        ax,
                        bim_layout,
                        label_fontsize=manh_loc_fontsize,
                    )
                elif len(args.bimrange_tuples) == 1:
                    if len(bim_layout) == 1:
                        seg = bim_layout[0]
                        _apply_bimrange_manhattan_axis(
                            ax,
                            seg["chrom"],
                            int(seg["start"]),
                            int(seg["end"]),
                        )
                    else:
                        bchrom, bstart, bend = args.bimrange_tuples[0]
                        _apply_bimrange_manhattan_axis(ax, bchrom, bstart, bend)
            if args.ylim_min is not None or args.ylim_max is not None:
                _y0, _y1 = ax.get_ylim()
                lo = float(args.ylim_min) if args.ylim_min is not None else 0.0
                hi = float(args.ylim_max) if args.ylim_max is not None else float(_y1)
                if not (hi > lo):
                    hi = lo + max(1e-9, abs(lo) * 1e-9)
                ax.set_ylim(lo, hi)
            ax.xaxis.label.set_size(manh_fontsize_target)
            ax.yaxis.label.set_size(manh_fontsize_target)
            ax.tick_params(axis="both", labelsize=manh_fontsize_target)
            if hide_axis_labels:
                ax.set_xlabel("")
                ax.set_ylabel("")
            manh_tick_values = apply_integer_yticks(ax)

            return (
                ax.get_ylim(),
                np.asarray(manh_tick_values, dtype=float),
                float(manh_fontsize_target),
                ax.get_xlim(),
            )

        # ----------------- Manhattan plot -----------------
        if args.manh_ratio is not None:
            fig, ax, _panel_w_in, manh_panel_h_in = _create_ratio_panel_figure(
                ratio=float(args.manh_ratio),
                dpi=dpi,
                panel_width_in=width_in,
            )
            manh_ylim, manh_yticks, manh_fontsize, manh_xlim = _draw_manhattan_axis(ax)
            manh_height_in = float(manh_panel_h_in)
            manh_axes_bounds = ax.get_position().bounds
            manh_path = os.path.join(args.out, f"{output_stem}.manh.{args.format}")
            if enable_pure_manhqq_parallel:
                pending_manh_fig = fig
                pending_manh_path = manh_path
            else:
                _save_figure_and_close(fig, manh_path)
        else:
            manh_path = None

        # ----------------- Circle Manhattan plot -----------------
        if args.circle_size is not None:
            if plotmodel is None:
                logger.warning("Warning: circular Manhattan skipped because no SNPs are available.")
                circle_path = None
            else:
                fig_circle, ax_circle = plt.subplots(
                    figsize=(float(args.circle_size), float(args.circle_size)),
                    dpi=dpi,
                )
                plotmodel.circle_manhattan(
                    threshold=(
                        _postgwas_threshold_to_logp(
                            threshold,
                            no_logtrans=no_logtrans,
                        )
                        if np.isfinite(threshold)
                        else None
                    ),
                    color_set=plot_colors,
                    ax=ax_circle,
                    links_df=circle_links_df,
                    link_type_col="link_type",
                    link_pvalue_col="link_pvalue",
                    link_pvalue_is_log10=no_logtrans,
                    draw_threshold_line=draw_threshold_line,
                    marker=single_marker,
                    scatter_size=single_scatter_size,
                    scatter_alpha=(
                        float(single_alpha) if single_alpha is not None else 0.76
                    ),
                    track_ratio=float(args.circle_track_ratio),
                    link_interval=float(args.circle_interval),
                    link_linewidth=float(args.circle_lw),
                    min_logp=manh_min_logp,
                    max_logp=manh_max_logp,
                    y_min=manh_ymin,
                    circle_direction=str(args.circle_direction),
                    rasterized=rasterized,
                )
                circle_path = os.path.join(args.out, f"{output_stem}.circle.{args.format}")
                _save_figure_and_close(fig_circle, circle_path)
        else:
            circle_path = None

        # ----------------- QQ plot -----------------
        if args.qq_ratio is not None:
            manh_save_executor: Optional[cf.ThreadPoolExecutor] = None
            manh_save_future: Optional[cf.Future] = None
            if (
                enable_pure_manhqq_parallel
                and pending_manh_fig is not None
                and pending_manh_path is not None
            ):
                manh_save_executor = cf.ThreadPoolExecutor(max_workers=1)
                manh_save_future = manh_save_executor.submit(
                    _save_figure_and_close,
                    pending_manh_fig,
                    pending_manh_path,
                )
                pending_manh_fig = None
                pending_manh_path = None
            try:
                if plotmodel_qq is None:
                    logger.warning("Warning: QQ plotting skipped because no SNPs are available.")
                    qq_path = None
                else:
                    qq_lower = (
                        float(manh_ylim[0])
                        if (manh_ylim is not None and len(manh_ylim) >= 1)
                        else (
                            float(args.ylim_min)
                            if args.ylim_min is not None
                            else 0.0
                        )
                    )
                    qq_upper_target = (
                        float(manh_ylim[1])
                        if (manh_ylim is not None and len(manh_ylim) >= 2)
                        else (
                            float(args.ylim_max)
                            if args.ylim_max is not None
                            else None
                        )
                    )
                    qq_y_in = 4.0
                    if manh_height_in is not None:
                        qq_y_in = manh_height_in
                    fig, ax2, _qq_panel_w_in, _qq_panel_h_in = _create_ratio_panel_figure(
                        ratio=float(args.qq_ratio),
                        dpi=dpi,
                        panel_height_in=qq_y_in,
                    )
                    plotmodel_qq.qq(
                        ax=ax2,
                        color_set=[qq_point_color, qq_band_color],
                        marker=single_marker,
                        line_color="black",
                        scatter_size=single_scatter_size,
                        scatter_alpha=(
                            float(single_alpha)
                            if single_alpha is not None
                            else 0.75
                        ),
                        qq_mode=("full" if args.fullscatter else "auto"),
                        qq_fast_max_points=_QQ_FAST_MAX_POINTS,
                        sig_p_threshold=(
                            float(threshold)
                            if (np.isfinite(threshold) and float(threshold) > 0.0)
                            else None
                        ),
                        axis_min=qq_lower,
                        axis_max=qq_upper_target,
                        band_color=qq_band_color,
                        rasterized=rasterized,
                    )
                    qq_xmax = float(np.log10(plotmodel_qq.df.shape[0] + 1))
                    if np.isfinite(qq_xmax) and qq_xmax > 0:
                        x_right = float(qq_xmax)
                    else:
                        _x_right = ax2.get_xlim()[1]
                        x_right = float(_x_right)
                    x_right = max(1.0, x_right)
                    qq_lower, qq_upper = _resolve_qq_ylim(
                        ax2,
                        lower=qq_lower,
                        upper=qq_upper_target,
                    )
                    _apply_qq_axes(
                        ax2,
                        y_lower=qq_lower,
                        y_upper=qq_upper,
                        x_right=x_right,
                        y_ticks=(
                            manh_yticks
                            if (manh_yticks is not None and manh_ylim is not None)
                            else None
                        ),
                    )
                    apply_integer_xticks(ax2)
                    if manh_fontsize is not None:
                        ax2.xaxis.label.set_size(manh_fontsize)
                        ax2.yaxis.label.set_size(manh_fontsize)
                        ax2.tick_params(axis="both", labelsize=manh_fontsize)
                    if hide_axis_labels:
                        ax2.set_xlabel("")
                        ax2.set_ylabel("")
                    qq_path = os.path.join(args.out, f"{output_stem}.qq.{args.format}")
                    _save_figure_and_close(fig, qq_path)
            finally:
                if manh_save_future is not None:
                    manh_save_future.result()
                if manh_save_executor is not None:
                    manh_save_executor.shutdown(wait=True)
        else:
            qq_path = None
        if pending_manh_fig is not None and pending_manh_path is not None:
            _save_figure_and_close(pending_manh_fig, pending_manh_path)
            pending_manh_fig = None
            pending_manh_path = None

        # ----------------- LD block -----------------
        gene_path = None
        if effective_ldblock_ratio is not None:
            ld_panel_xspan = args.ldblock_xspan
            ld_sites = _extract_ld_site_set(
                df,
                chr_col,
                pos_col,
                p_col,
                threshold,
                use_all_sites=ld_use_all_sites,
                no_logtrans=no_logtrans,
            )
            n_sig_sites = max(2, len(ld_sites))
            ld_overlay_text = None
            ld_site_keys: list[tuple[str, int]] = sorted(ld_sites, key=lambda x: (x[0], x[1]))

            if args.genofile is None:
                logger.warning(
                    "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; drawing zero-correlation LD block."
                )
                ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                ld_overlay_text = "No genotype"
            else:
                if len(ld_sites) < 2:
                    if ld_use_all_sites:
                        logger.warning(
                            "Warning: Fewer than 2 valid SNPs in selected region; drawing empty LD block."
                        )
                    else:
                        logger.warning(
                            "Warning: Fewer than 2 threshold-passing SNPs in selected region; drawing empty LD block."
                        )
                    ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                    ld_overlay_text = "Not enough SNPs"
                else:
                    ld_mat, sig_keys = _compute_ld_from_bed_rust(
                        str(args.genofile),
                        args.bimrange_tuples if args.bimrange_tuples is not None else [],
                        selected_sites=ld_sites,
                        threads=int(max(0, int(getattr(args, "thread", 0)))),
                        logger=logger,
                    )
                    # Keep Manhattan->gene/LD mapping lines even when genotype lookup fails:
                    # fallback to GWAS-derived ld_site_keys if no matched genotype SNP is returned.
                    if len(sig_keys) > 0:
                        ld_site_keys = sig_keys
                    if ld_mat.shape[0] < 2:
                        logger.warning(
                            "Warning: Requested SNPs were not found in genotype data; drawing empty LD block."
                        )
                        ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                        ld_overlay_text = "No matched SNPs"
                    else:
                        mode_text = "all SNPs" if ld_use_all_sites else "threshold-passing SNPs"
                        logger.info(f"LD block built from {len(sig_keys)} {mode_text}.")

            region_ranges = args.bimrange_tuples if args.bimrange_tuples is not None else []
            region_layout = (
                bim_layout
                if len(bim_layout) > 0
                else _build_layout_from_bimrange_tuples(
                    region_ranges,
                    interval_ratio=float(args.interval),
                )
            )
            use_segmented_gene_x = len(region_layout) > 1
            gene_track_df = pd.DataFrame(
                columns=["feature", "strand", "attribute", "x_start", "x_end"]
            )
            use_gene_bridge = False
            if args.anno_file:
                if anno_is_gff and gff_rust_index_cache is None and gff_query_cache is None:
                    gff_query_cache = _postgwas_get_gff_query(
                        args.anno_file,
                        use_shared=use_shared_gff,
                        current=gff_query_cache,
                    )
                gene_raw = _load_gene_like_records_from_anno(
                    args.anno_file,
                    region_ranges,
                    logger,
                    gff_query=gff_query_cache,
                    gff_rust_index=gff_rust_index_cache,
                )
                gene_track_df = _project_gene_records_to_plot_x(
                    gene_raw,
                    region_ranges,
                    region_layout,
                    use_segmented_x=use_segmented_gene_x,
                )
                if gene_track_df.shape[0] == 0:
                    logger.warning(
                        "Warning: No gene-structure records found in selected --bimrange; "
                        "gene panel and gene-overlaid Manhattan+LD transition are skipped."
                    )
                else:
                    use_gene_bridge = True

            ld_cmap = "Greys"
            gene_block_color = "grey"
            gene_line_color = "black"
            if ldblock_style is not None:
                ld_cmap = ldblock_style["ld_cmap"]
                gene_block_color = str(ldblock_style["gene_block_color"])
                gene_line_color = str(ldblock_style["gene_line_color"])

            def _draw_ld_axis(ax: plt.Axes) -> None:
                LDblock(
                    ld_mat,
                    ax=ax,
                    vmin=0,
                    vmax=1,
                    cmap=ld_cmap,
                    rasterize_threshold=100,
                )
                n_ld = max(2, int(ld_mat.shape[0]))
                # Keep LD triangle body filling the whole panel width.
                ax.set_xlim(0.5, float(n_ld) - 0.5)
                ax.margins(x=0.0)
                if ld_overlay_text:
                    ax.text(
                        n_ld / 2.0,
                        -n_ld / 2.0,
                        ld_overlay_text,
                        ha="center",
                        va="center",
                        fontsize=float(manh_fontsize_target),
                    )

            def _build_manh_ld_pairs(
                ax_manh: plt.Axes,
                ax_ld: plt.Axes,
            ) -> list[tuple[float, float, bool]]:
                if plotmodel is None or len(ld_site_keys) == 0:
                    return []

                # Keep mapping behavior consistent with Manhattan plotting:
                # use compressed subset (minidx), then select by ld mode.
                map_df = plotmodel.df.iloc[plotmodel.minidx, -3:].copy()
                p_vals = pd.to_numeric(map_df["y"], errors="coerce").to_numpy(dtype=float)
                keep_mask = _postgwas_valid_p_mask(
                    p_vals,
                    no_logtrans=no_logtrans,
                )
                if not ld_use_all_sites:
                    keep_mask = keep_mask & _postgwas_significant_p_mask(
                        p_vals,
                        threshold,
                        no_logtrans=no_logtrans,
                    )
                if not bool(np.any(keep_mask)):
                    return []

                chr_id_to_norm = {
                    i + 1: _normalize_chr(label) for i, label in enumerate(plotmodel.chr_labels)
                }
                chr_ids = np.asarray(map_df.index.get_level_values(0), dtype=np.int64)
                pos_vals = np.asarray(map_df.index.get_level_values(1), dtype=np.int64)
                x_vals = np.asarray(map_df["x"], dtype=float)
                key_to_meta: dict[tuple[str, int], tuple[float, bool]] = {}
                for cid, p, x, keep, pv in zip(chr_ids, pos_vals, x_vals, keep_mask, p_vals):
                    if not keep:
                        continue
                    chrom_norm = chr_id_to_norm.get(int(cid))
                    if chrom_norm is None:
                        continue
                    key = (chrom_norm, int(p))
                    is_sig = bool(
                        _postgwas_significant_p_mask(
                            [pv],
                            threshold,
                            no_logtrans=no_logtrans,
                        )[0]
                    )
                    if key not in key_to_meta:
                        key_to_meta[key] = (float(x), is_sig)
                    else:
                        old_x, old_sig = key_to_meta[key]
                        # For duplicated SNP keys, keep x and merge significance with OR.
                        key_to_meta[key] = (old_x, bool(old_sig or is_sig))

                n_ld = int(ld_mat.shape[0])
                keys = ld_site_keys[:n_ld]
                pairs: list[tuple[float, float, bool]] = []
                for i, key in enumerate(keys):
                    meta = key_to_meta.get((str(key[0]), int(key[1])))
                    if meta is None:
                        continue
                    x_top, is_sig = meta
                    # In LDblock, SNP i is centered around x=i+0.5 near the top edge.
                    x_ld = float(i) + 0.5
                    pairs.append((x_top, x_ld, bool(is_sig)))
                return pairs

            def _draw_bridge_axis(
                ax_bridge: plt.Axes,
                ax_manh: plt.Axes,
                ax_ld: plt.Axes,
                pairs: list[tuple[float, float, bool]],
                *,
                line_color: str = "grey",
                sig_line_color: str = "red",
            ) -> None:
                ax_bridge.set_xlim(0.0, 1.0)
                ax_bridge.set_ylim(0.0, 1.0)
                ax_bridge.set_xticks([])
                ax_bridge.set_yticks([])
                for spine in ax_bridge.spines.values():
                    spine.set_visible(False)
                if len(pairs) == 0:
                    return

                slashes_x: list[float] = []
                edge_margin_n = 0.006
                y_manh_ref = float(ax_manh.get_ylim()[0])
                y_ld_ref = float(_POSTGWAS_LD_LINK_Y)
                to_bridge = ax_bridge.transAxes.inverted()
                for x_top, x_ld, is_sig in pairs:
                    p_top_disp = ax_manh.transData.transform((float(x_top), y_manh_ref))
                    p_ld_disp = ax_ld.transData.transform((float(x_ld), y_ld_ref))
                    x_top_n = float(to_bridge.transform(p_top_disp)[0])
                    x_ld_n = float(to_bridge.transform(p_ld_disp)[0])
                    if not (np.isfinite(x_top_n) and np.isfinite(x_ld_n)):
                        continue
                    x_top_n = float(np.clip(x_top_n, edge_margin_n, 1.0 - edge_margin_n))
                    x_ld_n = float(np.clip(x_ld_n, edge_margin_n, 1.0 - edge_margin_n))
                    joint_y = 0.84
                    bottom_y = 0.06
                    # Draw as one polyline to avoid tiny rendering gap at the joint.
                    poly_color = str(sig_line_color) if bool(is_sig) else str(line_color)
                    ax_bridge.plot(
                        [x_top_n, x_top_n, x_ld_n],
                        [1.04, joint_y, bottom_y],
                        color=poly_color,
                        linewidth=0.25,
                        alpha=0.8,
                        clip_on=False,
                        solid_joinstyle="round",
                    )
                    slashes_x.append(float(x_top))

            if use_segmented_gene_x and len(region_layout) > 0:
                gene_xlim = (
                    float(region_layout[0]["x_start"]),
                    float(region_layout[-1]["x_end"]),
                )
            elif len(region_ranges) > 0:
                gene_xlim = (
                    float(min(int(x[1]) for x in region_ranges)),
                    float(max(int(x[2]) for x in region_ranges)),
                )
            else:
                gene_xlim = (0.0, 1.0)
            gene_plot_xlim = manh_xlim if manh_xlim is not None else gene_xlim

            ld_n_sites = max(2, int(ld_mat.shape[0]))
            ld_h_in = width_in / effective_ldblock_ratio
            ld_min_h_in = width_in * _ld_min_height_over_width(ld_n_sites)
            ld_h_in = max(float(ld_h_in), float(ld_min_h_in))
            fig_ld = plt.figure(
                figsize=(width_in, ld_h_in),
                dpi=dpi,
            )
            ax_ld = fig_ld.add_subplot(111)
            ld_path = os.path.join(args.out, f"{output_stem}.ldblock.{args.format}")
            _draw_ld_axis(ax_ld)

            fig_ld.subplots_adjust(
                left=0.08,
                right=0.98,
                top=0.98,
                bottom=0.08,
            )
            # Keep LD block x-axis drawable length consistent with Manhattan;
            # optionally constrain to user-specified x-span of Manhattan width.
            if manh_axes_bounds is not None or ld_panel_xspan is not None:
                _cx0, cy0, _cw, ch = ax_ld.get_position().bounds
                if manh_axes_bounds is not None:
                    mx0, _my0, mw, _mh = manh_axes_bounds
                    ref_x0 = float(mx0)
                    ref_w = float(mw)
                else:
                    ref_x0 = float(_cx0)
                    ref_w = float(_cw)
                if ld_panel_xspan is None:
                    fx0, fx1 = (0.0, 1.0)
                else:
                    fx0, fx1 = ld_panel_xspan
                new_x0 = ref_x0 + ref_w * float(fx0)
                new_w = ref_w * float(fx1 - fx0)
                ax_ld.set_position([new_x0, cy0, new_w, ch])
                ax_ld.set_anchor("N")
            _save_figure_and_close(fig_ld, ld_path)

            if use_gene_bridge:
                gene_h_in = gene_panel_h_in
                fig_gene = plt.figure(
                    figsize=(width_in, gene_h_in),
                    dpi=dpi,
                )
                ax_gene = fig_gene.add_subplot(111)
                _draw_gene_structure_axis(
                    ax_gene,
                    gene_track_df,
                    arrow_color=gene_line_color,
                    block_color=gene_block_color,
                    line_width=1.0,
                    arrow_step=1_000.0,
                    gene_text_size=manh_fontsize_target,
                )
                ax_gene.set_xlim(gene_plot_xlim)
                _apply_postgwas_gene_panel_layout(
                    fig_gene,
                    ax_gene,
                    x_align_bounds=manh_axes_bounds,
                )
                gene_path = os.path.join(args.out, f"{output_stem}.gene.{args.format}")
                _save_figure_and_close(fig_gene, gene_path)
            else:
                gene_path = None

            if args.manh_ratio is not None:
                # Combined Manhattan + LD panel
                manhld_manh_ratio = float(args.manh_ratio)
                manhld_manh_h_in = width_in / manhld_manh_ratio
                gene_bridge_scale = 0.8
                mid_gene_y_offset = 0.03
                mid_gene_ymin = -0.18
                mid_gene_ymax = 0.22
                mid_h_in = _resolve_postgwas_bridge_panel_height_in(
                    float(manh_fontsize_target),
                    width_in=width_in,
                    use_gene_bridge=use_gene_bridge,
                )
                if ld_panel_xspan is None:
                    ld_panel_frac_in_manh = 1.0
                else:
                    ld_panel_frac_in_manh = float(ld_panel_xspan[1] - ld_panel_xspan[0])
                ld_h_in_combo = (
                    width_in
                    * ld_panel_frac_in_manh
                    / effective_ldblock_ratio
                )
                ld_h_in_combo_min = (
                    width_in
                    * ld_panel_frac_in_manh
                    * _ld_min_height_over_width(max(2, int(ld_mat.shape[0])))
                )
                ld_h_in_combo = max(0.5, float(ld_h_in_combo), float(ld_h_in_combo_min))
                fig_manhld, combo_axes, _combo_panel_w_in, _combo_panel_heights = _create_stacked_panel_figure(
                    panel_width_in=width_in,
                    panel_heights_in=[manhld_manh_h_in, mid_h_in, ld_h_in_combo],
                    dpi=dpi,
                    vspace_in=_PANEL_STACK_VSPACE_IN,
                )
                ax_manhld_top, ax_manhld_mid, ax_manhld_bot = combo_axes
                # Keep transition axis below Manhattan axis so loc labels remain visible.
                ax_manhld_top.set_zorder(5)
                ax_manhld_mid.set_zorder(2)
                ax_manhld_bot.set_zorder(1)
                _set_postgwas_axis_transparent(ax_manhld_mid)

                _tmp_ylim, _tmp_yticks, _tmp_fontsize, manhld_top_xlim = _draw_manhattan_axis(ax_manhld_top)
                _draw_ld_axis(ax_manhld_bot)
                if use_gene_bridge:
                    _draw_gene_structure_axis(
                        ax_manhld_mid,
                        gene_track_df,
                        arrow_color=gene_line_color,
                        block_color=gene_block_color,
                        line_width=0.8,
                        arrow_step=1_000.0,
                        thickness_scale=gene_bridge_scale,
                        y_offset=mid_gene_y_offset,
                        gene_text_size=float(_tmp_fontsize),
                    )
                    ax_manhld_mid.set_xlim(manhld_top_xlim)
                    ax_manhld_mid.set_ylim(mid_gene_ymin, mid_gene_ymax)

                # Keep the two panels strictly left/right aligned.
                # Manh/Gene keep full width; optionally narrow only LD panel width.
                fig_manhld.canvas.draw()
                bx0, by0, bw, bh = ax_manhld_bot.get_position().bounds
                tx0, _ty0, tw, _th = ax_manhld_top.get_position().bounds
                if ld_panel_xspan is None:
                    ld_panel_width_scale = 1.0
                    new_bw = bw * float(ld_panel_width_scale)
                    new_bx0 = bx0 + 0.5 * (bw - new_bw)
                else:
                    fx0, fx1 = ld_panel_xspan
                    new_bw = float(tw) * float(fx1 - fx0)
                    new_bx0 = float(tx0) + float(tw) * float(fx0)
                ax_manhld_bot.set_position([new_bx0, by0, new_bw, bh])
                ax_manhld_bot.set_anchor("N")
                fig_manhld.canvas.draw()

                bridge_pairs = _build_manh_ld_pairs(ax_manhld_top, ax_manhld_bot)
                if use_gene_bridge:
                    ax_manhld_mid.set_xlim(ax_manhld_top.get_xlim())
                    ax_manhld_mid.set_ylim(mid_gene_ymin, mid_gene_ymax)
                    _draw_manh_gene_ld_links(
                        fig_manhld,
                        ax_manhld_mid,
                        ax_manhld_top,
                        ax_manhld_bot,
                        bridge_pairs,
                        gene_route_y=-0.17 * gene_bridge_scale + mid_gene_y_offset,
                        nonsig_line_color="grey",
                    )
                else:
                    _draw_bridge_axis(
                        ax_manhld_mid,
                        ax_manhld_top,
                        ax_manhld_bot,
                        bridge_pairs,
                        line_color="grey",
                        sig_line_color="red",
                    )
                if args.bimrange_tuples is None:
                    _show_end_locs_without_xticks(
                        ax_manhld_top,
                        label_fontsize=float(_tmp_fontsize),
                    )

                manhld_path = os.path.join(args.out, f"{output_stem}.manhld.{args.format}")
                _save_figure_and_close(fig_manhld, manhld_path)
            else:
                manhld_path = None
        else:
            ld_path = None
            gene_path = None
            manhld_path = None

        saved_paths: list[tuple[str, str]] = []
        if manh_path is not None:
            saved_paths.append(("Manhattan", manh_path))
        if circle_path is not None:
            saved_paths.append(("Circle Manhattan", circle_path))
        if qq_path is not None:
            saved_paths.append(("QQ", qq_path))
        if ld_path is not None:
            saved_paths.append(("LD block", ld_path))
        if gene_path is not None:
            saved_paths.append(("Gene structure", gene_path))
        if manhld_path is not None:
            saved_paths.append(("Manhattan+LD", manhld_path))

        if len(saved_paths) == 1:
            log_success(
                logger,
                f"{saved_paths[0][0]} plot saved to:\n  {format_path_for_display(saved_paths[0][1])}",
            )
        elif len(saved_paths) > 1:
            title = ", ".join([x[0] for x in saved_paths])
            body = "\n".join([f"  {format_path_for_display(x[1])}" for x in saved_paths])
            log_success(logger, f"{title} plots saved to:\n{body}")
        log_success(logger, f"Visualizing GWAS results from {src} ...Finished [{format_elapsed(time.time() - t_plot)}]")

    # ------------------------------------------------------------------
    # 2. Annotation of significant loci
    # ------------------------------------------------------------------
    if args.anno and args.thr is not None:
        if not status_enabled:
            logger.info(f"Annotating significant SNPs from {src}... [{format_elapsed(0.0)}]")
        with CliStatus(
            f"Annotating significant SNPs from {src}...",
            enabled=status_enabled,
        ) as anno_status:
            if not args.anno_file or (not os.path.exists(args.anno_file)):
                anno_status.complete(
                    "Annotating significant SNPs ...Skipped (annotation file not found)"
                )
            else:
                try:
                    gff_annotation_ctx: Optional[dict[str, object]] = None
                    bed_annotation_ctx: Optional[dict[str, object]] = None
                    use_gff_batch_annotation = False

                    # Keep SNPs passing threshold
                    df_filter_raw = df.loc[
                        _postgwas_significant_p_mask(
                            df[p_col],
                            threshold,
                            no_logtrans=no_logtrans,
                        )
                    ].copy()
                    original_columns = df_filter_raw.columns.tolist()
                    annotation_col_map = _resolve_annotation_append_colnames(
                        [str(col) for col in original_columns]
                    )
                    df_filter = _prepare_annotation_base_rows(
                        df_filter_raw,
                        chr_col=chr_col,
                        pos_col=pos_col,
                    )

                    if args.ldclump_window_bp is not None:
                        n_before = int(df_filter_raw.shape[0])
                        logger.info(
                            "Applying LD clump on threshold-passing SNPs for annotation: "
                            f"window={args.ldclump_window_bp} bp, r2>={args.ldclump_r2:g}"
                        )
                        df_clump_meta, _clump_dict = _ldclump_significant_snps(
                            df_filter_raw.loc[:, [chr_col, pos_col, p_col]].copy(),
                            chr_col=chr_col,
                            pos_col=pos_col,
                            p_col=p_col,
                            genofile=args.genofile,
                            window_bp=int(args.ldclump_window_bp),
                            r2_thr=float(args.ldclump_r2),
                            logger=logger,
                            show_progress=(len(args.gwasfile) == 1),
                            pvalue_is_log10=no_logtrans,
                        )
                        idx_chr = pd.Index(df_clump_meta.index.get_level_values(0).astype(str))
                        idx_pos = pd.to_numeric(
                            df_clump_meta.index.get_level_values(1),
                            errors="coerce",
                        ).fillna(0).astype(int)
                        df_clump_meta.index = pd.MultiIndex.from_arrays(
                            [idx_chr, idx_pos],
                            names=[chr_col, pos_col],
                        )
                        df_filter = df_filter.reindex(df_clump_meta.index).copy()
                        for src_col in ("start", "end", "nsnps", "MeanR2", "LDclump"):
                            if src_col in df_clump_meta.columns:
                                df_filter[annotation_col_map[src_col]] = df_clump_meta[src_col].values
                        n_after = int(df_filter.shape[0])
                        logger.info(
                            f"LD clump completed: kept {n_after}/{n_before} threshold-passing SNPs."
                        )

                    if anno_is_gff and int(df_filter.shape[0]) > 0:
                        if gff_rust_index_cache is None:
                            if gff_query_cache is None:
                                gff_query_cache = _postgwas_get_gff_query(
                                    args.anno_file,
                                    use_shared=use_shared_gff,
                                    current=gff_query_cache,
                                )
                            use_gff_batch_annotation = (
                                int(df_filter.shape[0]) >= int(_POSTGWAS_GFF_BATCH_ANNOTATION_MIN_SITES)
                            )
                            if use_gff_batch_annotation:
                                gff_annotation_ctx = _postgwas_get_gff_annotation_context(
                                    args.anno_file,
                                    use_shared=use_shared_gff,
                                    gff_query=gff_query_cache,
                                )

                    if anno_is_gff and gff_rust_index_cache is not None:
                        df_filter[annotation_col_map["desc"]] = _format_postgwas_gff_site_desc_many_rust(
                            df_filter.index,
                            gff_rust_index=gff_rust_index_cache,
                        )
                    elif anno_is_gff and gff_annotation_ctx is not None:
                        df_filter[annotation_col_map["desc"]] = _format_postgwas_gff_site_desc_many(
                            df_filter.index,
                            annotation_ctx=gff_annotation_ctx,
                        )
                    elif anno_is_gff and gff_query_cache is not None:
                        df_filter[annotation_col_map["desc"]] = [
                            _format_postgwas_gff_site_desc_direct(
                                chrom=idx[0],
                                pos=idx[1],
                                gff_query=gff_query_cache,
                            )
                            for idx in df_filter.index
                        ]
                    else:
                        if bed_annotation_ctx is None:
                            anno = readanno(
                                args.anno_file,
                                _ANNO_DESC_KEY,
                                annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
                            )
                            bed_annotation_ctx = _build_postgwas_bed_annotation_context(anno)
                        df_filter[annotation_col_map["desc"]] = _format_postgwas_bed_site_desc_many(
                            df_filter.index,
                            annotation_ctx=bed_annotation_ctx,
                        )

                    # Optional broadened window around SNP (卤 annobroaden kb)
                    if args.annobroaden is not None:
                        kb = args.annobroaden * 1_000
                        if anno_is_gff and gff_rust_index_cache is not None:
                            df_filter[annotation_col_map["broaden"]] = _format_postgwas_gff_broaden_many_rust(
                                df_filter.index,
                                gff_rust_index=gff_rust_index_cache,
                                window_bp=int(kb),
                            )
                        elif anno_is_gff and gff_annotation_ctx is not None:
                            df_filter[annotation_col_map["broaden"]] = _format_postgwas_gff_broaden_many(
                                df_filter.index,
                                annotation_ctx=gff_annotation_ctx,
                                window_bp=int(kb),
                            )
                        elif anno_is_gff and gff_query_cache is not None:
                            df_filter[annotation_col_map["broaden"]] = [
                                _format_postgwas_gff_broaden_direct(
                                    chrom=idx[0],
                                    pos=idx[1],
                                    gff_query=gff_query_cache,
                                    gff_rust_index=gff_rust_index_cache,
                                    window_bp=int(kb),
                                )
                                for idx in df_filter.index
                            ]
                        else:
                            if bed_annotation_ctx is None:
                                anno = readanno(
                                    args.anno_file,
                                    _ANNO_DESC_KEY,
                                    annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
                                )
                                bed_annotation_ctx = _build_postgwas_bed_annotation_context(anno)
                            df_filter[annotation_col_map["broaden"]] = _format_postgwas_bed_broaden_many(
                                df_filter.index,
                                annotation_ctx=bed_annotation_ctx,
                                window_bp=int(kb),
                            )
                    else:
                        broaden_col = annotation_col_map["broaden"]
                        if broaden_col in df_filter.columns:
                            df_filter = df_filter.drop(columns=[broaden_col])

                    df_out = _finalize_annotation_output_df(
                        df_filter,
                        chr_col=chr_col,
                        pos_col=pos_col,
                        original_columns=[str(col) for col in original_columns],
                        annotation_col_map=annotation_col_map,
                    )

                    anno_path = os.path.join(args.out, f"{output_stem}.{threshold}.anno.tsv")
                    df_out.to_csv(anno_path, sep="\t", index=False)
                except Exception:
                    anno_status.fail(f"Annotating significant SNPs from {src} ...Failed")
                    raise
                anno_status.complete(
                    f"Annotating significant SNPs from {src} ...Finished (nLead={int(df_out.shape[0])})"
                )
                log_success(logger, f"Annotation table saved to {format_path_for_display(anno_path)}")


def _read_merge_gwas_table(
    file: str,
    chr_col: str,
    pos_col: str,
    p_col: str,
    logger: logging.Logger,
    pvalue_is_log10: bool = False,
) -> pd.DataFrame:
    try:
        df = pd.read_csv(file, sep="\t", usecols=[chr_col, pos_col, p_col])
    except Exception as e:
        logger.error(
            f"Failed to read required columns from {file}: "
            f"{chr_col}, {pos_col}, {p_col}"
        )
        raise SystemExit(1) from e

    if df.shape[0] == 0:
        return df

    df = df.loc[:, [chr_col, pos_col, p_col]].copy()
    df[chr_col] = df[chr_col].astype(str)
    pos_num = pd.to_numeric(df[pos_col], errors="coerce")
    p_num = pd.to_numeric(df[p_col], errors="coerce")
    mask = (
        pos_num.notna()
        & np.isfinite(pos_num.to_numpy(dtype=float))
        & p_num.notna()
        & np.isfinite(p_num.to_numpy(dtype=float))
        & (p_num >= 0.0 if bool(pvalue_is_log10) else p_num > 0.0)
    )
    df = df.loc[mask, [chr_col, pos_col, p_col]].copy()
    if df.shape[0] == 0:
        return df

    df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce").astype(int)
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce").astype(float)
    return df


def _run_postgwas_merge_manhattan(args, logger: logging.Logger) -> None:
    _apply_postgwas_matplotlib_style(args)
    files = [str(f) for f in args.merge_files]
    if len(files) == 0:
        logger.warning("Warning: merged plotting requested but no GWAS input file is available.")
        return

    chr_col, pos_col, p_col = args.chr, args.pos, args.pvalue
    no_logtrans = bool(getattr(args, "no_logtrans", False))
    t_merge = time.time()
    logger.info("Visualizing merged post-GWAS results...")
    merge_manh_ratio = args._merge_manh_ratio
    merge_qq_ratio = args._merge_qq_ratio
    needs_positional_merge = bool(
        (merge_manh_ratio is not None)
        or (args.ldblock_ratio is not None)
    )

    series_colors = _resolve_merge_series_colors(args.palette_spec, len(files))
    series_markers = (
        list(getattr(args, "_postgwas_merge_markers", []))
        if len(getattr(args, "_postgwas_merge_markers", [])) == len(files)
        else _resolve_merge_markers(args.marker_spec, len(files))
    )
    series_sizes = (
        [float(x) for x in list(getattr(args, "_postgwas_merge_scatter_sizes", []))]
        if len(getattr(args, "_postgwas_merge_scatter_sizes", [])) == len(files)
        else _resolve_merge_series_values(
            getattr(args, "scatter_size_spec", None),
            len(files),
            default=float(getattr(args, "_postgwas_single_scatter_size", _DEFAULT_SCATTER_SIZE)),
        )
    )
    series_alphas = (
        [float(x) for x in list(getattr(args, "_postgwas_merge_alphas", []))]
        if len(getattr(args, "_postgwas_merge_alphas", [])) == len(files)
        else _resolve_merge_series_values(
            getattr(args, "alpha_spec", None),
            len(files),
            default=float(_DEFAULT_MERGE_ALPHA),
        )
    )
    legend_alpha = max(series_alphas) if len(series_alphas) > 0 else float(_DEFAULT_MERGE_ALPHA)
    legend_size_base = (
        max(series_sizes)
        if len(series_sizes) > 0
        else float(getattr(args, "_postgwas_single_scatter_size", _DEFAULT_SCATTER_SIZE))
    )
    legend_size = max(
        float(legend_size_base) * float(_POSTGWAS_LEGEND_SIZE_SCALE),
        float(legend_size_base) + float(_POSTGWAS_LEGEND_SIZE_MIN_BONUS),
    )
    series_labels = [f"{i + 1}:{os.path.basename(path)}" for i, path in enumerate(files)]
    ldblock_style = _resolve_ldblock_style(args.ldblock_palette_spec)

    frames_raw: list[pd.DataFrame] = []
    chrom_sets: list[tuple[int, str, set[str]]] = []
    pvals_by_series: dict[int, np.ndarray] = {}
    for i, file in enumerate(files):
        df = _read_merge_gwas_table(
            file,
            chr_col,
            pos_col,
            p_col,
            logger,
            pvalue_is_log10=no_logtrans,
        )
        if df.shape[0] == 0:
            logger.warning(f"Warning: no valid SNP rows in merged file {i}: {file}; skipped.")
            continue

        chrom_set = set(df[chr_col].astype(str).tolist())
        chrom_sets.append((i, file, chrom_set))
        pvals = np.asarray(df[p_col], dtype=float)
        if no_logtrans:
            pvals = _postgwas_plot_logp_values(pvals, no_logtrans=True)
        else:
            pvals = np.clip(pvals, np.nextafter(0.0, 1.0), np.inf)
        pvals_by_series[i] = pvals

        dfi = pd.DataFrame(
            {
                chr_col: df[chr_col].astype(str).to_numpy(),
                pos_col: np.asarray(df[pos_col], dtype=np.int64),
                p_col: pvals,
                "_series_idx": i,
            }
        )
        frames_raw.append(dfi)

    if len(frames_raw) == 0:
        logger.error("No valid SNPs available for merged plotting.")
        return

    # Local plots share explicit physical ranges, so input files need not
    # cover the same chromosomes outside those ranges (e.g. sparse highlights).
    if needs_positional_merge and args.bimrange_tuples is None and len(chrom_sets) >= 2:
        ref_i, ref_file, ref_chroms = chrom_sets[0]
        mismatch_items: list[tuple[int, str, int, int]] = []
        for i, file, chroms in chrom_sets[1:]:
            if chroms != ref_chroms:
                n_missing = int(len(ref_chroms - chroms))
                n_extra = int(len(chroms - ref_chroms))
                mismatch_items.append((i, file, n_missing, n_extra))
        if len(mismatch_items) > 0:
            ref_label = f"{ref_i}-{ref_file}"
            logger.warning(
                "Warning: Chromosome sets are inconsistent across merge GWAS files; "
                "merge mode is ignored and fallback to single-GWAS plotting."
            )
            for i, file, n_missing, n_extra in mismatch_items[:3]:
                logger.warning(
                    f"  File {i}-{file}: missing {n_missing} / extra {n_extra} chromosomes "
                    f"vs reference {ref_label}."
                )
            if len(mismatch_items) > 3:
                logger.warning(
                    f"  ... {len(mismatch_items) - 3} more mismatched files omitted."
                )
            if bool(getattr(args, "_postgwas_single_requested", False)):
                logger.info(
                    "Visualizing merged post-GWAS results ...Skipped (per-file plotting will still run)"
                )
                return
            fallback_file = str(args.gwasfile[0]) if len(args.gwasfile) > 0 else files[0]
            logger.info("Visualizing merged post-GWAS results ...Skipped (fallback single-file plotting)")
            logger.info(f"Fallback single GWAS plotting file: {fallback_file}")
            GWASplot(fallback_file, args, logger)
            return

    plot_df = pd.concat(frames_raw, axis=0, ignore_index=True)
    bim_layout: list[dict[str, object]] = []
    if args.bimrange_tuples is not None:
        df_sel, seg_defs, n_before = _filter_df_by_bimranges(
            plot_df,
            chr_col,
            pos_col,
            args.bimrange_tuples,
            logger,
            "merge",
        )
        n_after = int(df_sel.shape[0])
        if n_after == 0:
            logger.warning(
                "No SNPs found in all bimrange settings for merged GWAS; merged Manhattan/LD/Gene plotting may be empty."
            )
            plot_df = df_sel
        else:
            plot_df = df_sel
            bim_layout = _build_bimrange_layout(
                seg_defs,
                interval_ratio=float(args.interval),
            )
            logger.info(
                f"Applied {len(args.bimrange_tuples)} bimrange settings in merge mode: kept {n_after}/{n_before} SNPs."
            )

    threshold_merge = (
        float(args.thr)
        if args.thr is not None
        else _postgwas_default_threshold(
            int(plot_df.shape[0]),
            no_logtrans=no_logtrans,
        )
    )
    draw_threshold_line = _postgwas_should_draw_threshold_line(
        args.thr,
        no_logtrans=no_logtrans,
    )

    xticks: list[float] = []
    xticklabels: list[str] = []
    x_separators: list[float] = []
    use_segmented_layout = len(bim_layout) > 0
    x_axis_left: Optional[float] = None
    x_axis_right: Optional[float] = None
    x_edge_padding: float = 0.0

    if needs_positional_merge:
        if use_segmented_layout:
            plot_df = plot_df.copy()
            plot_df["_x"] = np.nan
            for seg in bim_layout:
                sid = int(seg["id"])
                start = int(seg["start"])
                offset = float(seg["offset"])
                length = float(seg["length"])
                mask = plot_df["__seg_id"].to_numpy(dtype=np.int64) == sid
                if not bool(np.any(mask)):
                    continue
                posv = pd.to_numeric(plot_df.loc[mask, pos_col], errors="coerce").to_numpy(dtype=float)
                rel = np.clip(posv - float(start), 0.0, float(length))
                plot_df.loc[mask, "_x"] = offset + rel
            plot_df = plot_df[np.isfinite(pd.to_numeric(plot_df["_x"], errors="coerce"))].copy()
            xticks = [0.5 * (float(seg["x_start"]) + float(seg["x_end"])) for seg in bim_layout]
            xticklabels = [_sanitize_plot_text(seg["label"]) for seg in bim_layout]
            for i in range(len(bim_layout) - 1):
                x_end = float(bim_layout[i]["x_end"])
                x_next = float(bim_layout[i + 1]["x_start"])
                x_separators.append(0.5 * (x_end + x_next))
            x_axis_left = float(bim_layout[0]["x_start"])
            x_axis_right = float(bim_layout[-1]["x_end"])
            if len(bim_layout) > 1:
                seg_gaps = np.asarray(
                    [
                        float(bim_layout[i + 1]["x_start"]) - float(bim_layout[i]["x_end"])
                        for i in range(len(bim_layout) - 1)
                    ],
                    dtype=float,
                )
                seg_gaps = seg_gaps[np.isfinite(seg_gaps) & (seg_gaps > 0.0)]
                if seg_gaps.size > 0:
                    x_edge_padding = 0.5 * float(np.median(seg_gaps))
        else:
            min_pos_by_chr: dict[str, int] = {}
            max_pos_by_chr: dict[str, int] = {}
            if plot_df.shape[0] > 0:
                chr_pos = plot_df.groupby(chr_col)[pos_col]
                chr_min = chr_pos.min()
                chr_max = chr_pos.max()
                for chrom, min_pos in chr_min.items():
                    min_pos_by_chr[str(chrom)] = int(min_pos)
                for chrom, max_pos in chr_max.items():
                    max_pos_by_chr[str(chrom)] = int(max_pos)
            chrom_order = sorted(max_pos_by_chr.keys(), key=_chrom_sort_key)
            if len(chrom_order) == 0:
                logger.error("No chromosome labels available for merged Manhattan plotting.")
                return
            chr_lens = np.asarray(
                [max(1, int(max_pos_by_chr[c])) for c in chrom_order],
                dtype=float,
            )
            gap = float(resolve_manhattan_chr_gap(
                chr_lens,
                interval_ratio=float(args.interval),
            ))
            offsets: dict[str, float] = {}
            cursor = 0.0
            for i, chrom in enumerate(chrom_order):
                length = max(1, int(max_pos_by_chr[chrom]))
                offsets[chrom] = cursor
                xticks.append(float(cursor) + float(length) / 2.0)
                xticklabels.append(_sanitize_plot_text(chrom))
                if i < len(chrom_order) - 1:
                    x_separators.append(float(cursor) + float(length) + float(gap) / 2.0)
                cursor += length + gap
            plot_df = plot_df.copy()
            plot_df["_x"] = (
                plot_df[chr_col].astype(str).map(offsets).fillna(0).astype(float)
                + pd.to_numeric(plot_df[pos_col], errors="coerce").fillna(0.0).astype(float)
            )
            first_chr = chrom_order[0]
            last_chr = chrom_order[-1]
            x_axis_left = float(offsets[first_chr]) + float(min_pos_by_chr.get(first_chr, 0))
            x_axis_right = float(offsets[last_chr]) + float(max_pos_by_chr.get(last_chr, 1))
            if gap > 0.0:
                x_edge_padding = 0.5 * float(gap)

    if plot_df.shape[0] > 0:
        pvals_draw = np.asarray(plot_df[p_col], dtype=float)
        plot_df["_ylog"] = _postgwas_plot_logp_values(
            pvals_draw,
            no_logtrans=no_logtrans,
        )
    else:
        plot_df["_ylog"] = np.asarray([], dtype=float)

    width_in = float(_PANEL_WIDTH_IN)
    # Merged plots can contain multiple dense layers; always rasterize point/band
    # artists while keeping axes/text/legend vector-friendly.
    rasterized = True

    def _draw_merge_manhattan_axis(
        ax: plt.Axes,
        *,
        font_size: float,
        include_legend: bool,
    ) -> tuple[tuple[float, float], np.ndarray, tuple[float, float]]:
        legend_handles: list[object] = []
        draw_xmins: list[float] = []
        draw_xmaxs: list[float] = []
        thr_log = (
            _postgwas_threshold_to_logp(
                threshold_merge,
                no_logtrans=no_logtrans,
            )
            if draw_threshold_line and np.isfinite(threshold_merge)
            else None
        )
        if thr_log is not None and np.isfinite(thr_log):
            ax.axhline(
                y=thr_log,
                color="grey",
                linewidth=1.0,
                linestyle="--",
                zorder=0,
            )

        for i, _file in enumerate(files):
            dfi = plot_df.loc[plot_df["_series_idx"] == i]
            if dfi.shape[0] == 0:
                continue
            yy = np.asarray(dfi["_ylog"], dtype=float)
            pvals_i = np.asarray(dfi[p_col], dtype=float)
            keep = np.isfinite(yy)
            if args.ylim_min is not None:
                keep = keep & (yy >= float(args.ylim_min))
            if args.ylim_max is not None:
                keep = keep & (yy <= float(args.ylim_max))
            if not bool(np.any(keep)):
                continue
            x_keep = np.asarray(dfi.loc[keep, "_x"], dtype=float)
            y_keep = yy[keep]
            p_keep = pvals_i[keep]
            sig_keep = (
                _postgwas_significant_p_mask(
                    p_keep,
                    threshold_merge,
                    no_logtrans=no_logtrans,
                )
                if args.thr is not None
                else np.ones(p_keep.shape, dtype=bool)
            )
            nonsig_keep = ~sig_keep
            alpha_i = float(series_alphas[i])
            size_i = float(series_sizes[i])
            if bool(np.any(nonsig_keep)):
                ax.scatter(
                    x_keep[nonsig_keep],
                    y_keep[nonsig_keep],
                    color="lightgrey",
                    marker=series_markers[i],
                    alpha=alpha_i,
                    s=size_i,
                    rasterized=rasterized,
                    **_marker_scatter_style(series_markers[i]),
                )
            if bool(np.any(sig_keep)):
                ax.scatter(
                    x_keep[sig_keep],
                    y_keep[sig_keep],
                    color=series_colors[i],
                    marker=series_markers[i],
                    alpha=alpha_i,
                    s=size_i,
                    rasterized=rasterized,
                    **_marker_scatter_style(series_markers[i]),
                )
            if include_legend:
                legend_handles.append(
                    ax.scatter(
                        [],
                        [],
                        color=series_colors[i],
                        marker=series_markers[i],
                        alpha=float(legend_alpha),
                        s=float(legend_size),
                        label=series_labels[i],
                        **_marker_scatter_style(series_markers[i]),
                    )
                )
            if x_keep.size > 0:
                draw_xmins.append(float(np.nanmin(x_keep)))
                draw_xmaxs.append(float(np.nanmax(x_keep)))

        ax.set_xlabel("Chromosome")
        ax.set_ylabel(
            _postgwas_manhattan_y_label(
                p_col,
                no_logtrans=no_logtrans,
            )
            or "-log10(p)"
        )
        if (
            x_axis_left is not None
            and x_axis_right is not None
            and np.isfinite(float(x_axis_left))
            and np.isfinite(float(x_axis_right))
        ):
            xmin = float(x_axis_left) - float(max(0.0, x_edge_padding))
            xmax = float(x_axis_right) + float(max(0.0, x_edge_padding))
        elif len(draw_xmins) > 0 and len(draw_xmaxs) > 0:
            xmin = float(np.nanmin(np.asarray(draw_xmins, dtype=float)))
            xmax = float(np.nanmax(np.asarray(draw_xmaxs, dtype=float)))
        else:
            xall = pd.to_numeric(plot_df["_x"], errors="coerce").to_numpy(dtype=float)
            xall = xall[np.isfinite(xall)]
            if xall.size > 0:
                xmin = float(np.nanmin(xall))
                xmax = float(np.nanmax(xall))
            else:
                xmin, xmax = (0.0, 1.0)
        if xmax > xmin:
            ax.set_xlim(xmin, xmax)
        else:
            eps = max(1e-9, abs(xmin) * 1e-9)
            ax.set_xlim(xmin - eps, xmax + eps)
        ax.margins(x=0.0)

        if use_segmented_layout and len(bim_layout) > 0:
            _apply_multi_bimrange_manhattan_axis(
                ax,
                bim_layout,
                label_fontsize=float(font_size),
            )
            if (
                x_axis_left is not None
                and x_axis_right is not None
                and np.isfinite(float(x_axis_left))
                and np.isfinite(float(x_axis_right))
            ):
                left = float(x_axis_left) - float(max(0.0, x_edge_padding))
                right = float(x_axis_right) + float(max(0.0, x_edge_padding))
                if right > left:
                    ax.set_xlim(left, right)
        else:
            for xsep in x_separators:
                ax.axvline(
                    xsep,
                    ymin=0.0,
                    ymax=1.0 / 3.0,
                    linestyle="--",
                    color="lightgrey",
                    linewidth=0.6,
                    alpha=0.8,
                    zorder=8,
                )
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=0)
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)
        ax.tick_params(axis="both", labelsize=font_size)

        if include_legend and len(legend_handles) > 0:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                borderaxespad=0.0,
                ncol=1,
                markerscale=1.0,
                fontsize=font_size,
            )
        _y0, _y1 = ax.get_ylim()
        lo = float(args.ylim_min) if args.ylim_min is not None else 0.0
        hi = float(args.ylim_max) if args.ylim_max is not None else float(_y1)
        if not (hi > lo):
            hi = lo + max(1e-9, abs(lo) * 1e-9)
        ax.set_ylim(lo, hi)
        manh_tick_values = apply_integer_yticks(ax)
        return (
            (float(ax.get_ylim()[0]), float(ax.get_ylim()[1])),
            np.asarray(manh_tick_values, dtype=float),
            (float(ax.get_xlim()[0]), float(ax.get_xlim()[1])),
        )

    def _draw_merge_ld_axis(ax: plt.Axes, *, font_size: float) -> None:
        LDblock(ld_mat, ax=ax, vmin=0, vmax=1, cmap=ld_cmap, rasterize_threshold=100)
        n_ld = max(2, int(ld_mat.shape[0]))
        ax.set_xlim(0.5, float(n_ld) - 0.5)
        ax.margins(x=0.0)
        if ld_overlay_text:
            ax.text(
                n_ld / 2.0,
                -n_ld / 2.0,
                ld_overlay_text,
                ha="center",
                va="center",
                fontsize=float(font_size),
            )

    def _build_merge_ld_bridge_pairs(
        ld_keys: list[tuple[str, int]],
    ) -> list[tuple[float, float, bool]]:
        if len(ld_keys) == 0 or plot_df.shape[0] == 0:
            return []
        chr_vals = plot_df[chr_col].astype(str).map(_normalize_chr).to_numpy(dtype=object)
        pos_vals = pd.to_numeric(plot_df[pos_col], errors="coerce").to_numpy(dtype=float)
        x_vals = pd.to_numeric(plot_df["_x"], errors="coerce").to_numpy(dtype=float)
        p_vals = pd.to_numeric(plot_df[p_col], errors="coerce").to_numpy(dtype=float)
        key_to_meta: dict[tuple[str, int], tuple[float, bool]] = {}
        for chrom_norm, posv, xv, pv in zip(chr_vals, pos_vals, x_vals, p_vals):
            if not (
                np.isfinite(posv)
                and np.isfinite(xv)
                and bool(
                    _postgwas_valid_p_mask(
                        [pv],
                        no_logtrans=no_logtrans,
                    )[0]
                )
            ):
                continue
            key = (str(chrom_norm), int(round(float(posv))))
            is_sig = args.thr is not None and bool(
                _postgwas_significant_p_mask(
                    [pv],
                    threshold_merge,
                    no_logtrans=no_logtrans,
                )[0]
            )
            if key not in key_to_meta:
                key_to_meta[key] = (float(xv), is_sig)
            else:
                old_x, old_sig = key_to_meta[key]
                key_to_meta[key] = (old_x, bool(old_sig or is_sig))
        pairs: list[tuple[float, float, bool]] = []
        n_ld = int(ld_mat.shape[0])
        for i, key in enumerate(ld_keys[:n_ld]):
            meta = key_to_meta.get((str(key[0]), int(key[1])))
            if meta is None:
                continue
            x_top, is_sig = meta
            pairs.append((float(x_top), float(i) + 0.5, bool(is_sig)))
        return pairs

    def _draw_merge_bridge_axis(
        ax_bridge: plt.Axes,
        ax_manh: plt.Axes,
        ax_ld: plt.Axes,
        pairs: list[tuple[float, float, bool]],
        *,
        line_color: str = "grey",
        sig_line_color: str = "red",
    ) -> None:
        ax_bridge.set_xlim(0.0, 1.0)
        ax_bridge.set_ylim(0.0, 1.0)
        ax_bridge.set_xticks([])
        ax_bridge.set_yticks([])
        for spine in ax_bridge.spines.values():
            spine.set_visible(False)
        if len(pairs) == 0:
            return

        edge_margin_n = 0.006
        y_manh_ref = float(ax_manh.get_ylim()[0])
        y_ld_ref = float(_POSTGWAS_LD_LINK_Y)
        to_bridge = ax_bridge.transAxes.inverted()
        for x_top, x_ld, is_sig in pairs:
            p_top_disp = ax_manh.transData.transform((float(x_top), y_manh_ref))
            p_ld_disp = ax_ld.transData.transform((float(x_ld), y_ld_ref))
            x_top_n = float(to_bridge.transform(p_top_disp)[0])
            x_ld_n = float(to_bridge.transform(p_ld_disp)[0])
            if not (np.isfinite(x_top_n) and np.isfinite(x_ld_n)):
                continue
            x_top_n = float(np.clip(x_top_n, edge_margin_n, 1.0 - edge_margin_n))
            x_ld_n = float(np.clip(x_ld_n, edge_margin_n, 1.0 - edge_margin_n))
            joint_y = 0.84
            bottom_y = 0.06
            poly_color = str(sig_line_color) if bool(is_sig) else str(line_color)
            ax_bridge.plot(
                [x_top_n, x_top_n, x_ld_n],
                [1.04, joint_y, bottom_y],
                color=poly_color,
                linewidth=0.25,
                alpha=0.8,
                clip_on=False,
                solid_joinstyle="round",
            )

    manh_path = None
    manh_height_in = None
    manh_fontsize: Optional[float] = None
    manh_ylim_pair: Optional[tuple[float, float]] = None
    manh_yticks_pair: Optional[np.ndarray] = None
    manh_xlim_pair: Optional[tuple[float, float]] = None
    manh_axes_bounds: Optional[tuple[float, float, float, float]] = None

    if merge_manh_ratio is not None:
        manh_ratio = float(merge_manh_ratio)
        manh_fontsize = _postgwas_resolve_fontsize(
            args,
            manh_ratio=manh_ratio,
        )
        fig, ax, _manh_panel_w_in, manh_panel_h_in = _create_ratio_panel_figure(
            ratio=manh_ratio,
            dpi=300,
            panel_width_in=width_in,
            reserve_right_in=_PANEL_LEGEND_RIGHT_IN,
        )
        manh_ylim_pair, manh_yticks_pair, manh_xlim_pair = _draw_merge_manhattan_axis(
            ax,
            font_size=float(manh_fontsize),
            include_legend=True,
        )

        manh_axes_bounds = ax.get_position().bounds
        manh_path = os.path.join(args.out, f"{args.prefix}.merge.manh.{args.format}")
        _save_figure(fig, manh_path)
        manh_height_in = float(manh_panel_h_in)
        plt.close(fig)

    qq_path = None
    if merge_qq_ratio is not None:
        qq_fontsize = (
            float(manh_fontsize)
            if manh_fontsize is not None
            else _postgwas_resolve_fontsize(
                args,
                manh_ratio=float(merge_qq_ratio),
            )
        )
        qq_y_in = manh_height_in if manh_height_in is not None else 4.0
        fig, ax, _qq_panel_w_in, _qq_panel_h_in = _create_ratio_panel_figure(
            ratio=float(merge_qq_ratio),
            dpi=300,
            panel_height_in=qq_y_in,
            reserve_right_in=_PANEL_LEGEND_RIGHT_IN,
        )
        legend_handles: list[object] = []

        exp_xmin = np.inf
        exp_xmax = -np.inf
        qq_band_n = 0
        for i, _file in enumerate(files):
            pvals = pvals_by_series.get(i)
            if pvals is None or pvals.size == 0:
                continue
            pvals_arr = np.asarray(pvals, dtype=float)
            qq_band_n = max(
                int(qq_band_n),
                int(
                    np.sum(
                        _postgwas_valid_p_mask(
                            pvals_arr,
                            no_logtrans=no_logtrans,
                        )
                    )
                ),
            )
            exp, obs = _qq_select_points_with_threshold(
                pvals_arr,
                sig_p_threshold=(
                    float(threshold_merge)
                    if (np.isfinite(threshold_merge) and float(threshold_merge) > 0.0)
                    else None
                ),
                max_points=_QQ_FAST_MAX_POINTS,
                keep_all=bool(args.fullscatter),
                no_logtrans=no_logtrans,
            )
            if exp.size == 0 or obs.size == 0:
                continue
            if exp.size > 0:
                exp_xmin = min(exp_xmin, float(np.nanmin(exp)))
                exp_xmax = max(exp_xmax, float(np.nanmax(exp)))
            ax.scatter(
                exp,
                obs,
                s=float(series_sizes[i]),
                marker=series_markers[i],
                alpha=float(series_alphas[i]),
                rasterized=rasterized,
                color=series_colors[i],
                **_marker_scatter_style(series_markers[i]),
            )
            legend_handles.append(
                ax.scatter(
                    [],
                    [],
                    s=float(legend_size),
                    marker=series_markers[i],
                    alpha=float(legend_alpha),
                    color=series_colors[i],
                    label=series_labels[i],
                    **_marker_scatter_style(series_markers[i]),
                )
            )

        if qq_band_n > 0:
            x_band, lower_band, upper_band = _qq_confidence_band_from_n(
                int(qq_band_n),
                max_points=_QQ_BAND_MAX_POINTS,
            )
            if x_band.size > 0:
                exp_xmin = min(exp_xmin, float(np.nanmin(x_band)))
                exp_xmax = max(exp_xmax, float(np.nanmax(x_band)))
                ax.fill_between(
                    x_band,
                    lower_band,
                    upper_band,
                    color=str(_QQ_BAND_COLOR),
                    alpha=0.25,
                    rasterized=rasterized,
                    zorder=0,
                )

        if not np.isfinite(exp_xmin) or not np.isfinite(exp_xmax):
            exp_xmin, exp_xmax = (0.0, 1.0)
        qq_lower = (
            float(manh_ylim_pair[0])
            if manh_ylim_pair is not None
            else (
                float(args.ylim_min)
                if args.ylim_min is not None
                else 0.0
            )
        )
        qq_upper_target = (
            float(manh_ylim_pair[1])
            if manh_ylim_pair is not None
            else (
                float(args.ylim_max)
                if args.ylim_max is not None
                else None
            )
        )
        if exp_xmax > exp_xmin:
            x_right = float(exp_xmax)
        else:
            eps = max(1e-9, abs(exp_xmin) * 1e-9)
            x_right = float(exp_xmax + eps)
        qq_lower, qq_upper = _resolve_qq_ylim(
            ax,
            lower=qq_lower,
            upper=qq_upper_target,
        )
        _apply_qq_axes(
            ax,
            y_lower=qq_lower,
            y_upper=qq_upper,
            x_right=x_right,
            y_ticks=manh_yticks_pair,
        )
        apply_integer_xticks(ax)
        line_left, line_right = ax.get_xlim()
        ax.plot([line_left, line_right], [line_left, line_right], lw=1.0, color="black")
        ax.set_xlabel("Expected -log10(p-value)")
        ax.set_ylabel("Observed -log10(p-value)")
        ax.xaxis.label.set_size(qq_fontsize)
        ax.yaxis.label.set_size(qq_fontsize)
        ax.tick_params(axis="both", labelsize=qq_fontsize)
        if len(legend_handles) > 0:
            ax.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                borderaxespad=0.0,
                ncol=1,
                markerscale=1.0,
                fontsize=qq_fontsize,
            )
        qq_path = os.path.join(args.out, f"{args.prefix}.merge.qq.{args.format}")
        _save_figure(fig, qq_path)
        plt.close(fig)

    ld_path = None
    gene_path = None
    manhld_path = None
    effective_ldblock_ratio = (
        args.ldblock_ratio
        if bool(getattr(args, "_postgwas_merge_ldblock_requested", False))
        else None
    )
    region_ranges = args.bimrange_tuples if args.bimrange_tuples is not None else []
    if effective_ldblock_ratio is not None:
        ld_use_all_sites = bool(args.ldblock_all is not None)
        ld_sites = _extract_ld_site_set(
            plot_df,
            chr_col,
            pos_col,
            p_col,
            threshold_merge,
            use_all_sites=ld_use_all_sites,
            no_logtrans=no_logtrans,
        )
        n_sig_sites = max(2, len(ld_sites))
        ld_overlay_text = None
        ld_site_keys: list[tuple[str, int]] = sorted(ld_sites, key=lambda x: (x[0], x[1]))

        if args.genofile is None:
            logger.warning(
                "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; drawing zero-correlation LD block."
            )
            ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
            ld_overlay_text = "No genotype"
        else:
            if len(ld_sites) < 2:
                if ld_use_all_sites:
                    logger.warning(
                        "Warning: Fewer than 2 valid SNPs in selected region; drawing empty LD block."
                    )
                else:
                    logger.warning(
                        "Warning: Fewer than 2 threshold-passing SNPs in selected region; drawing empty LD block."
                    )
                ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                ld_overlay_text = "Not enough SNPs"
            else:
                ld_mat, sig_keys = _compute_ld_from_bed_rust(
                    str(args.genofile),
                    region_ranges,
                    selected_sites=ld_sites,
                    threads=int(max(0, int(getattr(args, "thread", 0)))),
                    logger=logger,
                )
                missing_n = int(len(ld_sites) - len(set(sig_keys)))
                if missing_n > 0:
                    logger.warning(
                        f"Warning: {missing_n} requested LD sites are not in genotype data and were ignored."
                    )
                if len(sig_keys) > 0:
                    ld_site_keys = sig_keys
                if ld_mat.shape[0] < 2:
                    logger.warning(
                        "Warning: Requested SNPs were not found in genotype data; drawing empty LD block."
                    )
                    ld_mat = np.zeros((n_sig_sites, n_sig_sites), dtype=np.float32)
                    ld_overlay_text = "No matched SNPs"
                else:
                    mode_text = "all SNPs" if ld_use_all_sites else "threshold-passing SNPs"
                    logger.info(f"Merge LD block built from {len(sig_keys)} {mode_text}.")

        ld_cmap = "Greys"
        gene_block_color = "grey"
        gene_line_color = "black"
        if ldblock_style is not None:
            ld_cmap = ldblock_style["ld_cmap"]
            gene_block_color = str(ldblock_style["gene_block_color"])
            gene_line_color = str(ldblock_style["gene_line_color"])

        gene_track_df = pd.DataFrame(
            columns=["feature", "strand", "attribute", "x_start", "x_end"]
        )
        use_gene_bridge = False
        if args.anno_file:
            if len(region_ranges) == 0:
                logger.warning(
                    "Warning: annotation source in merge mode requires --bimrange for gene-structure plotting; gene plot skipped."
                )
            else:
                anno_is_gff = _postgwas_annotation_is_gff(
                    args.anno_file,
                    annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
                )
                use_shared_gff = bool(getattr(args, "_postgwas_use_shared_gff", False))
                gff_query_cache: Optional[GFFQuery] = None
                gff_rust_index_cache: Optional[object] = None
                if anno_is_gff:
                    gff_rust_index_cache = _postgwas_get_gff_rust_index(
                        args.anno_file,
                        use_shared=use_shared_gff,
                    )
                    if gff_rust_index_cache is None:
                        gff_query_cache = _postgwas_get_gff_query(
                            args.anno_file,
                            use_shared=use_shared_gff,
                        )
                gene_raw = _load_gene_like_records_from_anno(
                    args.anno_file,
                    region_ranges,
                    logger,
                    annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
                    gff_query=gff_query_cache,
                    gff_rust_index=gff_rust_index_cache,
                )
                region_layout = (
                    bim_layout
                    if len(bim_layout) > 0
                    else _build_layout_from_bimrange_tuples(
                        region_ranges,
                        interval_ratio=float(args.interval),
                    )
                )
                use_segmented_gene_x = len(region_layout) > 0
                gene_track_df = _project_gene_records_to_plot_x(
                    gene_raw,
                    region_ranges,
                    region_layout,
                    use_segmented_x=use_segmented_gene_x,
                )
                if gene_track_df.shape[0] == 0:
                    logger.warning(
                        "Warning: No gene-structure records found in selected --bimrange; gene plot is skipped."
                    )
                else:
                    use_gene_bridge = True
                    if len(region_layout) > 0:
                        gene_xlim = (
                            float(region_layout[0]["x_start"]),
                            float(region_layout[-1]["x_end"]),
                        )
                    else:
                        gene_xlim = (
                            float(min(int(x[1]) for x in region_ranges)),
                            float(max(int(x[2]) for x in region_ranges)),
                        )
                    gene_plot_xlim = manh_xlim_pair if manh_xlim_pair is not None else gene_xlim
                    gene_fontsize = float(
                        manh_fontsize
                        if manh_fontsize is not None
                        else _postgwas_resolve_fontsize(args, manh_ratio=float(effective_ldblock_ratio))
                    )
                    gene_h_in = _resolve_postgwas_gene_panel_height_in(
                        gene_fontsize,
                        width_in=width_in,
                    )

                    fig_gene = plt.figure(
                        figsize=(width_in, gene_h_in),
                        dpi=300,
                    )
                    ax_gene = fig_gene.add_subplot(111)
                    _draw_gene_structure_axis(
                        ax_gene,
                        gene_track_df,
                        arrow_color=gene_line_color,
                        block_color=gene_block_color,
                        line_width=1.0,
                        arrow_step=1_000.0,
                        gene_text_size=gene_fontsize,
                    )
                    ax_gene.set_xlim(gene_plot_xlim)
                    _apply_postgwas_gene_panel_layout(
                        fig_gene,
                        ax_gene,
                        x_align_bounds=manh_axes_bounds,
                    )
                    gene_path = os.path.join(args.out, f"{args.prefix}.merge.gene.{args.format}")
                    _save_figure_and_close(fig_gene, gene_path)

        ld_h_in = width_in / effective_ldblock_ratio
        ld_h_in = max(float(ld_h_in), float(width_in * _ld_min_height_over_width(max(2, int(ld_mat.shape[0])))))
        fig_ld = plt.figure(figsize=(width_in, ld_h_in), dpi=300)
        ax_ld = fig_ld.add_subplot(111)
        ld_fontsize = float(
            manh_fontsize
            if manh_fontsize is not None
            else _postgwas_resolve_fontsize(args, manh_ratio=float(effective_ldblock_ratio))
        )
        _draw_merge_ld_axis(
            ax_ld,
            font_size=ld_fontsize,
        )
        fig_ld.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
        if manh_axes_bounds is not None:
            _cx0, cy0, _cw, ch = ax_ld.get_position().bounds
            mx0, _my0, mw, _mh = manh_axes_bounds
            if args.ldblock_xspan is None:
                fx0, fx1 = (0.0, 1.0)
            else:
                fx0, fx1 = args.ldblock_xspan
            new_x0 = float(mx0) + float(mw) * float(fx0)
            new_w = float(mw) * float(fx1 - fx0)
            ax_ld.set_position([new_x0, cy0, new_w, ch])
            ax_ld.set_anchor("N")
        ld_path = os.path.join(args.out, f"{args.prefix}.merge.ldblock.{args.format}")
        _save_figure_and_close(fig_ld, ld_path)

        manhld_ratio = (
            float(merge_manh_ratio)
            if merge_manh_ratio is not None
            else 2.0
        )
        manhld_fontsize = float(
            manh_fontsize
            if manh_fontsize is not None
            else _postgwas_resolve_fontsize(args, manh_ratio=manhld_ratio)
        )
        manhld_manh_h_in = width_in / float(manhld_ratio)
        gene_bridge_scale = 0.8
        mid_gene_y_offset = 0.03
        mid_gene_ymin = -0.18
        mid_gene_ymax = 0.22
        mid_h_in = _resolve_postgwas_bridge_panel_height_in(
            manhld_fontsize,
            width_in=width_in,
            use_gene_bridge=use_gene_bridge,
        )
        if args.ldblock_xspan is None:
            ld_panel_frac_in_manh = 1.0
        else:
            ld_panel_frac_in_manh = float(args.ldblock_xspan[1] - args.ldblock_xspan[0])
        ld_h_in_combo = (
            width_in
            * ld_panel_frac_in_manh
            / effective_ldblock_ratio
        )
        ld_h_in_combo_min = (
            width_in
            * ld_panel_frac_in_manh
            * _ld_min_height_over_width(max(2, int(ld_mat.shape[0])))
        )
        ld_h_in_combo = max(0.5, float(ld_h_in_combo), float(ld_h_in_combo_min))
        fig_manhld, combo_axes, _combo_panel_w_in, _combo_panel_heights = _create_stacked_panel_figure(
            panel_width_in=width_in,
            panel_heights_in=[manhld_manh_h_in, mid_h_in, ld_h_in_combo],
            dpi=300,
            vspace_in=_PANEL_STACK_VSPACE_IN,
        )
        ax_manhld_top, ax_manhld_mid, ax_manhld_bot = combo_axes
        ax_manhld_top.set_zorder(5)
        ax_manhld_mid.set_zorder(2)
        ax_manhld_bot.set_zorder(1)
        _set_postgwas_axis_transparent(ax_manhld_mid)

        _tmp_ylim, _tmp_yticks, manhld_top_xlim = _draw_merge_manhattan_axis(
            ax_manhld_top,
            font_size=manhld_fontsize,
            include_legend=False,
        )
        _draw_merge_ld_axis(
            ax_manhld_bot,
            font_size=ld_fontsize,
        )
        if use_gene_bridge:
            _draw_gene_structure_axis(
                ax_manhld_mid,
                gene_track_df,
                arrow_color=gene_line_color,
                block_color=gene_block_color,
                line_width=1.0,
                arrow_step=1_000.0,
                thickness_scale=gene_bridge_scale,
                y_offset=mid_gene_y_offset,
                gene_text_size=manhld_fontsize,
            )
            ax_manhld_mid.set_xlim(manhld_top_xlim)
            ax_manhld_mid.set_ylim(mid_gene_ymin, mid_gene_ymax)

        fig_manhld.canvas.draw()
        bx0, by0, bw, bh = ax_manhld_bot.get_position().bounds
        tx0, _ty0, tw, _th = ax_manhld_top.get_position().bounds
        if args.ldblock_xspan is None:
            new_bw = bw
            new_bx0 = bx0
        else:
            fx0, fx1 = args.ldblock_xspan
            new_bw = float(tw) * float(fx1 - fx0)
            new_bx0 = float(tx0) + float(tw) * float(fx0)
        ax_manhld_bot.set_position([new_bx0, by0, new_bw, bh])
        ax_manhld_bot.set_anchor("N")
        fig_manhld.canvas.draw()

        bridge_pairs = _build_merge_ld_bridge_pairs(ld_site_keys)
        if use_gene_bridge:
            ax_manhld_mid.set_xlim(ax_manhld_top.get_xlim())
            ax_manhld_mid.set_ylim(mid_gene_ymin, mid_gene_ymax)
            _draw_manh_gene_ld_links(
                fig_manhld,
                ax_manhld_mid,
                ax_manhld_top,
                ax_manhld_bot,
                bridge_pairs,
                gene_route_y=-0.17 * gene_bridge_scale + mid_gene_y_offset,
                nonsig_line_color="grey",
            )
        else:
            _draw_merge_bridge_axis(
                ax_manhld_mid,
                ax_manhld_top,
                ax_manhld_bot,
                bridge_pairs,
                line_color="grey",
                sig_line_color="red",
            )
        if args.bimrange_tuples is None:
            _show_end_locs_without_xticks(
                ax_manhld_top,
                label_fontsize=float(manhld_fontsize),
            )
        manhld_path = os.path.join(args.out, f"{args.prefix}.merge.manhld.{args.format}")
        _save_figure_and_close(fig_manhld, manhld_path)

    saved_paths: list[tuple[str, str]] = []
    if manh_path is not None:
        saved_paths.append(("Merged Manhattan", manh_path))
    if qq_path is not None:
        saved_paths.append(("Merged QQ", qq_path))
    if ld_path is not None:
        saved_paths.append(("Merged LD block", ld_path))
    if gene_path is not None:
        saved_paths.append(("Merged Gene structure", gene_path))
    if manhld_path is not None:
        saved_paths.append(("Merged Manhattan+LD", manhld_path))

    if len(saved_paths) == 0:
        logger.warning("Warning: no merged figure was generated (both --manh and --qq are off).")
    elif len(saved_paths) == 1:
        log_success(
            logger,
            f"{saved_paths[0][0]} plot saved to:\n  {format_path_for_display(saved_paths[0][1])}\n",
        )
    else:
        title = ", ".join([x[0] for x in saved_paths])
        body = "\n".join([f"  {format_path_for_display(x[1])}" for x in saved_paths])
        log_success(logger, f"{title} plots saved to:\n{body}\n")
    log_success(
        logger,
        f"Visualizing merged post-GWAS results ...Finished [{format_elapsed(time.time() - t_merge)}]",
    )


def _run_one_postgwas_task(file: str, args, logger: logging.Logger) -> str:
    logger = _ensure_postgwas_worker_file_logging(args, logger)
    mute_stream = bool(getattr(args, "_postgwas_worker_mute_stream", False))
    detached_handlers: list[logging.Handler] = []
    thread_ctx = (
        runtime_thread_stage(blas_threads=1, rayon_threads=1)
        if int(getattr(args, "_postgwas_job_workers", 1)) > 1
        else nullcontext()
    )
    if mute_stream:
        detached_handlers = _detach_stream_handlers(logger)
    try:
        with thread_ctx:
            if mute_stream:
                # In parallel worker mode, fully silence worker stdout/stderr to
                # avoid corrupting parent-side progress/spinner rendering.
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        GWASplot(file, args, logger)
            else:
                GWASplot(file, args, logger)
    finally:
        if mute_stream:
            _restore_handlers(logger, detached_handlers)
    return str(file)


def _run_postgwas_ldblock_only(args, logger: logging.Logger) -> None:
    """
    Draw LD block using genotype input only (no GWAS table).
    """
    if args.ldblock_ratio is None:
        logger.warning("Warning: LD-only mode requires --ldblock/--ldblock-all.")
        return
    if args.bimrange_tuples is None or len(args.bimrange_tuples) == 0:
        logger.warning("Warning: LD-only mode requires --bimrange; skipped.")
        return

    if args.ldblock_mode == "threshold":
        logger.warning(
            "Warning: --ldblock (threshold mode) requires GWAS p-values; "
            "without --gwasfile it falls back to all SNPs in --bimrange."
        )

    _apply_postgwas_matplotlib_style(args)

    t_ld = time.time()
    logger.info("Visualizing LD block...")
    width_in = 8.0
    ld_overlay_text: Optional[str] = None
    ld_fontsize = _postgwas_resolve_fontsize(
        args,
        manh_ratio=float(args.ldblock_ratio),
    )

    if args.genofile is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; "
            "drawing zero-correlation LD block."
        )
        ld_mat = np.zeros((2, 2), dtype=np.float32)
        ld_site_keys: list[tuple[str, int]] = []
        ld_overlay_text = "No genotype"
        n_sites = 0
    else:
        ld_mat, ld_site_keys = _compute_ld_from_bed_rust(
            str(args.genofile),
            list(args.bimrange_tuples),
            selected_sites=None,
            threads=int(max(0, int(getattr(args, "thread", 0)))),
            logger=logger,
        )
        n_sites = int(ld_mat.shape[0])
        if n_sites < 2:
            logger.warning(
                "Warning: Fewer than 2 SNPs were found in selected --bimrange; "
                "drawing empty LD block."
            )
            ld_mat = np.zeros((max(2, n_sites), max(2, n_sites)), dtype=np.float32)
            ld_overlay_text = "Not enough SNPs"
        else:
            logger.info(f"LD-only block built from {n_sites} SNPs.")
    if args.genofile is None:
        ld_site_keys = []

    ldblock_style = _resolve_ldblock_style(args.ldblock_palette_spec)
    ld_cmap = "Greys"
    gene_block_color = "grey"
    gene_line_color = "black"
    if ldblock_style is not None:
        ld_cmap = ldblock_style["ld_cmap"]
        gene_block_color = str(ldblock_style["gene_block_color"])
        gene_line_color = str(ldblock_style["gene_line_color"])

    region_ranges = list(args.bimrange_tuples)
    ld_spans = _ld_bimrange_spans(ld_site_keys, region_ranges)
    show_ld_titles = True  # LD-only has no Manhattan panel above.

    gene_track_df = pd.DataFrame(columns=["feature", "strand", "attribute", "x_start", "x_end"])
    use_gene_panel = False
    if args.anno_file:
        anno_is_gff = _postgwas_annotation_is_gff(
            args.anno_file,
            annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
        )
        use_shared_gff = bool(getattr(args, "_postgwas_use_shared_gff", False))
        gff_query_cache: Optional[GFFQuery] = None
        gff_rust_index_cache: Optional[object] = None
        if anno_is_gff:
            gff_rust_index_cache = _postgwas_get_gff_rust_index(
                args.anno_file,
                use_shared=use_shared_gff,
            )
            if gff_rust_index_cache is None:
                gff_query_cache = _postgwas_get_gff_query(
                    args.anno_file,
                    use_shared=use_shared_gff,
                )
        gene_raw = _load_gene_like_records_from_anno(
            args.anno_file,
            region_ranges,
            logger,
            annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
            gff_query=gff_query_cache,
            gff_rust_index=gff_rust_index_cache,
        )
        gene_track_df = _project_gene_records_to_ld_spans(
            gene_raw,
            region_ranges,
            ld_spans,
        )
        if gene_track_df.shape[0] == 0:
            logger.warning(
                "Warning: No gene-structure records found in selected --bimrange; "
                "gene track above LD block is skipped."
            )
        else:
            use_gene_panel = True

    ld_h_in = width_in / float(args.ldblock_ratio)
    ld_h_in = max(
        float(ld_h_in),
        float(width_in * _ld_min_height_over_width(max(2, int(ld_mat.shape[0])))),
    )
    if use_gene_panel:
        gene_h_in = _resolve_postgwas_gene_panel_height_in(
            float(ld_fontsize),
            width_in=width_in,
        )
        fig_ld = plt.figure(figsize=(width_in, gene_h_in + ld_h_in), dpi=300)
        gs = fig_ld.add_gridspec(2, 1, height_ratios=[gene_h_in, ld_h_in], hspace=0.03)
        ax_gene = fig_ld.add_subplot(gs[0, 0])
        ax_ld = fig_ld.add_subplot(gs[1, 0])
        _draw_gene_structure_axis(
            ax_gene,
            gene_track_df,
            arrow_color=gene_line_color,
            block_color=gene_block_color,
            line_width=0.9,
            arrow_step=1.0,
            gene_text_size=ld_fontsize,
        )
    else:
        fig_ld = plt.figure(figsize=(width_in, ld_h_in), dpi=300)
        ax_ld = fig_ld.add_subplot(111)
        ax_gene = None

    LDblock(ld_mat, ax=ax_ld, vmin=0, vmax=1, cmap=ld_cmap, rasterize_threshold=100)
    n_ld = max(2, int(ld_mat.shape[0]))
    ax_ld.set_xlim(0.5, float(n_ld) - 0.5)
    ax_ld.margins(x=0.0)
    if ld_overlay_text:
        ax_ld.text(
            n_ld / 2.0,
            -n_ld / 2.0,
            ld_overlay_text,
            ha="center",
            va="center",
            fontsize=ld_fontsize,
        )
    _draw_ld_bimrange_titles(
        ax_ld,
        ld_spans,
        enabled=show_ld_titles,
        font_size=ld_fontsize,
    )

    if ax_gene is not None:
        ax_gene.set_xlim(0.5, float(n_ld) - 0.5)

    fig_ld.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
    if args.ldblock_xspan is not None:
        fx0, fx1 = args.ldblock_xspan
        if ax_gene is not None:
            gx0, gy0, gw, gh = ax_gene.get_position().bounds
            new_gx0 = float(gx0) + float(gw) * float(fx0)
            new_gw = float(gw) * float(fx1 - fx0)
            ax_gene.set_position([new_gx0, gy0, new_gw, gh])
        cx0, cy0, cw, ch = ax_ld.get_position().bounds
        new_x0 = float(cx0) + float(cw) * float(fx0)
        new_w = float(cw) * float(fx1 - fx0)
        ax_ld.set_position([new_x0, cy0, new_w, ch])
        ax_ld.set_anchor("N")
        if ax_gene is not None:
            ax_gene.set_anchor("N")

    ld_path = os.path.join(args.out, f"{args.prefix}.ldblock.{args.format}")
    _save_figure_and_close(fig_ld, ld_path)
    log_success(
        logger,
        f"LD block plot saved to:\n  {format_path_for_display(ld_path)}\n",
    )
    log_success(
        logger,
        f"Visualizing LD block ...Finished [{format_elapsed(time.time() - t_ld)}]",
    )


def _run_postgwas_tasks_serial(
    files: list[str],
    args,
    logger: logging.Logger,
    *,
    total_start_ts: float,
    done_count: int = 0,
    file_to_idx: Optional[dict[str, int]] = None,
    skip_files: Optional[set[str]] = None,
    emit_final_success: bool = True,
) -> int:
    n_total = len(files)
    idx_map = (
        {str(k): int(v) for k, v in file_to_idx.items()}
        if file_to_idx is not None
        else {str(f): i for i, f in enumerate(files, start=1)}
    )
    skip = {str(x) for x in (skip_files or set())}
    setattr(args, "_postgwas_job_workers", 1)
    setattr(args, "_postgwas_worker_mute_stream", False)
    for file_path in files:
        if file_path in skip:
            continue
        idx = idx_map.get(file_path, 0)
        task_start_ts = time.monotonic()
        try:
            GWASplot(file_path, args, logger)
        except Exception:
            elapsed = format_elapsed(time.monotonic() - task_start_ts)
            print_failure(
                f"Task {idx}/{n_total}: "
                f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
            )
            raise
        done_count += 1
    if emit_final_success:
        total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
        print_success(
            f"Task {done_count}/{n_total} ...Finished [{total_elapsed}]",
            force_color=True,
        )
    return int(done_count)


def _run_postgwas_tasks(args, logger: logging.Logger) -> None:
    files = [str(f) for f in args.gwasfile]
    if len(files) == 0:
        return
    if len(files) == 1:
        setattr(args, "_postgwas_job_workers", 1)
        setattr(args, "_postgwas_worker_mute_stream", False)
        GWASplot(files[0], args, logger)
        return

    total_start_ts = time.monotonic()
    done_count = 0
    req_threads = int(args.thread)
    logical_workers = int(
        max(
            1,
            int(
                getattr(
                    args,
                    "_postgwas_outer_workers",
                    _resolve_postgwas_worker_count(req_threads, len(files)),
                )
            ),
        )
    )
    serial_reason = str(getattr(args, "_postgwas_serial_reason", "") or "").strip()

    if (len(files) > 1) and (os.name == "nt") and (not _allow_windows_postgwas_process_pool()):
        logger.warning(
            "Warning: Windows multi-process postgwas plotting is unstable in this "
            "environment; falling back to serial execution. "
            "Set JANUSX_POSTGWAS_WINDOWS_PROCESS_POOL=1 to force experimental "
            "process-pool mode."
        )
        _run_postgwas_tasks_serial(
            files,
            args,
            logger,
            total_start_ts=total_start_ts,
            done_count=done_count,
        )
        return

    if logical_workers <= 1:
        if serial_reason != "":
            logger.info(serial_reason)
        _run_postgwas_tasks_serial(
            files,
            args,
            logger,
            total_start_ts=total_start_ts,
            done_count=done_count,
        )
        return

    # In parallel mode, keep worker logs in file only to avoid spinner/pbar corruption.
    setattr(args, "_postgwas_worker_mute_stream", True)

    if rich_progress_available():
        n_total = len(files)
        basenames = [os.path.basename(f) for f in files]
        name_width = max((len(x) for x in basenames), default=0)
        idx_width = len(str(n_total))
        max_visible = min(5, n_total)
        n_workers = int(logical_workers)
        setattr(args, "_postgwas_job_workers", int(n_workers))
        file_to_idx = {f: i for i, f in enumerate(files, start=1)}
        progress = build_rich_progress(
            description_template="Task {task.fields[task_label]}: {task.fields[file_pad]}",
            show_bar=False,
            show_percentage=False,
            show_elapsed=False,
            show_remaining=False,
            finished_text=" ",
            transient=True,
        ) if should_animate_status("Loading merged post-GWAS tasks...") else None
        with (progress if progress is not None else nullcontext()):
            task_start_ts: dict[str, float] = {}
            task_map: dict[str, int] = {}
            future_map: dict[cf.Future[str], str] = {}
            completed_files: set[str] = set()
            pending_iter = iter(files)

            def _add_visible_task(file_path: str) -> None:
                if file_path in task_map:
                    return
                if len(task_map) >= max_visible:
                    return
                idx = file_to_idx[file_path]
                if progress is None:
                    return
                task_map[file_path] = progress.add_task(
                    description="",
                    total=None,
                    task_label=f"{idx:>{idx_width}}/{n_total}",
                    file_pad=os.path.basename(file_path).ljust(name_width),
                )

            def _submit_next(executor: cf.ProcessPoolExecutor) -> bool:
                try:
                    f_next = next(pending_iter)
                except StopIteration:
                    return False
                fut = executor.submit(_run_one_postgwas_task, f_next, args, logger)
                future_map[fut] = f_next
                task_start_ts[f_next] = time.monotonic()
                return True

            def _fill_visible_from_running() -> None:
                if len(task_map) >= max_visible:
                    return
                for running_file in list(future_map.values()):
                    if len(task_map) >= max_visible:
                        break
                    _add_visible_task(running_file)

            try:
                with _build_postgwas_process_pool(n_workers) as ex:
                    for _ in range(min(n_workers, n_total)):
                        if not _submit_next(ex):
                            break
                    _fill_visible_from_running()

                    while len(future_map) > 0:
                        done, _ = cf.wait(
                            list(future_map.keys()),
                            timeout=0.1,
                            return_when=cf.FIRST_COMPLETED,
                        )
                        if len(done) == 0:
                            _fill_visible_from_running()
                            continue
                        fut = min(
                            done,
                            key=lambda x: file_to_idx.get(
                                future_map.get(x, ""),
                                10**9,
                            ),
                        )
                        file_path = future_map.pop(fut)
                        tid = task_map.pop(file_path, None)
                        if tid is not None:
                            try:
                                progress.remove_task(tid)
                            except Exception:
                                pass
                        elapsed = format_elapsed(
                            time.monotonic()
                            - task_start_ts.get(file_path, time.monotonic())
                        )
                        try:
                            done_file = str(fut.result())
                        except BrokenProcessPool:
                            raise
                        except Exception:
                            idx = file_to_idx.get(file_path, 0)
                            print_failure(
                                f"Task {idx}/{n_total}: "
                                f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
                            )
                            raise
                        _ = done_file
                        completed_files.add(file_path)
                        done_count += 1
                        _submit_next(ex)
                        _fill_visible_from_running()
            except BrokenProcessPool:
                _log_postgwas_broken_pool_hint(
                    logger,
                    n_workers=n_workers,
                    req_threads=req_threads,
                    n_files=n_total,
                )
                print_warning(
                    f"PostGWAS worker pool broke after {done_count}/{n_total} tasks; "
                    "retrying remaining tasks serially."
                )
                for f, tid in list(task_map.items()):
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    task_map.pop(f, None)
                _run_postgwas_tasks_serial(
                    files,
                    args,
                    logger,
                    total_start_ts=total_start_ts,
                    done_count=done_count,
                    file_to_idx=file_to_idx,
                    skip_files=completed_files,
                )
                return
            except PermissionError as exc:
                print_warning(
                    "PostGWAS worker pool is unavailable in this environment; "
                    f"retrying serially ({exc})."
                )
                for f, tid in list(task_map.items()):
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    task_map.pop(f, None)
                _run_postgwas_tasks_serial(
                    files,
                    args,
                    logger,
                    total_start_ts=total_start_ts,
                    done_count=done_count,
                    file_to_idx=file_to_idx,
                    skip_files=completed_files,
                )
                return
            except Exception:
                for f, tid in list(task_map.items()):
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    task_map.pop(f, None)
                raise
        total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
        print_success(f"Task {done_count}/{n_total} ...Finished [{total_elapsed}]", force_color=True)
        return

    if _HAS_TQDM and stdout_is_tty() and should_animate_status("Loading merged post-GWAS tasks..."):
        setattr(args, "_postgwas_job_workers", int(logical_workers))
        pbar = tqdm(
            total=len(files),
            desc="PostGWAS tasks",
            unit="file",
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                       "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        task_start_ts: dict[str, float] = {}
        file_to_idx = {f: i for i, f in enumerate(files, start=1)}
        completed_files: set[str] = set()
        try:
            future_map: dict[cf.Future[str], str] = {}
            with _build_postgwas_process_pool(logical_workers) as ex:
                for file_path in files:
                    future_map[ex.submit(_run_one_postgwas_task, file_path, args, logger)] = file_path
                    task_start_ts[file_path] = time.monotonic()
                for fut in cf.as_completed(future_map):
                    file_path = future_map[fut]
                    elapsed = format_elapsed(
                        time.monotonic()
                        - task_start_ts.get(file_path, time.monotonic())
                    )
                    try:
                        done_file = str(fut.result())
                    except BrokenProcessPool:
                        raise
                    except Exception:
                        idx = file_to_idx.get(file_path, 0)
                        print_failure(
                            f"Task {idx}/{len(files)}: "
                            f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
                        )
                        raise
                    pbar.update(1)
                    pbar.set_postfix(file=os.path.basename(str(done_file)))
                    completed_files.add(file_path)
                    done_count += 1
        except BrokenProcessPool:
            _log_postgwas_broken_pool_hint(
                logger,
                n_workers=logical_workers,
                req_threads=req_threads,
                n_files=len(files),
            )
            print_warning(
                f"PostGWAS worker pool broke after {done_count}/{len(files)} tasks; "
                "retrying remaining tasks serially."
            )
            _run_postgwas_tasks_serial(
                files,
                args,
                logger,
                total_start_ts=total_start_ts,
                done_count=done_count,
                file_to_idx=file_to_idx,
                skip_files=completed_files,
            )
            return
        except PermissionError as exc:
            print_warning(
                "PostGWAS worker pool is unavailable in this environment; "
                f"retrying serially ({exc})."
            )
            _run_postgwas_tasks_serial(
                files,
                args,
                logger,
                total_start_ts=total_start_ts,
                done_count=done_count,
                file_to_idx=file_to_idx,
                skip_files=completed_files,
            )
            return
        finally:
            pbar.close()
        total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
        print_success(f"Task {done_count}/{len(files)} ...Finished [{total_elapsed}]", force_color=True)
        return

    setattr(args, "_postgwas_job_workers", int(logical_workers))
    task_start_ts: dict[str, float] = {}
    file_to_idx = {f: i for i, f in enumerate(files, start=1)}
    future_map: dict[cf.Future[str], str] = {}
    completed_files: set[str] = set()
    try:
        with _build_postgwas_process_pool(logical_workers) as ex:
            for file_path in files:
                future_map[ex.submit(_run_one_postgwas_task, file_path, args, logger)] = file_path
                task_start_ts[file_path] = time.monotonic()
            for fut in cf.as_completed(future_map):
                file_path = future_map[fut]
                elapsed = format_elapsed(
                    time.monotonic()
                    - task_start_ts.get(file_path, time.monotonic())
                )
                try:
                    _ = str(fut.result())
                except BrokenProcessPool:
                    raise
                except Exception:
                    idx = file_to_idx.get(file_path, 0)
                    print_failure(
                        f"Task {idx}/{len(files)}: "
                        f"{os.path.basename(file_path)} ...Failed [{elapsed}]"
                    )
                    raise
                completed_files.add(file_path)
                done_count += 1
    except BrokenProcessPool:
        _log_postgwas_broken_pool_hint(
            logger,
            n_workers=logical_workers,
            req_threads=req_threads,
            n_files=len(files),
        )
        print_warning(
            f"PostGWAS worker pool broke after {done_count}/{len(files)} tasks; "
            "retrying remaining tasks serially."
        )
        _run_postgwas_tasks_serial(
            files,
            args,
            logger,
            total_start_ts=total_start_ts,
            done_count=done_count,
            file_to_idx=file_to_idx,
            skip_files=completed_files,
        )
        return
    except PermissionError as exc:
        print_warning(
            "PostGWAS worker pool is unavailable in this environment; "
            f"retrying serially ({exc})."
        )
        _run_postgwas_tasks_serial(
            files,
            args,
            logger,
            total_start_ts=total_start_ts,
            done_count=done_count,
            file_to_idx=file_to_idx,
            skip_files=completed_files,
        )
        return
    total_elapsed = format_elapsed(time.monotonic() - total_start_ts)
    print_success(f"Task {done_count}/{len(files)} ...Finished [{total_elapsed}]", force_color=True)


def main(argv: Optional[list[str]] = None):
    warn_deprecated_alias_usage(("-threshold", "--threshold"), replacement="-thr/--thr")
    t_start = time.time()
    show_dev_help = _postgwas_dev_help_requested(argv)

    parser = CliArgumentParser(
        prog="jx postgwas",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx postgwas -gwasfile result.lmm.tsv -manh -qq",
            "jx postgwas -i a.tsv b.tsv -manh-merge -qq-merge -marker '1,o,x'",
            "jx postgwas -gwasfile result.lmm.tsv -a 50 -gff genes.gff3",
            "jx postgwas -bfile test/geno -bimrange 1:1-2 -ldblock-all -bed genes.bed",
            "jx postgwas -i result.lmm.tsv -bfile test/geno -bimrange 1:10-12 -finemap susie -o result -prefix locus",
        ]),
    )

    # ------------------------------------------------------------------
    # Required GWAS input
    # ------------------------------------------------------------------
    required_group = parser.add_argument_group("Required GWAS Input")
    required_group.add_argument(
        "-i", "-gwasfile", "--gwasfile", nargs="+", type=str, required=False, default=None,
        help=(
            "One or more GWAS result files (tab-delimited). "
            "Optional only when running LD block only with genotype input."
        ),
    )
    parser.add_argument(
        "-dev",
        "--dev",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )

    # ------------------------------------------------------------------
    # Fine-mapping
    # ------------------------------------------------------------------
    finemap_group = parser.add_argument_group("Fine-mapping")
    finemap_group.add_argument(
        "-finemap",
        "--finemap",
        choices=("susie",),
        default=None,
        help="Run SuSiE-RSS fine-mapping for the requested -bimrange loci.",
    )
    add_common_memory_arg(
        finemap_group,
        default=_POSTGWAS_FINEMAP_DEFAULT_MEMORY_GB,
        help_text=(
            "Dense signed-LD fine-mapping memory limit in GB "
            "(default: %(default)s)."
        ),
    )
    finemap_group.add_argument(
        "-finemap-L",
        "--finemap-L",
        dest="finemap_l",
        type=int,
        default=10,
        help=(
            "Maximum number of SuSiE single effects (default: %(default)s)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    finemap_group.add_argument(
        "-finemap-max-iter",
        "--finemap-max-iter",
        dest="finemap_max_iter",
        type=int,
        default=100,
        help=(
            "Maximum SuSiE iterations (default: %(default)s)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    finemap_group.add_argument(
        "-finemap-tol",
        "--finemap-tol",
        dest="finemap_tol",
        type=float,
        default=1e-4,
        help=(
            "SuSiE convergence tolerance (default: %(default)g)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )
    finemap_group.add_argument(
        "-finemap-purity-r2",
        "--finemap-purity-r2",
        dest="finemap_purity_r2",
        type=float,
        default=_POSTGWAS_FINEMAP_PURITY_R2_DEFAULT,
        help=(
            "Minimum within-CS pairwise tag LD r2 for retaining a SuSiE credible set "
            f"(default: %(default)g)."
            if show_dev_help
            else argparse.SUPPRESS
        ),
    )

    # ------------------------------------------------------------------
    # Manhattan Plot
    # ------------------------------------------------------------------
    manh_group = parser.add_argument_group("Manhattan Plot")
    manh_group.add_argument(
        "-manh", "--manh", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable Manhattan plotting with aspect ratio (width/height). "
            "Examples: --manh (default 2), --manh 2, --manh 3/2."
        ),
    )
    manh_group.add_argument(
        "-manh-merge", "--manh-merge", dest="manh_merge", type=str, nargs="?", const="2", default=None,
        help=(
            "Draw one merged Manhattan plot from the GWAS files passed by -i. "
            "Ratio parsing matches --manh."
        ),
    )
    manh_group.add_argument(
        "-palette", "--palette", dest="palette", type=str, default=None,
        help=(
            "Manhattan color palette and QQ scatter color palette "
            "(QQ confidence band always stays grey). "
            "Supports cmap names (e.g. tab10, tab20) or ';'-separated colors "
            "(e.g. #1f77b4;#ff7f0e or (215,123,254);(1,1,1)). "
            "A single color is also accepted. "
            "If omitted, use default black for QQ scatter and black/grey for Manhattan."
        ),
    )
    manh_group.add_argument(
        "-interval", "--interval", type=float, default=0.5,
        help=(
            "Chromosome-gap ratio for Manhattan x-axis spacing in [0,1]. "
            "Gap = ratio * median(chromosome length) / 10. "
            "Default: %(default)s."
        ),
    )
    manh_group.add_argument(
        "-circle", "--circle", dest="circle", type=float, nargs="*", default=None,
        help=(
            "Enable circular Manhattan/Circos plotting. "
            "Accepts 0, 1, or 2 values: "
            "--circle uses defaults; "
            "--circle <size_in> sets the square figure size; "
            "--circle <size_in> <track_ratio> also sets the Manhattan scatter-track share in [0,1]. "
            f"Defaults: size={_DEFAULT_CIRCLE_SIZE_IN:g} in, track_ratio={_DEFAULT_CIRCLE_TRACK_RATIO:g}."
        ),
    )
    manh_group.add_argument(
        "-circle-interval",
        "--circle-interval",
        dest="circle_interval",
        type=float,
        default=_DEFAULT_CIRCLE_INTERVAL,
        help=(
            "Relative gap between the Manhattan scatter ring and interaction links in [0,1]. "
            f"1 keeps the current farthest spacing; 0 moves links closest to the scatter ring. Default: {_DEFAULT_CIRCLE_INTERVAL:g}."
        ),
    )
    manh_group.add_argument(
        "-circle-lw",
        "--circle-lw",
        dest="circle_lw",
        type=float,
        default=_DEFAULT_CIRCLE_LW,
        help=(
            "Interaction curve line width for circular Manhattan. "
            f"Default: {_DEFAULT_CIRCLE_LW:g}."
        ),
    )
    circle_dir_group = manh_group.add_mutually_exclusive_group(required=False)
    circle_dir_group.add_argument(
        "-circle-in",
        "--circle-in",
        dest="circle_direction",
        action="store_const",
        const="in",
        help="Draw circular Manhattan values toward the center (0 at the outer ring edge).",
    )
    circle_dir_group.add_argument(
        "-circle-out",
        "--circle-out",
        dest="circle_direction",
        action="store_const",
        const="out",
        help="Draw circular Manhattan values away from the center (current default).",
    )
    manh_group.add_argument(
        "-interact",
        "--interact",
        dest="interact",
        nargs="+",
        default=None,
        help=(
            "Optional interaction source for circular Manhattan. "
            "Usage: --interact <file> ['snp;chrom;pos;pvalue;group1;group2;...']. "
            "If spec is omitted, defaults are GARFIELD-compatible: "
            "'snp;chrom;pos;pwald;|;&;*'."
        ),
    )

    # ------------------------------------------------------------------
    # QQ Plot
    # ------------------------------------------------------------------
    qq_group = parser.add_argument_group("Q-Q Plot")
    qq_group.add_argument(
        "-qq", "--qq", type=str, nargs="?", const="5/4", default=None,
        help=(
            "Enable QQ plotting in auto mode with aspect ratio (width/height). "
            "Examples: --qq (default 5/4), --qq 5/4, --qq 2."
        ),
    )
    qq_group.add_argument(
        "-qq-merge", "--qq-merge", dest="qq_merge", type=str, nargs="?", const="5/4", default=None,
        help=(
            "Draw one merged QQ plot from the GWAS files passed by -i. "
            "Ratio parsing matches --qq."
        ),
    )

    # ------------------------------------------------------------------
    # LDBlock Plot
    # ------------------------------------------------------------------
    ldblock_group = parser.add_argument_group("LDBlock Plot")
    geno_group = ldblock_group.add_mutually_exclusive_group(required=False)
    geno_group.add_argument(
        "-bfile", "--bfile", type=str, default=None,
        help="Genotype PLINK prefix (for fine-mapping, LD block, or LD clump).",
    )
    geno_group.add_argument(
        "-vcf", "--vcf", type=str, default=None,
        help="Genotype VCF/VCF.GZ file (for LD block/LD clump).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp", type=str, default=None,
        help="Genotype HMP/HMP.GZ file (for LD block/LD clump).",
    )
    geno_group.add_argument(
        "-file", "--file", dest="geno", type=str, default=None,
        help=(
            "Genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. For LD/LDclump, also requires real prefix.site or prefix.bim."
        ),
    )
    ldblock_mode_group = ldblock_group.add_mutually_exclusive_group(required=False)
    ldblock_mode_group.add_argument(
        "-ldblock", "--ldblock", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable LD block inverted triangle plotting with aspect ratio (width/height), "
            "using only threshold-passing SNPs. Requires --bimrange. "
            "You can also pass x-span in Manhattan-width fraction, e.g. 0.2:0.8 or 0.2-0.8 "
            "(then ratio defaults to 2). "
            "If only ratio is given, x-span defaults to 0:1. "
            "You may also pass a colormap/palette token here (e.g. tab10 or white;yellow;red), "
            "which keeps ratio=2 by default."
        ),
    )
    ldblock_mode_group.add_argument(
        "-ldblock-all", "--ldblock-all", dest="ldblock_all", type=str, nargs="?", const="2", default=None,
        help=(
            "Enable LD block inverted triangle plotting with aspect ratio (width/height), "
            "using all SNPs in selected bimrange. Requires --bimrange. "
            "You can also pass x-span in Manhattan-width fraction, e.g. 0.2:0.8 or 0.2-0.8 "
            "(then ratio defaults to 2). "
            "If only ratio is given, x-span defaults to 0:1. "
            "You may also pass a colormap/palette token here (e.g. tab10 or white;yellow;red), "
            "which keeps ratio=2 by default."
        ),
    )
    ldblock_group.add_argument(
        "-ldblock-palette", "--ldblock-palette",
        dest="ldblock_palette",
        type=str,
        default=None,
        help=(
            "LD block colormap only (independent from --palette). "
            "Supports cmap names (e.g. tab10/tab20) or color lists "
            "like 'white;yellow;red' or 'white,yellow,red'. "
            "If omitted, LD block keeps default greyscale."
        ),
    )

    # ------------------------------------------------------------------
    # Variant Annotation
    # ------------------------------------------------------------------
    anno_group = parser.add_argument_group("Variant Annotation")
    anno_group.add_argument(
        "-a", "--anno", nargs="?", const="on", default=None, metavar="EXT_KB",
        help=(
            "Enable significant-variant annotation table output. "
            "Requires an explicit --thr; otherwise annotation output is skipped. "
            "Requires one annotation source from --gff or --bed. "
            "Optionally pass EXT_KB to also append broadened annotation within that window."
        ),
    )
    anno_group.add_argument(
        "-LDclump", "--LDclump", dest="ldclump", nargs=2, default=None,
        metavar=("WINDOW", "R2"),
        help=(
            "Enable LD clumping for annotation output using threshold-passing SNPs only. "
            "Format: --LDclump <window> <r2>, e.g. --LDclump 500kb 0.8. "
            "Window supports kb/mb/bp (no unit defaults to kb). "
            "Requires genotype input via --bfile/--vcf/--hmp/--file."
        ),
    )

    # ------------------------------------------------------------------
    # Common Parameters
    # ------------------------------------------------------------------
    common_group = parser.add_argument_group("Common Parameters")
    common_group.add_argument(
        "-chr", "--chr", type=str, default="chrom",
        help="Column name for chromosome (default: %(default)s).",
    )
    common_group.add_argument(
        "-pos", "--pos", type=str, default="pos",
        help="Column name for base position (default: %(default)s).",
    )
    common_group.add_argument(
        "-pvalue", "--pvalue", type=str, default="pwald",
        help="Column name for p-value (default: %(default)s).",
    )
    common_group.add_argument(
        "--no-logtrans",
        dest="no_logtrans",
        action="store_true",
        help=(
            "Treat the selected p-value column as already being -log10(p); "
            "--thr is then used directly on the logP scale, and no automatic "
            "threshold line is drawn unless --thr is supplied."
        ),
    )
    common_group.add_argument(
        "-thr", "--thr", dest="thr", type=float, default=None,
        help=(
            "P-value threshold; if not set, no threshold line is drawn "
            "and --anno output is skipped. With --no-logtrans, "
            "an explicitly supplied value is interpreted directly as -log10(p)."
        ),
    )
    common_group.add_argument(
        "-threshold", "--threshold", dest="thr", type=float, default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    common_group.add_argument(
        "-bimrange", "--bimrange", type=str, action="append", default=None,
        help=(
            "Independent fine-mapping locus and shared plotting range filter for "
            "single/merged Manhattan, LD block, and Manhattan+LD/gene-track layouts. "
            "Format chr:start-end "
            "(also accepts chr:start:end). If start/end are integer-like and >6 digits, "
            "they are auto-treated as bp (with warning) and axis labels are shown in Mb. "
            "Can be specified multiple times. QQ/QQ-merge are disabled when this is set."
        ),
    )
    common_group.add_argument(
        "-ylim", "--ylim", nargs="+", type=str, default=None,
        help=(
            "Shared y-range control for Manhattan, merged Manhattan, QQ, and merged QQ "
            "as <max>, <min:max>, <min:>, <:max>, or <min> <max> "
            "(also accepts '-' as separator). "
            "Examples: --ylim 6, --ylim 0:6, --ylim 2:, --ylim :6, --ylim 2 10."
        ),
    )
    common_group.add_argument(
        "-scatter-size", "--scatter-size", nargs="+", type=str, default=None,
        help=(
            "Shared scatter marker size for Manhattan, merged Manhattan, QQ, and merged QQ "
            "Single/non-merge uses the first value only; merge mode cycles values by GWAS file. "
            f"Default: {_DEFAULT_SCATTER_SIZE:g}."
        ),
    )
    common_group.add_argument(
        "-alpha", "--alpha", nargs="+", type=str, default=None,
        help=(
            "Shared scatter alpha in [0,1] for Manhattan, merged Manhattan, QQ, and merged QQ. "
            "Single/non-merge uses the first value only; merge mode cycles values by GWAS file."
        ),
    )
    common_group.add_argument(
        "-marker", "--marker", type=str, default=None,
        help=(
            "Shared scatter marker control for single Manhattan/QQ and merged Manhattan/QQ. "
            "Single/non-merge uses the first marker only (default: o). "
            "Merge plotting cycles markers by input GWAS file; default cycle is 1,2,3,4,*,+,x. "
            "Example: --marker '1,o,x'."
        ),
    )
    common_group.add_argument(
        "-fontsize", "--fontsize", type=float, default=None,
        help=(
            "Unified plot font size. If omitted, postgwas uses a larger readable base size "
            "with ratio-aware auto scaling for Manhattan-style layouts."
        ),
    )
    common_group.add_argument(
        "-fontstyle", "--fontstyle", "-fontstype", "--fontstype",
        dest="fontstyle",
        type=str,
        default=None,
        help=(
            "Unified plot font family or font file path (.ttf/.otf/.ttc/.otc). "
            "Font family names support fuzzy matching, for example --fontstyle arial."
        ),
    )
    common_group.add_argument(
        "-full", "--full", "-fullscatter", "--fullscatter",
        dest="fullscatter",
        action="store_true",
        default=False,
        help=(
            "Disable scatter compression for single/merged Manhattan and single/merged QQ "
            "and draw all points (no fast optimization; no 0.5 cut)."
        ),
    )
    common_group.add_argument(
        "-highlight", "--highlight",
        dest="highlight",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    common_group.add_argument(
        "-highlight-color", "--highlight-color",
        dest="highlight_color",
        type=str,
        default="red",
        help=argparse.SUPPRESS,
    )
    common_group.add_argument(
        "-highlight-marker", "--highlight-marker",
        dest="highlight_marker",
        type=str,
        default="D",
        help=argparse.SUPPRESS,
    )
    _add_postgwas_annotation_source_args(common_group)
    common_group.add_argument(
        "-fmt", "--fmt", dest="format", type=str, default="png",
        help="Output figure format: pdf, png, svg, tif (default: %(default)s).",
    )
    add_common_out_arg(common_group, default=".", help_profile="plot_annotation")
    add_common_prefix_arg(
        common_group,
        default=None,
        help_text=(
            "Output prefix. For single-GWAS plotting, figures are saved as "
            "<prefix>.<input-stem>.* ; if omitted, use the input filename stem. "
            "The same prefix is also used for the run log stem."
        ),
    )
    add_common_thread_arg(
        common_group,
        default_threads=detect_effective_threads(),
        help_profile="default",
    )

    args = parser.parse_args(argv)
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)
    args.gwasfile = (
        [str(x) for x in list(args.gwasfile)]
        if args.gwasfile is not None
        else []
    )
    args.anno_file = _resolve_postgwas_annotation_file(args)
    args._postgwas_annotation_kind = _resolve_postgwas_annotation_kind(
        gff=getattr(args, "gff", None),
        bed=getattr(args, "bed", None),
        anno_file=args.anno_file,
    )

    out_dir, outprefix_base, out_stem = apply_output_prefix_compat(
        args,
        "JanusX",
        argv=argv,
        fallback_prefix="JanusX",
    )
    args._postgwas_plot_prefix = (
        str(out_stem)
        if bool(getattr(args, "_out_was_explicit", False) or getattr(args, "_prefix_was_explicit", False))
        else ""
    )
    os.makedirs(out_dir, mode=0o755, exist_ok=True)
    configure_genotype_cache_from_out(out_dir)
    log_path = f"{outprefix_base}.postGWAS.log"
    logger = setup_logging(log_path)
    args._postgwas_log_path = str(log_path)
    try:
        args.anno, args.annobroaden = _resolve_postgwas_anno_cli(getattr(args, "anno", None))
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    if args.anno and args.thr is None:
        logger.warning(
            "Warning: --anno requires an explicit --thr; annotation table output is skipped."
        )
        args.anno = False
        args.annobroaden = None
    if thread_capped:
        logger.warning(
            f"Warning: Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )
    apply_blas_thread_env(int(args.thread))
    # maybe_warn_non_openblas(
    #     logger=logger,
    #     strict=require_openblas_by_default(),
    # )

    # ------------------------------------------------------------------
    # Basic checks and configuration
    # ------------------------------------------------------------------
    args.format = str(args.format).lower()
    if args.format not in ["pdf", "png", "svg", "tif"]:
        logger.error(
            f"Unsupported figure format: {args.format} "
            "(choose from: pdf, png, svg, tif)"
        )
        raise SystemExit(1)
    if args.format == "pdf" and _postgwas_resolve_pdf_backend() is None:
        logger.warning(
            "Warning: Cairo PDF backend is unavailable because `pycairo` is not installed. "
            "PDF export will fall back to the default Matplotlib backend. "
            "If you need to edit the PDF in Adobe Illustrator, please run: "
            "`pip install pycairo`."
        )
    try:
        args.scatter_size_spec = _parse_scatter_size_spec(args.scatter_size)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    args.scatter_size = _resolve_single_series_value(
        args.scatter_size_spec,
        _DEFAULT_SCATTER_SIZE,
    )
    args._postgwas_single_scatter_size = float(args.scatter_size)
    try:
        args.alpha_spec = _parse_alpha_spec(args.alpha)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    args._postgwas_single_alpha = (
        float(args.alpha_spec[0])
        if args.alpha_spec is not None and len(args.alpha_spec) > 0
        else None
    )
    if args.scatter_size <= 0:
        logger.error("scatter-size must be > 0.")
        raise SystemExit(1)
    if args.fontsize is not None:
        try:
            args.fontsize = float(args.fontsize)
        except Exception:
            logger.error("fontsize must be a finite number > 0.")
            raise SystemExit(1)
        if (not np.isfinite(args.fontsize)) or float(args.fontsize) <= 0.0:
            logger.error("fontsize must be > 0.")
            raise SystemExit(1)
    args._postgwas_base_fontsize = (
        float(args.fontsize)
        if args.fontsize is not None
        else float(_POSTGWAS_DEFAULT_FONT_SIZE)
    )
    if args.fontstyle is not None:
        try:
            font_family, font_display, font_match_mode = _resolve_postgwas_fontstyle(
                args.fontstyle
            )
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
        args._postgwas_font_family = str(font_family)
        args._postgwas_font_display = str(font_display)
        args._postgwas_font_match_mode = str(font_match_mode)
        if font_match_mode == "file":
            logger.info(
                f"Loaded font file {format_path_for_display(font_display)} "
                f"as family '{font_family}'."
            )
        elif font_match_mode not in {"exact", "generic"}:
            logger.info(f"Resolved fontstyle '{args.fontstyle}' -> '{font_family}'.")
    else:
        args._postgwas_font_family = ""
        args._postgwas_font_display = "auto"
        args._postgwas_font_match_mode = "auto"
    try:
        args.interval = float(args.interval)
    except Exception:
        logger.error("interval must be a finite number in [0,1].")
        raise SystemExit(1)
    if (not np.isfinite(args.interval)) or (args.interval < 0.0) or (args.interval > 1.0):
        logger.error("interval must be in [0,1].")
        raise SystemExit(1)
    if args.circle is not None:
        circle_items = list(args.circle)
        if len(circle_items) > 2:
            logger.error("circle accepts at most two values: <size_in> <track_ratio>.")
            raise SystemExit(1)
        if len(circle_items) == 0:
            args.circle_size = float(_DEFAULT_CIRCLE_SIZE_IN)
            args.circle_track_ratio = float(_DEFAULT_CIRCLE_TRACK_RATIO)
        else:
            try:
                args.circle_size = float(circle_items[0])
            except Exception:
                logger.error("circle size must be a finite number > 0.")
                raise SystemExit(1)
            if (not np.isfinite(args.circle_size)) or float(args.circle_size) <= 0.0:
                logger.error("circle size must be > 0.")
                raise SystemExit(1)
            if len(circle_items) >= 2:
                try:
                    args.circle_track_ratio = float(circle_items[1])
                except Exception:
                    logger.error("circle track ratio must be a finite number in [0,1].")
                    raise SystemExit(1)
            else:
                args.circle_track_ratio = float(_DEFAULT_CIRCLE_TRACK_RATIO)
        if (
            (not np.isfinite(args.circle_track_ratio))
            or float(args.circle_track_ratio) < 0.0
            or float(args.circle_track_ratio) > 1.0
        ):
            logger.error("circle track ratio must be in [0,1].")
            raise SystemExit(1)
    else:
        args.circle_size = None
        args.circle_track_ratio = float(_DEFAULT_CIRCLE_TRACK_RATIO)
    try:
        args.circle_interval = float(args.circle_interval)
    except Exception:
        logger.error("circle-interval must be a finite number in [0,1].")
        raise SystemExit(1)
    if (not np.isfinite(args.circle_interval)) or float(args.circle_interval) < 0.0 or float(args.circle_interval) > 1.0:
        logger.error("circle-interval must be in [0,1].")
        raise SystemExit(1)
    try:
        args.circle_lw = float(args.circle_lw)
    except Exception:
        logger.error("circle-lw must be a finite number > 0.")
        raise SystemExit(1)
    if (not np.isfinite(args.circle_lw)) or float(args.circle_lw) <= 0.0:
        logger.error("circle-lw must be > 0.")
        raise SystemExit(1)
    args.circle_direction = str(getattr(args, "circle_direction", "out") or "out").strip().lower()
    if args.circle_direction not in {"in", "out"}:
        logger.error("circle direction must be either circle-in or circle-out.")
        raise SystemExit(1)
    if args.interact is None:
        args._circle_interact_path = None
        args._circle_interact_spec = None
    else:
        interact_items = list(args.interact)
        if len(interact_items) < 1 or len(interact_items) > 2:
            logger.error("interact expects 1 or 2 values: <file> ['snp;chrom;pos;pvalue;group1;group2;...']")
            raise SystemExit(1)
        args._circle_interact_path = str(interact_items[0]).strip()
        if args._circle_interact_path == "":
            logger.error("interact file path cannot be empty.")
            raise SystemExit(1)
        try:
            args._circle_interact_spec = _postgwas_parse_interact_spec(
                interact_items[1] if len(interact_items) >= 2 else None
            )
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    if bool(args.anno) and args.anno_file is None:
        logger.error("Variant annotation requires one annotation source from --gff or --bed.")
        raise SystemExit(1)
    try:
        if args.ylim is not None:
            args.ylim_min, args.ylim_max = _parse_ylim_spec(args.ylim)
            args.ylim = _format_ylim_spec_text(args.ylim, args.ylim_min, args.ylim_max)
        else:
            args.ylim_min, args.ylim_max = (None, None)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        args.palette_spec = _parse_palette_spec(args.palette)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    try:
        args.ldblock_palette_spec = _parse_palette_spec(args.ldblock_palette)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        args.manh_ratio = _parse_ratio(args.manh, "Manhattan") if args.manh is not None else None
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    if args.qq is None:
        args.qq_ratio = None
    else:
        try:
            args.qq_ratio = _parse_ratio(args.qq, "QQ")
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    try:
        args.manh_merge_ratio = (
            _parse_ratio(args.manh_merge, "Merged Manhattan")
            if args.manh_merge is not None
            else None
        )
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    if args.qq_merge is None:
        args.qq_merge_ratio = None
    else:
        try:
            args.qq_merge_ratio = _parse_ratio(args.qq_merge, "Merged QQ")
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    try:
        args.marker_spec = _parse_marker_spec(args.marker)
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)
    args._postgwas_single_marker = _resolve_single_marker(args.marker_spec)
    args._postgwas_merge_markers = []

    try:
        args.ldblock_ratio = None
        args.ldblock_xspan = None
        args.ldblock_mode = None
        if args.ldblock_all is not None:
            try:
                args.ldblock_ratio, args.ldblock_xspan = _parse_ldblock_spec(
                    args.ldblock_all, "LDBlock-all", logger
                )
            except ValueError as ratio_err:
                try:
                    pal_spec = _parse_palette_spec(args.ldblock_all)
                except ValueError:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                if pal_spec is None:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                args.ldblock_ratio, args.ldblock_xspan = 2.0, (0.0, 1.0)
                if args.ldblock_palette_spec is None:
                    args.ldblock_palette_spec = pal_spec
                    args.ldblock_palette = str(args.ldblock_all)
            args.ldblock_mode = "all"
        elif args.ldblock is not None:
            try:
                args.ldblock_ratio, args.ldblock_xspan = _parse_ldblock_spec(
                    args.ldblock, "LDBlock", logger
                )
            except ValueError as ratio_err:
                try:
                    pal_spec = _parse_palette_spec(args.ldblock)
                except ValueError:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                if pal_spec is None:
                    logger.error(str(ratio_err))
                    raise SystemExit(1)
                args.ldblock_ratio, args.ldblock_xspan = 2.0, (0.0, 1.0)
                if args.ldblock_palette_spec is None:
                    args.ldblock_palette_spec = pal_spec
                    args.ldblock_palette = str(args.ldblock)
            args.ldblock_mode = "threshold"
    except SystemExit:
        raise
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    try:
        if args.ldclump is not None:
            args.ldclump_window_bp, args.ldclump_r2 = _parse_ldclump_spec(args.ldclump)
        else:
            args.ldclump_window_bp = None
            args.ldclump_r2 = None
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    if args.bimrange is not None:
        try:
            parsed_bimrange_tuples = [
                _parse_bimrange(x, logger) for x in args.bimrange
            ]
            args.finemap_bimrange_tuples = list(parsed_bimrange_tuples)
            finemap_requested_for_ranges = (
                str(getattr(args, "finemap", "") or "").lower() == "susie"
            )
            non_finemap_range_work = bool(
                (args.manh_ratio is not None)
                or (args.manh_merge_ratio is not None)
                or (args.circle_size is not None)
                or bool(args.anno)
                or (args.ldblock_ratio is not None)
            )
            args.bimrange_tuples = sorted(
                parsed_bimrange_tuples,
                key=lambda x: (_chrom_sort_key(x[0]), int(x[1]), int(x[2])),
            )
            args.bimrange_tuples = _merge_overlapping_bimranges(
                args.bimrange_tuples,
                logger,
                warn_overlaps=(
                    (not finemap_requested_for_ranges) or non_finemap_range_work
                ),
            )
        except ValueError as e:
            logger.error(str(e))
            raise SystemExit(1)
    else:
        args.bimrange_tuples = None
        args.finemap_bimrange_tuples = None
    if args.bimrange_tuples is not None and (
        args.qq_ratio is not None or args.qq_merge_ratio is not None
    ):
        logger.info(
            "QQ is disabled when --bimrange is set."
        )
        args.qq_ratio = None
        args.qq_merge_ratio = None
    if args.ldblock_ratio is not None and args.bimrange_tuples is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all requires --bimrange; LD block and Manhattan+LD plotting are skipped."
        )
        args.ldblock_ratio = None
        args.ldblock_xspan = None
        args.ldblock_mode = None

    args.genofile = args.bfile or args.vcf or args.hmp or args.geno
    _validate_postgwas_finemap_args(args, parser)
    if args.ldblock_ratio is not None and args.genofile is None:
        logger.warning(
            "Warning: --ldblock/--ldblock-all enabled but no genotype file provided; zero-correlation LD block will be drawn."
        )
    if args.ldclump_window_bp is not None and args.genofile is None:
        logger.warning(
            "Warning: --LDclump enabled but no genotype file provided; LD clump is skipped."
        )
        args.ldclump_window_bp = None
        args.ldclump_r2 = None
    if args.ldclump_window_bp is not None and (not bool(args.anno)):
        logger.warning(
            "Warning: --LDclump only affects --anno output; --LDclump is ignored."
        )
        args.ldclump_window_bp = None
        args.ldclump_r2 = None
    merge_plot_requested = bool(
        (args.manh_merge_ratio is not None)
        or (args.qq_merge_ratio is not None)
    )
    single_ldblock_requested = bool(
        (args.ldblock_ratio is not None)
        and (len(args.gwasfile) > 0)
        and (
            (args.manh_ratio is not None)
            or (args.manh_merge_ratio is None)
        )
    )
    merge_ldblock_requested = bool(
        (args.ldblock_ratio is not None)
        and (args.manh_merge_ratio is not None)
        and (len(args.gwasfile) > 0)
    )
    single_plot_requested = bool(
        (args.manh_ratio is not None)
        or (args.qq_ratio is not None)
        or (args.circle_size is not None)
        or bool(args.anno)
        or bool(single_ldblock_requested)
    )
    args.merge_mode = bool(merge_plot_requested)
    args._postgwas_merge_requested = bool(merge_plot_requested)
    args._postgwas_single_requested = bool(single_plot_requested)
    args._postgwas_single_ldblock_requested = bool(single_ldblock_requested)
    args._postgwas_merge_ldblock_requested = bool(merge_ldblock_requested)
    args._postgwas_use_shared_gff = bool(
        _postgwas_annotation_is_gff(
            args.anno_file,
            annotation_kind=getattr(args, "_postgwas_annotation_kind", None),
        )
    )
    args._postgwas_serial_reason = ""
    if merge_plot_requested:
        args.merge_files = [str(x) for x in list(args.gwasfile)]
        if len(args.merge_files) == 0:
            logger.error(
                "Merged plotting requires at least one GWAS file from -i/--gwasfile."
            )
            raise SystemExit(1)
        args._merge_manh_ratio = args.manh_merge_ratio
        args._merge_qq_ratio = args.qq_merge_ratio
        if args._merge_manh_ratio is None and args._merge_qq_ratio is None:
            logger.info("Merge plotting detected; forcing merged Manhattan ratio to 2.")
            args._merge_manh_ratio = 2.0
        args._postgwas_merge_markers = _resolve_merge_markers(
            args.marker_spec,
            len(args.merge_files),
        )
        args._postgwas_merge_scatter_sizes = _resolve_merge_series_values(
            args.scatter_size_spec,
            len(args.merge_files),
            default=float(args._postgwas_single_scatter_size),
        )
        args._postgwas_merge_alphas = _resolve_merge_series_values(
            args.alpha_spec,
            len(args.merge_files),
            default=float(_DEFAULT_MERGE_ALPHA),
        )
    else:
        args.merge_files = []
        args._merge_manh_ratio = None
        args._merge_qq_ratio = None
        args._postgwas_merge_scatter_sizes = []
        args._postgwas_merge_alphas = []

    args._postgwas_outer_workers = int(
        _resolve_postgwas_worker_count(int(args.thread), len(args.gwasfile))
    ) if (single_plot_requested and len(args.gwasfile) > 1) else 1
    if (
        bool(args._postgwas_use_shared_gff)
        and bool(single_plot_requested)
        and len(args.gwasfile) > 1
    ):
        args._postgwas_outer_workers = 1
        args._postgwas_serial_reason = (
            "GFF/GFF3 annotation source detected across multiple GWAS files; "
            "using serial shared-GFF mode to avoid per-worker GFF duplication "
            "and reduce memory pressure."
        )

    args.ldblock_only_mode = bool(
        (len(args.gwasfile) == 0) and (args.ldblock_ratio is not None)
    )
    if len(args.gwasfile) == 0:
        if args.manh_ratio is not None:
            logger.warning(
                "Warning: --manh requires GWAS result file(s); it is ignored in LD-only mode."
            )
            args.manh_ratio = None
        if args.qq_ratio is not None:
            logger.warning(
                "Warning: --qq requires GWAS result file(s); it is ignored in LD-only mode."
            )
            args.qq_ratio = None
        if args.anno_file is not None:
            logger.info(
                "LD-only mode: annotation source will be used for gene-structure track above LD block."
            )
        if args.ldclump_window_bp is not None:
            logger.warning(
                "Warning: --LDclump requires GWAS result file(s); it is ignored in LD-only mode."
            )
            args.ldclump_window_bp = None
            args.ldclump_r2 = None
        if args.ldblock_ratio is None:
            logger.error(
                "No GWAS input file provided. "
                "Use --gwasfile, or run LD-only mode with --ldblock/--ldblock-all + genotype input."
            )
            raise SystemExit(1)
        single_plot_requested = False
        args._postgwas_single_requested = False

    if not hasattr(args, "fullscatter"):
        args.fullscatter = bool(getattr(args, "full", False))

    no_plot_or_anno = (
        (not merge_plot_requested)
        and (not single_plot_requested)
        and (not bool(getattr(args, "ldblock_only_mode", False)))
        and (not bool(getattr(args, "finemap_requested", False)))
    )
    args.disable_compression = bool(args.fullscatter or (args.bimrange_tuples is not None))

    # ------------------------------------------------------------------
    # Configuration summary
    # ------------------------------------------------------------------
    config_title = "JanusX - Post-GWAS"
    host_text = socket.gethostname()
    input_files = list(args.gwasfile or [])
    input_files_text = _format_input_files(input_files)
    if bool(getattr(args, "ldblock_only_mode", False)):
        mode_text = "ldblock-only"
    elif merge_plot_requested and single_plot_requested:
        mode_text = "single+merge"
    elif merge_plot_requested:
        mode_text = "merge"
    else:
        mode_text = "single-file" if len(input_files) <= 1 else "multi-file"
    threshold_text = str(args.thr if args.thr is not None else "no line (annotation output skipped)")
    finemap_requested = bool(getattr(args, "finemap_requested", False))
    combined_finemap_range_work = bool(
        finemap_requested and (merge_plot_requested or single_plot_requested)
    )
    bimrange_text = _format_bimrange_summary(
        args.bimrange_tuples
        if (not finemap_requested or combined_finemap_range_work)
        else args.finemap_bimrange_tuples
    )
    merge_map_rows = (
        [(str(i), str(file)) for i, file in enumerate(args.merge_files)]
        if merge_plot_requested
        else []
    )
    threads_text = format_requested_thread_usage(
        requested_threads=int(requested_threads),
        using_threads=int(args.thread),
        detected_threads=int(detected_threads),
    )

    base_rows: list[tuple[str, str]] = [
        ("Mode", mode_text),
        ("GWAS files", input_files_text),
        ("Chr|Pos|Pvalue", f"{args.chr}|{args.pos}|{args.pvalue}"),
        ("Genotype file", str(args.genofile) if args.genofile is not None else "NA"),
        ("Threshold", threshold_text),
        ("Bimrange", bimrange_text),
    ]
    if bool(getattr(args, "no_logtrans", False)):
        base_rows.insert(4, ("P-value scale", "precomputed -log10(p)"))
    if finemap_requested:
        if combined_finemap_range_work:
            base_rows.append(
                (
                    "Fine-map loci",
                    _format_bimrange_summary(args.finemap_bimrange_tuples),
                )
            )
        base_rows.append(
            (
                "Fine-mapping",
                "SuSiE "
                f"(L={int(args.finemap_l)}, max_iter={int(args.finemap_max_iter)}, "
                f"tol={float(args.finemap_tol):g}, "
                f"purity_r2>={float(args.finemap_purity_r2):g})",
            )
        )
        base_rows.append(("Fine-map memory", f"{float(args.memory):g} GB limit"))

    vis_rows: Optional[list[tuple[str, str]]] = None
    if (
        args.manh_ratio is not None
        or args.qq_ratio is not None
        or args.circle_size is not None
        or args._merge_manh_ratio is not None
        or args._merge_qq_ratio is not None
        or args.ldblock_ratio is not None
    ):
        font_size_text = (
            f"{float(args.fontsize):g} (manual)"
            if args.fontsize is not None
            else f"auto (base={float(args._postgwas_base_fontsize):g}, ratio-aware)"
        )
        font_family_text = (
            str(getattr(args, "_postgwas_font_display", "auto"))
            if str(getattr(args, "_postgwas_font_display", "auto")) != "auto"
            else "auto (matplotlib + CJK fallback)"
        )
        single_manh_pal_text = (
            "default (black/grey)" if args.palette_spec is None else str(args.palette)
        )
        single_qq_pal_text = (
            "default (black; band=grey)"
            if args.palette_spec is None
            else f"{args.palette} (band=grey)"
        )
        if args.palette_spec is None:
            merge_default_cmap = "tab10" if len(args.merge_files) <= 10 else "tab20"
            merge_manh_pal_text = f"default ({merge_default_cmap})"
            merge_qq_pal_text = f"default ({merge_default_cmap}; band=grey)"
        else:
            merge_manh_pal_text = str(args.palette)
            merge_qq_pal_text = f"{args.palette} (band=grey)"
        if args.disable_compression:
            if args.fullscatter and args.bimrange_tuples is not None:
                comp_text = "off (--full, auto for --bimrange)"
            elif args.fullscatter:
                comp_text = "off (--full)"
            else:
                comp_text = "off (auto for --bimrange)"
        else:
            comp_text = "on"
        single_alpha_text = (
            f"{float(args._postgwas_single_alpha):g}"
            if getattr(args, "_postgwas_single_alpha", None) is not None
            else "default(auto)"
        )
        merge_size_text = (
            _format_float_series([float(x) for x in list(getattr(args, "_postgwas_merge_scatter_sizes", []))])
            if len(getattr(args, "_postgwas_merge_scatter_sizes", [])) > 0
            else f"{float(args.scatter_size):g}"
        )
        merge_alpha_text = (
            _format_float_series([float(x) for x in list(getattr(args, "_postgwas_merge_alphas", []))])
            if len(getattr(args, "_postgwas_merge_alphas", [])) > 0
            else f"{float(_DEFAULT_MERGE_ALPHA):g}"
        )
        vis_rows = [
            ("Format", str(args.format)),
            ("Font", f"size={font_size_text}, family={font_family_text}"),
        ]
        if args.manh_ratio is not None:
            vis_rows.append(
                (
                    "Single Manhattan",
                    f"ratio={args.manh_ratio}, palette={single_manh_pal_text}, "
                    f"interval={args.interval:g}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}, "
                    f"compression={comp_text}",
                )
            )
        if args.circle_size is not None:
            circle_desc = (
                f"size={float(args.circle_size):g}in, links use p<=thr, track=scatter, "
                f"track_ratio={float(args.circle_track_ratio):g}, "
                f"gap={float(args.circle_interval):g}, lw={float(args.circle_lw):g}, "
                f"dir={str(args.circle_direction)}"
            )
            if getattr(args, "_circle_interact_path", None) is not None:
                interact_file = os.path.basename(str(args._circle_interact_path))
                interact_spec = getattr(args, "_circle_interact_spec", None) or {}
                circle_desc += (
                    f", interact={interact_file}"
                    f" [{interact_spec.get('snp_col', 'snp')};"
                    f"{interact_spec.get('chr_col', 'chrom')};"
                    f"{interact_spec.get('pos_col', 'pos')};"
                    f"{interact_spec.get('p_col', 'pwald')}]"
                )
            vis_rows.append(
                (
                    "Circle Manhattan",
                    circle_desc,
                )
            )
        if args.qq_ratio is not None:
            vis_rows.append(
                (
                    "Single QQ",
                    f"ratio={args.qq_ratio}, palette={single_qq_pal_text}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}",
                )
            )
        if args.manh_ratio is not None or args.qq_ratio is not None:
            vis_rows.append(
                (
                    "Single Scatter",
                    f"size={float(args.scatter_size):g}, alpha={single_alpha_text}, "
                    f"marker={args._postgwas_single_marker}",
                )
            )
        if args._merge_manh_ratio is not None:
            vis_rows.append(
                (
                    "Merged Manhattan",
                    f"ratio={args._merge_manh_ratio}, palette={merge_manh_pal_text}, "
                    f"interval={args.interval:g}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}",
                )
            )
        if args._merge_qq_ratio is not None:
            vis_rows.append(
                (
                    "Merged QQ",
                    f"ratio={args._merge_qq_ratio}, palette={merge_qq_pal_text}, "
                    f"ylim={args.ylim if args.ylim is not None else 'auto'}",
                )
            )
        if merge_plot_requested:
            vis_rows.append(
                (
                    "Merged Scatter",
                    f"sizes={merge_size_text}, alphas={merge_alpha_text}, "
                    f"markers={','.join(args._postgwas_merge_markers)}",
                )
            )
        if args.ldblock_ratio is None:
            vis_rows.append(("LDBlock", "off"))
        else:
            ld_mode_text = "all SNPs" if args.ldblock_mode == "all" else "threshold SNPs"
            ld_xspan_text = (
                "full width (default)"
                if args.ldblock_xspan is None
                else f"{args.ldblock_xspan[0]:g}-{args.ldblock_xspan[1]:g} (fraction of Manhattan width)"
            )
            ld_pal_text = (
                "default (greys)"
                if args.ldblock_palette_spec is None
                else str(args.ldblock_palette)
            )
            vis_rows.append(
                (
                    "LDBlock",
                    f"ratio={args.ldblock_ratio}, mode={ld_mode_text}, "
                    f"x-span={ld_xspan_text}, palette={ld_pal_text}",
                ),
            )
    anno_rows: Optional[list[tuple[str, str]]] = None
    if bool(args.anno) or args.anno_file is not None:
        anno_rows = [
            ("Variant annotation", "on" if bool(args.anno) else "off"),
            ("Annotation source", str(args.anno_file) if args.anno_file is not None else "NA"),
            ("Window (kb)", str(args.annobroaden)),
        ]
        if (not bool(args.anno)) or args.ldclump_window_bp is None:
            anno_rows.append(("LD clump", "off"))
        else:
            anno_rows.append(
                (
                    "LD clump",
                    f"on ({args.ldclump_window_bp / 1000.0:g} kb, r2>={args.ldclump_r2:g})",
                )
            )

    sections: list[tuple[str, list[tuple[str, object]]]] = [("General", base_rows)]
    if len(merge_map_rows) > 0:
        sections.append(("Merge files", merge_map_rows))
    if vis_rows is None:
        sections.append(("Visualization", [("Status", "disabled")]))
    else:
        sections.append(("Visualization", vis_rows))
    if anno_rows is not None:
        sections.append(("Annotation", anno_rows))
    emit_cli_configuration(
        logger,
        app_title=config_title,
        config_title="POST-GWAS CONFIG",
        host=host_text,
        sections=sections,
        footer_rows=[
            ("Threads", threads_text),
            ("PostGWAS workers", int(getattr(args, "_postgwas_outer_workers", 1))),
        ],
        emit_to_stdout=True,
        line_max_chars=_CONFIG_LINE_MAX_CHARS,
        overflow_mark=_CONFIG_OVERFLOW_MARK,
    )
    _emit_info_to_file_handlers(logger, "")
    _emit_info_to_file_handlers(logger, "[ Command ]")
    _emit_info_to_file_handlers(logger, f"  {_postgwas_invocation_command(argv)}")
    if no_plot_or_anno:
        logger.warning(
            "Warning: No --manh/--circle/--qq/--manh-merge/--qq-merge/--ldblock/--ldblock-all/--anno provided. Nothing will be plotted or annotated."
        )

    check_gwas_files = list(args.gwasfile or [])
    checks: list[bool] = [
        ensure_file_exists(logger, f, "GWAS result file") for f in check_gwas_files
    ]
    if args._circle_interact_path is not None:
        checks.append(
            ensure_file_exists(
                logger,
                args._circle_interact_path,
                "Interaction file",
            )
        )
    if args.anno_file is not None:
        checks.append(ensure_file_exists(logger, args.anno_file, "Annotation file"))
    if args.vcf:
        checks.append(ensure_file_exists(logger, args.vcf, "Genotype VCF file"))
    if args.hmp:
        checks.append(ensure_file_exists(logger, args.hmp, "Genotype HMP file"))
    if args.geno:
        checks.append(ensure_file_input_exists(logger, args.geno, "Genotype FILE input"))
        if args.ldblock_ratio is not None or args.ldclump_window_bp is not None:
            checks.append(
                ensure_file_input_site_metadata_exists(
                    logger,
                    args.geno,
                    "Genotype FILE site metadata for LD/LDclump",
                )
            )
    mixed_model_finemap = (
        bool(getattr(args, "finemap_requested", False))
        and len(check_gwas_files) == 1
        and _postgwas_finemap_result_model(check_gwas_files[0]) is not None
    )
    if args.bfile and not mixed_model_finemap:
        checks.append(ensure_plink_prefix_exists(logger, args.bfile, "Genotype PLINK prefix"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    if bool(getattr(args, "finemap_requested", False)):
        try:
            args.finemap_output = _run_postgwas_susie_finemap(args, logger)
        except Exception as exc:
            logger.error("SuSiE fine-mapping failed: %s", exc)
            raise SystemExit(1) from exc

    if (
        bool(getattr(args, "_postgwas_use_shared_gff", False))
        and args.anno_file is not None
        and int(getattr(args, "_postgwas_outer_workers", 1)) <= 1
        and (
            bool(args.anno)
            or bool(getattr(args, "ldblock_only_mode", False))
            or (args.ldblock_ratio is not None)
        )
    ):
        preloaded = _postgwas_get_shared_gff_rust_index(args.anno_file)
        if preloaded is None:
            _postgwas_get_shared_gff_query(args.anno_file)
        logger.info(
            "Preloaded shared GFF annotation index: "
            f"{format_path_for_display(args.anno_file)}"
        )

    # ------------------------------------------------------------------
    # Parallel processing of all input files
    # ------------------------------------------------------------------
    if bool(getattr(args, "ldblock_only_mode", False)):
        _run_postgwas_ldblock_only(args, logger)
    else:
        if merge_plot_requested:
            _run_postgwas_merge_manhattan(args, logger)
        if single_plot_requested:
            _run_postgwas_tasks(args, logger)

    # ------------------------------------------------------------------
    # Final logging
    # ------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    log_success(logger, endinfo)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
