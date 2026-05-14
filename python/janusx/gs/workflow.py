# -*- coding: utf-8 -*-
"""
JanusX: Genomic Selection Command-Line Interface

Supported models
----------------
  - GBLUP  : Genomic Best Linear Unbiased Prediction.
             Kernels: additive(a), dominance(d), additive+dominance(ad)
  - rrBLUP : Ridge regression BLUP (rrBLUP, kinship = None)
  - BayesA : Bayesian marker effect model (via pyBLUP.bayes)
  - BayesB : Bayesian variable selection model (via pyBLUP.bayes)
  - BayesCpi : Bayesian variable selection model with shared variance (via pyBLUP.bayes)
  - RF     : Random forest regression with GS-oriented inner tuning
  - ET     : Extra-trees regression with GS-oriented inner tuning
  - GBDT   : Histogram gradient boosting regression with inner tuning
  - XGB    : XGBoost regression with compact inner tuning
  - SVM    : RBF-support vector regression with compact inner tuning
  - ENET   : ElasticNet regression with compact inner tuning

Genotype input formats
----------------------
  - VCF   : .vcf or .vcf.gz (using gfreader.vcfreader)
  - PLINK : Binary PLINK (.bed/.bim/.fam) via prefix (using gfreader.breader)

Phenotype input format
----------------------
  - Tab/comma/whitespace-delimited text file
  - First column: sample IDs
  - Remaining columns: phenotype traits
  - Duplicated IDs will be averaged.

Cross-validation
----------------
  - 5-fold cross-validation is performed within the training population for each model.
  - For each method, the fold with the highest R^2 on the validation set is reported
    and visualized.

Genomic selection workflow
--------------------------
  1. Load genotypes and phenotypes.
  2. Filter SNPs by MAF/missing rate thresholds (default 0.02/0.05) and
     mean-impute missing genotypes during Rust `gfreader` loading.
  3. For each phenotype column:
       - Split individuals into training (non-missing phenotype) and test sets.
       - Run 5-fold CV on the training set for each selected model.
       - Report Pearson, Spearman, and R2 per fold.
       - Use the best fold for diagnostic plotting.
       - Refit model on full training set and predict the test set.
  4. Write prediction results to {prefix}.{trait}.gs.tsv.

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import logging
import typing
import os
import time
import json
import socket
import argparse
import sys
import re
import math
import gc
import glob
import pickle
import threading
import subprocess
import shutil
import hashlib
from dataclasses import dataclass
from collections import OrderedDict
import multiprocessing as mp
import queue as queue_mod
import concurrent.futures as cf

# ----------------------------------------------------------------------
# Matplotlib backend configuration (non-interactive, server-safe)
# ----------------------------------------------------------------------
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr, spearmanr
try:
    from sklearn.model_selection import KFold
except Exception:
    KFold = None  # type: ignore[assignment]
from janusx.bioplotkit.sci_set import color_set
from janusx.bioplotkit import gsplot
from janusx._optional_deps import format_missing_dependency_message
from janusx.gfreader import (
    inspect_genotype_file,
    load_genotype_chunks,
    load_bed_2bit_packed,
    prepare_cli_input_cache,
)
from janusx.pyBLUP.kfold import kfold
from janusx.pyBLUP.mlm import BLUP as MLMBLUP
from janusx.pyBLUP.QK2 import GRM as _QK2_GRM
from janusx.pyBLUP.bayes import BAYES
from janusx.pyBLUP.ml import (
    MLGS,
    _HAS_SKLEARN,
    _SKLEARN_IMPORT_ERROR,
    _HAS_XGBOOST,
    _XGBOOST_IMPORT_ERROR,
)
try:
    from janusx.pyBLUP.blup import BLUP as KernelBLUP, Gmatrix
    _HAS_ADBLUP_PY = True
    _ADBLUP_PY_IMPORT_ERROR: Exception | None = None
except Exception as _adblup_py_exc:
    KernelBLUP = None  # type: ignore[assignment]
    Gmatrix = None  # type: ignore[assignment]
    _HAS_ADBLUP_PY = False
    _ADBLUP_PY_IMPORT_ERROR = _adblup_py_exc
from janusx.script._common.log import setup_logging
from janusx.script._common.config_render import emit_cli_configuration
from janusx.script._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from janusx.script._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from janusx.script._common.progress import ProgressAdapter, build_rich_progress, rich_progress_available
from janusx.script._common.status import (
    CliStatus,
    failure_symbol,
    log_success,
    print_success,
    print_failure,
    format_elapsed,
    stdout_is_tty,
    success_symbol,
)
from janusx.script._common.genocache import configure_genotype_cache_from_out
from janusx.script._common.colspec import parse_zero_based_index_specs
from janusx.script._common.threads import (
    detect_rust_blas_backend,
    get_rust_blas_threads,
    maybe_warn_non_openblas,
    require_openblas_by_default,
    runtime_thread_stage,
)
from janusx.script._common.packedctx import prepare_packed_ctx_from_plink
from janusx.gs import output as gs_output

try:
    from janusx import janusx as _jxrs
except Exception:
    _jxrs = None

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None  # type: ignore[assignment]
    _HAS_TQDM = False


# ======================================================================
# Core API for single-trait genomic prediction
# ======================================================================

_ML_METHOD_MAP: dict[str, str] = {
    "RF": "rf",
    "ET": "et",
    "GBDT": "gbdt",
    "XGB": "xgb",
    "SVM": "svm",
    "ENET": "enet",
}
_HASH_KERNEL_QC_SAMPLE_N = 2000
_HASH_KERNEL_QC_MIN_N = 64
_RRBLUP_AUTO_PCG_MIN_N = 10_000
_RRBLUP_LAMBDA_SUBSAMPLE_MIN_N = 2_000
_RRBLUP_LAMBDA_SUBSAMPLE_MAX_N = 5_000
_RRBLUP_LAMBDA_SUBSAMPLE_REPEATS = 9
_RUST_GBLUP_BACKEND_WARN_LOCK = threading.Lock()
_RUST_GBLUP_BACKEND_WARNED: set[str] = set()

_GBLUP_METHOD_ADD = "GBLUP"
_GBLUP_METHOD_DOM = "GBLUP_D"
_GBLUP_METHOD_AD = "GBLUP_AD"
_GBLUP_METHOD_SET = {_GBLUP_METHOD_ADD, _GBLUP_METHOD_DOM, _GBLUP_METHOD_AD}


def _default_rrblup_he_thread_policy() -> str:
    # Current rrBLUP-HE benchmarks favor BLAS serial + Rayon outer parallelism.
    return "rayon_parallel_blas_serial"


_RRBLUP_HE_THREAD_POLICY_DEFAULT = _default_rrblup_he_thread_policy()


def _default_rrblup_pcg_thread_policy() -> str:
    # Current rrBLUP-PCG benchmarks favor BLAS serial + Rayon outer parallelism.
    return "rayon_parallel_blas_serial"


_RRBLUP_PCG_THREAD_POLICY_DEFAULT = _default_rrblup_pcg_thread_policy()


def _rrblup_vc_method_display(name: object) -> str:
    txt = str(name).strip()
    if txt == "":
        return ""
    norm = re.sub(r"[^a-z0-9]+", "", txt.lower())
    if norm == "he":
        return "HE"
    if norm == "grmreml":
        return "GRMreml"
    if norm == "fastreml":
        return "FaSTreml"
    return txt


def _format_debug_bytes(n_bytes: object) -> str:
    try:
        value = int(n_bytes)
    except Exception:
        return "NA"
    if value < 0:
        return "NA"
    if value >= 1024 ** 3:
        return f"{value / (1024 ** 3):.2f} GiB"
    if value >= 1024 ** 2:
        return f"{value / (1024 ** 2):.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def _get_process_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _debug_malloc_trim_enabled() -> bool:
    raw = str(os.getenv("JX_GS_DEBUG_MALLOC_TRIM", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on", "y"}


def _maybe_linux_malloc_trim() -> tuple[bool, int | None, int | None, str]:
    if not _debug_malloc_trim_enabled():
        return False, None, None, "disabled"
    before = _get_process_rss_bytes()
    if not sys.platform.startswith("linux"):
        return False, before, before, "unsupported_platform"
    try:
        gc.collect()
        import ctypes
        import ctypes.util

        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        libc = ctypes.CDLL(libc_name)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is None:
            return False, before, _get_process_rss_bytes(), "malloc_trim_unavailable"
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        rc = int(malloc_trim(0))
        after = _get_process_rss_bytes()
        return True, before, after, f"rc={rc}"
    except Exception as ex:
        return False, before, _get_process_rss_bytes(), f"error={type(ex).__name__}:{ex}"


def _emit_debug_malloc_trim(prefix: str) -> None:
    trimmed, rss_before, rss_after, note = _maybe_linux_malloc_trim()
    if (not trimmed) and (note == "disabled"):
        return
    delta_txt = "NA"
    if (rss_before is not None) and (rss_after is not None):
        delta_txt = _format_debug_bytes(int(rss_after) - int(rss_before))
    print(
        (
            f"{prefix} malloc_trim "
            f"{note} "
            f"rss_before_trim={_format_debug_bytes(rss_before)} "
            f"rss_after_trim={_format_debug_bytes(rss_after)} "
            f"delta={delta_txt}"
        ),
        flush=True,
    )


def _emit_packed_load_debug(
    packed_ctx: dict[str, typing.Any] | None,
    *,
    label: str = "",
    enabled: bool = False,
) -> None:
    if (not bool(enabled)) and (not _GS_DEBUG_STAGE) and (not _debug_malloc_trim_enabled()):
        return
    if packed_ctx is None:
        return
    try:
        packed = np.asarray(packed_ctx["packed"], dtype=np.uint8)
        if packed.ndim != 2:
            print(
                f"[GS-DEBUG] Packed load {label or 'main'} invalid packed ndim={packed.ndim}",
                flush=True,
            )
            return
        packed_rows = int(packed.shape[0])
        bytes_per_snp = int(packed.shape[1])
        active_rows = int(
            packed_ctx.get(
                "n_active_sites",
                np.asarray(
                    packed_ctx.get("active_row_idx", np.arange(packed_rows, dtype=np.int64)),
                    dtype=np.int64,
                ).reshape(-1).shape[0],
            )
        )
        filter_mode = str(packed_ctx.get("packed_filter_mode", "compact")).strip() or "compact"
        site_keep_raw = packed_ctx.get("site_keep", None)
        source_rows = packed_rows
        if site_keep_raw is not None:
            try:
                source_rows = int(np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1).shape[0])
            except Exception:
                source_rows = packed_rows
        source_rows = int(max(packed_rows, source_rows, active_rows))
        packed_bytes = int(packed.nbytes)
        source_bed_payload_bytes = int(source_rows * bytes_per_snp)
        dropped_rows = int(max(0, source_rows - active_rows))
        label_txt = str(label).strip()
        if label_txt != "":
            label_txt = f" {label_txt}"
        print(
            (
                f"[GS-DEBUG] Packed load{label_txt} "
                f"rows={packed_rows} "
                f"active_rows={active_rows} "
                f"source_rows={source_rows} "
                f"dropped={dropped_rows} "
                f"mode={filter_mode} "
                f"bytes_per_snp={bytes_per_snp} "
                f"packed={_format_debug_bytes(packed_bytes)} "
                f"source_bed_payload={_format_debug_bytes(source_bed_payload_bytes)} "
                f"transient_peak_est={_format_debug_bytes(source_bed_payload_bytes + packed_bytes)} "
                f"rss_after_load={_format_debug_bytes(_get_process_rss_bytes())}"
            ),
            flush=True,
        )
        _emit_debug_malloc_trim(f"[GS-DEBUG] Packed load{label_txt}")
    except Exception as ex:
        print(
            f"[GS-DEBUG] Packed load {label or 'main'} debug_failed={type(ex).__name__}:{ex}",
            flush=True,
        )


def _packed_ctx_active_rows(packed_ctx: dict[str, typing.Any]) -> int:
    try:
        return int(packed_ctx.get("n_active_sites", 0)) or int(
            np.asarray(
                packed_ctx.get(
                    "active_row_idx",
                    np.arange(int(np.asarray(packed_ctx["packed"]).shape[0]), dtype=np.int64),
                ),
                dtype=np.int64,
            ).reshape(-1).shape[0]
        )
    except Exception:
        return int(np.asarray(packed_ctx["packed"]).shape[0])


def _packed_ctx_active_row_idx(packed_ctx: dict[str, typing.Any]) -> np.ndarray:
    raw = packed_ctx.get("active_row_idx", None)
    if raw is not None:
        arr = np.ascontiguousarray(np.asarray(raw, dtype=np.int64).reshape(-1), dtype=np.int64)
        if arr.size > 0:
            return arr
    site_keep_raw = packed_ctx.get("site_keep", None)
    if site_keep_raw is not None:
        site_keep = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        return np.ascontiguousarray(
            np.flatnonzero(site_keep).astype(np.int64, copy=False),
            dtype=np.int64,
        )
    m = _packed_ctx_active_rows(packed_ctx)
    return np.ascontiguousarray(np.arange(m, dtype=np.int64), dtype=np.int64)


def _packed_ctx_is_lazy_full(packed_ctx: dict[str, typing.Any]) -> bool:
    return str(packed_ctx.get("packed_filter_mode", "")).strip().lower() == "lazy_full"


def _normalize_he_thread_policy_name(
    raw: object,
    *,
    default: str = _RRBLUP_HE_THREAD_POLICY_DEFAULT,
    allow_compare: bool = False,
) -> str:
    txt = str(raw).strip().lower()
    if txt == "":
        txt = str(default).strip().lower()
    norm = re.sub(r"[^a-z0-9]+", "", txt)
    if norm in {
        "",
        "default",
        "auto",
        "rayon",
        "rayonparallelblasserial",
        "rayononly",
    }:
        return "rayon_parallel_blas_serial"
    if norm in {"blas", "blasparallelrayonserial", "blasonly"}:
        return "blas_parallel_rayon_serial"
    if norm in {"split", "splithalf", "halfhalf", "balanced"}:
        return "split_half"
    if norm in {"serial", "single", "singlethread", "one"}:
        return "serial"
    if norm in {"compare", "all", "sweep"}:
        return "compare" if allow_compare else str(default)
    raise ValueError(
        "Unsupported HE thread policy "
        f"{raw!r}. Use one of: rayon_parallel_blas_serial, "
        "blas_parallel_rayon_serial, split_half."
    )


def _resolve_he_thread_policy_spec(policy_name: object, total_threads: int) -> dict[str, typing.Any]:
    total = max(1, int(total_threads))
    policy = _normalize_he_thread_policy_name(policy_name, allow_compare=False)
    if (policy == "serial") or (total <= 1):
        blas_threads = 1
        rayon_threads = 1
        he_threads = 1
    elif policy == "rayon_parallel_blas_serial":
        blas_threads = 1
        rayon_threads = total
        he_threads = total
    elif policy == "blas_parallel_rayon_serial":
        blas_threads = total
        rayon_threads = 1
        he_threads = 1
    elif policy == "split_half":
        blas_threads = max(1, total // 2)
        rayon_threads = max(1, total - blas_threads)
        he_threads = rayon_threads
    else:
        raise ValueError(f"Unsupported HE thread policy {policy_name!r}")
    return {
        "policy": str(policy),
        "target_threads": int(total),
        "blas_threads": int(blas_threads),
        "rayon_threads": int(rayon_threads),
        "he_threads": int(he_threads),
        "label": (
            f"{policy} "
            f"(blas={int(blas_threads)}, rayon={int(rayon_threads)}, he={int(he_threads)})"
        ),
    }


def _resolve_pcg_thread_policy_spec(policy_name: object, total_threads: int) -> dict[str, typing.Any]:
    spec = dict(_resolve_he_thread_policy_spec(policy_name, total_threads))
    pcg_threads = int(spec.get("he_threads", 1))
    spec["pcg_threads"] = pcg_threads
    spec["label"] = (
        f"{str(spec.get('policy', '')).strip()} "
        f"(blas={int(spec.get('blas_threads', 1))}, "
        f"rayon={int(spec.get('rayon_threads', 1))}, "
        f"pcg={pcg_threads})"
    )
    return spec


def _warn_rust_gblup_backend_fallback_once(backend: str, allowed: set[str]) -> None:
    b = str(backend).strip().lower() or "unknown"
    with _RUST_GBLUP_BACKEND_WARN_LOCK:
        if b in _RUST_GBLUP_BACKEND_WARNED:
            return
        _RUST_GBLUP_BACKEND_WARNED.add(b)
    allowed_txt = ",".join(sorted(str(x).strip().lower() for x in set(allowed) if str(x).strip() != ""))
    if allowed_txt == "":
        allowed_txt = "none"
    logging.getLogger(__name__).warning(
        "Warning: Packed Rust GBLUP detected rust BLAS backend='%s' (allowed=%s on %s). "
        "Falling back to compatibility GS path; this may run slower.",
        b,
        allowed_txt,
        sys.platform,
    )


def _rust_gblup_allowed_blas_backends() -> set[str]:
    # macOS: Accelerate is a first-class backend and can be faster than OpenBLAS
    # on this path; do not force fallback.
    if sys.platform == "darwin":
        return {"openblas", "accelerate"}
    return {"openblas"}


def _is_gblup_method(method: str) -> bool:
    return str(method) in _GBLUP_METHOD_SET


def _gblup_method_kernel_mode(method: str) -> str:
    m = str(method)
    if m == _GBLUP_METHOD_DOM:
        return "d"
    if m == _GBLUP_METHOD_AD:
        return "ad"
    return "a"


def _gblup_method_display(method: str) -> str:
    mode = _gblup_method_kernel_mode(str(method))
    if mode == "a":
        return "GBLUP"
    return f"GBLUP({mode})"


def _method_display_name(method: str) -> str:
    m = str(method)
    if _is_gblup_method(m):
        return _gblup_method_display(m)
    return m


def _gblup_mode_label(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "d":
        return "dominance"
    if m == "ad":
        return "additive+dominance"
    return "additive"


def _select_top_method_for_trait(
    methods: list[str],
    method_result_map: dict[str, dict[str, typing.Any]],
    train_truth: np.ndarray,
    metric: str,
) -> tuple[str, dict[str, dict[str, float]]]:
    metric_key = str(metric).strip().lower()
    scores: dict[str, dict[str, float]] = {}
    best_method = ""
    best_score = np.inf if metric_key in {"rmse", "nrmse"} else -np.inf
    for m in methods:
        key = str(m)
        res = method_result_map.get(key)
        if res is None:
            continue
        oof_pred = res.get("oof_pred", None)
        oof_truth = res.get("oof_truth", train_truth)
        if oof_pred is None:
            continue
        y_true = np.asarray(oof_truth, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(oof_pred, dtype=np.float64).reshape(-1)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if int(mask.sum()) < 2:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        pear = float(pearsonr(yt, yp).statistic)
        spear = float(spearmanr(yt, yp).statistic)
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(ss_res / max(1, int(mask.sum()))))
        y_std = float(np.std(yt, ddof=0))
        nrmse = float(rmse / y_std) if np.isfinite(y_std) and y_std > 0.0 else float("nan")
        scores[key] = {
            "pearson": pear,
            "spearman": spear,
            "r2": r2,
            "rmse": rmse,
            "nrmse": nrmse,
        }
        score = scores[key].get(metric_key, float("nan"))
        if not np.isfinite(score):
            continue
        if metric_key in {"rmse", "nrmse"}:
            better = (best_method == "") or (score < best_score)
        else:
            better = (best_method == "") or (score > best_score)
        if better:
            best_method = key
            best_score = float(score)
    if best_method == "":
        # Fallback: first available method if all metrics are NaN.
        for m in methods:
            key = str(m)
            if key in method_result_map:
                best_method = key
                break
    if best_method == "":
        raise RuntimeError("No valid method result found for TOP selection.")
    return best_method, scores


def _normalize_gblup_kernel_token(token: str) -> str:
    t = str(token).strip().lower()
    if t in {"a", "add", "additive"}:
        return "a"
    if t in {"d", "dom", "dominance"}:
        return "d"
    if t in {"ad", "da", "adddom", "add+dom", "additive+dominance", "both"}:
        return "ad"
    raise ValueError(
        f"Invalid -GBLUP kernel token: {token!r}. "
        "Allowed: a, d, ad."
    )


def _parse_gblup_kernel_modes(raw: list[list[str]] | None) -> list[str]:
    if raw is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for group in raw:
        vals = [str(x).strip() for x in (group or []) if str(x).strip() != ""]
        if len(vals) == 0:
            vals = ["a"]
        for tok in vals:
            mode = _normalize_gblup_kernel_token(tok)
            if mode in seen:
                continue
            seen.add(mode)
            out.append(mode)
    return out


def _gblup_mode_to_method(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "d":
        return _GBLUP_METHOD_DOM
    if m == "ad":
        return _GBLUP_METHOD_AD
    return _GBLUP_METHOD_ADD


def _methods_need_additive_dense(methods: list[str]) -> bool:
    return any(
        _is_gblup_method(str(m)) and (_gblup_method_kernel_mode(str(m)) in {"d", "ad"})
        for m in methods
    )


_JXMODEL_FORMAT = "janusx.gs.jxmodel.v1"
_JXMODEL_TOP_BUNDLE_METHOD = "GS_TOP_BUNDLE"
_SELECT_INTERACTIVE_SENTINEL = "__JX_INTERACTIVE_TOP_SELECT__"


def _is_top_bundle_method(method: str) -> bool:
    return str(method).strip().upper() == _JXMODEL_TOP_BUNDLE_METHOD


def _is_top_bundle_payload(payload: typing.Mapping[str, typing.Any]) -> bool:
    return _is_top_bundle_method(str(payload.get("method", "")))


def _is_jxmodel_export_supported(method: str) -> bool:
    m = str(method)
    return bool(
        (m == "rrBLUP")
        or (m in {"BayesA", "BayesB", "BayesCpi"})
        or (m in _ML_METHOD_MAP)
    )


def _is_effect_export_supported(method: str) -> bool:
    m = str(method)
    return bool(_is_gblup_method(m) or _is_jxmodel_export_supported(m))


def _known_jxmodel_methods() -> set[str]:
    return _GBLUP_METHOD_SET | {"rrBLUP", "BayesA", "BayesB", "BayesCpi"} | set(_ML_METHOD_MAP.keys())


def _parse_jxmodel_method_from_name(path_like: str) -> str | None:
    base = os.path.basename(str(path_like))
    if not base.endswith(".jxmodel"):
        return None
    stem = base[: -len(".jxmodel")]
    if stem == "":
        return None
    token = str(stem.split(".")[-1]).strip()
    if token in _known_jxmodel_methods():
        return token
    return None


def _discover_jxmodel_methods(model_arg: str) -> list[str]:
    def _from_bundle(payload: dict[str, typing.Any]) -> list[str]:
        trait_models = payload.get("trait_models", None)
        out: list[str] = []
        seen: set[str] = set()
        if isinstance(trait_models, dict):
            for item in trait_models.values():
                if not isinstance(item, dict):
                    continue
                meth = str(item.get("selected_method", "")).strip()
                if meth == "" or meth in seen:
                    continue
                seen.add(meth)
                out.append(meth)
        return out

    p = str(model_arg)
    if os.path.isfile(p):
        one = _parse_jxmodel_method_from_name(p)
        if one is not None:
            return [one]
        try:
            payload = _load_jxmodel(p)
            method = str(payload.get("method", "")).strip()
            if _is_top_bundle_method(method):
                return _from_bundle(payload)
            return [method] if method != "" else []
        except Exception:
            return []
    if not os.path.isdir(p):
        return []
    found: list[str] = []
    seen: set[str] = set()
    for fp in sorted(glob.glob(os.path.join(p, "*.jxmodel"))):
        m = _parse_jxmodel_method_from_name(fp)
        if m is None:
            try:
                payload = _load_jxmodel(fp)
                meth = str(payload.get("method", "")).strip()
                if _is_top_bundle_method(meth):
                    for x in _from_bundle(payload):
                        if x not in seen:
                            seen.add(x)
                            found.append(x)
            except Exception:
                pass
            continue
        if m in seen:
            continue
        seen.add(m)
        found.append(m)
    return found


def _resolve_jxmodel_file(
    *,
    model_arg: str,
    trait_name: str,
    method: str,
    prefix_hint: str | None,
) -> str:
    p = str(model_arg)
    if os.path.isfile(p):
        return p
    if not os.path.isdir(p):
        raise FileNotFoundError(
            f"--model path does not exist: {model_arg}"
        )

    trait_token = str(trait_name)
    method_token = str(method)
    candidates: list[str] = []
    # New naming: {trait}.{method}.jxmodel
    new_exact = os.path.join(p, f"{trait_token}.{method_token}.jxmodel")
    if os.path.isfile(new_exact):
        candidates.append(new_exact)
    candidates.extend(sorted(glob.glob(os.path.join(p, f"*.{trait_token}.{method_token}.jxmodel"))))
    # Legacy naming: {prefix}.{trait}.gs.{method}.jxmodel
    if prefix_hint is not None and str(prefix_hint).strip() != "":
        exact = os.path.join(
            p,
            f"{str(prefix_hint)}.{trait_token}.gs.{method_token}.jxmodel",
        )
        if os.path.isfile(exact):
            candidates.append(exact)
    candidates.extend(sorted(glob.glob(os.path.join(p, f"*.{trait_token}.gs.{method_token}.jxmodel"))))
    if len(candidates) == 0:
        # Accept generic fallback files inside model directory, e.g. *.rrBLUP.jxmodel
        candidates.extend(sorted(glob.glob(os.path.join(p, f"*.{method_token}.jxmodel"))))
    if len(candidates) == 0:
        candidates.extend(sorted(glob.glob(os.path.join(p, f"*.gs.{method_token}.jxmodel"))))
    uniq = list(dict.fromkeys(candidates))
    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) == 0:
        raise FileNotFoundError(
            f"Cannot find model file for trait={trait_token}, method={method_token} under {p}."
        )
    raise RuntimeError(
        f"Ambiguous model files for trait={trait_token}, method={method_token}: "
        + ", ".join(uniq[:5])
        + (" ..." if len(uniq) > 5 else "")
    )


def _save_jxmodel(path: str, payload: dict[str, typing.Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_jxmodel(path: str) -> dict[str, typing.Any]:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid jxmodel payload type: {type(obj)!r}")
    fmt = str(obj.get("format", "")).strip()
    if fmt != _JXMODEL_FORMAT:
        raise ValueError(
            f"Unsupported jxmodel format: {fmt!r}. Expected {_JXMODEL_FORMAT!r}."
        )
    return typing.cast(dict[str, typing.Any], obj)


def _strip_plink_suffix(path_or_prefix: str) -> str:
    p = str(path_or_prefix).strip()
    low = p.lower()
    if low.endswith(".bed") or low.endswith(".bim") or low.endswith(".fam"):
        return p[:-4]
    return p


def _read_plink_bim_effect_meta(
    prefix_or_bim: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prefix = _strip_plink_suffix(str(prefix_or_bim))
    bim_path = prefix if str(prefix).lower().endswith(".bim") else f"{prefix}.bim"
    if not os.path.isfile(bim_path):
        raise FileNotFoundError(f"Cannot find BIM file: {bim_path}")

    chrom: list[str] = []
    pos: list[int] = []
    allele0: list[str] = []
    allele1: list[str] = []
    with open(bim_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            s = str(line).strip()
            if s == "" or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Invalid BIM row at {bim_path}:{line_no} (expected >= 6 columns)."
                )
            chrom.append(str(parts[0]))
            try:
                p = int(parts[3])
            except Exception:
                p = int(float(parts[3]))
            pos.append(int(p))
            allele0.append(str(parts[4]))
            allele1.append(str(parts[5]))
    return (
        np.asarray(chrom, dtype=object),
        np.asarray(pos, dtype=np.int64),
        np.asarray(allele0, dtype=object),
        np.asarray(allele1, dtype=object),
    )


def _extract_ml_effect_vector_from_estimator(
    estimator: typing.Any,
) -> tuple[np.ndarray | None, str]:
    if estimator is None:
        return None, "missing_estimator"
    base_est = estimator
    scaler = None
    named_steps = getattr(estimator, "named_steps", None)
    if isinstance(named_steps, dict) and len(named_steps) > 0:
        scaler = named_steps.get("scale", None)
        try:
            base_est = list(named_steps.values())[-1]
        except Exception:
            base_est = estimator

    if hasattr(base_est, "feature_importances_"):
        arr = np.asarray(getattr(base_est, "feature_importances_"), dtype=np.float64).reshape(-1)
        if int(arr.size) > 0:
            return np.ascontiguousarray(arr, dtype=np.float64), "feature_importances_"

    if hasattr(base_est, "coef_"):
        arr = np.asarray(getattr(base_est, "coef_"), dtype=np.float64).reshape(-1)
        src = "coef_"
        if scaler is not None and hasattr(scaler, "scale_"):
            scale = np.asarray(getattr(scaler, "scale_"), dtype=np.float64).reshape(-1)
            if int(scale.size) == int(arr.size):
                safe_scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
                arr = arr / safe_scale
                src = "coef_/scaled_back"
        if int(arr.size) > 0:
            return np.ascontiguousarray(arr, dtype=np.float64), src
    return None, "no_ml_feature_score"


def _infer_effect_kind_from_state(
    model_state: dict[str, typing.Any],
    *,
    ml_source: str = "",
) -> str:
    kind = str(model_state.get("kind", "")).strip().lower()
    method = str(model_state.get("method", "")).strip()
    explicit = str(model_state.get("effect_kind", "")).strip().lower()
    if explicit in {"signed_beta", "importance", "attribution", "kernel_projection"}:
        return explicit
    if kind in {"rrblup_linear", "bayes_linear"}:
        return "signed_beta"
    if kind in {"gblup_kernel_projection", "gblup_kernel_projection_ad"}:
        return "kernel_projection"
    if _is_gblup_method(method):
        return "kernel_projection"
    if kind == "mlsk_model":
        src = str(ml_source).strip().lower()
        if src.startswith("feature_importances_"):
            return "importance"
        if src.startswith("coef_"):
            return "signed_beta"
        return "importance"
    return "signed_beta"


_EFFECT_KIND_COLUMNS: dict[str, str] = {
    "signed_beta": "signed_beta",
    "importance": "importance",
    "attribution": "attribution",
    "kernel_projection": "kernel_projection",
}


def _primary_effect_col_for_kind(effect_kind: str) -> str:
    k = str(effect_kind).strip().lower()
    return _EFFECT_KIND_COLUMNS.get(k, "signed_beta")


def _candidate_effect_cols_for_kind(effect_kind: str) -> list[str]:
    primary = _primary_effect_col_for_kind(effect_kind)
    cols = [primary]
    # Backward compatibility: older artifacts only contain `beta`.
    if primary != "beta":
        cols.append("beta")
    return cols


def _as_opt_f64_vector(value: typing.Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    return np.ascontiguousarray(arr, dtype=np.float64)


def _prepare_model_state_for_export_raw_012(
    *,
    model_state: dict[str, typing.Any],
    packed_ctx: dict[str, typing.Any] | None,
) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    state_export = dict(model_state)
    kind = str(state_export.get("kind", "")).strip().lower()
    is_std = bool(state_export.get("standardized", True))
    notes: list[str] = []

    meta: dict[str, typing.Any] = {
        "model_kind": str(kind),
        "converted": False,
        "scale": ("standardized" if is_std else "raw_012"),
        "notes": notes,
    }

    beta_vec = _as_opt_f64_vector(state_export.get("beta", None))
    if beta_vec is None:
        notes.append("beta_missing_or_unreadable")
        return state_export, meta
    if int(beta_vec.shape[0]) == 0:
        notes.append("beta_empty")
        state_export["export_scale"] = "raw_012"
        meta["converted"] = True
        meta["scale"] = "raw_012"
        return state_export, meta

    alpha_val = float(state_export.get("alpha", 0.0))
    alpha_raw = float(alpha_val)
    beta_raw = np.ascontiguousarray(beta_vec, dtype=np.float64)
    cov_beta_raw: np.ndarray | None = None
    cov_beta_source = "none"
    can_apply_raw_affine = not bool(is_std)

    linear_kind = kind in {"rrblup_linear", "bayes_linear"}
    if linear_kind and is_std:
        row_mean_vec = _as_opt_f64_vector(state_export.get("row_mean", None))
        row_inv_vec = _as_opt_f64_vector(state_export.get("row_inv_sd", None))
        row_stats_source = "model_state"
        if row_mean_vec is None or row_inv_vec is None:
            if _looks_like_packed_payload(packed_ctx):
                try:
                    mean_auto, inv_auto = _ensure_packed_standard_stats_cached(
                        typing.cast(dict[str, typing.Any], packed_ctx)
                    )
                    row_mean_vec = np.ascontiguousarray(
                        np.asarray(mean_auto, dtype=np.float64).reshape(-1),
                        dtype=np.float64,
                    )
                    row_inv_vec = np.ascontiguousarray(
                        np.asarray(inv_auto, dtype=np.float64).reshape(-1),
                        dtype=np.float64,
                    )
                    row_stats_source = "packed_ctx"
                except Exception as ex:
                    notes.append(f"row_stats_from_packed_failed:{ex}")
        m = int(beta_vec.shape[0])
        if (
            row_mean_vec is not None
            and row_inv_vec is not None
            and int(row_mean_vec.shape[0]) == m
            and int(row_inv_vec.shape[0]) == m
        ):
            inv_safe = np.where(np.isfinite(row_inv_vec), row_inv_vec, 0.0)
            beta_raw = np.ascontiguousarray(beta_vec * inv_safe, dtype=np.float64)
            alpha_raw = float(alpha_val - float(np.dot(row_mean_vec, beta_raw)))
            state_export["standardized"] = False
            state_export["alpha_standardized"] = float(alpha_val)
            state_export["alpha"] = float(alpha_raw)
            state_export["beta"] = np.ascontiguousarray(
                np.asarray(beta_raw, dtype=np.float64).reshape(-1),
                dtype=np.float64,
            )
            state_export["export_scale"] = "raw_012"
            state_export["export_marker_transform"] = "converted_from_standardized"
            state_export["export_marker_stats_source"] = str(row_stats_source)
            meta["converted"] = True
            meta["scale"] = "raw_012"
            meta["marker_stats_source"] = str(row_stats_source)
            can_apply_raw_affine = True
        else:
            notes.append(
                "missing_or_mismatched_row_stats_for_standardized_marker_conversion"
            )
    elif linear_kind:
        state_export["export_scale"] = "raw_012"
        meta["converted"] = True
        meta["scale"] = "raw_012"
        can_apply_raw_affine = True
    else:
        notes.append("non_linear_or_non_marker_effect_model")

    cov_beta_key = None
    for key in ("cov_beta", "cov_coef", "cov_coeff", "covariate_beta", "gamma"):
        if key in state_export:
            cov_beta_key = key
            break
    cov_beta_vec = (
        _as_opt_f64_vector(state_export.get(cov_beta_key, None))
        if cov_beta_key is not None
        else None
    )
    cov_mean_vec = _as_opt_f64_vector(
        state_export.get("cov_means", state_export.get("cov_mean", None))
    )
    cov_inv_vec = _as_opt_f64_vector(
        state_export.get("cov_inv_sd", state_export.get("cov_inv_std", None))
    )

    if cov_beta_vec is not None and int(cov_beta_vec.shape[0]) > 0:
        cov_beta_raw = np.ascontiguousarray(cov_beta_vec, dtype=np.float64)
        cov_beta_source = str(cov_beta_key or "cov_beta")
        if bool(can_apply_raw_affine):
            if is_std and cov_inv_vec is not None and int(cov_inv_vec.shape[0]) == int(
                cov_beta_vec.shape[0]
            ):
                cov_beta_raw = np.ascontiguousarray(
                    cov_beta_vec * np.where(np.isfinite(cov_inv_vec), cov_inv_vec, 0.0),
                    dtype=np.float64,
                )
                if cov_mean_vec is not None and int(cov_mean_vec.shape[0]) == int(
                    cov_beta_raw.shape[0]
                ):
                    alpha_raw = float(alpha_raw - float(np.dot(cov_mean_vec, cov_beta_raw)))
                    state_export["alpha"] = float(alpha_raw)
                cov_beta_source = f"{cov_beta_source}:converted_from_standardized"
        elif bool(is_std):
            notes.append(
                "skip_covariate_conversion_due_to_unconverted_marker_scale"
            )
        state_export["export_cov_beta_raw_012"] = np.ascontiguousarray(
            np.asarray(cov_beta_raw, dtype=np.float64).reshape(-1),
            dtype=np.float64,
        )
    else:
        if cov_beta_key is not None:
            notes.append("covariate_beta_unreadable")

    scale_now = str(
        state_export.get("export_scale", "raw_012" if (not is_std) else "standardized")
    ).strip().lower()
    alpha_export_val: float | None = None
    if scale_now == "raw_012":
        try:
            alpha_export_val = float(state_export.get("alpha", alpha_raw))
        except Exception:
            alpha_export_val = None
    state_export["export_alpha_raw_012"] = alpha_export_val
    state_export["export_cov_beta_source"] = str(cov_beta_source)
    state_export["export_conversion_notes"] = list(notes)
    meta["alpha_raw_012"] = alpha_export_val
    meta["cov_beta_source"] = str(cov_beta_source)
    return state_export, meta


def _extract_effect_vector_from_model_state(
    model_state: dict[str, typing.Any],
) -> tuple[np.ndarray | None, str, str]:
    inferred_kind = _infer_effect_kind_from_state(model_state)
    cand_keys = _candidate_effect_cols_for_kind(inferred_kind)
    # Try all named semantics first, then `beta` fallback.
    for k in list(_EFFECT_KIND_COLUMNS.values()) + ["beta"]:
        if k not in cand_keys:
            cand_keys.append(k)
    for key in cand_keys:
        vec_raw = model_state.get(str(key), None)
        if vec_raw is None:
            continue
        vec = np.asarray(vec_raw, dtype=np.float64).reshape(-1)
        if int(vec.size) <= 0:
            continue
        src = f"model_state.{key}"
        eff_kind = _infer_effect_kind_from_state(model_state, ml_source=src)
        if str(key).strip().lower() in _EFFECT_KIND_COLUMNS:
            eff_kind = str(key).strip().lower()
        return np.ascontiguousarray(vec, dtype=np.float64), src, eff_kind

    kind = str(model_state.get("kind", "")).strip().lower()
    if kind == "mlsk_model":
        arr, src = _extract_ml_effect_vector_from_estimator(model_state.get("estimator", None))
        if arr is None:
            return None, src, _infer_effect_kind_from_state(model_state, ml_source=src)
        marker_means = np.asarray(
            model_state.get("marker_means", np.zeros((0,), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)
        m_marker = int(marker_means.shape[0])
        if m_marker > 0:
            if int(arr.size) < m_marker:
                short_src = f"{src}:short_feature_vector({arr.size}<{m_marker})"
                return None, short_src, _infer_effect_kind_from_state(model_state, ml_source=short_src)
            slice_src = f"{src}:marker_slice"
            return (
                np.ascontiguousarray(arr[:m_marker], dtype=np.float64),
                slice_src,
                _infer_effect_kind_from_state(model_state, ml_source=slice_src),
            )
        return (
            np.ascontiguousarray(arr, dtype=np.float64),
            src,
            _infer_effect_kind_from_state(model_state, ml_source=src),
        )
    src = "no_beta_in_model_state"
    return None, src, _infer_effect_kind_from_state(model_state, ml_source=src)


def _build_method_effect_table(
    *,
    model_state: dict[str, typing.Any],
    packed_ctx: dict[str, typing.Any] | None,
    genotype_prefix_hint: str | None,
    fallback_marker_count: int | None,
) -> tuple[pd.DataFrame, dict[str, typing.Any]]:
    notes: list[str] = []
    beta_vec, beta_source, effect_kind = _extract_effect_vector_from_model_state(model_state)
    beta_len = 0 if beta_vec is None else int(beta_vec.shape[0])

    metadata_source = "placeholder"
    chrom_meta = np.zeros((0,), dtype=object)
    pos_meta = np.zeros((0,), dtype=np.int64)
    allele0_meta = np.zeros((0,), dtype=object)
    allele1_meta = np.zeros((0,), dtype=object)
    maf_vec = np.zeros((0,), dtype=np.float64)
    row_flip = np.zeros((0,), dtype=np.bool_)

    packed_local = (
        typing.cast(dict[str, typing.Any], packed_ctx)
        if _looks_like_packed_payload(packed_ctx)
        else None
    )

    if packed_local is not None:
        try:
            maf_vec = np.ascontiguousarray(
                np.asarray(packed_local.get("maf", np.zeros((0,), dtype=np.float32)), dtype=np.float64).reshape(-1),
                dtype=np.float64,
            )
        except Exception:
            maf_vec = np.zeros((0,), dtype=np.float64)
        try:
            row_flip = np.ascontiguousarray(
                np.asarray(packed_local.get("row_flip", np.zeros((0,), dtype=np.bool_)), dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            )
        except Exception:
            row_flip = np.zeros((0,), dtype=np.bool_)

    prefix_candidates: list[str] = []
    if packed_local is not None:
        src_prefix = str(packed_local.get("source_prefix", "") or "").strip()
        if src_prefix != "":
            prefix_candidates.append(_strip_plink_suffix(src_prefix))
    if genotype_prefix_hint is not None and str(genotype_prefix_hint).strip() != "":
        prefix_candidates.append(_strip_plink_suffix(str(genotype_prefix_hint)))

    prefix_candidates = list(dict.fromkeys(prefix_candidates))
    for cand in prefix_candidates:
        if not _is_plink_prefix(cand):
            continue
        try:
            chrom_meta, pos_meta, allele0_meta, allele1_meta = _read_plink_bim_effect_meta(cand)
            metadata_source = str(cand)
            if packed_local is not None:
                site_keep_raw = packed_local.get("site_keep", None)
                if site_keep_raw is not None:
                    site_keep = np.ascontiguousarray(
                        np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
                        dtype=np.bool_,
                    )
                    if int(site_keep.shape[0]) == int(chrom_meta.shape[0]):
                        chrom_meta = np.asarray(chrom_meta[site_keep], dtype=object)
                        pos_meta = np.asarray(pos_meta[site_keep], dtype=np.int64)
                        allele0_meta = np.asarray(allele0_meta[site_keep], dtype=object)
                        allele1_meta = np.asarray(allele1_meta[site_keep], dtype=object)
                    else:
                        notes.append("site_keep length mismatch with BIM rows; skipped site_keep alignment")
                if int(row_flip.shape[0]) == int(chrom_meta.shape[0]) and int(row_flip.shape[0]) > 0:
                    swap = np.asarray(row_flip, dtype=np.bool_)
                    if np.any(swap):
                        a0_tmp = allele0_meta.copy()
                        allele0_meta[swap] = allele1_meta[swap]
                        allele1_meta[swap] = a0_tmp[swap]
            break
        except Exception as ex:
            notes.append(f"failed to read BIM metadata from {cand}: {ex}")

    n_rows = 0
    if beta_len > 0:
        n_rows = int(beta_len)
    elif int(maf_vec.shape[0]) > 0:
        n_rows = int(maf_vec.shape[0])
    elif int(chrom_meta.shape[0]) > 0:
        n_rows = int(chrom_meta.shape[0])
    elif fallback_marker_count is not None and int(fallback_marker_count) > 0:
        n_rows = int(fallback_marker_count)

    chrom = np.full((n_rows,), ".", dtype=object)
    pos = np.full((n_rows,), -1, dtype=np.int64)
    allele0 = np.full((n_rows,), "N", dtype=object)
    allele1 = np.full((n_rows,), "N", dtype=object)
    maf = np.full((n_rows,), np.nan, dtype=np.float64)
    effect_col = _primary_effect_col_for_kind(effect_kind)
    effect_values = np.full((n_rows,), np.nan, dtype=np.float64)

    if int(chrom_meta.shape[0]) > 0 and n_rows > 0:
        use_n = min(n_rows, int(chrom_meta.shape[0]))
        chrom[:use_n] = chrom_meta[:use_n]
        pos[:use_n] = pos_meta[:use_n]
        allele0[:use_n] = allele0_meta[:use_n]
        allele1[:use_n] = allele1_meta[:use_n]
        if int(chrom_meta.shape[0]) != n_rows:
            notes.append(f"metadata rows={int(chrom_meta.shape[0])}, effect rows={int(n_rows)}")
    if int(maf_vec.shape[0]) > 0 and n_rows > 0:
        use_n = min(n_rows, int(maf_vec.shape[0]))
        maf[:use_n] = np.asarray(maf_vec[:use_n], dtype=np.float64)
        if int(maf_vec.shape[0]) != n_rows:
            notes.append(f"maf rows={int(maf_vec.shape[0])}, effect rows={int(n_rows)}")
    if beta_vec is not None and n_rows > 0:
        use_n = min(n_rows, int(beta_vec.shape[0]))
        effect_values[:use_n] = np.asarray(beta_vec[:use_n], dtype=np.float64)
        if int(beta_vec.shape[0]) != n_rows:
            notes.append(f"beta rows={int(beta_vec.shape[0])}, effect rows={int(n_rows)}")
    for note in list(model_state.get("export_conversion_notes", []) or []):
        n = str(note).strip()
        if n != "":
            notes.append(f"model_export:{n}")

    table = pd.DataFrame(
        {
            "chrom": chrom,
            "pos": pos,
            "allele0": allele0,
            "allele1": allele1,
            "maf": maf,
            effect_col: effect_values,
            # Backward-compatible alias for older downstream code.
            "beta": effect_values,
        }
    )
    alpha_meta: float | None = None
    try:
        alpha_val = float(model_state.get("export_alpha_raw_012", np.nan))
        if np.isfinite(alpha_val):
            alpha_meta = float(alpha_val)
    except Exception:
        alpha_meta = None
    meta: dict[str, typing.Any] = {
        "rows": int(n_rows),
        "effect_kind": str(effect_kind),
        "effect_source": str(beta_source),
        "effect_column": str(effect_col),
        "beta_source": str(beta_source),
        "beta_available": bool(beta_vec is not None),
        "beta_non_nan_rows": int(np.sum(np.isfinite(effect_values))),
        "beta_scale": (
            str(model_state.get("export_scale", "raw_012")).strip()
            if (not bool(model_state.get("standardized", True)))
            else str(model_state.get("export_scale", "standardized")).strip()
        ),
        "alpha_raw_012": alpha_meta,
        "cov_beta_source": str(model_state.get("export_cov_beta_source", "none")),
        "metadata_source": str(metadata_source),
        "notes": list(notes),
    }
    return table, meta


def _write_method_effect_tsv(
    *,
    out_path: str,
    model_state: dict[str, typing.Any],
    packed_ctx: dict[str, typing.Any] | None,
    genotype_prefix_hint: str | None,
    fallback_marker_count: int | None,
) -> dict[str, typing.Any]:
    table, meta = _build_method_effect_table(
        model_state=model_state,
        packed_ctx=packed_ctx,
        genotype_prefix_hint=genotype_prefix_hint,
        fallback_marker_count=fallback_marker_count,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table.to_csv(str(out_path), sep="\t", index=False, float_format="%.6g")
    out = dict(meta)
    out["effect_file"] = str(out_path)
    return out


def _sanitize_artifact_token(token: str) -> str:
    s = str(token).strip()
    if s == "":
        return "NA"
    s = re.sub(r"[\\/:*?\"<>|\s]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "NA"


def _reindex_effect_table(
    table: pd.DataFrame | None,
    n_rows: int,
) -> pd.DataFrame:
    if n_rows <= 0:
        return pd.DataFrame(
            {
                "chrom": np.zeros((0,), dtype=object),
                "pos": np.zeros((0,), dtype=np.int64),
                "allele0": np.zeros((0,), dtype=object),
                "allele1": np.zeros((0,), dtype=object),
                "maf": np.zeros((0,), dtype=np.float64),
            }
        )

    chrom = np.full((n_rows,), ".", dtype=object)
    pos = np.full((n_rows,), -1, dtype=np.int64)
    allele0 = np.full((n_rows,), "N", dtype=object)
    allele1 = np.full((n_rows,), "N", dtype=object)
    maf = np.full((n_rows,), np.nan, dtype=np.float64)
    if table is not None and int(table.shape[0]) > 0:
        t = table
        use_n = min(n_rows, int(t.shape[0]))
        if "chrom" in t.columns:
            chrom[:use_n] = np.asarray(t["chrom"], dtype=object).reshape(-1)[:use_n]
        if "pos" in t.columns:
            pos[:use_n] = np.asarray(t["pos"], dtype=np.int64).reshape(-1)[:use_n]
        if "allele0" in t.columns:
            allele0[:use_n] = np.asarray(t["allele0"], dtype=object).reshape(-1)[:use_n]
        if "allele1" in t.columns:
            allele1[:use_n] = np.asarray(t["allele1"], dtype=object).reshape(-1)[:use_n]
        if "maf" in t.columns:
            maf[:use_n] = np.asarray(t["maf"], dtype=np.float64).reshape(-1)[:use_n]
    return pd.DataFrame(
        {
            "chrom": chrom,
            "pos": pos,
            "allele0": allele0,
            "allele1": allele1,
            "maf": maf,
        }
    )


def _resolve_effect_col_from_table(tb: pd.DataFrame | None) -> str | None:
    if tb is None:
        return None
    for c in ("signed_beta", "kernel_projection", "importance", "attribution", "beta"):
        if c in tb.columns:
            return str(c)
    return None


def _build_trait_merged_effect_table(
    *,
    method_tables: dict[str, pd.DataFrame],
    method_order: list[str],
    method_display_map: dict[str, str],
) -> pd.DataFrame:
    n_rows = 0
    for _k, _tb in method_tables.items():
        n_rows = max(n_rows, int(_tb.shape[0]))
    if n_rows <= 0:
        n_rows = 1

    best_key = ""
    best_score = -1
    for k, tb in method_tables.items():
        if int(tb.shape[0]) <= 0:
            continue
        score = 0
        if "chrom" in tb.columns:
            chrom_v = np.asarray(tb["chrom"], dtype=object).reshape(-1)
            score += int(np.sum(chrom_v != "."))
        if "pos" in tb.columns:
            pos_v = np.asarray(tb["pos"], dtype=np.int64).reshape(-1)
            score += int(np.sum(pos_v >= 0))
        if "maf" in tb.columns:
            maf_v = np.asarray(tb["maf"], dtype=np.float64).reshape(-1)
            score += int(np.sum(np.isfinite(maf_v)))
        if score > best_score:
            best_key = str(k)
            best_score = int(score)

    base_table = _reindex_effect_table(method_tables.get(best_key, None), n_rows)
    used_cols: set[str] = set(base_table.columns.tolist())
    for method in list(method_order):
        m_key = str(method)
        disp_raw = str(method_display_map.get(m_key, m_key))
        col = disp_raw if disp_raw != "" else m_key
        if col in used_cols:
            i = 2
            while f"{col}_{i}" in used_cols:
                i += 1
            col = f"{col}_{i}"
        used_cols.add(col)

        beta_full = np.full((n_rows,), np.nan, dtype=np.float64)
        tb = method_tables.get(m_key, None)
        eff_col = _resolve_effect_col_from_table(tb)
        if tb is not None and eff_col is not None:
            beta_vec = np.asarray(tb[eff_col], dtype=np.float64).reshape(-1)
            use_n = min(n_rows, int(beta_vec.shape[0]))
            beta_full[:use_n] = beta_vec[:use_n]
        base_table[col] = beta_full
    return base_table


def _json_safe(value: typing.Any) -> typing.Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(x) for x in value.tolist()]
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _order_gs_saved_result_paths(paths: typing.Sequence[str]) -> list[str]:
    dedup = list(dict.fromkeys([str(p) for p in paths if str(p).strip() != ""]))
    primary: list[str] = []
    model_effect: list[str] = []
    summary: list[str] = []
    for p in dedup:
        low = str(p).lower()
        if low.endswith("summary.json"):
            summary.append(str(p))
        elif low.endswith(".jxmodel") or low.endswith(".effect.tsv") or low.endswith(".gs.effect"):
            model_effect.append(str(p))
        else:
            primary.append(str(p))
    return primary + model_effect + summary


def _format_gs_saved_result_report(
    paths: typing.Sequence[str],
    *,
    trait_order: typing.Sequence[str] | None = None,
) -> str:
    dedup = list(dict.fromkeys([str(p) for p in paths if str(p).strip() != ""]))
    trait_blocks: "OrderedDict[str, dict[str, list[str]]]" = OrderedDict()
    summary_path: str | None = None
    extras: list[str] = []

    if trait_order is not None:
        for t in list(trait_order):
            tt = str(t).strip()
            if tt != "" and tt not in trait_blocks:
                trait_blocks[tt] = {"pred": [], "effect": [], "model": []}

    def _ensure_trait(trait_name: str) -> dict[str, list[str]]:
        t = str(trait_name).strip()
        if t == "":
            t = "unknown"
        if t not in trait_blocks:
            trait_blocks[t] = {"pred": [], "effect": [], "model": []}
        return trait_blocks[t]

    for p in dedup:
        low = str(p).lower()
        if low.endswith(".svg"):
            continue
        if low.endswith("summary.json"):
            summary_path = str(p)
            continue

        base = os.path.basename(str(p))
        parts = base.split(".")
        # Prediction table: {prefix}.{trait}.gs.tsv
        if len(parts) >= 4 and parts[-2] == "gs" and parts[-1] == "tsv":
            bucket = _ensure_trait(parts[-3])
            if str(p) not in bucket["pred"]:
                bucket["pred"].append(str(p))
            continue
        # Effect table: {prefix}.{trait}.gs.effect (or legacy *.effect.tsv)
        if len(parts) >= 4 and parts[-2] == "gs" and parts[-1] == "effect":
            bucket = _ensure_trait(parts[-3])
            if str(p) not in bucket["effect"]:
                bucket["effect"].append(str(p))
            continue
        if len(parts) >= 3 and parts[-2] == "effect" and parts[-1] == "tsv":
            bucket = _ensure_trait(parts[-3])
            if str(p) not in bucket["effect"]:
                bucket["effect"].append(str(p))
            continue
        # Model: {trait}.{method}.jxmodel (exclude TOP bundle)
        if low.endswith(".jxmodel") and (not low.endswith(".gs.top.jxmodel")):
            stem = base[: -len(".jxmodel")]
            if "." in stem:
                trait_token = stem.rsplit(".", 1)[0]
                bucket = _ensure_trait(trait_token)
                if str(p) not in bucket["model"]:
                    bucket["model"].append(str(p))
                continue

        extras.append(str(p))

    lines: list[str] = []
    for trait_name, bucket in trait_blocks.items():
        rows: list[str] = []
        for key in ("pred", "effect", "model"):
            for fp in list(bucket.get(key, [])):
                rows.append(f"  {key:<10}{format_path_for_display(str(fp))}")
        if len(rows) <= 0:
            continue
        lines.append(str(trait_name))
        lines.extend(rows)

    for p in extras:
        lines.append(f"extra    {format_path_for_display(str(p))}")

    if summary_path is not None:
        lines.append(f"summary  {format_path_for_display(str(summary_path))}")

    return "\n".join(lines).strip()


def _resolve_top_bundle_model_file(
    *,
    model_arg: str,
    prefix_hint: str | None,
) -> str:
    p = str(model_arg)
    if os.path.isfile(p):
        payload = _load_jxmodel(p)
        if not _is_top_bundle_payload(payload):
            raise ValueError(
                f"--model file is not a GS TOP bundle: {p} "
                f"(method={payload.get('method', '')!r})."
            )
        return p
    if not os.path.isdir(p):
        raise FileNotFoundError(f"--model path does not exist: {p}")

    cand: list[str] = []
    if prefix_hint is not None and str(prefix_hint).strip() != "":
        hint = os.path.join(p, f"{str(prefix_hint)}.gs.TOP.jxmodel")
        if os.path.isfile(hint):
            cand.append(hint)
    cand.extend(sorted(glob.glob(os.path.join(p, "*.gs.TOP.jxmodel"))))
    cand.extend(sorted(glob.glob(os.path.join(p, "*.jxmodel"))))
    uniq = list(dict.fromkeys(cand))

    valid: list[str] = []
    for fp in uniq:
        try:
            payload = _load_jxmodel(fp)
        except Exception:
            continue
        if _is_top_bundle_payload(payload):
            valid.append(fp)
    if len(valid) == 1:
        return valid[0]
    if len(valid) == 0:
        raise FileNotFoundError(
            f"Cannot find GS TOP bundle (*.gs.TOP.jxmodel) under: {p}"
        )
    raise RuntimeError(
        "Ambiguous GS TOP bundle files: " + ", ".join(valid[:5]) + (" ..." if len(valid) > 5 else "")
    )


@dataclass
class _GsLdPruneSpec:
    window_variants: int | None
    window_bp: int | None
    step_variants: int
    r2_threshold: float

    def label(self) -> str:
        if self.window_bp is not None:
            return (
                f"window={int(self.window_bp)}bp, "
                f"step={int(self.step_variants)}, r2={float(self.r2_threshold):.4f}"
            )
        return (
            f"window={int(self.window_variants or 0)} variants, "
            f"step={int(self.step_variants)}, r2={float(self.r2_threshold):.4f}"
        )


def _parse_ld_prune_window(token: str) -> tuple[int | None, int | None]:
    t = str(token).strip().lower()
    if t == "":
        raise ValueError("Empty LD prune window token.")
    if t.endswith("kb"):
        v = float(t[:-2].strip())
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid LD prune window (kb): {token}")
        return None, int(max(1, round(v * 1000.0)))
    if t.endswith("bp"):
        v = float(t[:-2].strip())
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"Invalid LD prune window (bp): {token}")
        return None, int(max(1, round(v)))
    v = float(t)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"Invalid LD prune window: {token}")
    # Keep gformat-compatible semantics: numeric window defaults to kb.
    return None, int(max(1, round(v * 1000.0)))


def _parse_ld_prune_args(values: list[str] | None) -> _GsLdPruneSpec | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(
            "Invalid --ldprune usage. Expected 3 values: "
            "--ldprune <window size[kb|bp]> <step size (variant ct)> <r^2 threshold>"
        )
    w_var, w_bp = _parse_ld_prune_window(values[0])
    step = int(float(str(values[1]).strip()))
    if step <= 0:
        raise ValueError(f"--ldprune step must be > 0, got {values[1]!r}")
    r2 = float(str(values[2]).strip())
    if (not np.isfinite(r2)) or (r2 <= 0.0) or (r2 > 1.0):
        raise ValueError(
            f"--ldprune r^2 threshold must be in (0, 1], got {values[2]!r}"
        )
    return _GsLdPruneSpec(
        window_variants=w_var,
        window_bp=w_bp,
        step_variants=int(step),
        r2_threshold=float(r2),
    )


def _is_plink_prefix(path_or_prefix: str) -> bool:
    p = str(path_or_prefix).strip()
    if p == "":
        return False
    return all(os.path.isfile(f"{p}.{ext}") for ext in ("bed", "bim", "fam"))


def _is_txt_like_input(path_or_prefix: str) -> bool:
    p = str(path_or_prefix).strip().lower()
    return p.endswith(".txt") or p.endswith(".tsv") or p.endswith(".csv")


def _is_vcf_or_hmp_like_input(path_or_prefix: str) -> bool:
    p = str(path_or_prefix).strip().lower()
    return (
        p.endswith(".vcf")
        or p.endswith(".vcf.gz")
        or p.endswith(".hmp")
        or p.endswith(".hmp.gz")
    )


def _decode_packed_subset_to_dense_raw_f32(
    packed_ctx: dict[str, typing.Any],
    sample_indices: np.ndarray,
) -> np.ndarray:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_decode_rows_f32")):
        raise RuntimeError(
            "Rust packed BED decode helper is unavailable. Rebuild/install JanusX extension."
        )
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8), dtype=np.uint8)
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
    n_samples = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    if int(maf.shape[0]) != int(packed.shape[0]):
        raise ValueError("Packed payload mismatch: maf length does not match packed SNP rows.")
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    if np.any(sidx < 0) or np.any(sidx >= int(n_samples)):
        raise ValueError("Packed sample_indices are out of range.")
    ridx = np.ascontiguousarray(np.arange(int(packed.shape[0]), dtype=np.int64), dtype=np.int64)
    row_flip = _ensure_packed_row_flip_cached(packed_ctx)
    decoded = _jxrs.bed_packed_decode_rows_f32(  # type: ignore[union-attr]
        packed,
        int(n_samples),
        ridx,
        row_flip,
        maf,
        sidx,
    )
    dense = np.ascontiguousarray(np.asarray(decoded, dtype=np.float32), dtype=np.float32)
    if dense.ndim != 2 or int(dense.shape[0]) != int(packed.shape[0]):
        raise ValueError(f"Packed subset decode returned invalid shape: {dense.shape}")
    return dense


def _build_memmap_cache_from_chunks(
    *,
    genotype_path: str,
    maf: float,
    missing_rate: float,
    cache_root: str,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.memmap]:
    sample_ids_raw, n_snps_hint_raw = inspect_genotype_file(
        genotype_path,
        snps_only=False,
        maf=float(maf),
        missing_rate=float(missing_rate),
    )
    sample_ids = np.asarray(sample_ids_raw, dtype=str)
    n_samples = int(sample_ids.shape[0])
    n_snps_hint = int(max(0, int(n_snps_hint_raw)))
    if n_samples <= 0:
        raise ValueError("No samples detected for memmap genotype input.")

    os.makedirs(cache_root, mode=0o755, exist_ok=True)
    abs_src = os.path.abspath(str(genotype_path))
    src_tag = os.path.basename(str(genotype_path).rstrip("/\\")).replace(" ", "_")
    key = hashlib.sha1(
        f"{abs_src}|maf={float(maf):.8g}|miss={float(missing_rate):.8g}".encode("utf-8")
    ).hexdigest()[:16]
    mm_path = os.path.join(cache_root, f"{src_tag}.gs.filtered.{key}.npy")

    def _open_existing_or_none(path: str) -> np.memmap | None:
        if not os.path.isfile(path):
            return None
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            return None
        if not isinstance(arr, np.memmap):
            return None
        if arr.ndim != 2:
            return None
        if int(arr.shape[1]) != int(n_samples):
            return None
        return arr

    mm_existing = _open_existing_or_none(mm_path)
    if mm_existing is not None:
        return sample_ids, mm_existing

    # Build filtered SNP-major float32 cache by streaming Rust chunks.
    n_rows_alloc = int(max(1, n_snps_hint))
    mm_out = np.lib.format.open_memmap(
        mm_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_rows_alloc, n_samples),
    )
    written = 0
    try:
        chunks = load_genotype_chunks(
            genotype_path,
            chunk_size=int(max(1, int(chunk_size))),
            maf=float(maf),
            missing_rate=float(missing_rate),
            impute=True,
        )
        for geno_chunk, _sites in chunks:
            arr = np.asarray(geno_chunk, dtype=np.float32)
            if arr.ndim != 2 or int(arr.shape[0]) == 0:
                continue
            n_rows = int(arr.shape[0])
            need_rows = int(written + n_rows)
            if need_rows > int(mm_out.shape[0]):
                grow_rows = int(max(need_rows, int(mm_out.shape[0]) * 2))
                tmp_path = f"{mm_path}.grow.tmp"
                mm_grow = np.lib.format.open_memmap(
                    tmp_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(grow_rows, n_samples),
                )
                if written > 0:
                    step = 10_000
                    for st in range(0, int(written), step):
                        ed = min(int(written), st + step)
                        mm_grow[st:ed, :] = mm_out[st:ed, :]
                mm_grow.flush()
                del mm_out
                del mm_grow
                os.replace(tmp_path, mm_path)
                mm_out = np.lib.format.open_memmap(
                    mm_path,
                    mode="r+",
                    dtype=np.float32,
                    shape=(grow_rows, n_samples),
                )
            mm_out[written:written + n_rows, :] = arr
            written += n_rows
    finally:
        mm_out.flush()
        del mm_out

    if int(written) <= 0:
        raise ValueError(
            "No SNPs left after Rust-side filtering for memmap genotype input. "
            "Please relax --maf/--geno thresholds."
        )

    if int(written) != int(n_rows_alloc):
        tmp_trim = f"{mm_path}.trim.tmp"
        mm_trim = np.lib.format.open_memmap(
            tmp_trim,
            mode="w+",
            dtype=np.float32,
            shape=(int(written), n_samples),
        )
        mm_src = np.load(mm_path, mmap_mode="r")
        step = 10_000
        for st in range(0, int(written), step):
            ed = min(int(written), st + step)
            mm_trim[st:ed, :] = np.asarray(mm_src[st:ed, :], dtype=np.float32)
        mm_trim.flush()
        del mm_trim
        del mm_src
        os.replace(tmp_trim, mm_path)

    mm_final = np.load(mm_path, mmap_mode="r")
    if not isinstance(mm_final, np.memmap):
        raise RuntimeError(f"Failed to open memmap cache as np.memmap: {mm_path}")
    return sample_ids, mm_final


def _env_truthy(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default)).strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


_GS_DEBUG_STAGE = _env_truthy("JX_GS_DEBUG_STAGE", "0")


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
    Detect usable thread count in HPC/container environments.

    Priority:
      1) Scheduler allocation vars (SLURM/PBS/LSF/SGE)
      2) CPU affinity mask
      3) cgroup CPU quota
      4) os.cpu_count()

    Important:
      Do not let existing BLAS/OpenMP thread env vars shrink the detected job
      allocation here. Those env vars are runtime caps that may have been set
      by a previous shell/session default and should not override the current
      scheduler allocation for GS CLI default `-t`.
    """
    # Scheduler-aware allocation hints (most reliable on HPC)
    scheduler_envs = [
        "SLURM_CPUS_PER_TASK",
        "PBS_NP",
        "LSB_DJOB_NUMPROC",
        "NSLOTS",
        "NCPUS",
    ]
    for name in scheduler_envs:
        v = _parse_positive_env_int(name)
        if v is not None:
            detected = v
            break
    else:
        detected = None

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
    return max(1, int(detected))


_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _parse_nonnegative_int(raw: object) -> int | None:
    try:
        v = int(str(raw).strip())
    except Exception:
        return None
    return v if v >= 0 else None


def _split_runtime_threads(total_threads: int, policy: str) -> tuple[int, int, str]:
    total = max(1, int(total_threads))
    p = str(policy).strip().lower()
    if p not in {"auto", "align", "outer", "blas", "balanced", "split", "rust"}:
        p = "auto"
    if p in {"align", "outer"}:
        # Keep BLAS and Rust thread caps identical to `-t`.
        return total, total, p
    if p == "auto":
        # Auto heuristic: favor BLAS on small/medium cores, split on larger hosts.
        p = "blas" if total <= 4 else ("balanced" if total <= 12 else "split")
    if p == "blas":
        return total, 1, p
    if p == "rust":
        return 1, total, p
    if p == "split":
        rust_threads = max(1, total // 2)
        blas_threads = max(1, total - rust_threads)
        return blas_threads, rust_threads, p
    # balanced
    rust_threads = max(1, total // 3)
    blas_threads = max(1, total - rust_threads)
    return blas_threads, rust_threads, p


def configure_thread_runtime(
    *,
    total_threads: int,
    methods: list[str],
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    enabled = _env_truthy("JX_THREAD_COORDINATE", "1")
    # Default to outer-cap mode: one global `-t` cap for both BLAS and Rayon.
    policy_env_raw = str(os.getenv("JX_THREAD_POLICY", "")).strip()
    policy_raw = str(policy_env_raw or "outer").strip().lower() or "outer"
    explicit_blas = _parse_nonnegative_int(os.getenv("JX_MLM_BLAS_THREADS", "").strip())
    explicit_rust = _parse_nonnegative_int(os.getenv("JX_MLM_RUST_THREADS", "").strip())
    gblup_only = bool(
        len(methods) > 0 and all(m in {_GBLUP_METHOD_ADD, "rrBLUP"} for m in methods)
    )

    plan: dict[str, object] = {
        "enabled": bool(enabled),
        "applied": False,
        "reason": "",
        "policy": policy_raw,
        "blas_threads": max(1, int(total_threads)),
        "rust_threads": 0,
        "explicit": bool((explicit_blas is not None) or (explicit_rust is not None)),
    }
    if not enabled:
        plan["reason"] = "disabled_by_env"
        return plan
    if (
        (not gblup_only)
        and (not bool(plan["explicit"]))
        and (policy_env_raw == "")
        and (policy_raw != "outer")
    ):
        plan["reason"] = "skipped_non_gblup_mixed_models"
        return plan

    if (explicit_blas is not None) or (explicit_rust is not None):
        # Keep explicit overrides predictable: missing side falls back to `--thread`.
        blas_threads = explicit_blas if explicit_blas is not None else int(total_threads)
        rust_threads = explicit_rust if explicit_rust is not None else int(total_threads)
        policy_used = "manual"
    else:
        blas_threads, rust_threads, policy_used = _split_runtime_threads(
            int(total_threads), str(policy_raw)
        )

    max_threads = max(1, int(total_threads))
    blas_threads = min(max_threads, max(1, int(blas_threads)))
    rust_threads = min(max_threads, max(1, int(rust_threads)))

    text_blas = str(int(blas_threads))
    text_rust = str(int(rust_threads))
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = text_blas
    os.environ["JX_MLM_BLAS_THREADS"] = text_blas
    os.environ["JX_MLM_RUST_THREADS"] = text_rust
    os.environ["RAYON_NUM_THREADS"] = text_rust
    os.environ["JX_THREADS"] = str(max_threads)

    plan["policy"] = policy_used
    plan["blas_threads"] = int(blas_threads)
    plan["rust_threads"] = int(rust_threads)
    plan["applied"] = True
    plan["reason"] = "ok"

    if logger is not None:
        logger.info(
            "Thread runtime coordination: "
            f"policy={plan['policy']}, total={max_threads}, "
            f"blas={int(blas_threads)}, rust={int(rust_threads)}."
        )
    return plan


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


def _load_phenotype_flexible(path: str) -> pd.DataFrame:
    """
    Load phenotype table with delimiter auto-detection.

    Assumes the first column is sample IDs and averages duplicated IDs.
    """
    sniffed = _sniff_sep(path)
    read_err: Exception | None = None
    df: pd.DataFrame | None = None
    for mode in _candidate_orders(sniffed):
        try:
            kwargs: dict[str, typing.Any] = {}
            if mode == "tab":
                kwargs["sep"] = "\t"
                kwargs["engine"] = "c"
            elif mode == "comma":
                kwargs["sep"] = ","
                kwargs["engine"] = "c"
            else:
                kwargs["sep"] = r"\s+"
                kwargs["engine"] = "c"
            tmp = pd.read_csv(path, **kwargs)
            if tmp.shape[1] <= 1:
                continue
            df = tmp
            break
        except Exception as ex:
            read_err = ex
            continue

    if df is None:
        if read_err is not None:
            raise read_err
        raise ValueError("Failed to read phenotype file.")
    if df.empty:
        raise ValueError("Phenotype file is empty.")

    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str).str.strip()
    out = df.groupby(id_col, sort=False).mean(numeric_only=True)
    out.index = out.index.astype(str)
    return out


def _extract_target_values_from_dataframe(
    df: pd.DataFrame,
    *,
    trait_names: list[str],
    source_label: str,
) -> np.ndarray:
    k = int(len(trait_names))
    if k <= 0:
        raise ValueError("No trait names provided for TOP target parsing.")

    if all(str(t) in df.columns for t in trait_names):
        sub = df.loc[:, [str(t) for t in trait_names]]
    else:
        if int(df.shape[1]) < k:
            raise ValueError(
                f"TOP target file {source_label} has {int(df.shape[1])} columns, "
                f"but {k} trait values are required."
            )
        sub = df.iloc[:, :k]
    arr = sub.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if arr.ndim != 2 or int(arr.shape[1]) != k:
        raise ValueError(
            f"TOP target values from {source_label} have invalid shape: {arr.shape} (expected * x {k})."
        )
    finite_cnt = np.sum(np.isfinite(arr), axis=1)
    if int(np.max(finite_cnt)) <= 0:
        raise ValueError(
            f"TOP target file {source_label} has no row with finite trait values "
            f"for requested traits: {', '.join(trait_names)}."
        )
    first = int(np.argmax(finite_cnt))
    return np.asarray(arr[first, :], dtype=np.float64).reshape(-1)


def _load_top_target_values_from_file(
    path: str,
    *,
    trait_names: list[str],
) -> np.ndarray:
    p = str(path)
    # 1) try phenotype-style loader (header + sample-id first column).
    try:
        ph = _load_phenotype_flexible(p)
        return _extract_target_values_from_dataframe(
            ph,
            trait_names=trait_names,
            source_label=p,
        )
    except Exception:
        pass

    # 2) fallback: headerless table; prefer skipping first column as possible sample ID.
    sniffed = _sniff_sep(p)
    sep: str
    if sniffed == "tab":
        sep = "\t"
    elif sniffed == "comma":
        sep = ","
    else:
        sep = r"\s+"
    raw = pd.read_csv(p, sep=sep, header=None, engine="python")
    if raw.empty:
        raise ValueError(f"TOP target file is empty: {p}")
    arr_full = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    k = int(len(trait_names))
    candidates: list[np.ndarray] = []
    if int(arr_full.shape[1]) >= (k + 1):
        candidates.append(np.asarray(arr_full[:, 1 : 1 + k], dtype=np.float64))
    if int(arr_full.shape[1]) >= k:
        candidates.append(np.asarray(arr_full[:, :k], dtype=np.float64))
    for cand in candidates:
        finite_cnt = np.sum(np.isfinite(cand), axis=1)
        if int(np.max(finite_cnt)) > 0:
            first = int(np.argmax(finite_cnt))
            return np.asarray(cand[first, :], dtype=np.float64).reshape(-1)
    raise ValueError(
        f"Failed to parse TOP target values from file: {p}. "
        "Provide a phenotype-like table with at least one row of finite values."
    )


def _resolve_top_target_values(
    *,
    select_arg: str | None,
    trait_names: list[str],
    trait_mean: np.ndarray | None,
    trait_std: np.ndarray | None,
    logger: logging.Logger,
) -> tuple[np.ndarray | None, str]:
    k = int(len(trait_names))
    if k <= 0:
        return None, "none"

    if select_arg is not None and str(select_arg).strip() == _SELECT_INTERACTIVE_SENTINEL:
        select_arg = None

    if select_arg is not None:
        token = str(select_arg).strip()
        if token == "":
            raise ValueError("--select must not be empty.")
        if os.path.isfile(token):
            values = _load_top_target_values_from_file(token, trait_names=trait_names)
            if int(values.size) != k:
                raise ValueError(
                    f"--select file resolved to {int(values.size)} values, but expected {k}."
                )
            return np.asarray(values, dtype=np.float64).reshape(-1), "file"

        parts = [x for x in re.split(r"[,\s]+", token) if str(x).strip() != ""]
        if len(parts) != k:
            raise ValueError(
                f"--select requires exactly {k} numeric values for traits "
                f"({', '.join(trait_names)}); got {len(parts)}."
            )
        vals: list[float] = []
        for i, p in enumerate(parts):
            pt = str(p).strip().lower()
            if pt in {"na", "nan", ".", "null", "none"}:
                vals.append(float("nan"))
                continue
            try:
                v = float(p)
            except Exception as ex:
                raise ValueError(
                    f"--select value at position {i + 1} is not numeric: {p!r}"
                ) from ex
            vals.append(float(v))
        if not np.isfinite(np.asarray(vals, dtype=np.float64)).any():
            raise ValueError("--select must provide at least one finite target value.")
        return np.asarray(vals, dtype=np.float64).reshape(-1), "inline"

    if not sys.stdin.isatty():
        logger.warning(
            "TOP --select is not provided and stdin is not interactive; skip TOP ranking output."
        )
        return None, "none"

    vals: list[float] = []
    for i, trait in enumerate(trait_names):
        mu = float(trait_mean[i]) if (trait_mean is not None and i < int(trait_mean.size) and np.isfinite(trait_mean[i])) else float("nan")
        sd = float(trait_std[i]) if (trait_std is not None and i < int(trait_std.size) and np.isfinite(trait_std[i])) else float("nan")
        while True:
            raw = input(
                f"Input target value for {trait} (mean={mu:.4g}, std={sd:.4g}): "
            ).strip()
            if raw == "":
                print("Please input a numeric value.")
                continue
            try:
                v = float(raw)
            except Exception:
                print("Invalid number, please retry.")
                continue
            if not np.isfinite(v):
                print("Value must be finite, please retry.")
                continue
            vals.append(float(v))
            break
    return np.asarray(vals, dtype=np.float64).reshape(-1), "interactive"


def build_cv_splits(
    n_samples: int,
    n_splits: int,
    seed: int | None = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build CV splits with sklearn KFold and keep legacy output order:
    each item is (test_idx, train_idx).
    """
    n_samples = int(n_samples)
    n_splits = int(n_splits)
    if n_samples < 2:
        raise ValueError(f"CV requires at least 2 samples, got {n_samples}.")
    if n_splits < 2:
        raise ValueError(f"CV folds must be >=2, got {n_splits}.")
    if n_splits > n_samples:
        raise ValueError(f"CV folds ({n_splits}) cannot exceed sample size ({n_samples}).")

    try:
        if KFold is None:
            raise ImportError("sklearn.model_selection.KFold is unavailable")
        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        row = np.arange(n_samples, dtype=int)
        return [(test_idx, train_idx) for train_idx, test_idx in splitter.split(row)]
    except Exception:
        # Fallback keeps historical behavior if sklearn splitter is unavailable.
        return list(kfold(n_samples, k=n_splits, seed=seed))


def _mlgs_inner_cv(n_samples: int) -> int:
    if n_samples >= 120:
        return 3
    return 2


def _apply_optional_pca(
    Xtrain: np.ndarray,
    Xtest: np.ndarray,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return Xtrain, Xtest
    Xtt = np.concatenate([Xtrain, Xtest], axis=1)
    Xtt = (Xtt - np.mean(Xtt, axis=1, keepdims=True)) / (
        np.std(Xtt, axis=1, keepdims=True) + 1e-8
    )
    val, vec = np.linalg.eigh(Xtt.T @ Xtt / Xtt.shape[0])
    idx = np.argsort(val)[::-1]
    val, vec = val[idx], vec[:, idx]
    dim = np.sum(np.cumsum(val) / np.sum(val) <= 0.9)
    vec = val[:dim] * vec[:, :dim]
    return vec[: Xtrain.shape[1], :].T, vec[Xtrain.shape[1] :, :].T


def _looks_like_packed_payload(obj: typing.Any) -> bool:
    return (
        isinstance(obj, dict)
        and ("packed" in obj)
        and ("n_samples" in obj)
        and ("maf" in obj)
    )


def _normalize_train_pred_indices(
    train_pred_indices: np.ndarray | None,
    n_train: int,
) -> np.ndarray | None:
    if train_pred_indices is None:
        return None
    idx = np.asarray(train_pred_indices, dtype=np.int64).reshape(-1)
    if int(idx.size) == 0:
        return np.zeros((0,), dtype=np.int64)
    if np.any(idx < 0) or np.any(idx >= int(n_train)):
        raise ValueError(
            f"train prediction indices out of range: valid [0, {int(n_train) - 1}]"
        )
    return np.ascontiguousarray(idx, dtype=np.int64)


def _resolve_train_pred_local_indices(
    n_train_fold: int,
    limit_predtrain: int | None,
    fold_id: int,
) -> np.ndarray | None:
    if limit_predtrain is None:
        return None
    limit_i = int(limit_predtrain)
    if limit_i <= 0:
        return np.zeros((0,), dtype=np.int64)
    if limit_i >= int(n_train_fold):
        return None
    # Deterministic fold-local subsampling so all methods are compared on
    # the same train-prediction subset.
    rng = np.random.default_rng(4242 + int(fold_id))
    picked = np.asarray(
        rng.choice(int(n_train_fold), size=int(limit_i), replace=False),
        dtype=np.int64,
    )
    picked.sort()
    return np.ascontiguousarray(picked, dtype=np.int64)


def _build_gblup_cv_grm_once(
    *,
    train_snp: np.ndarray | None,
    packed_ctx: dict[str, typing.Any] | None,
    train_sample_indices: np.ndarray | None,
    n_jobs: int,
) -> np.ndarray | None:
    """
    Build one GRM for GBLUP-CV and reuse it by slicing in each fold.
    """
    if packed_ctx is not None:
        if train_sample_indices is None:
            raise ValueError("Packed GBLUP-CV requires train sample indices.")
        if _jxrs is None:
            return None
        if (not hasattr(_jxrs, "grm_packed_f32")) or (
            not hasattr(_jxrs, "bed_packed_row_flip_mask")
        ):
            return None

        packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
        maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
        n_samples = int(packed_ctx["n_samples"])
        if packed.ndim != 2:
            raise ValueError("Invalid packed payload: packed must be 2D.")
        if maf.shape[0] != packed.shape[0]:
            raise ValueError(
                f"Packed payload mismatch: maf={maf.shape[0]}, packed_rows={packed.shape[0]}."
            )
        exp_bps = (n_samples + 3) // 4
        if int(packed.shape[1]) != int(exp_bps):
            raise ValueError(
                f"Packed payload mismatch: bytes_per_snp={packed.shape[1]}, expected={exp_bps}."
            )

        row_flip_raw = packed_ctx.get("row_flip", None)
        if row_flip_raw is None:
            row_flip = np.asarray(
                _jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
                dtype=np.bool_,
            )
            row_flip = np.ascontiguousarray(row_flip.reshape(-1), dtype=np.bool_)
            packed_ctx["row_flip"] = row_flip
        else:
            row_flip = np.ascontiguousarray(
                np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
                dtype=np.bool_,
            )
            if row_flip.shape[0] != packed.shape[0]:
                raise ValueError(
                    "Packed payload mismatch: row_flip length does not match packed rows."
                )

        sidx = np.ascontiguousarray(np.asarray(train_sample_indices, dtype=np.int64).reshape(-1))
        if np.any(sidx < 0) or np.any(sidx >= int(n_samples)):
            raise ValueError("train sample indices contain out-of-range values.")
        full_identity = bool(
            int(sidx.size) == int(n_samples)
            and np.array_equal(sidx, np.arange(n_samples, dtype=np.int64))
        )
        sidx_arg = None if full_identity else sidx
        block_cols = max(1, min(4096, int(sidx.size) if sidx.size > 0 else int(n_samples)))

        grm_raw = _jxrs.grm_packed_f32(
            packed,
            int(n_samples),
            row_flip,
            maf,
            sample_indices=sidx_arg,
            method=1,
            block_cols=int(block_cols),
            threads=max(1, int(n_jobs)),
            progress_callback=None,
            progress_every=0,
        )
        grm = np.asarray(grm_raw, dtype=np.float64)
    else:
        if train_snp is None:
            return None
        grm = np.asarray(_QK2_GRM(np.asarray(train_snp, dtype=np.float32), log=False), dtype=np.float64)

    if grm.ndim != 2 or grm.shape[0] != grm.shape[1]:
        raise ValueError(f"GBLUP CV GRM must be square; got shape={grm.shape}.")
    grm = 0.5 * (grm + grm.T)
    return np.ascontiguousarray(grm, dtype=np.float64)


def _fit_gblup_reml_from_grm(
    y: np.ndarray,
    grm: np.ndarray,
) -> dict[str, typing.Any]:
    """
    Fit intercept-only GBLUP from a precomputed train-train GRM block.
    """
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    g_base = np.asarray(grm, dtype=np.float64)
    if y_vec.shape[0] != g_base.shape[0] or g_base.shape[0] != g_base.shape[1]:
        raise ValueError(
            f"GBLUP fit expects y and GRM with matching n. got y={y_vec.shape}, grm={g_base.shape}."
        )
    n = int(y_vec.shape[0])
    if n == 0:
        raise ValueError("GBLUP fit received empty training set.")

    # Degenerate tiny folds: return mean-only predictor.
    if n <= 1:
        beta = float(y_vec[0, 0]) if n == 1 else 0.0
        alpha = np.zeros((n, 1), dtype=np.float64)
        return {
            "beta": beta,
            "alpha": alpha,
            "pve": float("nan"),
            "lambda_reml": float("nan"),
            "sigma_g2": float("nan"),
            "sigma_e2": float("nan"),
        }

    x = np.ones((n, 1), dtype=np.float64)
    g_fit = np.ascontiguousarray(0.5 * (g_base + g_base.T), dtype=np.float64)
    g_fit += np.eye(n, dtype=np.float64) * np.float64(1e-6)
    evals, evec = np.linalg.eigh(g_fit)
    evals = np.asarray(evals, dtype=np.float64)
    evec = np.asarray(evec, dtype=np.float64)
    max_eval = float(np.max(evals)) if evals.size > 0 else 0.0
    eig_floor = max(
        1e-12,
        float(np.finfo(np.float64).eps * max(1.0, max_eval)),
    )
    evals = np.maximum(evals, eig_floor)

    x_rot = evec.T @ x
    y_rot = evec.T @ y_vec
    v_floor = max(1e-12, float(np.finfo(np.float64).tiny))
    n_eff = int(n - 1)
    if n_eff <= 0:
        beta = float(np.mean(y_vec))
        alpha = np.zeros((n, 1), dtype=np.float64)
        return {
            "beta": beta,
            "alpha": alpha,
            "pve": float("nan"),
            "lambda_reml": float("nan"),
            "sigma_g2": float("nan"),
            "sigma_e2": float("nan"),
        }

    def _reml_at_lambda(lbd: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        lam = float(lbd)
        if (not np.isfinite(lam)) or (lam <= 0.0):
            lam = float(v_floor)
        v = np.maximum(evals + lam, v_floor)
        v_inv = 1.0 / v
        x_w = v_inv[:, None] * x_rot
        y_w = v_inv[:, None] * y_rot
        xtvx = x_rot.T @ x_w
        xtvy = x_rot.T @ y_w
        xtvx_00 = float(xtvx[0, 0]) if xtvx.size > 0 else 0.0
        if (not np.isfinite(xtvx_00)) or (xtvx_00 <= v_floor):
            xtvx_00 = float(v_floor)
        beta = np.array([[float(xtvy[0, 0]) / xtvx_00]], dtype=np.float64)
        r = y_rot - x_rot @ beta
        rtv = float(np.sum((r.reshape(-1) ** 2) * v_inv))
        if (not np.isfinite(rtv)) or (rtv <= v_floor):
            rtv = float(v_floor)
        log_det_v = float(np.sum(np.log(v)))
        log_det_xtvx = float(np.log(xtvx_00))
        total_log = float(n_eff) * float(np.log(rtv)) + log_det_v + log_det_xtvx
        c = float(n_eff) * (float(np.log(float(n_eff))) - 1.0 - float(np.log(2.0 * np.pi))) / 2.0
        return c - total_log / 2.0, beta, r, v_inv, rtv

    def _objective(log10_lbd: float) -> float:
        lbd = 10.0 ** float(log10_lbd)
        ll, _beta, _r, _v_inv, _rtv = _reml_at_lambda(lbd)
        return -float(ll)

    opt = minimize_scalar(_objective, bounds=(-6, 6), method="bounded")
    log10_lbd = float(opt.x) if np.isfinite(float(opt.x)) else 0.0
    lbd = float(10.0 ** log10_lbd)
    _ll, beta_mat, r, v_inv, rtv = _reml_at_lambda(lbd)

    rhs = v_inv[:, None] * r
    alpha = evec @ rhs
    sigma_g2 = float(rtv) / float(max(1, n_eff))
    sigma_e2 = float(lbd) * sigma_g2
    var_g = sigma_g2 * float(np.mean(evals))
    denom = var_g + sigma_e2
    pve = float(var_g / denom) if (np.isfinite(denom) and denom > 0.0) else float("nan")
    beta = float(beta_mat[0, 0])

    return {
        "beta": beta,
        "alpha": np.ascontiguousarray(alpha, dtype=np.float64),
        "pve": float(pve),
        "lambda_reml": float(lbd),
        "sigma_g2": float(sigma_g2),
        "sigma_e2": float(sigma_e2),
    }


def _predict_gblup_from_cross(
    *,
    cross_kernel: np.ndarray,
    alpha: np.ndarray,
    beta: float,
) -> np.ndarray:
    cross = np.asarray(cross_kernel, dtype=np.float64)
    a = np.asarray(alpha, dtype=np.float64).reshape(-1, 1)
    if cross.ndim != 2:
        raise ValueError(f"cross kernel must be 2D, got shape={cross.shape}.")
    if cross.shape[1] != a.shape[0]:
        raise ValueError(
            f"cross kernel width ({cross.shape[1]}) != alpha length ({a.shape[0]})."
        )
    out = cross @ a + float(beta)
    return np.asarray(out, dtype=np.float64).reshape(-1, 1)


def _build_gblup_test_train_cross_from_markers(
    test_snp: np.ndarray,
    train_snp: np.ndarray,
    *,
    chunk_rows: int = 50_000,
) -> np.ndarray:
    """
    Build cross-kernel K_test_train using train-side centering/variance:
      K = (M_test - mu_train)^T (M_train - mu_train) / sum(var_train)
    """
    m, n_train = train_snp.shape
    m_test, n_test = test_snp.shape
    if int(m_test) != int(m):
        raise ValueError(
            f"test/train marker mismatch: test m={m_test}, train m={m}"
        )
    if int(n_test) == 0:
        return np.zeros((0, int(n_train)), dtype=np.float64)

    step = max(1, int(chunk_rows))
    cross = np.zeros((int(n_test), int(n_train)), dtype=np.float64)
    var_sum_total = 0.0

    for st in range(0, int(m), step):
        ed = min(st + step, int(m))
        tr_blk = np.asarray(train_snp[st:ed], dtype=np.float32)
        te_blk = np.asarray(test_snp[st:ed], dtype=np.float32)
        mu_blk = np.mean(tr_blk, axis=1, dtype=np.float32, keepdims=True)
        var_blk = np.var(tr_blk, axis=1, dtype=np.float32, keepdims=True)
        var_sum_total += float(np.sum(var_blk, dtype=np.float64))
        tr_ctr = tr_blk - mu_blk
        te_ctr = te_blk - mu_blk
        cross += np.asarray(te_ctr.T @ tr_ctr, dtype=np.float64)

    if (not np.isfinite(var_sum_total)) or (var_sum_total <= 0.0):
        raise ValueError("Invalid GBLUP denominator in cross kernel build (sum(var)<=0).")
    cross /= float(var_sum_total)
    return np.ascontiguousarray(cross, dtype=np.float64)


def _tune_ml_method_once(
    method: str,
    Y: np.ndarray,
    Xtrain: np.ndarray,
    PCAdec: bool,
    n_jobs: int,
    progress_hook: typing.Any = None,
) -> dict[str, typing.Any]:
    empty_test = np.zeros((Xtrain.shape[0], 0), dtype=Xtrain.dtype)
    Xtrain_tuned, _ = _apply_optional_pca(Xtrain, empty_test, enabled=PCAdec)

    def _on_ml_search_progress(event: str, payload: dict[str, typing.Any]) -> None:
        if progress_hook is None:
            return
        try:
            progress_hook(str(event), dict(payload))
        except Exception:
            return

    model = MLGS(
        y=Y.reshape(-1),
        M=Xtrain_tuned,
        method=typing.cast(typing.Any, _ML_METHOD_MAP[method]),
        seed=42,
        cv=_mlgs_inner_cv(int(Y.shape[0])),
        n_jobs=max(1, int(n_jobs)),
        fit_on_init=False,
        verbose=False,
        progress_callback=_on_ml_search_progress,
    )
    try:
        planned_total = int(max(1, int(model.planned_search_steps())))
        _on_ml_search_progress("search_total", {"count": planned_total})
    except Exception:
        pass
    model.fit()
    return {
        "params": model.get_final_params(),
        "pve": float((model.best_metrics_ or {}).get("r2", np.nan)),
        "best_metrics": dict(model.best_metrics_ or {}),
    }


def _iter_row_blocks(
    n_rows: int,
    block_size: int,
) -> list[tuple[int, int, np.ndarray]]:
    n = int(max(0, int(n_rows)))
    step = max(1, int(block_size))
    out: list[tuple[int, int, np.ndarray]] = []
    for st in range(0, n, step):
        ed = min(st + step, n)
        ridx = np.ascontiguousarray(np.arange(st, ed, dtype=np.int64))
        out.append((st, ed, ridx))
    return out


def _resolve_rrblup_solver(
    *,
    solver: str,
    n_train: int,
    n_snp: int,
    cfg: dict[str, typing.Any] | None,
) -> str:
    mode = str(solver).strip().lower()
    if mode not in {"exact", "adamw", "pcg", "auto"}:
        mode = "auto"
    if mode != "auto":
        return mode
    cfg_use = cfg or {}
    auto_pcg_min_n = int(max(2, int(cfg_use.get("auto_pcg_min_n", _RRBLUP_AUTO_PCG_MIN_N))))
    n_ref = int(cfg_use.get("auto_pcg_ref_n", n_train))
    n_use = int(max(0, int(n_ref)))
    # Adaptive rrBLUP policy:
    #   n <= 10k: exact REML/FaST path.
    #   n > 10k : PCG path (HE-based lambda auto-estimation when enabled).
    return "pcg" if n_use > auto_pcg_min_n else "exact"


def _rrblup_lambda_raw_to_equation(
    *,
    lambda_raw: float,
    lambda_scale: str,
    n_train: int,
) -> float:
    lam_raw = float(lambda_raw)
    scale = str(lambda_scale).strip().lower()
    if scale == "mean-loss":
        return float(lam_raw * float(max(1, int(n_train))))
    return float(lam_raw)


def _rrblup_lambda_equation_to_raw(
    *,
    lambda_equation: float,
    lambda_scale: str,
    n_train: int,
) -> float:
    lam_eq = float(lambda_equation)
    scale = str(lambda_scale).strip().lower()
    if scale == "mean-loss":
        return float(lam_eq / float(max(1, int(n_train))))
    return float(lam_eq)


def _estimate_rrblup_lambda_subsample_reml(
    *,
    y_train: np.ndarray,
    packed_ctx: dict[str, typing.Any],
    train_sample_indices: np.ndarray,
    n_jobs: int,
    cfg: dict[str, typing.Any] | None,
    progress_hook: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
) -> dict[str, typing.Any]:
    cfg_use = cfg or {}
    y_vec = np.asarray(y_train, dtype=np.float64).reshape(-1)
    train_abs = np.ascontiguousarray(
        np.asarray(train_sample_indices, dtype=np.int64).reshape(-1),
        dtype=np.int64,
    )
    n_train = int(train_abs.shape[0])
    if n_train <= 1 or y_vec.shape[0] != n_train:
        return {
            "lambda_equation": float("nan"),
            "lambda_k": float("nan"),
            "n_sub": 0,
            "repeats": 0,
            "ok_repeats": 0,
            "strategy": "invalid_input",
        }
    sub_min = int(max(16, int(cfg_use.get("lambda_subsample_min_n", _RRBLUP_LAMBDA_SUBSAMPLE_MIN_N))))
    sub_max = int(max(sub_min, int(cfg_use.get("lambda_subsample_max_n", _RRBLUP_LAMBDA_SUBSAMPLE_MAX_N))))
    sub_target = int(max(sub_min, int(cfg_use.get("lambda_subsample_n", _RRBLUP_LAMBDA_SUBSAMPLE_MAX_N))))
    n_sub = int(min(max(1, n_train), min(sub_max, max(sub_min, sub_target))))
    rep_cfg = int(cfg_use.get("lambda_subsample_repeats", _RRBLUP_LAMBDA_SUBSAMPLE_REPEATS))
    repeats = int(min(20, max(5, rep_cfg)))
    seed = int(cfg_use.get("lambda_subsample_seed", 42))
    rng = np.random.default_rng(seed)
    std_eps = float(max(np.finfo(np.float32).eps, float(cfg_use.get("pcg_std_eps", 1e-12))))
    maf_all = np.asarray(packed_ctx.get("maf", np.asarray([], dtype=np.float32)), dtype=np.float64).reshape(-1)
    site_keep_raw = packed_ctx.get("site_keep", None)
    if site_keep_raw is not None and int(maf_all.size) > 0:
        site_keep = np.asarray(site_keep_raw, dtype=bool).reshape(-1)
        if int(site_keep.size) == int(maf_all.size):
            maf_use = maf_all[site_keep]
        else:
            maf_use = maf_all
    else:
        maf_use = maf_all
    if int(maf_use.size) > 0:
        p_use = np.clip(maf_use, 0.0, 0.5)
        var_use = 2.0 * p_use * (1.0 - p_use)
        m_effective = int(np.count_nonzero(np.isfinite(var_use) & (var_use > float(std_eps))))
    else:
        m_effective = 0
    m_scale = float(max(1, int(m_effective)))

    he_diag: dict[str, typing.Any] = {
        "sigma_g2": float("nan"),
        "sigma_e2": float("nan"),
        "h2": float("nan"),
        "he_converged": None,
        "he_iters": None,
        "he_rel_res": float("nan"),
        "lambda_he_raw_k": float("nan"),
        "lambda_he_raw_equation": float("nan"),
        "lambda_he_lambda_k": float("nan"),
        "lambda_he_lambda_equation": float("nan"),
        "trace_samples_used": 0,
        "he_tr_k2": float("nan"),
        "he_tr_k2_solve": float("nan"),
        "he_y_ky": float("nan"),
        "he_y_y": float("nan"),
        "he_tr_k": float("nan"),
        "he_tr_p": float("nan"),
        "he_nnls_projected": None,
        "he_boundary_status": "",
        "he_accepted": None,
        "he_reject_reason": "",
        "he_error": "",
        "he_call_path": "",
        "he_call_api": "",
        "he_site_keep_mode": "",
        "thread_policy": "",
        "stage_blas_threads": 0,
        "stage_rayon_threads": 0,
        "he_threads_arg": 0,
    }
    lambda_auto_strategy = str(
        cfg_use.get("lambda_auto_strategy", "he_first")
    ).strip().lower()
    if lambda_auto_strategy not in {"he_first", "he_only", "reml_first"}:
        lambda_auto_strategy = "he_first"

    def _merge_he_diag(payload: dict[str, typing.Any]) -> dict[str, typing.Any]:
        out = dict(payload)
        out.update(dict(he_diag))
        return out

    def _emit(event: str, **payload: typing.Any) -> None:
        if progress_hook is None:
            return
        try:
            progress_hook(str(event), dict(payload))
        except Exception:
            return

    he_debug_mode = bool(cfg_use.get("debug_mode", False)) or bool(_GS_DEBUG_STAGE)

    def _emit_he_call_path(
        call_path: str,
        call_api: str,
        *,
        note: str = "",
    ) -> None:
        he_diag["he_call_path"] = str(call_path).strip()
        he_diag["he_call_api"] = str(call_api).strip()
        payload = {
            "vc_method": "HE",
            "strategy": "he",
            "he_call_path": str(call_path).strip(),
            "he_call_api": str(call_api).strip(),
        }
        note_txt = str(note).strip()
        if note_txt != "":
            payload["note"] = note_txt
        _emit("pcg_lambda_he_call_path", **payload)
        if he_debug_mode:
            msg = (
                f"[rrBLUP-DEBUG] HE call path={payload['he_call_path']} "
                f"api={payload['he_call_api']}"
            )
            if note_txt != "":
                msg += f" note={note_txt}"
            print(msg, flush=True)

    def _describe_he_site_keep_mode(
        site_keep_raw_obj: typing.Any,
        site_keep_arr: np.ndarray | None,
        packed_rows: int,
    ) -> tuple[str, str]:
        if site_keep_arr is None:
            if site_keep_raw_obj is None:
                return "none", "site_keep not provided"
            try:
                raw_len = int(np.asarray(site_keep_raw_obj).reshape(-1).shape[0])
            except Exception:
                raw_len = -1
            if raw_len > 0 and raw_len != int(packed_rows):
                return (
                    "suppressed_len_mismatch",
                    f"site_keep len={raw_len} != packed_rows={int(packed_rows)}",
                )
            return "none", "site_keep unavailable after normalization"
        keep_len = int(site_keep_arr.shape[0])
        keep_true = int(np.count_nonzero(site_keep_arr))
        if keep_true == keep_len and bool(np.all(site_keep_arr)):
            return "identity", f"site_keep len={keep_len}, all_true=1"
        return (
            "subset_copy_candidate",
            f"site_keep len={keep_len}, keep_true={keep_true}, packed_rows={int(packed_rows)}",
        )

    he_enable = _cfg_truthy(cfg_use.get("he_enable", "on"), default=True)
    can_try_he = bool(
        he_enable
        and
        (_jxrs is not None)
        and hasattr(_jxrs, "he_pcg_bed")
        and _looks_like_packed_payload(packed_ctx)
    )
    if can_try_he:
        try:
            packed_src_view = np.asarray(packed_ctx["packed"], dtype=np.uint8)
            packed_arg = np.ascontiguousarray(
                packed_src_view,
                dtype=np.uint8,
            )
            packed_is_lazy = _packed_ctx_is_lazy_full(packed_ctx)
            maf_arg = np.ascontiguousarray(
                np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )
            n_samples_full = int(packed_ctx["n_samples"])
            if packed_arg.ndim != 2:
                raise ValueError("HE lambda-auto requires packed matrix with ndim=2.")
            if int(maf_arg.shape[0]) != int(packed_arg.shape[0]):
                raise ValueError("HE lambda-auto packed payload mismatch: maf length != packed SNP rows.")
            exp_bps = (n_samples_full + 3) // 4
            if int(packed_arg.shape[1]) != int(exp_bps):
                raise ValueError(
                    "HE lambda-auto packed payload mismatch: "
                    f"bytes_per_snp={packed_arg.shape[1]} expected={exp_bps}."
                )

            row_flip_raw = packed_ctx.get("row_flip", None)
            if row_flip_raw is None:
                row_flip_arg = np.ascontiguousarray(
                    np.asarray(
                        _jxrs.bed_packed_row_flip_mask(packed_arg, int(n_samples_full)),  # type: ignore[union-attr]
                        dtype=np.bool_,
                    ).reshape(-1),
                    dtype=np.bool_,
                )
                packed_ctx["row_flip"] = row_flip_arg
            else:
                row_flip_arg = np.ascontiguousarray(
                    np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
                    dtype=np.bool_,
                )
            if int(row_flip_arg.shape[0]) != int(packed_arg.shape[0]):
                raise ValueError("HE lambda-auto packed payload mismatch: row_flip length != packed SNP rows.")

            site_keep_arg: np.ndarray | None = None
            site_keep_raw = packed_ctx.get("site_keep", None)
            if site_keep_raw is not None:
                site_keep_arg = np.ascontiguousarray(
                    np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
                    dtype=np.bool_,
                )
                if (not packed_is_lazy) and (int(site_keep_arg.shape[0]) != int(maf_arg.shape[0])):
                    site_keep_arg = None
            he_site_keep_mode, he_site_keep_note = _describe_he_site_keep_mode(
                site_keep_raw,
                site_keep_arg,
                int(maf_arg.shape[0]),
            )
            he_diag["he_site_keep_mode"] = str(he_site_keep_mode)

            source_prefix = str(packed_ctx.get("source_prefix", "") or "").strip()
            he_trace_samples = int(
                max(
                    8,
                    int(
                        cfg_use.get(
                            "he_trace_samples",
                            cfg_use.get("pcg_he_trace_samples", 64),
                        )
                    ),
                )
            )
            he_trace_probe_batch = int(
                max(
                    1,
                    int(
                        cfg_use.get(
                            "he_trace_probe_batch",
                            cfg_use.get(
                                "pcg_he_trace_probe_batch",
                                min(64, int(he_trace_samples)),
                            ),
                        )
                    ),
                )
            )
            he_trace_probe_batch = int(
                min(
                    max(1, int(he_trace_probe_batch)),
                    max(1, int(he_trace_samples)),
                )
            )
            he_block_rows = int(
                max(
                    1,
                    int(
                        cfg_use.get(
                            "he_block_rows",
                            cfg_use.get("pcg_block_rows", cfg_use.get("snp_block_size", 4096)),
                        )
                    ),
                )
            )
            he_tol = float(
                max(
                    np.finfo(np.float64).eps,
                    float(cfg_use.get("he_tol", cfg_use.get("pcg_tol", 1e-6))),
                )
            )
            he_max_iter = int(max(1, int(cfg_use.get("he_max_iter", cfg_use.get("pcg_max_iter", 32)))))
            he_seed = int(cfg_use.get("he_seed", seed))
            he_use_train_maf = _cfg_truthy(cfg_use.get("he_use_train_maf", "on"), default=True)
            he_thread_policy_raw = cfg_use.get(
                "he_thread_policy",
                cfg_use.get(
                    "pcg_he_thread_policy",
                    os.getenv(
                        "JX_GS_HE_THREAD_POLICY",
                        os.getenv(
                            "JX_RRBLUP_HE_THREAD_POLICY",
                            _RRBLUP_HE_THREAD_POLICY_DEFAULT,
                        ),
                    ),
                ),
            )
            he_thread_spec = _resolve_he_thread_policy_spec(
                he_thread_policy_raw,
                int(max(1, int(n_jobs))),
            )
            he_diag["thread_policy"] = str(he_thread_spec["policy"])
            he_diag["stage_blas_threads"] = int(he_thread_spec["blas_threads"])
            he_diag["stage_rayon_threads"] = int(he_thread_spec["rayon_threads"])
            he_diag["he_threads_arg"] = int(he_thread_spec["he_threads"])
            he_rss_before: int | None = None
            if he_debug_mode:
                packed_c_contig = bool(getattr(packed_arg, "flags", {}).c_contiguous)
                packed_same_obj = bool(packed_arg is packed_src_view)
                try:
                    packed_shares_memory = bool(np.shares_memory(packed_arg, packed_src_view))
                except Exception:
                    packed_shares_memory = False
                he_block_bytes = int(int(he_block_rows) * int(n_train) * np.dtype(np.float32).itemsize)
                he_probe_bytes = int(
                    int(n_train) * int(he_trace_probe_batch) * np.dtype(np.float32).itemsize
                )
                he_rss_before = _get_process_rss_bytes()
                print(
                    (
                        "[rrBLUP-DEBUG] HE site_keep mode="
                        f"{he_site_keep_mode} ({he_site_keep_note})"
                    ),
                    flush=True,
                )
                print(
                    (
                        "[rrBLUP-DEBUG] HE packed payload "
                        f"rows={int(packed_arg.shape[0])} "
                        f"bytes_per_snp={int(packed_arg.shape[1])} "
                        f"c_contig={int(packed_c_contig)} "
                        f"shared_with_ctx={int(packed_shares_memory)} "
                        f"same_obj={int(packed_same_obj)} "
                        f"nbytes={_format_debug_bytes(int(packed_arg.nbytes))}"
                    ),
                    flush=True,
                )
                print(
                    (
                        "[rrBLUP-DEBUG] HE workspace estimate "
                        f"packed={_format_debug_bytes(int(packed_arg.nbytes))} "
                        f"block={_format_debug_bytes(he_block_bytes)} "
                        f"probe={_format_debug_bytes(he_probe_bytes)} x2 "
                        f"rss_before={_format_debug_bytes(he_rss_before)}"
                    ),
                    flush=True,
                )

            _emit(
                "pcg_lambda_vc_start",
                vc_method="HE",
                strategy="he",
                trace_samples=int(he_trace_samples),
                thread_policy=str(he_thread_spec["policy"]),
                stage_blas_threads=int(he_thread_spec["blas_threads"]),
                stage_rayon_threads=int(he_thread_spec["rayon_threads"]),
                he_threads=int(he_thread_spec["he_threads"]),
            )
            try:
                with runtime_thread_stage(
                    blas_threads=int(he_thread_spec["blas_threads"]),
                    rayon_threads=int(he_thread_spec["rayon_threads"]),
                ):
                    he_call_kwargs = dict(
                        site_keep=site_keep_arg,
                        trace_samples=int(he_trace_samples),
                        trace_probe_batch=int(he_trace_probe_batch),
                        tol=float(he_tol),
                        max_iter=int(he_max_iter),
                        block_rows=int(he_block_rows),
                        std_eps=float(std_eps),
                        use_train_maf=bool(he_use_train_maf),
                        threads=int(he_thread_spec["he_threads"]),
                        blas_threads=int(he_thread_spec["blas_threads"]),
                        seed=int(he_seed),
                        packed=packed_arg,
                        packed_n_samples=int(n_samples_full),
                        maf=maf_arg,
                        row_flip=row_flip_arg,
                    )
                    try:
                        he_out = _jxrs.he_pcg_bed(  # type: ignore[union-attr]
                            source_prefix,
                            train_abs,
                            y_vec,
                            **he_call_kwargs,
                        )
                        _emit_he_call_path(
                            "external_packed",
                            "kwargs_blas_threads",
                        )
                    except TypeError:
                        he_call_kwargs.pop("blas_threads", None)
                        try:
                            he_out = _jxrs.he_pcg_bed(  # type: ignore[union-attr]
                                source_prefix,
                                train_abs,
                                y_vec,
                                **he_call_kwargs,
                            )
                            _emit_he_call_path(
                                "external_packed",
                                "kwargs_no_blas_threads",
                                note="extension_missing_he_blas_threads_kw",
                            )
                        except TypeError:
                            _emit_he_call_path(
                                "legacy_prefix_reload",
                                "legacy_positional",
                                note="extension_missing_packed_he_signature",
                            )
                            he_out = _jxrs.he_pcg_bed(  # type: ignore[union-attr]
                                source_prefix,
                                train_abs,
                                y_vec,
                                site_keep_arg,
                                int(he_trace_samples),
                                float(he_tol),
                                int(he_max_iter),
                                int(he_block_rows),
                                float(std_eps),
                                bool(he_use_train_maf),
                                int(he_thread_spec["he_threads"]),
                                int(he_seed),
                            )
            finally:
                if he_debug_mode:
                    he_rss_after = _get_process_rss_bytes()
                    delta_txt = "NA"
                    if (he_rss_before is not None) and (he_rss_after is not None):
                        delta_txt = _format_debug_bytes(int(he_rss_after) - int(he_rss_before))
                    print(
                        (
                            "[rrBLUP-DEBUG] HE rss after call="
                            f"{_format_debug_bytes(he_rss_after)} "
                            f"delta={delta_txt}"
                        ),
                        flush=True,
                    )
                    _emit_debug_malloc_trim("[rrBLUP-DEBUG] HE")
                _emit(
                    "pcg_lambda_vc_end",
                    vc_method="HE",
                    strategy="he",
                    thread_policy=str(he_thread_spec["policy"]),
                    stage_blas_threads=int(he_thread_spec["blas_threads"]),
                    stage_rayon_threads=int(he_thread_spec["rayon_threads"]),
                    he_threads=int(he_thread_spec["he_threads"]),
                )

            he_t = tuple(he_out)
            if len(he_t) >= 2:
                sigma_g2 = float(he_t[0])
                sigma_e2 = float(he_t[1])
                h2 = float(he_t[2]) if len(he_t) >= 3 else float("nan")
                he_converged = bool(he_t[3]) if len(he_t) >= 4 else None
                he_iters = int(he_t[4]) if len(he_t) >= 5 else None
                he_rel_res = float(he_t[5]) if len(he_t) >= 6 else float("nan")
                he_m_eff = int(he_t[6]) if len(he_t) >= 7 else int(max(1, int(m_effective)))
                he_tr_k2 = float(he_t[7]) if len(he_t) >= 8 else float("nan")
                he_y_ky = float(he_t[8]) if len(he_t) >= 9 else float("nan")
                he_y_y = float(he_t[9]) if len(he_t) >= 10 else float("nan")
                he_lambda_k = float(he_t[10]) if len(he_t) >= 11 else float("nan")
                he_tr_k2_solve = float(he_t[11]) if len(he_t) >= 12 else float("nan")
                he_tr_k = float(he_t[12]) if len(he_t) >= 13 else float("nan")
                he_tr_p = float(he_t[13]) if len(he_t) >= 14 else float("nan")
                he_boundary_status = str(he_t[14]).strip() if len(he_t) >= 15 else ""

                if (not np.isfinite(he_lambda_k)) and np.isfinite(sigma_g2) and np.isfinite(sigma_e2) and sigma_g2 > 0.0:
                    he_lambda_k = float(sigma_e2 / sigma_g2)
                he_m_for_eq = int(max(1, he_m_eff if he_m_eff > 0 else int(m_effective)))
                he_lambda_eq = (
                    float(he_lambda_k * float(he_m_for_eq))
                    if np.isfinite(he_lambda_k) and he_lambda_k >= 0.0
                    else float("nan")
                )
                he_lambda_eq_sel = (
                    float(max(1e-8, he_lambda_eq))
                    if np.isfinite(he_lambda_eq) and he_lambda_eq >= 0.0
                    else float("nan")
                )
                he_lambda_k_sel = (
                    float(he_lambda_eq_sel / float(max(1, he_m_for_eq)))
                    if np.isfinite(he_lambda_eq_sel)
                    else float("nan")
                )
                he_nnls_projected = None
                if np.isfinite(sigma_g2) and np.isfinite(sigma_e2):
                    he_nnls_projected = bool((sigma_g2 <= 0.0) or (sigma_e2 <= 0.0))

                he_diag.update(
                    {
                        "sigma_g2": float(sigma_g2),
                        "sigma_e2": float(sigma_e2),
                        "h2": float(h2),
                        "he_converged": he_converged,
                        "he_iters": he_iters,
                        "he_rel_res": float(he_rel_res),
                        "lambda_he_raw_k": float(he_lambda_k),
                        "lambda_he_raw_equation": float(he_lambda_eq),
                        "lambda_he_lambda_k": float(he_lambda_k_sel),
                        "lambda_he_lambda_equation": float(he_lambda_eq_sel),
                        "trace_samples_used": int(he_trace_samples),
                        "thread_policy": str(he_thread_spec["policy"]),
                        "stage_blas_threads": int(he_thread_spec["blas_threads"]),
                        "stage_rayon_threads": int(he_thread_spec["rayon_threads"]),
                        "he_threads_arg": int(he_thread_spec["he_threads"]),
                        "he_tr_k2": float(he_tr_k2),
                        "he_tr_k2_solve": float(he_tr_k2_solve),
                        "he_y_ky": float(he_y_ky),
                        "he_y_y": float(he_y_y),
                        "he_tr_k": float(he_tr_k),
                        "he_tr_p": float(he_tr_p),
                        "he_nnls_projected": he_nnls_projected,
                        "he_boundary_status": str(he_boundary_status),
                        "he_error": "",
                    }
                )
        except Exception as he_ex:
            he_diag["he_error"] = str(he_ex)

    he_lam_eq_raw = float(he_diag.get("lambda_he_raw_equation", np.nan))
    he_lam_k_raw = float(he_diag.get("lambda_he_raw_k", np.nan))
    he_lam_eq = float(he_diag.get("lambda_he_lambda_equation", np.nan))
    he_lam_k = float(he_diag.get("lambda_he_lambda_k", np.nan))
    he_sigma_g2 = float(he_diag.get("sigma_g2", np.nan))
    he_sigma_e2 = float(he_diag.get("sigma_e2", np.nan))
    he_rel_res = float(he_diag.get("he_rel_res", np.nan))
    he_converged = he_diag.get("he_converged", None)
    he_nnls_projected = he_diag.get("he_nnls_projected", None)
    he_accept_rel_res_max = float(
        max(
            1e-6,
            float(cfg_use.get("he_accept_rel_res_max", 1e-2)),
        )
    )
    he_is_clamped = bool(
        np.isfinite(he_lam_eq_raw)
        and np.isfinite(he_lam_eq)
        and (he_lam_eq > (he_lam_eq_raw + 1e-12))
    )
    he_rel_ok = bool(np.isfinite(he_rel_res) and (he_rel_res <= he_accept_rel_res_max))
    if (not he_rel_ok) and (not np.isfinite(he_rel_res)) and (he_converged is not None):
        he_rel_ok = bool(he_converged)
    he_reject_reasons: list[str] = []
    if not (np.isfinite(he_lam_eq_raw) and he_lam_eq_raw > 0.0):
        he_reject_reasons.append("lambda_raw_nonpositive")
    if not (np.isfinite(he_sigma_g2) and he_sigma_g2 > 0.0):
        he_reject_reasons.append("sigma_g2_nonpositive")
    if not (np.isfinite(he_sigma_e2) and he_sigma_e2 > 0.0):
        he_reject_reasons.append("sigma_e2_nonpositive")
    if bool(he_nnls_projected):
        he_reject_reasons.append("nnls_projected")
    if not he_rel_ok:
        he_reject_reasons.append("rel_res_too_large")
    if he_is_clamped:
        he_reject_reasons.append("lambda_clamped_to_floor")
    he_ok = len(he_reject_reasons) == 0
    he_diag["he_accepted"] = bool(he_ok)
    he_diag["he_reject_reason"] = ",".join(he_reject_reasons)
    if lambda_auto_strategy in {"he_first", "he_only"}:
        if he_ok:
            return _merge_he_diag(
                {
                    "lambda_equation": float(he_lam_eq_raw),
                    "lambda_k": float(he_lam_k_raw) if np.isfinite(he_lam_k_raw) else float("nan"),
                    "n_sub": int(n_sub),
                    "repeats": 0,
                    "ok_repeats": 0,
                    "m_effective": int(m_effective),
                    "strategy": "he_primary",
                    "vc_method": "HE",
                    "vc_sigma_g2": float(he_sigma_g2),
                    "vc_sigma_e2": float(he_sigma_e2),
                    "vc_pve": float(he_diag.get("h2", np.nan)),
                }
            )
        if lambda_auto_strategy == "he_only":
            return _merge_he_diag(
                {
                    "lambda_equation": float("nan"),
                    "lambda_k": float("nan"),
                    "n_sub": int(n_sub),
                    "repeats": 0,
                    "ok_repeats": 0,
                    "m_effective": int(m_effective),
                    "strategy": "he_only_rejected",
                    "vc_method": "HE",
                    "vc_sigma_g2": float(he_sigma_g2),
                    "vc_sigma_e2": float(he_sigma_e2),
                    "vc_pve": float(he_diag.get("h2", np.nan)),
                }
            )

    def _method_payload(
        *,
        lambda_equation: float,
        lambda_k: float,
        n_sub_used: int,
        repeats_used: int,
        ok_repeats_used: int,
        strategy: str,
        vc_method: str,
        vc_sigma_g2: float,
        vc_sigma_e2: float,
        vc_pve: float,
    ) -> dict[str, typing.Any]:
        return _merge_he_diag(
            {
                "lambda_equation": float(lambda_equation),
                "lambda_k": float(lambda_k),
                "n_sub": int(n_sub_used),
                "repeats": int(repeats_used),
                "ok_repeats": int(ok_repeats_used),
                "m_effective": int(m_effective),
                "strategy": str(strategy),
                "vc_method": str(vc_method),
                "vc_sigma_g2": float(vc_sigma_g2),
                "vc_sigma_e2": float(vc_sigma_e2),
                "vc_pve": float(vc_pve),
            }
        )

    source_prefix = str(packed_ctx.get("source_prefix", "") or "").strip()
    site_keep_arg: np.ndarray | None = None
    site_keep_raw = packed_ctx.get("site_keep", None)
    if site_keep_raw is not None:
        site_keep_arr = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        maf_len = int(np.asarray(packed_ctx.get("maf", np.asarray([], dtype=np.float32))).reshape(-1).shape[0])
        if (maf_len <= 0) or (int(site_keep_arr.shape[0]) == int(maf_len)):
            site_keep_arg = site_keep_arr

    gblup_g_eps = float(max(0.0, float(np.finfo(np.float64).eps)))
    gblup_reml_low = float(cfg_use.get("gblup_reml_low", -6.0))
    gblup_reml_high = float(cfg_use.get("gblup_reml_high", 6.0))
    if (not np.isfinite(gblup_reml_low)) or (not np.isfinite(gblup_reml_high)) or (gblup_reml_low >= gblup_reml_high):
        gblup_reml_low, gblup_reml_high = -6.0, 6.0
    gblup_reml_max_iter = int(max(1, int(cfg_use.get("gblup_reml_max_iter", 50))))
    gblup_reml_tol = float(
        max(
            np.finfo(np.float64).eps,
            float(cfg_use.get("gblup_reml_tol", 1e-4)),
        )
    )
    gblup_block_rows = int(
        max(
            1,
            int(
                cfg_use.get(
                    "gblup_block_rows",
                    cfg_use.get("pcg_block_rows", cfg_use.get("snp_block_size", 4096)),
                )
            ),
        )
    )
    can_use_rust_gblup = bool(
        (_jxrs is not None)
        and hasattr(_jxrs, "gblup_reml_packed_bed")
        and (source_prefix != "")
    )

    def _run_rust_reml(
        local_idx: np.ndarray | None,
        *,
        force_grm_eig: bool,
    ) -> tuple[dict[str, float], str]:
        if not can_use_rust_gblup:
            raise RuntimeError(
                "Rust GBLUP packed REML path is unavailable for lambda-auto "
                "(missing source_prefix or gblup_reml_packed_bed binding)."
            )
        if local_idx is None:
            sub_abs = np.ascontiguousarray(train_abs, dtype=np.int64)
            y_sub = np.ascontiguousarray(y_vec, dtype=np.float64)
        else:
            sub_abs = np.ascontiguousarray(train_abs[local_idx], dtype=np.int64)
            y_sub = np.ascontiguousarray(y_vec[local_idx], dtype=np.float64)
        train_pred_local_empty = np.zeros((0,), dtype=np.int64)

        old_force = os.environ.get("JX_GBLUP_PACKED_FORCE_GRM_EIG", None)
        try:
            if force_grm_eig:
                os.environ["JX_GBLUP_PACKED_FORCE_GRM_EIG"] = "1"
            elif old_force is None:
                os.environ.pop("JX_GBLUP_PACKED_FORCE_GRM_EIG", None)

            try:
                g_out = _jxrs.gblup_reml_packed_bed(  # type: ignore[union-attr]
                    source_prefix,
                    sub_abs,
                    y_sub,
                    None,
                    train_pred_local_empty,
                    site_keep_arg,
                    float(gblup_g_eps),
                    float(gblup_reml_low),
                    float(gblup_reml_high),
                    int(gblup_reml_max_iter),
                    float(gblup_reml_tol),
                    int(gblup_block_rows),
                    int(max(1, int(n_jobs))),
                    True,
                    True,
                )
            except TypeError:
                try:
                    g_out = _jxrs.gblup_reml_packed_bed(  # type: ignore[union-attr]
                        source_prefix,
                        sub_abs,
                        y_sub,
                        None,
                        train_pred_local_empty,
                        site_keep_arg,
                        float(gblup_g_eps),
                        float(gblup_reml_low),
                        float(gblup_reml_high),
                        int(gblup_reml_max_iter),
                        float(gblup_reml_tol),
                        int(gblup_block_rows),
                        int(max(1, int(n_jobs))),
                        True,
                    )
                except TypeError:
                    g_out = _jxrs.gblup_reml_packed_bed(  # type: ignore[union-attr]
                        source_prefix,
                        sub_abs,
                        y_sub,
                        None,
                        train_pred_local_empty,
                        site_keep_arg,
                        float(gblup_g_eps),
                        float(gblup_reml_low),
                        float(gblup_reml_high),
                        int(gblup_reml_max_iter),
                        float(gblup_reml_tol),
                        int(gblup_block_rows),
                        int(max(1, int(n_jobs))),
                    )
        finally:
            if old_force is None:
                os.environ.pop("JX_GBLUP_PACKED_FORCE_GRM_EIG", None)
            else:
                os.environ["JX_GBLUP_PACKED_FORCE_GRM_EIG"] = str(old_force)

        g_t = tuple(g_out)
        if len(g_t) >= 11:
            _pred_train, _pred_test, pve, lam_k, _ml, _reml, evd_backend, _evd_elapsed, _m_eff, sigma_g2, sigma_e2 = g_t[:11]
        elif len(g_t) >= 9:
            _pred_train, _pred_test, pve, lam_k, _ml, _reml, evd_backend, _evd_elapsed, _m_eff = g_t[:9]
            sigma_g2 = float("nan")
            sigma_e2 = float("nan")
        else:
            raise RuntimeError(
                "gblup_reml_packed_bed returned unexpected payload size "
                f"for lambda-auto: {len(g_t)}"
            )

        lam_sub_k = float(lam_k)
        lam_sub_eq = (
            float(lam_sub_k * m_scale)
            if (np.isfinite(lam_sub_k) and lam_sub_k > 0.0)
            else float("nan")
        )
        fit = {
            "lambda_k": float(lam_sub_k),
            "lambda_eq": float(lam_sub_eq),
            "sigma_g2": float(sigma_g2),
            "sigma_e2": float(sigma_e2),
            "pve": float(pve),
        }
        return fit, str(evd_backend)

    def _fit_once_grm(local_idx: np.ndarray) -> dict[str, float]:
        if can_use_rust_gblup:
            fit_sub, _evd_backend = _run_rust_reml(
                local_idx,
                force_grm_eig=True,
            )
            return fit_sub

        sub_abs = np.ascontiguousarray(train_abs[local_idx], dtype=np.int64)
        y_sub = np.ascontiguousarray(y_vec[local_idx], dtype=np.float64)
        grm_sub = _build_gblup_cv_grm_once(
            train_snp=None,
            packed_ctx=packed_ctx,
            train_sample_indices=sub_abs,
            n_jobs=max(1, int(n_jobs)),
        )
        if grm_sub is None:
            raise RuntimeError("Packed GRM builder returned None.")
        fit_sub = _fit_gblup_reml_from_grm(y_sub, grm_sub)
        lam_sub_k = float(fit_sub.get("lambda_reml", np.nan))
        lam_sub_eq = (
            float(lam_sub_k * m_scale)
            if (np.isfinite(lam_sub_k) and lam_sub_k > 0.0)
            else float("nan")
        )
        return {
            "lambda_k": float(lam_sub_k),
            "lambda_eq": float(lam_sub_eq),
            "sigma_g2": float(fit_sub.get("sigma_g2", np.nan)),
            "sigma_e2": float(fit_sub.get("sigma_e2", np.nan)),
            "pve": float(fit_sub.get("pve", np.nan)),
        }

    def _recover_lambda_from_model(model: typing.Any) -> tuple[float, float, float]:
        sigma_g2 = float("nan")
        sigma_e2 = float("nan")
        lambda_k = float("nan")
        try:
            rtv_invr = float(getattr(model, "rTV_invr", np.nan))
            n_model = int(getattr(model, "n", n_train))
            p_model = int(getattr(model, "p", 1))
            n_eff_model = int(max(1, n_model - p_model))
            svals = np.asarray(getattr(model, "S", []), dtype=float).reshape(-1)
            v_inv = np.asarray(getattr(model, "V_inv", []), dtype=float).reshape(-1)
            if svals.size > 0 and v_inv.size == svals.size:
                eps_v = float(np.finfo(np.float64).eps)
                valid = np.isfinite(svals) & np.isfinite(v_inv) & (v_inv > eps_v)
                if np.any(valid):
                    lam_cand = (1.0 / v_inv[valid]) - svals[valid]
                    lam_cand = lam_cand[np.isfinite(lam_cand) & (lam_cand >= 0.0)]
                    if lam_cand.size > 0:
                        lambda_k = float(np.median(lam_cand))
            if np.isfinite(rtv_invr) and rtv_invr > 0.0:
                sigma_g2 = float(rtv_invr / float(max(1, n_eff_model)))
                if np.isfinite(lambda_k) and lambda_k >= 0.0:
                    sigma_e2 = float(lambda_k * sigma_g2)
            if (
                (not np.isfinite(lambda_k))
                and np.isfinite(sigma_g2)
                and np.isfinite(sigma_e2)
                and sigma_g2 > 0.0
            ):
                lambda_k = float(sigma_e2 / sigma_g2)
        except Exception:
            pass
        return float(lambda_k), float(sigma_g2), float(sigma_e2)

    def _fit_full_fast_reml() -> dict[str, float]:
        if can_use_rust_gblup:
            fit_fast, _evd_backend = _run_rust_reml(
                None,
                force_grm_eig=False,
            )
            return fit_fast

        y_fit = np.ascontiguousarray(y_vec.reshape(-1, 1), dtype=np.float64)
        model = MLMBLUP(
            y_fit,
            packed_ctx,
            kinship=1,
            sample_indices=np.ascontiguousarray(train_abs, dtype=np.int64),
            force_fast=True,
        )
        lam_k, sigma_g2, sigma_e2 = _recover_lambda_from_model(model)
        lam_eq = (
            float(lam_k * m_scale)
            if (np.isfinite(lam_k) and lam_k > 0.0)
            else float("nan")
        )
        return {
            "lambda_k": float(lam_k),
            "lambda_eq": float(lam_eq),
            "sigma_g2": float(sigma_g2),
            "sigma_e2": float(sigma_e2),
            "pve": float(getattr(model, "pve", np.nan)),
        }

    large_cutoff = int(max(2, int(cfg_use.get("lambda_large_cutoff", 15_000))))
    many_samples = bool(int(n_train) >= int(large_cutoff))
    many_snps = bool(int(m_effective) >= int(large_cutoff))
    prefer_subsample = bool(many_samples and many_snps)
    prefer_fast = bool(many_samples and (not many_snps))

    # 1) GRM+REML (small-sample bucket under 15k threshold)
    if not prefer_subsample and not prefer_fast:
        fit_one = {
            "lambda_k": float("nan"),
            "lambda_eq": float("nan"),
            "sigma_g2": float("nan"),
            "sigma_e2": float("nan"),
            "pve": float("nan"),
        }
        try:
            _emit(
                "pcg_lambda_vc_start",
                vc_method="GRMreml",
                strategy="grm_reml",
            )
            fit_one = _fit_once_grm(np.arange(n_train, dtype=np.int64))
        except Exception:
            pass
        finally:
            _emit(
                "pcg_lambda_vc_end",
                vc_method="GRMreml",
                strategy="grm_reml",
            )
        lam_ok = bool(np.isfinite(float(fit_one["lambda_eq"])) and float(fit_one["lambda_eq"]) > 0.0)
        if lam_ok:
            return _method_payload(
                lambda_equation=float(fit_one["lambda_eq"]),
                lambda_k=float(fit_one["lambda_k"]),
                n_sub_used=int(n_train),
                repeats_used=1,
                ok_repeats_used=1,
                strategy="grm_reml",
                vc_method="GRMreml",
                vc_sigma_g2=float(fit_one["sigma_g2"]),
                vc_sigma_e2=float(fit_one["sigma_e2"]),
                vc_pve=float(fit_one["pve"]),
            )

    # 2) FaST+REML (many-sample + few-SNP bucket under 15k threshold)
    if prefer_fast:
        fit_fast = {
            "lambda_k": float("nan"),
            "lambda_eq": float("nan"),
            "sigma_g2": float("nan"),
            "sigma_e2": float("nan"),
            "pve": float("nan"),
        }
        try:
            _emit(
                "pcg_lambda_vc_start",
                vc_method="FaSTreml",
                strategy="fast_reml",
            )
            fit_fast = _fit_full_fast_reml()
        except Exception:
            pass
        finally:
            _emit(
                "pcg_lambda_vc_end",
                vc_method="FaSTreml",
                strategy="fast_reml",
            )
        lam_ok = bool(np.isfinite(float(fit_fast["lambda_eq"])) and float(fit_fast["lambda_eq"]) > 0.0)
        if lam_ok:
            return _method_payload(
                lambda_equation=float(fit_fast["lambda_eq"]),
                lambda_k=float(fit_fast["lambda_k"]),
                n_sub_used=int(n_train),
                repeats_used=1,
                ok_repeats_used=1,
                strategy="fast_reml",
                vc_method="FaSTreml",
                vc_sigma_g2=float(fit_fast["sigma_g2"]),
                vc_sigma_e2=float(fit_fast["sigma_e2"]),
                vc_pve=float(fit_fast["pve"]),
            )

    # 3) Subsample REML (many-sample + many-SNP)
    _emit(
        "pcg_lambda_subsample_start",
        total=int(repeats),
        n_sub=int(n_sub),
        n_train=int(n_train),
        strategy="subsample_reml",
    )
    log10_vals: list[float] = []
    sigma_g2_vals: list[float] = []
    sigma_e2_vals: list[float] = []
    pve_vals: list[float] = []
    lam_k_vals: list[float] = []
    for rep in range(repeats):
        if n_sub >= n_train:
            local_idx = np.arange(n_train, dtype=np.int64)
        else:
            local_idx = np.asarray(
                rng.choice(n_train, size=n_sub, replace=False),
                dtype=np.int64,
            )
            local_idx.sort()
        fit_one = {
            "lambda_k": float("nan"),
            "lambda_eq": float("nan"),
            "sigma_g2": float("nan"),
            "sigma_e2": float("nan"),
            "pve": float("nan"),
        }
        try:
            fit_one = _fit_once_grm(local_idx)
        except Exception:
            pass
        lam_sub_k = float(fit_one["lambda_k"])
        lam_sub_eq = float(fit_one["lambda_eq"])
        if np.isfinite(lam_sub_eq) and lam_sub_eq > 0.0:
            log10_vals.append(float(np.log10(lam_sub_eq)))
        if np.isfinite(lam_sub_k) and lam_sub_k > 0.0:
            lam_k_vals.append(float(lam_sub_k))
        if np.isfinite(float(fit_one["sigma_g2"])):
            sigma_g2_vals.append(float(fit_one["sigma_g2"]))
        if np.isfinite(float(fit_one["sigma_e2"])):
            sigma_e2_vals.append(float(fit_one["sigma_e2"]))
        if np.isfinite(float(fit_one["pve"])):
            pve_vals.append(float(fit_one["pve"]))
        _emit(
            "pcg_lambda_subsample_iter",
            iter=int(rep + 1),
            total=int(repeats),
            lambda_equation=float(lam_sub_eq),
            lambda_k=float(lam_sub_k),
            m_effective=int(m_effective),
            strategy="subsample_reml",
        )

    if len(log10_vals) == 0:
        _emit(
            "pcg_lambda_subsample_end",
            repeats=int(repeats),
            ok_repeats=0,
            lambda_equation=float("nan"),
            m_effective=int(m_effective),
            strategy="subsample_reml",
        )
        he_lam_eq = float(he_diag.get("lambda_he_raw_equation", np.nan))
        he_lam_k = float(he_diag.get("lambda_he_raw_k", np.nan))
        if np.isfinite(he_lam_eq) and he_lam_eq > 0.0:
            return _method_payload(
                lambda_equation=float(he_lam_eq),
                lambda_k=float(he_lam_k) if np.isfinite(he_lam_k) else float("nan"),
                n_sub_used=int(n_sub),
                repeats_used=int(repeats),
                ok_repeats_used=0,
                strategy="he_fallback_subsample_reml_failed",
                vc_method="HE",
                vc_sigma_g2=float(he_diag.get("sigma_g2", np.nan)),
                vc_sigma_e2=float(he_diag.get("sigma_e2", np.nan)),
                vc_pve=float(he_diag.get("h2", np.nan)),
            )
        return _method_payload(
            lambda_equation=float("nan"),
            lambda_k=float("nan"),
            n_sub_used=int(n_sub),
            repeats_used=int(repeats),
            ok_repeats_used=0,
            strategy="subsample_reml_failed",
            vc_method="Sub/REML",
            vc_sigma_g2=float("nan"),
            vc_sigma_e2=float("nan"),
            vc_pve=float("nan"),
        )

    lam_eq = float(10.0 ** float(np.median(np.asarray(log10_vals, dtype=np.float64))))
    lam_k = float(lam_eq / m_scale) if m_scale > 0.0 else float("nan")
    vc_sigma_g2 = (
        float(np.median(np.asarray(sigma_g2_vals, dtype=np.float64)))
        if len(sigma_g2_vals) > 0
        else float("nan")
    )
    vc_sigma_e2 = (
        float(np.median(np.asarray(sigma_e2_vals, dtype=np.float64)))
        if len(sigma_e2_vals) > 0
        else float("nan")
    )
    vc_pve = (
        float(np.median(np.asarray(pve_vals, dtype=np.float64)))
        if len(pve_vals) > 0
        else float("nan")
    )
    _emit(
        "pcg_lambda_subsample_end",
        repeats=int(repeats),
        ok_repeats=int(len(log10_vals)),
        lambda_equation=float(lam_eq),
        m_effective=int(m_effective),
        strategy="subsample_reml",
    )
    return _method_payload(
        lambda_equation=float(lam_eq),
        lambda_k=float(lam_k),
        n_sub_used=int(n_sub),
        repeats_used=int(repeats),
        ok_repeats_used=int(len(log10_vals)),
        strategy="subsample_reml",
        vc_method="Sub/REML",
        vc_sigma_g2=float(vc_sigma_g2),
        vc_sigma_e2=float(vc_sigma_e2),
        vc_pve=float(vc_pve),
    )


def _cfg_truthy(value: typing.Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text == "":
        return bool(default)
    if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return bool(default)


def _estimate_bayes_auto_r2_from_blup(
    *,
    y: np.ndarray,
    Xtrain: typing.Any,
    packed_train_indices: np.ndarray | None = None,
    cfg: dict[str, typing.Any] | None = None,
    seed_offset: int = 0,
) -> tuple[float, str, int, int]:
    """
    Estimate Bayes auto-r2 (BLUP PVE) with optional large-n subsampling.

    Returns
    -------
    r2_blup : float
        Estimated BLUP PVE before clipping.
    source : str
        Estimation source label.
    n_used : int
        Number of samples used in BLUP fit.
    n_total : int
        Total training sample size.
    """
    y_vec = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1), dtype=np.float64)
    n_total = int(y_vec.shape[0])
    if n_total <= 1:
        return float("nan"), "blup_auto", int(max(0, n_total)), int(max(0, n_total))

    cfg_use = dict(cfg or {})
    sub_trigger_n = int(max(2, int(cfg_use.get("subsample_min_n", 10000))))
    sub_min = int(max(16, int(cfg_use.get("subsample_min_samples", _RRBLUP_LAMBDA_SUBSAMPLE_MIN_N))))
    sub_max = int(max(sub_min, int(cfg_use.get("subsample_max_n", _RRBLUP_LAMBDA_SUBSAMPLE_MAX_N))))
    sub_target = int(max(sub_min, int(cfg_use.get("subsample_n", _RRBLUP_LAMBDA_SUBSAMPLE_MAX_N))))
    rep_cfg = int(cfg_use.get("subsample_repeats", _RRBLUP_LAMBDA_SUBSAMPLE_REPEATS))
    repeats = int(min(20, max(5, rep_cfg)))
    sub_seed = int(cfg_use.get("subsample_seed", 42))

    is_packed = _looks_like_packed_payload(Xtrain)
    packed_train: dict[str, typing.Any] | None = None
    base_abs: np.ndarray | None = None
    dense: np.ndarray | None = None
    if is_packed:
        packed_train = typing.cast(dict[str, typing.Any], Xtrain)
        n_samples_full = int(packed_train["n_samples"])
        if packed_train_indices is None:
            if n_samples_full != n_total:
                raise ValueError(
                    "Packed Bayes auto-r2 requires packed_train_indices when "
                    "len(y) differs from packed n_samples."
                )
            base_abs = np.ascontiguousarray(np.arange(n_samples_full, dtype=np.int64), dtype=np.int64)
        else:
            base_abs = np.ascontiguousarray(
                np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
            if int(base_abs.shape[0]) != n_total:
                raise ValueError(
                    "Packed Bayes auto-r2 index length mismatch: "
                    f"indices={base_abs.shape[0]}, y={n_total}."
                )
    else:
        dense = np.asarray(Xtrain)
        if dense.ndim != 2:
            raise ValueError(f"Bayes auto-r2 expects 2D dense genotype matrix, got {dense.shape}")
        if int(dense.shape[1]) != n_total:
            raise ValueError(
                f"Bayes auto-r2 dense genotype sample mismatch: X n={dense.shape[1]}, y n={n_total}."
            )

    def _fit_once(local_idx: np.ndarray | None) -> float:
        if local_idx is None:
            y_fit = np.ascontiguousarray(y_vec.reshape(-1, 1), dtype=np.float64)
        else:
            y_fit = np.ascontiguousarray(y_vec[local_idx].reshape(-1, 1), dtype=np.float64)

        if is_packed:
            assert packed_train is not None
            assert base_abs is not None
            fit_abs = (
                base_abs
                if local_idx is None
                else np.ascontiguousarray(base_abs[local_idx], dtype=np.int64)
            )
            model = MLMBLUP(y_fit, packed_train, kinship=1, sample_indices=fit_abs)
        else:
            assert dense is not None
            x_fit = dense if local_idx is None else np.asarray(dense)[:, local_idx]
            model = MLMBLUP(y_fit, x_fit, kinship=1)
        return float(model.pve)

    use_subsample = bool(n_total > sub_trigger_n and sub_target < n_total)
    if (not use_subsample) or (sub_target <= 0):
        r2 = float(_fit_once(None))
        return r2, "blup_auto", int(n_total), int(n_total)

    n_sub = int(min(max(1, n_total), min(sub_max, max(sub_min, sub_target))))
    if n_sub >= n_total:
        r2 = float(_fit_once(None))
        return r2, "blup_auto", int(n_total), int(n_total)

    rng = np.random.default_rng(int(sub_seed) + int(seed_offset) + int(n_total) * 17)
    vals: list[float] = []
    for _rep in range(int(repeats)):
        local_idx = np.asarray(
            rng.choice(int(n_total), size=int(n_sub), replace=False),
            dtype=np.int64,
        )
        local_idx.sort()
        try:
            v = float(_fit_once(local_idx))
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            vals.append(float(v))

    if len(vals) == 0:
        r2 = float(_fit_once(None))
        return r2, "blup_auto_fallback", int(n_total), int(n_total)

    r2 = float(np.median(np.asarray(vals, dtype=np.float64)))
    return r2, f"blup_subsample(n={int(n_sub)},rep={len(vals)}/{int(repeats)})", int(n_sub), int(n_total)


def _build_rrblup_adamw_grid(
    cfg: dict[str, typing.Any] | None,
) -> list[tuple[float, float]]:
    cfg_use = cfg or {}
    lam0 = float(cfg_use.get("lambda_value", 1.0))
    lr0 = float(cfg_use.get("lr", 1e-2))
    if (not np.isfinite(lam0)) or lam0 < 0.0:
        lam0 = 1.0
    if (not np.isfinite(lr0)) or lr0 <= 0.0:
        lr0 = 1e-2
    grid_size = int(max(1, min(4, int(cfg_use.get("grid_size", 4)))))
    lam_anchor = float(max(lam0, 1e-8))
    cands = [
        (lam_anchor, lr0),
        (lam_anchor * 0.5, lr0),
        (lam_anchor * 2.0, lr0),
        (lam_anchor * 4.0, lr0),
    ]
    out: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for lam, lr in cands:
        lam_use = float(max(0.0, lam))
        lr_use = float(max(np.finfo(np.float32).eps, lr))
        key = (float(f"{lam_use:.12g}"), float(f"{lr_use:.12g}"))
        if key in seen:
            continue
        seen.add(key)
        out.append((lam_use, lr_use))
        if len(out) >= grid_size:
            break
    if len(out) == 0:
        out = [(float(max(0.0, lam0)), float(max(np.finfo(np.float32).eps, lr0)))]
    return out


def _build_rrblup_validation_indices(
    *,
    n_train: int,
    cfg: dict[str, typing.Any] | None,
) -> np.ndarray | None:
    cfg_use = cfg or {}
    n = int(max(0, int(n_train)))
    if n <= 0:
        return None
    val_frac = float(cfg_use.get("es_val_frac", 0.1))
    val_frac = min(0.45, max(0.0, val_frac))
    val_min = int(max(1, int(cfg_use.get("es_val_min", 64))))
    min_train = int(max(8, int(cfg_use.get("es_min_train", 128))))
    if n <= min_train:
        return None
    if val_frac <= 0.0 and val_min <= 0:
        return None
    val_n = int(round(float(n) * float(val_frac)))
    val_n = max(val_min, val_n)
    val_n = min(val_n, max(1, n - min_train))
    if val_n <= 0 or val_n >= n:
        return None
    seed = int(cfg_use.get("seed", 42))
    grid_seed = int(cfg_use.get("grid_seed", seed))
    rng = np.random.default_rng(grid_seed + 7919)
    idx = np.ascontiguousarray(rng.permutation(n)[:val_n], dtype=np.int64)
    idx.sort()
    return idx


def _adamw_update_inplace_f32(
    *,
    param: np.ndarray,
    grad: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    if int(step) <= 0:
        raise ValueError("AdamW step must be > 0.")
    grad32 = np.asarray(grad, dtype=np.float32)
    m1 *= np.float32(beta1)
    m1 += np.float32(1.0 - beta1) * grad32
    m2 *= np.float32(beta2)
    m2 += np.float32(1.0 - beta2) * (grad32 * grad32)

    b1_corr = 1.0 - (float(beta1) ** float(step))
    b2_corr = 1.0 - (float(beta2) ** float(step))
    if b1_corr <= 0.0 or b2_corr <= 0.0:
        return
    m_hat = m1 / np.float32(b1_corr)
    v_hat = m2 / np.float32(b2_corr)
    denom = np.sqrt(v_hat, dtype=np.float32) + np.float32(eps)
    param -= np.float32(lr) * (m_hat / denom)
    if float(weight_decay) > 0.0:
        decay = max(0.0, 1.0 - float(lr) * float(weight_decay))
        param *= np.float32(decay)


def _ensure_packed_row_flip_cached(
    packed_ctx: dict[str, typing.Any],
) -> np.ndarray:
    row_flip_raw = packed_ctx.get("row_flip", None)
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    n_samples = int(packed_ctx["n_samples"])
    if row_flip_raw is None:
        if _jxrs is None or (not hasattr(_jxrs, "bed_packed_row_flip_mask")):
            raise RuntimeError(
                "Rust packed row-flip kernel is unavailable. Rebuild/install JanusX extension."
            )
        row_flip = np.asarray(
            _jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        )
        row_flip = np.ascontiguousarray(row_flip.reshape(-1), dtype=np.bool_)
        packed_ctx["row_flip"] = row_flip
    else:
        row_flip = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
    if int(row_flip.shape[0]) != int(packed.shape[0]):
        raise ValueError(
            "Packed payload mismatch: row_flip length does not match packed SNP rows."
        )
    return row_flip


def _decode_packed_block_standardized(
    *,
    packed_ctx: dict[str, typing.Any],
    row_idx: np.ndarray,
    sample_indices: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
) -> np.ndarray:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_decode_rows_f32")):
        raise RuntimeError(
            "Rust packed BED decode helper is unavailable. Rebuild/install JanusX extension."
        )
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    n_samples = int(packed_ctx["n_samples"])
    row_flip = _ensure_packed_row_flip_cached(packed_ctx)
    ridx = np.ascontiguousarray(np.asarray(row_idx, dtype=np.int64).reshape(-1))
    if _packed_ctx_is_lazy_full(packed_ctx):
        active_row_idx = _packed_ctx_active_row_idx(packed_ctx)
        ridx_decode = np.ascontiguousarray(active_row_idx[ridx], dtype=np.int64)
    else:
        ridx_decode = ridx
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1))
    blk = _jxrs.bed_packed_decode_rows_f32(
        packed,
        int(n_samples),
        ridx_decode,
        row_flip,
        maf,
        sidx,
    )
    x = np.ascontiguousarray(np.asarray(blk, dtype=np.float32), dtype=np.float32)
    mu = np.ascontiguousarray(np.asarray(row_mean[ridx], dtype=np.float32).reshape(-1, 1))
    inv_sd = np.ascontiguousarray(np.asarray(row_inv_sd[ridx], dtype=np.float32).reshape(-1, 1))
    x -= mu
    x *= inv_sd
    return x


def _ensure_packed_standard_stats_cached(
    packed_ctx: dict[str, typing.Any],
) -> tuple[np.ndarray, np.ndarray]:
    mean_raw = packed_ctx.get("__std_row_mean__", None)
    inv_raw = packed_ctx.get("__std_row_inv_sd__", None)
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    active_row_idx = _packed_ctx_active_row_idx(packed_ctx)
    m = int(active_row_idx.shape[0])
    if mean_raw is not None and inv_raw is not None:
        mean = np.ascontiguousarray(np.asarray(mean_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
        inv = np.ascontiguousarray(np.asarray(inv_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
        if int(mean.shape[0]) == m and int(inv.shape[0]) == m:
            return mean, inv

    maf_full = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    if _packed_ctx_is_lazy_full(packed_ctx):
        if int(maf_full.shape[0]) != int(packed.shape[0]):
            raise ValueError("Packed context mismatch: maf length != packed rows.")
        maf = np.ascontiguousarray(maf_full[active_row_idx], dtype=np.float32)
    else:
        maf = maf_full
        if int(maf.shape[0]) != m:
            raise ValueError("Packed context mismatch: maf length != packed rows.")
    n_samples = int(packed_ctx["n_samples"])
    if n_samples <= 0:
        raise ValueError("Packed context invalid: n_samples must be > 0.")

    mean_f64: np.ndarray
    var_f64: np.ndarray
    use_rust_stats = bool(
        (not _packed_ctx_is_lazy_full(packed_ctx))
        and (_jxrs is not None)
        and hasattr(_jxrs, "bed_packed_decode_stats_f64")
    )
    if use_rust_stats:
        row_flip = _ensure_packed_row_flip_cached(packed_ctx)
        sum_rows = np.zeros((m,), dtype=np.float64)
        sq_rows = np.zeros((m,), dtype=np.float64)
        step = int(max(1, min(8192, n_samples)))
        threads_raw = os.getenv("JX_MLM_RUST_THREADS", "").strip()
        try:
            rust_threads = int(threads_raw) if threads_raw != "" else 0
        except Exception:
            rust_threads = 0
        for st in range(0, n_samples, step):
            ed = min(st + step, n_samples)
            sidx = np.ascontiguousarray(np.arange(st, ed, dtype=np.int64), dtype=np.int64)
            _blk, sum_blk, sq_blk = _jxrs.bed_packed_decode_stats_f64(  # type: ignore[union-attr]
                packed,
                int(n_samples),
                row_flip,
                maf,
                sidx,
                threads=int(rust_threads),
            )
            sum_rows += np.asarray(sum_blk, dtype=np.float64).reshape(-1)
            sq_rows += np.asarray(sq_blk, dtype=np.float64).reshape(-1)
        mean_f64 = np.asarray(sum_rows / float(n_samples), dtype=np.float64)
        var_f64 = np.asarray((sq_rows / float(n_samples)) - mean_f64 * mean_f64, dtype=np.float64)
    else:
        mean_f64 = np.asarray(2.0 * maf, dtype=np.float64)
        var_f64 = np.asarray(2.0 * maf * (1.0 - maf), dtype=np.float64)

    # Match dense GS standardization exactly:
    #   z = (x - mean) / (std + 1e-6)
    # where std = sqrt(var) computed on the full cohort.
    var_f64 = np.maximum(var_f64, 0.0)
    std_f64 = np.sqrt(var_f64, dtype=np.float64)
    inv_f64 = 1.0 / (std_f64 + 1e-6)
    inv_f64 = np.asarray(np.where(np.isfinite(inv_f64), inv_f64, 0.0), dtype=np.float64)
    mean = np.ascontiguousarray(np.asarray(mean_f64, dtype=np.float32), dtype=np.float32)
    inv = np.ascontiguousarray(np.asarray(inv_f64, dtype=np.float32), dtype=np.float32)
    packed_ctx["__std_row_mean__"] = mean
    packed_ctx["__std_row_inv_sd__"] = inv
    return mean, inv


def _decode_packed_subset_to_dense_standardized(
    *,
    packed_ctx: dict[str, typing.Any],
    sample_indices: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    row_block_size: int = 2048,
    out_dtype: typing.Any = np.float32,
) -> np.ndarray:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_decode_rows_f32")):
        raise RuntimeError(
            "Rust packed BED decode helper is unavailable. Rebuild/install JanusX extension."
        )
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    m = int(packed.shape[0])
    n_samples = int(packed_ctx["n_samples"])
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
    if int(maf.shape[0]) != m:
        raise ValueError("Packed context mismatch: maf length != packed rows.")
    row_flip = _ensure_packed_row_flip_cached(packed_ctx)
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    n_out = int(sidx.shape[0])
    dtype_out = np.dtype(out_dtype)
    if dtype_out != np.dtype(np.float64):
        dtype_out = np.dtype(np.float32)
    out = np.empty((m, n_out), dtype=dtype_out)
    blk_step = int(max(1, int(row_block_size)))
    for st in range(0, m, blk_step):
        ed = min(st + blk_step, m)
        ridx = np.ascontiguousarray(np.arange(st, ed, dtype=np.int64), dtype=np.int64)
        blk = _jxrs.bed_packed_decode_rows_f32(  # type: ignore[union-attr]
            packed,
            int(n_samples),
            ridx,
            row_flip,
            maf,
            sidx,
        )
        x = np.ascontiguousarray(np.asarray(blk, dtype=dtype_out), dtype=dtype_out)
        mu = np.ascontiguousarray(
            np.asarray(row_mean[st:ed], dtype=dtype_out).reshape(-1, 1),
            dtype=dtype_out,
        )
        inv_sd = np.ascontiguousarray(
            np.asarray(row_inv_sd[st:ed], dtype=dtype_out).reshape(-1, 1),
            dtype=dtype_out,
        )
        x -= mu
        x *= inv_sd
        out[st:ed, :] = x
    return np.ascontiguousarray(out, dtype=dtype_out)


def _predict_rrblup_dense_from_beta(
    *,
    X: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    snp_block_size: int,
) -> np.ndarray:
    mat = np.asarray(X, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"Dense rrBLUP predict expects 2D matrix, got {mat.shape}.")
    m, n = int(mat.shape[0]), int(mat.shape[1])
    b = np.asarray(beta, dtype=np.float32).reshape(-1)
    if int(b.shape[0]) != m:
        raise ValueError(f"beta length mismatch: got {b.shape[0]}, expected {m}.")
    pred = np.full((n,), np.float32(alpha), dtype=np.float32)
    for st, ed, _ in _iter_row_blocks(m, snp_block_size):
        blk = np.ascontiguousarray(mat[st:ed], dtype=np.float32)
        pred += np.asarray(blk.T @ b[st:ed], dtype=np.float32).reshape(-1)
    return np.asarray(pred, dtype=np.float64).reshape(-1, 1)


def _predict_rrblup_packed_from_beta(
    *,
    packed_ctx: dict[str, typing.Any],
    sample_indices: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    snp_block_size: int,
    sample_chunk_size: int,
) -> np.ndarray:
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1))
    n_out = int(sidx.shape[0])
    if n_out == 0:
        return np.zeros((0, 1), dtype=np.float64)

    m = _packed_ctx_active_rows(packed_ctx)
    b = np.asarray(beta, dtype=np.float32).reshape(-1)
    if int(b.shape[0]) != m:
        raise ValueError(f"beta length mismatch: got {b.shape[0]}, expected {m}.")

    out = np.zeros((n_out,), dtype=np.float32)
    row_blocks = _iter_row_blocks(m, snp_block_size)
    chunk = max(1, int(sample_chunk_size))
    for cst in range(0, n_out, chunk):
        ced = min(cst + chunk, n_out)
        blk_idx = np.ascontiguousarray(sidx[cst:ced], dtype=np.int64)
        pred = np.full((int(blk_idx.shape[0]),), np.float32(alpha), dtype=np.float32)
        for st, ed, ridx in row_blocks:
            x = _decode_packed_block_standardized(
                packed_ctx=packed_ctx,
                row_idx=ridx,
                sample_indices=blk_idx,
                row_mean=row_mean,
                row_inv_sd=row_inv_sd,
            )
            pred += np.asarray(x.T @ b[st:ed], dtype=np.float32).reshape(-1)
        out[cst:ced] = pred
    return np.asarray(out, dtype=np.float64).reshape(-1, 1)


def _predict_rrblup_packed_raw_from_beta(
    *,
    packed_ctx: dict[str, typing.Any],
    sample_indices: np.ndarray,
    alpha: float,
    beta: np.ndarray,
    row_block_size: int,
    sample_chunk_size: int,
) -> np.ndarray:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_decode_rows_f32")):
        raise RuntimeError(
            "Rust packed BED decode helper is unavailable. Rebuild/install JanusX extension."
        )
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    n_out = int(sidx.shape[0])
    if n_out == 0:
        return np.zeros((0, 1), dtype=np.float64)

    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8), dtype=np.uint8)
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
    n_samples = int(packed_ctx["n_samples"])
    row_flip = _ensure_packed_row_flip_cached(packed_ctx)
    m = int(packed.shape[0])
    b = np.ascontiguousarray(np.asarray(beta, dtype=np.float32).reshape(-1), dtype=np.float32)
    if int(b.shape[0]) != m:
        raise ValueError(f"beta length mismatch: got {b.shape[0]}, expected {m}.")

    out = np.zeros((n_out,), dtype=np.float32)
    row_step = int(max(1, int(row_block_size)))
    sample_step = int(max(1, int(sample_chunk_size)))
    for cst in range(0, n_out, sample_step):
        ced = min(cst + sample_step, n_out)
        blk_idx = np.ascontiguousarray(sidx[cst:ced], dtype=np.int64)
        pred = np.full((int(blk_idx.shape[0]),), np.float32(alpha), dtype=np.float32)
        for st in range(0, m, row_step):
            ed = min(st + row_step, m)
            ridx = np.ascontiguousarray(np.arange(st, ed, dtype=np.int64), dtype=np.int64)
            x_blk = _jxrs.bed_packed_decode_rows_f32(  # type: ignore[union-attr]
                packed,
                int(n_samples),
                ridx,
                row_flip,
                maf,
                blk_idx,
            )
            x = np.ascontiguousarray(np.asarray(x_blk, dtype=np.float32), dtype=np.float32)
            pred += np.asarray(x.T @ b[st:ed], dtype=np.float32).reshape(-1)
        out[cst:ced] = pred
    return np.asarray(out, dtype=np.float64).reshape(-1, 1)


def _kernel_projection_beta_from_dense(
    *,
    marker_by_sample: np.ndarray,
    alpha: np.ndarray,
    coef: float = 1.0,
) -> np.ndarray | None:
    try:
        m = np.asarray(marker_by_sample, dtype=np.float64)
        a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if m.ndim != 2:
        return None
    if int(m.shape[1]) != int(a.shape[0]) or int(m.shape[0]) <= 0:
        return None
    row_mean = np.mean(m, axis=1)
    row_var = np.var(m, axis=1, ddof=0)
    valid = np.isfinite(row_var) & (row_var > 0.0)
    var_sum = float(np.sum(row_var[valid], dtype=np.float64))
    if (not np.isfinite(var_sum)) or (var_sum <= 0.0):
        var_sum = 1e-12
    alpha_sum = float(np.sum(a, dtype=np.float64))
    m_alpha = np.asarray(m @ a, dtype=np.float64).reshape(-1)
    beta = (m_alpha - row_mean * alpha_sum) / float(var_sum)
    coef_v = float(coef)
    if np.isfinite(coef_v) and coef_v != 1.0:
        beta = beta * coef_v
    return np.ascontiguousarray(beta, dtype=np.float64)


def _kernel_projection_beta_from_packed(
    *,
    packed_ctx: dict[str, typing.Any],
    sample_indices: np.ndarray,
    alpha: np.ndarray,
    mode: str,
    coef: float = 1.0,
    block_rows: int = 4096,
    threads: int = 0,
) -> np.ndarray | None:
    if _jxrs is None or not hasattr(_jxrs, "packed_malpha_mode_f64"):
        return None
    if not _looks_like_packed_payload(packed_ctx):
        return None
    try:
        packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8), dtype=np.uint8)
        maf = np.ascontiguousarray(
            np.asarray(packed_ctx.get("maf", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        row_flip = _ensure_packed_row_flip_cached(packed_ctx)
        n_samples = int(packed_ctx["n_samples"])
        sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
        a = np.ascontiguousarray(np.asarray(alpha, dtype=np.float64).reshape(-1), dtype=np.float64)
        m = int(packed.shape[0])
        if int(maf.shape[0]) != m or int(row_flip.shape[0]) != m:
            return None
        if int(sidx.shape[0]) == 0 or int(a.shape[0]) != int(sidx.shape[0]):
            return None

        mode_norm = str(mode).strip().lower()
        mode_arg = "d" if mode_norm in {"d", "dom", "dominance"} else "a"
        m_alpha_raw = _jxrs.packed_malpha_mode_f64(  # type: ignore[union-attr]
            packed,
            int(n_samples),
            row_flip,
            maf,
            sidx,
            a,
            mode=mode_arg,
            block_rows=int(max(1, int(block_rows))),
            threads=int(max(0, int(threads))),
        )
        m_alpha = np.ascontiguousarray(np.asarray(m_alpha_raw, dtype=np.float64).reshape(-1), dtype=np.float64)
        if int(m_alpha.shape[0]) != m:
            return None

        p_raw = np.asarray(maf, dtype=np.float64).reshape(-1)
        p_raw = np.clip(p_raw, 0.0, 1.0)
        flip = np.asarray(row_flip, dtype=np.bool_).reshape(-1)
        p_oriented = np.where(
            p_raw <= 0.5,
            p_raw,
            np.where(flip, 1.0 - p_raw, p_raw),
        )
        p_oriented = np.clip(p_oriented, 0.0, 1.0)
        if mode_arg == "d":
            row_mean = 2.0 * p_oriented * (1.0 - p_oriented)
            row_var = row_mean * (1.0 - row_mean)
        else:
            row_mean = 2.0 * p_oriented
            row_var = 2.0 * p_oriented * (1.0 - p_oriented)
        row_var = np.where(np.isfinite(row_var) & (row_var > 0.0), row_var, 0.0)
        var_sum = float(np.sum(row_var, dtype=np.float64))
        if (not np.isfinite(var_sum)) or (var_sum <= 0.0):
            var_sum = 1e-12
        alpha_sum = float(np.sum(a, dtype=np.float64))
        beta = (m_alpha - row_mean * alpha_sum) / float(var_sum)
        coef_v = float(coef)
        if np.isfinite(coef_v) and coef_v != 1.0:
            beta = beta * coef_v
        return np.ascontiguousarray(beta, dtype=np.float64)
    except Exception:
        return None


def _predict_bayes_packed_from_effects(
    *,
    packed_ctx: dict[str, typing.Any],
    sample_indices: np.ndarray,
    alpha0: float,
    beta: np.ndarray,
    row_mean: np.ndarray,
    row_inv_sd: np.ndarray,
    snp_block_size: int,
    sample_chunk_size: int,
) -> np.ndarray:
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    n_out = int(sidx.shape[0])
    if n_out == 0:
        return np.zeros((0, 1), dtype=np.float64)

    m = int(np.asarray(packed_ctx["packed"]).shape[0])
    b = np.ascontiguousarray(np.asarray(beta, dtype=np.float64).reshape(-1), dtype=np.float64)
    if int(b.shape[0]) != m:
        raise ValueError(f"Bayes beta length mismatch: got {b.shape[0]}, expected {m}.")

    out = np.zeros((n_out,), dtype=np.float64)
    row_blocks = _iter_row_blocks(m, snp_block_size)
    chunk = int(max(1, int(sample_chunk_size)))
    alpha = float(alpha0)
    for cst in range(0, n_out, chunk):
        ced = min(cst + chunk, n_out)
        blk_idx = np.ascontiguousarray(sidx[cst:ced], dtype=np.int64)
        pred = np.full((int(blk_idx.shape[0]),), alpha, dtype=np.float64)
        for st, ed, ridx in row_blocks:
            x = _decode_packed_block_standardized(
                packed_ctx=packed_ctx,
                row_idx=ridx,
                sample_indices=blk_idx,
                row_mean=row_mean,
                row_inv_sd=row_inv_sd,
            )
            x64 = np.ascontiguousarray(np.asarray(x, dtype=np.float64), dtype=np.float64)
            pred += np.asarray(x64.T @ b[st:ed], dtype=np.float64).reshape(-1)
        out[cst:ced] = pred
    return np.asarray(out, dtype=np.float64).reshape(-1, 1)


def _predict_bayes_packed_raw_from_effects(
    *,
    packed_ctx: dict[str, typing.Any],
    sample_indices: np.ndarray,
    alpha0: float,
    beta: np.ndarray,
    row_block_size: int,
    sample_chunk_size: int,
) -> np.ndarray:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_decode_rows_f32")):
        raise RuntimeError(
            "Rust packed BED decode helper is unavailable. Rebuild/install JanusX extension."
        )
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1), dtype=np.int64)
    n_out = int(sidx.shape[0])
    if n_out == 0:
        return np.zeros((0, 1), dtype=np.float64)

    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8), dtype=np.uint8)
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
    n_samples = int(packed_ctx["n_samples"])
    row_flip = _ensure_packed_row_flip_cached(packed_ctx)
    m = int(packed.shape[0])
    b = np.ascontiguousarray(np.asarray(beta, dtype=np.float64).reshape(-1), dtype=np.float64)
    if int(b.shape[0]) != m:
        raise ValueError(f"Bayes beta length mismatch: got {b.shape[0]}, expected {m}.")

    out = np.zeros((n_out,), dtype=np.float64)
    row_step = int(max(1, int(row_block_size)))
    sample_step = int(max(1, int(sample_chunk_size)))
    alpha = float(alpha0)
    for cst in range(0, n_out, sample_step):
        ced = min(cst + sample_step, n_out)
        blk_idx = np.ascontiguousarray(sidx[cst:ced], dtype=np.int64)
        pred = np.full((int(blk_idx.shape[0]),), alpha, dtype=np.float64)
        for st in range(0, m, row_step):
            ed = min(st + row_step, m)
            ridx = np.ascontiguousarray(np.arange(st, ed, dtype=np.int64), dtype=np.int64)
            x_blk = _jxrs.bed_packed_decode_rows_f32(  # type: ignore[union-attr]
                packed,
                int(n_samples),
                ridx,
                row_flip,
                maf,
                blk_idx,
            )
            x = np.ascontiguousarray(np.asarray(x_blk, dtype=np.float64), dtype=np.float64)
            pred += np.asarray(x.T @ b[st:ed], dtype=np.float64).reshape(-1)
        out[cst:ced] = pred
    return np.asarray(out, dtype=np.float64).reshape(-1, 1)


def _fit_rrblup_adamw_cpu(
    *,
    y: np.ndarray,
    Xtrain: typing.Any,
    is_packed_input: bool,
    packed_train_indices: np.ndarray | None,
    cfg: dict[str, typing.Any] | None,
    progress_hook: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
) -> dict[str, typing.Any]:
    cfg_use = cfg or {}
    y_vec = np.asarray(y, dtype=np.float32).reshape(-1)
    n_train = int(y_vec.shape[0])
    if n_train <= 0:
        raise ValueError("rrBLUP AdamW training requires at least one sample.")

    lambda_value = float(cfg_use.get("lambda_value", 1.0))
    lambda_scale = str(cfg_use.get("lambda_scale", "equation")).strip().lower()
    lr = float(cfg_use.get("lr", 1e-2))
    epochs = int(max(1, int(cfg_use.get("epochs", 60))))
    batch_size = int(max(1, int(cfg_use.get("batch_size", 1024))))
    snp_block_size = int(max(1, int(cfg_use.get("snp_block_size", 2048))))
    beta1 = float(cfg_use.get("beta1", 0.9))
    beta2 = float(cfg_use.get("beta2", 0.999))
    eps = float(cfg_use.get("eps", 1e-8))
    seed = int(cfg_use.get("seed", 42))
    log_every = int(max(0, int(cfg_use.get("log_every", 0))))
    sample_chunk_size = int(max(1, int(cfg_use.get("sample_chunk_size", 4096))))
    batch_threads_cfg = int(max(0, int(cfg_use.get("batch_threads", 0))))
    std_eps = float(max(np.finfo(np.float32).eps, float(cfg_use.get("std_eps", 1e-12))))
    early_stop_patience = int(max(0, int(cfg_use.get("early_stop_patience", 0))))
    early_stop_warmup = int(max(1, int(cfg_use.get("early_stop_warmup", 5))))
    early_stop_min_delta = float(max(0.0, float(cfg_use.get("early_stop_min_delta", 1e-5))))
    exclude_val_from_train = bool(_cfg_truthy(cfg_use.get("exclude_val_from_train", False), default=False))
    # Dense AdamW sample-batch threading:
    # - 0 (default): auto, only enable when BLAS threads are not already > 1.
    # - >0: explicit worker count for sample-sharded batch execution.
    total_threads_hint = int(
        max(
            1,
            int(_parse_nonnegative_int(os.getenv("JX_THREADS", "").strip()) or 1),
        )
    )
    blas_threads_hint = int(
        max(
            0,
            int(_parse_nonnegative_int(os.getenv("JX_MLM_BLAS_THREADS", "").strip()) or 0),
        )
    )

    if lambda_value < 0.0 or (not np.isfinite(lambda_value)):
        raise ValueError(f"rrBLUP lambda must be finite and >= 0, got {lambda_value!r}.")
    if lambda_scale not in {"equation", "mean-loss"}:
        raise ValueError(
            f"rrBLUP lambda_scale must be one of {{'equation', 'mean-loss'}}, got {lambda_scale!r}."
        )
    if (not np.isfinite(lr)) or lr <= 0.0:
        raise ValueError(f"rrBLUP learning rate must be > 0, got {lr!r}.")
    if (not np.isfinite(beta1)) or (not np.isfinite(beta2)):
        raise ValueError("rrBLUP beta1/beta2 must be finite.")
    if beta1 < 0.0 or beta1 >= 1.0 or beta2 < 0.0 or beta2 >= 1.0:
        raise ValueError("rrBLUP beta1/beta2 must be in [0,1).")
    if (not np.isfinite(eps)) or eps <= 0.0:
        raise ValueError(f"rrBLUP eps must be > 0, got {eps!r}.")

    lambda_effective = (
        (lambda_value / float(max(1, n_train)))
        if lambda_scale == "equation"
        else lambda_value
    )
    lambda_equation = (
        lambda_value
        if lambda_scale == "equation"
        else (lambda_value * float(max(1, n_train)))
    )

    row_mean: np.ndarray | None = None
    row_inv_sd: np.ndarray | None = None
    row_mean64: np.ndarray | None = None
    row_inv_sd64: np.ndarray | None = None
    train_abs_idx: np.ndarray | None = None
    X_dense: np.ndarray | None = None
    m_snp: int
    m_effective: int
    packed_maf: np.ndarray | None = None
    packed_row_flip: np.ndarray | None = None
    packed_n_samples = 0
    use_packed_malpha_grad = False
    packed_malpha_block_rows = 0
    packed_malpha_threads = 0

    if is_packed_input:
        packed_ctx = typing.cast(dict[str, typing.Any], Xtrain)
        packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
        maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
        if packed.ndim != 2:
            raise ValueError("Packed rrBLUP input must have 2D packed SNP matrix.")
        m_snp = int(packed.shape[0])
        if int(maf.shape[0]) != m_snp:
            raise ValueError("Packed rrBLUP input mismatch: maf length != packed SNP rows.")
        n_samples = int(packed_ctx["n_samples"])
        if packed_train_indices is None:
            if int(n_samples) != n_train:
                raise ValueError(
                    "Packed rrBLUP requires train sample indices when train size differs from packed n_samples."
                )
            train_abs_idx = np.ascontiguousarray(np.arange(n_samples, dtype=np.int64))
        else:
            train_abs_idx = np.ascontiguousarray(
                np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
            if int(train_abs_idx.shape[0]) != n_train:
                raise ValueError(
                    f"Packed rrBLUP train index length mismatch: got {train_abs_idx.shape[0]}, expected {n_train}."
                )
            if np.any(train_abs_idx < 0) or np.any(train_abs_idx >= int(n_samples)):
                raise ValueError("Packed rrBLUP train sample indices are out of range.")
        _ensure_packed_row_flip_cached(packed_ctx)
        row_mean = np.ascontiguousarray((2.0 * maf).astype(np.float32, copy=False))
        var = np.asarray(2.0 * maf * (1.0 - maf), dtype=np.float32)
        row_inv_sd = np.zeros_like(var, dtype=np.float32)
        good = var > np.float32(std_eps)
        row_inv_sd[good] = np.asarray(1.0 / np.sqrt(var[good], dtype=np.float32), dtype=np.float32)
        row_mean64 = np.ascontiguousarray(np.asarray(row_mean, dtype=np.float64), dtype=np.float64)
        row_inv_sd64 = np.ascontiguousarray(np.asarray(row_inv_sd, dtype=np.float64), dtype=np.float64)
        packed_maf = np.ascontiguousarray(maf, dtype=np.float32)
        packed_row_flip = np.ascontiguousarray(
            np.asarray(_ensure_packed_row_flip_cached(packed_ctx), dtype=np.bool_).reshape(-1),
            dtype=np.bool_,
        )
        packed_n_samples = int(n_samples)
        # Packed AdamW fast gradient path can be enabled explicitly via env.
        # Keep default conservative (off) to prioritize numerical stability.
        use_packed_malpha_grad = bool(
            (_jxrs is not None)
            and hasattr(_jxrs, "packed_malpha_f64")
            and _env_truthy("JX_RRBLUP_ADAM_PACKED_MALPHA", "0")
        )
        packed_malpha_block_rows = int(max(1, min(4096, int(snp_block_size), int(m_snp))))
        rust_threads_raw = _parse_nonnegative_int(os.getenv("JX_MLM_RUST_THREADS", "").strip())
        packed_malpha_threads = int(0 if rust_threads_raw is None else rust_threads_raw)
        m_effective = int(np.count_nonzero(good))
    else:
        X_dense = np.asarray(Xtrain, dtype=np.float32)
        if X_dense.ndim != 2:
            raise ValueError(f"Dense rrBLUP input must be 2D, got shape={X_dense.shape}.")
        m_snp = int(X_dense.shape[0])
        if int(X_dense.shape[1]) != n_train:
            raise ValueError(
                f"Dense rrBLUP input sample mismatch: Xtrain n={X_dense.shape[1]}, y n={n_train}."
            )
        m_effective = int(m_snp)

    if is_packed_input:
        dense_batch_threads = 1
    elif batch_threads_cfg > 0:
        dense_batch_threads = int(max(1, int(batch_threads_cfg)))
    else:
        # Auto policy: if BLAS is already multithreaded, avoid nested oversubscription.
        dense_batch_threads = int(total_threads_hint if blas_threads_hint <= 1 else 1)

    val_idx_raw = cfg_use.get("val_indices_local", None)
    val_loc: np.ndarray | None = None
    if val_idx_raw is not None:
        try:
            v = np.asarray(val_idx_raw, dtype=np.int64).reshape(-1)
            if int(v.size) > 0:
                v = v[(v >= 0) & (v < n_train)]
                if int(v.size) > 0:
                    val_loc = np.unique(np.ascontiguousarray(v, dtype=np.int64))
        except Exception:
            val_loc = None

    if (
        val_loc is not None
        and exclude_val_from_train
        and int(val_loc.size) > 0
        and int(val_loc.size) < n_train
    ):
        opt_mask = np.ones((n_train,), dtype=np.bool_)
        opt_mask[val_loc] = False
        opt_loc = np.ascontiguousarray(np.flatnonzero(opt_mask), dtype=np.int64)
    else:
        opt_loc = np.ascontiguousarray(np.arange(n_train, dtype=np.int64))
    n_opt = int(opt_loc.size)
    if n_opt <= 0:
        raise ValueError("rrBLUP AdamW has zero optimization samples after validation split.")
    opt_is_full = bool(n_opt == n_train)

    beta = np.zeros((m_snp,), dtype=np.float32)
    alpha = np.zeros((1,), dtype=np.float32)
    m_alpha = np.zeros_like(alpha, dtype=np.float32)
    v_alpha = np.zeros_like(alpha, dtype=np.float32)
    m_beta = np.zeros_like(beta, dtype=np.float32)
    v_beta = np.zeros_like(beta, dtype=np.float32)
    step_alpha = 0
    step_beta = 0
    row_blocks = _iter_row_blocks(m_snp, snp_block_size)
    rng = np.random.default_rng(seed)
    full_batch_mode = bool(batch_size >= n_opt)
    full_loc = np.ascontiguousarray(opt_loc, dtype=np.int64)
    y_full = np.ascontiguousarray(y_vec[opt_loc], dtype=np.float32)
    dense_full_batch_t: np.ndarray | None = None
    if (not is_packed_input) and full_batch_mode:
        assert X_dense is not None
        if opt_is_full:
            dense_full_batch_t = np.ascontiguousarray(X_dense.T, dtype=np.float32)
        else:
            dense_full_batch_t = np.ascontiguousarray(
                np.take(X_dense, full_loc, axis=1).T,
                dtype=np.float32,
            )
    dense_batch_pool: cf.ThreadPoolExecutor | None = None
    if (not is_packed_input) and int(dense_batch_threads) > 1:
        dense_batch_pool = cf.ThreadPoolExecutor(max_workers=int(dense_batch_threads))

    def _batch_shard_ranges(batch_n: int, workers: int) -> list[tuple[int, int]]:
        n = int(max(0, int(batch_n)))
        k = int(max(1, int(workers)))
        if n <= 0:
            return []
        if k > n:
            k = n
        base = n // k
        rem = n % k
        st = 0
        out: list[tuple[int, int]] = []
        for i in range(k):
            step = int(base + (1 if i < rem else 0))
            ed = st + step
            if ed > st:
                out.append((st, ed))
            st = ed
        return out

    use_val_monitor = bool((val_loc is not None) and (int(val_loc.size) > 0))
    use_early_stop = bool(use_val_monitor and (early_stop_patience > 0))
    best_val_loss = np.inf
    best_epoch = 0
    best_alpha = float(alpha[0])
    best_beta: np.ndarray | None = None
    bad_epochs = 0
    epochs_ran = 0
    stopped_early = False

    def _emit_progress(event: str, **payload: typing.Any) -> None:
        if progress_hook is None:
            return
        try:
            progress_hook(str(event), dict(payload))
        except Exception:
            return

    def _compute_val_mse() -> float:
        if (val_loc is None) or int(val_loc.size) <= 0:
            return float("nan")
        y_val = np.asarray(y_vec[val_loc], dtype=np.float32)
        if is_packed_input:
            assert train_abs_idx is not None
            assert row_mean is not None
            assert row_inv_sd is not None
            pred_val = _predict_rrblup_packed_from_beta(
                packed_ctx=typing.cast(dict[str, typing.Any], Xtrain),
                sample_indices=np.ascontiguousarray(train_abs_idx[val_loc], dtype=np.int64),
                alpha=float(alpha[0]),
                beta=beta,
                row_mean=row_mean,
                row_inv_sd=row_inv_sd,
                snp_block_size=snp_block_size,
                sample_chunk_size=sample_chunk_size,
            )
        else:
            assert X_dense is not None
            pred_val = _predict_rrblup_dense_from_beta(
                X=np.ascontiguousarray(np.take(X_dense, val_loc, axis=1), dtype=np.float32),
                alpha=float(alpha[0]),
                beta=beta,
                snp_block_size=snp_block_size,
            )
        pv = np.asarray(pred_val, dtype=np.float32).reshape(-1)
        if int(pv.size) != int(y_val.size):
            return float("nan")
        diff = np.asarray(pv - y_val, dtype=np.float32)
        return float(np.mean(diff * diff, dtype=np.float64))

    _emit_progress(
        "adam_start",
        total=int(epochs),
        use_val_monitor=bool(use_val_monitor),
        use_early_stop=bool(use_early_stop),
        val_samples=int(0 if val_loc is None else int(val_loc.size)),
        opt_samples=int(max(1, n_opt)),
    )

    try:
        for ep in range(epochs):
            order = (
                full_loc
                if full_batch_mode
                else np.ascontiguousarray(rng.permutation(full_loc), dtype=np.int64)
            )
            loss_sum = 0.0
            batch_ct = 0
            batch_stride = n_opt if full_batch_mode else batch_size
            for b_st in range(0, n_opt, batch_stride):
                b_ed = min(b_st + batch_size, n_opt)
                if full_batch_mode:
                    loc = full_loc
                else:
                    loc = np.ascontiguousarray(order[b_st:b_ed], dtype=np.int64)
                bs = int(loc.shape[0])
                if bs <= 0:
                    continue
                yb = y_full if full_batch_mode else np.ascontiguousarray(y_vec[loc], dtype=np.float32)
                x_batch_t: np.ndarray | None = None
                resid: np.ndarray | None = None
                grad_beta_dense: np.ndarray | None = None
                sum_resid_dense: float | None = None
                pred = np.full((bs,), alpha[0], dtype=np.float32)

                if is_packed_input:
                    assert train_abs_idx is not None
                    assert row_mean is not None
                    assert row_inv_sd is not None
                    abs_idx = (
                        train_abs_idx
                        if (full_batch_mode and opt_is_full)
                        else np.ascontiguousarray(train_abs_idx[loc], dtype=np.int64)
                    )
                    for st, ed, ridx in row_blocks:
                        x = _decode_packed_block_standardized(
                            packed_ctx=typing.cast(dict[str, typing.Any], Xtrain),
                            row_idx=ridx,
                            sample_indices=abs_idx,
                            row_mean=row_mean,
                            row_inv_sd=row_inv_sd,
                        )
                        pred += np.asarray(x.T @ beta[st:ed], dtype=np.float32).reshape(-1)
                    resid = np.asarray(pred - yb, dtype=np.float32)
                    loss_sum += float(np.mean(resid * resid, dtype=np.float64))
                    batch_ct += 1
                else:
                    assert X_dense is not None
                    use_sharded_dense = bool(
                        dense_batch_pool is not None
                        and int(dense_batch_threads) > 1
                        and int(bs) >= int(dense_batch_threads)
                    )
                    if use_sharded_dense:
                        def _dense_shard_job(local_st: int, local_ed: int) -> tuple[float, float, np.ndarray]:
                            if full_batch_mode and dense_full_batch_t is not None:
                                x_sub = np.ascontiguousarray(
                                    dense_full_batch_t[local_st:local_ed],
                                    dtype=np.float32,
                                )
                            else:
                                loc_sub = np.ascontiguousarray(
                                    loc[local_st:local_ed],
                                    dtype=np.int64,
                                )
                                x_sub = np.ascontiguousarray(
                                    np.take(X_dense, loc_sub, axis=1).T,
                                    dtype=np.float32,
                                )
                            y_sub = np.ascontiguousarray(
                                yb[local_st:local_ed],
                                dtype=np.float32,
                            )
                            pred_sub = np.asarray(x_sub @ beta, dtype=np.float32).reshape(-1)
                            pred_sub += np.float32(alpha[0])
                            resid_sub = np.asarray(pred_sub - y_sub, dtype=np.float32).reshape(-1)
                            resid64 = np.asarray(resid_sub, dtype=np.float64)
                            sq_sum = float(np.sum(resid64 * resid64, dtype=np.float64))
                            res_sum = float(np.sum(resid64, dtype=np.float64))
                            grad_sub = np.asarray(x_sub.T @ resid_sub, dtype=np.float64).reshape(-1)
                            return sq_sum, res_sum, grad_sub

                        shard_ranges = _batch_shard_ranges(int(bs), int(dense_batch_threads))
                        grad_acc = np.zeros((m_snp,), dtype=np.float64)
                        sq_sum_acc = 0.0
                        res_sum_acc = 0.0
                        futures = [
                            dense_batch_pool.submit(_dense_shard_job, int(st), int(ed))
                            for st, ed in shard_ranges
                        ]
                        for fut in futures:
                            sq_part, res_part, grad_part = fut.result()
                            sq_sum_acc += float(sq_part)
                            res_sum_acc += float(res_part)
                            grad_acc += np.asarray(grad_part, dtype=np.float64).reshape(-1)
                        loss_sum += float(sq_sum_acc / float(max(1, bs)))
                        batch_ct += 1
                        sum_resid_dense = float(res_sum_acc)
                        grad_beta_dense = np.ascontiguousarray(
                            np.asarray(grad_acc / float(max(1, bs)), dtype=np.float32).reshape(-1),
                            dtype=np.float32,
                        )
                    else:
                        # Dense mini-batch optimization:
                        # cache sample-batch matrix once and use BLAS for both
                        # prediction and gradient.
                        if full_batch_mode and dense_full_batch_t is not None:
                            x_batch_t = dense_full_batch_t
                        else:
                            x_batch_t = np.ascontiguousarray(
                                np.take(X_dense, loc, axis=1).T,
                                dtype=np.float32,
                            )
                        pred = np.asarray(x_batch_t @ beta, dtype=np.float32).reshape(-1)
                        pred += np.float32(alpha[0])
                        resid = np.asarray(pred - yb, dtype=np.float32)
                        loss_sum += float(np.mean(resid * resid, dtype=np.float64))
                        batch_ct += 1

                step_alpha += 1
                if sum_resid_dense is not None:
                    grad_alpha = np.asarray([float(sum_resid_dense / float(max(1, bs)))], dtype=np.float32)
                else:
                    assert resid is not None
                    grad_alpha = np.asarray([float(np.mean(resid, dtype=np.float64))], dtype=np.float32)
                _adamw_update_inplace_f32(
                    param=alpha,
                    grad=grad_alpha,
                    m1=m_alpha,
                    m2=v_alpha,
                    step=step_alpha,
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=0.0,
                )

                step_beta += 1
                inv_bs = np.float32(1.0 / float(bs))
                if is_packed_input:
                    assert resid is not None
                    assert train_abs_idx is not None
                    assert row_mean is not None
                    assert row_inv_sd is not None
                    assert row_mean64 is not None
                    assert row_inv_sd64 is not None
                    assert packed_maf is not None
                    assert packed_row_flip is not None
                    abs_idx = (
                        train_abs_idx
                        if (full_batch_mode and opt_is_full)
                        else np.ascontiguousarray(train_abs_idx[loc], dtype=np.int64)
                    )
                    used_fast_grad = False
                    if use_packed_malpha_grad:
                        try:
                            resid64 = np.ascontiguousarray(
                                np.asarray(resid, dtype=np.float64).reshape(-1),
                                dtype=np.float64,
                            )
                            m_alpha_raw = _jxrs.packed_malpha_f64(  # type: ignore[union-attr]
                                typing.cast(dict[str, typing.Any], Xtrain)["packed"],
                                int(packed_n_samples),
                                packed_row_flip,
                                packed_maf,
                                np.ascontiguousarray(abs_idx, dtype=np.int64),
                                resid64,
                                block_rows=int(packed_malpha_block_rows),
                                threads=int(packed_malpha_threads),
                            )
                            m_alpha_vec = np.ascontiguousarray(
                                np.asarray(m_alpha_raw, dtype=np.float64).reshape(-1),
                                dtype=np.float64,
                            )
                            if int(m_alpha_vec.shape[0]) != int(m_snp):
                                raise RuntimeError(
                                    "packed_malpha_f64 length mismatch: "
                                    f"got {m_alpha_vec.shape[0]}, expected {m_snp}"
                                )
                            sum_resid = float(np.sum(resid64, dtype=np.float64))
                            grad_all = (
                                (m_alpha_vec - (row_mean64 * sum_resid))
                                * row_inv_sd64
                                * float(inv_bs)
                            )
                            grad_all_f32 = np.ascontiguousarray(
                                np.asarray(grad_all, dtype=np.float32).reshape(-1),
                                dtype=np.float32,
                            )
                            _adamw_update_inplace_f32(
                                param=beta,
                                grad=grad_all_f32,
                                m1=m_beta,
                                m2=v_beta,
                                step=step_beta,
                                lr=lr,
                                beta1=beta1,
                                beta2=beta2,
                                eps=eps,
                                weight_decay=float(lambda_effective),
                            )
                            used_fast_grad = True
                        except Exception:
                            use_packed_malpha_grad = False
                    if not used_fast_grad:
                        for st, ed, ridx in row_blocks:
                            x = _decode_packed_block_standardized(
                                packed_ctx=typing.cast(dict[str, typing.Any], Xtrain),
                                row_idx=ridx,
                                sample_indices=abs_idx,
                                row_mean=row_mean,
                                row_inv_sd=row_inv_sd,
                            )
                            grad_beta = np.asarray((x @ resid) * inv_bs, dtype=np.float32).reshape(-1)
                            _adamw_update_inplace_f32(
                                param=beta[st:ed],
                                grad=grad_beta,
                                m1=m_beta[st:ed],
                                m2=v_beta[st:ed],
                                step=step_beta,
                                lr=lr,
                                beta1=beta1,
                                beta2=beta2,
                                eps=eps,
                                weight_decay=float(lambda_effective),
                            )
                else:
                    if grad_beta_dense is None:
                        assert x_batch_t is not None
                        assert resid is not None
                        grad_beta_dense = np.asarray(
                            (x_batch_t.T @ resid) * inv_bs,
                            dtype=np.float32,
                        ).reshape(-1)
                    _adamw_update_inplace_f32(
                        param=beta,
                        grad=grad_beta_dense,
                        m1=m_beta,
                        m2=v_beta,
                        step=step_beta,
                        lr=lr,
                        beta1=beta1,
                        beta2=beta2,
                        eps=eps,
                        weight_decay=float(lambda_effective),
                    )

            epochs_ran = ep + 1
            mean_loss = (loss_sum / float(max(1, batch_ct)))
            val_mse = _compute_val_mse() if use_val_monitor else float("nan")
            _emit_progress(
                "adam_epoch",
                epoch=int(ep + 1),
                total=int(epochs),
                train_mse=float(mean_loss),
                val_mse=float(val_mse),
                best_epoch=int(max(0, int(best_epoch))),
                best_val_loss=float(best_val_loss),
            )
            if use_val_monitor and np.isfinite(val_mse):
                improved = (best_epoch <= 0) or (val_mse < (best_val_loss - early_stop_min_delta))
                if improved:
                    best_val_loss = float(val_mse)
                    best_epoch = ep + 1
                    best_alpha = float(alpha[0])
                    best_beta = np.ascontiguousarray(beta.copy(), dtype=np.float32)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if use_early_stop and (ep + 1) >= early_stop_warmup and bad_epochs >= early_stop_patience:
                    if _GS_DEBUG_STAGE:
                        print(
                            f"[GS-DEBUG] rrBLUP-AdamW early-stop at epoch={ep + 1}, "
                            f"best_epoch={best_epoch}, best_val_mse={best_val_loss:.6g}",
                            flush=True,
                        )
                    stopped_early = True
                    break

            if _GS_DEBUG_STAGE and log_every > 0 and ((ep + 1) % log_every == 0):
                if np.isfinite(val_mse):
                    print(
                        f"[GS-DEBUG] rrBLUP-AdamW epoch={ep + 1}/{epochs} "
                        f"train_mse={mean_loss:.6g} val_mse={float(val_mse):.6g}",
                        flush=True,
                    )
                else:
                    print(
                        f"[GS-DEBUG] rrBLUP-AdamW epoch={ep + 1}/{epochs} train_mse={mean_loss:.6g}",
                        flush=True,
                    )
    finally:
        if dense_batch_pool is not None:
            dense_batch_pool.shutdown(wait=True)

    if use_val_monitor and (best_beta is not None) and (best_epoch > 0):
        alpha[0] = np.float32(best_alpha)
        beta[:] = np.asarray(best_beta, dtype=np.float32)
    else:
        best_epoch = max(1, int(epochs_ran))
        if not np.isfinite(best_val_loss):
            best_val_loss = float("nan")

    _emit_progress(
        "adam_end",
        epochs_ran=int(max(1, int(epochs_ran))),
        total=int(epochs),
        best_epoch=int(max(1, int(best_epoch))),
        best_val_loss=float(best_val_loss),
        stopped_early=bool(stopped_early),
    )

    return {
        "alpha": float(alpha[0]),
        "beta": np.ascontiguousarray(beta, dtype=np.float32),
        "lambda_effective": float(lambda_effective),
        "lambda_equation": float(lambda_equation),
        "snp_block_size": int(snp_block_size),
        "sample_chunk_size": int(sample_chunk_size),
        "batch_threads": int(max(1, int(dense_batch_threads))),
        "is_packed": bool(is_packed_input),
        "row_mean": row_mean,
        "row_inv_sd": row_inv_sd,
        "train_abs_idx": train_abs_idx,
        "m_effective": int(max(1, m_effective)),
        "best_epoch": int(max(1, int(best_epoch))),
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(max(1, int(epochs_ran))),
        "stopped_early": bool(stopped_early),
        "opt_samples": int(max(1, n_opt)),
        "val_samples": int(0 if val_loc is None else int(val_loc.size)),
    }


def GSapi(
    Y: np.ndarray,
    Xtrain: typing.Any,
    Xtest: typing.Any,
    method: typing.Literal["GBLUP", "GBLUP_D", "GBLUP_AD", "rrBLUP", "BayesA", "BayesB", "BayesCpi", "RF", "ET", "GBDT", "XGB", "SVM", "ENET"],
    PCAdec: bool = False,
    n_jobs: int = 1,
    force_fast: bool = False,
    ml_fixed_params: dict[str, typing.Any] | None = None,
    need_train_pred: bool = True,
    packed_train_indices: np.ndarray | None = None,
    packed_test_indices: np.ndarray | None = None,
    train_pred_indices: np.ndarray | None = None,
    rrblup_solver: typing.Literal["exact", "adamw", "pcg", "auto"] = "pcg",
    rrblup_adamw_cfg: dict[str, typing.Any] | None = None,
    rrblup_runtime_state: dict[str, typing.Any] | None = None,
    rrblup_progress_hook: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
    gblup_runtime_state: dict[str, typing.Any] | None = None,
    bayes_auto_r2: float | None = None,
    bayes_runtime_state: dict[str, typing.Any] | None = None,
    bayes_auto_cfg: dict[str, typing.Any] | None = None,
    model_state: dict[str, typing.Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Core genomic selection API.

    Parameters
    ----------
    Y : np.ndarray
        Phenotype values for training individuals, shape (n_train, 1) or (n_train,).
    Xtrain : np.ndarray
        Genotype matrix for training individuals, shape (m_markers, n_train).
    Xtest : np.ndarray
        Genotype matrix for test individuals, shape (m_markers, n_test).
    method : {'GBLUP', 'GBLUP_D', 'GBLUP_AD', 'rrBLUP', 'BayesA', 'BayesB', 'BayesCpi', 'RF', 'ET', 'GBDT', 'XGB', 'SVM', 'ENET'}
        Prediction model.
    PCAdec : bool, optional
        If True, perform PCA-based dimensionality reduction before modeling.
        PCA is computed on the concatenated matrix [Xtrain, Xtest].
    n_jobs : int, optional
        Thread count for ML models. Non-ML models currently ignore this setting.
    need_train_pred : bool, optional
        If False, skip generating train-set predictions when the caller does not need them.
    train_pred_indices : np.ndarray, optional
        Optional local indices (relative to Xtrain columns) used for train-set
        prediction. When provided, only these training samples are predicted.
    rrblup_solver : {'exact', 'adamw', 'pcg', 'auto'}, optional
        rrBLUP backend policy. `exact` keeps the legacy solver; `adamw`
        enables mini-batch AdamW; `pcg` enables Rust packed-BED PCG solver;
        `auto` chooses exact for n<=10k and PCG for n>10k.
    rrblup_adamw_cfg : dict, optional
        AdamW rrBLUP configuration used when backend resolves to `adamw`.
    rrblup_runtime_state : dict, optional
        Mutable runtime state sink for rrBLUP AdamW (selected tuning params, diagnostics).
    rrblup_progress_hook : callable, optional
        Progress callback for rrBLUP AdamW epochs and auto-grid trials.
    gblup_runtime_state : dict, optional
        Mutable runtime state sink for packed Rust GBLUP diagnostics
        (eigh backend/time, REML lambda/likelihood).
    bayes_auto_r2 : float, optional
        Optional precomputed BLUP PVE used as Bayes `r2` prior. When omitted,
        Bayes models estimate it internally from BLUP.
    bayes_runtime_state : dict, optional
        Mutable state sink for Bayes diagnostics (e.g. resolved `r2` source/value).
    bayes_auto_cfg : dict, optional
        Bayes auto-r2 configuration. Supports optional BLUP subsampling on
        large sample sizes to reduce memory usage.
    model_state : dict, optional
        Mutable sink for exporting a lightweight fitted-model artifact.

    Returns
    -------
    yhat_train : np.ndarray
        Predicted phenotypes for training individuals, shape (n_train, 1).
    yhat_test : np.ndarray
        Predicted phenotypes for test individuals, shape (n_test, 1).
    pve : float
        Variance-component PVE/h2 for mixed models and Bayes models,
        or predictive PVE (inner-CV R2) for ML models.
    """
    train_pred_idx = _normalize_train_pred_indices(
        train_pred_indices,
        int(Y.reshape(-1).shape[0]),
    )
    is_packed_input = _looks_like_packed_payload(Xtrain) or _looks_like_packed_payload(Xtest)
    if is_packed_input:
        if PCAdec:
            raise ValueError("PCAdec is not supported with packed genotype payloads.")
    else:
        Xtrain, Xtest = _apply_optional_pca(
            np.asarray(Xtrain),
            np.asarray(Xtest),
            enabled=PCAdec,
        )

    # Additive+dominance GBLUP variant (pyBLUP/JAX backend)
    if _is_gblup_method(str(method)) and (_gblup_method_kernel_mode(str(method)) == "ad"):
        mode = _gblup_method_kernel_mode(str(method))
        if (not _HAS_ADBLUP_PY) or KernelBLUP is None or Gmatrix is None:
            raise ImportError(
                "GBLUP(ad) (JAX backend) is unavailable. "
                f"Original import error: {_ADBLUP_PY_IMPORT_ERROR}"
            )
        if is_packed_input:
            if not _looks_like_packed_payload(Xtrain):
                raise ValueError("GBLUP(ad) packed route requires packed train payload.")
            packed_train = typing.cast(dict[str, typing.Any], Xtrain)
            n_train_expected = int(np.asarray(Y).reshape(-1).shape[0])
            if packed_train_indices is None:
                if int(packed_train["n_samples"]) != n_train_expected:
                    raise ValueError(
                        "Packed GBLUP(ad) requires packed_train_indices when "
                        "n_train differs from packed n_samples."
                    )
                train_abs = np.ascontiguousarray(
                    np.arange(int(packed_train["n_samples"]), dtype=np.int64),
                    dtype=np.int64,
                )
            else:
                train_abs = np.ascontiguousarray(
                    np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                )
            if int(train_abs.shape[0]) != n_train_expected:
                raise ValueError(
                    "Packed GBLUP(ad) train index mismatch: "
                    f"indices={train_abs.shape[0]}, y={n_train_expected}."
                )
            if packed_test_indices is None:
                test_abs = np.zeros((0,), dtype=np.int64)
            else:
                test_abs = np.ascontiguousarray(
                    np.asarray(packed_test_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                )
            gadd_train = _decode_packed_subset_to_dense_raw_f32(
                packed_train,
                train_abs,
            )
            gadd_test = (
                np.zeros((int(gadd_train.shape[0]), 0), dtype=np.float32)
                if int(test_abs.size) == 0
                else _decode_packed_subset_to_dense_raw_f32(
                    typing.cast(dict[str, typing.Any], Xtest),
                    test_abs,
                )
            )
        else:
            gadd_train = np.asarray(Xtrain, dtype=np.float32)
            gadd_test = np.asarray(Xtest, dtype=np.float32)
        if gadd_train.ndim != 2 or gadd_test.ndim != 2:
            raise ValueError(
                f"GBLUP(ad) expects 2D genotype matrices. got train={gadd_train.shape}, test={gadd_test.shape}"
            )
        if gadd_train.shape[0] != gadd_test.shape[0]:
            raise ValueError(
                f"GBLUP(ad) marker mismatch: train={gadd_train.shape[0]}, test={gadd_test.shape[0]}"
            )

        ghet_train = (gadd_train == 1.0).astype(np.float32, copy=False)
        Ghet_train = Gmatrix(ghet_train)
        Gadd_train = Gmatrix(gadd_train)
        model = KernelBLUP(Y.reshape(-1, 1), G=[Gadd_train, Ghet_train], progress=False)

        if need_train_pred:
            yhat_train = model.predict(G=[Gadd_train, Ghet_train])
        else:
            yhat_train = np.zeros((0, 1), dtype=float)
        n_train = int(gadd_train.shape[1])
        if int(gadd_test.shape[1]) > 0:
            gadd_all = np.concatenate([gadd_train, gadd_test], axis=1)
            ghet_all = (gadd_all == 1.0).astype(np.float32, copy=False)
            Ghet_all = Gmatrix(ghet_all)
            Ghet_cross = Ghet_all[n_train:, :n_train]
            Gadd_all = Gmatrix(gadd_all)
            Gadd_cross = Gadd_all[n_train:, :n_train]
            yhat_test = model.predict(G=[Gadd_cross, Ghet_cross])
        else:
            yhat_test = np.zeros((0, 1), dtype=float)

        pve = float("nan")
        sigma_g2 = float("nan")
        sigma_e2 = float("nan")
        sigma_a2 = float("nan")
        sigma_d2 = float("nan")
        lambda_reml = float("nan")
        try:
            var = np.asarray(getattr(model, "var", []), dtype=float).reshape(-1)
            if var.size >= 2 and np.all(np.isfinite(var)):
                den = float(np.sum(var))
                if den > 0:
                    pve = float(np.sum(var[:-1]) / den)
                    sigma_e2 = float(var[-1])
                    sigma_a2 = float(var[0]) if int(var.size) >= 3 else float("nan")
                    sigma_d2 = float(var[1]) if int(var.size) >= 3 else float("nan")
                    sigma_g2 = float(np.sum(var[:-1]))
                    if np.isfinite(sigma_g2) and sigma_g2 > 0.0 and np.isfinite(sigma_e2):
                        lambda_reml = float(sigma_e2 / sigma_g2)
        except Exception:
            pve = float("nan")
            sigma_g2 = float("nan")
            sigma_e2 = float("nan")
            sigma_a2 = float("nan")
            sigma_d2 = float("nan")
            lambda_reml = float("nan")
        if gblup_runtime_state is not None:
            gblup_runtime_state["variance_component_path"] = "grm"
            gblup_runtime_state["variance_component"] = "REML/GRM"
            gblup_runtime_state["backend"] = "pyBLUP/JAX"
            gblup_runtime_state["kernel_mode"] = str(mode)
            if np.isfinite(sigma_g2):
                gblup_runtime_state["sigma_g2"] = float(sigma_g2)
            if np.isfinite(sigma_e2):
                gblup_runtime_state["sigma_e2"] = float(sigma_e2)
            if np.isfinite(sigma_a2):
                gblup_runtime_state["sigma_a2"] = float(sigma_a2)
            if np.isfinite(sigma_d2):
                gblup_runtime_state["sigma_d2"] = float(sigma_d2)
            if np.isfinite(lambda_reml):
                gblup_runtime_state["lambda_reml"] = float(lambda_reml)
            if np.isfinite(pve):
                gblup_runtime_state["h2"] = float(pve)
        if model_state is not None:
            beta_a = np.zeros((int(gadd_train.shape[0]),), dtype=np.float64)
            beta_d = np.zeros((int(gadd_train.shape[0]),), dtype=np.float64)
            used_rust_kernel_projection = False
            try:
                vinvr = np.asarray(getattr(model, "_Vinvr", np.zeros((0, 1))), dtype=np.float64).reshape(-1)
                g_theta = np.asarray(getattr(model, "_g_theta", np.zeros((0,))), dtype=np.float64).reshape(-1)
                g_scales = np.asarray(getattr(model, "_g_scales", np.ones((0,))), dtype=np.float64).reshape(-1)
                if int(vinvr.size) == int(gadd_train.shape[1]):
                    coef_a = (
                        float(g_theta[0]) / float(max(1e-12, float(g_scales[0])))
                        if int(g_theta.size) >= 2 and int(g_scales.size) >= 2
                        else 1.0
                    )
                    coef_d = (
                        float(g_theta[1]) / float(max(1e-12, float(g_scales[1])))
                        if int(g_theta.size) >= 2 and int(g_scales.size) >= 2
                        else 1.0
                    )
                    if is_packed_input and _looks_like_packed_payload(Xtrain):
                        beta_a_rust = _kernel_projection_beta_from_packed(
                            packed_ctx=typing.cast(dict[str, typing.Any], Xtrain),
                            sample_indices=np.asarray(train_abs, dtype=np.int64),
                            alpha=vinvr,
                            mode="a",
                            coef=float(coef_a),
                            block_rows=4096,
                            threads=max(0, int(n_jobs)),
                        )
                        beta_d_rust = _kernel_projection_beta_from_packed(
                            packed_ctx=typing.cast(dict[str, typing.Any], Xtrain),
                            sample_indices=np.asarray(train_abs, dtype=np.int64),
                            alpha=vinvr,
                            mode="d",
                            coef=float(coef_d),
                            block_rows=4096,
                            threads=max(0, int(n_jobs)),
                        )
                        if beta_a_rust is not None and beta_d_rust is not None:
                            beta_a = np.ascontiguousarray(beta_a_rust, dtype=np.float64)
                            beta_d = np.ascontiguousarray(beta_d_rust, dtype=np.float64)
                            used_rust_kernel_projection = True
                    if int(g_theta.size) >= 2 and int(g_scales.size) >= 2:
                        if not used_rust_kernel_projection:
                            beta_a_dense = _kernel_projection_beta_from_dense(
                                marker_by_sample=gadd_train,
                                alpha=vinvr,
                                coef=float(coef_a),
                            )
                            beta_d_dense = _kernel_projection_beta_from_dense(
                                marker_by_sample=ghet_train,
                                alpha=vinvr,
                                coef=float(coef_d),
                            )
                            if beta_a_dense is not None and beta_d_dense is not None:
                                beta_a = np.ascontiguousarray(beta_a_dense, dtype=np.float64)
                                beta_d = np.ascontiguousarray(beta_d_dense, dtype=np.float64)
            except Exception:
                beta_a = np.zeros((int(gadd_train.shape[0]),), dtype=np.float64)
                beta_d = np.zeros((int(gadd_train.shape[0]),), dtype=np.float64)
            model_state["kind"] = "gblup_kernel_projection_ad"
            model_state["method"] = str(method)
            model_state["effect_kind"] = "kernel_projection"
            model_state["alpha"] = float(np.mean(np.asarray(Y, dtype=np.float64)))
            model_state["beta_a"] = np.ascontiguousarray(beta_a, dtype=np.float64)
            model_state["beta_d"] = np.ascontiguousarray(beta_d, dtype=np.float64)
            model_state["kernel_projection_a"] = np.ascontiguousarray(beta_a, dtype=np.float64)
            model_state["kernel_projection_d"] = np.ascontiguousarray(beta_d, dtype=np.float64)
            model_state["kernel_projection"] = np.ascontiguousarray(beta_a + beta_d, dtype=np.float64)
            model_state["beta"] = np.ascontiguousarray(beta_a + beta_d, dtype=np.float64)
            model_state["packed"] = bool(is_packed_input)
            model_state["standardized"] = False
            model_state["kernel_mode"] = "ad"
            model_state["pve"] = float(pve)
        return np.asarray(yhat_train, dtype=float), np.asarray(yhat_test, dtype=float), pve

    # Dominance-only GBLUP: keep a dedicated fast path (separate from AD kernel stack).
    if _is_gblup_method(str(method)) and (_gblup_method_kernel_mode(str(method)) == "d"):
        if is_packed_input:
            if not _looks_like_packed_payload(Xtrain):
                raise ValueError("GBLUP(d) packed route requires packed train payload.")
            packed_train = typing.cast(dict[str, typing.Any], Xtrain)
            n_train_expected = int(np.asarray(Y).reshape(-1).shape[0])
            if packed_train_indices is None:
                if int(packed_train["n_samples"]) != n_train_expected:
                    raise ValueError(
                        "Packed GBLUP(d) requires packed_train_indices when "
                        "n_train differs from packed n_samples."
                    )
                train_abs = np.ascontiguousarray(
                    np.arange(int(packed_train["n_samples"]), dtype=np.int64),
                    dtype=np.int64,
                )
            else:
                train_abs = np.ascontiguousarray(
                    np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                )
            if int(train_abs.shape[0]) != n_train_expected:
                raise ValueError(
                    "Packed GBLUP(d) train index mismatch: "
                    f"indices={train_abs.shape[0]}, y={n_train_expected}."
                )
            if packed_test_indices is None:
                test_abs = np.zeros((0,), dtype=np.int64)
            else:
                test_abs = np.ascontiguousarray(
                    np.asarray(packed_test_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                )
            gadd_train = _decode_packed_subset_to_dense_raw_f32(
                packed_train,
                train_abs,
            )
            gadd_test = (
                np.zeros((int(gadd_train.shape[0]), 0), dtype=np.float32)
                if int(test_abs.size) == 0
                else _decode_packed_subset_to_dense_raw_f32(
                    typing.cast(dict[str, typing.Any], Xtest),
                    test_abs,
                )
            )
        else:
            gadd_train = np.asarray(Xtrain, dtype=np.float32)
            gadd_test = np.asarray(Xtest, dtype=np.float32)

        if gadd_train.ndim != 2 or gadd_test.ndim != 2:
            raise ValueError(
                f"GBLUP(d) expects 2D genotype matrices. got train={gadd_train.shape}, test={gadd_test.shape}"
            )
        if gadd_train.shape[0] != gadd_test.shape[0]:
            raise ValueError(
                f"GBLUP(d) marker mismatch: train={gadd_train.shape[0]}, test={gadd_test.shape[0]}"
            )

        gdom_train = (gadd_train == 1.0).astype(np.float32, copy=False)
        gdom_test = (gadd_test == 1.0).astype(np.float32, copy=False)
        model = MLMBLUP(
            Y.reshape(-1, 1),
            gdom_train,
            kinship=1,
            force_fast=True,
        )

        if need_train_pred:
            if train_pred_idx is None:
                yhat_train = model.predict(gdom_train)
            elif int(train_pred_idx.size) == 0:
                yhat_train = np.zeros((0, 1), dtype=float)
            else:
                yhat_train = model.predict(np.asarray(gdom_train)[:, train_pred_idx])
        else:
            yhat_train = np.zeros((0, 1), dtype=float)
        if int(gdom_test.shape[1]) > 0:
            yhat_test = model.predict(gdom_test)
        else:
            yhat_test = np.zeros((0, 1), dtype=float)

        pve = float(getattr(model, "pve", np.nan))
        sigma_g2 = float("nan")
        sigma_e2 = float("nan")
        lambda_reml = float("nan")
        try:
            rtv = float(getattr(model, "rTV_invr", np.nan))
            n_m = int(getattr(model, "n", int(np.asarray(Y).reshape(-1).shape[0])))
            p_m = int(getattr(model, "p", 1))
            lambda_reml = float(getattr(model, "lbd", np.nan))
            if np.isfinite(rtv) and (rtv > 0.0):
                sigma_g2 = float(rtv / float(max(1, n_m - p_m)))
                if np.isfinite(lambda_reml) and lambda_reml >= 0.0:
                    sigma_e2 = float(lambda_reml * sigma_g2)
        except Exception:
            sigma_g2 = float("nan")
            sigma_e2 = float("nan")
            lambda_reml = float("nan")

        if gblup_runtime_state is not None:
            gblup_runtime_state["variance_component_path"] = "fast"
            gblup_runtime_state["variance_component"] = "REML/FaST"
            gblup_runtime_state["backend"] = "pyBLUP/FaST(d)"
            gblup_runtime_state["kernel_mode"] = "d"
            if np.isfinite(sigma_g2):
                gblup_runtime_state["sigma_g2"] = float(sigma_g2)
                gblup_runtime_state["sigma_d2"] = float(sigma_g2)
            if np.isfinite(sigma_e2):
                gblup_runtime_state["sigma_e2"] = float(sigma_e2)
            if np.isfinite(lambda_reml):
                gblup_runtime_state["lambda_reml"] = float(lambda_reml)
            if np.isfinite(pve):
                gblup_runtime_state["h2"] = float(pve)

        if model_state is not None:
            model_state["kind"] = "gblup_kernel_projection"
            model_state["method"] = str(method)
            model_state["effect_kind"] = "kernel_projection"
            model_state["alpha"] = float(np.mean(np.asarray(Y, dtype=np.float64)))
            beta_dom: np.ndarray | None = None
            alpha_vec = np.asarray(getattr(model, "alpha", np.zeros((0, 1))), dtype=np.float64).reshape(-1)
            if int(alpha_vec.size) == int(gdom_train.shape[1]):
                if is_packed_input and _looks_like_packed_payload(Xtrain):
                    beta_dom = _kernel_projection_beta_from_packed(
                        packed_ctx=typing.cast(dict[str, typing.Any], Xtrain),
                        sample_indices=np.asarray(train_abs, dtype=np.int64),
                        alpha=alpha_vec,
                        mode="d",
                        coef=1.0,
                        block_rows=4096,
                        threads=max(0, int(n_jobs)),
                    )
                if beta_dom is None:
                    beta_dom = _kernel_projection_beta_from_dense(
                        marker_by_sample=gdom_train,
                        alpha=alpha_vec,
                        coef=1.0,
                    )
            if beta_dom is None:
                beta_u = np.asarray(getattr(model, "u", np.zeros((0, 1))), dtype=np.float64).reshape(-1)
                if int(beta_u.size) == int(gdom_train.shape[0]):
                    beta_dom = np.ascontiguousarray(beta_u, dtype=np.float64)
            model_state["kernel_projection"] = (
                np.ascontiguousarray(beta_dom, dtype=np.float64)
                if beta_dom is not None
                else None
            )
            model_state["beta"] = model_state["kernel_projection"]
            model_state["packed"] = bool(is_packed_input)
            model_state["standardized"] = False
            model_state["pve"] = float(pve)
            model_state["kernel_mode"] = "d"
        return np.asarray(yhat_train, dtype=float), np.asarray(yhat_test, dtype=float), pve

    # Linear mixed models
    if method in ("GBLUP", "rrBLUP"):
        n_train_local = int(Y.reshape(-1).shape[0])
        n_snp_local = int(
            _packed_ctx_active_rows(typing.cast(dict[str, typing.Any], Xtrain))
            if _looks_like_packed_payload(Xtrain)
            else np.asarray(Xtrain).shape[0]
        )
        resolved_rr_solver = (
            _resolve_rrblup_solver(
                solver=str(rrblup_solver),
                n_train=n_train_local,
                n_snp=n_snp_local,
                cfg=rrblup_adamw_cfg,
            )
            if method == "rrBLUP"
            else "exact"
        )
        requested_rr_solver = str(rrblup_solver).strip().lower()
        if requested_rr_solver not in {"exact", "adamw", "pcg", "auto"}:
            requested_rr_solver = "auto"
        kinship = 1 if method == "GBLUP" else None

        def _set_rrblup_solver_state(
            solver_name: str,
            fallback_reason: str | None = None,
        ) -> None:
            nonlocal resolved_rr_solver
            resolved_rr_solver = str(solver_name).strip().lower()
            if (method != "rrBLUP") or (rrblup_runtime_state is None):
                return
            rrblup_runtime_state["solver_requested"] = str(requested_rr_solver)
            rrblup_runtime_state["solver_effective"] = str(resolved_rr_solver)
            rrblup_runtime_state["solver"] = str(resolved_rr_solver)
            if fallback_reason is not None and str(fallback_reason).strip() != "":
                rrblup_runtime_state["solver_fallback_reason"] = str(fallback_reason).strip()
            elif "solver_fallback_reason" in rrblup_runtime_state:
                rrblup_runtime_state.pop("solver_fallback_reason", None)

        if method == "rrBLUP":
            _set_rrblup_solver_state(str(resolved_rr_solver))

        def _emit_rrblup_progress(event: str, **payload: typing.Any) -> None:
            if rrblup_progress_hook is None:
                return
            try:
                rrblup_progress_hook(str(event), dict(payload))
            except Exception:
                return

        def _sync_rrblup_he_state(
            runtime_state: dict[str, typing.Any],
            lambda_info: dict[str, typing.Any] | None,
        ) -> None:
            he_keys = (
                "lambda_he_sigma_g2",
                "lambda_he_sigma_e2",
                "lambda_he_h2",
                "lambda_he_converged",
                "lambda_he_iters",
                "lambda_he_rel_res",
                "lambda_he_lambda_k",
                "lambda_he_lambda_equation",
                "lambda_he_raw_k",
                "lambda_he_raw_equation",
                "lambda_he_trace_samples_used",
                "lambda_he_tr_k2",
                "lambda_he_tr_k2_solve",
                "lambda_he_y_ky",
                "lambda_he_y_y",
                "lambda_he_tr_k",
                "lambda_he_tr_p",
                "lambda_he_nnls_projected",
                "lambda_he_boundary_status",
                "lambda_he_accepted",
                "lambda_he_reject_reason",
                "lambda_he_error",
                "lambda_he_call_path",
                "lambda_he_call_api",
                "lambda_he_thread_policy",
                "lambda_he_stage_blas_threads",
                "lambda_he_stage_rayon_threads",
                "lambda_he_threads_arg",
                "lambda_vc_method",
                "lambda_vc_sigma_g2",
                "lambda_vc_sigma_e2",
                "lambda_vc_pve",
            )
            if lambda_info is None:
                for k in he_keys:
                    runtime_state.pop(str(k), None)
                return

            def _set_float(dst: str, src: str) -> None:
                raw = lambda_info.get(src, None)
                if raw is None:
                    runtime_state.pop(dst, None)
                    return
                try:
                    val = float(raw)
                except Exception:
                    runtime_state.pop(dst, None)
                    return
                if np.isfinite(val):
                    runtime_state[dst] = float(val)
                else:
                    runtime_state.pop(dst, None)

            def _set_int(dst: str, src: str) -> None:
                raw = lambda_info.get(src, None)
                if raw is None:
                    runtime_state.pop(dst, None)
                    return
                try:
                    runtime_state[dst] = int(raw)
                except Exception:
                    runtime_state.pop(dst, None)

            def _set_bool(dst: str, src: str) -> None:
                raw = lambda_info.get(src, None)
                if raw is None:
                    runtime_state.pop(dst, None)
                    return
                runtime_state[dst] = bool(raw)

            def _set_str(dst: str, src: str) -> None:
                raw = lambda_info.get(src, None)
                if raw is None:
                    runtime_state.pop(dst, None)
                    return
                txt = str(raw).strip()
                if txt == "":
                    runtime_state.pop(dst, None)
                else:
                    runtime_state[dst] = txt

            _set_float("lambda_he_sigma_g2", "sigma_g2")
            _set_float("lambda_he_sigma_e2", "sigma_e2")
            _set_float("lambda_he_h2", "h2")
            _set_bool("lambda_he_converged", "he_converged")
            _set_int("lambda_he_iters", "he_iters")
            _set_float("lambda_he_rel_res", "he_rel_res")
            _set_float("lambda_he_lambda_k", "lambda_he_lambda_k")
            _set_float("lambda_he_lambda_equation", "lambda_he_lambda_equation")
            _set_float("lambda_he_raw_k", "lambda_he_raw_k")
            _set_float("lambda_he_raw_equation", "lambda_he_raw_equation")
            _set_int("lambda_he_trace_samples_used", "trace_samples_used")
            _set_float("lambda_he_tr_k2", "he_tr_k2")
            _set_float("lambda_he_tr_k2_solve", "he_tr_k2_solve")
            _set_float("lambda_he_y_ky", "he_y_ky")
            _set_float("lambda_he_y_y", "he_y_y")
            _set_float("lambda_he_tr_k", "he_tr_k")
            _set_float("lambda_he_tr_p", "he_tr_p")
            _set_bool("lambda_he_nnls_projected", "he_nnls_projected")
            _set_str("lambda_he_boundary_status", "he_boundary_status")
            _set_bool("lambda_he_accepted", "he_accepted")
            _set_str("lambda_he_reject_reason", "he_reject_reason")
            _set_str("lambda_he_error", "he_error")
            _set_str("lambda_he_call_path", "he_call_path")
            _set_str("lambda_he_call_api", "he_call_api")
            _set_str("lambda_he_thread_policy", "thread_policy")
            _set_int("lambda_he_stage_blas_threads", "stage_blas_threads")
            _set_int("lambda_he_stage_rayon_threads", "stage_rayon_threads")
            _set_int("lambda_he_threads_arg", "he_threads_arg")
            _set_str("lambda_vc_method", "vc_method")
            _set_float("lambda_vc_sigma_g2", "vc_sigma_g2")
            _set_float("lambda_vc_sigma_e2", "vc_sigma_e2")
            _set_float("lambda_vc_pve", "vc_pve")

        if _GS_DEBUG_STAGE:
            t_fit = time.time()
            n_train_dbg = int(
                packed_train_indices.shape[0]
                if (packed_train_indices is not None)
                else (Xtrain.shape[1] if not is_packed_input else int(Y.shape[0]))
            )
            n_test_dbg = int(
                packed_test_indices.shape[0]
                if (packed_test_indices is not None)
                else (Xtest.shape[1] if not is_packed_input else 0)
            )
            n_snp_dbg = int(
                _packed_ctx_active_rows(typing.cast(dict[str, typing.Any], Xtrain))
                if _looks_like_packed_payload(Xtrain)
                else Xtrain.shape[0]
            )
            print(
                f"[GS-DEBUG] GSapi start method={method} "
                f"n_train={n_train_dbg} n_test={n_test_dbg} n_snp={n_snp_dbg} "
                f"force_fast={int(bool(force_fast))} rr_solver={resolved_rr_solver}",
                flush=True,
            )
        if method == "GBLUP":
            can_use_rust_gblup = bool(
                is_packed_input
                and _looks_like_packed_payload(Xtrain)
                and (_jxrs is not None)
                and hasattr(_jxrs, "gblup_reml_packed_bed")
            )
            rust_backend = str(detect_rust_blas_backend()).strip().lower() if can_use_rust_gblup else "unknown"
            allowed_rust_backends = _rust_gblup_allowed_blas_backends()
            if can_use_rust_gblup and (rust_backend not in allowed_rust_backends):
                _warn_rust_gblup_backend_fallback_once(rust_backend, allowed_rust_backends)
                can_use_rust_gblup = False
            if can_use_rust_gblup:
                packed_train = typing.cast(dict[str, typing.Any], Xtrain)
                source_prefix_raw = packed_train.get("source_prefix", None)
                if source_prefix_raw is None or str(source_prefix_raw).strip() == "":
                    raise ValueError(
                        "Packed GBLUP Rust backend requires source PLINK prefix in packed context."
                    )
                source_prefix = str(source_prefix_raw)

                site_keep_raw = packed_train.get("site_keep", None)
                site_keep_arg = None
                if site_keep_raw is not None:
                    site_keep_arg = np.ascontiguousarray(
                        np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
                        dtype=np.bool_,
                    )

                n_train_expected = int(np.asarray(Y).reshape(-1).shape[0])
                if packed_train_indices is None:
                    if int(packed_train["n_samples"]) != n_train_expected:
                        raise ValueError(
                            "Packed GBLUP Rust backend requires packed_train_indices when "
                            "n_train differs from packed n_samples."
                        )
                    train_abs = np.ascontiguousarray(
                        np.arange(int(packed_train["n_samples"]), dtype=np.int64),
                        dtype=np.int64,
                    )
                else:
                    train_abs = np.ascontiguousarray(
                        np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    )
                if int(train_abs.shape[0]) != n_train_expected:
                    raise ValueError(
                        f"Packed GBLUP Rust backend train index mismatch: got {train_abs.shape[0]}, "
                        f"expected {n_train_expected}."
                    )

                test_abs: np.ndarray | None
                if (packed_test_indices is None) or (int(np.asarray(packed_test_indices).size) == 0):
                    test_abs = None
                else:
                    test_abs = np.ascontiguousarray(
                        np.asarray(packed_test_indices, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    )

                if need_train_pred:
                    if train_pred_idx is None:
                        train_pred_local_arg = None
                    else:
                        train_pred_local_arg = np.ascontiguousarray(
                            np.asarray(train_pred_idx, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        )
                else:
                    train_pred_local_arg = np.zeros((0,), dtype=np.int64)

                y_train_vec = np.ascontiguousarray(
                    np.asarray(Y, dtype=np.float64).reshape(-1),
                    dtype=np.float64,
                )
                gblup_g_eps = float(max(0.0, float(np.finfo(np.float64).eps)))
                gblup_reml_low = -6.0
                gblup_reml_high = 6.0
                gblup_reml_max_iter = 50
                gblup_reml_tol = 1e-4
                gblup_block_rows = _parse_nonnegative_int(
                    os.getenv("JX_GBLUP_PACKED_BLOCK_ROWS", "4096")
                )
                gblup_block_rows = int(max(1, int(4096 if gblup_block_rows is None else gblup_block_rows)))
                gblup_return_variance_components = bool(
                    (gblup_runtime_state is not None)
                    or _env_truthy("JX_GBLUP_PACKED_RETURN_VC", "0")
                )
                gblup_return_effect = bool(
                    (model_state is not None)
                    and _cfg_truthy(os.getenv("JX_GS_GBLUP_RETURN_EFFECT", "1"), default=True)
                )

                if _GS_DEBUG_STAGE:
                    rust_blas_before = get_rust_blas_threads()
                    print(
                        "[GS-DEBUG] GBLUP additive packed route=rust_gblup_reml_packed_bed "
                        f"threads={int(max(1, int(n_jobs)))}, "
                        f"stage_threads=(blas={int(max(1, int(n_jobs)))},rayon=1), "
                        f"rust_blas_threads_before={rust_blas_before}",
                        flush=True,
                    )
                rust_blas_in_stage: int | None = None
                gblup_effect_vec = None
                with runtime_thread_stage(
                    blas_threads=int(max(1, int(n_jobs))),
                    rayon_threads=1,
                ):
                    rust_blas_in_stage = get_rust_blas_threads()
                    try:
                        g_out = _jxrs.gblup_reml_packed_bed(  # type: ignore[union-attr]
                            source_prefix,
                            train_abs,
                            y_train_vec,
                            test_abs,
                            train_pred_local_arg,
                            site_keep_arg,
                            float(gblup_g_eps),
                            float(gblup_reml_low),
                            float(gblup_reml_high),
                            int(gblup_reml_max_iter),
                            float(gblup_reml_tol),
                            int(gblup_block_rows),
                            int(max(1, int(n_jobs))),
                            bool(gblup_return_variance_components),
                            False,
                            bool(gblup_return_effect),
                        )
                    except TypeError:
                        try:
                            g_out = _jxrs.gblup_reml_packed_bed(  # type: ignore[union-attr]
                                source_prefix,
                                train_abs,
                                y_train_vec,
                                test_abs,
                                train_pred_local_arg,
                                site_keep_arg,
                                float(gblup_g_eps),
                                float(gblup_reml_low),
                                float(gblup_reml_high),
                                int(gblup_reml_max_iter),
                                float(gblup_reml_tol),
                                int(gblup_block_rows),
                                int(max(1, int(n_jobs))),
                                bool(gblup_return_variance_components),
                            )
                        except TypeError:
                            g_out = _jxrs.gblup_reml_packed_bed(  # type: ignore[union-attr]
                                source_prefix,
                                train_abs,
                                y_train_vec,
                                test_abs,
                                train_pred_local_arg,
                                site_keep_arg,
                                float(gblup_g_eps),
                                float(gblup_reml_low),
                                float(gblup_reml_high),
                                int(gblup_reml_max_iter),
                                float(gblup_reml_tol),
                                int(gblup_block_rows),
                                int(max(1, int(n_jobs))),
                            )
                    g_t = tuple(g_out)
                    if len(g_t) >= 12:
                        (
                            pred_train_raw,
                            pred_test_raw,
                            pve,
                            _lambda_opt,
                            _ml,
                            _reml,
                            _eigh_backend,
                            _eigh_elapsed,
                            _m_eff,
                            _sigma_g2,
                            _sigma_e2,
                            gblup_effect_vec,
                        ) = g_t[:12]
                    elif len(g_t) >= 11:
                        (
                            pred_train_raw,
                            pred_test_raw,
                            pve,
                            _lambda_opt,
                            _ml,
                            _reml,
                            _eigh_backend,
                            _eigh_elapsed,
                            _m_eff,
                            _sigma_g2,
                            _sigma_e2,
                        ) = g_t[:11]
                        gblup_effect_vec = None
                    elif len(g_t) >= 9:
                        (
                            pred_train_raw,
                            pred_test_raw,
                            pve,
                            _lambda_opt,
                            _ml,
                            _reml,
                            _eigh_backend,
                            _eigh_elapsed,
                            _m_eff,
                        ) = g_t[:9]
                        _sigma_g2 = float("nan")
                        _sigma_e2 = float("nan")
                        gblup_effect_vec = None
                    else:
                        raise RuntimeError(
                            "gblup_reml_packed_bed returned unexpected payload size: "
                            f"{len(g_t)}"
                        )
                if gblup_runtime_state is not None:
                    _evd_backend_l = str(_eigh_backend).strip().lower()
                    _vc_path = (
                        "fast"
                        if _evd_backend_l.startswith("marker_fast")
                        else "grm"
                    )
                    gblup_runtime_state["rust_backend"] = str(rust_backend)
                    gblup_runtime_state["eigh_backend"] = str(_eigh_backend)
                    gblup_runtime_state["eigh_sec"] = float(_eigh_elapsed)
                    gblup_runtime_state["lambda_reml"] = float(_lambda_opt)
                    gblup_runtime_state["ml"] = float(_ml)
                    gblup_runtime_state["reml"] = float(_reml)
                    gblup_runtime_state["m_effective"] = int(_m_eff)
                    gblup_runtime_state["variance_component_path"] = str(_vc_path)
                    gblup_runtime_state["variance_component"] = (
                        "REML/FaST" if _vc_path == "fast" else "REML/GRM"
                    )
                    if np.isfinite(float(_sigma_g2)):
                        gblup_runtime_state["sigma_g2"] = float(_sigma_g2)
                    if np.isfinite(float(_sigma_e2)):
                        gblup_runtime_state["sigma_e2"] = float(_sigma_e2)
                if (model_state is not None) and (gblup_return_effect or (gblup_effect_vec is not None)):
                    model_state["kind"] = "gblup_kernel_projection"
                    model_state["method"] = str(method)
                    model_state["effect_kind"] = "kernel_projection"
                    model_state["alpha"] = float(np.mean(np.asarray(Y, dtype=np.float64)))
                    model_state["kernel_projection"] = (
                        np.ascontiguousarray(
                            np.asarray(gblup_effect_vec, dtype=np.float64).reshape(-1),
                            dtype=np.float64,
                        )
                        if gblup_effect_vec is not None
                        else None
                    )
                    model_state["beta"] = model_state["kernel_projection"]
                    model_state["packed"] = True
                    model_state["standardized"] = False
                    model_state["pve"] = float(pve)
                pred_train = (
                    np.zeros((0, 1), dtype=float)
                    if (not need_train_pred)
                    else np.asarray(pred_train_raw, dtype=float).reshape(-1, 1)
                )
                pred_test = np.asarray(pred_test_raw, dtype=float).reshape(-1, 1)
                if _GS_DEBUG_STAGE:
                    rust_blas_after = get_rust_blas_threads()
                    print(
                        f"[GS-DEBUG] GSapi model_fit_done method={method}(rust-packed) "
                        f"elapsed={time.time() - t_fit:.3f}s "
                        f"eigh_backend={str(_eigh_backend)} "
                        f"eigh_sec={float(_eigh_elapsed):.3f} "
                        f"rust_blas_threads_in_stage={rust_blas_in_stage} "
                        f"rust_blas_threads_after={rust_blas_after}",
                        flush=True,
                    )
                return pred_train, pred_test, float(pve)
        if method == "rrBLUP" and resolved_rr_solver == "pcg":
            if not is_packed_input:
                if requested_rr_solver == "pcg":
                    raise ValueError(
                        "rrBLUP solver=pcg requires packed PLINK input (bfile) in this build."
                    )
                _set_rrblup_solver_state("exact", "packed_input_required_for_pcg")
            elif (_jxrs is None) or (not hasattr(_jxrs, "rrblup_pcg_bed")):
                if requested_rr_solver == "pcg":
                    raise RuntimeError(
                        "Rust rrBLUP-PCG kernel is unavailable. Rebuild/install JanusX extension."
                    )
                _set_rrblup_solver_state("exact", "rust_rrblup_pcg_kernel_unavailable")
            else:
                packed_train = typing.cast(dict[str, typing.Any], Xtrain)
                pcg_debug_mode = bool((rrblup_adamw_cfg or {}).get("debug_mode", False)) or bool(
                    _GS_DEBUG_STAGE
                )
                source_prefix_raw = packed_train.get("source_prefix", None)
                source_prefix = (
                    ""
                    if source_prefix_raw is None
                    else str(source_prefix_raw).strip()
                )
                packed_src_view = np.asarray(packed_train["packed"], dtype=np.uint8)
                packed_arg = np.ascontiguousarray(
                    packed_src_view,
                    dtype=np.uint8,
                )
                packed_is_lazy = _packed_ctx_is_lazy_full(packed_train)
                maf_arg = np.ascontiguousarray(
                    np.asarray(packed_train["maf"], dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
                row_flip_arg = np.ascontiguousarray(
                    np.asarray(_ensure_packed_row_flip_cached(packed_train), dtype=np.bool_).reshape(-1),
                    dtype=np.bool_,
                )
                packed_n_samples = int(packed_train["n_samples"])
                if packed_n_samples <= 0:
                    raise ValueError("Packed rrBLUP-PCG requires n_samples > 0 in packed context.")
                if packed_arg.ndim != 2:
                    raise ValueError("Packed rrBLUP-PCG requires packed array shape (m, bytes_per_snp).")
                if int(packed_arg.shape[0]) != int(maf_arg.shape[0]):
                    raise ValueError(
                        "Packed rrBLUP-PCG payload mismatch: packed SNP rows != maf length."
                    )
                if int(packed_arg.shape[0]) != int(row_flip_arg.shape[0]):
                    raise ValueError(
                        "Packed rrBLUP-PCG payload mismatch: packed SNP rows != row_flip length."
                    )
                expected_bps = int((packed_n_samples + 3) // 4)
                if int(packed_arg.shape[1]) != expected_bps:
                    raise ValueError(
                        f"Packed rrBLUP-PCG payload mismatch: packed bytes_per_snp={packed_arg.shape[1]} "
                        f"!= expected {expected_bps}."
                    )
                site_keep_raw = packed_train.get("site_keep", None)
                site_keep_arg_legacy = None
                if site_keep_raw is not None:
                    site_keep_arg_legacy = np.ascontiguousarray(
                        np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1),
                        dtype=np.bool_,
                    )
                site_keep_arg = None
                if site_keep_arg_legacy is not None and (
                    packed_is_lazy
                    or int(site_keep_arg_legacy.shape[0]) == int(packed_arg.shape[0])
                ):
                    site_keep_arg = site_keep_arg_legacy

                n_train_expected = int(np.asarray(Y).reshape(-1).shape[0])
                if packed_train_indices is None:
                    if int(packed_train["n_samples"]) != n_train_expected:
                        raise ValueError(
                            "Packed rrBLUP-PCG requires packed_train_indices when n_train "
                            "differs from packed n_samples."
                        )
                    train_abs = np.ascontiguousarray(
                        np.arange(int(packed_train["n_samples"]), dtype=np.int64)
                    )
                else:
                    train_abs = np.ascontiguousarray(
                        np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    )
                if int(train_abs.shape[0]) != n_train_expected:
                    raise ValueError(
                        f"Packed rrBLUP-PCG train index mismatch: got {train_abs.shape[0]}, "
                        f"expected {n_train_expected}."
                    )

                test_abs: np.ndarray | None
                if (packed_test_indices is None) or (int(np.asarray(packed_test_indices).size) == 0):
                    test_abs = None
                else:
                    test_abs = np.ascontiguousarray(
                        np.asarray(packed_test_indices, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    )

                rr_cfg = dict(rrblup_adamw_cfg or {})
                lambda_raw = float(rr_cfg.get("lambda_value", 1.0))
                lambda_scale = str(rr_cfg.get("lambda_scale", "equation")).strip().lower()
                stage_rayon_threads = int(max(1, int(n_jobs)))
                if (not np.isfinite(lambda_raw)) or (lambda_raw < 0.0):
                    raise ValueError(f"rrBLUP lambda must be finite and >= 0, got {lambda_raw!r}.")
                if lambda_scale not in {"equation", "mean-loss"}:
                    raise ValueError(
                        "rrBLUP lambda_scale must be one of {'equation', 'mean-loss'}, "
                        f"got {lambda_scale!r}."
                    )
                lambda_equation = _rrblup_lambda_raw_to_equation(
                    lambda_raw=float(lambda_raw),
                    lambda_scale=lambda_scale,
                    n_train=int(n_train_local),
                )
                lambda_source = "manual"
                lambda_auto_info: dict[str, typing.Any] | None = None
                lambda_auto_enabled = _cfg_truthy(rr_cfg.get("lambda_auto", "on"), default=True)
                if bool(lambda_auto_enabled):
                    lambda_auto_info = _estimate_rrblup_lambda_subsample_reml(
                        y_train=np.asarray(Y, dtype=np.float64).reshape(-1),
                        packed_ctx=packed_train,
                        train_sample_indices=train_abs,
                        n_jobs=max(1, int(n_jobs)),
                        cfg=rr_cfg,
                        progress_hook=lambda event, payload: _emit_rrblup_progress(
                            str(event),
                            **dict(payload),
                        ),
                    )
                    lam_auto = float(lambda_auto_info.get("lambda_equation", np.nan))
                    lam_auto_strategy = (
                        str(lambda_auto_info.get("strategy", "lambda_auto")).strip() or "lambda_auto"
                    )
                    if np.isfinite(lam_auto) and lam_auto >= 0.0:
                        lambda_equation = float(lam_auto)
                        lambda_raw = _rrblup_lambda_equation_to_raw(
                            lambda_equation=float(lambda_equation),
                            lambda_scale=lambda_scale,
                            n_train=int(n_train_local),
                        )
                        lambda_source = lam_auto_strategy
                    else:
                        lambda_source = f"{lam_auto_strategy}_fallback_manual"
                if (not np.isfinite(lambda_equation)) or (lambda_equation < 0.0):
                    lambda_equation = float(1e-8)
                lambda_equation = float(max(lambda_equation, 1e-8))
                lambda_raw = _rrblup_lambda_equation_to_raw(
                    lambda_equation=float(lambda_equation),
                    lambda_scale=lambda_scale,
                    n_train=int(n_train_local),
                )
                pcg_tol = float(rr_cfg.get("pcg_tol", 1e-4))
                pcg_max_iter = int(max(1, int(rr_cfg.get("pcg_max_iter", 100))))
                pcg_block_rows = int(
                    max(1, int(rr_cfg.get("pcg_block_rows", rr_cfg.get("snp_block_size", 4096))))
                )
                pcg_std_eps = float(max(np.finfo(np.float32).eps, float(rr_cfg.get("pcg_std_eps", 1e-12))))
                pcg_thread_policy_raw = rr_cfg.get(
                    "pcg_thread_policy",
                    os.getenv(
                        "JX_GS_PCG_THREAD_POLICY",
                        os.getenv(
                            "JX_RRBLUP_PCG_THREAD_POLICY",
                            _RRBLUP_PCG_THREAD_POLICY_DEFAULT,
                        ),
                    ),
                )
                pcg_thread_spec = _resolve_pcg_thread_policy_spec(
                    pcg_thread_policy_raw,
                    int(max(1, int(n_jobs))),
                )

                if not (np.isfinite(pcg_tol) and pcg_tol > 0.0):
                    raise ValueError(f"rrBLUP PCG tol must be finite and > 0, got {pcg_tol!r}.")
                if not (np.isfinite(pcg_std_eps) and pcg_std_eps > 0.0):
                    raise ValueError(
                        f"rrBLUP PCG std_eps must be finite and > 0, got {pcg_std_eps!r}."
                    )
                pve_mode = str(rr_cfg.get("pve_mode", "lambda")).strip().lower()
                if pve_mode not in {"lambda", "trainvar"}:
                    pve_mode = "lambda"
                # For PCG, default to skipping full train prediction unless
                # train-variance PVE is explicitly requested.
                pcg_compute_trainvar = bool(pve_mode == "trainvar")

                if need_train_pred:
                    if train_pred_idx is None:
                        train_pred_local_arg = None
                    else:
                        train_pred_local_arg = np.ascontiguousarray(
                            np.asarray(train_pred_idx, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        )
                else:
                    train_pred_local_arg = np.zeros((0,), dtype=np.int64)

                y_train_vec = np.ascontiguousarray(
                    np.asarray(Y, dtype=np.float64).reshape(-1),
                    dtype=np.float64,
                )
                pcg_rel_res_trace: list[dict[str, typing.Any]] = []
                _emit_rrblup_progress(
                    "pcg_start",
                    stage="pcg",
                    total=int(pcg_max_iter),
                    tol=float(pcg_tol),
                    lambda_equation=float(lambda_equation),
                    thread_policy=str(pcg_thread_spec["policy"]),
                    stage_blas_threads=int(pcg_thread_spec["blas_threads"]),
                    stage_rayon_threads=int(pcg_thread_spec["rayon_threads"]),
                    pcg_threads=int(pcg_thread_spec["pcg_threads"]),
                )
                def _on_pcg_progress(
                    done: int,
                    total_from_backend: int,
                    rel_res_now: float = float("nan"),
                ) -> None:
                    iter_now = int(max(0, int(done)))
                    total_now = int(max(1, int(total_from_backend)))
                    rel_res_f = float(rel_res_now)
                    if (
                        len(pcg_rel_res_trace) == 0
                        or int(pcg_rel_res_trace[-1].get("iter", -1)) != iter_now
                    ):
                        pcg_rel_res_trace.append(
                            {
                                "iter": int(iter_now),
                                "total": int(total_now),
                                "rel_res": float(rel_res_f),
                            }
                        )
                    try:
                        _emit_rrblup_progress(
                            "pcg_iter",
                            stage="pcg",
                            iter=int(iter_now),
                            total=int(total_now),
                            rel_res=float(rel_res_f),
                        )
                    except Exception:
                        return
                pcg_progress_cb = (
                    _on_pcg_progress
                    if ((rrblup_progress_hook is not None) or (rrblup_runtime_state is not None))
                    else None
                )
                pcg_progress_every = (
                    int(max(1, int(rr_cfg.get("pcg_progress_every", 1))))
                    if pcg_progress_cb is not None
                    else 0
                )
                pcg_rss_before: int | None = None
                if pcg_debug_mode:
                    packed_c_contig = bool(getattr(packed_arg, "flags", {}).c_contiguous)
                    packed_same_obj = bool(packed_arg is packed_src_view)
                    try:
                        packed_shares_memory = bool(np.shares_memory(packed_arg, packed_src_view))
                    except Exception:
                        packed_shares_memory = False
                    pcg_block_bytes = int(
                        int(pcg_block_rows) * int(n_train_expected) * np.dtype(np.float32).itemsize
                    )
                    pcg_rss_before = _get_process_rss_bytes()
                    print(
                        (
                            "[rrBLUP-DEBUG] PCG packed payload "
                            f"rows={int(packed_arg.shape[0])} "
                            f"bytes_per_snp={int(packed_arg.shape[1])} "
                            f"c_contig={int(packed_c_contig)} "
                            f"shared_with_ctx={int(packed_shares_memory)} "
                            f"same_obj={int(packed_same_obj)} "
                            f"nbytes={_format_debug_bytes(int(packed_arg.nbytes))}"
                        ),
                        flush=True,
                    )
                    print(
                        (
                            "[rrBLUP-DEBUG] PCG workspace estimate "
                            f"packed={_format_debug_bytes(int(packed_arg.nbytes))} "
                            f"block={_format_debug_bytes(pcg_block_bytes)} "
                            f"rss_before={_format_debug_bytes(pcg_rss_before)}"
                        ),
                        flush=True,
                    )
                try:
                    with runtime_thread_stage(
                        blas_threads=int(pcg_thread_spec["blas_threads"]),
                        rayon_threads=int(pcg_thread_spec["rayon_threads"]),
                    ):
                        pcg_call_kwargs = dict(
                            progress_callback=pcg_progress_cb,
                            progress_every=int(pcg_progress_every),
                            compute_trainvar=bool(pcg_compute_trainvar),
                            packed=packed_arg,
                            packed_n_samples=int(packed_n_samples),
                            maf=maf_arg,
                            row_flip=row_flip_arg,
                            blas_threads=int(pcg_thread_spec["blas_threads"]),
                        )
                        try:
                            rr_out = _jxrs.rrblup_pcg_bed(  # type: ignore[union-attr]
                                source_prefix,
                                train_abs,
                                y_train_vec,
                                test_abs,
                                train_pred_local_arg,
                                site_keep_arg,
                                float(lambda_equation),
                                float(pcg_tol),
                                int(pcg_max_iter),
                                int(pcg_block_rows),
                                float(pcg_std_eps),
                                int(pcg_thread_spec["pcg_threads"]),
                                **pcg_call_kwargs,
                            )
                        except TypeError:
                            pcg_call_kwargs.pop("blas_threads", None)
                            rr_out = _jxrs.rrblup_pcg_bed(  # type: ignore[union-attr]
                                source_prefix,
                                train_abs,
                                y_train_vec,
                                test_abs,
                                train_pred_local_arg,
                                site_keep_arg,
                                float(lambda_equation),
                                float(pcg_tol),
                                int(pcg_max_iter),
                                int(pcg_block_rows),
                                float(pcg_std_eps),
                                int(pcg_thread_spec["pcg_threads"]),
                                **pcg_call_kwargs,
                            )
                except TypeError:
                    _emit_rrblup_progress(
                        "pcg_callback_unavailable",
                        stage="pcg",
                        reason="legacy_rrblup_pcg_bed_signature",
                    )
                    if source_prefix == "":
                        raise RuntimeError(
                            "Current JanusX extension does not support direct packed rrBLUP-PCG "
                            "arguments, and packed context has no source_prefix for legacy fallback."
                        )
                    with runtime_thread_stage(
                        blas_threads=int(pcg_thread_spec["blas_threads"]),
                        rayon_threads=int(pcg_thread_spec["rayon_threads"]),
                    ):
                        rr_out = _jxrs.rrblup_pcg_bed(  # type: ignore[union-attr]
                            source_prefix,
                            train_abs,
                            y_train_vec,
                            test_abs,
                            train_pred_local_arg,
                            site_keep_arg_legacy,
                            float(lambda_equation),
                            float(pcg_tol),
                            int(pcg_max_iter),
                            int(pcg_block_rows),
                            float(pcg_std_eps),
                            int(pcg_thread_spec["pcg_threads"]),
                        )
                finally:
                    if pcg_debug_mode:
                        pcg_rss_after = _get_process_rss_bytes()
                        delta_txt = "NA"
                        if (pcg_rss_before is not None) and (pcg_rss_after is not None):
                            delta_txt = _format_debug_bytes(int(pcg_rss_after) - int(pcg_rss_before))
                        print(
                            (
                                "[rrBLUP-DEBUG] PCG rss after call="
                                f"{_format_debug_bytes(pcg_rss_after)} "
                                f"delta={delta_txt}"
                            ),
                            flush=True,
                        )
                        _emit_debug_malloc_trim("[rrBLUP-DEBUG] PCG")
                rr_out_t = tuple(rr_out)
                beta_export = None
                if len(rr_out_t) >= 10:
                    (
                        pred_train_raw,
                        pred_test_raw,
                        pve_trainvar,
                        pcg_converged,
                        pcg_iters,
                        pcg_rel_res,
                        m_effective,
                        pve_lambda_trace,
                        k_trace_mean,
                        beta_export,
                    ) = rr_out_t[:10]
                elif len(rr_out_t) >= 9:
                    (
                        pred_train_raw,
                        pred_test_raw,
                        pve_trainvar,
                        pcg_converged,
                        pcg_iters,
                        pcg_rel_res,
                        m_effective,
                        pve_lambda_trace,
                        k_trace_mean,
                    ) = rr_out_t[:9]
                elif len(rr_out_t) == 7:
                    (
                        pred_train_raw,
                        pred_test_raw,
                        pve_trainvar,
                        pcg_converged,
                        pcg_iters,
                        pcg_rel_res,
                        m_effective,
                    ) = rr_out_t
                    pve_lambda_trace = float("nan")
                    k_trace_mean = float("nan")
                else:
                    raise RuntimeError(
                        "rrblup_pcg_bed returned unexpected payload size: "
                        f"{len(rr_out_t)}"
                    )
                _emit_rrblup_progress(
                    "pcg_end",
                    stage="pcg",
                    iters=int(pcg_iters),
                    converged=bool(pcg_converged),
                    rel_res=float(pcg_rel_res),
                    total=int(pcg_max_iter),
                )
                if len(pcg_rel_res_trace) == 0:
                    pcg_rel_res_trace.append(
                        {
                            "iter": int(max(0, int(pcg_iters))),
                            "total": int(max(1, int(pcg_max_iter))),
                            "rel_res": float(pcg_rel_res),
                        }
                    )
                elif int(pcg_rel_res_trace[-1].get("iter", -1)) != int(max(0, int(pcg_iters))):
                    pcg_rel_res_trace.append(
                        {
                            "iter": int(max(0, int(pcg_iters))),
                            "total": int(max(1, int(pcg_max_iter))),
                            "rel_res": float(pcg_rel_res),
                        }
                    )

                pred_train = (
                    np.zeros((0, 1), dtype=float)
                    if (not need_train_pred)
                    else np.asarray(pred_train_raw, dtype=float).reshape(-1, 1)
                )
                pred_test = np.asarray(pred_test_raw, dtype=float).reshape(-1, 1)

                m_eff = int(max(1, int(m_effective)))
                pve_lambda_formula = float(m_eff / (m_eff + float(lambda_equation)))
                pve_lambda = (
                    float(pve_lambda_trace)
                    if np.isfinite(float(pve_lambda_trace))
                    else float(pve_lambda_formula)
                )
                pve = float(pve_lambda if pve_mode == "lambda" else float(pve_trainvar))

                _set_rrblup_solver_state("pcg")
                if rrblup_runtime_state is not None:
                    rrblup_runtime_state["pcg_converged"] = bool(pcg_converged)
                    rrblup_runtime_state["pcg_iters"] = int(pcg_iters)
                    rrblup_runtime_state["pcg_rel_res"] = float(pcg_rel_res)
                    rrblup_runtime_state["pcg_rel_res_trace"] = [dict(x) for x in pcg_rel_res_trace]
                    rrblup_runtime_state["m_effective"] = int(m_eff)
                    rrblup_runtime_state["thread_policy"] = str(pcg_thread_spec["policy"])
                    rrblup_runtime_state["stage_blas_threads"] = int(pcg_thread_spec["blas_threads"])
                    rrblup_runtime_state["stage_rayon_threads"] = int(pcg_thread_spec["rayon_threads"])
                    rrblup_runtime_state["pcg_threads_arg"] = int(pcg_thread_spec["pcg_threads"])
                    rrblup_runtime_state["lambda_equation"] = float(lambda_equation)
                    rrblup_runtime_state["selected_lambda"] = float(lambda_raw)
                    rrblup_runtime_state["lambda_source"] = str(lambda_source)
                    rrblup_runtime_state["lambda_auto_enabled"] = bool(lambda_auto_enabled)
                    if lambda_auto_info is not None:
                        rrblup_runtime_state["lambda_subsample_n"] = int(
                            lambda_auto_info.get("n_sub", 0)
                        )
                        rrblup_runtime_state["lambda_subsample_repeats"] = int(
                            lambda_auto_info.get("repeats", 0)
                        )
                        rrblup_runtime_state["lambda_subsample_ok_repeats"] = int(
                            lambda_auto_info.get("ok_repeats", 0)
                        )
                    _sync_rrblup_he_state(rrblup_runtime_state, lambda_auto_info)
                    rrblup_runtime_state["pve_mode_used"] = str(pve_mode)
                    rrblup_runtime_state["pve_used"] = float(pve)
                    rrblup_runtime_state["pve_lambda"] = float(pve_lambda)
                    rrblup_runtime_state["pve_lambda_formula"] = float(pve_lambda_formula)
                    rrblup_runtime_state["pve_lambda_trace"] = float(pve_lambda_trace)
                    rrblup_runtime_state["pve_trainvar"] = float(pve_trainvar)
                    rrblup_runtime_state["k_trace_mean"] = float(k_trace_mean)
                if model_state is not None and beta_export is not None:
                    row_mean_pc, row_inv_pc = _ensure_packed_standard_stats_cached(packed_train)
                    model_state["kind"] = "rrblup_linear"
                    model_state["method"] = "rrBLUP"
                    model_state["solver"] = "pcg"
                    model_state["alpha"] = float(np.mean(np.asarray(Y, dtype=np.float64)))
                    model_state["beta"] = np.ascontiguousarray(
                        np.asarray(beta_export, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    model_state["packed"] = True
                    model_state["standardized"] = True
                    model_state["row_mean"] = np.ascontiguousarray(
                        np.asarray(row_mean_pc, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    model_state["row_inv_sd"] = np.ascontiguousarray(
                        np.asarray(row_inv_pc, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    model_state["snp_block_size"] = int(max(1, int(pcg_block_rows)))
                    model_state["sample_chunk_size"] = int(
                        max(1, int(rr_cfg.get("sample_chunk_size", 4096)))
                    )
                    model_state["pve"] = float(pve)

                if _GS_DEBUG_STAGE:
                    print(
                        f"[GS-DEBUG] GSapi model_fit_done method={method}(pcg) "
                        f"elapsed={time.time() - t_fit:.3f}s iters={int(pcg_iters)} "
                        f"converged={int(bool(pcg_converged))} rel_res={float(pcg_rel_res):.6g}",
                        flush=True,
                    )
                return (
                    np.asarray(pred_train, dtype=float).reshape(-1, 1),
                    np.asarray(pred_test, dtype=float).reshape(-1, 1),
                    float(pve),
                )
        if method == "rrBLUP" and resolved_rr_solver == "adamw":
            rr_cfg_base = dict(rrblup_adamw_cfg or {})
            adamw_lambda_auto_info: dict[str, typing.Any] | None = None
            adamw_lambda_auto_enabled = _cfg_truthy(
                rr_cfg_base.get("lambda_auto", "on"),
                default=True,
            )
            if bool(adamw_lambda_auto_enabled) and bool(is_packed_input):
                try:
                    packed_train = typing.cast(dict[str, typing.Any], Xtrain)
                    n_train_expected = int(np.asarray(Y).reshape(-1).shape[0])
                    if packed_train_indices is None:
                        if int(packed_train["n_samples"]) != n_train_expected:
                            raise ValueError(
                                "Packed rrBLUP-AdamW lambda-auto requires packed_train_indices "
                                "when n_train differs from packed n_samples."
                            )
                        train_abs_auto = np.ascontiguousarray(
                            np.arange(int(packed_train["n_samples"]), dtype=np.int64),
                            dtype=np.int64,
                        )
                    else:
                        train_abs_auto = np.ascontiguousarray(
                            np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        )
                    if int(train_abs_auto.shape[0]) != n_train_expected:
                        raise ValueError(
                            "Packed rrBLUP-AdamW lambda-auto train index mismatch: "
                            f"indices={train_abs_auto.shape[0]}, y={n_train_expected}."
                        )
                    adamw_lambda_auto_info = _estimate_rrblup_lambda_subsample_reml(
                        y_train=np.asarray(Y, dtype=np.float64).reshape(-1),
                        packed_ctx=packed_train,
                        train_sample_indices=train_abs_auto,
                        n_jobs=max(1, int(n_jobs)),
                        cfg=rr_cfg_base,
                        progress_hook=lambda event, payload: _emit_rrblup_progress(
                            str(event),
                            **dict(payload),
                        ),
                    )
                    lam_auto = float(adamw_lambda_auto_info.get("lambda_equation", np.nan))
                    if np.isfinite(lam_auto) and lam_auto > 0.0:
                        lam_auto_raw = _rrblup_lambda_equation_to_raw(
                            lambda_equation=float(lam_auto),
                            lambda_scale=str(rr_cfg_base.get("lambda_scale", "equation")),
                            n_train=int(n_train_local),
                        )
                        rr_cfg_base["lambda_value"] = float(lam_auto_raw)
                except Exception:
                    adamw_lambda_auto_info = None

            auto_grid_enabled = _cfg_truthy(rr_cfg_base.get("auto_grid", "on"), default=True)
            grid_candidates = _build_rrblup_adamw_grid(rr_cfg_base) if auto_grid_enabled else []
            min_grid_samples = int(max(16, int(rr_cfg_base.get("grid_min_samples", 256))))
            use_auto_grid = bool(
                auto_grid_enabled
                and len(grid_candidates) > 1
                and n_train_local >= min_grid_samples
            )
            selected_cfg = dict(rr_cfg_base)
            trial_rows: list[dict[str, typing.Any]] = []

            if use_auto_grid:
                val_idx = _build_rrblup_validation_indices(
                    n_train=n_train_local,
                    cfg=rr_cfg_base,
                )
                if val_idx is not None and int(val_idx.size) > 0:
                    trial_epochs = int(
                        max(
                            1,
                            min(
                                int(rr_cfg_base.get("epochs", 60)),
                                int(rr_cfg_base.get("grid_trial_epochs", 40)),
                            ),
                        )
                    )
                    best_score = np.inf
                    best_row: dict[str, typing.Any] | None = None
                    base_lambda = float(max(float(rr_cfg_base.get("lambda_value", 1.0)), 1e-8))
                    base_lr = float(rr_cfg_base.get("lr", 1e-2))
                    baseline_row: dict[str, typing.Any] | None = None
                    for trial_id, (lam_try, lr_try) in enumerate(grid_candidates, start=1):
                        cfg_trial = dict(rr_cfg_base)
                        cfg_trial["lambda_value"] = float(lam_try)
                        cfg_trial["lr"] = float(lr_try)
                        cfg_trial["epochs"] = int(trial_epochs)
                        cfg_trial["val_indices_local"] = np.ascontiguousarray(
                            np.asarray(val_idx, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        )
                        cfg_trial["exclude_val_from_train"] = True
                        cfg_trial["early_stop_patience"] = int(
                            max(0, int(rr_cfg_base.get("es_patience", 5)))
                        )
                        cfg_trial["early_stop_warmup"] = int(
                            max(1, int(rr_cfg_base.get("es_warmup", 5)))
                        )
                        cfg_trial["early_stop_min_delta"] = float(
                            max(0.0, float(rr_cfg_base.get("es_min_delta", 1e-5)))
                        )
                        fit_trial = _fit_rrblup_adamw_cpu(
                            y=np.asarray(Y, dtype=np.float32).reshape(-1),
                            Xtrain=Xtrain,
                            is_packed_input=bool(is_packed_input),
                            packed_train_indices=packed_train_indices,
                            cfg=cfg_trial,
                            progress_hook=(
                                lambda event, payload, trial_id=trial_id, trial_total=len(grid_candidates):
                                    _emit_rrblup_progress(
                                        event,
                                        stage="grid",
                                        trial_id=int(trial_id),
                                        trial_total=int(trial_total),
                                        **dict(payload),
                                    )
                            ),
                        )
                        trial_val = float(fit_trial.get("best_val_loss", np.nan))
                        trial_epoch = int(max(1, int(fit_trial.get("best_epoch", trial_epochs))))
                        trial_row = {
                            "id": int(trial_id),
                            "lambda": float(lam_try),
                            "lr": float(lr_try),
                            "val_mse": float(trial_val),
                            "best_epoch": int(trial_epoch),
                        }
                        trial_rows.append(trial_row)
                        if np.isclose(float(lam_try), base_lambda) and np.isclose(float(lr_try), base_lr):
                            baseline_row = trial_row
                        if np.isfinite(trial_val) and (trial_val < best_score):
                            best_score = float(trial_val)
                            best_row = trial_row
                    if best_row is not None and baseline_row is not None:
                        best_val = float(best_row["val_mse"])
                        base_val = float(baseline_row["val_mse"])
                        min_rel_improve = float(
                            max(0.0, float(rr_cfg_base.get("grid_switch_min_improve", 0.02)))
                        )
                        if np.isfinite(best_val) and np.isfinite(base_val):
                            rel_improve = float((base_val - best_val) / max(1e-12, abs(base_val)))
                            if rel_improve < min_rel_improve:
                                best_row = baseline_row
                    if best_row is None and len(grid_candidates) > 0:
                        # Guardrail: when all trial scores are non-finite, fall back to
                        # the first grid candidate (anchor lambda) instead of user-input
                        # low-lambda defaults that can destabilize AdamW training.
                        best_row = {
                            "id": 0,
                            "lambda": float(grid_candidates[0][0]),
                            "lr": float(grid_candidates[0][1]),
                            "val_mse": float("nan"),
                            "best_epoch": int(trial_epochs),
                        }
                    if best_row is not None:
                        selected_cfg["lambda_value"] = float(best_row["lambda"])
                        selected_cfg["lr"] = float(best_row["lr"])
                        selected_cfg["epochs"] = int(max(1, int(rr_cfg_base.get("epochs", 60))))
                        if _GS_DEBUG_STAGE:
                            print(
                                "[GS-DEBUG] rrBLUP-AdamW auto-grid selected "
                                f"lambda={float(best_row['lambda']):.6g}, "
                                f"lr={float(best_row['lr']):.6g}, "
                                f"epochs={int(selected_cfg['epochs'])}, "
                                f"val_mse={float(best_row['val_mse']):.6g}",
                                flush=True,
                            )
                elif _GS_DEBUG_STAGE:
                    print(
                        "[GS-DEBUG] rrBLUP-AdamW auto-grid skipped: validation split unavailable.",
                        flush=True,
                    )

            fit = _fit_rrblup_adamw_cpu(
                y=np.asarray(Y, dtype=np.float32).reshape(-1),
                Xtrain=Xtrain,
                is_packed_input=bool(is_packed_input),
                packed_train_indices=packed_train_indices,
                cfg=selected_cfg,
                progress_hook=(
                    lambda event, payload: _emit_rrblup_progress(
                        event,
                        stage="fit",
                        **dict(payload),
                    )
                ),
            )
            if len(trial_rows) > 0:
                fit["auto_grid_trials"] = trial_rows
                fit["auto_grid_selected_lambda"] = float(selected_cfg.get("lambda_value", np.nan))
                fit["auto_grid_selected_lr"] = float(selected_cfg.get("lr", np.nan))
                fit["auto_grid_selected_epochs"] = int(selected_cfg.get("epochs", fit.get("best_epoch", 1)))
            _set_rrblup_solver_state("adamw")
            if rrblup_runtime_state is not None:
                rrblup_runtime_state["auto_grid_enabled"] = bool(use_auto_grid)
                rrblup_runtime_state["selected_lambda"] = float(selected_cfg.get("lambda_value", np.nan))
                rrblup_runtime_state["lambda_source"] = (
                    str(adamw_lambda_auto_info.get("strategy", "lambda_auto")).strip()
                    if (adamw_lambda_auto_info is not None)
                    else "manual"
                )
                rrblup_runtime_state["lambda_auto_enabled"] = bool(adamw_lambda_auto_enabled)
                rrblup_runtime_state["selected_lr"] = float(selected_cfg.get("lr", np.nan))
                rrblup_runtime_state["selected_epochs"] = int(selected_cfg.get("epochs", fit.get("best_epoch", 1)))
                rrblup_runtime_state["best_epoch"] = int(max(1, int(fit.get("best_epoch", 1))))
                rrblup_runtime_state["epochs_ran"] = int(max(1, int(fit.get("epochs_ran", fit.get("best_epoch", 1)))))
                rrblup_runtime_state["stopped_early"] = bool(fit.get("stopped_early", False))
                rrblup_runtime_state["best_val_loss"] = float(fit.get("best_val_loss", np.nan))
                _sync_rrblup_he_state(rrblup_runtime_state, adamw_lambda_auto_info)
                if len(trial_rows) > 0:
                    rrblup_runtime_state["auto_grid_trials"] = trial_rows
            if _GS_DEBUG_STAGE:
                print(
                    f"[GS-DEBUG] GSapi model_fit_done method={method}(adamw) "
                    f"elapsed={time.time() - t_fit:.3f}s",
                    flush=True,
                )
                t_pred = time.time()

            alpha_hat = float(fit["alpha"])
            beta_hat = np.ascontiguousarray(np.asarray(fit["beta"], dtype=np.float32).reshape(-1))
            snp_block = int(fit["snp_block_size"])
            sample_chunk = int(fit["sample_chunk_size"])

            if is_packed_input:
                packed_train = typing.cast(dict[str, typing.Any], Xtrain)
                n_train_expected = int(np.asarray(Y).reshape(-1).shape[0])
                if packed_train_indices is None:
                    if int(packed_train["n_samples"]) != n_train_expected:
                        raise ValueError(
                            "Packed rrBLUP AdamW prediction requires packed_train_indices when n_train differs from packed n_samples."
                        )
                    train_abs = np.ascontiguousarray(
                        np.arange(int(packed_train["n_samples"]), dtype=np.int64)
                    )
                else:
                    train_abs = np.ascontiguousarray(
                        np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    )
                if int(train_abs.shape[0]) != n_train_expected:
                    raise ValueError(
                        f"Packed rrBLUP AdamW train index mismatch: got {train_abs.shape[0]}, expected {n_train_expected}."
                    )
                row_mean = np.ascontiguousarray(
                    np.asarray(fit["row_mean"], dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
                row_inv_sd = np.ascontiguousarray(
                    np.asarray(fit["row_inv_sd"], dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
                pred_train_full = _predict_rrblup_packed_from_beta(
                    packed_ctx=packed_train,
                    sample_indices=train_abs,
                    alpha=alpha_hat,
                    beta=beta_hat,
                    row_mean=row_mean,
                    row_inv_sd=row_inv_sd,
                    snp_block_size=snp_block,
                    sample_chunk_size=sample_chunk,
                )
                if (packed_test_indices is None) or (int(np.asarray(packed_test_indices).size) == 0):
                    pred_test = np.zeros((0, 1), dtype=float)
                else:
                    pred_test = _predict_rrblup_packed_from_beta(
                        packed_ctx=typing.cast(dict[str, typing.Any], Xtest),
                        sample_indices=np.ascontiguousarray(
                            np.asarray(packed_test_indices, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        ),
                        alpha=alpha_hat,
                        beta=beta_hat,
                        row_mean=row_mean,
                        row_inv_sd=row_inv_sd,
                        snp_block_size=snp_block,
                        sample_chunk_size=sample_chunk,
                    )
            else:
                dense_train = np.asarray(Xtrain, dtype=np.float32)
                pred_train_full = _predict_rrblup_dense_from_beta(
                    X=dense_train,
                    alpha=alpha_hat,
                    beta=beta_hat,
                    snp_block_size=snp_block,
                )
                dense_test = np.asarray(Xtest, dtype=np.float32)
                if int(dense_test.shape[1]) <= 0:
                    pred_test = np.zeros((0, 1), dtype=float)
                else:
                    pred_test = _predict_rrblup_dense_from_beta(
                        X=dense_test,
                        alpha=alpha_hat,
                        beta=beta_hat,
                        snp_block_size=snp_block,
                    )

            y_train_vec = np.asarray(Y, dtype=np.float64).reshape(-1)
            pred_train_vec = np.asarray(pred_train_full, dtype=np.float64).reshape(-1)
            g_hat = pred_train_vec - float(alpha_hat)
            var_g = float(np.var(g_hat, ddof=1))
            var_e = float(np.var(y_train_vec - pred_train_vec, ddof=1))
            denom = var_g + var_e
            pve_trainvar = float(var_g / denom) if (np.isfinite(denom) and denom > 0.0) else float("nan")
            lambda_eq = float(fit.get("lambda_equation", np.nan))
            m_eff = int(max(1, int(fit.get("m_effective", int(beta_hat.shape[0])))))
            pve_lambda = float("nan")
            if np.isfinite(lambda_eq) and lambda_eq >= 0.0:
                pve_lambda = float(m_eff / (m_eff + lambda_eq))
            rr_cfg = rrblup_adamw_cfg or {}
            pve_mode = str(rr_cfg.get("pve_mode", "lambda")).strip().lower()
            if pve_mode not in {"lambda", "trainvar"}:
                pve_mode = "lambda"
            pve = pve_lambda if (pve_mode == "lambda") else pve_trainvar

            if rrblup_runtime_state is not None:
                rrblup_runtime_state["pve_mode_used"] = str(pve_mode)
                rrblup_runtime_state["pve_used"] = float(pve)
                rrblup_runtime_state["pve_lambda"] = float(pve_lambda)
                rrblup_runtime_state["pve_trainvar"] = float(pve_trainvar)
                rrblup_runtime_state["lambda_equation"] = float(lambda_eq)
                rrblup_runtime_state["m_effective"] = int(m_eff)
            if model_state is not None:
                model_state["kind"] = "rrblup_linear"
                model_state["method"] = "rrBLUP"
                model_state["solver"] = "adamw"
                model_state["alpha"] = float(alpha_hat)
                model_state["beta"] = np.ascontiguousarray(
                    np.asarray(beta_hat, dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
                model_state["packed"] = bool(is_packed_input)
                model_state["standardized"] = True
                if is_packed_input:
                    model_state["row_mean"] = np.ascontiguousarray(
                        np.asarray(row_mean, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    model_state["row_inv_sd"] = np.ascontiguousarray(
                        np.asarray(row_inv_sd, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                model_state["snp_block_size"] = int(max(1, int(snp_block)))
                model_state["sample_chunk_size"] = int(max(1, int(sample_chunk)))
                model_state["pve"] = float(pve)

            if need_train_pred:
                if train_pred_idx is None:
                    pred_train = pred_train_full
                elif int(train_pred_idx.size) == 0:
                    pred_train = np.zeros((0, 1), dtype=float)
                else:
                    pred_train = np.asarray(
                        np.asarray(pred_train_full, dtype=np.float64)[train_pred_idx],
                        dtype=np.float64,
                    ).reshape(-1, 1)
            else:
                pred_train = np.zeros((0, 1), dtype=float)
            if _GS_DEBUG_STAGE:
                print(
                    f"[GS-DEBUG] GSapi predict_done method={method}(adamw) "
                    f"elapsed={time.time() - t_pred:.3f}s pve_mode={pve_mode} "
                    f"pve={float(pve):.6g} pve_lambda={float(pve_lambda):.6g} "
                    f"pve_trainvar={float(pve_trainvar):.6g}",
                    flush=True,
                )
            return (
                np.asarray(pred_train, dtype=float).reshape(-1, 1),
                np.asarray(pred_test, dtype=float).reshape(-1, 1),
                float(pve),
            )

        model = MLMBLUP(
            Y.reshape(-1, 1),
            Xtrain,
            kinship=kinship,
            sample_indices=packed_train_indices,
            force_fast=bool(force_fast),
        )
        if _GS_DEBUG_STAGE:
            print(
                f"[GS-DEBUG] GSapi model_fit_done method={method} "
                f"elapsed={time.time() - t_fit:.3f}s",
                flush=True,
            )
            t_pred = time.time()
        if need_train_pred:
            if train_pred_idx is None:
                pred_train = model.predict(Xtrain, sample_indices=packed_train_indices)
            elif int(train_pred_idx.size) == 0:
                pred_train = np.zeros((0, 1), dtype=float)
            else:
                if is_packed_input:
                    base_indices = (
                        np.arange(int(Y.reshape(-1).shape[0]), dtype=np.int64)
                        if packed_train_indices is None
                        else np.asarray(packed_train_indices, dtype=np.int64).reshape(-1)
                    )
                    picked_indices = np.ascontiguousarray(base_indices[train_pred_idx], dtype=np.int64)
                    pred_train = model.predict(Xtrain, sample_indices=picked_indices)
                else:
                    pred_train = model.predict(np.asarray(Xtrain)[:, train_pred_idx])
        else:
            pred_train = np.zeros((0, 1), dtype=float)
        pred_test = model.predict(Xtest, sample_indices=packed_test_indices)
        if method == "GBLUP" and gblup_runtime_state is not None:
            vc_path = (
                "fast"
                if bool(getattr(model, "_implicit_kinship_fast", False))
                else "grm"
            )
            gblup_runtime_state["variance_component_path"] = str(vc_path)
            gblup_runtime_state["variance_component"] = (
                "REML/FaST" if vc_path == "fast" else "REML/GRM"
            )
            sigma_g2 = float("nan")
            sigma_e2 = float("nan")
            h2 = float("nan")
            lambda_reml = float("nan")
            try:
                var = np.asarray(getattr(model, "var", []), dtype=float).reshape(-1)
                if var.size >= 2 and np.all(np.isfinite(var)):
                    sigma_g2 = float(np.sum(var[:-1]))
                    sigma_e2 = float(var[-1])
                    den = sigma_g2 + sigma_e2
                    if np.isfinite(den) and den > 0.0:
                        h2 = float(sigma_g2 / den)
                    if np.isfinite(sigma_g2) and sigma_g2 > 0.0 and np.isfinite(sigma_e2):
                        lambda_reml = float(sigma_e2 / sigma_g2)
            except Exception:
                sigma_g2 = float("nan")
                sigma_e2 = float("nan")
                h2 = float("nan")
                lambda_reml = float("nan")
            if (not np.isfinite(sigma_g2)) or (not np.isfinite(sigma_e2)):
                # Packed/FaST paths in MLMBLUP may not populate `model.var`.
                # Recover lambda from spectral V_inv and derive (sigma_g2, sigma_e2).
                try:
                    rtv_invr = float(getattr(model, "rTV_invr", np.nan))
                    n_model = int(getattr(model, "n", int(Y.reshape(-1).shape[0])))
                    p_model = int(getattr(model, "p", 1))
                    n_eff_model = int(max(1, n_model - p_model))
                    svals = np.asarray(getattr(model, "S", []), dtype=float).reshape(-1)
                    v_inv = np.asarray(getattr(model, "V_inv", []), dtype=float).reshape(-1)
                    lam = float("nan")
                    if svals.size > 0 and v_inv.size == svals.size:
                        eps_v = float(np.finfo(np.float64).eps)
                        valid = np.isfinite(svals) & np.isfinite(v_inv) & (v_inv > eps_v)
                        if np.any(valid):
                            lam_cand = (1.0 / v_inv[valid]) - svals[valid]
                            lam_cand = lam_cand[np.isfinite(lam_cand) & (lam_cand >= 0.0)]
                            if lam_cand.size > 0:
                                lam = float(np.median(lam_cand))
                    if np.isfinite(rtv_invr) and rtv_invr > 0.0 and np.isfinite(lam) and lam >= 0.0:
                        sigma_g2 = float(rtv_invr / float(max(1, n_eff_model)))
                        sigma_e2 = float(lam * sigma_g2)
                        den = sigma_g2 + sigma_e2
                        if np.isfinite(den) and den > 0.0:
                            h2 = float(sigma_g2 / den)
                        if np.isfinite(sigma_g2) and sigma_g2 > 0.0 and np.isfinite(sigma_e2):
                            lambda_reml = float(sigma_e2 / sigma_g2)
                except Exception:
                    pass
            if np.isfinite(sigma_g2):
                gblup_runtime_state["sigma_g2"] = float(sigma_g2)
            if np.isfinite(sigma_e2):
                gblup_runtime_state["sigma_e2"] = float(sigma_e2)
            if np.isfinite(h2):
                gblup_runtime_state["h2"] = float(h2)
            if np.isfinite(lambda_reml):
                gblup_runtime_state["lambda_reml"] = float(lambda_reml)
        if method == "rrBLUP":
            _set_rrblup_solver_state(str(resolved_rr_solver))
        if method == "rrBLUP" and rrblup_runtime_state is not None:
            rrblup_runtime_state["pve_used"] = float(model.pve)
            rrblup_runtime_state["pve_exact"] = float(model.pve)
            # Exact rrBLUP follows REML/variance-component PVE from pyBLUP.
            # Keep trainvar slot aligned to the reported PVE for unified logging.
            rrblup_runtime_state["pve_trainvar"] = float(model.pve)
            rrblup_runtime_state["pve_lambda"] = float("nan")
            rrblup_runtime_state["pve_mode_used"] = "exact_reml"
        if method == "rrBLUP" and model_state is not None:
            beta_fix = np.asarray(getattr(model, "beta", np.asarray([[0.0]], dtype=np.float64)), dtype=np.float64)
            alpha0 = float(beta_fix.reshape(-1)[0]) if int(beta_fix.size) > 0 else 0.0
            u_vec = np.ascontiguousarray(
                np.asarray(getattr(model, "u", np.zeros((0, 1), dtype=np.float64)), dtype=np.float64).reshape(-1),
                dtype=np.float64,
            )
            model_state["kind"] = "rrblup_linear"
            model_state["method"] = "rrBLUP"
            model_state["solver"] = "exact"
            model_state["alpha"] = float(alpha0)
            model_state["beta"] = u_vec
            model_state["packed"] = bool(is_packed_input)
            if bool(is_packed_input):
                model_state["standardized"] = False
            else:
                model_state["standardized"] = True
            model_state["snp_block_size"] = 2048
            model_state["sample_chunk_size"] = 4096
            model_state["pve"] = float(model.pve)
        if method == "GBLUP" and model_state is not None:
            n_train_cur = int(np.asarray(Y).reshape(-1).shape[0])
            alpha_vec = np.asarray(getattr(model, "alpha", np.zeros((0, 1))), dtype=np.float64).reshape(-1)
            beta_kp: np.ndarray | None = None
            if int(alpha_vec.size) == n_train_cur:
                if is_packed_input and _looks_like_packed_payload(Xtrain):
                    packed_train_local = typing.cast(dict[str, typing.Any], Xtrain)
                    if packed_train_indices is None:
                        if int(packed_train_local["n_samples"]) == n_train_cur:
                            sample_idx_local = np.ascontiguousarray(
                                np.arange(int(n_train_cur), dtype=np.int64),
                                dtype=np.int64,
                            )
                        else:
                            sample_idx_local = None
                    else:
                        sample_idx_local = np.ascontiguousarray(
                            np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        )
                    if sample_idx_local is not None and int(sample_idx_local.shape[0]) == n_train_cur:
                        beta_kp = _kernel_projection_beta_from_packed(
                            packed_ctx=packed_train_local,
                            sample_indices=sample_idx_local,
                            alpha=alpha_vec,
                            mode="a",
                            coef=1.0,
                            block_rows=4096,
                            threads=max(0, int(n_jobs)),
                        )
                if beta_kp is None and (not is_packed_input):
                    beta_kp = _kernel_projection_beta_from_dense(
                        marker_by_sample=np.asarray(Xtrain, dtype=np.float64),
                        alpha=alpha_vec,
                        coef=1.0,
                    )
            if beta_kp is None:
                u_vec = np.asarray(getattr(model, "u", np.zeros((0, 1))), dtype=np.float64).reshape(-1)
                if not is_packed_input and int(u_vec.size) == int(np.asarray(Xtrain).shape[0]):
                    beta_kp = np.ascontiguousarray(u_vec, dtype=np.float64)
            model_state["kind"] = "gblup_kernel_projection"
            model_state["method"] = "GBLUP"
            model_state["effect_kind"] = "kernel_projection"
            model_state["alpha"] = float(np.mean(np.asarray(Y, dtype=np.float64)))
            model_state["kernel_projection"] = (
                np.ascontiguousarray(beta_kp, dtype=np.float64)
                if beta_kp is not None
                else None
            )
            model_state["beta"] = model_state["kernel_projection"]
            model_state["packed"] = bool(is_packed_input)
            model_state["standardized"] = False
            model_state["kernel_mode"] = "a"
            model_state["pve"] = float(model.pve)
        if _GS_DEBUG_STAGE:
            print(
                f"[GS-DEBUG] GSapi predict_done method={method} "
                f"elapsed={time.time() - t_pred:.3f}s",
                flush=True,
            )
        return pred_train, pred_test, model.pve

    if method in ("BayesA", "BayesB", "BayesCpi"):
        resolved_bayes_r2: float | None = None
        if bayes_auto_r2 is not None:
            try:
                cand_r2 = float(bayes_auto_r2)
            except Exception:
                cand_r2 = float("nan")
            if np.isfinite(cand_r2):
                resolved_bayes_r2 = float(cand_r2)
        if is_packed_input:
            if not _looks_like_packed_payload(Xtrain):
                raise ValueError("Packed Bayes requires packed train payload.")
            packed_train = typing.cast(dict[str, typing.Any], Xtrain)
            if packed_train_indices is None:
                n_total = int(packed_train["n_samples"])
                if n_total != int(np.asarray(Y).reshape(-1).shape[0]):
                    raise ValueError(
                        "Packed Bayes requires packed_train_indices when n_train differs from packed n_samples."
                    )
                train_abs = np.ascontiguousarray(np.arange(n_total, dtype=np.int64), dtype=np.int64)
            else:
                train_abs = np.ascontiguousarray(
                    np.asarray(packed_train_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                )
            if int(train_abs.shape[0]) != int(np.asarray(Y).reshape(-1).shape[0]):
                raise ValueError(
                    f"Packed Bayes train index mismatch: got {train_abs.shape[0]}, "
                    f"expected {np.asarray(Y).reshape(-1).shape[0]}."
                )
            if packed_test_indices is None:
                test_abs = np.zeros((0,), dtype=np.int64)
            else:
                test_abs = np.ascontiguousarray(
                    np.asarray(packed_test_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                )

            row_mean, row_inv_sd = _ensure_packed_standard_stats_cached(packed_train)
            row_flip = _ensure_packed_row_flip_cached(packed_train)
            packed = np.ascontiguousarray(np.asarray(packed_train["packed"], dtype=np.uint8))
            maf = np.ascontiguousarray(np.asarray(packed_train["maf"], dtype=np.float32).reshape(-1), dtype=np.float32)
            n_total_samples = int(packed_train["n_samples"])
            train_abs = np.ascontiguousarray(train_abs, dtype=np.int64)

            bayes_snp_block = _parse_nonnegative_int(
                os.getenv("JX_BAYES_PACKED_SNP_BLOCK", "2048")
            )
            bayes_sample_chunk = _parse_nonnegative_int(
                os.getenv("JX_BAYES_PACKED_SAMPLE_CHUNK", "4096")
            )
            bayes_snp_block = int(max(1, int(2048 if bayes_snp_block is None else bayes_snp_block)))
            bayes_sample_chunk = int(max(1, int(4096 if bayes_sample_chunk is None else bayes_sample_chunk)))

            packed_func_name = {
                "BayesA": "bayesa_packed",
                "BayesB": "bayesb_packed",
                "BayesCpi": "bayescpi_packed",
            }.get(str(method))
            use_packed_native = bool(
                (_jxrs is not None)
                and (packed_func_name is not None)
                and hasattr(_jxrs, str(packed_func_name))
            )

            if use_packed_native:
                y_vec = np.ascontiguousarray(np.asarray(Y, dtype=np.float64).reshape(-1), dtype=np.float64)
                r2_blup = float("nan")
                r2_source = "provided"
                r2_n_used = int(y_vec.shape[0])
                r2_n_total = int(y_vec.shape[0])
                if resolved_bayes_r2 is None or (not np.isfinite(float(resolved_bayes_r2))):
                    r2_blup, r2_source, r2_n_used, r2_n_total = _estimate_bayes_auto_r2_from_blup(
                        y=y_vec,
                        Xtrain=packed_train,
                        packed_train_indices=train_abs,
                        cfg=bayes_auto_cfg,
                        seed_offset=int(train_abs.shape[0]),
                    )
                    r2_fit = float(r2_blup)
                else:
                    r2_fit = float(resolved_bayes_r2)
                r2_used = float(min(0.95, max(0.05, float(r2_fit))))

                packed_kwargs: dict[str, typing.Any] = {
                    "y": y_vec,
                    "packed": packed,
                    "n_samples": int(n_total_samples),
                    "row_flip": np.ascontiguousarray(np.asarray(row_flip, dtype=np.bool_).reshape(-1), dtype=np.bool_),
                    "row_maf": maf,
                    "row_mean": np.ascontiguousarray(np.asarray(row_mean, dtype=np.float32).reshape(-1), dtype=np.float32),
                    "row_inv_sd": np.ascontiguousarray(np.asarray(row_inv_sd, dtype=np.float32).reshape(-1), dtype=np.float32),
                    "sample_indices": train_abs,
                    "x": None,
                    "n_iter": 400,
                    "burnin": 200,
                    "thin": 1,
                    "r2": float(r2_used),
                    "threads": int(max(0, int(n_jobs))),
                    "seed": None,
                }
                if str(method) in {"BayesB", "BayesCpi"}:
                    packed_kwargs["prob_in"] = 0.5
                    packed_kwargs["counts"] = 5.0
                if str(method) == "BayesA":
                    packed_kwargs["min_abs_beta"] = 1e-9

                packed_fit_fn = getattr(_jxrs, str(packed_func_name))
                packed_threads_compat_fallback = False
                try:
                    packed_fit_ret = packed_fit_fn(**packed_kwargs)
                except TypeError as e:
                    msg = str(e)
                    if "unexpected keyword argument 'threads'" in msg:
                        packed_kwargs_retry = dict(packed_kwargs)
                        packed_kwargs_retry.pop("threads", None)
                        packed_fit_ret = packed_fit_fn(**packed_kwargs_retry)
                        packed_threads_compat_fallback = True
                    else:
                        raise
                if str(method) == "BayesCpi":
                    beta_raw, alpha_raw, _varb_mean, _vare, h2_mean, _var_h2, *diag_tail = packed_fit_ret
                else:
                    beta_raw, alpha_raw, _varb, _vare, h2_mean, _var_h2, *diag_tail = packed_fit_ret

                pve = float(h2_mean)
                bayes_beta = np.ascontiguousarray(np.asarray(beta_raw, dtype=np.float64).reshape(-1), dtype=np.float64)
                bayes_alpha = np.ascontiguousarray(np.asarray(alpha_raw, dtype=np.float64).reshape(-1), dtype=np.float64)
                bayes_alpha0 = float(bayes_alpha[0]) if int(bayes_alpha.shape[0]) > 0 else 0.0

                if bayes_runtime_state is not None:
                    bayes_runtime_state["r2_used"] = float(r2_used)
                    bayes_runtime_state["r2_blup"] = float(r2_blup)
                    bayes_runtime_state["r2_source"] = str(r2_source)
                    bayes_runtime_state["r2_n_used"] = int(r2_n_used)
                    bayes_runtime_state["r2_n_total"] = int(r2_n_total)
                    bayes_runtime_state["threads_requested"] = int(max(0, int(n_jobs)))
                    bayes_runtime_state["threads_compat_fallback"] = bool(
                        packed_threads_compat_fallback
                    )
                    if len(diag_tail) >= 2:
                        bayes_runtime_state["prob_in_mean"] = float(diag_tail[0])
                        bayes_runtime_state["n_active_mean"] = float(diag_tail[1])

                if need_train_pred:
                    if train_pred_idx is None:
                        train_pred_abs = train_abs
                    elif int(train_pred_idx.size) == 0:
                        train_pred_abs = np.zeros((0,), dtype=np.int64)
                    else:
                        train_pred_abs = np.ascontiguousarray(
                            train_abs[np.asarray(train_pred_idx, dtype=np.int64).reshape(-1)],
                            dtype=np.int64,
                        )
                    if int(train_pred_abs.size) > 0:
                        train_pred = _predict_bayes_packed_from_effects(
                            packed_ctx=packed_train,
                            sample_indices=train_pred_abs,
                            alpha0=bayes_alpha0,
                            beta=bayes_beta,
                            row_mean=row_mean,
                            row_inv_sd=row_inv_sd,
                            snp_block_size=bayes_snp_block,
                            sample_chunk_size=bayes_sample_chunk,
                        )
                    else:
                        train_pred = np.zeros((0, 1), dtype=float)
                else:
                    train_pred = np.zeros((0, 1), dtype=float)

                if int(test_abs.size) > 0:
                    test_pred = _predict_bayes_packed_from_effects(
                        packed_ctx=typing.cast(dict[str, typing.Any], Xtest),
                        sample_indices=test_abs,
                        alpha0=bayes_alpha0,
                        beta=bayes_beta,
                        row_mean=row_mean,
                        row_inv_sd=row_inv_sd,
                        snp_block_size=bayes_snp_block,
                        sample_chunk_size=bayes_sample_chunk,
                    )
                else:
                    test_pred = np.zeros((0, 1), dtype=float)
                if model_state is not None:
                    model_state["kind"] = "bayes_linear"
                    model_state["method"] = str(method)
                    model_state["alpha"] = float(bayes_alpha0)
                    model_state["beta"] = np.ascontiguousarray(
                        np.asarray(bayes_beta, dtype=np.float64).reshape(-1),
                        dtype=np.float64,
                    )
                    model_state["packed"] = True
                    model_state["standardized"] = True
                    model_state["row_mean"] = np.ascontiguousarray(
                        np.asarray(row_mean, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    model_state["row_inv_sd"] = np.ascontiguousarray(
                        np.asarray(row_inv_sd, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    model_state["snp_block_size"] = int(max(1, int(bayes_snp_block)))
                    model_state["sample_chunk_size"] = int(max(1, int(bayes_sample_chunk)))
                    model_state["pve"] = float(pve)
                return (
                    np.asarray(train_pred, dtype=float).reshape(-1, 1),
                    np.asarray(test_pred, dtype=float).reshape(-1, 1),
                    pve,
                )

            # Fallback for old extension builds without packed Bayes kernels.
            r2_blup_fb = float("nan")
            r2_source_fb = "provided"
            r2_n_used_fb = int(np.asarray(Y).reshape(-1).shape[0])
            r2_n_total_fb = int(np.asarray(Y).reshape(-1).shape[0])
            if resolved_bayes_r2 is None or (not np.isfinite(float(resolved_bayes_r2))):
                r2_blup_fb, r2_source_fb, r2_n_used_fb, r2_n_total_fb = _estimate_bayes_auto_r2_from_blup(
                    y=Y,
                    Xtrain=packed_train,
                    packed_train_indices=train_abs,
                    cfg=bayes_auto_cfg,
                    seed_offset=int(train_abs.shape[0]) + 17,
                )
                resolved_bayes_r2 = float(r2_blup_fb)
            dense_train = _decode_packed_subset_to_dense_standardized(
                packed_ctx=packed_train,
                sample_indices=train_abs,
                row_mean=row_mean,
                row_inv_sd=row_inv_sd,
                out_dtype=np.float64,
            )
            model = BAYES(Y.reshape(-1, 1), dense_train, method=method, r2=resolved_bayes_r2)
            pve = model.pve
            if bayes_runtime_state is not None:
                bayes_runtime_state["r2_used"] = float(getattr(model, "r2_used", np.nan))
                model_r2_blup = float(getattr(model, "r2_blup", np.nan))
                bayes_runtime_state["r2_blup"] = (
                    float(model_r2_blup)
                    if np.isfinite(model_r2_blup)
                    else float(r2_blup_fb)
                )
                model_r2_source = str(getattr(model, "r2_source", "")).strip()
                bayes_runtime_state["r2_source"] = (
                    model_r2_source
                    if model_r2_source != ""
                    else str(r2_source_fb)
                )
                bayes_runtime_state["r2_n_used"] = int(r2_n_used_fb)
                bayes_runtime_state["r2_n_total"] = int(r2_n_total_fb)
            bayes_beta = np.ascontiguousarray(
                np.asarray(getattr(model, "beta_hat", np.zeros((0, 1))), dtype=np.float64).reshape(-1),
                dtype=np.float64,
            )
            bayes_alpha = np.ascontiguousarray(
                np.asarray(getattr(model, "alpha_hat", np.asarray([[0.0]], dtype=np.float64)), dtype=np.float64).reshape(-1),
                dtype=np.float64,
            )
            bayes_alpha0 = float(bayes_alpha[0]) if int(bayes_alpha.shape[0]) > 0 else 0.0

            if need_train_pred:
                if train_pred_idx is None:
                    train_pred = model.predict(dense_train)
                elif int(train_pred_idx.size) == 0:
                    train_pred = np.zeros((0, 1), dtype=float)
                else:
                    train_pred = model.predict(np.asarray(dense_train)[:, train_pred_idx])
            else:
                train_pred = np.zeros((0, 1), dtype=float)

            del dense_train
            gc.collect()

            if int(test_abs.size) > 0:
                test_pred = _predict_bayes_packed_from_effects(
                    packed_ctx=typing.cast(dict[str, typing.Any], Xtest),
                    sample_indices=test_abs,
                    alpha0=bayes_alpha0,
                    beta=bayes_beta,
                    row_mean=row_mean,
                    row_inv_sd=row_inv_sd,
                    snp_block_size=bayes_snp_block,
                    sample_chunk_size=bayes_sample_chunk,
                )
            else:
                test_pred = np.zeros((0, 1), dtype=float)
            if model_state is not None:
                model_state["kind"] = "bayes_linear"
                model_state["method"] = str(method)
                model_state["alpha"] = float(bayes_alpha0)
                model_state["beta"] = np.ascontiguousarray(
                    np.asarray(bayes_beta, dtype=np.float64).reshape(-1),
                    dtype=np.float64,
                )
                model_state["packed"] = True
                model_state["standardized"] = True
                model_state["row_mean"] = np.ascontiguousarray(
                    np.asarray(row_mean, dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
                model_state["row_inv_sd"] = np.ascontiguousarray(
                    np.asarray(row_inv_sd, dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
                model_state["snp_block_size"] = int(max(1, int(bayes_snp_block)))
                model_state["sample_chunk_size"] = int(max(1, int(bayes_sample_chunk)))
                model_state["pve"] = float(pve)
            return (
                np.asarray(train_pred, dtype=float).reshape(-1, 1),
                np.asarray(test_pred, dtype=float).reshape(-1, 1),
                pve,
            )

        if resolved_bayes_r2 is None or (not np.isfinite(float(resolved_bayes_r2))):
            r2_blup, r2_source, r2_n_used, r2_n_total = _estimate_bayes_auto_r2_from_blup(
                y=Y,
                Xtrain=Xtrain,
                packed_train_indices=None,
                cfg=bayes_auto_cfg,
                seed_offset=int(np.asarray(Y).reshape(-1).shape[0]),
            )
            resolved_bayes_r2 = float(r2_blup)
        else:
            r2_blup = float("nan")
            r2_source = "provided"
            r2_n_used = int(np.asarray(Y).reshape(-1).shape[0])
            r2_n_total = int(np.asarray(Y).reshape(-1).shape[0])
        model = BAYES(Y.reshape(-1, 1), Xtrain, method=method, r2=resolved_bayes_r2)
        pve = model.pve
        if bayes_runtime_state is not None:
            bayes_runtime_state["r2_used"] = float(getattr(model, "r2_used", np.nan))
            model_r2_blup = float(getattr(model, "r2_blup", np.nan))
            if np.isfinite(model_r2_blup):
                bayes_runtime_state["r2_blup"] = float(model_r2_blup)
            else:
                bayes_runtime_state["r2_blup"] = float(r2_blup)
            model_r2_source = str(getattr(model, "r2_source", "")).strip()
            bayes_runtime_state["r2_source"] = (
                model_r2_source
                if model_r2_source != ""
                else str(r2_source)
            )
            bayes_runtime_state["r2_n_used"] = int(r2_n_used)
            bayes_runtime_state["r2_n_total"] = int(r2_n_total)
        if need_train_pred:
            if train_pred_idx is None:
                train_pred = model.predict(Xtrain)
            elif int(train_pred_idx.size) == 0:
                train_pred = np.zeros((0, 1), dtype=float)
            else:
                train_pred = model.predict(np.asarray(Xtrain)[:, train_pred_idx])
        else:
            train_pred = np.zeros((0, 1), dtype=float)
        if model_state is not None:
            model_state["kind"] = "bayes_linear"
            model_state["method"] = str(method)
            model_state["alpha"] = float(
                np.asarray(getattr(model, "alpha_hat", np.asarray([[0.0]], dtype=np.float64)), dtype=np.float64).reshape(-1)[0]
                if int(np.asarray(getattr(model, "alpha_hat", np.asarray([[0.0]], dtype=np.float64))).size) > 0
                else 0.0
            )
            model_state["beta"] = np.ascontiguousarray(
                np.asarray(getattr(model, "beta_hat", np.zeros((0, 1))), dtype=np.float64).reshape(-1),
                dtype=np.float64,
            )
            model_state["packed"] = False
            model_state["standardized"] = True
            model_state["snp_block_size"] = 2048
            model_state["sample_chunk_size"] = 4096
            model_state["pve"] = float(pve)
        return (
            np.asarray(train_pred, dtype=float).reshape(-1, 1),
            np.asarray(model.predict(Xtest), dtype=float).reshape(-1, 1),
            pve,
        )

    if method in _ML_METHOD_MAP:
        Xtrain_ml = np.ascontiguousarray(np.asarray(Xtrain, dtype=np.float32), dtype=np.float32)
        Xtest_ml = np.ascontiguousarray(np.asarray(Xtest, dtype=np.float32), dtype=np.float32)
        model = MLGS(
            y=Y.reshape(-1),
            M=Xtrain_ml,
            method=typing.cast(typing.Any, _ML_METHOD_MAP[method]),
            seed=42,
            cv=_mlgs_inner_cv(int(Y.shape[0])),
            n_jobs=max(1, int(n_jobs)),
            fit_on_init=False,
            verbose=False,
        )
        if ml_fixed_params is not None:
            model.fit_with_params(ml_fixed_params)
        else:
            model.fit()
        if need_train_pred:
            if train_pred_idx is None:
                yhat_train = model.predict(Xtrain_ml)
            elif int(train_pred_idx.size) == 0:
                yhat_train = np.zeros((0, 1), dtype=float)
            else:
                yhat_train = model.predict(np.asarray(Xtrain_ml)[:, train_pred_idx])
        else:
            yhat_train = np.zeros((0, 1), dtype=float)
        if int(Xtest_ml.shape[1]) > 0:
            yhat_test = model.predict(Xtest_ml)
        else:
            yhat_test = np.zeros((0, 1), dtype=float)
        pve = float(
            (model.best_metrics_ or {}).get("r2", np.nan)
        )
        if model_state is not None:
            model_state["kind"] = "mlsk_model"
            model_state["method"] = str(method)
            model_state["alpha"] = float(0.0)
            model_state["beta"] = None
            model_state["packed"] = bool(is_packed_input)
            model_state["standardized"] = False
            model_state["estimator"] = model.model
            model_state["feature_axis"] = str(getattr(model, "feature_axis_", "auto"))
            model_state["marker_means"] = np.ascontiguousarray(
                np.asarray(getattr(model, "marker_means_", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )
            model_state["cov_means"] = (
                None
                if getattr(model, "cov_means_", None) is None
                else np.ascontiguousarray(
                    np.asarray(getattr(model, "cov_means_", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
                    dtype=np.float32,
                )
            )
            model_state["cov_count"] = int(getattr(model, "cov_count_", 0))
            model_state["snp_block_size"] = int(getattr(model, "best_n_estimators_", 0) or 0)
            model_state["sample_chunk_size"] = int(getattr(model, "best_boost_rounds_", 0) or 0)
            model_state["pve"] = float(pve)
        return (
            np.asarray(yhat_train, dtype=float).reshape(-1, 1),
            np.asarray(yhat_test, dtype=float).reshape(-1, 1),
            pve,
        )

    raise ValueError(f"Unsupported GS method: {method}")


def _run_method_task(
    method: str,
    train_pheno: np.ndarray,
    train_snp: np.ndarray | None,
    test_snp: np.ndarray | None,
    train_snp_add: np.ndarray | None,
    test_snp_add: np.ndarray | None,
    train_snp_ml: np.ndarray | None,
    test_snp_ml: np.ndarray | None,
    pca_dec: bool,
    cv_splits: typing.Optional[list[tuple[np.ndarray, np.ndarray]]],
    n_jobs: int,
    strict_cv: bool,
    force_fast: bool,
    packed_ctx: dict[str, typing.Any] | None = None,
    train_sample_indices: np.ndarray | None = None,
    test_sample_indices: np.ndarray | None = None,
    train_sample_ids: np.ndarray | None = None,
    progress_queue: typing.Any = None,
    progress_hook: typing.Any = None,
    search_progress_hook: typing.Any = None,
    rrblup_progress_hook: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
    limit_predtrain: int | None = None,
    rrblup_solver: str = "pcg",
    rrblup_adamw_cfg: dict[str, typing.Any] | None = None,
    bayes_auto_r2_cache: dict[str, float] | None = None,
    bayes_auto_r2_cfg: dict[str, typing.Any] | None = None,
    export_model: bool = True,
    stage_hook: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
) -> dict[str, typing.Any]:
    _t_method_begin = time.time()
    def _emit_stage(event: str, **payload: typing.Any) -> None:
        if stage_hook is None:
            return
        try:
            stage_hook(str(event), dict(payload))
        except Exception:
            return
    def _stage_elapsed_metrics(elapsed_total: float) -> tuple[float, float, float]:
        cv_elapsed = float(
            np.nansum(
                np.asarray(
                    [float(r[6]) for r in fold_rows],
                    dtype=np.float64,
                )
            )
        ) if len(fold_rows) > 0 else 0.0
        cv_elapsed = float(max(0.0, cv_elapsed))
        fit_elapsed = float(max(0.0, float(elapsed_total) - cv_elapsed))
        # Predict is included in final fit/inference stage for current kernels.
        predict_elapsed = 0.0
        return cv_elapsed, fit_elapsed, predict_elapsed

    fold_rows: list[tuple[str, int, float, float, float, float, float]] = []
    oof_pred: np.ndarray | None = None
    oof_truth: np.ndarray | None = None
    rrblup_pve_rows: list[dict[str, typing.Any]] = []
    rrblup_pcg_trace_rows: list[dict[str, typing.Any]] = []
    rrblup_pve_final: dict[str, typing.Any] | None = None
    rrblup_final_state: dict[str, typing.Any] | None = None
    rrblup_final_cfg: dict[str, typing.Any] | None = None
    gblup_final_state: dict[str, typing.Any] | None = None
    bayes_r2_rows: list[float] = []
    bayes_r2_final = float("nan")
    bayes_r2_source_final = ""
    bayes_r2_n_used_final = 0
    bayes_r2_n_total_final = 0
    model_state_final: dict[str, typing.Any] | None = None
    best_test = None
    best_train = None
    best_r2 = -np.inf
    ml_tuning_cache: dict[str, typing.Any] | None = None
    rrblup_cfg_base = dict(rrblup_adamw_cfg or {})
    if method == "rrBLUP":
        rrblup_cfg_base["auto_pcg_ref_n"] = int(np.asarray(train_pheno).reshape(-1).shape[0])
    packed_lmm_methods = {
        _GBLUP_METHOD_ADD,
        _GBLUP_METHOD_DOM,
        _GBLUP_METHOD_AD,
        "rrBLUP",
        "BayesA",
        "BayesB",
        "BayesCpi",
    }
    bayes_methods = {"BayesA", "BayesB", "BayesCpi"}
    bayes_cfg_base = dict(bayes_auto_r2_cfg or {})
    bayes_cv_reuse_enabled = bool(
        method in bayes_methods
        and cv_splits is not None
        and _cfg_truthy(bayes_cfg_base.get("cv_reuse", "on"), default=True)
    )
    bayes_cv_shared_key = f"{method}:cv_shared"
    bayes_cv_shared_r2 = float("nan")
    bayes_cv_shared_source = ""
    bayes_cv_shared_n_used = 0
    bayes_cv_shared_n_total = 0
    if bayes_cv_reuse_enabled:
        if bayes_auto_r2_cache is not None:
            cached_shared = float(bayes_auto_r2_cache.get(bayes_cv_shared_key, np.nan))
            if np.isfinite(cached_shared):
                bayes_cv_shared_r2 = float(cached_shared)
                bayes_cv_shared_source = "cache"
        if not np.isfinite(bayes_cv_shared_r2):
            try:
                if (packed_ctx is not None) and _looks_like_packed_payload(packed_ctx):
                    if train_sample_indices is None:
                        raise ValueError("Packed Bayes CV-r2 reuse requires train_sample_indices.")
                    bayes_cv_shared_r2, _src, _n_used, _n_total = _estimate_bayes_auto_r2_from_blup(
                        y=train_pheno,
                        Xtrain=typing.cast(dict[str, typing.Any], packed_ctx),
                        packed_train_indices=np.ascontiguousarray(
                            np.asarray(train_sample_indices, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        ),
                        cfg=bayes_cfg_base,
                        seed_offset=int(np.asarray(train_pheno).reshape(-1).shape[0]) + 101,
                    )
                else:
                    if train_snp is None:
                        raise ValueError("Bayes CV-r2 reuse requires train_snp when packed context is absent.")
                    bayes_cv_shared_r2, _src, _n_used, _n_total = _estimate_bayes_auto_r2_from_blup(
                        y=train_pheno,
                        Xtrain=train_snp,
                        packed_train_indices=None,
                        cfg=bayes_cfg_base,
                        seed_offset=int(np.asarray(train_pheno).reshape(-1).shape[0]) + 101,
                    )
                bayes_cv_shared_source = str(_src)
                bayes_cv_shared_n_used = int(_n_used)
                bayes_cv_shared_n_total = int(_n_total)
                if np.isfinite(bayes_cv_shared_r2) and bayes_auto_r2_cache is not None:
                    bayes_auto_r2_cache[bayes_cv_shared_key] = float(bayes_cv_shared_r2)
            except Exception:
                # Fail-open: if shared pre-estimation fails, fallback to fold-level auto-r2.
                bayes_cv_shared_r2 = float("nan")

    rrblup_cv_reuse_enabled = bool(
        method == "rrBLUP"
        and cv_splits is not None
        and _cfg_truthy(rrblup_cfg_base.get("grid_reuse_cv", "on"), default=True)
    )
    rrblup_cv_selected: dict[str, float] | None = None
    packed_payload = (
        typing.cast(dict[str, typing.Any], packed_ctx)
        if _looks_like_packed_payload(packed_ctx)
        else None
    )
    precomputed_train_grm = None
    precomputed_test_train_grm = None
    precomputed_hash_ztz_test_pred = None
    precomputed_hash_ztz_pve = None
    precomputed_hash_cv_compact = False
    hash_cv_dim: int | None = None
    hash_cv_seed: int | None = None
    hash_cv_standardize = True
    if isinstance(packed_ctx, dict):
        precomputed_train_grm = packed_ctx.get("__gblup_train_grm__", None)
        precomputed_test_train_grm = packed_ctx.get("__gblup_test_train_grm__", None)
        precomputed_hash_ztz_test_pred = packed_ctx.get("__gblup_hash_ztz_test_pred__", None)
        precomputed_hash_ztz_pve = packed_ctx.get("__gblup_hash_ztz_pve__", None)
        precomputed_hash_cv_compact = bool(
            packed_ctx.get("__gblup_hash_cv_compact__", False)
        )
        hash_dim_raw = packed_ctx.get("__gblup_hash_dim__", None)
        hash_seed_raw = packed_ctx.get("__gblup_hash_seed__", None)
        hash_std_raw = packed_ctx.get("__gblup_hash_standardize__", None)
        if hash_dim_raw is not None:
            hash_cv_dim = int(hash_dim_raw)
        if hash_seed_raw is not None:
            hash_cv_seed = int(hash_seed_raw)
        if hash_std_raw is not None:
            hash_cv_standardize = bool(hash_std_raw)

    def _float_or_nan(value: typing.Any) -> float:
        try:
            out = float(value)
        except Exception:
            return float("nan")
        return float(out) if np.isfinite(out) else float("nan")
    use_hash_cv_compact = bool(
        method == "GBLUP"
        and cv_splits is not None
        and packed_payload is not None
        and precomputed_hash_cv_compact
        and hash_cv_dim is not None
        and hash_cv_seed is not None
    )
    gblup_cv_grm_full: np.ndarray | None = None
    gblup_final_test_train: np.ndarray | None = None
    gblup_force_kinship = False
    prefer_rust_packed_gblup = bool(
        method == "GBLUP"
        and packed_payload is not None
        and (_jxrs is not None)
        and hasattr(_jxrs, "gblup_reml_packed_bed")
    )

    if (
        method == "GBLUP"
        and cv_splits is None
        and precomputed_hash_ztz_test_pred is not None
    ):
        _elapsed_total = float(max(0.0, time.time() - _t_method_begin))
        _cv_elapsed, _fit_elapsed, _predict_elapsed = _stage_elapsed_metrics(_elapsed_total)
        pred = np.ascontiguousarray(
            np.asarray(precomputed_hash_ztz_test_pred, dtype=np.float64).reshape(-1, 1),
            dtype=np.float64,
        )
        pve_final = (
            float(precomputed_hash_ztz_pve)
            if precomputed_hash_ztz_pve is not None
            else float("nan")
        )
        _emit_stage("fit_start", method=str(method))
        _emit_stage("fit_end", method=str(method), elapsed=float(_fit_elapsed))
        _emit_stage("predict_start", method=str(method))
        _emit_stage("predict_end", method=str(method), elapsed=float(_predict_elapsed))
        return {
            "method": method,
            "method_display": _method_display_name(method),
            "fold_rows": [],
            "best_test": None,
            "best_train": None,
            "oof_pred": None,
            "oof_truth": None,
            "oof_sample_ids": None,
            "test_pred": pred,
            "pve_final": float(pve_final),
            "ml_tuning": None,
            "rrblup_final_state": None,
            "rrblup_final_cfg": None,
            "rrblup_cv_selected": None,
            "model_state": None,
            "cv_mode": "train_cv",
            "elapsed_total_sec": _elapsed_total,
            "elapsed_cv_sec": _cv_elapsed,
            "elapsed_fit_sec": _fit_elapsed,
            "elapsed_predict_sec": _predict_elapsed,
        }

    if method in _ML_METHOD_MAP and (not strict_cv):
        if train_snp_ml is None:
            raise ValueError(f"{method} requires raw genotype matrix.")
        ml_tuning_cache = _tune_ml_method_once(
            method=method,
            Y=train_pheno,
            Xtrain=train_snp_ml,
            PCAdec=pca_dec,
            n_jobs=n_jobs,
            progress_hook=search_progress_hook,
        )

    if method == "GBLUP":
        if prefer_rust_packed_gblup:
            gblup_force_kinship = False
            if _GS_DEBUG_STAGE:
                print(
                    "[GS-DEBUG] GBLUP packed policy: route via GSapi->rust_gblup_reml_packed_bed "
                    "(skip CV GRM prebuild).",
                    flush=True,
                )
        elif precomputed_train_grm is not None:
            gblup_cv_grm_full = np.asarray(precomputed_train_grm, dtype=np.float64)
            if precomputed_test_train_grm is not None:
                gblup_final_test_train = np.asarray(
                    precomputed_test_train_grm,
                    dtype=np.float64,
                )
            gblup_force_kinship = True
        elif packed_payload is not None:
            # Packed backend policy aligns with dense behavior:
            # - n > m: prefer marker-space/FaST route (avoid explicit n x n GRM prebuild)
            # - m >= n: explicit kinship prebuild can be reused across CV folds
            n_train = int(train_pheno.shape[0])
            m_snp = int(np.asarray(packed_payload["packed"]).shape[0])
            gblup_force_kinship = bool(m_snp >= n_train)
        elif packed_payload is None and train_snp is not None:
            # GBLUP backend policy (dense markers only):
            # - n > m: use marker-space ZTZ/FaST route (no explicit kinship build)
            # - m >= n: build kinship once and reuse by slicing
            n_train = int(train_pheno.shape[0])
            m_snp = int(train_snp.shape[0])
            gblup_force_kinship = bool(m_snp >= n_train)
            if gblup_force_kinship:
                try:
                    gblup_cv_grm_full = _build_gblup_cv_grm_once(
                        train_snp=train_snp,
                        packed_ctx=None,
                        train_sample_indices=None,
                        n_jobs=max(1, int(n_jobs)),
                    )
                    if test_snp is not None and int(test_snp.shape[1]) > 0:
                        gblup_final_test_train = _build_gblup_test_train_cross_from_markers(
                            test_snp=test_snp,
                            train_snp=train_snp,
                        )
                    if _GS_DEBUG_STAGE and gblup_cv_grm_full is not None:
                        print(
                            f"[GS-DEBUG] GBLUP kinship backend prebuilt GRM shape={gblup_cv_grm_full.shape}",
                            flush=True,
                        )
                except Exception as ex:
                    # Fallback to legacy GSapi route on unexpected numeric issues.
                    gblup_cv_grm_full = None
                    gblup_final_test_train = None
                    gblup_force_kinship = False
                    if _GS_DEBUG_STAGE:
                        print(
                            f"[GS-DEBUG] GBLUP kinship backend failed -> fallback GSapi: {ex}",
                            flush=True,
                        )

    need_cv_grm_prebuild = bool(
        (method == "GBLUP")
        and (
            gblup_force_kinship
            or (precomputed_train_grm is not None)
        )
    )
    if use_hash_cv_compact:
        # Compact hash+CV path builds fold-level ZTZ stats directly and does
        # not need a full n_train x n_train GRM prebuild.
        need_cv_grm_prebuild = False
    if (
        (method == "GBLUP")
        and (cv_splits is not None)
        and (gblup_cv_grm_full is None)
        and need_cv_grm_prebuild
    ):
        try:
            gblup_cv_grm_full = _build_gblup_cv_grm_once(
                train_snp=train_snp,
                packed_ctx=packed_payload,
                train_sample_indices=train_sample_indices,
                n_jobs=max(1, int(n_jobs)),
            )
            if _GS_DEBUG_STAGE and gblup_cv_grm_full is not None:
                print(
                    f"[GS-DEBUG] GBLUP CV prebuilt GRM shape={gblup_cv_grm_full.shape}",
                    flush=True,
                )
        except Exception as ex:
            # Keep backward-compatible fallback path if one-shot GRM cannot be built.
            gblup_cv_grm_full = None
            if _GS_DEBUG_STAGE:
                print(
                    f"[GS-DEBUG] GBLUP CV prebuilt GRM failed -> fallback per-fold fit: {ex}",
                    flush=True,
                )

    if cv_splits is not None:
        cv_total = int(max(1, len(cv_splits)))
        oof_pred = np.full((int(train_pheno.shape[0]),), np.nan, dtype=np.float64)
        oof_truth = np.asarray(train_pheno, dtype=np.float64).reshape(-1)
        _emit_stage("cv_start", method=str(method), total=int(cv_total))
        for fold_id, (test_idx, train_idx) in enumerate(cv_splits, start=1):
            t_fold = time.time()
            fold_train_idx_arg = None
            fold_test_idx_arg = None
            fold_train_pred_local_idx: np.ndarray | None = _resolve_train_pred_local_indices(
                n_train_fold=int(np.asarray(train_idx).shape[0]),
                limit_predtrain=limit_predtrain,
                fold_id=int(fold_id),
            )
            need_fold_train_pred = bool(
                (fold_train_pred_local_idx is None) or (int(fold_train_pred_local_idx.size) > 0)
            )
            fold_train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
            fold_test_idx = np.asarray(test_idx, dtype=np.int64).reshape(-1)
            if (method == "GBLUP") and use_hash_cv_compact:
                if train_sample_indices is None:
                    raise ValueError(
                        "Compact hash+CV mode requires train_sample_indices for packed genotype."
                    )
                train_abs_all = np.ascontiguousarray(
                    np.asarray(train_sample_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                )
                fold_train_abs = np.ascontiguousarray(
                    train_abs_all[fold_train_idx],
                    dtype=np.int64,
                )
                fold_test_abs = np.ascontiguousarray(
                    train_abs_all[fold_test_idx],
                    dtype=np.int64,
                )
                y_fold = np.asarray(train_pheno[fold_train_idx], dtype=np.float64).reshape(-1)
                gram, row_sum, row_sq_sum, zy_vec, _hash_scale, _hash_kept = (
                    _hash_packed_ztz_stats_for_gblup(
                        packed_ctx=typing.cast(dict[str, typing.Any], packed_payload),
                        hash_dim=int(hash_cv_dim),
                        hash_seed=int(hash_cv_seed),
                        standardize=bool(hash_cv_standardize),
                        n_jobs=max(1, int(n_jobs)),
                        train_sample_indices=fold_train_abs,
                        y_train=y_fold,
                    )
                )
                fit_fast = _fit_gblup_reml_from_hash_ztz_stats(
                    gram=np.asarray(gram, dtype=np.float64),
                    row_sum=np.asarray(row_sum, dtype=np.float64),
                    row_sq_sum=np.asarray(row_sq_sum, dtype=np.float64),
                    zy=np.asarray(zy_vec, dtype=np.float64),
                    n_train=int(fold_train_idx.shape[0]),
                )
                yhat_test = _predict_hashed_gblup_test_from_compact(
                    packed_ctx=typing.cast(dict[str, typing.Any], packed_payload),
                    hash_dim=int(hash_cv_dim),
                    hash_seed=int(hash_cv_seed),
                    standardize=bool(hash_cv_standardize),
                    n_jobs=max(1, int(n_jobs)),
                    test_sample_indices=fold_test_abs,
                    beta=float(fit_fast["beta"]),
                    m_mean=np.asarray(fit_fast["m_mean"], dtype=np.float64),
                    m_var_sum=float(fit_fast["m_var_sum"]),
                    m_alpha=np.asarray(fit_fast["m_alpha"], dtype=np.float64),
                    alpha_sum=float(fit_fast["alpha_sum"]),
                    mean_sq=float(fit_fast["mean_sq"]),
                    mean_malpha=float(fit_fast["mean_malpha"]),
                )
                if need_fold_train_pred:
                    if fold_train_pred_local_idx is None:
                        pred_train_abs = fold_train_abs
                    elif int(fold_train_pred_local_idx.size) == 0:
                        pred_train_abs = np.zeros((0,), dtype=np.int64)
                    else:
                        local_idx = np.asarray(
                            fold_train_pred_local_idx, dtype=np.int64
                        ).reshape(-1)
                        pred_train_abs = np.ascontiguousarray(
                            fold_train_abs[local_idx],
                            dtype=np.int64,
                        )
                    if int(pred_train_abs.size) == 0:
                        yhat_train = np.zeros((0, 1), dtype=float)
                    else:
                        yhat_train = _predict_hashed_gblup_test_from_compact(
                            packed_ctx=typing.cast(dict[str, typing.Any], packed_payload),
                            hash_dim=int(hash_cv_dim),
                            hash_seed=int(hash_cv_seed),
                            standardize=bool(hash_cv_standardize),
                            n_jobs=max(1, int(n_jobs)),
                            test_sample_indices=pred_train_abs,
                            beta=float(fit_fast["beta"]),
                            m_mean=np.asarray(fit_fast["m_mean"], dtype=np.float64),
                            m_var_sum=float(fit_fast["m_var_sum"]),
                            m_alpha=np.asarray(fit_fast["m_alpha"], dtype=np.float64),
                            alpha_sum=float(fit_fast["alpha_sum"]),
                            mean_sq=float(fit_fast["mean_sq"]),
                            mean_malpha=float(fit_fast["mean_malpha"]),
                        )
                else:
                    yhat_train = np.zeros((0, 1), dtype=float)
                pve = float(fit_fast["pve"])
            elif (method == "GBLUP") and (gblup_cv_grm_full is not None):
                y_fold = np.asarray(train_pheno[fold_train_idx], dtype=np.float64).reshape(-1)
                grm_train = np.asarray(
                    gblup_cv_grm_full[np.ix_(fold_train_idx, fold_train_idx)],
                    dtype=np.float64,
                )
                fit = _fit_gblup_reml_from_grm(y_fold, grm_train)
                alpha = np.asarray(fit["alpha"], dtype=np.float64).reshape(-1, 1)
                beta = float(fit["beta"])
                pve = float(fit["pve"])

                grm_test_train = np.asarray(
                    gblup_cv_grm_full[np.ix_(fold_test_idx, fold_train_idx)],
                    dtype=np.float64,
                )
                yhat_test = _predict_gblup_from_cross(
                    cross_kernel=grm_test_train,
                    alpha=alpha,
                    beta=beta,
                )
                if need_fold_train_pred:
                    if fold_train_pred_local_idx is None:
                        yhat_train = _predict_gblup_from_cross(
                            cross_kernel=grm_train,
                            alpha=alpha,
                            beta=beta,
                        )
                    elif int(fold_train_pred_local_idx.size) == 0:
                        yhat_train = np.zeros((0, 1), dtype=float)
                    else:
                        local_idx = np.asarray(fold_train_pred_local_idx, dtype=np.int64).reshape(-1)
                        grm_pick = np.asarray(grm_train[local_idx, :], dtype=np.float64)
                        yhat_train = _predict_gblup_from_cross(
                            cross_kernel=grm_pick,
                            alpha=alpha,
                            beta=beta,
                        )
                else:
                    yhat_train = np.zeros((0, 1), dtype=float)
            else:
                if (packed_payload is not None) and (method in packed_lmm_methods):
                    if train_sample_indices is None:
                        raise ValueError("Packed LMM requires train sample indices.")
                    fold_train = packed_payload
                    fold_test = packed_payload
                    fold_train_idx_arg = np.ascontiguousarray(train_sample_indices[fold_train_idx], dtype=np.int64)
                    fold_test_idx_arg = np.ascontiguousarray(train_sample_indices[fold_test_idx], dtype=np.int64)
                    fold_pca = False
                elif _is_gblup_method(str(method)) and (_gblup_method_kernel_mode(str(method)) in {"d", "ad"}):
                    if train_snp_add is None:
                        raise ValueError(f"{method} requires additive raw genotype matrix.")
                    fold_train = train_snp_add[:, fold_train_idx]
                    fold_test = train_snp_add[:, fold_test_idx]
                    fold_pca = False
                elif method in _ML_METHOD_MAP:
                    if train_snp_ml is None:
                        raise ValueError(f"{method} requires raw genotype matrix.")
                    fold_train = train_snp_ml[:, fold_train_idx]
                    fold_test = train_snp_ml[:, fold_test_idx]
                    fold_pca = pca_dec
                else:
                    fold_train = train_snp[:, fold_train_idx]
                    fold_test = train_snp[:, fold_test_idx]
                    fold_pca = pca_dec
                rr_cfg_call: dict[str, typing.Any] | None = rrblup_adamw_cfg
                rr_state_call: dict[str, typing.Any] | None = None
                if method == "rrBLUP":
                    rr_cfg_call = dict(rrblup_cfg_base)
                    rr_lambda_auto_strategy = str(
                        rr_cfg_call.get("lambda_auto_strategy", "he_first")
                    ).strip().lower()
                    if rr_lambda_auto_strategy in {"he_first", "he_only"}:
                        rr_cfg_call["he_enable"] = "on"
                    else:
                        # REML-first mode can keep fold-level HE disabled.
                        rr_cfg_call["he_enable"] = "off"
                    if (
                        rrblup_cv_reuse_enabled
                        and rrblup_cv_selected is not None
                        and ("lr" in rrblup_cv_selected)
                    ):
                        rr_cfg_call["lambda_value"] = float(rrblup_cv_selected["lambda_value"])
                        rr_sel_lr = float(rrblup_cv_selected.get("lr", np.nan))
                        if np.isfinite(rr_sel_lr) and rr_sel_lr > 0.0:
                            rr_cfg_call["lr"] = float(rr_sel_lr)
                        rr_cfg_call["lambda_auto"] = "off"
                        rr_cfg_call["auto_grid"] = "off"
                    rr_state_call = {}
                rr_progress_call = None
                gblup_state_call: dict[str, typing.Any] | None = None
                bayes_r2_call: float | None = None
                bayes_state_call: dict[str, typing.Any] | None = None
                bayes_r2_cache_key: str | None = None
                if _is_gblup_method(str(method)):
                    gblup_state_call = {}
                if method == "rrBLUP" and rrblup_progress_hook is not None:
                    def _fold_rr_progress(event: str, payload: dict[str, typing.Any], *, _fold_id: int = int(fold_id), _cv_total: int = int(cv_total)) -> None:
                        pp = dict(payload)
                        pp.setdefault("phase_label", f"fold {_fold_id}/{_cv_total}")
                        rrblup_progress_hook(str(event), pp)
                    rr_progress_call = _fold_rr_progress
                if method in {"BayesA", "BayesB", "BayesCpi"}:
                    if bayes_cv_reuse_enabled and np.isfinite(bayes_cv_shared_r2):
                        bayes_r2_call = float(bayes_cv_shared_r2)
                        bayes_r2_cache_key = bayes_cv_shared_key
                    else:
                        bayes_r2_cache_key = f"cv:{int(fold_id)}"
                        if bayes_auto_r2_cache is not None:
                            cached_r2 = float(bayes_auto_r2_cache.get(bayes_r2_cache_key, np.nan))
                            if np.isfinite(cached_r2):
                                bayes_r2_call = float(cached_r2)
                    bayes_state_call = {}
                yhat_train, yhat_test, pve = GSapi(
                    train_pheno[fold_train_idx],
                    fold_train,
                    fold_test,
                    method=method,
                    PCAdec=fold_pca,
                    n_jobs=n_jobs,
                    force_fast=force_fast,
                    ml_fixed_params=(None if ml_tuning_cache is None else ml_tuning_cache.get("params")),
                    need_train_pred=need_fold_train_pred,
                    packed_train_indices=fold_train_idx_arg,
                    packed_test_indices=fold_test_idx_arg,
                    train_pred_indices=fold_train_pred_local_idx,
                    rrblup_solver=typing.cast(typing.Any, rrblup_solver),
                    rrblup_adamw_cfg=rr_cfg_call,
                    rrblup_runtime_state=rr_state_call,
                    rrblup_progress_hook=rr_progress_call,
                    gblup_runtime_state=gblup_state_call,
                    bayes_auto_r2=bayes_r2_call,
                    bayes_runtime_state=bayes_state_call,
                    bayes_auto_cfg=bayes_auto_r2_cfg,
                )
                if method in {"BayesA", "BayesB", "BayesCpi"} and bayes_state_call is not None:
                    r2_used = float(bayes_state_call.get("r2_used", np.nan))
                    if np.isfinite(r2_used):
                        bayes_r2_rows.append(float(r2_used))
                        if (
                            (bayes_auto_r2_cache is not None)
                            and (bayes_r2_cache_key is not None)
                            and (not (bayes_cv_reuse_enabled and np.isfinite(bayes_cv_shared_r2)))
                        ):
                            bayes_auto_r2_cache[bayes_r2_cache_key] = float(r2_used)
                if (
                    method == "rrBLUP"
                    and rrblup_cv_reuse_enabled
                    and rrblup_cv_selected is None
                    and rr_state_call is not None
                ):
                    sel_lam = float(rr_state_call.get("selected_lambda", np.nan))
                    sel_lr = float(rr_state_call.get("selected_lr", np.nan))
                    solver_eff_call = str(
                        rr_state_call.get(
                            "solver_effective",
                            rr_state_call.get("solver", ""),
                        )
                    ).strip().lower()
                    if (
                        np.isfinite(sel_lam)
                        and sel_lam >= 0.0
                        and solver_eff_call == "adamw"
                        and np.isfinite(sel_lr)
                        and sel_lr > 0.0
                    ):
                        rrblup_cv_selected = {"lambda_value": float(sel_lam)}
                        rrblup_cv_selected["lr"] = float(sel_lr)
                        if _GS_DEBUG_STAGE:
                            print(
                                "[GS-DEBUG] rrBLUP CV reuse armed "
                                f"lambda={sel_lam:.6g} lr={float(sel_lr):.6g}",
                                flush=True,
                            )
                if method == "rrBLUP" and rr_state_call is not None:
                    rr_cfg_mode = str(
                        rr_state_call.get(
                            "pve_mode_used",
                            (rr_cfg_call or {}).get("pve_mode", "lambda"),
                        )
                    ).strip().lower()
                    if rr_cfg_mode not in {"lambda", "trainvar"}:
                        rr_cfg_mode = "lambda"
                    pve_used_f = _float_or_nan(pve)
                    pve_trainvar_f = _float_or_nan(rr_state_call.get("pve_trainvar", np.nan))
                    pve_lambda_f = _float_or_nan(rr_state_call.get("pve_lambda", np.nan))
                    if (not np.isfinite(pve_trainvar_f)) and rr_cfg_mode == "trainvar":
                        pve_trainvar_f = pve_used_f
                    if (not np.isfinite(pve_lambda_f)) and rr_cfg_mode == "lambda":
                        pve_lambda_f = pve_used_f
                    iter_like = _float_or_nan(rr_state_call.get("pcg_iters", np.nan))
                    if not np.isfinite(iter_like):
                        iter_like = _float_or_nan(
                            rr_state_call.get(
                                "epochs_ran",
                                rr_state_call.get("best_epoch", np.nan),
                            )
                        )
                    solver_req_fold = str(
                        rr_state_call.get(
                            "solver_requested",
                            str(rrblup_solver).strip().lower(),
                        )
                    ).strip().lower()
                    solver_eff_fold = str(
                        rr_state_call.get(
                            "solver_effective",
                            rr_state_call.get("solver", solver_req_fold),
                        )
                    ).strip().lower()
                    solver_reason_fold = str(
                        rr_state_call.get("solver_fallback_reason", "")
                    ).strip()
                    rrblup_pve_rows.append(
                        {
                            "fold": int(fold_id),
                            "solver": str(solver_eff_fold),
                            "solver_requested": str(solver_req_fold),
                            "solver_effective": str(solver_eff_fold),
                            "solver_fallback_reason": str(solver_reason_fold),
                            "pve_mode": str(rr_cfg_mode),
                            "pve_used": pve_used_f,
                            "pve_trainvar": pve_trainvar_f,
                            "pve_lambda": pve_lambda_f,
                            "pve_exact": _float_or_nan(rr_state_call.get("pve_exact", np.nan)),
                            "iter_like": iter_like,
                        }
                    )
                    trace_items_raw = list(
                        typing.cast(
                            list[dict[str, typing.Any]] | None,
                            rr_state_call.get("pcg_rel_res_trace"),
                        )
                        or []
                    )
                    if len(trace_items_raw) > 0:
                        rrblup_pcg_trace_rows.append(
                            {
                                "phase": "cv",
                                "phase_label": f"fold {int(fold_id)}/{int(cv_total)}",
                                "fold": int(fold_id),
                                "solver": str(solver_eff_fold),
                                "converged": bool(rr_state_call.get("pcg_converged", False)),
                                "iters": int(max(0, int(rr_state_call.get("pcg_iters", 0) or 0))),
                                "max_iter": int(
                                    max(
                                        1,
                                        int(
                                            (rr_cfg_call or {}).get(
                                                "pcg_max_iter",
                                                rrblup_cfg_base.get("pcg_max_iter", 100),
                                            )
                                        ),
                                    )
                                ),
                                "trace": [
                                    {
                                        "iter": int(max(0, int(rec.get("iter", 0) or 0))),
                                        "total": int(max(1, int(rec.get("total", 1) or 1))),
                                        "rel_res": float(rec.get("rel_res", np.nan)),
                                    }
                                    for rec in trace_items_raw
                                ],
                            }
                        )
            if oof_pred is not None:
                try:
                    yhat_oof = np.asarray(yhat_test, dtype=np.float64).reshape(-1)
                    if int(yhat_oof.shape[0]) == int(fold_test_idx.shape[0]):
                        oof_pred[fold_test_idx] = yhat_oof
                except Exception:
                    pass
            if ml_tuning_cache is not None:
                pve = float(ml_tuning_cache.get("pve", np.nan))
            ttest = np.concatenate([train_pheno[fold_test_idx], yhat_test], axis=1)
            ttrain = None
            if need_fold_train_pred:
                y_train_fold = train_pheno[fold_train_idx]
                if fold_train_pred_local_idx is not None:
                    y_train_fold = y_train_fold[fold_train_pred_local_idx]
                ttrain = np.concatenate([y_train_fold, yhat_train], axis=1)

            ss_res = np.sum((ttest[:, 0] - ttest[:, 1]) ** 2)
            ss_tot = np.sum((ttest[:, 0] - ttest[:, 0].mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            pear = float(pearsonr(ttest[:, 0], ttest[:, 1]).statistic)
            spear = float(spearmanr(ttest[:, 0], ttest[:, 1]).statistic)
            elapsed = float(time.time() - t_fold)
            fold_rows.append((method, int(fold_id), pear, spear, r2, float(pve), elapsed))
            if progress_queue is not None:
                try:
                    progress_queue.put((method, 1))
                except Exception:
                    pass
            if progress_hook is not None:
                try:
                    progress_hook(method, 1)
                except Exception:
                    pass

            if r2 > best_r2:
                best_r2 = r2
                best_test = ttest
                best_train = ttrain
        cv_elapsed_now = float(
            np.nansum(np.asarray([float(r[6]) for r in fold_rows], dtype=np.float64))
        ) if len(fold_rows) > 0 else 0.0
        _emit_stage("cv_end", method=str(method), total=int(cv_total), elapsed=float(max(0.0, cv_elapsed_now)))

    if use_hash_cv_compact:
        _elapsed_total = float(max(0.0, time.time() - _t_method_begin))
        _cv_elapsed, _fit_elapsed, _predict_elapsed = _stage_elapsed_metrics(_elapsed_total)
        if train_sample_indices is None:
            raise ValueError(
                "Compact hash+CV mode requires train_sample_indices for final fit."
            )
        train_abs = np.ascontiguousarray(
            np.asarray(train_sample_indices, dtype=np.int64).reshape(-1),
            dtype=np.int64,
        )
        y_full = np.asarray(train_pheno, dtype=np.float64).reshape(-1)
        gram_f, row_sum_f, row_sq_sum_f, zy_f, _hs_f, _hk_f = _hash_packed_ztz_stats_for_gblup(
            packed_ctx=typing.cast(dict[str, typing.Any], packed_payload),
            hash_dim=int(hash_cv_dim),
            hash_seed=int(hash_cv_seed),
            standardize=bool(hash_cv_standardize),
            n_jobs=max(1, int(n_jobs)),
            train_sample_indices=train_abs,
            y_train=y_full,
        )
        fit_final = _fit_gblup_reml_from_hash_ztz_stats(
            gram=np.asarray(gram_f, dtype=np.float64),
            row_sum=np.asarray(row_sum_f, dtype=np.float64),
            row_sq_sum=np.asarray(row_sq_sum_f, dtype=np.float64),
            zy=np.asarray(zy_f, dtype=np.float64),
            n_train=int(train_abs.shape[0]),
        )
        if test_sample_indices is None:
            test_pred = np.zeros((0, 1), dtype=float)
        else:
            test_abs = np.ascontiguousarray(
                np.asarray(test_sample_indices, dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
            if int(test_abs.size) == 0:
                test_pred = np.zeros((0, 1), dtype=float)
            else:
                test_pred = _predict_hashed_gblup_test_from_compact(
                    packed_ctx=typing.cast(dict[str, typing.Any], packed_payload),
                    hash_dim=int(hash_cv_dim),
                    hash_seed=int(hash_cv_seed),
                    standardize=bool(hash_cv_standardize),
                    n_jobs=max(1, int(n_jobs)),
                    test_sample_indices=test_abs,
                    beta=float(fit_final["beta"]),
                    m_mean=np.asarray(fit_final["m_mean"], dtype=np.float64),
                    m_var_sum=float(fit_final["m_var_sum"]),
                    m_alpha=np.asarray(fit_final["m_alpha"], dtype=np.float64),
                    alpha_sum=float(fit_final["alpha_sum"]),
                    mean_sq=float(fit_final["mean_sq"]),
                    mean_malpha=float(fit_final["mean_malpha"]),
                )
        return {
            "method": method,
            "method_display": _method_display_name(method),
            "fold_rows": fold_rows,
            "best_test": best_test,
            "best_train": best_train,
            "oof_pred": (
                None
                if oof_pred is None
                else np.asarray(oof_pred, dtype=np.float64).reshape(-1)
            ),
            "oof_truth": (
                None
                if oof_truth is None
                else np.asarray(oof_truth, dtype=np.float64).reshape(-1)
            ),
            "oof_sample_ids": (
                None
                if train_sample_ids is None
                else np.asarray(train_sample_ids, dtype=str).reshape(-1)
            ),
            "test_pred": np.asarray(test_pred, dtype=float).reshape(-1, 1),
            "pve_final": float(fit_final["pve"]),
            "ml_tuning": ml_tuning_cache,
            "rrblup_final_state": None,
            "rrblup_final_cfg": None,
            "rrblup_cv_selected": None,
            "model_state": None,
            "cv_mode": "train_cv",
            "elapsed_total_sec": _elapsed_total,
            "elapsed_cv_sec": _cv_elapsed,
            "elapsed_fit_sec": _fit_elapsed,
            "elapsed_predict_sec": _predict_elapsed,
        }

    final_train_idx_arg = None
    final_test_idx_arg = None
    if (packed_payload is not None) and (method in packed_lmm_methods):
        if train_sample_indices is None or test_sample_indices is None:
            raise ValueError("Packed LMM requires train/test sample indices.")
        final_train = packed_payload
        final_test = packed_payload
        final_pca = False
        final_train_idx_arg = np.ascontiguousarray(train_sample_indices, dtype=np.int64)
        final_test_idx_arg = np.ascontiguousarray(test_sample_indices, dtype=np.int64)
    elif _is_gblup_method(str(method)) and (_gblup_method_kernel_mode(str(method)) in {"d", "ad"}):
        if train_snp_add is None or test_snp_add is None:
            raise ValueError(f"{method} requires additive raw genotype matrices.")
        final_train = train_snp_add
        final_test = test_snp_add
        final_pca = False
    elif method in _ML_METHOD_MAP:
        if train_snp_ml is None or test_snp_ml is None:
            raise ValueError(f"{method} requires raw genotype matrices.")
        final_train = train_snp_ml
        final_test = test_snp_ml
        final_pca = pca_dec
    else:
        final_train = train_snp
        final_test = test_snp
        final_pca = pca_dec

    # In CV mode with no external holdout (n_test == 0), the extra final
    # full-data fit does not contribute to fold metrics and only adds runtime.
    # Keep default behavior for all other cases.
    final_test_n = 0
    if (packed_payload is not None) and (method in packed_lmm_methods):
        final_test_n = int(0 if final_test_idx_arg is None else int(final_test_idx_arg.size))
    elif hasattr(final_test, "shape") and len(getattr(final_test, "shape")) >= 2:
        final_test_n = int(final_test.shape[1])
    has_external_test = bool(final_test_n > 0)

    skip_final_fit = False
    use_custom_gblup_final = bool(
        method == "GBLUP"
        and gblup_cv_grm_full is not None
        and (gblup_force_kinship or precomputed_train_grm is not None)
    )
    # Packed-GRM CV prebuild currently only provides train-train GRM; if we
    # still have external holdout samples but no precomputed test-train cross
    # kernel, fall back to the generic GSapi final route so test predictions
    # are produced instead of an empty vector.
    if (
        use_custom_gblup_final
        and has_external_test
        and (gblup_final_test_train is None)
    ):
        use_custom_gblup_final = False
    if cv_splits is not None:
        if use_custom_gblup_final:
            skip_final_fit = bool(
                gblup_final_test_train is None
                or int(gblup_final_test_train.shape[0]) == 0
            )
        elif (packed_payload is not None) and (method in packed_lmm_methods):
            skip_final_fit = bool(final_test_idx_arg is not None and int(final_test_idx_arg.size) == 0)
        else:
            skip_final_fit = bool(
                hasattr(final_test, "shape")
                and len(getattr(final_test, "shape")) >= 2
                and int(final_test.shape[1]) == 0
            )
    if bool(export_model) and _is_effect_export_supported(method):
        skip_final_fit = False

    if skip_final_fit:
        _emit_stage("fit_start", method=str(method))
        _emit_stage("fit_end", method=str(method), elapsed=0.0)
        _emit_stage("predict_start", method=str(method))
        test_pred = np.zeros((0, 1), dtype=float)
        _emit_stage("predict_end", method=str(method), elapsed=0.0)
        if len(fold_rows) > 0:
            pve_final = float(np.nanmean([float(r[5]) for r in fold_rows]))
        else:
            pve_final = float("nan")
        if method == "rrBLUP":
            rrblup_final_cfg = dict(rrblup_cfg_base)
            if (
                rrblup_cv_reuse_enabled
                and rrblup_cv_selected is not None
                and ("lr" in rrblup_cv_selected)
            ):
                rrblup_final_cfg["lambda_value"] = float(rrblup_cv_selected["lambda_value"])
                rr_sel_lr = float(rrblup_cv_selected.get("lr", np.nan))
                if np.isfinite(rr_sel_lr) and rr_sel_lr > 0.0:
                    rrblup_final_cfg["lr"] = float(rr_sel_lr)
                rrblup_final_cfg["lambda_auto"] = "off"
                rrblup_final_cfg["auto_grid"] = "off"
            tv = np.asarray(
                [float(x.get("pve_trainvar", np.nan)) for x in rrblup_pve_rows],
                dtype=np.float64,
            )
            lv = np.asarray(
                [float(x.get("pve_lambda", np.nan)) for x in rrblup_pve_rows],
                dtype=np.float64,
            )
            itv = np.asarray(
                [float(x.get("iter_like", np.nan)) for x in rrblup_pve_rows],
                dtype=np.float64,
            )
            solver_req_final = str(rrblup_solver).strip().lower()
            solver_eff_final = str(solver_req_final)
            solver_reason_final = ""
            for _row in rrblup_pve_rows:
                cand_req = str(_row.get("solver_requested", "")).strip().lower()
                cand_eff = str(_row.get("solver_effective", _row.get("solver", ""))).strip().lower()
                cand_reason = str(_row.get("solver_fallback_reason", "")).strip()
                if cand_req != "":
                    solver_req_final = str(cand_req)
                if cand_eff != "":
                    solver_eff_final = str(cand_eff)
                if solver_reason_final == "" and cand_reason != "":
                    solver_reason_final = str(cand_reason)
            rrblup_pve_final = {
                "phase": "final(skipped)",
                "solver": str(solver_eff_final),
                "solver_requested": str(solver_req_final),
                "solver_effective": str(solver_eff_final),
                "solver_fallback_reason": str(solver_reason_final),
                "pve_mode": str(rrblup_cfg_base.get("pve_mode", "lambda")).strip().lower(),
                "pve_used": _float_or_nan(pve_final),
                "pve_trainvar": (
                    float(np.nanmean(tv)) if int(tv.size) > 0 and np.any(np.isfinite(tv)) else float("nan")
                ),
                "pve_lambda": (
                    float(np.nanmean(lv)) if int(lv.size) > 0 and np.any(np.isfinite(lv)) else float("nan")
                ),
                "iter_like": (
                    float(np.nanmean(itv)) if int(itv.size) > 0 and np.any(np.isfinite(itv)) else float("nan")
                ),
            }
            rrblup_final_state = {
                "solver": str(solver_eff_final),
                "solver_requested": str(solver_req_final),
                "solver_effective": str(solver_eff_final),
            }
            if solver_reason_final != "":
                rrblup_final_state["solver_fallback_reason"] = str(solver_reason_final)
    elif use_custom_gblup_final:
        _emit_stage("fit_start", method=str(method))
        _t_fit_stage = time.time()
        fit = _fit_gblup_reml_from_grm(
            np.asarray(train_pheno, dtype=np.float64).reshape(-1),
            np.asarray(gblup_cv_grm_full, dtype=np.float64),
        )
        _emit_stage("fit_end", method=str(method), elapsed=float(max(0.0, time.time() - _t_fit_stage)))
        _emit_stage("predict_start", method=str(method))
        _t_pred_stage = time.time()
        alpha = np.asarray(fit["alpha"], dtype=np.float64).reshape(-1, 1)
        beta = float(fit["beta"])
        pve_final = float(fit["pve"])
        if gblup_final_test_train is None or int(gblup_final_test_train.shape[0]) == 0:
            test_pred = np.zeros((0, 1), dtype=float)
        else:
            test_pred = _predict_gblup_from_cross(
                cross_kernel=np.asarray(gblup_final_test_train, dtype=np.float64),
                alpha=alpha,
                beta=beta,
            )
        gblup_final_state = {
            "variance_component_path": "grm",
            "variance_component": "REML/GRM",
            "backend": ("packed active" if packed_payload is not None else "dense"),
        }
        sigma_g2_fit = float(fit.get("sigma_g2", np.nan))
        sigma_e2_fit = float(fit.get("sigma_e2", np.nan))
        lambda_reml_fit = float(fit.get("lambda_reml", np.nan))
        h2_fit = float(fit.get("pve", np.nan))
        if np.isfinite(sigma_g2_fit):
            gblup_final_state["sigma_g2"] = float(sigma_g2_fit)
        if np.isfinite(sigma_e2_fit):
            gblup_final_state["sigma_e2"] = float(sigma_e2_fit)
        if np.isfinite(lambda_reml_fit):
            gblup_final_state["lambda_reml"] = float(lambda_reml_fit)
        if np.isfinite(h2_fit):
            gblup_final_state["h2"] = float(h2_fit)
        _emit_stage("predict_end", method=str(method), elapsed=float(max(0.0, time.time() - _t_pred_stage)))
    else:
        rr_cfg_final: dict[str, typing.Any] | None = rrblup_adamw_cfg
        rr_state_final: dict[str, typing.Any] | None = None
        rr_progress_final = None
        gblup_state_final: dict[str, typing.Any] | None = None
        model_state_call: dict[str, typing.Any] | None = None
        bayes_r2_final_call: float | None = None
        bayes_state_final: dict[str, typing.Any] | None = None
        bayes_r2_final_key: str | None = None
        if method == "rrBLUP":
            rr_cfg_final = dict(rrblup_cfg_base)
            if (
                rrblup_cv_reuse_enabled
                and rrblup_cv_selected is not None
                and ("lr" in rrblup_cv_selected)
            ):
                rr_cfg_final["lambda_value"] = float(rrblup_cv_selected["lambda_value"])
                rr_sel_lr = float(rrblup_cv_selected.get("lr", np.nan))
                if np.isfinite(rr_sel_lr) and rr_sel_lr > 0.0:
                    rr_cfg_final["lr"] = float(rr_sel_lr)
                rr_cfg_final["lambda_auto"] = "off"
                rr_cfg_final["auto_grid"] = "off"
            rrblup_final_cfg = dict(rr_cfg_final)
            rr_state_final = {}
            if rrblup_progress_hook is not None:
                def _final_rr_progress(event: str, payload: dict[str, typing.Any]) -> None:
                    pp = dict(payload)
                    pp.setdefault("phase_label", "final")
                    rrblup_progress_hook(str(event), pp)
                rr_progress_final = _final_rr_progress
        if method in {"BayesA", "BayesB", "BayesCpi"}:
            if bayes_cv_reuse_enabled and np.isfinite(bayes_cv_shared_r2):
                bayes_r2_final_key = bayes_cv_shared_key
                bayes_r2_final_call = float(bayes_cv_shared_r2)
            else:
                bayes_r2_final_key = "final"
                if bayes_auto_r2_cache is not None:
                    cached_r2 = float(bayes_auto_r2_cache.get(bayes_r2_final_key, np.nan))
                    if np.isfinite(cached_r2):
                        bayes_r2_final_call = float(cached_r2)
            bayes_state_final = {}
        if _is_gblup_method(str(method)):
            gblup_state_final = {}
        if bool(export_model) and _is_effect_export_supported(method):
            model_state_call = {}
        _emit_stage("fit_start", method=str(method))
        _t_fit_stage = time.time()
        _, test_pred, pve_final = GSapi(
            train_pheno,
            final_train,
            final_test,
            method=method,
            PCAdec=final_pca,
            n_jobs=n_jobs,
            force_fast=force_fast,
            ml_fixed_params=(None if ml_tuning_cache is None else ml_tuning_cache.get("params")),
            need_train_pred=False,
            packed_train_indices=final_train_idx_arg,
            packed_test_indices=final_test_idx_arg,
            rrblup_solver=typing.cast(typing.Any, rrblup_solver),
            rrblup_adamw_cfg=rr_cfg_final,
            rrblup_runtime_state=rr_state_final,
            rrblup_progress_hook=rr_progress_final,
            gblup_runtime_state=gblup_state_final,
            bayes_auto_r2=bayes_r2_final_call,
            bayes_runtime_state=bayes_state_final,
            bayes_auto_cfg=bayes_auto_r2_cfg,
            model_state=model_state_call,
        )
        _emit_stage("fit_end", method=str(method), elapsed=float(max(0.0, time.time() - _t_fit_stage)))
        _emit_stage("predict_start", method=str(method))
        _emit_stage("predict_end", method=str(method), elapsed=0.0)
        if model_state_call is not None and len(model_state_call) > 0:
            model_state_final = dict(model_state_call)
        if _is_gblup_method(str(method)) and gblup_state_final is not None:
            gblup_final_state = dict(gblup_state_final)
        if method in {"BayesA", "BayesB", "BayesCpi"} and bayes_state_final is not None:
            r2_used_final = float(bayes_state_final.get("r2_used", np.nan))
            if np.isfinite(r2_used_final):
                bayes_r2_final = float(r2_used_final)
                bayes_r2_source_final = str(bayes_state_final.get("r2_source", "")).strip()
                try:
                    bayes_r2_n_used_final = int(bayes_state_final.get("r2_n_used", 0))
                except Exception:
                    bayes_r2_n_used_final = 0
                try:
                    bayes_r2_n_total_final = int(bayes_state_final.get("r2_n_total", 0))
                except Exception:
                    bayes_r2_n_total_final = 0
                if bayes_cv_reuse_enabled and np.isfinite(bayes_cv_shared_r2):
                    bayes_r2_source_final = "cv_shared_reuse"
                    if bayes_cv_shared_n_used > 0:
                        bayes_r2_n_used_final = int(bayes_cv_shared_n_used)
                    if bayes_cv_shared_n_total > 0:
                        bayes_r2_n_total_final = int(bayes_cv_shared_n_total)
                if (
                    (bayes_auto_r2_cache is not None)
                    and (bayes_r2_final_key is not None)
                    and (not (bayes_cv_reuse_enabled and np.isfinite(bayes_cv_shared_r2)))
                ):
                    bayes_auto_r2_cache[bayes_r2_final_key] = float(r2_used_final)
        if method == "rrBLUP" and rr_state_final is not None:
            mode_used = str((rr_cfg_final or {}).get("pve_mode", "lambda")).strip().lower()
            if mode_used not in {"lambda", "trainvar"}:
                mode_used = "lambda"
            pve_used_f = _float_or_nan(pve_final)
            pve_trainvar_f = _float_or_nan(rr_state_final.get("pve_trainvar", np.nan))
            pve_lambda_f = _float_or_nan(rr_state_final.get("pve_lambda", np.nan))
            if (not np.isfinite(pve_trainvar_f)) and mode_used == "trainvar":
                pve_trainvar_f = pve_used_f
            if (not np.isfinite(pve_lambda_f)) and mode_used == "lambda":
                pve_lambda_f = pve_used_f
            solver_req_final = str(
                rr_state_final.get("solver_requested", str(rrblup_solver).strip().lower())
            ).strip().lower()
            solver_eff_final = str(
                rr_state_final.get(
                    "solver_effective",
                    rr_state_final.get("solver", solver_req_final),
                )
            ).strip().lower()
            solver_reason_final = str(rr_state_final.get("solver_fallback_reason", "")).strip()
            rrblup_pve_final = {
                "phase": "final",
                "solver": str(solver_eff_final),
                "solver_requested": str(solver_req_final),
                "solver_effective": str(solver_eff_final),
                "solver_fallback_reason": str(solver_reason_final),
                "pve_mode": str(mode_used),
                "pve_used": pve_used_f,
                "pve_trainvar": pve_trainvar_f,
                "pve_lambda": pve_lambda_f,
                "pve_exact": _float_or_nan(rr_state_final.get("pve_exact", np.nan)),
                "iter_like": _float_or_nan(
                    rr_state_final.get(
                        "pcg_iters",
                        rr_state_final.get("epochs_ran", rr_state_final.get("best_epoch", np.nan)),
                    )
                ),
            }
            rrblup_final_state = dict(rr_state_final)
            trace_items_final = list(
                typing.cast(
                    list[dict[str, typing.Any]] | None,
                    rr_state_final.get("pcg_rel_res_trace"),
                )
                or []
            )
            if len(trace_items_final) > 0:
                rrblup_pcg_trace_rows.append(
                    {
                        "phase": "final",
                        "phase_label": "final",
                        "fold": None,
                        "solver": str(solver_eff_final),
                        "converged": bool(rr_state_final.get("pcg_converged", False)),
                        "iters": int(max(0, int(rr_state_final.get("pcg_iters", 0) or 0))),
                        "max_iter": int(
                            max(
                                1,
                                int(
                                    (rr_cfg_final or {}).get(
                                        "pcg_max_iter",
                                        rrblup_cfg_base.get("pcg_max_iter", 100),
                                    )
                                ),
                            )
                        ),
                        "trace": [
                            {
                                "iter": int(max(0, int(rec.get("iter", 0) or 0))),
                                "total": int(max(1, int(rec.get("total", 1) or 1))),
                                "rel_res": float(rec.get("rel_res", np.nan)),
                            }
                            for rec in trace_items_final
                        ],
                    }
                )
        if ml_tuning_cache is not None:
            pve_final = float(ml_tuning_cache.get("pve", np.nan))
    if (
        method in {"BayesA", "BayesB", "BayesCpi"}
        and (not np.isfinite(bayes_r2_final))
        and len(bayes_r2_rows) > 0
    ):
        bayes_r2_final = float(np.nanmean(np.asarray(bayes_r2_rows, dtype=np.float64)))
    if (
        method in {"BayesA", "BayesB", "BayesCpi"}
        and (bayes_r2_source_final == "")
        and bayes_cv_reuse_enabled
        and np.isfinite(bayes_cv_shared_r2)
    ):
        bayes_r2_source_final = (
            "cv_shared_reuse"
            if bayes_cv_shared_source == ""
            else f"cv_shared_reuse:{bayes_cv_shared_source}"
        )
        if bayes_r2_n_used_final <= 0 and bayes_cv_shared_n_used > 0:
            bayes_r2_n_used_final = int(bayes_cv_shared_n_used)
        if bayes_r2_n_total_final <= 0 and bayes_cv_shared_n_total > 0:
            bayes_r2_n_total_final = int(bayes_cv_shared_n_total)
    _elapsed_total = float(max(0.0, time.time() - _t_method_begin))
    _cv_elapsed, _fit_elapsed, _predict_elapsed = _stage_elapsed_metrics(_elapsed_total)
    return {
        "method": method,
        "method_display": _method_display_name(method),
        "fold_rows": fold_rows,
        "rrblup_pve_rows": rrblup_pve_rows,
        "rrblup_pcg_trace_rows": rrblup_pcg_trace_rows,
        "rrblup_pve_final": rrblup_pve_final,
        "best_test": best_test,
        "best_train": best_train,
        "oof_pred": (
            None
            if oof_pred is None
            else np.asarray(oof_pred, dtype=np.float64).reshape(-1)
        ),
        "oof_truth": (
            None
            if oof_truth is None
            else np.asarray(oof_truth, dtype=np.float64).reshape(-1)
        ),
        "oof_sample_ids": (
            None
            if train_sample_ids is None
            else np.asarray(train_sample_ids, dtype=str).reshape(-1)
        ),
        "test_pred": test_pred,
        "pve_final": float(pve_final),
        "bayes_r2_final": float(bayes_r2_final),
        "bayes_r2_source_final": str(bayes_r2_source_final),
        "bayes_r2_n_used_final": int(bayes_r2_n_used_final),
        "bayes_r2_n_total_final": int(bayes_r2_n_total_final),
        "ml_tuning": ml_tuning_cache,
        "rrblup_final_state": rrblup_final_state,
        "rrblup_final_cfg": rrblup_final_cfg,
        "rrblup_cv_selected": rrblup_cv_selected,
        "gblup_final_state": gblup_final_state,
        "model_state": model_state_final,
        "cv_mode": "train_cv",
        "elapsed_total_sec": _elapsed_total,
        "elapsed_cv_sec": _cv_elapsed,
        "elapsed_fit_sec": _fit_elapsed,
        "elapsed_predict_sec": _predict_elapsed,
    }


def _predict_from_loaded_model_state(
    *,
    method: str,
    model_state: dict[str, typing.Any],
    dense_matrix: np.ndarray | None,
    packed_ctx: dict[str, typing.Any] | None,
    packed_sample_indices: np.ndarray | None,
) -> np.ndarray:
    kind = str(model_state.get("kind", "")).strip().lower()
    m = str(method)
    if kind == "rrblup_linear":
        alpha = float(model_state.get("alpha", 0.0))
        beta = np.asarray(model_state.get("beta", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        block_rows = int(max(1, int(model_state.get("snp_block_size", 2048))))
        sample_chunk = int(max(1, int(model_state.get("sample_chunk_size", 4096))))
        is_std = bool(model_state.get("standardized", True))
        if packed_ctx is not None and packed_sample_indices is not None:
            if is_std:
                row_mean = model_state.get("row_mean", None)
                row_inv = model_state.get("row_inv_sd", None)
                if row_mean is None or row_inv is None:
                    row_mean_arr, row_inv_arr = _ensure_packed_standard_stats_cached(packed_ctx)
                else:
                    row_mean_arr = np.ascontiguousarray(
                        np.asarray(row_mean, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    row_inv_arr = np.ascontiguousarray(
                        np.asarray(row_inv, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                return _predict_rrblup_packed_from_beta(
                    packed_ctx=packed_ctx,
                    sample_indices=np.ascontiguousarray(
                        np.asarray(packed_sample_indices, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    ),
                    alpha=alpha,
                    beta=beta,
                    row_mean=row_mean_arr,
                    row_inv_sd=row_inv_arr,
                    snp_block_size=block_rows,
                    sample_chunk_size=sample_chunk,
                )
            return _predict_rrblup_packed_raw_from_beta(
                packed_ctx=packed_ctx,
                sample_indices=np.ascontiguousarray(
                    np.asarray(packed_sample_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                ),
                alpha=alpha,
                beta=beta,
                row_block_size=block_rows,
                sample_chunk_size=sample_chunk,
            )
        if dense_matrix is None:
            raise ValueError(f"Loaded model {m} requires dense genotype matrix for prediction.")
        return _predict_rrblup_dense_from_beta(
            X=np.asarray(dense_matrix, dtype=np.float32),
            alpha=alpha,
            beta=beta,
            snp_block_size=block_rows,
        )

    if kind == "bayes_linear":
        alpha0 = float(model_state.get("alpha", 0.0))
        beta = np.asarray(model_state.get("beta", np.zeros((0,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        block_rows = int(max(1, int(model_state.get("snp_block_size", 2048))))
        sample_chunk = int(max(1, int(model_state.get("sample_chunk_size", 4096))))
        is_std = bool(model_state.get("standardized", True))
        if packed_ctx is not None and packed_sample_indices is not None:
            if is_std:
                row_mean = model_state.get("row_mean", None)
                row_inv = model_state.get("row_inv_sd", None)
                if row_mean is None or row_inv is None:
                    row_mean_arr, row_inv_arr = _ensure_packed_standard_stats_cached(packed_ctx)
                else:
                    row_mean_arr = np.ascontiguousarray(
                        np.asarray(row_mean, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                    row_inv_arr = np.ascontiguousarray(
                        np.asarray(row_inv, dtype=np.float32).reshape(-1),
                        dtype=np.float32,
                    )
                return _predict_bayes_packed_from_effects(
                    packed_ctx=packed_ctx,
                    sample_indices=np.ascontiguousarray(
                        np.asarray(packed_sample_indices, dtype=np.int64).reshape(-1),
                        dtype=np.int64,
                    ),
                    alpha0=alpha0,
                    beta=beta,
                    row_mean=row_mean_arr,
                    row_inv_sd=row_inv_arr,
                    snp_block_size=block_rows,
                    sample_chunk_size=sample_chunk,
                )
            return _predict_bayes_packed_raw_from_effects(
                packed_ctx=packed_ctx,
                sample_indices=np.ascontiguousarray(
                    np.asarray(packed_sample_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                ),
                alpha0=alpha0,
                beta=beta,
                row_block_size=block_rows,
                sample_chunk_size=sample_chunk,
            )
        if dense_matrix is None:
            raise ValueError(f"Loaded model {m} requires dense genotype matrix for prediction.")
        x = np.asarray(dense_matrix, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Dense Bayes predict expects 2D matrix, got {x.shape}.")
        if int(x.shape[0]) != int(beta.shape[0]):
            raise ValueError(
                f"Dense Bayes marker mismatch: matrix m={x.shape[0]} vs beta m={beta.shape[0]}."
            )
        pred = np.full((int(x.shape[1]),), float(alpha0), dtype=np.float64)
        pred += np.asarray(x.T @ beta, dtype=np.float64).reshape(-1)
        return np.asarray(pred, dtype=np.float64).reshape(-1, 1)

    if kind == "mlsk_model":
        estimator = model_state.get("estimator", None)
        if estimator is None:
            raise ValueError(f"Loaded ML model for {m} is missing estimator payload.")
        if dense_matrix is None:
            raise ValueError(f"Loaded model {m} requires dense genotype matrix for prediction.")
        mat = np.asarray(dense_matrix, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"Loaded ML model expects 2D matrix, got {mat.shape}.")
        marker_means = np.ascontiguousarray(
            np.asarray(model_state.get("marker_means", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
        m_expect = int(marker_means.shape[0])
        if int(mat.shape[0]) == m_expect:
            x = np.ascontiguousarray(mat.T, dtype=np.float32)
        elif int(mat.shape[1]) == m_expect:
            x = np.ascontiguousarray(mat, dtype=np.float32)
        else:
            raise ValueError(
                f"Loaded ML model marker mismatch: expected {m_expect}, got matrix {mat.shape}."
            )
        if np.isnan(x).any():
            idx = np.where(np.isnan(x))
            x = x.copy()
            x[idx] = marker_means[idx[1]]
        pred = np.asarray(estimator.predict(x), dtype=np.float64).reshape(-1, 1)
        return pred

    raise ValueError(f"Unsupported loaded model kind for method={m}: {kind!r}")


def _run_loaded_model_task(
    *,
    method: str,
    model_state: dict[str, typing.Any],
    train_pheno: np.ndarray,
    train_snp: np.ndarray | None,
    test_snp: np.ndarray | None,
    train_snp_ml: np.ndarray | None,
    test_snp_ml: np.ndarray | None,
    packed_ctx: dict[str, typing.Any] | None,
    train_sample_indices: np.ndarray | None,
    test_sample_indices: np.ndarray | None,
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None,
    limit_predtrain: int | None,
) -> dict[str, typing.Any]:
    _t_loaded_begin = time.time()
    fold_rows: list[tuple[str, int, float, float, float, float, float]] = []
    best_test = None
    best_train = None
    best_r2 = -np.inf
    pve_final = float(model_state.get("pve", np.nan))

    def _predict_local(
        *,
        idx_local: np.ndarray | None,
        use_test: bool,
    ) -> np.ndarray:
        if method in _ML_METHOD_MAP:
            base = test_snp_ml if use_test else train_snp_ml
            if base is None:
                return np.zeros((0, 1), dtype=np.float64)
            if int(np.asarray(base).shape[1]) == 0:
                return np.zeros((0, 1), dtype=np.float64)
            if idx_local is None:
                x = base
            else:
                x = np.asarray(base[:, idx_local], dtype=np.float32)
            if int(np.asarray(x).shape[1]) == 0:
                return np.zeros((0, 1), dtype=np.float64)
            return _predict_from_loaded_model_state(
                method=method,
                model_state=model_state,
                dense_matrix=np.asarray(x, dtype=np.float32),
                packed_ctx=None,
                packed_sample_indices=None,
            )
        if packed_ctx is not None and train_sample_indices is not None:
            if use_test:
                if test_sample_indices is None:
                    base_abs = np.zeros((0,), dtype=np.int64)
                else:
                    base_abs = np.asarray(test_sample_indices, dtype=np.int64).reshape(-1)
            else:
                base_abs = np.asarray(train_sample_indices, dtype=np.int64).reshape(-1)
            if int(base_abs.size) == 0:
                return np.zeros((0, 1), dtype=np.float64)
            if idx_local is None:
                abs_idx = np.ascontiguousarray(base_abs, dtype=np.int64)
            else:
                abs_idx = np.ascontiguousarray(base_abs[np.asarray(idx_local, dtype=np.int64).reshape(-1)], dtype=np.int64)
            return _predict_from_loaded_model_state(
                method=method,
                model_state=model_state,
                dense_matrix=None,
                packed_ctx=packed_ctx,
                packed_sample_indices=abs_idx,
            )
        base_dense = test_snp if use_test else train_snp
        if base_dense is None:
            return np.zeros((0, 1), dtype=np.float64)
        if int(np.asarray(base_dense).shape[1]) == 0:
            return np.zeros((0, 1), dtype=np.float64)
        if idx_local is None:
            x_dense = base_dense
        else:
            x_dense = np.asarray(base_dense[:, idx_local], dtype=np.float32)
        if int(np.asarray(x_dense).shape[1]) == 0:
            return np.zeros((0, 1), dtype=np.float64)
        return _predict_from_loaded_model_state(
            method=method,
            model_state=model_state,
            dense_matrix=np.asarray(x_dense, dtype=np.float32),
            packed_ctx=None,
            packed_sample_indices=None,
        )

    if cv_splits is not None:
        for fold_id, (test_idx, train_idx) in enumerate(cv_splits, start=1):
            t0 = time.time()
            fold_test_idx = np.asarray(test_idx, dtype=np.int64).reshape(-1)
            fold_train_idx = np.asarray(train_idx, dtype=np.int64).reshape(-1)
            y_test = np.asarray(train_pheno[fold_test_idx], dtype=np.float64).reshape(-1, 1)
            yhat_test = _predict_local(idx_local=fold_test_idx, use_test=False)

            pick_local = _resolve_train_pred_local_indices(
                n_train_fold=int(fold_train_idx.shape[0]),
                limit_predtrain=limit_predtrain,
                fold_id=int(fold_id),
            )
            if pick_local is None:
                pred_train_idx = fold_train_idx
            elif int(pick_local.size) == 0:
                pred_train_idx = np.zeros((0,), dtype=np.int64)
            else:
                pred_train_idx = np.asarray(fold_train_idx[pick_local], dtype=np.int64).reshape(-1)

            if int(pred_train_idx.size) > 0:
                y_train_part = np.asarray(train_pheno[pred_train_idx], dtype=np.float64).reshape(-1, 1)
                yhat_train = _predict_local(idx_local=pred_train_idx, use_test=False)
                ttrain = np.concatenate([y_train_part, yhat_train], axis=1)
            else:
                ttrain = np.zeros((0, 2), dtype=np.float64)

            ttest = np.concatenate([y_test, yhat_test], axis=1)
            ss_res = np.sum((ttest[:, 0] - ttest[:, 1]) ** 2)
            ss_tot = np.sum((ttest[:, 0] - ttest[:, 0].mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            pear = float(pearsonr(ttest[:, 0], ttest[:, 1]).statistic)
            spear = float(spearmanr(ttest[:, 0], ttest[:, 1]).statistic)
            elapsed = float(time.time() - t0)
            fold_rows.append((method, int(fold_id), pear, spear, r2, float(pve_final), elapsed))
            if r2 > best_r2:
                best_r2 = r2
                best_test = ttest
                best_train = ttrain

    test_pred = _predict_local(idx_local=None, use_test=True)
    _elapsed_total = float(max(0.0, time.time() - _t_loaded_begin))
    _cv_elapsed = float(
        np.nansum(
            np.asarray(
                [float(r[6]) for r in fold_rows],
                dtype=np.float64,
            )
        )
    ) if len(fold_rows) > 0 else 0.0
    _cv_elapsed = float(max(0.0, _cv_elapsed))
    _predict_elapsed = float(max(0.0, _elapsed_total - _cv_elapsed))
    return {
        "method": str(method),
        "method_display": _method_display_name(str(method)),
        "fold_rows": fold_rows,
        "best_test": best_test,
        "best_train": best_train,
        "oof_pred": None,
        "oof_truth": None,
        "oof_sample_ids": None,
        "test_pred": np.asarray(test_pred, dtype=np.float64).reshape(-1, 1),
        "pve_final": float(pve_final),
        "ml_tuning": None,
        "rrblup_final_state": None,
        "rrblup_final_cfg": None,
        "rrblup_cv_selected": None,
        "gblup_final_state": None,
        "model_state": dict(model_state),
        "cv_mode": ("model_eval" if cv_splits is not None else "predict_only"),
        "elapsed_total_sec": _elapsed_total,
        "elapsed_cv_sec": _cv_elapsed,
        "elapsed_fit_sec": 0.0,
        "elapsed_predict_sec": _predict_elapsed,
    }


def _run_methods_parallel(
    methods: list[str],
    train_pheno: np.ndarray,
    train_snp: np.ndarray | None,
    test_snp: np.ndarray | None,
    train_snp_add: np.ndarray | None,
    test_snp_add: np.ndarray | None,
    train_snp_ml: np.ndarray | None,
    test_snp_ml: np.ndarray | None,
    pca_dec: bool,
    cv_splits: typing.Optional[list[tuple[np.ndarray, np.ndarray]]],
    n_jobs: int,
    strict_cv: bool,
    force_fast: bool = False,
    packed_ctx: dict[str, typing.Any] | None = None,
    train_sample_indices: np.ndarray | None = None,
    test_sample_indices: np.ndarray | None = None,
    train_sample_ids: np.ndarray | None = None,
    limit_predtrain: int | None = None,
    elapsed_offset_by_method: dict[str, float] | None = None,
    rrblup_solver: str = "pcg",
    rrblup_adamw_cfg: dict[str, typing.Any] | None = None,
    bayes_auto_r2_cfg: dict[str, typing.Any] | None = None,
    emit_cv_progress_bar: bool = True,
    emit_method_summary: bool = True,
    on_method_start: typing.Callable[[str], None] | None = None,
    on_method_complete: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
) -> list[dict[str, typing.Any]]:
    if len(methods) == 0:
        return []

    model_n_jobs = max(1, int(n_jobs))
    offset_map = dict(elapsed_offset_by_method or {})
    method_display_map = {
        str(m): _method_display_name(str(m))
        for m in methods
    }

    def _method_display(name: str) -> str:
        return str(method_display_map.get(str(name), str(name)))

    def _method_offset(name: str) -> float:
        try:
            return max(0.0, float(offset_map.get(str(name), 0.0)))
        except Exception:
            return 0.0

    def _method_detail_rows(
        method: str,
        result: dict[str, typing.Any],
    ) -> list[tuple[str, str]]:
        m = str(method)
        rows: list[tuple[str, str]] = []
        if _is_gblup_method(m):
            mode = _gblup_method_kernel_mode(m)
            rows.append(("kernel", _gblup_mode_label(mode)))
            gblup_state = dict(
                typing.cast(
                    dict[str, typing.Any] | None,
                    result.get("gblup_final_state"),
                )
                or {}
            )
            vc_label = str(gblup_state.get("variance_component", "")).strip()
            vc_path = str(gblup_state.get("variance_component_path", "")).strip().lower()
            eigh_backend = str(gblup_state.get("eigh_backend", "")).strip().lower()
            if vc_label == "":
                if mode == "ad":
                    vc_label = "REML/GRM"
                elif mode == "d":
                    vc_label = "REML/FaST"
                elif vc_path in {"fast", "marker_fast"} or eigh_backend.startswith("marker_fast"):
                    vc_label = "REML/FaST"
                elif vc_path == "grm" or eigh_backend != "":
                    vc_label = "REML/GRM"
                else:
                    vc_label = "REML/FaST"
            rows.append(("method", vc_label))
            if mode == "ad":
                rows.append(("backend", "pyBLUP/JAX"))
            elif mode == "d":
                backend_d = str(gblup_state.get("backend", "pyBLUP/FaST(d)")).strip()
                rows.append(("backend", backend_d if backend_d != "" else "pyBLUP/FaST(d)"))
            else:
                rows.append(
                    (
                        "backend",
                        ("packed active" if _looks_like_packed_payload(packed_ctx) else "dense"),
                    )
                )
            sigma_g2 = float(gblup_state.get("sigma_g2", np.nan))
            sigma_e2 = float(gblup_state.get("sigma_e2", np.nan))
            sigma_a2 = float(gblup_state.get("sigma_a2", np.nan))
            sigma_d2 = float(gblup_state.get("sigma_d2", np.nan))
            va = sigma_a2 if np.isfinite(sigma_a2) else sigma_g2
            vd = sigma_d2 if np.isfinite(sigma_d2) else sigma_g2
            def _fmt_var_or_na(x: float) -> str:
                return f"{x:.6g}" if np.isfinite(x) else "NA"
            if mode in {"a", "ad"}:
                rows.append(("Va", _fmt_var_or_na(va)))
            if mode in {"d", "ad"}:
                rows.append(("Vd", _fmt_var_or_na(vd)))
            rows.append(("Ve", _fmt_var_or_na(sigma_e2)))
            try:
                pve_final = float(result.get("pve_final", np.nan))
            except Exception:
                pve_final = float("nan")
            rows.append(("PVE", f"{pve_final:.3f}" if np.isfinite(pve_final) else "NA"))
            return rows

        if m == "rrBLUP":
            rr_state = dict(
                typing.cast(
                    dict[str, typing.Any] | None,
                    result.get("rrblup_final_state"),
                )
                or {}
            )
            rr_cfg = dict(
                typing.cast(
                    dict[str, typing.Any] | None,
                    result.get("rrblup_final_cfg"),
                )
                or dict(rrblup_adamw_cfg or {})
            )
            solver_req = str(
                rr_state.get("solver_requested", str(rrblup_solver).strip().lower())
            ).strip().lower()
            solver_used = str(
                rr_state.get(
                    "solver_effective",
                    rr_state.get("solver", solver_req),
                )
            ).strip().lower()

            if solver_used == "pcg":
                rows.append(
                    (
                        "PCG",
                        (
                            f"tol={float(rr_cfg.get('pcg_tol', np.nan)):.1e}, "
                            f"max_iter={int(max(1, int(rr_cfg.get('pcg_max_iter', 0))))}"
                        ),
                    )
                )
                vc_method = str(rr_state.get("lambda_vc_method", "")).strip()
                if vc_method == "":
                    lam_src = str(rr_state.get("lambda_source", "")).strip().lower()
                    if lam_src.startswith("he"):
                        vc_method = "HE"
                    elif "fast_reml" in lam_src:
                        vc_method = "FaSTreml"
                    elif "subsample" in lam_src:
                        vc_method = "Sub/REML"
                    elif lam_src != "":
                        vc_method = "GRMreml"
                vc_method = _rrblup_vc_method_display(vc_method)
                if vc_method != "":
                    rows.append(("method", vc_method))
                he_thread_policy = str(rr_state.get("lambda_he_thread_policy", "")).strip()
                if he_thread_policy != "":
                    he_blas_threads = int(max(0, int(rr_state.get("lambda_he_stage_blas_threads", 0) or 0)))
                    he_rayon_threads = int(max(0, int(rr_state.get("lambda_he_stage_rayon_threads", 0) or 0)))
                    he_threads_arg = int(max(0, int(rr_state.get("lambda_he_threads_arg", 0) or 0)))
                    # rows.append(
                    #     (
                    #         "HE threads",
                    #         (
                    #             f"{he_thread_policy} "
                    #             f"(blas={he_blas_threads}, "
                    #             f"rayon={he_rayon_threads}, "
                    #             f"he={he_threads_arg})"
                    #         ),
                    #     )
                    # )
                va = float(rr_state.get("lambda_vc_sigma_g2", np.nan))
                ve = float(rr_state.get("lambda_vc_sigma_e2", np.nan))
                pve_vc = float(rr_state.get("lambda_vc_pve", np.nan))
                if not np.isfinite(pve_vc):
                    pve_vc = float(rr_state.get("pve_used", np.nan))
                rows.append(("Va", f"{va:.6g}" if np.isfinite(va) else "NA"))
                rows.append(("Ve", f"{ve:.6g}" if np.isfinite(ve) else "NA"))
                rows.append(("PVE", f"{pve_vc:.3f}" if np.isfinite(pve_vc) else "NA"))
                return rows
            elif solver_used == "adamw":
                if solver_req != solver_used:
                    rows.append(("solver", f"{solver_req} -> {solver_used.upper()}"))
                else:
                    rows.append(("solver", solver_used.upper()))
                solver_fallback_reason = str(rr_state.get("solver_fallback_reason", "")).strip()
                if solver_fallback_reason != "":
                    rows.append(("solver note", solver_fallback_reason))
                lambda_source = str(rr_state.get("lambda_source", "")).strip()
                if lambda_source != "":
                    rows.append(("lambda source", lambda_source))
                lam_raw = float(rr_state.get("selected_lambda", rr_cfg.get("lambda_value", np.nan)))
                if np.isfinite(lam_raw):
                    lam_scale = str(rr_cfg.get("lambda_scale", "equation")).strip().lower()
                    rows.append(("lambda", f"{lam_raw:.6g} ({lam_scale})"))
                rows.append(
                    (
                        "AdamW",
                        (
                            f"lr={float(rr_cfg.get('lr', np.nan)):.6g}, "
                            f"batch={int(max(1, int(rr_cfg.get('batch_size', 0))))}, "
                            f"epochs={int(max(1, int(rr_cfg.get('epochs', 0))))}"
                        ),
                    )
                )
            return rows

        if m in {"BayesA", "BayesB", "BayesCpi"}:
            rows.append(("n_iter/burnin/thin", "400/200/1"))
            r2_blup = float(result.get("bayes_r2_final", np.nan))
            r2_src = str(result.get("bayes_r2_source_final", "")).strip()
            r2_n_used = int(max(0, int(result.get("bayes_r2_n_used_final", 0) or 0)))
            r2_n_total = int(max(0, int(result.get("bayes_r2_n_total_final", 0) or 0)))
            if np.isfinite(r2_blup):
                if r2_n_total > 0 and r2_n_used > 0 and r2_n_used < r2_n_total:
                    rows.append(
                        ("r2", f"auto (BLUP pve={r2_blup:.3g}, n={r2_n_used}/{r2_n_total})")
                    )
                elif r2_src.startswith("cv_shared_reuse"):
                    rows.append(("r2", f"auto (BLUP pve={r2_blup:.6g}, cv-shared)"))
                else:
                    rows.append(("r2", f"auto (BLUP pve={r2_blup:.6g})"))
            else:
                rows.append(("r2", "auto (BLUP pve=NA)"))
            if m in {"BayesB", "BayesCpi"}:
                rows.append(("prob_in/counts", "0.5/5.0"))
            return rows

        if m in _ML_METHOD_MAP:
            tune = typing.cast(dict[str, typing.Any] | None, result.get("ml_tuning"))
            if tune is None:
                return rows
            params = dict(typing.cast(dict[str, typing.Any], tune.get("params", {})))
            key_pref = {
                "RF": ["n_estimators", "max_features", "max_depth"],
                "ET": ["n_estimators", "max_features", "max_depth"],
                "GBDT": ["max_iter", "learning_rate", "max_depth"],
                "XGB": ["n_estimators", "learning_rate", "max_depth"],
                "SVM": ["C", "gamma", "epsilon"],
                "ENET": ["alpha", "l1_ratio"],
            }
            shown = 0
            for k in key_pref.get(m, []):
                if k in params:
                    rows.append((str(k), str(params.get(k))))
                    shown += 1
            if shown == 0:
                for k, v in list(sorted(params.items(), key=lambda kv: str(kv[0])))[:3]:
                    rows.append((str(k), str(v)))
            return rows

        return rows

    # Keep detail value column aligned across methods in the same run.
    # Example: GBLUP `additive`, rrBLUP `auto -> EXACT`, Bayes `400/200/1`
    # should start from one shared column.
    detail_key_width = 0
    for _m in methods:
        try:
            for _k, _v in _method_detail_rows(str(_m), {}):
                detail_key_width = max(detail_key_width, len(str(_k)))
        except Exception:
            continue
    detail_key_width = max(8, int(detail_key_width))

    def _attach_method_detail_rows(
        method: str,
        result: dict[str, typing.Any] | None,
    ) -> list[tuple[str, str]]:
        if not isinstance(result, dict):
            return []
        try:
            rows = _method_detail_rows(str(method), result)
        except Exception:
            rows = []
        safe_rows = [(str(k), str(v)) for k, v in rows]
        result["method_detail_rows"] = safe_rows
        return safe_rows

    def _emit_method_details_plain(
        method: str,
        result: dict[str, typing.Any],
        *,
        line_printer: typing.Callable[[str], None],
    ) -> None:
        rows = _attach_method_detail_rows(str(method), result)
        if len(rows) == 0:
            return
        key_w = max(int(detail_key_width), max(len(str(k)) for k, _ in rows))
        for k, v in rows:
            line_printer(f"{str(k):<{key_w}}  {str(v)}")

    def _method_expects_rrblup_iter_progress(name: str) -> bool:
        if str(name) != "rrBLUP":
            return False
        solver_mode = str(rrblup_solver).strip().lower()
        if solver_mode == "exact":
            return False
        if solver_mode == "pcg":
            return True
        if solver_mode == "adamw":
            return True
        n_snp = 0
        if _looks_like_packed_payload(packed_ctx):
            try:
                n_snp = int(typing.cast(dict[str, typing.Any], packed_ctx)["packed"].shape[0])
            except Exception:
                n_snp = 0
        elif train_snp is not None:
            n_snp = int(np.asarray(train_snp).shape[0])
        if n_snp <= 0:
            return False
        n_train_candidates = [int(np.asarray(train_pheno).reshape(-1).shape[0])]
        if cv_splits is not None:
            for _test_idx, train_idx in cv_splits:
                n_train_candidates.append(int(np.asarray(train_idx).reshape(-1).shape[0]))
        for n_train_try in n_train_candidates:
            resolved = _resolve_rrblup_solver(
                solver=solver_mode,
                n_train=int(max(0, n_train_try)),
                n_snp=int(n_snp),
                cfg=rrblup_adamw_cfg,
            )
            if str(resolved) in {"adamw", "pcg"}:
                return True
        return False

    rrblup_progress_enabled_by_method = {
        str(m): bool(_method_expects_rrblup_iter_progress(m))
        for m in methods
    }

    show_method_progress = (cv_splits is not None) and bool(emit_cv_progress_bar)
    # Force serial execution for all GS methods to avoid cross-method contention
    # and keep runtime behavior consistent.
    method_parallel_jobs = 1
    bayes_auto_r2_cache_shared: dict[str, float] = {}
    train_snp_runtime = train_snp
    test_snp_runtime = test_snp
    train_snp_add_runtime = train_snp_add
    test_snp_add_runtime = test_snp_add
    train_snp_ml_runtime = train_snp_ml
    test_snp_ml_runtime = test_snp_ml
    ml_dense_materialized = False

    def _ensure_ml_dense_runtime() -> None:
        nonlocal train_snp_runtime, test_snp_runtime
        nonlocal train_snp_add_runtime, test_snp_add_runtime
        nonlocal train_snp_ml_runtime, test_snp_ml_runtime
        nonlocal ml_dense_materialized
        if ml_dense_materialized:
            return
        # If dense/ml arrays are already provided (including memmap-backed slices),
        # materialize once to in-RAM float32 for sklearn/XGBoost routes.
        if train_snp_ml_runtime is not None and test_snp_ml_runtime is not None:
            train_snp_ml_runtime = np.ascontiguousarray(
                np.asarray(train_snp_ml_runtime, dtype=np.float32),
                dtype=np.float32,
            )
            test_snp_ml_runtime = np.ascontiguousarray(
                np.asarray(test_snp_ml_runtime, dtype=np.float32),
                dtype=np.float32,
            )
            if train_snp_runtime is None:
                train_snp_runtime = train_snp_ml_runtime
            if test_snp_runtime is None:
                test_snp_runtime = test_snp_ml_runtime
            ml_dense_materialized = True
            return
        if _looks_like_packed_payload(packed_ctx):
            if train_sample_indices is None or test_sample_indices is None:
                raise ValueError("Packed->ML dense decode requires train/test sample indices.")
            packed_payload = typing.cast(dict[str, typing.Any], packed_ctx)
            train_snp_ml_runtime = _decode_packed_subset_to_dense_raw_f32(
                packed_payload,
                np.ascontiguousarray(
                    np.asarray(train_sample_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                ),
            )
            test_snp_ml_runtime = _decode_packed_subset_to_dense_raw_f32(
                packed_payload,
                np.ascontiguousarray(
                    np.asarray(test_sample_indices, dtype=np.int64).reshape(-1),
                    dtype=np.int64,
                ),
            )
            if train_snp_runtime is None:
                train_snp_runtime = train_snp_ml_runtime
            if test_snp_runtime is None:
                test_snp_runtime = test_snp_ml_runtime
            if train_snp_add_runtime is None:
                train_snp_add_runtime = train_snp_ml_runtime
            if test_snp_add_runtime is None:
                test_snp_add_runtime = test_snp_ml_runtime
            ml_dense_materialized = True
            return
        if train_snp_runtime is None or test_snp_runtime is None:
            raise ValueError("ML models require dense train/test genotype matrices.")
        train_snp_ml_runtime = np.ascontiguousarray(
            np.asarray(train_snp_runtime, dtype=np.float32),
            dtype=np.float32,
        )
        test_snp_ml_runtime = np.ascontiguousarray(
            np.asarray(test_snp_runtime, dtype=np.float32),
            dtype=np.float32,
        )
        ml_dense_materialized = True

    def _runtime_inputs_for_method(method_name: str) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        m_key = str(method_name)
        if m_key in _ML_METHOD_MAP:
            _ensure_ml_dense_runtime()
        return (
            train_snp_runtime,
            test_snp_runtime,
            train_snp_add_runtime,
            test_snp_add_runtime,
            train_snp_ml_runtime,
            test_snp_ml_runtime,
        )
    if method_parallel_jobs <= 1:
        out: list[dict[str, typing.Any]] = []
        fold_total_map = {
            m: (int(len(cv_splits)) if cv_splits is not None else 1)
            for m in methods
        }
        animate_method_progress = bool(
            show_method_progress or any(rrblup_progress_enabled_by_method.values())
        )
        if animate_method_progress and rich_progress_available():
            for m in methods:
                progress = build_rich_progress(
                    description_template="{task.fields[label]}",
                    show_percentage=False,
                    show_elapsed=True,
                    show_remaining=False,
                    field_templates_before_elapsed=["{task.fields[counter]}"],
                    finished_text=" ",
                    transient=True,
                )
                assert progress is not None
                m_disp = _method_display(str(m))
                if on_method_start is not None:
                    try:
                        on_method_start(str(m))
                    except Exception:
                        pass
                cv_total = int(max(1, fold_total_map.get(m, 1)))
                has_search_stage = bool((m in _ML_METHOD_MAP) and (not strict_cv))
                has_rrblup_iter_progress = bool(
                    rrblup_progress_enabled_by_method.get(str(m), False)
                )
                # Keep method UX stable: CV uses rich progress bar, fit/predict use spinner lines.
                # Avoid nested rrBLUP sub-progress bars here to prevent terminal live-render contention.
                use_rrblup_subprogress = bool(has_rrblup_iter_progress)
                search_task_id: int | None = None
                search_done = 0
                search_total = 0
                search_total_fixed = False
                cv_task_id: int | None = None
                cv_done = 0
                prefit_task_id: int | None = None
                adam_task_id: int | None = None
                adam_done = 0
                adam_total = 0
                adam_stage = "fit"
                adam_grid_trial_total = 0
                adam_grid_trial_epochs = 0
                adam_grid_completed_epochs = 0
                adam_grid_active_trial = 0
                adam_task_indeterminate = False
                lambda_task_id: int | None = None
                lambda_done = 0
                lambda_total = 0
                lambda_task_indeterminate = False
                cv_line_emitted = False
                stage_lines_emitted = False
                cv_stage_closed = False
                cv_started_at: float | None = None
                progress_live_stopped = False
                fit_status: CliStatus | None = None
                pred_status: CliStatus | None = None
                rr_subprogress_seen = False
                res: dict[str, typing.Any] | None = None
                t0 = time.monotonic()

                def _close_search_task() -> None:
                    nonlocal search_task_id, search_done, search_total
                    if search_task_id is None:
                        return
                    left = max(0, int(search_total) - int(search_done))
                    if left > 0:
                        progress.update(
                            search_task_id,
                            advance=left,
                            label=f"{m_disp} search {int(search_total)}/{int(search_total)}",
                        )
                    try:
                        progress.remove_task(search_task_id)
                    except Exception:
                        pass
                    search_task_id = None

                def _ensure_cv_task() -> None:
                    nonlocal cv_task_id, cv_line_emitted, cv_started_at
                    if not show_method_progress:
                        return
                    if cv_stage_closed:
                        return
                    if cv_task_id is None:
                        cv_started_at = time.monotonic()
                        cv_task_id = progress.add_task(
                            description="",
                            total=cv_total,
                            label="Cross-validation",
                            counter=f"{cv_done}/{cv_total}",
                        )

                def _finish_cv_stage() -> None:
                    nonlocal cv_task_id, cv_done, cv_line_emitted, cv_stage_closed, prefit_task_id
                    if cv_stage_closed:
                        return
                    if prefit_task_id is not None:
                        try:
                            progress.remove_task(prefit_task_id)
                        except Exception:
                            pass
                        prefit_task_id = None
                    if cv_task_id is None:
                        if show_method_progress:
                            cv_elapsed = float(max(0.0, time.monotonic() - (cv_started_at or t0)))
                            progress.console.print(
                                f"[green]{success_symbol()} Cross-validation ...Finished [{cv_elapsed:.1f}s][/green]"
                            )
                            cv_line_emitted = True
                        cv_stage_closed = True
                        return
                    left_cv = max(0, cv_total - int(cv_done))
                    if left_cv > 0:
                        cv_done += left_cv
                        progress.update(
                            cv_task_id,
                            advance=left_cv,
                            label="Cross-validation",
                            counter=f"{cv_done}/{cv_total}",
                        )
                    cv_elapsed = float(max(0.0, time.monotonic() - (cv_started_at or t0)))
                    try:
                        progress.remove_task(cv_task_id)
                    except Exception:
                        pass
                    cv_task_id = None
                    progress.console.print(
                        f"[green]{success_symbol()} Cross-validation ...Finished [{cv_elapsed:.1f}s][/green]"
                    )
                    cv_line_emitted = True
                    cv_stage_closed = True

                def _start_fit_stage() -> None:
                    nonlocal fit_status
                    if fit_status is not None:
                        return
                    fit_status = CliStatus(
                        "Computing Fitting...",
                        enabled=True,
                        use_process=False,
                        show_elapsed=True,
                    )
                    fit_status.__enter__()

                def _finish_fit_stage(elapsed_payload: float | None = None) -> None:
                    nonlocal fit_status, stage_lines_emitted
                    if use_rrblup_subprogress and fit_status is None:
                        elapsed = (
                            float(elapsed_payload)
                            if (elapsed_payload is not None and np.isfinite(float(elapsed_payload)))
                            else float(0.0)
                        )
                        progress.console.print(
                            f"[green]{success_symbol()} Fitting ...Finished [{elapsed:.3f}s][/green]"
                        )
                        stage_lines_emitted = True
                        return
                    if fit_status is None:
                        return
                    fit_status.complete("Fitting ...Finished")
                    fit_status.__exit__(None, None, None)
                    fit_status = None
                    stage_lines_emitted = True

                def _start_predict_stage() -> None:
                    nonlocal pred_status
                    if pred_status is not None:
                        return
                    pred_status = CliStatus(
                        "Computing Predicting...",
                        enabled=True,
                        use_process=False,
                        show_elapsed=True,
                    )
                    pred_status.__enter__()

                def _finish_predict_stage(elapsed_payload: float | None = None) -> None:
                    nonlocal pred_status, stage_lines_emitted
                    if use_rrblup_subprogress and pred_status is None:
                        elapsed = (
                            float(elapsed_payload)
                            if (elapsed_payload is not None and np.isfinite(float(elapsed_payload)))
                            else float(0.0)
                        )
                        progress.console.print(
                            f"[green]{success_symbol()} Predicting ...Finished [{elapsed:.3f}s][/green]"
                        )
                        stage_lines_emitted = True
                        return
                    if pred_status is None:
                        return
                    pred_status.complete("Predicting ...Finished")
                    pred_status.__exit__(None, None, None)
                    pred_status = None
                    stage_lines_emitted = True

                def _fail_fit_stage() -> None:
                    nonlocal fit_status
                    if fit_status is None:
                        return
                    fit_status.fail("Fitting ...Failed")
                    fit_status.__exit__(None, None, None)
                    fit_status = None

                def _fail_predict_stage() -> None:
                    nonlocal pred_status
                    if pred_status is None:
                        return
                    pred_status.fail("Predicting ...Failed")
                    pred_status.__exit__(None, None, None)
                    pred_status = None

                def _rr_phase_title(name: str, payload: dict[str, typing.Any]) -> str:
                    _ = payload
                    return str(name).strip()

                def _rr_lambda_label(payload: dict[str, typing.Any]) -> str:
                    vc_method = _rrblup_vc_method_display(payload.get("vc_method", ""))
                    strategy = str(payload.get("strategy", "")).strip().lower()
                    if vc_method != "":
                        return _rr_phase_title(f"Estimate lambda ({vc_method})", payload)
                    if strategy == "subsample_reml":
                        return _rr_phase_title("Estimate lambda (subsample REML)", payload)
                    return _rr_phase_title("Estimate lambda", payload)

                def _mark_rr_subprogress() -> None:
                    nonlocal fit_status, rr_subprogress_seen
                    rr_subprogress_seen = True
                    if fit_status is not None:
                        fit_status.__exit__(None, None, None)
                        fit_status = None

                def _close_rr_phase(*, fail: bool = False) -> None:
                    _ = fail
                    _close_adam_task()
                    _close_lambda_task()

                def _stage_hook(event: str, payload: dict[str, typing.Any]) -> None:
                    nonlocal progress_live_stopped
                    ev = str(event)
                    if ev == "fit_start":
                        _finish_cv_stage()
                        if (not use_rrblup_subprogress) and (not progress_live_stopped):
                            try:
                                progress.stop()
                            except Exception:
                                pass
                            progress_live_stopped = True
                        _start_fit_stage()
                        return
                    if ev == "fit_end":
                        _finish_fit_stage(payload.get("elapsed"))
                        return
                    if ev == "predict_start":
                        _start_predict_stage()
                        return
                    if ev == "predict_end":
                        _finish_predict_stage(payload.get("elapsed"))
                        return

                def _adam_stage_key(payload: dict[str, typing.Any]) -> str:
                    stage = str(payload.get("stage", "fit")).strip().lower()
                    if stage == "pcg":
                        return "pcg"
                    return "grid" if stage == "grid" else "fit"

                def _adam_label(stage: str, payload: dict[str, typing.Any]) -> str:
                    if str(stage) == "grid":
                        return _rr_phase_title("AdamW search", payload)
                    if str(stage) == "pcg":
                        return _rr_phase_title("PCG iteration", payload)
                    return _rr_phase_title("AdamW iteration", payload)

                def _adam_counter(done: int, total: int) -> str:
                    return f"{int(done)}/{int(total)}"

                def _adam_progress_values(
                    event: str,
                    payload: dict[str, typing.Any],
                ) -> tuple[str, int, int]:
                    nonlocal adam_done, adam_total
                    nonlocal adam_grid_trial_total, adam_grid_trial_epochs
                    nonlocal adam_grid_completed_epochs, adam_grid_active_trial
                    stage = _adam_stage_key(payload)
                    if stage == "pcg":
                        adam_grid_trial_total = 0
                        adam_grid_trial_epochs = 0
                        adam_grid_completed_epochs = 0
                        adam_grid_active_trial = 0
                        total = int(max(1, int(payload.get("total", max(1, int(adam_total))))))
                        if str(event) == "start":
                            done = 0
                        elif str(event) == "end":
                            done = int(
                                max(
                                    0,
                                    int(payload.get("iters", payload.get("iter", adam_done))),
                                )
                            )
                        else:
                            done = int(
                                max(
                                    0,
                                    int(payload.get("iter", payload.get("iters", adam_done))),
                                )
                            )
                        done = int(min(done, total))
                        return stage, done, total
                    if stage != "grid":
                        if str(event) == "start":
                            adam_grid_trial_total = 0
                            adam_grid_trial_epochs = 0
                            adam_grid_completed_epochs = 0
                            adam_grid_active_trial = 0
                        total = int(max(1, int(payload.get("total", max(1, int(adam_total))))))
                        if str(event) == "start":
                            done = 0
                        elif str(event) == "end":
                            done = int(max(0, int(payload.get("epochs_ran", adam_done))))
                        else:
                            done = int(max(0, int(payload.get("epoch", adam_done))))
                        done = int(min(done, total))
                        return stage, done, total

                    trial_total = int(
                        max(1, int(payload.get("trial_total", max(1, int(adam_grid_trial_total or 1)))))
                    )
                    trial_epochs = int(
                        max(1, int(payload.get("total", max(1, int(adam_grid_trial_epochs or 1)))))
                    )
                    trial_id = int(
                        max(1, int(payload.get("trial_id", max(1, int(adam_grid_active_trial or 1)))))
                    )
                    if str(event) == "start":
                        if trial_id <= 1 or trial_id < int(adam_grid_active_trial):
                            adam_grid_completed_epochs = 0
                        adam_grid_active_trial = int(trial_id)
                        trial_done = 0
                    elif str(event) == "end":
                        trial_done = int(max(0, int(payload.get("epochs_ran", payload.get("epoch", 0)))))
                    else:
                        trial_done = int(max(0, int(payload.get("epoch", 0))))
                    trial_done = int(min(trial_done, trial_epochs))

                    total = int(max(1, trial_total * trial_epochs))
                    done = int(max(0, int(adam_grid_completed_epochs) + trial_done))
                    done = int(min(done, total))
                    if str(event) == "end":
                        adam_grid_completed_epochs = int(max(int(adam_grid_completed_epochs), done))
                    adam_grid_trial_total = int(trial_total)
                    adam_grid_trial_epochs = int(trial_epochs)
                    return stage, done, total

                def _ensure_adam_task(
                    payload: dict[str, typing.Any],
                    *,
                    event: str = "start",
                    indeterminate: bool = False,
                ) -> None:
                    nonlocal adam_task_id, adam_done, adam_total, adam_stage
                    nonlocal adam_task_indeterminate
                    stage, target_done, target_total = _adam_progress_values(str(event), payload)
                    adam_stage = str(stage)
                    label = _adam_label(adam_stage, payload)
                    if indeterminate:
                        adam_done = 0
                        adam_total = 0
                        if (adam_task_id is not None) and (not adam_task_indeterminate):
                            try:
                                progress.remove_task(adam_task_id)
                            except Exception:
                                pass
                            adam_task_id = None
                        adam_task_indeterminate = True
                        if adam_task_id is None:
                            adam_task_id = progress.add_task(
                                description="",
                                total=None,
                                label=label,
                                counter="",
                            )
                        else:
                            progress.update(
                                adam_task_id,
                                total=None,
                                completed=0,
                                label=label,
                                counter="",
                            )
                        return
                    if (adam_task_id is not None) and adam_task_indeterminate:
                        try:
                            progress.remove_task(adam_task_id)
                        except Exception:
                            pass
                        adam_task_id = None
                    adam_task_indeterminate = False
                    adam_done = int(target_done)
                    adam_total = int(max(1, int(target_total)))
                    if adam_task_id is None:
                        adam_task_id = progress.add_task(
                            description="",
                            total=adam_total,
                            label=label,
                            counter=_adam_counter(adam_done, adam_total),
                        )
                        if adam_done > 0:
                            progress.update(
                                adam_task_id,
                                total=adam_total,
                                completed=adam_done,
                                label=label,
                                counter=_adam_counter(adam_done, adam_total),
                            )
                    else:
                        progress.update(
                            adam_task_id,
                            total=adam_total,
                            completed=adam_done,
                            label=label,
                            counter=_adam_counter(adam_done, adam_total),
                        )

                def _close_adam_task() -> None:
                    nonlocal adam_task_id, adam_done, adam_total, adam_stage
                    nonlocal adam_grid_trial_total, adam_grid_trial_epochs
                    nonlocal adam_grid_completed_epochs, adam_grid_active_trial
                    nonlocal adam_task_indeterminate
                    if adam_task_id is not None:
                        try:
                            progress.remove_task(adam_task_id)
                        except Exception:
                            pass
                    adam_task_id = None
                    adam_done = 0
                    adam_total = 0
                    adam_stage = "fit"
                    adam_grid_trial_total = 0
                    adam_grid_trial_epochs = 0
                    adam_grid_completed_epochs = 0
                    adam_grid_active_trial = 0
                    adam_task_indeterminate = False

                def _ensure_lambda_task(
                    *,
                    label: str,
                    done: int | None = None,
                    total: int | None = None,
                    indeterminate: bool = False,
                ) -> None:
                    nonlocal lambda_task_id, lambda_done, lambda_total
                    nonlocal lambda_task_indeterminate
                    if indeterminate:
                        lambda_done = 0
                        lambda_total = 0
                        if (lambda_task_id is not None) and (not lambda_task_indeterminate):
                            try:
                                progress.remove_task(lambda_task_id)
                            except Exception:
                                pass
                            lambda_task_id = None
                        lambda_task_indeterminate = True
                        if lambda_task_id is None:
                            lambda_task_id = progress.add_task(
                                description="",
                                total=None,
                                label=label,
                                counter="",
                                hide_bar=True,
                            )
                        else:
                            progress.update(
                                lambda_task_id,
                                total=None,
                                completed=0,
                                label=label,
                                counter="",
                                hide_bar=True,
                            )
                        return
                    if done is None:
                        done = 0
                    if total is None:
                        total = 1
                    if (lambda_task_id is not None) and lambda_task_indeterminate:
                        try:
                            progress.remove_task(lambda_task_id)
                        except Exception:
                            pass
                        lambda_task_id = None
                    lambda_task_indeterminate = False
                    lambda_done = int(max(0, int(done)))
                    lambda_total = int(max(1, int(total)))
                    if lambda_task_id is None:
                        lambda_task_id = progress.add_task(
                            description="",
                            total=lambda_total,
                            label=label,
                            counter="",
                            hide_bar=True,
                        )
                    progress.update(
                        lambda_task_id,
                        total=lambda_total,
                        completed=min(lambda_done, lambda_total),
                        label=label,
                        counter="",
                        hide_bar=True,
                    )

                def _close_lambda_task(*, complete: bool = False) -> None:
                    nonlocal lambda_task_id, lambda_done, lambda_total
                    nonlocal lambda_task_indeterminate
                    if lambda_task_id is not None:
                        if complete and (not lambda_task_indeterminate):
                            progress.update(
                                lambda_task_id,
                                total=max(1, int(lambda_total)),
                                completed=max(0, int(lambda_total)),
                                label=_rr_phase_title("Estimate lambda", {}),
                                counter="",
                                hide_bar=True,
                            )
                        try:
                            progress.remove_task(lambda_task_id)
                        except Exception:
                            pass
                    lambda_task_id = None
                    lambda_done = 0
                    lambda_total = 0
                    lambda_task_indeterminate = False

                def _search_hook(event: str, payload: dict[str, typing.Any]) -> None:
                    nonlocal search_task_id, search_done, search_total, search_total_fixed
                    ev = str(event)
                    if ev == "search_total":
                        count = int(max(0, int(payload.get("count", 0))))
                        if count <= 0:
                            return
                        search_total = int(count)
                        search_total_fixed = True
                        if search_task_id is None:
                            search_task_id = progress.add_task(
                                description="",
                                total=int(max(1, search_total)),
                                label=f"{m_disp} search {search_done}/{search_total}",
                                counter="",
                            )
                        else:
                            progress.update(
                                search_task_id,
                                total=int(max(1, search_total)),
                                label=f"{m_disp} search {search_done}/{search_total}",
                            )
                        return
                    if ev == "search_plan":
                        count = int(max(0, int(payload.get("count", 0))))
                        if count <= 0:
                            return
                        if not search_total_fixed:
                            search_total += count
                        if search_total <= 0:
                            search_total = count
                        if search_task_id is None:
                            search_task_id = progress.add_task(
                                description="",
                                total=int(max(1, search_total)),
                                label=f"{m_disp} search {search_done}/{search_total}",
                                counter="",
                            )
                        else:
                            progress.update(
                                search_task_id,
                                total=int(max(1, search_total)),
                                label=f"{m_disp} search {search_done}/{search_total}",
                            )
                        return
                    if ev == "search_advance":
                        inc = int(max(0, int(payload.get("inc", 0))))
                        if inc <= 0:
                            return
                        if (not search_total_fixed) and (search_done + inc > search_total):
                            search_total = search_done + inc
                        if search_task_id is None:
                            search_task_id = progress.add_task(
                                description="",
                                total=int(max(1, search_total)),
                                label=f"{m_disp} search {search_done}/{search_total}",
                                counter="",
                            )
                        left = max(0, int(search_total) - int(search_done))
                        adv = min(left, inc)
                        if adv > 0:
                            search_done += adv
                            progress.update(
                                search_task_id,
                                total=int(max(1, search_total)),
                                advance=int(adv),
                                label=f"{m_disp} search {search_done}/{search_total}",
                            )

                def _adam_hook(event: str, payload: dict[str, typing.Any]) -> None:
                    nonlocal adam_done, adam_total, adam_task_id, adam_stage
                    nonlocal lambda_done, lambda_total
                    nonlocal adam_task_indeterminate
                    if not has_rrblup_iter_progress:
                        return
                    ev = str(event)
                    _mark_rr_subprogress()
                    if ev == "pcg_lambda_vc_start":
                        _close_adam_task()
                        _close_lambda_task()
                        _ensure_lambda_task(
                            label=_rr_lambda_label(payload),
                            indeterminate=True,
                        )
                        return
                    if ev == "pcg_lambda_vc_end":
                        _close_lambda_task()
                        return
                    if ev == "pcg_lambda_subsample_start":
                        _close_adam_task()
                        _close_lambda_task()
                        total = int(max(1, int(payload.get("total", payload.get("repeats", 1)))))
                        _ensure_lambda_task(
                            label=_rr_lambda_label(payload),
                            done=0,
                            total=total,
                        )
                        return
                    if ev == "pcg_callback_unavailable":
                        _close_adam_task()
                        _ensure_adam_task(
                            payload,
                            event="start",
                            indeterminate=True,
                        )
                        return
                    if ev == "pcg_lambda_subsample_iter":
                        total = int(max(1, int(payload.get("total", max(1, int(lambda_total or 1))))))
                        done = int(max(0, int(payload.get("iter", lambda_done))))
                        _ensure_lambda_task(
                            label=_rr_lambda_label(payload),
                            done=min(done, total),
                            total=total,
                        )
                        return
                    if ev == "pcg_lambda_subsample_end":
                        total = int(max(1, int(payload.get("total", payload.get("repeats", max(1, int(lambda_total or 1)))))))
                        done = int(max(0, int(payload.get("iter", payload.get("ok_repeats", total)))))
                        _ensure_lambda_task(
                            label=_rr_lambda_label(payload),
                            done=min(done, total),
                            total=total,
                        )
                        _close_lambda_task(complete=True)
                        return
                    if ev == "pcg_start":
                        _close_lambda_task()
                        _ensure_adam_task(payload, event="start")
                        return
                    if ev == "pcg_iter":
                        _ensure_adam_task(payload, event="epoch")
                        return
                    if ev == "pcg_end":
                        if not adam_task_indeterminate:
                            _ensure_adam_task(payload, event="end")
                        _close_adam_task()
                        return
                    if ev == "adam_start":
                        _ensure_adam_task(payload, event="start")
                        return
                    if ev == "adam_epoch":
                        if adam_task_id is None:
                            _ensure_adam_task(payload, event="epoch")
                        if adam_task_id is None:
                            return
                        stage, target, total = _adam_progress_values("epoch", payload)
                        adam_stage = str(stage)
                        adam_done = int(target)
                        adam_total = int(max(1, int(total)))
                        progress.update(
                            adam_task_id,
                            total=adam_total,
                            completed=adam_done,
                            label=_adam_label(adam_stage, payload),
                            counter=_adam_counter(adam_done, adam_total),
                        )
                        return
                    if ev == "adam_end":
                        if adam_task_id is None:
                            return
                        stage, target, total = _adam_progress_values("end", payload)
                        adam_stage = str(stage)
                        adam_done = int(target)
                        adam_total = int(max(1, int(total)))
                        progress.update(
                            adam_task_id,
                            total=adam_total,
                            completed=adam_done,
                            label=_adam_label(adam_stage, payload),
                            counter=_adam_counter(adam_done, adam_total),
                        )

                def _cv_hook(method_name: str, inc: int) -> None:
                    nonlocal cv_done, prefit_task_id
                    if str(method_name) != str(m):
                        return
                    _close_lambda_task()
                    _close_search_task()
                    _ensure_cv_task()
                    left = max(0, cv_total - int(cv_done))
                    adv = min(left, int(max(0, int(inc))))
                    if adv > 0 and cv_task_id is not None:
                        cv_done += adv
                        progress.update(
                            cv_task_id,
                            advance=adv,
                            label="Cross-validation",
                            counter=f"{cv_done}/{cv_total}",
                        )
                        if int(cv_done) >= int(cv_total):
                            if prefit_task_id is None:
                                prefit_task_id = progress.add_task(
                                    description="",
                                    total=None,
                                    label="Preparing final fit",
                                    counter="",
                                )

                with progress:
                    if show_method_progress and (not has_search_stage):
                        _ensure_cv_task()
                    try:
                        (
                            train_snp_cur,
                            test_snp_cur,
                            train_snp_add_cur,
                            test_snp_add_cur,
                            train_snp_ml_cur,
                            test_snp_ml_cur,
                        ) = _runtime_inputs_for_method(str(m))
                        res = _run_method_task(
                            m,
                            train_pheno,
                            train_snp_cur,
                            test_snp_cur,
                            train_snp_add_cur,
                            test_snp_add_cur,
                            train_snp_ml_cur,
                            test_snp_ml_cur,
                            pca_dec,
                            cv_splits,
                            model_n_jobs,
                            strict_cv,
                            force_fast,
                            packed_ctx=packed_ctx,
                            train_sample_indices=train_sample_indices,
                            test_sample_indices=test_sample_indices,
                            train_sample_ids=train_sample_ids,
                            progress_queue=None,
                            progress_hook=_cv_hook,
                            search_progress_hook=_search_hook,
                            rrblup_progress_hook=(
                                _adam_hook
                                if (has_rrblup_iter_progress and use_rrblup_subprogress)
                                else None
                            ),
                            limit_predtrain=limit_predtrain,
                            rrblup_solver=rrblup_solver,
                            rrblup_adamw_cfg=rrblup_adamw_cfg,
                            bayes_auto_r2_cache=bayes_auto_r2_cache_shared,
                            bayes_auto_r2_cfg=bayes_auto_r2_cfg,
                            stage_hook=_stage_hook,
                        )
                    except Exception:
                        _close_search_task()
                        _close_adam_task()
                        _close_lambda_task()
                        _close_rr_phase(fail=True)
                        _fail_fit_stage()
                        _fail_predict_stage()
                        if cv_task_id is not None:
                            try:
                                progress.remove_task(cv_task_id)
                            except Exception:
                                pass
                        elapsed = format_elapsed(time.monotonic() - t0)
                        progress.console.print(
                            f"[red]{failure_symbol()} {m_disp} ...Failed [{elapsed}][/red]"
                        )
                        raise
                    _close_search_task()
                    _close_lambda_task()
                    _close_rr_phase()
                    if show_method_progress:
                        _ensure_cv_task()
                        _finish_cv_stage()
                    _close_adam_task()
                    _close_lambda_task()
                    if fit_status is not None:
                        _finish_fit_stage(None)
                    if pred_status is not None:
                        _finish_predict_stage(None)
                    if res is not None and stage_lines_emitted:
                        res["__stage_lines_emitted"] = True
                    if res is not None and cv_line_emitted:
                        res["__cv_line_emitted"] = True
                    if emit_method_summary and res is not None:
                        elapsed = format_elapsed(time.monotonic() - t0)
                        progress.console.print(
                            f"[green]{success_symbol()} {m_disp} ...Finished [{elapsed}][/green]"
                        )
                        _emit_method_details_plain(
                            m,
                            res,
                            line_printer=lambda s: progress.console.print(s, highlight=False),
                        )

                if res is None:
                    continue
                _attach_method_detail_rows(str(m), res)
                out.append(res)
                if on_method_complete is not None:
                    try:
                        on_method_complete(str(m), res)
                    except Exception:
                        pass
            return out
        for m in methods:
            t0 = time.monotonic()
            method_offset = _method_offset(str(m))
            m_disp = _method_display(str(m))
            if on_method_start is not None:
                try:
                    on_method_start(str(m))
                except Exception:
                    pass
            cv_total = int(max(1, fold_total_map.get(m, 1)))
            has_search_stage = bool((m in _ML_METHOD_MAP) and (not strict_cv))
            has_rrblup_iter_progress = bool(
                rrblup_progress_enabled_by_method.get(str(m), False)
            )
            enable_tqdm_progress = bool(
                _HAS_TQDM
                and stdout_is_tty()
                and (show_method_progress or has_rrblup_iter_progress)
            )
            enable_method_spinner = bool(
                emit_method_summary
                and
                (not enable_tqdm_progress)
                and stdout_is_tty()
                and (method_offset <= 0.0)
            )
            search_bar = None
            cv_bar = None
            adam_bar = None
            lambda_bar = None
            search_done = 0
            search_total = 0
            search_total_fixed = False
            cv_done = 0
            adam_done = 0
            adam_total = 0
            adam_stage = "fit"
            adam_grid_trial_total = 0
            adam_grid_trial_epochs = 0
            adam_grid_completed_epochs = 0
            adam_grid_active_trial = 0
            adam_bar_indeterminate = False
            lambda_done = 0
            lambda_total = 0
            lambda_bar_indeterminate = False
            fit_status: CliStatus | None = None
            pred_status: CliStatus | None = None
            stage_lines_emitted = False
            cv_line_emitted = bool(enable_tqdm_progress and show_method_progress)
            rr_subprogress_seen = False

            def _close_search_bar() -> None:
                nonlocal search_bar
                if search_bar is not None:
                    search_bar.close()
                    search_bar = None

            def _close_cv_bar() -> None:
                nonlocal cv_bar
                if cv_bar is not None:
                    cv_bar.close()
                    cv_bar = None

            def _ensure_cv_bar() -> None:
                nonlocal cv_bar
                if not show_method_progress:
                    return
                if (not enable_tqdm_progress) or cv_bar is not None:
                    return
                assert tqdm is not None
                cv_bar = tqdm(
                    total=cv_total,
                    desc="Cross-validation",
                    unit="fold",
                    leave=False,
                    dynamic_ncols=True,
                    position=0,
                    bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}, {rate_fmt}{postfix}]",
                )

            def _adam_stage_key(payload: dict[str, typing.Any]) -> str:
                stage = str(payload.get("stage", "fit")).strip().lower()
                if stage == "pcg":
                    return "pcg"
                return "grid" if stage == "grid" else "fit"

            def _adam_desc(stage: str, payload: dict[str, typing.Any]) -> str:
                if str(stage) == "grid":
                    return _rr_phase_title("AdamW search", payload)
                if str(stage) == "pcg":
                    return _rr_phase_title("PCG iteration", payload)
                return _rr_phase_title("AdamW iteration", payload)

            def _adam_progress_values(
                event: str,
                payload: dict[str, typing.Any],
            ) -> tuple[str, int, int]:
                nonlocal adam_done, adam_total
                nonlocal adam_grid_trial_total, adam_grid_trial_epochs
                nonlocal adam_grid_completed_epochs, adam_grid_active_trial
                stage = _adam_stage_key(payload)
                if stage == "pcg":
                    adam_grid_trial_total = 0
                    adam_grid_trial_epochs = 0
                    adam_grid_completed_epochs = 0
                    adam_grid_active_trial = 0
                    total = int(max(1, int(payload.get("total", max(1, int(adam_total))))))
                    if str(event) == "start":
                        done = 0
                    elif str(event) == "end":
                        done = int(
                            max(
                                0,
                                int(payload.get("iters", payload.get("iter", adam_done))),
                            )
                        )
                    else:
                        done = int(
                            max(
                                0,
                                int(payload.get("iter", payload.get("iters", adam_done))),
                            )
                        )
                    done = int(min(done, total))
                    return stage, done, total
                if stage != "grid":
                    if str(event) == "start":
                        adam_grid_trial_total = 0
                        adam_grid_trial_epochs = 0
                        adam_grid_completed_epochs = 0
                        adam_grid_active_trial = 0
                    total = int(max(1, int(payload.get("total", max(1, int(adam_total))))))
                    if str(event) == "start":
                        done = 0
                    elif str(event) == "end":
                        done = int(max(0, int(payload.get("epochs_ran", adam_done))))
                    else:
                        done = int(max(0, int(payload.get("epoch", adam_done))))
                    done = int(min(done, total))
                    return stage, done, total

                trial_total = int(
                    max(1, int(payload.get("trial_total", max(1, int(adam_grid_trial_total or 1)))))
                )
                trial_epochs = int(
                    max(1, int(payload.get("total", max(1, int(adam_grid_trial_epochs or 1)))))
                )
                trial_id = int(
                    max(1, int(payload.get("trial_id", max(1, int(adam_grid_active_trial or 1)))))
                )
                if str(event) == "start":
                    if trial_id <= 1 or trial_id < int(adam_grid_active_trial):
                        adam_grid_completed_epochs = 0
                    adam_grid_active_trial = int(trial_id)
                    trial_done = 0
                elif str(event) == "end":
                    trial_done = int(max(0, int(payload.get("epochs_ran", payload.get("epoch", 0)))))
                else:
                    trial_done = int(max(0, int(payload.get("epoch", 0))))
                trial_done = int(min(trial_done, trial_epochs))

                total = int(max(1, trial_total * trial_epochs))
                done = int(max(0, int(adam_grid_completed_epochs) + trial_done))
                done = int(min(done, total))
                if str(event) == "end":
                    adam_grid_completed_epochs = int(max(int(adam_grid_completed_epochs), done))
                adam_grid_trial_total = int(trial_total)
                adam_grid_trial_epochs = int(trial_epochs)
                return stage, done, total

            def _ensure_adam_bar(
                payload: dict[str, typing.Any],
                *,
                event: str = "start",
                indeterminate: bool = False,
            ) -> None:
                nonlocal adam_bar, adam_done, adam_total, adam_stage
                nonlocal adam_bar_indeterminate
                if not enable_tqdm_progress:
                    return
                stage, target_done, target_total = _adam_progress_values(str(event), payload)
                adam_stage = str(stage)
                desc = _adam_desc(adam_stage, payload)
                if indeterminate:
                    adam_done = 0
                    adam_total = 0
                    if (adam_bar is not None) and (not adam_bar_indeterminate):
                        adam_bar.close()
                        adam_bar = None
                    adam_bar_indeterminate = True
                    if adam_bar is None:
                        assert tqdm is not None
                        adam_bar = tqdm(
                            total=None,
                            desc=desc,
                            leave=False,
                            dynamic_ncols=True,
                            position=(1 if show_method_progress else 0),
                            bar_format="{desc} [{elapsed}]",
                        )
                    else:
                        adam_bar.set_description_str(desc, refresh=False)
                    adam_bar.refresh()
                    return
                if (adam_bar is not None) and adam_bar_indeterminate:
                    adam_bar.close()
                    adam_bar = None
                adam_bar_indeterminate = False
                adam_done = int(target_done)
                adam_total = int(max(1, int(target_total)))
                if adam_bar is None:
                    assert tqdm is not None
                    unit = "iter" if str(adam_stage) == "pcg" else "ep"
                    adam_bar = tqdm(
                        total=adam_total,
                        desc=desc,
                        unit=unit,
                        leave=False,
                        dynamic_ncols=True,
                        position=(1 if show_method_progress else 0),
                        bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}, {rate_fmt}{postfix}]",
                    )
                adam_bar.total = int(max(1, adam_total))
                adam_bar.n = int(min(max(0, adam_done), max(1, adam_total)))
                adam_bar.set_description_str(desc, refresh=False)
                adam_bar.set_postfix_str("", refresh=False)
                adam_bar.refresh()

            def _close_adam_bar() -> None:
                nonlocal adam_bar, adam_done, adam_total, adam_stage
                nonlocal adam_grid_trial_total, adam_grid_trial_epochs
                nonlocal adam_grid_completed_epochs, adam_grid_active_trial
                nonlocal adam_bar_indeterminate
                if adam_bar is not None:
                    adam_bar.close()
                    adam_bar = None
                adam_done = 0
                adam_total = 0
                adam_stage = "fit"
                adam_grid_trial_total = 0
                adam_grid_trial_epochs = 0
                adam_grid_completed_epochs = 0
                adam_grid_active_trial = 0
                adam_bar_indeterminate = False

            def _ensure_lambda_bar(
                desc: str,
                *,
                event: str = "start",
                payload: dict[str, typing.Any] | None = None,
                indeterminate: bool = False,
            ) -> None:
                nonlocal lambda_bar, lambda_done, lambda_total
                nonlocal lambda_bar_indeterminate
                if not enable_tqdm_progress:
                    return
                _ = event
                _ = payload
                _ = indeterminate
                was_indeterminate = bool(lambda_bar_indeterminate)
                lambda_done = 0
                lambda_total = 0
                if (lambda_bar is not None) and (not was_indeterminate):
                    lambda_bar.close()
                    lambda_bar = None
                lambda_bar_indeterminate = True
                if lambda_bar is None:
                    assert tqdm is not None
                    lambda_bar = tqdm(
                        total=None,
                        desc=desc,
                        leave=False,
                        dynamic_ncols=True,
                        position=(1 if show_method_progress else 0),
                        bar_format="{desc} [{elapsed}]",
                    )
                else:
                    lambda_bar.set_description_str(desc, refresh=False)
                lambda_bar.refresh()

            def _close_lambda_bar() -> None:
                nonlocal lambda_bar, lambda_done, lambda_total
                nonlocal lambda_bar_indeterminate
                if lambda_bar is not None:
                    lambda_bar.close()
                    lambda_bar = None
                lambda_done = 0
                lambda_total = 0
                lambda_bar_indeterminate = False

            def _start_fit_status() -> None:
                nonlocal fit_status
                if fit_status is not None:
                    return
                if not stdout_is_tty():
                    return
                fit_status = CliStatus(
                    "Computing Fitting...",
                    enabled=True,
                    use_process=True,
                    show_elapsed=False,
                )
                fit_status.__enter__()

            def _complete_fit_status(elapsed: float | None = None) -> None:
                nonlocal fit_status, stage_lines_emitted
                if fit_status is None and rr_subprogress_seen:
                    msg = "Fitting ...Finished"
                    if elapsed is not None and np.isfinite(float(elapsed)):
                        msg = f"Fitting ...Finished [{float(elapsed):.1f}s]"
                    print_success(msg, force_color=True)
                    stage_lines_emitted = True
                    return
                if fit_status is None:
                    return
                msg = "Fitting ...Finished"
                if elapsed is not None and np.isfinite(float(elapsed)):
                    msg = f"Fitting ...Finished [{float(elapsed):.1f}s]"
                fit_status.complete(msg)
                fit_status.__exit__(None, None, None)
                fit_status = None
                stage_lines_emitted = True

            def _fail_fit_status() -> None:
                nonlocal fit_status
                if fit_status is None:
                    return
                fit_status.fail("Fitting ...Failed")
                fit_status.__exit__(None, None, None)
                fit_status = None

            def _start_predict_status() -> None:
                nonlocal pred_status
                if pred_status is not None:
                    return
                if not stdout_is_tty():
                    return
                pred_status = CliStatus(
                    "Computing Predicting...",
                    enabled=True,
                    use_process=True,
                    show_elapsed=False,
                )
                pred_status.__enter__()

            def _complete_predict_status(elapsed: float | None = None) -> None:
                nonlocal pred_status, stage_lines_emitted
                if pred_status is None:
                    return
                msg = "Predicting ...Finished"
                if elapsed is not None and np.isfinite(float(elapsed)):
                    msg = f"Predicting ...Finished [{float(elapsed):.1f}s]"
                pred_status.complete(msg)
                pred_status.__exit__(None, None, None)
                pred_status = None
                stage_lines_emitted = True

            def _fail_predict_status() -> None:
                nonlocal pred_status
                if pred_status is None:
                    return
                pred_status.fail("Predicting ...Failed")
                pred_status.__exit__(None, None, None)
                pred_status = None

            def _rr_phase_title(name: str, payload: dict[str, typing.Any]) -> str:
                _ = payload
                return str(name).strip()

            def _rr_lambda_desc(payload: dict[str, typing.Any]) -> str:
                vc_method = _rrblup_vc_method_display(payload.get("vc_method", ""))
                strategy = str(payload.get("strategy", "")).strip().lower()
                if vc_method != "":
                    return _rr_phase_title(f"Estimate lambda ({vc_method})", payload)
                if strategy == "subsample_reml":
                    return _rr_phase_title("Estimate lambda (subsample REML)", payload)
                return _rr_phase_title("Estimate lambda", payload)

            def _mark_rr_subprogress() -> None:
                nonlocal fit_status, rr_subprogress_seen
                rr_subprogress_seen = True
                if fit_status is not None:
                    fit_status.__exit__(None, None, None)
                    fit_status = None

            def _close_rr_phase(*, fail: bool = False) -> None:
                _ = fail
                _close_adam_bar()
                _close_lambda_bar()

            def _stage_hook(event: str, payload: dict[str, typing.Any]) -> None:
                ev = str(event)
                if ev == "fit_start":
                    _start_fit_status()
                    return
                if ev == "fit_end":
                    _complete_fit_status(float(payload.get("elapsed", np.nan)))
                    return
                if ev == "predict_start":
                    _start_predict_status()
                    return
                if ev == "predict_end":
                    _complete_predict_status(float(payload.get("elapsed", np.nan)))
                    return

            def _ensure_search_bar() -> None:
                nonlocal search_bar, search_total
                if (not enable_tqdm_progress) or search_bar is not None:
                    return
                assert tqdm is not None
                search_bar = tqdm(
                    total=int(max(1, search_total)),
                    desc=f"{m_disp} search {search_done}/{max(1, int(search_total))}",
                    unit="cfg",
                    leave=False,
                    dynamic_ncols=True,
                    bar_format="{desc}: |{bar}| "
                    "[{elapsed}, {rate_fmt}{postfix}]",
                )

            def _search_hook(event: str, payload: dict[str, typing.Any]) -> None:
                nonlocal search_done, search_total, search_total_fixed
                if not enable_tqdm_progress:
                    return
                ev = str(event)
                stage = str(payload.get("stage", "")).strip()
                if ev == "search_total":
                    count = int(max(0, int(payload.get("count", 0))))
                    if count <= 0:
                        return
                    search_total = int(count)
                    search_total_fixed = True
                    _ensure_search_bar()
                    if search_bar is not None:
                        search_bar.total = int(max(1, search_total))
                        search_bar.set_description_str(
                            f"{m_disp} search {search_done}/{max(1, int(search_total))}",
                            refresh=False,
                        )
                        search_bar.refresh()
                    return
                if ev == "search_plan":
                    count = int(max(0, int(payload.get("count", 0))))
                    if count <= 0:
                        return
                    if not search_total_fixed:
                        search_total += count
                    if search_total <= 0:
                        search_total = count
                    _ensure_search_bar()
                    if search_bar is not None:
                        search_bar.total = int(max(1, search_total))
                        search_bar.set_description_str(
                            f"{m_disp} search {search_done}/{max(1, int(search_total))}",
                            refresh=False,
                        )
                        if stage != "":
                            search_bar.set_postfix_str(stage, refresh=False)
                        search_bar.refresh()
                    return
                if ev == "search_advance":
                    inc = int(max(0, int(payload.get("inc", 0))))
                    if inc <= 0:
                        return
                    if (not search_total_fixed) and (search_done + inc > search_total):
                        search_total = search_done + inc
                    _ensure_search_bar()
                    if search_bar is None:
                        return
                    left = max(0, int(search_total) - int(search_done))
                    adv = min(left, inc)
                    if adv > 0:
                        search_done += adv
                        search_bar.total = int(max(1, search_total))
                        search_bar.set_description_str(
                            f"{m_disp} search {search_done}/{max(1, int(search_total))}",
                            refresh=False,
                        )
                        if stage != "":
                            search_bar.set_postfix_str(stage, refresh=False)
                        search_bar.update(int(adv))

            def _adam_hook(event: str, payload: dict[str, typing.Any]) -> None:
                nonlocal adam_done, adam_total, adam_stage
                nonlocal adam_bar_indeterminate
                if not has_rrblup_iter_progress:
                    return
                ev = str(event)
                _mark_rr_subprogress()
                if ev == "pcg_lambda_vc_start":
                    _close_adam_bar()
                    _close_lambda_bar()
                    _ensure_lambda_bar(
                        _rr_lambda_desc(payload),
                        indeterminate=True,
                    )
                    return
                if ev == "pcg_lambda_vc_end":
                    _close_lambda_bar()
                    return
                if ev == "pcg_lambda_subsample_start":
                    _close_adam_bar()
                    _close_lambda_bar()
                    _ensure_lambda_bar(
                        _rr_lambda_desc(payload),
                        event="start",
                        payload=payload,
                    )
                    return
                if ev == "pcg_lambda_subsample_iter":
                    _ensure_lambda_bar(
                        _rr_lambda_desc(payload),
                        event="iter",
                        payload=payload,
                    )
                    return
                if ev == "pcg_lambda_subsample_end":
                    _ensure_lambda_bar(
                        _rr_lambda_desc(payload),
                        event="end",
                        payload=payload,
                    )
                    _close_lambda_bar()
                    return
                if ev == "pcg_callback_unavailable":
                    _close_adam_bar()
                    _ensure_adam_bar(
                        payload,
                        event="start",
                        indeterminate=True,
                    )
                    return
                if ev == "pcg_start":
                    _close_lambda_bar()
                    _ensure_adam_bar(payload, event="start")
                    return
                if ev == "pcg_iter":
                    _ensure_adam_bar(payload, event="epoch")
                    return
                if ev == "pcg_end":
                    if not adam_bar_indeterminate:
                        _ensure_adam_bar(payload, event="end")
                    _close_adam_bar()
                    return
                if ev == "adam_start":
                    _ensure_adam_bar(payload, event="start")
                    return
                if ev == "adam_epoch":
                    if adam_bar is None:
                        _ensure_adam_bar(payload, event="epoch")
                    if adam_bar is None:
                        return
                    stage, target, total = _adam_progress_values("epoch", payload)
                    adam_stage = str(stage)
                    adam_done = int(target)
                    adam_total = int(max(1, int(total)))
                    adam_bar.total = adam_total
                    adam_bar.n = int(min(max(0, adam_done), adam_total))
                    adam_bar.set_description_str(
                        _adam_desc(adam_stage, payload),
                        refresh=False,
                    )
                    adam_bar.set_postfix_str("", refresh=False)
                    adam_bar.refresh()
                    return
                if ev == "adam_end":
                    if adam_bar is None:
                        return
                    stage, target, total = _adam_progress_values("end", payload)
                    adam_stage = str(stage)
                    adam_done = int(target)
                    adam_total = int(max(1, int(total)))
                    adam_bar.total = adam_total
                    adam_bar.n = int(min(max(0, adam_done), adam_total))
                    adam_bar.set_description_str(
                        _adam_desc(adam_stage, payload),
                        refresh=False,
                    )
                    adam_bar.set_postfix_str("", refresh=False)
                    adam_bar.refresh()

            def _cv_hook(method_name: str, inc: int) -> None:
                nonlocal cv_done
                if str(method_name) != str(m):
                    return
                _close_lambda_bar()
                _close_search_bar()
                _ensure_cv_bar()
                if cv_bar is None:
                    return
                left = max(0, cv_total - int(cv_done))
                adv = min(left, int(max(0, int(inc))))
                if adv > 0:
                    cv_done += adv
                    cv_bar.update(adv)
                    cv_bar.set_description_str(
                        "Cross-validation",
                        refresh=True,
                    )

            if enable_tqdm_progress and show_method_progress and (not has_search_stage):
                _ensure_cv_bar()

            method_status = (
                CliStatus(f"Loading {m_disp}...", enabled=True, use_process=True)
                if enable_method_spinner
                else None
            )
            if method_status is not None:
                method_status.__enter__()
            try:
                try:
                    (
                        train_snp_cur,
                        test_snp_cur,
                        train_snp_add_cur,
                        test_snp_add_cur,
                        train_snp_ml_cur,
                        test_snp_ml_cur,
                    ) = _runtime_inputs_for_method(str(m))
                    res = _run_method_task(
                        m,
                        train_pheno,
                        train_snp_cur,
                        test_snp_cur,
                        train_snp_add_cur,
                        test_snp_add_cur,
                        train_snp_ml_cur,
                        test_snp_ml_cur,
                        pca_dec,
                        cv_splits,
                        model_n_jobs,
                        strict_cv,
                        force_fast,
                        packed_ctx=packed_ctx,
                        train_sample_indices=train_sample_indices,
                        test_sample_indices=test_sample_indices,
                        train_sample_ids=train_sample_ids,
                        progress_queue=None,
                        progress_hook=(_cv_hook if enable_tqdm_progress else None),
                        search_progress_hook=(_search_hook if enable_tqdm_progress else None),
                        rrblup_progress_hook=(
                            _adam_hook
                            if (has_rrblup_iter_progress and enable_tqdm_progress)
                            else None
                        ),
                        limit_predtrain=limit_predtrain,
                        rrblup_solver=rrblup_solver,
                        rrblup_adamw_cfg=rrblup_adamw_cfg,
                        bayes_auto_r2_cache=bayes_auto_r2_cache_shared,
                        bayes_auto_r2_cfg=bayes_auto_r2_cfg,
                        stage_hook=_stage_hook,
                    )
                except Exception:
                    _close_search_bar()
                    _close_adam_bar()
                    _close_lambda_bar()
                    _close_rr_phase(fail=True)
                    _close_cv_bar()
                    _fail_fit_status()
                    _fail_predict_status()
                    elapsed = format_elapsed((time.monotonic() - t0) + method_offset)
                    if method_status is not None:
                        method_status.fail(f"{m_disp} ...Failed")
                    else:
                        print_failure(f"{m_disp} ...Failed [{elapsed}]", force_color=True)
                    raise

                _close_search_bar()
                _close_lambda_bar()
                _close_rr_phase()
                if enable_tqdm_progress and show_method_progress:
                    _ensure_cv_bar()
                    if cv_bar is not None:
                        left_cv = max(0, cv_total - int(cv_done))
                        if left_cv > 0:
                            cv_done += left_cv
                            cv_bar.update(left_cv)
                        cv_bar.set_description_str(
                            "Cross-validation",
                            refresh=True,
                        )
                _close_adam_bar()
                _close_lambda_bar()
                _close_cv_bar()
                _complete_fit_status(None)
                _complete_predict_status(None)

                if stage_lines_emitted:
                    res["__stage_lines_emitted"] = True
                if cv_line_emitted:
                    res["__cv_line_emitted"] = True
                _attach_method_detail_rows(str(m), res)
                out.append(res)
                if on_method_complete is not None:
                    try:
                        on_method_complete(str(m), res)
                    except Exception:
                        pass
                elapsed = format_elapsed((time.monotonic() - t0) + method_offset)
                if emit_method_summary:
                    if method_status is not None:
                        method_status.complete(f"{m_disp} ...Finished")
                    else:
                        print_success(f"{m_disp} ...Finished [{elapsed}]", force_color=True)
                    _emit_method_details_plain(
                        m,
                        res,
                        line_printer=lambda s: print(s, flush=True),
                    )
            finally:
                if method_status is not None:
                    method_status.__exit__(None, None, None)
        return out

    results_by_method: "OrderedDict[str, dict[str, typing.Any]]" = OrderedDict()
    fold_total_map = {
        m: (int(len(cv_splits)) if cv_splits is not None else 1)
        for m in methods
    }
    task_total_map = {
        m: (fold_total_map[m] + 1) if show_method_progress else fold_total_map[m]
        for m in methods
    }

    if not show_method_progress:
        method_start_ts = {m: time.monotonic() for m in methods}
        with cf.ProcessPoolExecutor(max_workers=method_parallel_jobs) as ex:
            future_map = {
                ex.submit(
                    _run_method_task,
                    m,
                    train_pheno,
                    train_snp,
                    test_snp,
                    train_snp_add,
                    test_snp_add,
                    train_snp_ml,
                    test_snp_ml,
                    pca_dec,
                    cv_splits,
                    model_n_jobs,
                    strict_cv,
                    force_fast,
                    packed_ctx,
                    train_sample_indices,
                    test_sample_indices,
                    None,
                    None,
                    None,
                    limit_predtrain,
                    rrblup_solver,
                    rrblup_adamw_cfg,
                ): m
                for m in methods
            }
            for fut in cf.as_completed(list(future_map.keys())):
                method = future_map[fut]
                try:
                    res_obj = fut.result()
                except Exception:
                    elapsed = format_elapsed(
                        (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                        + _method_offset(str(method))
                    )
                    method_disp = _method_display(str(method))
                    print_failure(
                        f"{method_disp} ...Failed [{elapsed}]",
                        force_color=True,
                    )
                    raise
                _attach_method_detail_rows(str(method), typing.cast(dict[str, typing.Any], res_obj))
                results_by_method[method] = typing.cast(dict[str, typing.Any], res_obj)
                if on_method_complete is not None:
                    try:
                        on_method_complete(str(method), results_by_method[method])
                    except Exception:
                        pass
                elapsed = format_elapsed(
                    (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                    + _method_offset(str(method))
                )
                method_disp = _method_display(str(method))
                if emit_method_summary:
                    print_success(
                        f"{method_disp} ...Finished [{elapsed}]",
                        force_color=True,
                    )
                    _emit_method_details_plain(
                        method,
                        results_by_method[method],
                        line_printer=lambda s: print(s, flush=True),
                    )
    elif bool(show_method_progress and rich_progress_available()):
        progress = build_rich_progress(
            description_template="{task.fields[method]}",
            field_templates=["{task.fields[suffix]}"],
            show_percentage=False,
            show_elapsed=True,
            show_remaining=False,
            finished_text=" ",
            transient=True,
        )
        assert progress is not None
        with progress:
            def _rich_print_success(method_name: str, elapsed_text: str) -> None:
                progress.console.print(
                    f"[green]{success_symbol()} {method_name} ...Finished [{elapsed_text}][/green]"
                )

            def _rich_print_failure(method_name: str, elapsed_text: str) -> None:
                progress.console.print(
                    f"[red]{failure_symbol()} {method_name} ...Failed [{elapsed_text}][/red]"
                )

            task_map: dict[str, int] = {}
            method_start_ts: dict[str, float] = {}
            for method in methods:
                method_disp = _method_display(str(method))
                task_map[method] = progress.add_task(
                    description="",
                    total=task_total_map[method],
                    method=method_disp,
                    fold_done=0,
                    fold_total=fold_total_map[method],
                    suffix="",
                )
                method_start_ts[method] = time.monotonic()
            fold_advanced = {m: 0 for m in methods}
            manager = mp.Manager()
            prog_q = manager.Queue() if cv_splits is not None else None
            try:
                with cf.ProcessPoolExecutor(max_workers=method_parallel_jobs) as ex:
                    future_map = {
                        ex.submit(
                            _run_method_task,
                            m,
                            train_pheno,
                            train_snp,
                            test_snp,
                            train_snp_add,
                            test_snp_add,
                            train_snp_ml,
                            test_snp_ml,
                            pca_dec,
                            cv_splits,
                            model_n_jobs,
                            strict_cv,
                            force_fast,
                            packed_ctx,
                            train_sample_indices,
                            test_sample_indices,
                            prog_q,
                            None,
                            None,
                            limit_predtrain,
                            rrblup_solver,
                            rrblup_adamw_cfg,
                        ): m
                        for m in methods
                    }
                    while len(future_map) > 0:
                        if prog_q is not None:
                            while True:
                                try:
                                    m_name, inc = prog_q.get_nowait()
                                except queue_mod.Empty:
                                    break
                                tid = task_map.get(str(m_name))
                                if tid is not None:
                                    key = str(m_name)
                                    done_n = int(fold_advanced.get(key, 0))
                                    total_n = int(fold_total_map.get(key, 0))
                                    left = max(0, total_n - done_n)
                                    adv = min(left, int(max(0, int(inc))))
                                    if adv > 0:
                                        new_done = done_n + adv
                                        progress.update(
                                            tid,
                                            advance=adv,
                                            fold_done=new_done,
                                        )
                                        fold_advanced[key] = new_done
                        done, _ = cf.wait(
                            list(future_map.keys()),
                            timeout=0.1,
                            return_when=cf.FIRST_COMPLETED,
                        )
                        for fut in done:
                            method = future_map.pop(fut)
                            res = fut.result()
                            _attach_method_detail_rows(str(method), res)
                            results_by_method[method] = res
                            if on_method_complete is not None:
                                try:
                                    on_method_complete(str(method), res)
                                except Exception:
                                    pass
                            tid = task_map.get(method)
                            if tid is not None:
                                try:
                                    progress.remove_task(tid)
                                except Exception:
                                    pass
                                task_map.pop(method, None)
                            elapsed = format_elapsed(
                                (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                                + _method_offset(str(method))
                            )
                            if emit_method_summary:
                                _rich_print_success(_method_display(str(method)), elapsed)
                                _emit_method_details_plain(
                                    method,
                                    res,
                                    line_printer=lambda s: progress.console.print(s, highlight=False),
                                )
            except Exception:
                for method, tid in task_map.items():
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    elapsed = format_elapsed(
                        (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                        + _method_offset(str(method))
                    )
                    _rich_print_failure(_method_display(str(method)), elapsed)
                raise
            finally:
                try:
                    manager.shutdown()
                except Exception:
                    pass
    elif bool(show_method_progress and _HAS_TQDM and stdout_is_tty()):
        method_total = int(len(methods))
        methods_done = 0
        pbar = tqdm(
            total=method_total,
            desc=f"Methods {methods_done}/{method_total}",
            unit="method",
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc}: |{bar}| "
                       "[{elapsed}, {rate_fmt}{postfix}]",
        )
        method_start_ts = {m: time.monotonic() for m in methods}
        manager = mp.Manager()
        prog_q = manager.Queue() if cv_splits is not None else None
        try:
            with cf.ProcessPoolExecutor(max_workers=method_parallel_jobs) as ex:
                future_map = {
                    ex.submit(
                        _run_method_task,
                        m,
                        train_pheno,
                        train_snp,
                        test_snp,
                        train_snp_add,
                        test_snp_add,
                        train_snp_ml,
                        test_snp_ml,
                        pca_dec,
                        cv_splits,
                        model_n_jobs,
                        strict_cv,
                        force_fast,
                        packed_ctx,
                        train_sample_indices,
                        test_sample_indices,
                        prog_q,
                        None,
                        None,
                        limit_predtrain,
                        rrblup_solver,
                        rrblup_adamw_cfg,
                    ): m
                    for m in methods
                }
                while len(future_map) > 0:
                    done, _ = cf.wait(
                        list(future_map.keys()),
                        timeout=0.1,
                        return_when=cf.FIRST_COMPLETED,
                    )
                    for fut in done:
                        method = future_map.pop(fut)
                        res = fut.result()
                        _attach_method_detail_rows(str(method), res)
                        results_by_method[method] = res
                        if on_method_complete is not None:
                            try:
                                on_method_complete(str(method), res)
                            except Exception:
                                pass
                        methods_done = min(method_total, methods_done + 1)
                        pbar.update(1)
                        pbar.set_description_str(
                            f"Methods {methods_done}/{method_total}",
                            refresh=True,
                        )
                        pbar.set_postfix(method=_method_display(str(method)))
                        elapsed = format_elapsed(
                            (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                            + _method_offset(str(method))
                        )
                        if emit_method_summary:
                            print_success(
                                f"{_method_display(str(method))} ...Finished [{elapsed}]",
                                force_color=True,
                            )
                            _emit_method_details_plain(
                                method,
                                res,
                                line_printer=lambda s: print(s, flush=True),
                            )
        finally:
            pbar.close()
            try:
                manager.shutdown()
            except Exception:
                pass
    else:
        with cf.ProcessPoolExecutor(max_workers=method_parallel_jobs) as ex:
            future_map = {
                ex.submit(
                    _run_method_task,
                    m,
                    train_pheno,
                    train_snp,
                    test_snp,
                    train_snp_add,
                    test_snp_add,
                    train_snp_ml,
                    test_snp_ml,
                    pca_dec,
                    cv_splits,
                    model_n_jobs,
                    strict_cv,
                    force_fast,
                    packed_ctx,
                    train_sample_indices,
                    test_sample_indices,
                    None,
                    None,
                    None,
                    limit_predtrain,
                    rrblup_solver,
                    rrblup_adamw_cfg,
                ): m
                for m in methods
            }
            for fut in cf.as_completed(list(future_map.keys())):
                method = future_map[fut]
                res = fut.result()
                _attach_method_detail_rows(str(method), res)
                results_by_method[method] = res

    return [results_by_method[m] for m in methods if m in results_by_method]


def _load_genotype_with_rust_gfreader(
    genotype_path: str,
    *,
    maf: float,
    missing_rate: float,
    chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load full genotype matrix with Rust gfreader chunk decoder.

    Returns
    -------
    sample_ids : np.ndarray[str], shape (n_samples,)
    geno : np.ndarray[float32], shape (n_snps_filtered, n_samples)
    """
    sample_ids, _ = inspect_genotype_file(genotype_path)
    chunks = load_genotype_chunks(
        genotype_path,
        chunk_size=int(chunk_size),
        maf=float(maf),
        missing_rate=float(missing_rate),
        impute=True,
    )

    blocks: list[np.ndarray] = []
    for geno_chunk, _ in chunks:
        arr = np.asarray(geno_chunk, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        blocks.append(arr)

    if len(blocks) == 0:
        raise ValueError(
            "No SNPs left after Rust-side filtering. "
            "Please relax --maf/--geno thresholds."
        )

    geno = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    return np.asarray(sample_ids, dtype=str), geno


def _load_plink_packed_for_lmm(
    genotype_prefix: str,
    *,
    maf: float,
    missing_rate: float,
    filter_mode: str = "compact",
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """
    Load PLINK BED in packed format for low-memory LMM/GBLUP.

    Returns
    -------
    sample_ids : np.ndarray[str], shape (n_samples,)
    packed_ctx : dict
        Keys: packed(uint8), missing_rate(float32), maf(float32), n_samples(int)
    """
    sample_ids_arr, packed_ctx = prepare_packed_ctx_from_plink(
        str(genotype_prefix),
        maf=float(maf),
        missing_rate=float(missing_rate),
        snps_only=False,
        filter_mode=str(filter_mode),
    )
    return sample_ids_arr, packed_ctx


def _decode_packed_ctx_to_dense(
    packed_ctx: dict[str, typing.Any],
) -> np.ndarray:
    """
    Decode packed BED payload to dense float32 genotype matrix (m, n).
    """
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_decode_rows_f32")):
        raise RuntimeError(
            "Rust packed BED decode helper is unavailable. Rebuild/install JanusX extension."
        )
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_row_flip_mask")):
        raise RuntimeError(
            "Rust packed row-flip kernel is unavailable. Rebuild/install JanusX extension."
        )

    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    maf_full = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    n_samples = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    active_row_idx = _packed_ctx_active_row_idx(packed_ctx)
    if _packed_ctx_is_lazy_full(packed_ctx):
        maf = np.ascontiguousarray(maf_full[active_row_idx], dtype=np.float32)
    else:
        maf = maf_full
    if int(active_row_idx.shape[0]) != int(maf.shape[0]):
        raise ValueError("Packed payload shape mismatch between active rows and maf.")

    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is None:
        row_flip_full = np.asarray(
            _jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        )
        row_flip_full = np.ascontiguousarray(row_flip_full.reshape(-1), dtype=np.bool_)
        packed_ctx["row_flip"] = row_flip_full
        if _packed_ctx_is_lazy_full(packed_ctx):
            row_flip = np.ascontiguousarray(row_flip_full[active_row_idx], dtype=np.bool_)
        else:
            row_flip = row_flip_full
    else:
        row_flip_full = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if _packed_ctx_is_lazy_full(packed_ctx):
            if int(row_flip_full.shape[0]) != int(packed.shape[0]):
                raise ValueError("Packed payload mismatch: row_flip length does not match packed rows.")
            row_flip = np.ascontiguousarray(row_flip_full[active_row_idx], dtype=np.bool_)
        else:
            row_flip = row_flip_full
            if int(row_flip.shape[0]) != int(packed.shape[0]):
                raise ValueError("Packed payload mismatch: row_flip length does not match packed rows.")

    ridx = (
        active_row_idx
        if _packed_ctx_is_lazy_full(packed_ctx)
        else np.ascontiguousarray(np.arange(int(packed.shape[0]), dtype=np.int64))
    )
    decoded = _jxrs.bed_packed_decode_rows_f32(
        packed,
        int(n_samples),
        ridx,
        row_flip,
        maf,
        None,
    )
    dense = np.ascontiguousarray(np.asarray(decoded, dtype=np.float32), dtype=np.float32)
    expected_rows = int(active_row_idx.shape[0])
    if dense.ndim != 2 or dense.shape != (expected_rows, int(n_samples)):
        raise ValueError(
            "Packed decode shape mismatch: "
            f"expected ({expected_rows},{int(n_samples)}), got {dense.shape}"
        )
    return dense


def _read_plink_bim_chrom_pos(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    bim_path = str(prefix)
    if not bim_path.lower().endswith(".bim"):
        bim_path = f"{bim_path}.bim"
    if not os.path.isfile(bim_path):
        raise ValueError(f"Cannot find BIM file for PLINK prefix: {prefix}")

    chrom_codes: list[int] = []
    positions: list[int] = []
    code_map: dict[str, int] = {}
    next_code = 0
    with open(bim_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            s = str(line).strip()
            if s == "" or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Invalid BIM row at {bim_path}:{line_no} (expected >= 6 columns)."
                )
            chrom = str(parts[0]).strip()
            if chrom not in code_map:
                code_map[chrom] = int(next_code)
                next_code += 1
            chrom_codes.append(int(code_map[chrom]))
            try:
                pos = int(parts[3])
            except Exception:
                try:
                    pos = int(float(parts[3]))
                except Exception as ex:
                    raise ValueError(
                        f"Invalid BIM position at {bim_path}:{line_no}: {parts[3]!r}"
                    ) from ex
            positions.append(int(pos))
    if len(chrom_codes) == 0:
        raise ValueError(f"No SNP rows found in BIM: {bim_path}")
    return (
        np.ascontiguousarray(np.asarray(chrom_codes, dtype=np.int32)),
        np.ascontiguousarray(np.asarray(positions, dtype=np.int64)),
    )


def _read_plink_bim_marker_ids(prefix: str) -> list[str]:
    bim_path = str(prefix)
    if not bim_path.lower().endswith(".bim"):
        bim_path = f"{bim_path}.bim"
    if not os.path.isfile(bim_path):
        raise ValueError(f"Cannot find BIM file for PLINK prefix: {prefix}")
    ids: list[str] = []
    with open(bim_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            s = str(line).strip()
            if s == "" or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Invalid BIM row at {bim_path}:{line_no} (expected >= 6 columns)."
                )
            ids.append(str(parts[1]).strip())
    if len(ids) == 0:
        raise ValueError(f"No SNP rows found in BIM: {bim_path}")
    return ids


def _resolve_gs_ld_prune_backend() -> str:
    raw = str(
        os.getenv(
            "JX_GS_LDPRUNE_BACKEND",
            os.getenv("JX_LD_PRUNE_BACKEND", "rust"),
        )
    ).strip().lower()
    if raw in {"", "auto", "default", "rust"}:
        return "rust"
    if raw in {"plink", "plink-external", "external-plink"}:
        return "plink"
    return "rust"


def _format_gs_ld_prune_window_for_plink(spec: _GsLdPruneSpec) -> str:
    if spec.window_variants is not None:
        return str(int(spec.window_variants))
    if spec.window_bp is None:
        raise ValueError("Invalid LD prune window for PLINK backend.")
    kb = float(int(spec.window_bp)) / 1000.0
    tok = f"{kb:.6f}".rstrip("0").rstrip(".")
    if tok == "":
        tok = "0"
    return f"{tok}kb"


def _run_gs_external_command(cmd: list[str]) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = str(proc.stdout or "").strip()
    if int(proc.returncode) != 0:
        tail = "\n".join(out.splitlines()[-20:]) if out else "(no output)"
        raise RuntimeError(
            "External command failed "
            f"(exit={int(proc.returncode)}): {' '.join(cmd)}\n{tail}"
        )
    return out


def _apply_packed_ld_prune_for_gs_external_plink(
    *,
    packed_ctx: dict[str, typing.Any],
    genotype_prefix: str,
    spec: _GsLdPruneSpec,
    n_jobs: int,
) -> dict[str, typing.Any]:
    plink_bin_raw = str(os.getenv("JX_PLINK_BIN", "plink")).strip() or "plink"
    if os.path.sep in plink_bin_raw or plink_bin_raw.startswith("."):
        if os.path.isfile(plink_bin_raw) and os.access(plink_bin_raw, os.X_OK):
            plink_bin = plink_bin_raw
        else:
            plink_bin = ""
    else:
        found = shutil.which(plink_bin_raw)
        plink_bin = str(found) if found else ""
    if plink_bin == "":
        raise RuntimeError(
            "PLINK backend requested for GS LD prune but binary was not found. "
            "Please install PLINK 1.9 and/or set JX_PLINK_BIN=/path/to/plink."
        )

    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    miss = np.ascontiguousarray(
        np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1)
    )
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    m = int(packed.shape[0])
    if int(maf.shape[0]) != m or int(miss.shape[0]) != m:
        raise ValueError("Packed payload shape mismatch among packed/maf/missing_rate.")

    ids_full = _read_plink_bim_marker_ids(str(genotype_prefix))
    n_full = int(len(ids_full))
    site_keep_raw = packed_ctx.get("site_keep", None)
    if site_keep_raw is not None:
        site_keep = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if int(site_keep.shape[0]) != n_full:
            raise ValueError(
                "Packed payload mismatch: site_keep length does not match BIM row count."
            )
        filtered_ids = [ids_full[i] for i in np.flatnonzero(site_keep).astype(int).tolist()]
    else:
        site_keep = None
        filtered_ids = list(ids_full)

    if int(len(filtered_ids)) != m:
        raise ValueError(
            f"Packed/BIM row mismatch for LD prune: packed={m}, bim_aligned={len(filtered_ids)}"
        )
    if m == 0:
        raise ValueError("No SNPs available for LD prune after preprocessing.")

    tmp_prefix = (
        f"{str(genotype_prefix)}.jxgsprune_tmp_{int(os.getpid())}_{int(time.time() * 1000)}"
    )
    extract_path = f"{tmp_prefix}.extract.in"
    prune_in = f"{tmp_prefix}.prune.in"
    keep_tmp = _env_truthy("JX_KEEP_PLINK_TMP", "0")
    tmp_paths = [
        extract_path,
        f"{tmp_prefix}.prune.in",
        f"{tmp_prefix}.prune.out",
        f"{tmp_prefix}.log",
        f"{tmp_prefix}.nosex",
    ]

    try:
        with open(extract_path, "w", encoding="utf-8") as fw:
            for mid in filtered_ids:
                fw.write(f"{mid}\n")

        _run_gs_external_command(
            [
                str(plink_bin),
                "--bfile",
                str(genotype_prefix),
                "--extract",
                str(extract_path),
                "--indep-pairwise",
                str(_format_gs_ld_prune_window_for_plink(spec)),
                str(int(spec.step_variants)),
                str(f"{float(spec.r2_threshold):.12g}"),
                "--threads",
                str(max(1, int(n_jobs))),
                "--out",
                str(tmp_prefix),
            ]
        )
        if not os.path.isfile(prune_in):
            raise RuntimeError(f"PLINK did not produce prune list: {prune_in}")
        keep_id_set: set[str] = set()
        with open(prune_in, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                s = str(line).strip()
                if s != "":
                    keep_id_set.add(s)
        keep = np.asarray([mid in keep_id_set for mid in filtered_ids], dtype=np.bool_)
    finally:
        if not keep_tmp:
            for p in tmp_paths:
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                except Exception:
                    pass

    if int(keep.shape[0]) != m:
        raise ValueError(
            f"LD prune keep-mask mismatch: got {keep.shape[0]}, expected {m}"
        )
    if not np.any(keep):
        raise ValueError(
            "All SNPs were filtered out by LD prune; loosen --ldprune thresholds."
        )
    if np.all(keep):
        return packed_ctx

    packed_ctx["packed"] = np.ascontiguousarray(packed[keep], dtype=np.uint8)
    packed_ctx["maf"] = np.ascontiguousarray(maf[keep], dtype=np.float32)
    packed_ctx["missing_rate"] = np.ascontiguousarray(miss[keep], dtype=np.float32)
    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is not None:
        row_flip = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if int(row_flip.shape[0]) != m:
            raise ValueError("Packed payload mismatch: row_flip length does not match packed rows.")
        packed_ctx["row_flip"] = np.ascontiguousarray(row_flip[keep], dtype=np.bool_)

    if site_keep is not None:
        idx_kept = np.flatnonzero(site_keep).astype(np.int64, copy=False)
        if int(idx_kept.shape[0]) != m:
            raise ValueError("Packed payload mismatch: site_keep true-count does not match packed rows.")
        site_keep_new = np.zeros_like(site_keep, dtype=np.bool_)
        site_keep_new[idx_kept[keep]] = True
        packed_ctx["site_keep"] = np.ascontiguousarray(site_keep_new, dtype=np.bool_)

    return packed_ctx


def _apply_packed_ld_prune_for_gs(
    *,
    packed_ctx: dict[str, typing.Any],
    genotype_prefix: str,
    spec: _GsLdPruneSpec,
    n_jobs: int,
) -> dict[str, typing.Any]:
    backend = _resolve_gs_ld_prune_backend()
    if backend == "plink":
        return _apply_packed_ld_prune_for_gs_external_plink(
            packed_ctx=packed_ctx,
            genotype_prefix=genotype_prefix,
            spec=spec,
            n_jobs=n_jobs,
        )
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_ld_prune_maf_priority")):
        raise RuntimeError(
            "Rust packed LD prune kernel is unavailable. Rebuild/install JanusX extension."
        )
    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    miss = np.ascontiguousarray(
        np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1)
    )
    n_samples = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    if int(packed.shape[0]) != int(maf.shape[0]) or int(packed.shape[0]) != int(miss.shape[0]):
        raise ValueError("Packed payload shape mismatch among packed/maf/missing_rate.")

    chrom_codes_full, positions_full = _read_plink_bim_chrom_pos(str(genotype_prefix))
    n_full = int(chrom_codes_full.shape[0])
    if int(positions_full.shape[0]) != n_full:
        raise ValueError("BIM chrom/position arrays length mismatch.")
    site_keep_raw = packed_ctx.get("site_keep", None)
    if site_keep_raw is not None:
        site_keep = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if int(site_keep.shape[0]) != n_full:
            raise ValueError(
                "Packed payload mismatch: site_keep length does not match BIM row count."
            )
        chrom_codes = np.ascontiguousarray(chrom_codes_full[site_keep], dtype=np.int32)
        positions = np.ascontiguousarray(positions_full[site_keep], dtype=np.int64)
    else:
        chrom_codes = chrom_codes_full
        positions = positions_full

    m = int(packed.shape[0])
    if int(chrom_codes.shape[0]) != m or int(positions.shape[0]) != m:
        raise ValueError(
            f"Packed/BIM row mismatch for LD prune: packed={m}, bim_aligned={chrom_codes.shape[0]}"
        )

    keep = np.asarray(
        _jxrs.bed_packed_ld_prune_maf_priority(
            packed=packed,
            n_samples=int(n_samples),
            chrom_codes=chrom_codes,
            positions=positions,
            window_bp=(
                int(spec.window_bp)
                if spec.window_bp is not None
                else None
            ),
            window_variants=(
                int(spec.window_variants)
                if spec.window_variants is not None
                else None
            ),
            step_variants=int(spec.step_variants),
            r2_threshold=float(spec.r2_threshold),
            threads=max(1, int(n_jobs)),
        ),
        dtype=np.bool_,
    ).reshape(-1)
    if int(keep.shape[0]) != m:
        raise ValueError(
            f"LD prune keep-mask mismatch: got {keep.shape[0]}, expected {m}"
        )
    if not np.any(keep):
        raise ValueError(
            "All SNPs were filtered out by LD prune; loosen --ldprune thresholds."
        )
    if np.all(keep):
        return packed_ctx

    packed_ctx["packed"] = np.ascontiguousarray(packed[keep], dtype=np.uint8)
    packed_ctx["maf"] = np.ascontiguousarray(maf[keep], dtype=np.float32)
    packed_ctx["missing_rate"] = np.ascontiguousarray(miss[keep], dtype=np.float32)
    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is not None:
        row_flip = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if int(row_flip.shape[0]) != m:
            raise ValueError("Packed payload mismatch: row_flip length does not match packed rows.")
        packed_ctx["row_flip"] = np.ascontiguousarray(row_flip[keep], dtype=np.bool_)

    if site_keep_raw is not None:
        site_keep = np.ascontiguousarray(
            np.asarray(site_keep_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        idx_kept = np.flatnonzero(site_keep).astype(np.int64, copy=False)
        if int(idx_kept.shape[0]) != m:
            raise ValueError("Packed payload mismatch: site_keep true-count does not match packed rows.")
        site_keep_new = np.zeros_like(site_keep, dtype=np.bool_)
        site_keep_new[idx_kept[keep]] = True
        packed_ctx["site_keep"] = np.ascontiguousarray(site_keep_new, dtype=np.bool_)

    return packed_ctx


def _hash_packed_for_gs(
    *,
    packed_ctx: dict[str, typing.Any],
    hash_dim: int,
    hash_seed: int,
    standardize: bool,
    n_jobs: int,
    sample_indices: np.ndarray | None = None,
    progress_callback: typing.Callable[[int, int], None] | None = None,
    progress_every: int = 0,
) -> tuple[np.ndarray, float, int]:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_signed_hash_f32")):
        raise RuntimeError(
            "Rust signed-hash kernel is unavailable. Rebuild/install JanusX extension."
        )
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_row_flip_mask")):
        raise RuntimeError(
            "Rust packed row-flip kernel is unavailable. Rebuild/install JanusX extension."
        )
    if int(hash_dim) <= 0:
        raise ValueError("hash_dim must be > 0.")

    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    miss = np.ascontiguousarray(
        np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1)
    )
    n_samples = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    if int(packed.shape[0]) != int(maf.shape[0]) or int(packed.shape[0]) != int(miss.shape[0]):
        raise ValueError("Packed payload shape mismatch among packed/maf/missing_rate.")

    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is None:
        row_flip = np.asarray(
            _jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        )
        row_flip = np.ascontiguousarray(row_flip.reshape(-1), dtype=np.bool_)
        packed_ctx["row_flip"] = row_flip
    else:
        row_flip = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if int(row_flip.shape[0]) != int(packed.shape[0]):
            raise ValueError("Packed payload mismatch: row_flip length does not match packed rows.")

    sidx_arg = None
    if sample_indices is not None:
        sidx_arg = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1))
        if np.any(sidx_arg < 0) or np.any(sidx_arg >= int(n_samples)):
            raise ValueError("sample_indices contain out-of-range values for packed hashing.")

    try:
        z_raw, scale_raw, kept_raw = _jxrs.bed_packed_signed_hash_f32(
            packed=packed,
            n_samples=int(n_samples),
            row_flip=row_flip,
            row_maf=maf,
            n_buckets=int(hash_dim),
            seed=int(hash_seed),
            sample_indices=sidx_arg,
            row_missing=miss,
            min_maf=0.0,
            max_missing=1.0,
            standardize=bool(standardize),
            threads=max(1, int(n_jobs)),
            progress_callback=progress_callback,
            progress_every=max(0, int(progress_every)),
        )
    except TypeError:
        # Backward compatibility with old extension builds that don't expose
        # progress_callback/progress_every yet.
        z_raw, scale_raw, kept_raw = _jxrs.bed_packed_signed_hash_f32(
            packed=packed,
            n_samples=int(n_samples),
            row_flip=row_flip,
            row_maf=maf,
            n_buckets=int(hash_dim),
            seed=int(hash_seed),
            sample_indices=sidx_arg,
            row_missing=miss,
            min_maf=0.0,
            max_missing=1.0,
            standardize=bool(standardize),
            threads=max(1, int(n_jobs)),
        )
    z = np.ascontiguousarray(np.asarray(z_raw, dtype=np.float32))
    expected_n = int(n_samples if sidx_arg is None else sidx_arg.shape[0])
    if z.ndim != 2 or int(z.shape[0]) != int(hash_dim) or int(z.shape[1]) != expected_n:
        raise ValueError(
            f"Signed hash output shape mismatch: expected ({hash_dim},{expected_n}), got {z.shape}"
        )
    return z, float(scale_raw), int(kept_raw)


def _hash_packed_kernels_for_gblup(
    *,
    packed_ctx: dict[str, typing.Any],
    hash_dim: int,
    hash_seed: int,
    standardize: bool,
    n_jobs: int,
    train_sample_indices: np.ndarray,
    test_sample_indices: np.ndarray | None = None,
    progress_callback: typing.Callable[[int, int], None] | None = None,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_signed_hash_kernels_f64")):
        raise RuntimeError(
            "Rust signed-hash kernel->GRM path is unavailable. Rebuild/install JanusX extension."
        )
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_row_flip_mask")):
        raise RuntimeError(
            "Rust packed row-flip kernel is unavailable. Rebuild/install JanusX extension."
        )
    if int(hash_dim) <= 0:
        raise ValueError("hash_dim must be > 0.")

    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    miss = np.ascontiguousarray(
        np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1)
    )
    n_samples = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    if int(packed.shape[0]) != int(maf.shape[0]) or int(packed.shape[0]) != int(miss.shape[0]):
        raise ValueError("Packed payload shape mismatch among packed/maf/missing_rate.")

    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is None:
        row_flip = np.asarray(
            _jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        )
        row_flip = np.ascontiguousarray(row_flip.reshape(-1), dtype=np.bool_)
        packed_ctx["row_flip"] = row_flip
    else:
        row_flip = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if int(row_flip.shape[0]) != int(packed.shape[0]):
            raise ValueError("Packed payload mismatch: row_flip length does not match packed rows.")

    train_idx = np.ascontiguousarray(np.asarray(train_sample_indices, dtype=np.int64).reshape(-1))
    if train_idx.size == 0:
        raise ValueError("train_sample_indices must not be empty.")
    if np.any(train_idx < 0) or np.any(train_idx >= int(n_samples)):
        raise ValueError("train_sample_indices contain out-of-range values for packed hashing.")
    test_idx: np.ndarray | None = None
    if test_sample_indices is not None:
        t = np.ascontiguousarray(np.asarray(test_sample_indices, dtype=np.int64).reshape(-1))
        if np.any(t < 0) or np.any(t >= int(n_samples)):
            raise ValueError("test_sample_indices contain out-of-range values for packed hashing.")
        test_idx = t

    try:
        train_grm_raw, cross_raw, kept_raw = _jxrs.bed_packed_signed_hash_kernels_f64(
            packed=packed,
            n_samples=int(n_samples),
            row_flip=row_flip,
            row_maf=maf,
            n_buckets=int(hash_dim),
            train_sample_indices=train_idx,
            test_sample_indices=test_idx,
            seed=int(hash_seed),
            row_missing=miss,
            min_maf=0.0,
            max_missing=1.0,
            standardize=bool(standardize),
            threads=max(1, int(n_jobs)),
            progress_callback=progress_callback,
            progress_every=max(0, int(progress_every)),
        )
    except TypeError:
        train_grm_raw, cross_raw, kept_raw = _jxrs.bed_packed_signed_hash_kernels_f64(
            packed=packed,
            n_samples=int(n_samples),
            row_flip=row_flip,
            row_maf=maf,
            n_buckets=int(hash_dim),
            train_sample_indices=train_idx,
            test_sample_indices=test_idx,
            seed=int(hash_seed),
            row_missing=miss,
            min_maf=0.0,
            max_missing=1.0,
            standardize=bool(standardize),
            threads=max(1, int(n_jobs)),
        )
    train_grm = np.ascontiguousarray(np.asarray(train_grm_raw, dtype=np.float64))
    cross = np.ascontiguousarray(np.asarray(cross_raw, dtype=np.float64))

    n_train = int(train_idx.shape[0])
    n_test = 0 if test_idx is None else int(test_idx.shape[0])
    if train_grm.ndim != 2 or train_grm.shape != (n_train, n_train):
        raise ValueError(
            f"Signed hash kernel output mismatch: expected train GRM ({n_train},{n_train}), got {train_grm.shape}"
        )
    if cross.ndim != 2 or cross.shape != (n_test, n_train):
        raise ValueError(
            f"Signed hash kernel output mismatch: expected cross ({n_test},{n_train}), got {cross.shape}"
        )
    return train_grm, cross, int(kept_raw)


def _hash_packed_ztz_stats_for_gblup(
    *,
    packed_ctx: dict[str, typing.Any],
    hash_dim: int,
    hash_seed: int,
    standardize: bool,
    n_jobs: int,
    train_sample_indices: np.ndarray,
    y_train: np.ndarray,
    progress_callback: typing.Callable[[int, int], None] | None = None,
    progress_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_signed_hash_ztz_stats_f64")):
        raise RuntimeError(
            "Rust signed-hash ZTZ stats kernel is unavailable. Rebuild/install JanusX extension."
        )
    if _jxrs is None or (not hasattr(_jxrs, "bed_packed_row_flip_mask")):
        raise RuntimeError(
            "Rust packed row-flip kernel is unavailable. Rebuild/install JanusX extension."
        )
    if int(hash_dim) <= 0:
        raise ValueError("hash_dim must be > 0.")

    packed = np.ascontiguousarray(np.asarray(packed_ctx["packed"], dtype=np.uint8))
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    miss = np.ascontiguousarray(
        np.asarray(packed_ctx["missing_rate"], dtype=np.float32).reshape(-1)
    )
    n_samples = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    if int(packed.shape[0]) != int(maf.shape[0]) or int(packed.shape[0]) != int(miss.shape[0]):
        raise ValueError("Packed payload shape mismatch among packed/maf/missing_rate.")

    row_flip_raw = packed_ctx.get("row_flip", None)
    if row_flip_raw is None:
        row_flip = np.asarray(
            _jxrs.bed_packed_row_flip_mask(packed, int(n_samples)),
            dtype=np.bool_,
        )
        row_flip = np.ascontiguousarray(row_flip.reshape(-1), dtype=np.bool_)
        packed_ctx["row_flip"] = row_flip
    else:
        row_flip = np.ascontiguousarray(
            np.asarray(row_flip_raw, dtype=np.bool_).reshape(-1), dtype=np.bool_
        )
        if int(row_flip.shape[0]) != int(packed.shape[0]):
            raise ValueError("Packed payload mismatch: row_flip length does not match packed rows.")

    train_idx = np.ascontiguousarray(np.asarray(train_sample_indices, dtype=np.int64).reshape(-1))
    if train_idx.size == 0:
        raise ValueError("train_sample_indices must not be empty.")
    if np.any(train_idx < 0) or np.any(train_idx >= int(n_samples)):
        raise ValueError("train_sample_indices contain out-of-range values for packed hashing.")
    y_vec = np.ascontiguousarray(np.asarray(y_train, dtype=np.float64).reshape(-1))
    if int(y_vec.shape[0]) != int(train_idx.shape[0]):
        raise ValueError(
            f"y_train length mismatch: got {y_vec.shape[0]}, expected {train_idx.shape[0]}"
        )

    try:
        gram_raw, row_sum_raw, row_sq_sum_raw, zy_raw, scale_raw, kept_raw = (
            _jxrs.bed_packed_signed_hash_ztz_stats_f64(
                packed=packed,
                n_samples=int(n_samples),
                row_flip=row_flip,
                row_maf=maf,
                n_buckets=int(hash_dim),
                train_sample_indices=train_idx,
                y_train=y_vec,
                seed=int(hash_seed),
                row_missing=miss,
                min_maf=0.0,
                max_missing=1.0,
                standardize=bool(standardize),
                threads=max(1, int(n_jobs)),
                progress_callback=progress_callback,
                progress_every=max(0, int(progress_every)),
            )
        )
    except TypeError:
        gram_raw, row_sum_raw, row_sq_sum_raw, zy_raw, scale_raw, kept_raw = (
            _jxrs.bed_packed_signed_hash_ztz_stats_f64(
                packed=packed,
                n_samples=int(n_samples),
                row_flip=row_flip,
                row_maf=maf,
                n_buckets=int(hash_dim),
                train_sample_indices=train_idx,
                y_train=y_vec,
                seed=int(hash_seed),
                row_missing=miss,
                min_maf=0.0,
                max_missing=1.0,
                standardize=bool(standardize),
                threads=max(1, int(n_jobs)),
            )
        )
    gram = np.ascontiguousarray(np.asarray(gram_raw, dtype=np.float64))
    row_sum = np.ascontiguousarray(np.asarray(row_sum_raw, dtype=np.float64).reshape(-1))
    row_sq_sum = np.ascontiguousarray(np.asarray(row_sq_sum_raw, dtype=np.float64).reshape(-1))
    zy = np.ascontiguousarray(np.asarray(zy_raw, dtype=np.float64).reshape(-1))

    if gram.ndim != 2 or gram.shape != (int(hash_dim), int(hash_dim)):
        raise ValueError(
            f"Signed hash ZTZ output mismatch: expected gram ({hash_dim},{hash_dim}), got {gram.shape}"
        )
    for name, arr in (("row_sum", row_sum), ("row_sq_sum", row_sq_sum), ("zy", zy)):
        if arr.ndim != 1 or int(arr.shape[0]) != int(hash_dim):
            raise ValueError(
                f"Signed hash ZTZ output mismatch: expected {name} length {hash_dim}, got {arr.shape}"
            )
    return gram, row_sum, row_sq_sum, zy, float(scale_raw), int(kept_raw)


def _hash_progress_step(total_hint: int) -> int:
    t = int(max(1, int(total_hint)))
    return max(1, t // 200)


def _upper_offdiag_values(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be square for off-diagonal extraction, got {arr.shape}")
    n = int(arr.shape[0])
    if n <= 1:
        return np.zeros((0,), dtype=np.float64)
    tri = np.triu_indices(n, k=1)
    return np.ascontiguousarray(arr[tri], dtype=np.float64)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    if int(a.size) == 0 or int(b.size) == 0 or int(a.size) != int(b.size):
        return float("nan")
    ax = a - float(np.mean(a))
    bx = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(ax * ax) * np.sum(bx * bx)))
    if (not np.isfinite(denom)) or (denom <= 0.0):
        return float("nan")
    return float(np.sum(ax * bx) / denom)


def _snapshot_packed_ctx_for_qc(
    packed_ctx: dict[str, typing.Any] | None,
) -> dict[str, typing.Any] | None:
    """
    Keep a lightweight snapshot of packed payload references for pre/post
    preprocessing QC comparison. Arrays are not deep-copied.
    """
    if packed_ctx is None:
        return None
    snap: dict[str, typing.Any] = {
        "packed": packed_ctx["packed"],
        "maf": packed_ctx["maf"],
        "missing_rate": packed_ctx["missing_rate"],
        "n_samples": packed_ctx["n_samples"],
    }
    if "row_flip" in packed_ctx:
        snap["row_flip"] = packed_ctx["row_flip"]
    if "site_keep" in packed_ctx:
        snap["site_keep"] = packed_ctx["site_keep"]
    return snap


def _hash_kernel_sanity_check(
    *,
    packed_ctx: dict[str, typing.Any],
    sample_ids: np.ndarray,
    pheno: pd.DataFrame,
    hash_dim: int | None,
    hash_seed: int,
    standardize: bool,
    n_jobs: int,
    baseline_packed_ctx: dict[str, typing.Any] | None = None,
    sample_n: int = _HASH_KERNEL_QC_SAMPLE_N,
) -> dict[str, typing.Any]:
    sid = np.asarray(sample_ids, dtype=str).reshape(-1)
    if int(sid.size) == 0:
        raise ValueError("Hash QC requires non-empty sample IDs.")
    if int(pheno.shape[1]) <= 0:
        raise ValueError("Hash QC requires at least one phenotype column.")

    trait_name = str(pheno.columns[0])
    sid_index = pd.Index(sid, dtype=str)
    y_col = pd.to_numeric(
        pheno[str(trait_name)].reindex(sid_index),
        errors="coerce",
    )
    y_all = np.asarray(y_col.to_numpy(dtype=np.float64, copy=False), dtype=np.float64)
    valid = np.isfinite(y_all)
    pool = np.flatnonzero(valid).astype(np.int64, copy=False)
    if int(pool.size) < int(_HASH_KERNEL_QC_MIN_N):
        raise ValueError(
            f"Hash QC requires at least {_HASH_KERNEL_QC_MIN_N} non-missing samples "
            f"in phenotype column '{trait_name}', got {int(pool.size)}."
        )

    take = int(max(_HASH_KERNEL_QC_MIN_N, min(int(sample_n), int(pool.size))))
    if int(pool.size) > take:
        rng = np.random.default_rng(int(hash_seed) + 20260428)
        pick = np.asarray(rng.choice(pool, size=take, replace=False), dtype=np.int64)
        pick.sort()
    else:
        pick = np.asarray(pool, dtype=np.int64)
    sub_idx = np.ascontiguousarray(pick.reshape(-1), dtype=np.int64)
    y_sub = np.ascontiguousarray(y_all[sub_idx].reshape(-1, 1), dtype=np.float64)

    full_ctx = baseline_packed_ctx if baseline_packed_ctx is not None else packed_ctx
    full_snp = int(
        np.asarray(
            typing.cast(dict[str, typing.Any], full_ctx)["packed"]
        ).shape[0]
    )
    proc_snp = int(
        np.asarray(
            typing.cast(dict[str, typing.Any], packed_ctx)["packed"]
        ).shape[0]
    )

    k_full = _build_gblup_cv_grm_once(
        train_snp=None,
        packed_ctx=full_ctx,
        train_sample_indices=sub_idx,
        n_jobs=max(1, int(n_jobs)),
    )
    if k_full is None:
        raise RuntimeError("Hash QC failed to build full GRM from packed genotype.")
    k_full = np.ascontiguousarray(0.5 * (k_full + k_full.T), dtype=np.float64)

    mode = "ldprune"
    if hash_dim is not None:
        mode = "hash"
        if int(proc_snp) < int(full_snp):
            mode = "ldprune+hash"
        k_proc, _cross, kept_snp = _hash_packed_kernels_for_gblup(
            packed_ctx=packed_ctx,
            hash_dim=int(hash_dim),
            hash_seed=int(hash_seed),
            standardize=bool(standardize),
            n_jobs=max(1, int(n_jobs)),
            train_sample_indices=sub_idx,
            test_sample_indices=np.zeros((0,), dtype=np.int64),
        )
        k_proc = np.ascontiguousarray(0.5 * (k_proc + k_proc.T), dtype=np.float64)
    else:
        k_proc_tmp = _build_gblup_cv_grm_once(
            train_snp=None,
            packed_ctx=packed_ctx,
            train_sample_indices=sub_idx,
            n_jobs=max(1, int(n_jobs)),
        )
        if k_proc_tmp is None:
            raise RuntimeError("Kernel QC failed to build post-prune GRM from packed genotype.")
        k_proc = np.ascontiguousarray(0.5 * (k_proc_tmp + k_proc_tmp.T), dtype=np.float64)
        kept_snp = int(proc_snp)

    off_full = _upper_offdiag_values(k_full)
    off_proc = _upper_offdiag_values(k_proc)
    offdiag_r = _safe_pearson(off_full, off_proc)

    fit_full = _fit_gblup_reml_from_grm(y_sub, k_full)
    fit_proc = _fit_gblup_reml_from_grm(y_sub, k_proc)
    pve_full = float(fit_full.get("pve", float("nan")))
    pve_proc = float(fit_proc.get("pve", float("nan")))
    pve_delta = (
        abs(pve_proc - pve_full)
        if np.isfinite(pve_proc) and np.isfinite(pve_full)
        else float("nan")
    )
    return {
        "mode": str(mode),
        "trait": trait_name,
        "n": int(sub_idx.shape[0]),
        "full_snp": int(full_snp),
        "proc_snp": int(proc_snp),
        "kept_snp": int(kept_snp),
        "offdiag_r": float(offdiag_r),
        "pve_full": float(pve_full),
        "pve_proc": float(pve_proc),
        "pve_delta": float(pve_delta),
    }


def _report_hash_kernel_sanity_check(
    *,
    packed_ctx: dict[str, typing.Any] | None,
    sample_ids: np.ndarray | None,
    pheno: pd.DataFrame,
    hash_dim: int | None,
    hash_seed: int,
    hash_standardize: bool,
    n_jobs: int,
    use_spinner: bool,
    logger: logging.Logger,
    baseline_packed_ctx: dict[str, typing.Any] | None = None,
    debug_mode: bool = False,
) -> None:
    if packed_ctx is None or sample_ids is None:
        return
    t0 = time.monotonic()
    try:
        def _estimate_details(qc_row: dict[str, typing.Any]) -> list[str]:
            full_snp = int(qc_row["full_snp"])
            proc_snp = int(qc_row["proc_snp"])
            if hash_dim is None:
                m_part = f"m={full_snp}->{proc_snp}"
            else:
                m_part = f"m={full_snp}->{proc_snp}, k={int(hash_dim)}"
            return [
                f"n={int(qc_row['n'])}",
                m_part,
                f"offdiag_r={float(qc_row['offdiag_r']):.3f}",
                f"ΔPVE={float(qc_row['pve_delta']):.3f}",
            ]

        if use_spinner:
            qc = _run_indeterminate_task_with_progress_bar(
                desc="Estimate",
                enabled=True,
                runner=lambda: _hash_kernel_sanity_check(
                    packed_ctx=packed_ctx,
                    sample_ids=np.asarray(sample_ids, dtype=str),
                    pheno=pheno,
                    hash_dim=(None if hash_dim is None else int(hash_dim)),
                    hash_seed=int(hash_seed),
                    standardize=bool(hash_standardize),
                    n_jobs=max(1, int(n_jobs)),
                    baseline_packed_ctx=baseline_packed_ctx,
                    sample_n=int(_HASH_KERNEL_QC_SAMPLE_N),
                ),
            )
            msg = _compact_done_message("Estimate", _estimate_details(qc))
            print_success(
                f"{msg} [{format_elapsed(time.monotonic() - t0)}]",
                force_color=True,
            )
        else:
            with CliStatus("Running estimate...", enabled=use_spinner) as task:
                qc = _hash_kernel_sanity_check(
                    packed_ctx=packed_ctx,
                    sample_ids=np.asarray(sample_ids, dtype=str),
                    pheno=pheno,
                    hash_dim=(None if hash_dim is None else int(hash_dim)),
                    hash_seed=int(hash_seed),
                    standardize=bool(hash_standardize),
                    n_jobs=max(1, int(n_jobs)),
                    baseline_packed_ctx=baseline_packed_ctx,
                    sample_n=int(_HASH_KERNEL_QC_SAMPLE_N),
                )
                task.complete(_compact_done_message("Estimate", _estimate_details(qc)))
        if bool(debug_mode):
            logger.info(
                "Estimate details: "
                f"mode={qc['mode']}, "
                f"trait={qc['trait']}, "
                f"n={int(qc['n'])}, "
                f"full_snp={int(qc['full_snp'])}, "
                f"proc_snp={int(qc['proc_snp'])}, "
                f"offdiag_r={float(qc['offdiag_r']):.6f}, "
                f"pve_full={float(qc['pve_full']):.6f}, "
                f"pve_proc={float(qc['pve_proc']):.6f}, "
                f"delta_pve={float(qc['pve_delta']):.6f}, "
                f"hash_kept_snp={int(qc['kept_snp'])}"
            )
    except Exception as ex:
        logger.warning(
            "Estimate skipped due to diagnostic failure: "
            f"{ex}"
        )


def _run_hash_job_with_progress_bar(
    *,
    desc: str,
    total_hint: int,
    enabled: bool,
    runner: typing.Callable[[typing.Callable[[int, int], None] | None, int], typing.Any],
) -> typing.Any:
    total = int(max(0, int(total_hint)))
    if (not bool(enabled)) or total <= 0:
        return runner(None, 0)

    pbar = ProgressAdapter(
        total=total,
        desc=str(desc),
        show_spinner=False,
        show_postfix=False,
        show_remaining=False,
        keep_display=False,
        emit_done=False,
        force_animate=True,
    )
    shown = 0

    def _on_progress(done: int, total_from_backend: int) -> None:
        nonlocal shown
        b_total = int(max(1, int(total_from_backend)))
        b_done = int(min(max(0, int(done)), b_total))
        mapped = int(round(float(b_done) * float(total) / float(b_total)))
        target = int(min(max(shown, mapped), total))
        if target > shown:
            pbar.update(target - shown)
            shown = target

    try:
        out = runner(_on_progress, _hash_progress_step(total))
    except Exception:
        pbar.close()
        raise
    if shown < total:
        pbar.update(total - shown)
    pbar.finish()
    pbar.close()
    return out


def _terminal_columns(default: int = 100) -> int:
    try:
        cols = int(os.get_terminal_size().columns)
        return max(40, cols)
    except Exception:
        return int(max(40, default))


def _compact_done_message(name: str, details: list[str] | None = None) -> str:
    base = f"{str(name).strip()} ...Finished"
    items = [str(x).strip() for x in (details or []) if str(x).strip() != ""]
    if len(items) == 0:
        return base
    max_width = max(24, _terminal_columns() - 8)
    chosen: list[str] = []
    for item in items:
        trial = f"{base} ({', '.join(chosen + [item])})"
        if len(trial) <= max_width:
            chosen.append(item)
        else:
            break
    if len(chosen) == 0:
        return base
    if len(chosen) < len(items):
        trial_more = f"{base} ({', '.join(chosen + ['...'])})"
        if len(trial_more) <= max_width:
            chosen.append("...")
    return f"{base} ({', '.join(chosen)})"


def _run_indeterminate_task_with_progress_bar(
    *,
    desc: str,
    enabled: bool,
    runner: typing.Callable[[], typing.Any],
    interval_s: float = 0.2,
    total_ticks: int = 80,
    cap_ratio: float = 0.95,
) -> typing.Any:
    if not bool(enabled):
        return runner()
    total = max(20, int(total_ticks))
    cap = max(1, min(total - 1, int(round(total * float(cap_ratio)))))
    pbar = ProgressAdapter(
        total=total,
        desc=str(desc),
        show_spinner=False,
        show_postfix=False,
        show_remaining=False,
        keep_display=False,
        emit_done=False,
        force_animate=True,
    )
    ticked = {"n": 0}
    stop_evt = threading.Event()

    def _ticker() -> None:
        wait_s = max(0.05, float(interval_s))
        while not stop_evt.wait(wait_s):
            if ticked["n"] < cap:
                pbar.update(1)
                ticked["n"] += 1

    th = threading.Thread(target=_ticker, daemon=True)
    th.start()
    try:
        out = runner()
    except Exception:
        stop_evt.set()
        try:
            th.join(timeout=0.6)
        except Exception:
            pass
        pbar.close()
        raise

    stop_evt.set()
    try:
        th.join(timeout=0.6)
    except Exception:
        pass
    if ticked["n"] < total:
        pbar.update(total - ticked["n"])
    pbar.finish()
    pbar.close()
    return out


def _fit_gblup_reml_from_hash_ztz_stats(
    *,
    gram: np.ndarray,
    row_sum: np.ndarray,
    row_sq_sum: np.ndarray,
    zy: np.ndarray,
    n_train: int,
) -> dict[str, typing.Any]:
    g = np.ascontiguousarray(np.asarray(gram, dtype=np.float64))
    rs = np.ascontiguousarray(np.asarray(row_sum, dtype=np.float64).reshape(-1))
    r2 = np.ascontiguousarray(np.asarray(row_sq_sum, dtype=np.float64).reshape(-1))
    zy_vec = np.ascontiguousarray(np.asarray(zy, dtype=np.float64).reshape(-1))
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"gram must be square; got shape={g.shape}.")
    k = int(g.shape[0])
    if int(rs.shape[0]) != k or int(r2.shape[0]) != k or int(zy_vec.shape[0]) != k:
        raise ValueError("row_sum/row_sq_sum/zy length mismatch with gram dimension.")
    n = int(n_train)
    if n <= 1:
        beta = float(np.mean(zy_vec)) if zy_vec.size > 0 else 0.0
        return {
            "beta": float(beta),
            "pve": float("nan"),
            "m_mean": np.zeros((k, 1), dtype=np.float64),
            "m_var_sum": 1.0,
            "m_alpha": np.zeros((k, 1), dtype=np.float64),
            "alpha_sum": 0.0,
            "mean_sq": 0.0,
            "mean_malpha": 0.0,
        }

    m_mean = (rs / float(n)).reshape(-1, 1)
    m_var = (r2 / float(n)).reshape(-1, 1) - m_mean * m_mean
    m_var = np.maximum(m_var, 0.0)
    m_var_sum = float(np.sum(m_var, dtype=np.float64))
    if (not np.isfinite(m_var_sum)) or (m_var_sum <= 0.0):
        m_var_sum = 1e-12

    eigvals, eigvec = np.linalg.eigh(g)
    max_eval = float(eigvals[-1]) if eigvals.size > 0 else 0.0
    tol = np.finfo(np.float64).eps * max(1.0, max_eval) * max(1, k)
    keep_start = int(np.searchsorted(eigvals, tol, side="right"))
    if keep_start >= int(eigvals.shape[0]):
        raise RuntimeError("Numerically singular hashed ZTZ matrix (all eigenvalues dropped).")
    eigvals = np.asarray(eigvals[keep_start:], dtype=np.float64)
    eigvec = np.asarray(eigvec[:, keep_start:], dtype=np.float64)
    svals = np.sqrt(np.maximum(eigvals, 1e-18))
    inv_s = 1.0 / svals
    s_spec = svals * svals

    mx = rs.reshape(-1, 1)
    my = zy_vec.reshape(-1, 1)
    x_proj = (eigvec.T @ mx) * inv_s[:, None]
    y_proj = (eigvec.T @ my) * inv_s[:, None]
    v_floor = max(1e-12, float(np.finfo(np.float64).tiny))
    n_obj = int(x_proj.shape[0])
    n_eff_obj = int(max(1, n_obj - 1))

    def _reml_at_lambda(lbd: float) -> tuple[float, float, np.ndarray, np.ndarray, float]:
        lam = float(lbd)
        if (not np.isfinite(lam)) or (lam <= 0.0):
            lam = v_floor
        v = np.maximum(s_spec + lam, v_floor)
        v_inv = 1.0 / v
        x_w = v_inv[:, None] * x_proj
        xtvx = float((x_proj.T @ x_w)[0, 0])
        if (not np.isfinite(xtvx)) or (xtvx <= v_floor):
            xtvx = v_floor
        xtvy = float((x_proj.T @ (v_inv[:, None] * y_proj))[0, 0])
        beta = float(xtvy / xtvx)
        r = y_proj - x_proj * beta
        rtv = float(np.sum((r.reshape(-1) ** 2) * v_inv))
        if (not np.isfinite(rtv)) or (rtv <= v_floor):
            rtv = v_floor
        log_det_v = float(np.sum(np.log(v)))
        log_det_xtvx = float(np.log(xtvx))
        total_log = float(n_eff_obj) * float(np.log(rtv)) + log_det_v + log_det_xtvx
        c = float(n_eff_obj) * (
            float(np.log(float(n_eff_obj))) - 1.0 - float(np.log(2.0 * np.pi))
        ) / 2.0
        return c - total_log / 2.0, beta, r, v_inv, rtv

    def _objective(log10_lbd: float) -> float:
        lbd = 10.0 ** float(log10_lbd)
        ll, _beta, _r, _v_inv, _rtv = _reml_at_lambda(lbd)
        return -float(ll)

    opt = minimize_scalar(_objective, bounds=(-6, 6), method="bounded")
    log10_lbd = float(opt.x) if np.isfinite(float(opt.x)) else 0.0
    lbd = float(10.0 ** log10_lbd)
    _ll, beta, r, v_inv, rtv = _reml_at_lambda(lbd)

    rhs = (v_inv.reshape(-1, 1) * r).reshape(-1, 1)
    q_alpha = eigvec @ (inv_s[:, None] * rhs)
    m_alpha = eigvec @ (svals[:, None] * rhs)
    alpha_sum = float((rs.reshape(1, -1) @ q_alpha)[0, 0])
    mean_sq = float((m_mean.T @ m_mean)[0, 0])
    mean_malpha = float((m_mean.T @ m_alpha)[0, 0])

    sigma_g2 = float(rtv) / float(max(1, n - 1))
    sigma_e2 = float(lbd) * sigma_g2
    var_g = sigma_g2 * float(np.mean(s_spec))
    denom = var_g + sigma_e2
    pve = float(var_g / denom) if (np.isfinite(denom) and denom > 0.0) else float("nan")
    return {
        "beta": float(beta),
        "pve": float(pve),
        "m_mean": np.ascontiguousarray(m_mean, dtype=np.float64),
        "m_var_sum": float(m_var_sum),
        "m_alpha": np.ascontiguousarray(m_alpha, dtype=np.float64),
        "alpha_sum": float(alpha_sum),
        "mean_sq": float(mean_sq),
        "mean_malpha": float(mean_malpha),
    }


def _predict_hashed_gblup_test_from_compact(
    *,
    packed_ctx: dict[str, typing.Any],
    hash_dim: int,
    hash_seed: int,
    standardize: bool,
    n_jobs: int,
    test_sample_indices: np.ndarray,
    beta: float,
    m_mean: np.ndarray,
    m_var_sum: float,
    m_alpha: np.ndarray,
    alpha_sum: float,
    mean_sq: float,
    mean_malpha: float,
    sample_chunk: int = 4096,
) -> np.ndarray:
    test_idx = np.ascontiguousarray(np.asarray(test_sample_indices, dtype=np.int64).reshape(-1))
    n_test = int(test_idx.shape[0])
    if n_test == 0:
        return np.zeros((0, 1), dtype=np.float64)
    out = np.zeros((n_test, 1), dtype=np.float64)
    step = max(1, int(sample_chunk))
    m_mean64 = np.ascontiguousarray(np.asarray(m_mean, dtype=np.float64).reshape(-1, 1))
    m_alpha64 = np.ascontiguousarray(np.asarray(m_alpha, dtype=np.float64).reshape(-1, 1))
    denom = float(m_var_sum)
    if (not np.isfinite(denom)) or (denom <= 0.0):
        raise ValueError(f"Invalid hashed GBLUP denominator: {denom!r}")
    for st in range(0, n_test, step):
        ed = min(st + step, n_test)
        blk_idx = np.ascontiguousarray(test_idx[st:ed], dtype=np.int64)
        z_blk, _scale, _kept = _hash_packed_for_gs(
            packed_ctx=packed_ctx,
            hash_dim=int(hash_dim),
            hash_seed=int(hash_seed),
            standardize=bool(standardize),
            n_jobs=max(1, int(n_jobs)),
            sample_indices=blk_idx,
        )
        z64 = np.asarray(z_blk, dtype=np.float64)
        t1 = z64.T @ m_alpha64
        t2 = z64.T @ m_mean64
        cross = (t1 - t2 * float(alpha_sum) - float(mean_malpha) + float(mean_sq) * float(alpha_sum)) / denom
        out[st:ed] = cross + float(beta)
    return np.ascontiguousarray(out, dtype=np.float64)


# ======================================================================
# CLI
# ======================================================================

def parse_args(argv: typing.Optional[list[str]] = None):
    parser = CliArgumentParser(
        prog="jx gs",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx gs -vcf geno.vcf.gz -p pheno.tsv -GBLUP -cv 5",
            "jx gs -hmp geno.hmp.gz -p pheno.tsv -GBLUP -cv 5",
            "jx gs -file geno_prefix -p pheno.tsv -GBLUP -cv 5",
            "jx gs -vcf geno.vcf.gz -p pheno.tsv -GBLUP ad -cv 5",
            "jx gs -vcf geno.vcf.gz -p pheno.tsv -RF -ET -GBDT -SVM -ENET -cv 5",
            "jx gs -bfile geno_prefix -p pheno.tsv -GBLUP -rrBLUP",
        ]),
    )

    # ------------------------------------------------------------------
    # Required arguments
    # ------------------------------------------------------------------
    required_group = parser.add_argument_group("Required Arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=False)
    geno_group.add_argument(
        "-vcf", "--vcf",
        type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-hmp", "--hmp",
        type=str,
        help="Input genotype file in HMP format (.hmp or .hmp.gz).",
    )
    geno_group.add_argument(
        "-bfile", "--bfile",
        type=str,
        help="Input genotype files in PLINK binary format "
             "(prefix for .bed, .bim, .fam).",
    )
    geno_group.add_argument(
        "-file", "--file",
        type=str,
        help=(
            "Input genotype numeric matrix (.txt/.tsv/.csv/.npy) or prefix. "
            "Requires sibling prefix.id. Optional site metadata: prefix.site or prefix.bim."
        ),
    )
    required_group.add_argument(
        "-p", "--pheno",
        type=str,
        required=False,
        help="Phenotype file (tab/comma/whitespace-delimited, sample IDs in the first column).",
    )

    # ------------------------------------------------------------------
    # Model arguments
    # ------------------------------------------------------------------
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument(
        "-GBLUP", "--GBLUP",
        action="append",
        nargs="*",
        metavar="KERNEL",
        default=None,
        help=(
            "Use GBLUP model for training and prediction. Optional kernel token(s): "
            "a|d|ad. Example: -GBLUP, -GBLUP a, -GBLUP d, -GBLUP ad. "
            "No token defaults to additive (a)."
        ),
    )
    model_group.add_argument(
        "-adBLUP", "--adBLUP",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    model_group.add_argument(
        "-rrBLUP", "--rrBLUP",
        action="store_true",
        default=False,
        help="Use rrBLUP model for training and prediction "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-BayesA", "--BayesA",
        action="store_true",
        default=False,
        help="Use BayesA model for training and prediction "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-BayesB", "--BayesB",
        action="store_true",
        default=False,
        help="Use BayesB model for training and prediction "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-BayesCpi", "--BayesCpi",
        action="store_true",
        default=False,
        help="Use BayesCpi model for training and prediction "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-RF", "--RF",
        action="store_true",
        default=False,
        help="Use random forest genomic selection with compact inner tuning "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-ET", "--ET",
        action="store_true",
        default=False,
        help="Use extra-trees genomic selection with compact inner tuning "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-GBDT", "--GBDT",
        action="store_true",
        default=False,
        help="Use histogram gradient boosting genomic selection with compact inner tuning "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-XGB", "--XGB",
        action="store_true",
        default=False,
        help="Use XGBoost genomic selection with compact inner tuning "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-SVM", "--SVM",
        action="store_true",
        default=False,
        help="Use RBF-support vector regression genomic selection "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-ENET", "--ENET",
        action="store_true",
        default=False,
        help="Use ElasticNet genomic selection with compact inner tuning "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-model", "--model",
        type=str,
        default=None,
        help=(
            "Load saved .jxmodel file for direct prediction/evaluation. "
            "When set, selected methods are loaded from model artifact instead of refitting."
        ),
    )
    # ------------------------------------------------------------------
    # Optional arguments
    # ------------------------------------------------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-maf", "--maf",
        type=float,
        default=0.02,
        help="Exclude variants with minor allele frequency lower than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-geno", "--geno",
        type=float,
        default=0.05,
        help="Exclude variants with missing call frequencies greater than a threshold "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-n", "--n",
        action="extend",
        nargs="+",
        metavar="COL",
        type=str,
        default=None,
        dest="ncol",
        help=(
            "Phenotype column(s), zero-based index (excluding sample ID), "
            "comma list (e.g. 0,2), or numeric range (e.g. 0:2). "
            "Repeat this flag for multiple traits."
        ),
    )
    optional_group.add_argument(
        "--ncol",
        action="extend",
        nargs="+",
        metavar="COL",
        type=str,
        default=None,
        dest="ncol",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-cv", "--cv",
        type=int,
        default=None,
        help="K fold of cross-validation for models. "
             "(default: %(default)s).",
    )
    # optional_group.add_argument(
    #     "-select","--select",
    #     type=str,
    #     nargs="?",
    #     const=_SELECT_INTERACTIVE_SENTINEL,
    #     default=None,
    #     help=(
    #         "TOP target trait values for ranking. "
    #         "Accepts a comma list (e.g. 180,200,121), or a target-value file "
    #         "(phenotype-like table). If provided without a value, interactive input is used."
    #     ),
    # )
    optional_group.add_argument(
        "--model-select",
        type=str,
        choices=("per-trait", "global"),
        default="per-trait",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--model-select-metric",
        type=str,
        choices=("pearson", "spearman", "r2", "rmse", "nrmse"),
        default="pearson",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-mode",
        type=str,
        choices=("auto", "exact-newton", "exact-bfgs", "quasi-newton", "minibatch-adam"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-exact-threshold",
        type=int,
        default=20000,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-max-iter",
        type=int,
        default=50,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-tol",
        type=float,
        default=1e-6,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-l2",
        type=float,
        default=1e-4,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-batch-size",
        type=int,
        default=1024,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-epochs",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-lr",
        type=float,
        default=1e-2,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-seed",
        type=int,
        default=2026,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--top-calibration",
        type=str,
        choices=("linear", "none", "addmean"),
        default="linear",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-t", "--thread",
        type=int,
        default=detect_effective_threads(),
        help="Number of CPU threads (default: %(default)s).",
    )
    optional_group.add_argument(
        "-limit-predtrain", "--limit-predtrain",
        "-limit-train", "--limit-train",
        type=int,
        default=None,
        dest="limit_predtrain",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-strict-cv", "--strict-cv",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-force-fast", "--force-fast",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-solver",
        type=str,
        choices=("exact", "adamw", "pcg", "auto"),
        default="pcg",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lambda",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lambda-scale",
        type=str,
        choices=("equation", "mean-loss"),
        default="equation",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lambda-auto",
        type=str,
        choices=("on", "off"),
        default="on",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-auto-pcg-min-n",
        type=int,
        default=_RRBLUP_AUTO_PCG_MIN_N,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lambda-subsample-n",
        type=int,
        default=_RRBLUP_LAMBDA_SUBSAMPLE_MAX_N,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lambda-subsample-repeats",
        type=int,
        default=_RRBLUP_LAMBDA_SUBSAMPLE_REPEATS,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lambda-subsample-seed",
        type=int,
        default=42,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-he-thread-policy",
        type=str,
        default=os.getenv(
            "JX_RRBLUP_HE_THREAD_POLICY",
            os.getenv("JX_GS_HE_THREAD_POLICY", _RRBLUP_HE_THREAD_POLICY_DEFAULT),
        ),
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lr",
        type=float,
        default=1e-2,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-epochs",
        type=int,
        default=60,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-batchsize", "--batchsize",
        "--rrblup-batch-size",
        type=int,
        default=1024,
        dest="rrblup_batch_size",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-batch-threads",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-snp-block-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-beta1",
        type=float,
        default=0.9,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-beta2",
        type=float,
        default=0.999,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-eps",
        type=float,
        default=1e-8,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-seed",
        type=int,
        default=42,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-auto-min-cells",
        type=int,
        default=200000000,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-log-every",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-sample-chunk-size",
        type=int,
        default=4096,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-pve-mode",
        type=str,
        choices=("lambda", "trainvar"),
        default="lambda",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-auto-grid",
        type=str,
        choices=("on", "off"),
        default="on",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-grid-size",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-grid-min-samples",
        type=int,
        default=256,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-grid-trial-epochs",
        type=int,
        default=20,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-grid-switch-min-improve",
        type=float,
        default=0.15,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-grid-reuse-cv",
        type=str,
        choices=("on", "off"),
        default="on",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-grid-seed",
        type=int,
        default=42,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-es-val-frac",
        type=float,
        default=0.08,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-es-val-min",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-es-min-train",
        type=int,
        default=128,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-es-patience",
        type=int,
        default=3,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-es-warmup",
        type=int,
        default=4,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-es-min-delta",
        type=float,
        default=1e-5,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-pcg-tol",
        type=float,
        default=1e-4,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-pcg-max-iter",
        type=int,
        default=100,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-pcg-std-eps",
        type=float,
        default=1e-12,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--packed-lmm-auto",
        type=str,
        choices=("on", "off"),
        default="on",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--packed-lmm-auto-min-cells",
        type=int,
        default=200000000,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--bayes-r2-cv-reuse",
        type=str,
        choices=("on", "off"),
        default="on",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--bayes-r2-subsample-min-n",
        type=int,
        default=10000,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--bayes-r2-subsample-n",
        type=int,
        default=2000,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--bayes-r2-subsample-max-n",
        type=int,
        default=_RRBLUP_LAMBDA_SUBSAMPLE_MAX_N,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--bayes-r2-subsample-repeats",
        type=int,
        default=_RRBLUP_LAMBDA_SUBSAMPLE_REPEATS,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--bayes-r2-subsample-seed",
        type=int,
        default=42,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-pcd", "--pcd",
        action="store_true",
        default=False,
        help="Enable PCA-based dimensionality reduction on genotypes "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-ldprune", "--ldprune",
        nargs=3,
        default=None,
        metavar=("WINDOW", "STEP", "R2"),
        help=(
            "Apply packed BED LD prune before GS model fitting. "
            "Usage: --ldprune <window size[kb|bp]> <step size (variant ct)> <r^2 threshold>. "
            "Set env JX_GS_LDPRUNE_BACKEND=plink to use external PLINK backend "
            "(same prune strategy as PLINK indep-pairwise). "
            "Numeric window defaults to kb (e.g. 1=1kb, 0.1=100bp)."
        ),
    )
    optional_group.add_argument(
        "-hash", "--hash",
        nargs="*",
        default=None,
        metavar=("DIM", "SEED"),
        help=(
            "Enable signed feature hashing before GS. "
            "Usage: --hash [dim] [seed]. "
            "Defaults when enabled: dim=2048, seed=520."
        ),
    )
    # Backward-compatible legacy flags (hidden): --hash-dim / --hash-seed
    optional_group.add_argument("-hash-dim", "--hash-dim", type=int, default=None, help=argparse.SUPPRESS)
    optional_group.add_argument("-hash-seed", "--hash-seed", type=int, default=None, help=argparse.SUPPRESS)
    optional_group.add_argument(
        "-hash-raw", "--hash-raw",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "-debug", "--debug",
        action="store_true",
        default=False,
        help="Enable runtime debug logs (thread/runtime/model diagnostics).",
    )
    optional_group.add_argument(
        "-o", "--out",
        type=str,
        default=".",
        help="Output directory for results (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix",
        type=str,
        default=None,
        help="Prefix of output files "
             "(default: genotype basename).",
    )

    args, extras = parser.parse_known_args(argv)
    has_genotype = bool(args.vcf or args.hmp or args.bfile or args.file)
    has_pheno = bool(args.pheno)
    if (not has_pheno) and (not has_genotype):
        parser.error(
            "the following arguments are required: -p/--pheno & "
            "(-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)"
        )
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if not has_genotype:
        parser.error("the following arguments are required: (-vcf VCF | -hmp HMP | -file FILE | -bfile BFILE)")
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))
    try:
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))
    if (args.limit_predtrain is not None) and (int(args.limit_predtrain) < 0):
        parser.error("--limit-predtrain/--limit-train must be >= 0.")
    if (not np.isfinite(float(args.rrblup_lambda))) or (float(args.rrblup_lambda) < 0.0):
        parser.error("--rrblup-lambda must be a finite value >= 0.")
    if (not np.isfinite(float(args.rrblup_lr))) or (float(args.rrblup_lr) <= 0.0):
        parser.error("--rrblup-lr must be a finite value > 0.")
    if int(args.rrblup_epochs) <= 0:
        parser.error("--rrblup-epochs must be > 0.")
    if int(args.rrblup_batch_size) <= 0:
        parser.error("-batchsize/--batchsize/--rrblup-batch-size must be > 0.")
    if int(args.rrblup_batch_threads) < 0:
        parser.error("--rrblup-batch-threads must be >= 0.")
    if args.rrblup_snp_block_size is None:
        # Default mini-batch logic: use batch-size as the primary memory knob.
        args.rrblup_snp_block_size = int(args.rrblup_batch_size)
    if int(args.rrblup_snp_block_size) <= 0:
        parser.error("--rrblup-snp-block-size must be > 0.")
    if int(args.rrblup_auto_min_cells) <= 0:
        parser.error("--rrblup-auto-min-cells must be > 0.")
    if int(args.rrblup_auto_pcg_min_n) <= 1:
        parser.error("--rrblup-auto-pcg-min-n must be > 1.")
    if int(args.rrblup_lambda_subsample_n) <= 0:
        parser.error("--rrblup-lambda-subsample-n must be > 0.")
    if int(args.rrblup_lambda_subsample_repeats) <= 0:
        parser.error("--rrblup-lambda-subsample-repeats must be > 0.")
    if int(args.rrblup_log_every) < 0:
        parser.error("--rrblup-log-every must be >= 0.")
    if int(args.rrblup_sample_chunk_size) <= 0:
        parser.error("--rrblup-sample-chunk-size must be > 0.")
    if (not np.isfinite(float(args.rrblup_beta1))) or (not np.isfinite(float(args.rrblup_beta2))):
        parser.error("--rrblup-beta1/--rrblup-beta2 must be finite.")
    if float(args.rrblup_beta1) < 0.0 or float(args.rrblup_beta1) >= 1.0:
        parser.error("--rrblup-beta1 must be in [0, 1).")
    if float(args.rrblup_beta2) < 0.0 or float(args.rrblup_beta2) >= 1.0:
        parser.error("--rrblup-beta2 must be in [0, 1).")
    if (not np.isfinite(float(args.rrblup_eps))) or (float(args.rrblup_eps) <= 0.0):
        parser.error("--rrblup-eps must be a finite value > 0.")
    if int(args.rrblup_grid_size) <= 0 or int(args.rrblup_grid_size) > 4:
        parser.error("--rrblup-grid-size must be in [1, 4].")
    if int(args.rrblup_grid_min_samples) <= 0:
        parser.error("--rrblup-grid-min-samples must be > 0.")
    if int(args.rrblup_grid_trial_epochs) <= 0:
        parser.error("--rrblup-grid-trial-epochs must be > 0.")
    if (not np.isfinite(float(args.rrblup_grid_switch_min_improve))) or float(args.rrblup_grid_switch_min_improve) < 0.0:
        parser.error("--rrblup-grid-switch-min-improve must be a finite value >= 0.")
    if (not np.isfinite(float(args.rrblup_es_val_frac))) or float(args.rrblup_es_val_frac) < 0.0:
        parser.error("--rrblup-es-val-frac must be a finite value >= 0.")
    if int(args.rrblup_es_val_min) <= 0:
        parser.error("--rrblup-es-val-min must be > 0.")
    if int(args.rrblup_es_min_train) <= 0:
        parser.error("--rrblup-es-min-train must be > 0.")
    if int(args.rrblup_es_patience) < 0:
        parser.error("--rrblup-es-patience must be >= 0.")
    if int(args.rrblup_es_warmup) <= 0:
        parser.error("--rrblup-es-warmup must be > 0.")
    if (not np.isfinite(float(args.rrblup_es_min_delta))) or float(args.rrblup_es_min_delta) < 0.0:
        parser.error("--rrblup-es-min-delta must be a finite value >= 0.")
    if (not np.isfinite(float(args.rrblup_pcg_tol))) or (float(args.rrblup_pcg_tol) <= 0.0):
        parser.error("--rrblup-pcg-tol must be a finite value > 0.")
    if int(args.rrblup_pcg_max_iter) <= 0:
        parser.error("--rrblup-pcg-max-iter must be > 0.")
    if (not np.isfinite(float(args.rrblup_pcg_std_eps))) or (float(args.rrblup_pcg_std_eps) <= 0.0):
        parser.error("--rrblup-pcg-std-eps must be a finite value > 0.")
    if int(args.packed_lmm_auto_min_cells) <= 0:
        parser.error("--packed-lmm-auto-min-cells must be > 0.")
    if int(args.bayes_r2_subsample_min_n) <= 1:
        parser.error("--bayes-r2-subsample-min-n must be > 1.")
    if int(args.bayes_r2_subsample_n) <= 1:
        parser.error("--bayes-r2-subsample-n must be > 1.")
    if int(args.bayes_r2_subsample_max_n) <= 1:
        parser.error("--bayes-r2-subsample-max-n must be > 1.")
    if int(args.bayes_r2_subsample_repeats) <= 0:
        parser.error("--bayes-r2-subsample-repeats must be > 0.")
    try:
        args.ldprune_spec = _parse_ld_prune_args(args.ldprune)
    except ValueError as e:
        parser.error(str(e))
    hash_default_dim = 2048
    hash_default_seed = 520
    hash_tokens = typing.cast(list[str] | None, getattr(args, "hash", None))
    legacy_hash_dim = getattr(args, "hash_dim", None)
    legacy_hash_seed = getattr(args, "hash_seed", None)
    if (hash_tokens is not None) and ((legacy_hash_dim is not None) or (legacy_hash_seed is not None)):
        parser.error("Use either --hash or legacy --hash-dim/--hash-seed, not both.")

    if hash_tokens is not None:
        if len(hash_tokens) > 2:
            parser.error("--hash accepts at most two values: [dim] [seed].")
        if len(hash_tokens) >= 1:
            try:
                args.hash_dim = int(hash_tokens[0])
            except Exception:
                parser.error(f"Invalid --hash dim value: {hash_tokens[0]!r}")
        else:
            args.hash_dim = int(hash_default_dim)
        if len(hash_tokens) >= 2:
            try:
                args.hash_seed = int(hash_tokens[1])
            except Exception:
                parser.error(f"Invalid --hash seed value: {hash_tokens[1]!r}")
        else:
            args.hash_seed = int(hash_default_seed)
    else:
        if legacy_hash_dim is not None:
            args.hash_dim = int(legacy_hash_dim)
            args.hash_seed = int(hash_default_seed if legacy_hash_seed is None else legacy_hash_seed)
        else:
            args.hash_dim = None
            args.hash_seed = int(hash_default_seed if legacy_hash_seed is None else legacy_hash_seed)

    if (args.hash_dim is not None) and (int(args.hash_dim) <= 0):
        parser.error("--hash dim must be > 0.")
    if args.model is not None:
        model_path = str(args.model).strip()
        if model_path == "":
            parser.error("--model file must not be empty.")
        if not os.path.isfile(model_path):
            parser.error(f"--model must be a .jxmodel file (got: {model_path})")
        if not str(model_path).lower().endswith(".jxmodel"):
            parser.error(f"--model must point to a .jxmodel file (got: {model_path})")
        args.model = model_path
    # if args.select is not None:
    #     if str(args.select).strip() == _SELECT_INTERACTIVE_SENTINEL:
    #         args.select = _SELECT_INTERACTIVE_SENTINEL
    #     else:
    #         target = str(args.select).strip()
    #         if target == "":
    #             parser.error("--select must not be empty.")
    #         args.select = target
    if int(args.top_exact_threshold) <= 0:
        parser.error("--top-exact-threshold must be > 0.")
    if int(args.top_max_iter) <= 0:
        parser.error("--top-max-iter must be > 0.")
    if not np.isfinite(float(args.top_tol)) or float(args.top_tol) <= 0.0:
        parser.error("--top-tol must be a finite value > 0.")
    if not np.isfinite(float(args.top_l2)) or float(args.top_l2) < 0.0:
        parser.error("--top-l2 must be a finite value >= 0.")
    if int(args.top_batch_size) <= 0:
        parser.error("--top-batch-size must be > 0.")
    if int(args.top_epochs) <= 0:
        parser.error("--top-epochs must be > 0.")
    if not np.isfinite(float(args.top_lr)) or float(args.top_lr) <= 0.0:
        parser.error("--top-lr must be a finite value > 0.")
    if int(args.top_seed) < 0:
        parser.error("--top-seed must be >= 0.")
    return args


def _run_gs_pipeline(
    args=None,
    *,
    argv: typing.Optional[list[str]] = None,
    log: bool = True,
    return_result: bool = False,
) -> typing.Optional[dict[str, typing.Any]]:
    t_start = time.time()
    use_spinner = stdout_is_tty()
    if args is None:
        args = parse_args(argv)
    debug_mode = bool(getattr(args, "debug", False))

    gblup_modes = _parse_gblup_kernel_modes(
        typing.cast(list[list[str]] | None, args.GBLUP)
    )
    if bool(args.adBLUP):
        if "ad" not in gblup_modes:
            gblup_modes.append("ad")
        try:
            # Use stderr warning before logging setup.
            sys.stderr.write(
                "[warn] -adBLUP is deprecated; use -GBLUP ad instead.\n"
            )
        except Exception:
            pass
    gblup_methods = [_gblup_mode_to_method(m) for m in gblup_modes]

    rrblup_solver = str(args.rrblup_solver).strip().lower()
    rrblup_adamw_cfg: dict[str, typing.Any] = {
        "lambda_value": float(args.rrblup_lambda),
        "lambda_scale": str(args.rrblup_lambda_scale),
        "lambda_auto": str(args.rrblup_lambda_auto),
        "lambda_auto_strategy": "he_first",
        "he_enable": "on",
        "auto_pcg_min_n": int(args.rrblup_auto_pcg_min_n),
        "lambda_subsample_n": int(args.rrblup_lambda_subsample_n),
        "lambda_subsample_min_n": int(_RRBLUP_LAMBDA_SUBSAMPLE_MIN_N),
        "lambda_subsample_max_n": int(_RRBLUP_LAMBDA_SUBSAMPLE_MAX_N),
        "lambda_subsample_repeats": int(args.rrblup_lambda_subsample_repeats),
        "lambda_subsample_seed": int(args.rrblup_lambda_subsample_seed),
        "he_thread_policy": str(args.rrblup_he_thread_policy),
        "lambda_large_cutoff": 15_000,
        "lr": float(args.rrblup_lr),
        "epochs": int(args.rrblup_epochs),
        "batch_size": int(args.rrblup_batch_size),
        "batch_threads": int(args.rrblup_batch_threads),
        "snp_block_size": int(args.rrblup_snp_block_size),
        "beta1": float(args.rrblup_beta1),
        "beta2": float(args.rrblup_beta2),
        "eps": float(args.rrblup_eps),
        "seed": int(args.rrblup_seed),
        "auto_min_cells": int(args.rrblup_auto_min_cells),
        "log_every": int(args.rrblup_log_every),
        "sample_chunk_size": int(args.rrblup_sample_chunk_size),
        "pve_mode": str(args.rrblup_pve_mode),
        "auto_grid": str(args.rrblup_auto_grid),
        "grid_size": int(args.rrblup_grid_size),
        "grid_min_samples": int(args.rrblup_grid_min_samples),
        "grid_trial_epochs": int(args.rrblup_grid_trial_epochs),
        "grid_switch_min_improve": float(args.rrblup_grid_switch_min_improve),
        "grid_reuse_cv": str(args.rrblup_grid_reuse_cv),
        "grid_seed": int(args.rrblup_grid_seed),
        "es_val_frac": float(args.rrblup_es_val_frac),
        "es_val_min": int(args.rrblup_es_val_min),
        "es_min_train": int(args.rrblup_es_min_train),
        "es_patience": int(args.rrblup_es_patience),
        "es_warmup": int(args.rrblup_es_warmup),
        "es_min_delta": float(args.rrblup_es_min_delta),
        "pcg_tol": float(args.rrblup_pcg_tol),
        "pcg_max_iter": int(args.rrblup_pcg_max_iter),
        "pcg_std_eps": float(args.rrblup_pcg_std_eps),
        "pcg_progress_every": 1,
        "pcg_block_rows": int(args.rrblup_snp_block_size),
        "debug_mode": bool(debug_mode),
    }
    bayes_auto_r2_cfg: dict[str, typing.Any] = {
        "cv_reuse": str(args.bayes_r2_cv_reuse),
        "subsample_min_n": int(args.bayes_r2_subsample_min_n),
        "subsample_n": int(args.bayes_r2_subsample_n),
        "subsample_min_samples": int(_RRBLUP_LAMBDA_SUBSAMPLE_MIN_N),
        "subsample_max_n": int(args.bayes_r2_subsample_max_n),
        "subsample_repeats": int(args.bayes_r2_subsample_repeats),
        "subsample_seed": int(args.bayes_r2_subsample_seed),
    }

    # ------------------------------------------------------------------
    # Determine genotype file and output prefix
    # ------------------------------------------------------------------
    if args.vcf:
        gfile = args.vcf
        args.prefix = (
            os.path.basename(gfile)
            .replace(".gz", "")
            .replace(".vcf", "")
            if args.prefix is None else args.prefix
        )
    elif args.hmp:
        gfile = args.hmp
        args.prefix = (
            os.path.basename(gfile)
            .replace(".gz", "")
            .replace(".hmp", "")
            if args.prefix is None else args.prefix
        )
    elif args.file:
        gfile = args.file
        base = os.path.basename(gfile)
        for ext in (".npy", ".txt", ".tsv", ".csv"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
                break
        args.prefix = base if args.prefix is None else args.prefix
    elif args.bfile:
        gfile = args.bfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    else:
        raise ValueError("No genotype input detected. Use -vcf, -hmp, -file or -bfile.")

    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, args.prefix)
    gs_model_dir = os.path.join(args.out, f"{args.prefix}.gs.model")
    detected_threads = detect_effective_threads()
    requested_threads = int(args.thread)
    thread_capped = False
    if int(args.thread) <= 0:
        args.thread = int(detected_threads)
    if int(args.thread) > int(detected_threads):
        thread_capped = True
        args.thread = int(detected_threads)
    # Keep pyBLUP eigh fallback aligned with CLI `-t` even when runtime
    # thread coordination is skipped for mixed model sets.
    os.environ["JX_THREADS"] = str(int(args.thread))

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    os.makedirs(gs_model_dir, 0o755, exist_ok=True)
    configure_genotype_cache_from_out(args.out)
    log_path = f"{outprefix}.gs.log"
    logger = setup_logging(log_path)
    _file_only_logger = logging.getLogger("janusx.gs.file_only")
    _file_only_logger.handlers.clear()
    _file_only_logger.setLevel(logging.INFO)
    _file_only_logger.propagate = False
    for _h in list(logger.handlers):
        if isinstance(_h, logging.FileHandler):
            _file_only_logger.addHandler(_h)

    def _log_file_only(msg: str) -> None:
        text = str(msg).strip()
        if text != "":
            _file_only_logger.info(text)

    if thread_capped:
        logger.warning(
            f"Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )

    manual_packed_preprocess_requested = bool(
        (args.hash_dim is not None) or (getattr(args, "ldprune_spec", None) is not None)
    )
    rr_solver_mode = str(rrblup_solver).strip().lower()
    file_input_requested = bool(args.file)
    file_txt_like_requested = bool(file_input_requested and _is_txt_like_input(str(gfile)))
    input_vcf_hmp_requested = bool(
        bool(args.vcf)
        or bool(args.hmp)
        or (
            file_input_requested
            and _is_vcf_or_hmp_like_input(str(gfile))
        )
    )
    input_cacheable_requested = bool(input_vcf_hmp_requested or file_input_requested)
    file_memmap_preferred = bool(
        file_input_requested
        and (
            file_txt_like_requested
            or str(gfile).strip().lower().endswith(".npy")
            or str(gfile).strip().lower().endswith(".bin")
        )
    )
    pcg_requested = bool(
        bool(args.rrBLUP) and (rr_solver_mode == "pcg")
    )
    gfile_is_plink = _is_plink_prefix(str(gfile))
    gblup_add_requested = bool("a" in gblup_modes)
    gblup_nonadd_requested = bool(any(m in {"d", "ad"} for m in gblup_modes))
    bayes_requested = bool(bool(args.BayesA) or bool(args.BayesB) or bool(args.BayesCpi))
    packed_model_requested = bool(gblup_add_requested or bool(args.rrBLUP) or bayes_requested)
    if file_memmap_preferred and rr_solver_mode == "pcg" and (not packed_model_requested):
        rrblup_solver = "auto"
        rr_solver_mode = "auto"
        logger.warning(
            "rrBLUP solver=pcg is unavailable for FILE memmap inputs without packed Rust models; "
            "falling back to solver=auto."
        )
        pcg_requested = False
    packed_only_requested = bool(
        packed_model_requested
        and (not gblup_nonadd_requested)
        and (not bool(args.RF))
        and (not bool(args.ET))
        and (not bool(args.GBDT))
        and (not bool(args.XGB))
        and (not bool(args.SVM))
        and (not bool(args.ENET))
    )
    auto_packed_lmm_requested = False
    probe_n_samples: int | None = None
    packed_auto_cells: int | None = None
    packed_auto_min_cells = int(max(1, int(getattr(args, "packed_lmm_auto_min_cells", 200000000))))
    if (
        (not gfile_is_plink)
        and _cfg_truthy(getattr(args, "packed_lmm_auto", "on"), default=True)
        and packed_only_requested
        and input_cacheable_requested
    ):
        # For Bayes on VCF/HMP inputs, prefer packed path by default to avoid
        # dense genotype peak memory. Keep legacy size-threshold probing for
        # other packed-eligible model combinations.
        if bayes_requested and input_vcf_hmp_requested:
            auto_packed_lmm_requested = True
        else:
            try:
                _probe_ids, _probe_m = inspect_genotype_file(
                    str(gfile),
                    snps_only=False,
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                )
                probe_n_samples = int(len(_probe_ids))
                packed_auto_cells = int(max(0, len(_probe_ids))) * int(max(0, int(_probe_m)))
                auto_packed_lmm_requested = bool(packed_auto_cells >= packed_auto_min_cells)
            except Exception as ex:
                # Fail open for memory-safety: if probing fails, prefer packed fallback.
                auto_packed_lmm_requested = True
                if bool(debug_mode):
                    logger.warning(
                        "Packed LMM auto probe failed; defaulting to packed-preprocess path. "
                        f"reason: {ex}"
                    )
    if (
        (not pcg_requested)
        and bool(args.rrBLUP)
        and (rr_solver_mode == "auto")
        and (not gfile_is_plink)
        and input_cacheable_requested
    ):
        if probe_n_samples is None:
            try:
                probe_ids, _probe_m = inspect_genotype_file(
                    str(gfile),
                    snps_only=False,
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                )
                probe_n_samples = int(len(probe_ids))
            except Exception:
                probe_n_samples = None
        auto_pcg_min_n = int(
            max(2, int(rrblup_adamw_cfg.get("auto_pcg_min_n", _RRBLUP_AUTO_PCG_MIN_N)))
        )
        if (probe_n_samples is not None) and (int(probe_n_samples) > int(auto_pcg_min_n)):
            pcg_requested = True
    pcg_packed_preprocess_requested = bool(
        (not gfile_is_plink)
        and pcg_requested
        and input_cacheable_requested
    )
    hard_packed_preprocess_requested = bool(
        manual_packed_preprocess_requested or pcg_packed_preprocess_requested
    )
    always_packed_for_cacheable = bool(input_cacheable_requested and packed_model_requested)
    packed_preprocess_requested = bool(
        hard_packed_preprocess_requested or auto_packed_lmm_requested or always_packed_for_cacheable
    )

    checks: list[bool] = []
    if gfile_is_plink:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
    elif args.file:
        checks.append(ensure_file_input_exists(logger, gfile, "Genotype FILE input"))
    else:
        checks.append(ensure_file_exists(logger, gfile, "Genotype file"))
    checks.append(ensure_file_exists(logger, args.pheno, "Phenotype file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Collect methods to run
    # ------------------------------------------------------------------
    methods: list[str] = []
    methods.extend(list(gblup_methods))
    if args.rrBLUP:
        methods.append("rrBLUP")
    if args.BayesA:
        methods.append("BayesA")
    if args.BayesB:
        methods.append("BayesB")
    if args.BayesCpi:
        methods.append("BayesCpi")
    if args.RF:
        methods.append("RF")
    if args.ET:
        methods.append("ET")
    if args.GBDT:
        methods.append("GBDT")
    if args.XGB:
        methods.append("XGB")
    if args.SVM:
        methods.append("SVM")
    if args.ENET:
        methods.append("ENET")
    # keep order, drop duplicates
    if len(methods) > 1:
        _seen_methods: set[str] = set()
        _uniq_methods: list[str] = []
        for _m in methods:
            if _m in _seen_methods:
                continue
            _seen_methods.add(_m)
            _uniq_methods.append(_m)
        methods = _uniq_methods
    model_mode = bool(args.model is not None)
    top_requested = bool(getattr(args, "select", None) is not None)
    if model_mode and len(methods) == 0:
        inferred = _discover_jxmodel_methods(str(args.model))
        if len(inferred) > 0:
            methods = inferred
    loaded_bundle_file: str | None = None
    loaded_bundle_payload: dict[str, typing.Any] | None = None
    if model_mode:
        try:
            loaded_bundle_file = _resolve_top_bundle_model_file(
                model_arg=str(args.model),
                prefix_hint=str(args.prefix),
            )
            loaded_bundle_payload = _load_jxmodel(str(loaded_bundle_file))
            if top_requested:
                inferred_bundle = _discover_jxmodel_methods(str(loaded_bundle_file))
                if len(inferred_bundle) > 0:
                    methods = list(dict.fromkeys([str(x) for x in inferred_bundle]))
        except Exception:
            loaded_bundle_file = None
            loaded_bundle_payload = None
    if len(methods) == 0:
        logger.error(
            "No model selected. Use "
            "--GBLUP [a|d|ad]/--rrBLUP/--BayesA/--BayesB/--BayesCpi/--RF/--ET/--GBDT/--XGB/--SVM/--ENET "
            "or provide --model with discoverable *.jxmodel files."
        )
        raise SystemExit(1)
    if model_mode:
        unsupported = [m for m in methods if not _is_jxmodel_export_supported(m)]
        if len(unsupported) > 0:
            logger.error(
                "Loaded-model mode currently supports rrBLUP/Bayes/RF/ET/GBDT/XGB/SVM/ENET. "
                "Unsupported: " + ", ".join(unsupported)
            )
            raise SystemExit(1)
    cv_enabled = bool(args.cv is not None)
    if model_mode and cv_enabled:
        logger.warning(
            "--cv is ignored when --model is set. "
            "Loaded-model mode runs prediction only."
        )
        cv_enabled = False
    top_enabled = bool(top_requested)
    if top_enabled and (str(getattr(args, "model_select", "per-trait")).strip().lower() != "per-trait"):
        logger.error("TOP currently supports only --model-select per-trait.")
        raise SystemExit(1)
    if top_enabled and (_jxrs is None):
        logger.error("Rust backend is unavailable: cannot run TOP fitting/ranking.")
        raise SystemExit(1)
    if top_enabled and (not hasattr(_jxrs, "top_rank_to_target_values")):
        logger.error("Rust backend missing top_rank_to_target_values export. Please rebuild JanusX.")
        raise SystemExit(1)
    if top_enabled and model_mode:
        if loaded_bundle_payload is None or not _is_top_bundle_payload(loaded_bundle_payload):
            logger.error(
                "TOP with --model requires a single GS TOP bundle model file "
                "(*.gs.TOP.jxmodel)."
            )
            raise SystemExit(1)
    elif top_enabled and (not cv_enabled):
        logger.error("TOP fitting requires --cv (OOF predictions are required).")
        raise SystemExit(1)
    if top_enabled and (not model_mode) and (not hasattr(_jxrs, "top_fit_model")):
        logger.error("Rust backend missing top_fit_model export. Please rebuild JanusX.")
        raise SystemExit(1)
    ml_methods = {"RF", "ET", "GBDT", "XGB", "SVM", "ENET"}
    gblup_ad_methods = [
        m for m in methods
        if _is_gblup_method(str(m)) and (_gblup_method_kernel_mode(str(m)) == "ad")
    ]
    if (len(gblup_ad_methods) > 0) and (not _HAS_ADBLUP_PY):
        methods = [m for m in methods if m not in set(gblup_ad_methods)]
        logger.warning(
            "Skip GBLUP(ad) because JAX backend dependency is unavailable. "
            f"Original import error: {_ADBLUP_PY_IMPORT_ERROR}"
        )

    if any(m in ml_methods for m in methods) and (not _HAS_SKLEARN):
        skipped_ml = [m for m in methods if m in ml_methods]
        methods = [m for m in methods if m not in ml_methods]
        logger.warning(
            format_missing_dependency_message(
                "Skip ML models (RF/ET/GBDT/SVM/ENET/XGB) because scikit-learn is unavailable.",
                packages=("scikit-learn",),
                extra="ml",
                original_error=_SKLEARN_IMPORT_ERROR,
            )
        )
        logger.warning("Skipped models: " + ", ".join(skipped_ml))

    if ("XGB" in methods) and (not _HAS_XGBOOST):
        methods = [m for m in methods if m != "XGB"]
        logger.warning(
            format_missing_dependency_message(
                "Skip model XGB because xgboost could not be imported.",
                packages=("xgboost",),
                extra="ml",
                original_error=_XGBOOST_IMPORT_ERROR,
            )
        )

    if len(methods) == 0:
        logger.error(
            "All selected models were skipped due to missing optional dependencies."
        )
        raise SystemExit(1)

    if log:
        ncol_cfg = (
            "All"
            if args.ncol is None
            else ",".join(str(int(i)) for i in args.ncol)
        )
        cv_cfg = (
            f"{int(args.cv)}-fold, strict={bool(args.strict_cv)}"
            if cv_enabled
            else (
                "off (--model: predict-only)"
                if (model_mode and (args.cv is not None))
                else "off"
            )
        )
        ld_cfg = (
            args.ldprune_spec.label()
            if getattr(args, "ldprune_spec", None) is not None
            else "off"
        )
        if args.hash_dim is None:
            hash_cfg = "off"
        else:
            hash_cfg = (
                f"k={int(args.hash_dim)}, seed={int(args.hash_seed)}, "
                f"std={'on' if (not bool(args.hash_raw)) else 'off'}"
            )
        model_rows: list[tuple[str, object]] = []
        for i, m in enumerate(methods, start=1):
            detail = ""
            if _is_gblup_method(str(m)):
                detail = f"kernel={_gblup_mode_label(_gblup_method_kernel_mode(str(m)))}"
            elif str(m) == "rrBLUP":
                solver_txt = str(rrblup_solver).strip().lower()
                if solver_txt == "auto":
                    n_hint = probe_n_samples
                    if n_hint is not None:
                        solver_txt = (
                            "auto->PCG"
                            if int(n_hint) > int(max(2, int(args.rrblup_auto_pcg_min_n)))
                            else "auto->exact"
                        )
                    else:
                        solver_txt = "auto"
                detail = f"solver={solver_txt}"
            elif str(m) in {"RF", "ET", "GBDT", "XGB", "SVM", "ENET"}:
                detail = "compact tuning=on"
            elif str(m) in {"BayesA", "BayesB", "BayesCpi"}:
                detail = "bayesian marker model"
            model_rows.append((f"{i}. {_method_display_name(str(m))}", detail))

        emit_cli_configuration(
            logger,
            app_title="JanusX GS configuration",
            config_title="GS configuration",
            host=socket.gethostname(),
            sections=[
                (
                    "Input",
                    [
                        ("Genotype", gfile),
                        ("Phenotype", args.pheno),
                        ("Trait/Pcol", ncol_cfg),
                    ],
                ),
                (
                    "Data filters",
                    [
                        ("MAF", float(args.maf)),
                        ("Missing", float(args.geno)),
                        ("LD prune", ld_cfg),
                        ("Hash", hash_cfg),
                        ("PCA", ("on" if bool(args.pcd) else "off")),
                    ],
                ),
                (
                    "Runtime",
                    [
                        ("Threads", int(args.thread)),
                        ("CV", cv_cfg),
                        ("Model mode", ("on" if model_mode else "off")),
                        (
                            "TOP",
                            (
                                f"on ({str(args.model_select_metric).strip().lower()})"
                                if top_enabled
                                else "off"
                            ),
                        ),
                        ("Train prediction", ("All" if args.limit_predtrain is None else int(args.limit_predtrain))),
                    ],
                ),
                ("Enabled models", model_rows),
            ],
            footer_rows=[("Output prefix", outprefix)],
            line_max_chars=62,
        )

    if packed_preprocess_requested and (not gfile_is_plink):
        if bool(args.vcf) or bool(args.hmp) or bool(args.file):
            src_label = os.path.basename(str(gfile).rstrip("/\\")) or str(gfile)
            with CliStatus(
                f"Preparing PLINK cache for packed GS from {src_label}...",
                enabled=use_spinner,
            ) as task:
                try:
                    cached_prefix = prepare_cli_input_cache(
                        str(gfile),
                        snps_only=False,
                        delimiter=None,
                        # Prefer PLINK cache for GS packed Rust routes, including FILE inputs.
                        prefer_plink_for_txt=True,
                    )
                except Exception as ex:
                    if hard_packed_preprocess_requested:
                        task.fail("Preparing PLINK cache for packed GS ...Failed")
                        raise
                    task.complete("Preparing PLINK cache for packed GS ...Skipped (fallback dense)")
                    logger.warning(
                        "Auto packed LMM cache was requested but failed; fallback to dense path. "
                        f"reason: {ex}"
                    )
                    cached_prefix = None
                if cached_prefix is not None and _is_plink_prefix(str(cached_prefix)):
                    gfile = str(cached_prefix).replace("\\", "/")
                    gfile_is_plink = True
                    task.complete(
                        "Preparing PLINK cache for packed GS ...Finished "
                        f"({format_path_for_display(gfile)})"
                    )
                elif cached_prefix is not None:
                    if hard_packed_preprocess_requested:
                        task.fail("Preparing PLINK cache for packed GS ...Failed")
                        if pcg_packed_preprocess_requested:
                            logger.error(
                                "rrBLUP solver=pcg requires packed PLINK cache, but input could not be "
                                f"converted to PLINK BED: {cached_prefix}."
                            )
                            raise SystemExit(1)
                        logger.error(
                            "--hash/--ldprune require PLINK BED input. "
                            "Auto-cache to PLINK is supported for -vcf/-hmp and for -file "
                            "when source resolves to VCF/HMP/TXT/NPY/BIN. "
                            f"Current input could not be converted to BED: {cached_prefix}."
                        )
                        raise SystemExit(1)
                    task.complete("Preparing PLINK cache for packed GS ...Skipped (fallback dense)")
                    logger.warning(
                        "Auto packed LMM cache produced non-PLINK output; fallback to dense path. "
                        f"resolved={cached_prefix}"
                    )
        else:
            if hard_packed_preprocess_requested:
                if pcg_packed_preprocess_requested:
                    logger.error(
                        "rrBLUP solver=pcg requires packed PLINK cache. "
                        "Auto-cache to PLINK is supported for -vcf/-hmp/-file when convertible."
                    )
                    raise SystemExit(1)
                logger.error(
                    "--hash/--ldprune require PLINK BED input. "
                    "Auto-cache to PLINK is supported for -vcf/-hmp/-file when convertible."
                )
                raise SystemExit(1)
            logger.warning(
                "Auto packed LMM cache requested but input is not convertible; fallback to dense path."
            )

    # ------------------------------------------------------------------
    # Load phenotype
    # ------------------------------------------------------------------
    t_loading = time.time()
    psrc = os.path.basename(str(args.pheno).rstrip("/\\")) or str(args.pheno)
    with CliStatus(f"Loading phenotype from {psrc}...", enabled=use_spinner) as task:
        try:
            pheno = _load_phenotype_flexible(args.pheno)
        except Exception:
            task.fail(f"Loading phenotype from {psrc} ...Failed")
            raise
        task.complete(
            f"Loading phenotype from {psrc} (n={pheno.shape[0]}, npheno={pheno.shape[1]})"
        )

    if pheno.shape[1] <= 0:
        logger.error(
            "No phenotype data found. Please check the phenotype file format.\n"
            f"{pheno.head()}"
        )
        raise SystemExit(1)

    if args.ncol is not None:
        requested_cols = [int(i) for i in args.ncol]
        invalid_cols = [i for i in requested_cols if (i < 0 or i >= pheno.shape[1])]
        if invalid_cols:
            logger.error(
                "IndexError: phenotype column index out of range. "
                f"Invalid: {invalid_cols}; valid range: [0, {pheno.shape[1] - 1}]"
            )
            raise SystemExit(1)

        # Keep user-provided order and drop duplicates.
        seen: set[int] = set()
        unique_cols: list[int] = []
        for i in requested_cols:
            if i not in seen:
                unique_cols.append(i)
                seen.add(i)
        pheno = pheno.iloc[:, unique_cols]

    # Runtime-thread default tuning:
    # when policy is not explicitly set, prefer outer-cap mode so `-t`
    # consistently bounds both BLAS and Rust runtimes.
    policy_env_raw = str(os.getenv("JX_THREAD_POLICY", "")).strip()
    gblup_only_models = bool(len(methods) > 0 and all(m in {_GBLUP_METHOD_ADD, "rrBLUP"} for m in methods))
    if (args.hash_dim is not None) and any(
        _is_gblup_method(str(m)) and (_gblup_method_kernel_mode(str(m)) in {"d", "ad"})
        for m in methods
    ):
        logger.error("--hash is currently not supported together with -GBLUP d/ad.")
        raise SystemExit(1)
    hash_gblup_only_mode = bool(
        (args.hash_dim is not None)
        and (len(methods) == 1)
        and (methods[0] == _GBLUP_METHOD_ADD)
    )
    if (policy_env_raw == "") and bool(gfile_is_plink) and gblup_only_models:
        os.environ["JX_THREAD_POLICY"] = "outer"
        if bool(debug_mode):
            logger.info(
                "Thread runtime hint: packed GBLUP/rrBLUP detected; "
                "using outer policy by default (set JX_THREAD_POLICY to override)."
            )

    thread_plan = configure_thread_runtime(
        total_threads=int(args.thread),
        methods=methods,
        logger=(logger if bool(debug_mode) else None),
    )
    if bool(debug_mode) and (not bool(thread_plan.get("applied", False))):
        logger.info(
            "Thread runtime coordination: "
            f"skipped ({str(thread_plan.get('reason', 'unknown'))})."
        )
    if bool(debug_mode):
        np_blas_backend = maybe_warn_non_openblas(
            logger=logger,
            strict=require_openblas_by_default(),
        )
        rust_blas_backend = str(detect_rust_blas_backend()).strip().lower()
        if rust_blas_backend != "unknown":
            logger.info(f"Rust BLAS backend: {rust_blas_backend}.")
            if (str(np_blas_backend).strip().lower() != "openblas") and (
                rust_blas_backend == "openblas"
            ):
                logger.info(
                    "NumPy/SciPy BLAS is non-openblas, but packed Rust kernels "
                    "will use OpenBLAS."
                )

    if bool(debug_mode):
        logger.info("Effective models: " + ", ".join(methods))
    if any(
        _is_gblup_method(str(m)) and (_gblup_method_kernel_mode(str(m)) in {"d", "ad"})
        for m in methods
    ) and args.pcd:
        logger.info("Note: --pcd is ignored for -GBLUP d/ad.")

    # ------------------------------------------------------------------
    # Load genotype
    # ------------------------------------------------------------------
    gsrc = os.path.basename(str(gfile).rstrip("/\\")) or str(gfile)
    use_packed_lmm = bool(gfile_is_plink and len(methods) > 0)
    ldprune_spec = typing.cast(
        _GsLdPruneSpec | None,
        getattr(args, "ldprune_spec", None),
    )
    rrblup_only_mode = bool(args.rrBLUP) and all(str(m) == "rrBLUP" for m in methods)
    packed_lazy_main_requested = bool(
        rrblup_only_mode
        and gfile_is_plink
        and (args.hash_dim is None)
        and (ldprune_spec is None)
    )
    preprocess_qc_requested = bool(
        (args.hash_dim is not None) or (ldprune_spec is not None)
    )
    packed_lmm_ctx: dict[str, typing.Any] | None = None
    packed_qc_baseline_ctx: dict[str, typing.Any] | None = None
    geno_is_memmap = False
    rrblup_aux_packed_ctx_requested = bool(
        bool(args.rrBLUP)
        and (str(rr_solver_mode) in {"pcg", "auto"})
        and gfile_is_plink
        and (not use_packed_lmm)
        and (args.hash_dim is None)
        and (ldprune_spec is None)
    )
    rrblup_aux_packed_ctx: dict[str, typing.Any] | None = None
    rrblup_aux_dense_to_packed_idx: np.ndarray | None = None
    if use_packed_lmm:
        with CliStatus(f"Loading genotype from {gsrc}...", enabled=use_spinner) as task:
            try:
                sample_ids, packed_lmm_ctx = _load_plink_packed_for_lmm(
                    str(gfile),
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                    filter_mode=("lazy" if packed_lazy_main_requested else "compact"),
                )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            n = int(sample_ids.shape[0])
            m = (
                _packed_ctx_active_rows(packed_lmm_ctx)
                if packed_lmm_ctx is not None
                else 0
            )
            task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m}, packed)")
        _emit_packed_load_debug(
            packed_lmm_ctx,
            label="main",
            enabled=bool(debug_mode),
        )
        if preprocess_qc_requested:
            packed_qc_baseline_ctx = _snapshot_packed_ctx_for_qc(packed_lmm_ctx)
        if ldprune_spec is not None:
            if use_spinner:
                t_ld = time.monotonic()
                try:
                    packed_lmm_ctx = _run_indeterminate_task_with_progress_bar(
                        desc="LD prune",
                        enabled=True,
                        runner=lambda: _apply_packed_ld_prune_for_gs(
                            packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                            genotype_prefix=str(gfile),
                            spec=ldprune_spec,
                            n_jobs=max(1, int(args.thread)),
                        ),
                    )
                except Exception:
                    print_failure("LD prune ...Failed", force_color=True)
                    raise
                m_after = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
                msg = _compact_done_message(
                    "LD prune",
                    [f"kept={m_after}", f"spec={ldprune_spec.label()}"],
                )
                print_success(f"{msg} [{format_elapsed(time.monotonic() - t_ld)}]", force_color=True)
            else:
                with CliStatus("Applying LD prune...", enabled=use_spinner) as task:
                    try:
                        packed_lmm_ctx = _apply_packed_ld_prune_for_gs(
                            packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                            genotype_prefix=str(gfile),
                            spec=ldprune_spec,
                            n_jobs=max(1, int(args.thread)),
                        )
                    except Exception:
                        task.fail("LD prune ...Failed")
                        raise
                    m_after = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
                    task.complete(
                        _compact_done_message(
                            "LD prune",
                            [f"kept={m_after}", f"spec={ldprune_spec.label()}"],
                        )
                    )

        if preprocess_qc_requested:
            _report_hash_kernel_sanity_check(
                packed_ctx=packed_lmm_ctx,
                sample_ids=sample_ids,
                pheno=pheno,
                hash_dim=(int(args.hash_dim) if args.hash_dim is not None else None),
                hash_seed=int(args.hash_seed),
                hash_standardize=(not bool(args.hash_raw)),
                n_jobs=max(1, int(args.thread)),
                use_spinner=bool(use_spinner),
                logger=logger,
                baseline_packed_ctx=packed_qc_baseline_ctx,
                debug_mode=bool(debug_mode),
            )

        if args.hash_dim is not None:
            hash_dim = int(args.hash_dim)
            hash_std = not bool(args.hash_raw)
            if hash_gblup_only_mode:
                samples = np.asarray(sample_ids, dtype=str)
                geno = None
                geno_raw = None
                geno_add_raw = None
                geno_ml_raw = None
            else:
                if use_spinner:
                    hash_total = int(
                        np.asarray(
                            typing.cast(dict[str, typing.Any], packed_lmm_ctx)["packed"]
                        ).shape[0]
                    )

                    def _run_hash(
                        cb: typing.Callable[[int, int], None] | None,
                        pe: int,
                    ) -> tuple[np.ndarray, float, int]:
                        return _hash_packed_for_gs(
                            packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                            hash_dim=hash_dim,
                            hash_seed=int(args.hash_seed),
                            standardize=bool(hash_std),
                            n_jobs=max(1, int(args.thread)),
                            progress_callback=cb,
                            progress_every=pe,
                        )

                    t_hash = time.monotonic()
                    try:
                        geno_h, hash_scale, hash_kept = _run_hash_job_with_progress_bar(
                            desc="Hash",
                            total_hint=hash_total,
                            enabled=True,
                            runner=_run_hash,
                        )
                    except Exception:
                        print_failure("Hash ...Failed", force_color=True)
                        raise
                    msg = _compact_done_message(
                        "Hash",
                        [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                    )
                    print_success(f"{msg} [{format_elapsed(time.monotonic() - t_hash)}]", force_color=True)
                else:
                    with CliStatus(
                        f"Hash (k={hash_dim})...",
                        enabled=use_spinner,
                    ) as task:
                        try:
                            geno_h, hash_scale, hash_kept = _hash_packed_for_gs(
                                packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                hash_dim=hash_dim,
                                hash_seed=int(args.hash_seed),
                                standardize=bool(hash_std),
                                n_jobs=max(1, int(args.thread)),
                            )
                        except Exception:
                            task.fail("Hash ...Failed")
                            raise
                        task.complete(_compact_done_message(
                            "Hash",
                            [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                        ))

                samples = np.asarray(sample_ids, dtype=str)
                geno = np.ascontiguousarray(np.asarray(geno_h, dtype=np.float32))
                packed_lmm_ctx = None
                use_packed_lmm = False

                need_std_geno = bool(
                    any(m in {"rrBLUP", "BayesA", "BayesB", "BayesCpi"} for m in methods)
                )
                need_add_raw = _methods_need_additive_dense(methods)
                need_raw_geno = bool(need_add_raw or any(m in ml_methods for m in methods))
                geno_raw = (
                    np.asarray(geno, dtype=np.float32, copy=need_std_geno)
                    if need_raw_geno
                    else None
                )
                geno_add_raw = geno_raw if need_add_raw else None
                geno_ml_raw = geno_raw if any(m in ml_methods for m in methods) else None
                if need_std_geno:
                    geno = (geno - geno.mean(axis=1, keepdims=True)) / (
                        geno.std(axis=1, keepdims=True) + 1e-6
                    )
        else:
            samples = sample_ids
            geno = None
            geno_raw = None
            geno_add_raw = None
            geno_ml_raw = None
    elif gfile_is_plink and (args.hash_dim is not None):
        # Global hash path for non-packed-model sets:
        # load packed BED -> optional LD prune -> hash projection -> dense hashed matrix.
        with CliStatus(f"Loading genotype from {gsrc}...", enabled=use_spinner) as task:
            try:
                sample_ids, packed_lmm_ctx = _load_plink_packed_for_lmm(
                    str(gfile),
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                    filter_mode="compact",
                )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            n = int(sample_ids.shape[0])
            m = (
                _packed_ctx_active_rows(packed_lmm_ctx)
                if packed_lmm_ctx is not None
                else 0
            )
            task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m}, packed)")
        _emit_packed_load_debug(
            packed_lmm_ctx,
            label="hash",
            enabled=bool(debug_mode),
        )
        packed_qc_baseline_ctx = _snapshot_packed_ctx_for_qc(packed_lmm_ctx)

        if ldprune_spec is not None:
            _spec = typing.cast(_GsLdPruneSpec, ldprune_spec)
            if use_spinner:
                t_ld = time.monotonic()
                try:
                    packed_lmm_ctx = _run_indeterminate_task_with_progress_bar(
                        desc="LD prune",
                        enabled=True,
                        runner=lambda: _apply_packed_ld_prune_for_gs(
                            packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                            genotype_prefix=str(gfile),
                            spec=_spec,
                            n_jobs=max(1, int(args.thread)),
                        ),
                    )
                except Exception:
                    print_failure("LD prune ...Failed", force_color=True)
                    raise
                m_after = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
                msg = _compact_done_message(
                    "LD prune",
                    [f"kept={m_after}", f"spec={_spec.label()}"],
                )
                print_success(f"{msg} [{format_elapsed(time.monotonic() - t_ld)}]", force_color=True)
            else:
                with CliStatus("Applying LD prune...", enabled=use_spinner) as task:
                    try:
                        packed_lmm_ctx = _apply_packed_ld_prune_for_gs(
                            packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                            genotype_prefix=str(gfile),
                            spec=_spec,
                            n_jobs=max(1, int(args.thread)),
                        )
                    except Exception:
                        task.fail("LD prune ...Failed")
                        raise
                    m_after = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
                    task.complete(
                        _compact_done_message(
                            "LD prune",
                            [f"kept={m_after}", f"spec={_spec.label()}"],
                        )
                    )

        _report_hash_kernel_sanity_check(
            packed_ctx=packed_lmm_ctx,
            sample_ids=sample_ids,
            pheno=pheno,
            hash_dim=(int(args.hash_dim) if args.hash_dim is not None else None),
            hash_seed=int(args.hash_seed),
            hash_standardize=(not bool(args.hash_raw)),
            n_jobs=max(1, int(args.thread)),
            use_spinner=bool(use_spinner),
            logger=logger,
            baseline_packed_ctx=packed_qc_baseline_ctx,
            debug_mode=bool(debug_mode),
        )

        hash_dim = int(args.hash_dim)
        hash_std = not bool(args.hash_raw)
        if use_spinner:
            hash_total = int(
                np.asarray(
                    typing.cast(dict[str, typing.Any], packed_lmm_ctx)["packed"]
                ).shape[0]
            )

            def _run_hash(
                cb: typing.Callable[[int, int], None] | None,
                pe: int,
            ) -> tuple[np.ndarray, float, int]:
                return _hash_packed_for_gs(
                    packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                    hash_dim=hash_dim,
                    hash_seed=int(args.hash_seed),
                    standardize=bool(hash_std),
                    n_jobs=max(1, int(args.thread)),
                    progress_callback=cb,
                    progress_every=pe,
                )

            t_hash = time.monotonic()
            try:
                geno_h, hash_scale, hash_kept = _run_hash_job_with_progress_bar(
                    desc="Hash",
                    total_hint=hash_total,
                    enabled=True,
                    runner=_run_hash,
                )
            except Exception:
                print_failure("Hash ...Failed", force_color=True)
                raise
            msg = _compact_done_message(
                "Hash",
                [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
            )
            print_success(f"{msg} [{format_elapsed(time.monotonic() - t_hash)}]", force_color=True)
        else:
            with CliStatus(
                f"Hash (k={hash_dim})...",
                enabled=use_spinner,
            ) as task:
                try:
                    geno_h, hash_scale, hash_kept = _hash_packed_for_gs(
                        packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                        hash_dim=hash_dim,
                        hash_seed=int(args.hash_seed),
                        standardize=bool(hash_std),
                        n_jobs=max(1, int(args.thread)),
                    )
                except Exception:
                    task.fail("Hash ...Failed")
                    raise
                task.complete(_compact_done_message(
                    "Hash",
                    [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                ))

        samples = np.asarray(sample_ids, dtype=str)
        geno = np.ascontiguousarray(np.asarray(geno_h, dtype=np.float32))
        packed_lmm_ctx = None
        use_packed_lmm = False

        need_std_geno = bool(any(m in {"rrBLUP", "BayesA", "BayesB", "BayesCpi"} for m in methods))
        need_add_raw = _methods_need_additive_dense(methods)
        need_raw_geno = bool(need_add_raw or any(m in ml_methods for m in methods))
        geno_raw = (
            np.asarray(
                geno,
                dtype=np.float32,
                copy=bool(need_std_geno and (not geno_is_memmap)),
            )
            if need_raw_geno
            else None
        )
        geno_add_raw = geno_raw if need_add_raw else None
        geno_ml_raw = geno_raw if any(m in ml_methods for m in methods) else None
        if need_std_geno:
            geno = (geno - geno.mean(axis=1, keepdims=True)) / (
                geno.std(axis=1, keepdims=True) + 1e-6
            )
    elif gfile_is_plink and (ldprune_spec is not None):
        # For non-GBLUP/rrBLUP model sets, allow global LD prune by
        # pruning on packed BED first, then decoding once to dense matrix.
        with CliStatus(f"Loading genotype from {gsrc}...", enabled=use_spinner) as task:
            try:
                sample_ids, packed_lmm_ctx = _load_plink_packed_for_lmm(
                    str(gfile),
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                    filter_mode="compact",
                )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            n = int(sample_ids.shape[0])
            m = (
                _packed_ctx_active_rows(packed_lmm_ctx)
                if packed_lmm_ctx is not None
                else 0
            )
            task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m}, packed)")
        _emit_packed_load_debug(
            packed_lmm_ctx,
            label="ldprune",
            enabled=bool(debug_mode),
        )
        packed_qc_baseline_ctx = _snapshot_packed_ctx_for_qc(packed_lmm_ctx)

        _spec = typing.cast(_GsLdPruneSpec, ldprune_spec)
        if use_spinner:
            t_ld = time.monotonic()
            try:
                packed_lmm_ctx = _run_indeterminate_task_with_progress_bar(
                    desc="LD prune",
                    enabled=True,
                    runner=lambda: _apply_packed_ld_prune_for_gs(
                        packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                        genotype_prefix=str(gfile),
                        spec=_spec,
                        n_jobs=max(1, int(args.thread)),
                    ),
                )
            except Exception:
                print_failure("LD prune ...Failed", force_color=True)
                raise
            m_after = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
            msg = _compact_done_message(
                "LD prune",
                [f"kept={m_after}", f"spec={_spec.label()}"],
            )
            print_success(f"{msg} [{format_elapsed(time.monotonic() - t_ld)}]", force_color=True)
        else:
            with CliStatus("Applying LD prune...", enabled=use_spinner) as task:
                try:
                    packed_lmm_ctx = _apply_packed_ld_prune_for_gs(
                        packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                        genotype_prefix=str(gfile),
                        spec=_spec,
                        n_jobs=max(1, int(args.thread)),
                    )
                except Exception:
                    task.fail("LD prune ...Failed")
                    raise
                m_after = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
                task.complete(
                    _compact_done_message(
                        "LD prune",
                        [f"kept={m_after}", f"spec={_spec.label()}"],
                    )
                )

        _report_hash_kernel_sanity_check(
            packed_ctx=packed_lmm_ctx,
            sample_ids=sample_ids,
            pheno=pheno,
            hash_dim=None,
            hash_seed=int(args.hash_seed),
            hash_standardize=(not bool(args.hash_raw)),
            n_jobs=max(1, int(args.thread)),
            use_spinner=bool(use_spinner),
            logger=logger,
            baseline_packed_ctx=packed_qc_baseline_ctx,
            debug_mode=bool(debug_mode),
        )

        with CliStatus("Decoding packed genotype after LD prune...", enabled=use_spinner) as task:
            try:
                geno = _decode_packed_ctx_to_dense(
                    typing.cast(dict[str, typing.Any], packed_lmm_ctx)
                )
            except Exception:
                task.fail("Decoding packed genotype after LD prune ...Failed")
                raise
            task.complete(
                "Decoding packed genotype after LD prune ...Finished "
                f"(n={int(geno.shape[1])}, nSNP={int(geno.shape[0])})"
            )

        samples = np.asarray(sample_ids, dtype=str)
        packed_lmm_ctx = None
        use_packed_lmm = False

        need_std_geno = bool(any(m in {"rrBLUP", "BayesA", "BayesB", "BayesCpi"} for m in methods))
        need_add_raw = _methods_need_additive_dense(methods)
        need_raw_geno = bool(need_add_raw or any(m in ml_methods for m in methods))
        geno_raw = (
            np.asarray(geno, dtype=np.float32, copy=need_std_geno)
            if need_raw_geno
            else None
        )
        geno_add_raw = geno_raw if need_add_raw else None
        geno_ml_raw = geno_raw if any(m in ml_methods for m in methods) else None
        if need_std_geno:
            geno = (geno - geno.mean(axis=1, keepdims=True)) / (
                geno.std(axis=1, keepdims=True) + 1e-6
            )
    else:
        with CliStatus(f"Loading genotype from {gsrc}...", enabled=use_spinner) as task:
            try:
                if file_memmap_preferred:
                    sample_ids, geno = _build_memmap_cache_from_chunks(
                        genotype_path=str(gfile),
                        maf=float(args.maf),
                        missing_rate=float(args.geno),
                        cache_root=os.path.join(args.out, ".janusx_gs_memmap"),
                        chunk_size=50_000,
                    )
                    geno_is_memmap = isinstance(geno, np.memmap)
                else:
                    sample_ids, geno = _load_genotype_with_rust_gfreader(
                        gfile,
                        maf=args.maf,
                        missing_rate=args.geno,
                    )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            m, n = geno.shape
            if geno_is_memmap:
                task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m}, memmap)")
            else:
                task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m})")

        samples = sample_ids
        # GBLUP computes kinship-driven scaling inside MLMBLUP(kinship=1),
        # so we skip global z-score standardization here to avoid redundant work.
        need_std_geno = bool(any(m in {"rrBLUP", "BayesA", "BayesB", "BayesCpi"} for m in methods))
        need_add_raw = _methods_need_additive_dense(methods)
        need_raw_geno = bool(need_add_raw or any(m in ml_methods for m in methods))
        geno_raw = (
            np.asarray(geno, dtype=np.float32, copy=need_std_geno)
            if need_raw_geno
            else None
        )
        geno_add_raw = geno_raw if need_add_raw else None
        geno_ml_raw = geno_raw if any(m in ml_methods for m in methods) else None
        if need_std_geno and (not geno_is_memmap):
            geno = (geno - geno.mean(axis=1, keepdims=True)) / (
                geno.std(axis=1, keepdims=True) + 1e-6
            )  # standardization for marker-effect models

    if rrblup_aux_packed_ctx_requested:
        with CliStatus(
            f"Loading packed rrBLUP helper from {gsrc}...",
            enabled=use_spinner,
        ) as task:
            try:
                rrblup_aux_sample_ids, rrblup_aux_packed_ctx = _load_plink_packed_for_lmm(
                    str(gfile),
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                    filter_mode="lazy",
                )
            except Exception:
                task.fail("Loading packed rrBLUP helper ...Failed")
                raise
            dense_ids = np.asarray(samples, dtype=str).reshape(-1)
            packed_ids = np.asarray(rrblup_aux_sample_ids, dtype=str).reshape(-1)
            if int(dense_ids.shape[0]) != int(packed_ids.shape[0]):
                raise RuntimeError(
                    "Packed rrBLUP helper sample-size mismatch: "
                    f"dense n={int(dense_ids.shape[0])}, packed n={int(packed_ids.shape[0])}."
                )
            if np.array_equal(dense_ids, packed_ids):
                rrblup_aux_dense_to_packed_idx = np.arange(int(dense_ids.shape[0]), dtype=np.int64)
            else:
                packed_pos = {str(sid): int(i) for i, sid in enumerate(packed_ids)}
                mapped_idx = np.fromiter(
                    (packed_pos.get(str(sid), -1) for sid in dense_ids),
                    dtype=np.int64,
                    count=int(dense_ids.shape[0]),
                )
                if np.any(mapped_idx < 0):
                    missing_ids = dense_ids[mapped_idx < 0]
                    preview = ",".join([str(x) for x in missing_ids[:3]])
                    raise RuntimeError(
                        "Packed rrBLUP helper could not align sample IDs with dense genotype order. "
                        f"Missing examples: {preview}"
                    )
                rrblup_aux_dense_to_packed_idx = np.ascontiguousarray(mapped_idx, dtype=np.int64)
            m_aux = (
                _packed_ctx_active_rows(rrblup_aux_packed_ctx)
                if rrblup_aux_packed_ctx is not None
                else 0
            )
            task.complete(
                "Loading packed rrBLUP helper ...Finished "
                f"(n={int(packed_ids.shape[0])}, nSNP={m_aux})"
            )
        _emit_packed_load_debug(
            rrblup_aux_packed_ctx,
            label="rrblup_aux",
            enabled=bool(debug_mode),
        )

    # Cache for repeated trait train/test partitions in GBLUP-only hash kinship mode.
    # Key: (train_idx bytes, test_idx bytes) in genotype sample order.
    hash_kernel_cache: dict[tuple[bytes, bytes], tuple[np.ndarray, np.ndarray, int]] = {}
    if (
        hash_gblup_only_mode
        and bool(use_packed_lmm)
        and (packed_lmm_ctx is not None)
        and (args.hash_dim is not None)
        and (_jxrs is not None)
        and hasattr(_jxrs, "bed_packed_signed_hash_kernels_f64")
    ):
        # If all requested traits share the same train/test split and backend=kinship,
        # prebuild kernels once right after genotype loading.
        common_trainmask: np.ndarray | None = None
        shared_mask = True
        for _tname in pheno.columns:
            _p = pheno[_tname]
            _namark = _p.isna()
            _train_id_set = set(_p.index[~_namark])
            _trainmask = np.fromiter(
                (x in _train_id_set for x in samples),
                dtype=bool,
                count=len(samples),
            )
            if common_trainmask is None:
                common_trainmask = _trainmask
            elif not np.array_equal(common_trainmask, _trainmask):
                shared_mask = False
                break
        if shared_mask and common_trainmask is not None:
            train_idx_common = np.flatnonzero(common_trainmask).astype(np.int64, copy=False)
            test_idx_common = np.flatnonzero(~common_trainmask).astype(np.int64, copy=False)
            if int(train_idx_common.shape[0]) <= int(args.hash_dim):
                key_common = (
                    np.ascontiguousarray(train_idx_common, dtype=np.int64).tobytes(),
                    np.ascontiguousarray(test_idx_common, dtype=np.int64).tobytes(),
                )
                if key_common not in hash_kernel_cache:
                    hash_dim = int(args.hash_dim)
                    hash_std = not bool(args.hash_raw)
                    if use_spinner:
                        hash_total = int(
                            np.asarray(
                                typing.cast(dict[str, typing.Any], packed_lmm_ctx)["packed"]
                            ).shape[0]
                        )

                        def _run_hash_kernels_shared(
                            cb: typing.Callable[[int, int], None] | None,
                            pe: int,
                        ) -> tuple[np.ndarray, np.ndarray, int]:
                            return _hash_packed_kernels_for_gblup(
                                packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                hash_dim=hash_dim,
                                hash_seed=int(args.hash_seed),
                                standardize=bool(hash_std),
                                n_jobs=max(1, int(args.thread)),
                                train_sample_indices=np.ascontiguousarray(train_idx_common, dtype=np.int64),
                                test_sample_indices=np.ascontiguousarray(test_idx_common, dtype=np.int64),
                                progress_callback=cb,
                                progress_every=pe,
                            )

                        t_hash = time.monotonic()
                        try:
                            train_grm_c, test_train_grm_c, hash_kept_c = _run_hash_job_with_progress_bar(
                                desc="Hash",
                                total_hint=hash_total,
                                enabled=True,
                                runner=_run_hash_kernels_shared,
                            )
                        except Exception:
                            print_failure("Hash ...Failed", force_color=True)
                            raise
                        msg = _compact_done_message(
                            "Hash",
                            [f"k={hash_dim}", f"kept_snp={hash_kept_c}", "gram-only"],
                        )
                        print_success(f"{msg} [{format_elapsed(time.monotonic() - t_hash)}]", force_color=True)
                    else:
                        with CliStatus(
                            f"Hash (k={hash_dim})...",
                            enabled=use_spinner,
                        ) as task:
                            try:
                                train_grm_c, test_train_grm_c, hash_kept_c = _hash_packed_kernels_for_gblup(
                                    packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                    hash_dim=hash_dim,
                                    hash_seed=int(args.hash_seed),
                                    standardize=bool(hash_std),
                                    n_jobs=max(1, int(args.thread)),
                                    train_sample_indices=np.ascontiguousarray(train_idx_common, dtype=np.int64),
                                    test_sample_indices=np.ascontiguousarray(test_idx_common, dtype=np.int64),
                                )
                            except Exception:
                                task.fail("Hash ...Failed")
                                raise
                            task.complete(_compact_done_message(
                                "Hash",
                                [f"k={hash_dim}", f"kept_snp={hash_kept_c}", "gram-only"],
                            ))
                    hash_kernel_cache[key_common] = (
                        np.ascontiguousarray(train_grm_c, dtype=np.float64),
                        np.ascontiguousarray(test_train_grm_c, dtype=np.float64),
                        int(hash_kept_c),
                    )

    # ------------------------------------------------------------------
    # Genomic Selection for each phenotype
    # ------------------------------------------------------------------
    single_trait_mode = (pheno.shape[1] == 1)
    trait_order = [str(x) for x in list(pheno.columns)]
    gs_summary_rows: list[dict[str, typing.Any]] = []
    saved_result_paths: list[str] = []
    exported_model_artifacts: list[dict[str, typing.Any]] = []
    trait_method_summary: dict[str, dict[str, dict[str, typing.Any]]] = {}
    run_started_unix = float(time.time())
    top_trait_selected_method: dict[str, str] = {}
    top_trait_metric_scores: dict[str, dict[str, dict[str, float]]] = {}
    top_train_ids_by_trait: dict[str, np.ndarray] = {}
    top_train_true_by_trait: dict[str, np.ndarray] = {}
    top_train_oof_by_trait: dict[str, np.ndarray] = {}
    top_test_ids_by_trait: dict[str, np.ndarray] = {}
    top_test_pred_by_trait: dict[str, np.ndarray] = {}
    top_trait_model_state: dict[str, dict[str, typing.Any]] = {}
    top_model_payload_loaded: dict[str, typing.Any] | None = None
    top_bundle_trait_models: dict[str, dict[str, typing.Any]] = {}
    if loaded_bundle_payload is not None and _is_top_bundle_payload(loaded_bundle_payload):
        raw_top_state = loaded_bundle_payload.get("top_model_state", None)
        if isinstance(raw_top_state, dict) and len(raw_top_state) > 0:
            top_model_payload_loaded = dict(raw_top_state)
        raw_trait_models = loaded_bundle_payload.get("trait_models", None)
        if isinstance(raw_trait_models, dict):
            for tk, tv in raw_trait_models.items():
                if isinstance(tv, dict):
                    top_bundle_trait_models[str(tk)] = dict(tv)
    for trait_name in pheno.columns:
        t_trait = time.time()
        if _GS_DEBUG_STAGE:
            logger.info(f"[GS-DEBUG][{trait_name}] stage=trait_start")

        t_stage = time.time()
        p = pheno[trait_name]
        namark = p.isna()
        if _GS_DEBUG_STAGE:
            logger.info(
                f"[GS-DEBUG][{trait_name}] stage=build_na_mask "
                f"elapsed={time.time() - t_stage:.3f}s"
            )

        t_stage = time.time()
        train_id_set = set(p.index[~namark])
        trainmask = np.fromiter((x in train_id_set for x in samples), dtype=bool, count=len(samples))
        testmask = ~trainmask
        if _GS_DEBUG_STAGE:
            logger.info(
                f"[GS-DEBUG][{trait_name}] stage=build_train_test_mask "
                f"train={int(np.sum(trainmask))} test={int(np.sum(testmask))} "
                f"elapsed={time.time() - t_stage:.3f}s"
            )

        train_sample_idx = np.flatnonzero(trainmask).astype(np.int64, copy=False)
        test_sample_idx = np.flatnonzero(testmask).astype(np.int64, copy=False)
        train_snp = None
        test_snp = None
        train_snp_add = None
        test_snp_add = None
        train_snp_ml = None
        test_snp_ml = None
        trait_use_packed_lmm = bool(use_packed_lmm)
        trait_packed_ctx: dict[str, typing.Any] | None = packed_lmm_ctx
        trait_packed_index_map: np.ndarray | None = None
        if (not trait_use_packed_lmm) and (trait_packed_ctx is None) and (rrblup_aux_packed_ctx is not None):
            # Mixed-method runs may still need packed BED for rrBLUP(pcg/auto).
            trait_packed_ctx = typing.cast(dict[str, typing.Any], rrblup_aux_packed_ctx)
            trait_packed_index_map = rrblup_aux_dense_to_packed_idx
        gblup_backend_label: str | None = None
        gblup_backend_n: int | None = None
        gblup_backend_m: int | None = None
        method_elapsed_offsets: dict[str, float] = {}
        if trait_use_packed_lmm:
            if _GS_DEBUG_STAGE:
                logger.info(
                    f"[GS-DEBUG][{trait_name}] stage=build_sample_indices "
                    f"train={int(train_sample_idx.shape[0])} test={int(test_sample_idx.shape[0])}"
                )
        else:
            t_stage = time.time()
            assert geno is not None
            train_snp = geno[:, trainmask]
            if _GS_DEBUG_STAGE:
                logger.info(
                    f"[GS-DEBUG][{trait_name}] stage=slice_train_snp "
                    f"shape={train_snp.shape} bytes={int(train_snp.nbytes)} "
                    f"elapsed={time.time() - t_stage:.3f}s"
                )

        t_stage = time.time()
        train_pheno = p.loc[samples[trainmask]].values.reshape(-1, 1)
        if _GS_DEBUG_STAGE:
            logger.info(
                f"[GS-DEBUG][{trait_name}] stage=align_train_pheno "
                f"shape={train_pheno.shape} elapsed={time.time() - t_stage:.3f}s"
            )

        if train_pheno.size == 0:
            logger.warning(f"No non-missing phenotypes for trait {trait_name}; skipped.")
            continue

        if hash_gblup_only_mode:
            gblup_prep_t0 = time.monotonic()
            if packed_lmm_ctx is None:
                raise RuntimeError(
                    "Internal error: packed genotype context missing for GBLUP-only hash mode."
                )
            hash_dim = int(args.hash_dim)
            hash_std = not bool(args.hash_raw)
            n_train_now = int(train_sample_idx.shape[0])
            gblup_backend_label = "ZTZ" if n_train_now > hash_dim else "kinship"
            gblup_backend_n = int(n_train_now)
            gblup_backend_m = int(hash_dim)
            trait_use_packed_lmm = False
            if gblup_backend_label == "kinship":
                used_gram_only = False
                if _jxrs is not None and hasattr(_jxrs, "bed_packed_signed_hash_kernels_f64"):
                    key_kin = (
                        np.ascontiguousarray(train_sample_idx, dtype=np.int64).tobytes(),
                        np.ascontiguousarray(test_sample_idx, dtype=np.int64).tobytes(),
                    )
                    cached_kin = hash_kernel_cache.get(key_kin)
                    if cached_kin is not None:
                        train_grm, test_train_grm, _hash_kept = cached_kin
                    else:
                        if use_spinner:
                            hash_total = int(
                                np.asarray(
                                    typing.cast(dict[str, typing.Any], packed_lmm_ctx)["packed"]
                                ).shape[0]
                            )

                            def _run_hash_kernels_trait(
                                cb: typing.Callable[[int, int], None] | None,
                                pe: int,
                            ) -> tuple[np.ndarray, np.ndarray, int]:
                                return _hash_packed_kernels_for_gblup(
                                    packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                    hash_dim=hash_dim,
                                    hash_seed=int(args.hash_seed),
                                    standardize=bool(hash_std),
                                    n_jobs=max(1, int(args.thread)),
                                    train_sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                    test_sample_indices=np.ascontiguousarray(test_sample_idx, dtype=np.int64),
                                    progress_callback=cb,
                                    progress_every=pe,
                                )

                            t_hash = time.monotonic()
                            try:
                                train_grm, test_train_grm, hash_kept = _run_hash_job_with_progress_bar(
                                    desc="Hash",
                                    total_hint=hash_total,
                                    enabled=True,
                                    runner=_run_hash_kernels_trait,
                                )
                            except Exception:
                                print_failure("Hash ...Failed", force_color=True)
                                raise
                            msg = _compact_done_message(
                                "Hash",
                                [f"k={hash_dim}", f"kept_snp={hash_kept}", "gram-only"],
                            )
                            print_success(
                                f"{msg} [{format_elapsed(time.monotonic() - t_hash)}]",
                                force_color=True,
                            )
                        else:
                            with CliStatus(
                                f"Hash (k={hash_dim})...",
                                enabled=use_spinner,
                            ) as task:
                                try:
                                    train_grm, test_train_grm, hash_kept = _hash_packed_kernels_for_gblup(
                                        packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                        hash_dim=hash_dim,
                                        hash_seed=int(args.hash_seed),
                                        standardize=bool(hash_std),
                                        n_jobs=max(1, int(args.thread)),
                                        train_sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                        test_sample_indices=np.ascontiguousarray(test_sample_idx, dtype=np.int64),
                                    )
                                except Exception:
                                    task.fail("Hash ...Failed")
                                    raise
                                task.complete(_compact_done_message(
                                    "Hash",
                                    [f"k={hash_dim}", f"kept_snp={hash_kept}", "gram-only"],
                                ))
                        train_grm = np.ascontiguousarray(train_grm, dtype=np.float64)
                        test_train_grm = np.ascontiguousarray(test_train_grm, dtype=np.float64)
                        hash_kernel_cache[key_kin] = (train_grm, test_train_grm, int(hash_kept))
                    train_snp = None
                    test_snp = None
                    trait_packed_ctx = {
                        "__gblup_train_grm__": np.ascontiguousarray(train_grm, dtype=np.float64),
                        "__gblup_test_train_grm__": np.ascontiguousarray(test_train_grm, dtype=np.float64),
                    }
                    used_gram_only = True

                if not used_gram_only:
                    if use_spinner:
                        hash_total = int(
                            np.asarray(
                                typing.cast(dict[str, typing.Any], packed_lmm_ctx)["packed"]
                            ).shape[0]
                        )

                        def _run_hash_train(
                            cb: typing.Callable[[int, int], None] | None,
                            pe: int,
                        ) -> tuple[np.ndarray, float, int]:
                            return _hash_packed_for_gs(
                                packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                hash_dim=hash_dim,
                                hash_seed=int(args.hash_seed),
                                standardize=bool(hash_std),
                                n_jobs=max(1, int(args.thread)),
                                sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                progress_callback=cb,
                                progress_every=pe,
                            )

                        def _run_hash_test(
                            cb: typing.Callable[[int, int], None] | None,
                            pe: int,
                        ) -> tuple[np.ndarray, float, int]:
                            return _hash_packed_for_gs(
                                packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                hash_dim=hash_dim,
                                hash_seed=int(args.hash_seed),
                                standardize=bool(hash_std),
                                n_jobs=max(1, int(args.thread)),
                                sample_indices=np.ascontiguousarray(test_sample_idx, dtype=np.int64),
                                progress_callback=cb,
                                progress_every=pe,
                            )

                        t_hash = time.monotonic()
                        try:
                            z_train, hash_scale, hash_kept = _run_hash_job_with_progress_bar(
                                desc="Hash(train)",
                                total_hint=hash_total,
                                enabled=True,
                                runner=_run_hash_train,
                            )
                            if int(test_sample_idx.shape[0]) > 0:
                                z_test, _scale_t, _kept_t = _run_hash_job_with_progress_bar(
                                    desc="Hash(test)",
                                    total_hint=hash_total,
                                    enabled=True,
                                    runner=_run_hash_test,
                                )
                            else:
                                z_test = np.zeros((hash_dim, 0), dtype=np.float32)
                        except Exception:
                            print_failure("Hash ...Failed", force_color=True)
                            raise
                        msg = _compact_done_message(
                            "Hash",
                            [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                        )
                        print_success(
                            f"{msg} [{format_elapsed(time.monotonic() - t_hash)}]",
                            force_color=True,
                        )
                    else:
                        with CliStatus(
                            f"Hash (k={hash_dim})...",
                            enabled=use_spinner,
                        ) as task:
                            try:
                                z_train, hash_scale, hash_kept = _hash_packed_for_gs(
                                    packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                    hash_dim=hash_dim,
                                    hash_seed=int(args.hash_seed),
                                    standardize=bool(hash_std),
                                    n_jobs=max(1, int(args.thread)),
                                    sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                )
                                if int(test_sample_idx.shape[0]) > 0:
                                    z_test, _scale_t, _kept_t = _hash_packed_for_gs(
                                        packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                        hash_dim=hash_dim,
                                        hash_seed=int(args.hash_seed),
                                        standardize=bool(hash_std),
                                        n_jobs=max(1, int(args.thread)),
                                        sample_indices=np.ascontiguousarray(test_sample_idx, dtype=np.int64),
                                    )
                                else:
                                    z_test = np.zeros((hash_dim, 0), dtype=np.float32)
                            except Exception:
                                task.fail("Hash ...Failed")
                                raise
                            task.complete(_compact_done_message(
                                "Hash",
                                [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                            ))
                    with CliStatus(
                        "Building hashed GBLUP kinship kernels...",
                        enabled=use_spinner,
                    ) as task:
                        try:
                            train_grm = np.asarray(
                                _QK2_GRM(np.asarray(z_train, dtype=np.float32), log=False),
                                dtype=np.float64,
                            )
                            if int(z_test.shape[1]) > 0:
                                test_train_grm = _build_gblup_test_train_cross_from_markers(
                                    test_snp=np.asarray(z_test, dtype=np.float32),
                                    train_snp=np.asarray(z_train, dtype=np.float32),
                                )
                            else:
                                test_train_grm = np.zeros((0, int(z_train.shape[1])), dtype=np.float64)
                        except Exception:
                            task.fail("Building hashed GBLUP kinship kernels ...Failed")
                            raise
                        task.complete("Building hashed GBLUP kinship kernels ...Finished")
                    train_snp = None
                    test_snp = None
                    trait_packed_ctx = {
                        "__gblup_train_grm__": np.ascontiguousarray(train_grm, dtype=np.float64),
                        "__gblup_test_train_grm__": np.ascontiguousarray(test_train_grm, dtype=np.float64),
                    }
            else:
                used_stream_ztz = False
                if (
                    cv_enabled
                    and _jxrs is not None
                    and hasattr(_jxrs, "bed_packed_signed_hash_ztz_stats_f64")
                ):
                    # Compact hash+CV mode: keep packed payload and defer fold-level
                    # ZTZ stats + prediction to _run_method_task, avoiding dense Z build.
                    train_snp = None
                    test_snp = None
                    trait_packed_ctx = dict(
                        typing.cast(dict[str, typing.Any], packed_lmm_ctx)
                    )
                    trait_packed_ctx["__gblup_hash_cv_compact__"] = True
                    trait_packed_ctx["__gblup_hash_dim__"] = int(hash_dim)
                    trait_packed_ctx["__gblup_hash_seed__"] = int(args.hash_seed)
                    trait_packed_ctx["__gblup_hash_standardize__"] = bool(hash_std)
                    used_stream_ztz = True
                if (
                    (not cv_enabled)
                    and _jxrs is not None
                    and hasattr(_jxrs, "bed_packed_signed_hash_ztz_stats_f64")
                ):
                    if use_spinner:
                        hash_total = int(
                            np.asarray(
                                typing.cast(dict[str, typing.Any], packed_lmm_ctx)["packed"]
                            ).shape[0]
                        )

                        def _run_hash_ztz(
                            cb: typing.Callable[[int, int], None] | None,
                            pe: int,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
                            return _hash_packed_ztz_stats_for_gblup(
                                packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                hash_dim=hash_dim,
                                hash_seed=int(args.hash_seed),
                                standardize=bool(hash_std),
                                n_jobs=max(1, int(args.thread)),
                                train_sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                y_train=np.asarray(train_pheno, dtype=np.float64).reshape(-1),
                                progress_callback=cb,
                                progress_every=pe,
                            )

                        t_hash = time.monotonic()
                        try:
                            gram, row_sum, row_sq_sum, zy_vec, hash_scale, hash_kept = _run_hash_job_with_progress_bar(
                                desc="Hash",
                                total_hint=hash_total,
                                enabled=True,
                                runner=_run_hash_ztz,
                            )
                        except Exception:
                            print_failure("Hash ...Failed", force_color=True)
                            raise
                        msg = _compact_done_message(
                            "Hash",
                            [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                        )
                        print_success(
                            f"{msg} [{format_elapsed(time.monotonic() - t_hash)}]",
                            force_color=True,
                        )
                    else:
                        with CliStatus(
                            f"Hash (k={hash_dim})...",
                            enabled=use_spinner,
                        ) as task:
                            try:
                                gram, row_sum, row_sq_sum, zy_vec, hash_scale, hash_kept = (
                                    _hash_packed_ztz_stats_for_gblup(
                                        packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                        hash_dim=hash_dim,
                                        hash_seed=int(args.hash_seed),
                                        standardize=bool(hash_std),
                                        n_jobs=max(1, int(args.thread)),
                                        train_sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                        y_train=np.asarray(train_pheno, dtype=np.float64).reshape(-1),
                                    )
                                )
                            except Exception:
                                task.fail("Hash ...Failed")
                                raise
                            task.complete(_compact_done_message(
                                "Hash",
                                [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                            ))

                    try:
                        fit_fast = _fit_gblup_reml_from_hash_ztz_stats(
                            gram=np.asarray(gram, dtype=np.float64),
                            row_sum=np.asarray(row_sum, dtype=np.float64),
                            row_sq_sum=np.asarray(row_sq_sum, dtype=np.float64),
                            zy=np.asarray(zy_vec, dtype=np.float64),
                            n_train=int(train_sample_idx.shape[0]),
                        )
                        test_pred_fast = _predict_hashed_gblup_test_from_compact(
                            packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                            hash_dim=hash_dim,
                            hash_seed=int(args.hash_seed),
                            standardize=bool(hash_std),
                            n_jobs=max(1, int(args.thread)),
                            test_sample_indices=np.ascontiguousarray(test_sample_idx, dtype=np.int64),
                            beta=float(fit_fast["beta"]),
                            m_mean=np.asarray(fit_fast["m_mean"], dtype=np.float64),
                            m_var_sum=float(fit_fast["m_var_sum"]),
                            m_alpha=np.asarray(fit_fast["m_alpha"], dtype=np.float64),
                            alpha_sum=float(fit_fast["alpha_sum"]),
                            mean_sq=float(fit_fast["mean_sq"]),
                            mean_malpha=float(fit_fast["mean_malpha"]),
                        )
                    except Exception:
                        if bool(debug_mode):
                            print_failure("GBLUP compact solve ...Failed", force_color=True)
                        raise

                    train_snp = None
                    test_snp = None
                    trait_packed_ctx = {
                        "__gblup_hash_ztz_test_pred__": np.ascontiguousarray(
                            test_pred_fast, dtype=np.float64
                        ),
                        "__gblup_hash_ztz_pve__": float(fit_fast["pve"]),
                    }
                    used_stream_ztz = True

                if not used_stream_ztz:
                    if use_spinner:
                        hash_total = int(
                            np.asarray(
                                typing.cast(dict[str, typing.Any], packed_lmm_ctx)["packed"]
                            ).shape[0]
                        )

                        def _run_hash_train(
                            cb: typing.Callable[[int, int], None] | None,
                            pe: int,
                        ) -> tuple[np.ndarray, float, int]:
                            return _hash_packed_for_gs(
                                packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                hash_dim=hash_dim,
                                hash_seed=int(args.hash_seed),
                                standardize=bool(hash_std),
                                n_jobs=max(1, int(args.thread)),
                                sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                progress_callback=cb,
                                progress_every=pe,
                            )

                        def _run_hash_test(
                            cb: typing.Callable[[int, int], None] | None,
                            pe: int,
                        ) -> tuple[np.ndarray, float, int]:
                            return _hash_packed_for_gs(
                                packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                hash_dim=hash_dim,
                                hash_seed=int(args.hash_seed),
                                standardize=bool(hash_std),
                                n_jobs=max(1, int(args.thread)),
                                sample_indices=np.ascontiguousarray(test_sample_idx, dtype=np.int64),
                                progress_callback=cb,
                                progress_every=pe,
                            )

                        t_hash = time.monotonic()
                        try:
                            z_train, hash_scale, hash_kept = _run_hash_job_with_progress_bar(
                                desc="Hash(train)",
                                total_hint=hash_total,
                                enabled=True,
                                runner=_run_hash_train,
                            )
                            if int(test_sample_idx.shape[0]) > 0:
                                z_test, _scale_t, _kept_t = _run_hash_job_with_progress_bar(
                                    desc="Hash(test)",
                                    total_hint=hash_total,
                                    enabled=True,
                                    runner=_run_hash_test,
                                )
                            else:
                                z_test = np.zeros((hash_dim, 0), dtype=np.float32)
                        except Exception:
                            print_failure("Hash ...Failed", force_color=True)
                            raise
                        msg = _compact_done_message(
                            "Hash",
                            [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                        )
                        print_success(
                            f"{msg} [{format_elapsed(time.monotonic() - t_hash)}]",
                            force_color=True,
                        )
                    else:
                        with CliStatus(
                            f"Hash (k={hash_dim})...",
                            enabled=use_spinner,
                        ) as task:
                            try:
                                z_train, hash_scale, hash_kept = _hash_packed_for_gs(
                                    packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                    hash_dim=hash_dim,
                                    hash_seed=int(args.hash_seed),
                                    standardize=bool(hash_std),
                                    n_jobs=max(1, int(args.thread)),
                                    sample_indices=np.ascontiguousarray(train_sample_idx, dtype=np.int64),
                                )
                                if int(test_sample_idx.shape[0]) > 0:
                                    z_test, _scale_t, _kept_t = _hash_packed_for_gs(
                                        packed_ctx=typing.cast(dict[str, typing.Any], packed_lmm_ctx),
                                        hash_dim=hash_dim,
                                        hash_seed=int(args.hash_seed),
                                        standardize=bool(hash_std),
                                        n_jobs=max(1, int(args.thread)),
                                        sample_indices=np.ascontiguousarray(test_sample_idx, dtype=np.int64),
                                    )
                                else:
                                    z_test = np.zeros((hash_dim, 0), dtype=np.float32)
                            except Exception:
                                task.fail("Hash ...Failed")
                                raise
                            task.complete(_compact_done_message(
                                "Hash",
                                [f"k={hash_dim}", f"kept_snp={hash_kept}", f"scale={hash_scale:.6g}"],
                            ))
                    train_snp = np.ascontiguousarray(np.asarray(z_train, dtype=np.float32))
                    test_snp = np.ascontiguousarray(np.asarray(z_test, dtype=np.float32))
                    trait_packed_ctx = None
            method_elapsed_offsets["GBLUP"] = max(0.0, time.monotonic() - gblup_prep_t0)

        # 5-fold cross-validation on training population
        cv_splits = None
        if cv_enabled:
            cv_splits = build_cv_splits(
                n_samples=int(train_sample_idx.shape[0]),
                n_splits=int(args.cv),
                seed=42,
            )

        if (not trait_use_packed_lmm) and (test_snp is None) and (geno is not None):
            t_stage = time.time()
            test_snp = geno[:, testmask]
            if _GS_DEBUG_STAGE:
                logger.info(
                    f"[GS-DEBUG][{trait_name}] stage=slice_test_snp "
                    f"shape={test_snp.shape} bytes={int(test_snp.nbytes)} "
                    f"elapsed={time.time() - t_stage:.3f}s"
                )
            if _methods_need_additive_dense(methods):
                if geno_add_raw is None:
                    raise RuntimeError(
                        "Internal error: additive genotype matrix for GBLUP(d/ad) is missing."
                    )
                train_snp_add = geno_add_raw[:, trainmask]
                test_snp_add = geno_add_raw[:, testmask]
            if any(m in ml_methods for m in methods):
                if geno_ml_raw is None:
                    raise RuntimeError("Internal error: raw genotype matrix for MLGS is missing.")
                train_snp_ml = geno_ml_raw[:, trainmask]
                test_snp_ml = geno_ml_raw[:, testmask]
        if single_trait_mode and (not trait_use_packed_lmm) and (geno is not None):
            # In single-trait runs, release full matrices once train/test views are materialized.
            # This avoids carrying an extra full-genotype buffer through model fitting.
            geno = np.empty((0, 0), dtype=np.float32)
            geno_raw = None
            geno_add_raw = None
            geno_ml_raw = None
            gc.collect()
            if _GS_DEBUG_STAGE:
                logger.info(f"[GS-DEBUG][{trait_name}] stage=release_full_geno done=1")

        if hash_gblup_only_mode:
            eff_snp = int(args.hash_dim)
        else:
            eff_snp = (
                _packed_ctx_active_rows(typing.cast(dict[str, typing.Any], trait_packed_ctx))
                if (trait_use_packed_lmm and trait_packed_ctx is not None)
                else int(train_snp.shape[0] if train_snp is not None else 0)
            )
        if gblup_backend_label is None and (_GBLUP_METHOD_ADD in methods) and (train_snp is not None):
            gblup_backend_n = int(train_pheno.shape[0])
            gblup_backend_m = int(train_snp.shape[0])
            gblup_backend_label = "ZTZ" if gblup_backend_n > gblup_backend_m else "kinship"

        sep_w = max(40, min(80, _terminal_columns()))
        gs_output.emit_trait_header(
            logger,
            trait_name=str(trait_name),
            train_size=int(np.sum(trainmask)),
            test_size=int(np.sum(testmask)),
            eff_snp=int(eff_snp),
            sep_width=int(sep_w),
            debug_mode=bool(debug_mode),
            gblup_backend_label=gblup_backend_label,
            gblup_backend_n=gblup_backend_n,
            gblup_backend_m=gblup_backend_m,
        )
        bundle_mode = bool(
            model_mode
            and (loaded_bundle_payload is not None)
            and _is_top_bundle_payload(loaded_bundle_payload)
        )
        trait_methods = list(methods)
        trait_bundle_entry: dict[str, typing.Any] | None = None
        if bundle_mode:
            trait_bundle_entry = top_bundle_trait_models.get(str(trait_name), None)
            if trait_bundle_entry is None:
                logger.warning(
                    f"TOP bundle does not contain trait={trait_name}; skipped."
                )
                continue
            sel_from_bundle = str(trait_bundle_entry.get("selected_method", "")).strip()
            if sel_from_bundle == "":
                logger.warning(
                    f"TOP bundle trait entry missing selected_method for trait={trait_name}; skipped."
                )
                continue
            trait_methods = [sel_from_bundle]

        method_display_map = {str(m): _method_display_name(str(m)) for m in trait_methods}
        out_tsv = f"{outprefix}.{trait_name}.gs.tsv"
        out_effect_merged = f"{outprefix}.{trait_name}.gs.effect"
        expected_test_n = int(np.sum(testmask))
        pred_by_method: dict[str, np.ndarray] = {}
        trait_method_summary[str(trait_name)] = {}
        trait_effect_tables: dict[str, pd.DataFrame] = {}
        trait_effect_meta_by_method: dict[str, dict[str, typing.Any]] = {}

        def _emit_method_header(method_key: str) -> None:
            m_key = str(method_key)
            m_disp = method_display_map.get(m_key, _method_display_name(m_key))
            gs_output.emit_method_header(logger, m_disp)

        def _emit_method_block_and_artifacts(
            method_key: str,
            res_obj: dict[str, typing.Any],
        ) -> None:
            m_key = str(method_key)
            m_disp = method_display_map.get(m_key, _method_display_name(m_key))

            cv_sec = float(res_obj.get("elapsed_cv_sec", np.nan))
            fit_sec = float(res_obj.get("elapsed_fit_sec", np.nan))
            pred_sec = float(res_obj.get("elapsed_predict_sec", np.nan))
            stage_lines_emitted = bool(res_obj.get("__stage_lines_emitted", False))
            cv_line_emitted = bool(res_obj.get("__cv_line_emitted", False))
            gs_output.emit_method_stage_lines(
                logger,
                cv_folds=(
                    (int(len(cv_splits)) if cv_splits is not None else None)
                    if (not cv_line_emitted)
                    else None
                ),
                cv_elapsed_sec=float(cv_sec),
                fit_elapsed_sec=float(fit_sec),
                predict_elapsed_sec=float(pred_sec),
                include_fit=bool((args.model is None) and (not stage_lines_emitted)),
                include_predict=bool(not stage_lines_emitted),
                terminal_columns=int(_terminal_columns()),
                format_elapsed=format_elapsed,
                log_success=log_success,
            )

            cv_mode_row = str(res_obj.get("cv_mode", "train_cv")).strip().lower()
            gblup_variance_component_label: str | None = None
            if _is_gblup_method(m_key):
                gblup_mode = _gblup_method_kernel_mode(m_key)
                if gblup_mode == "ad":
                    # Additive+dominance path uses explicit dual-kernel REML.
                    gblup_variance_component_label = "REML/GRM"
                else:
                    gblup_state = dict(
                        typing.cast(
                            dict[str, typing.Any] | None,
                            res_obj.get("gblup_final_state"),
                        )
                        or {}
                    )
                    vc_label = str(gblup_state.get("variance_component", "")).strip()
                    vc_path = str(gblup_state.get("variance_component_path", "")).strip().lower()
                    eigh_backend = str(gblup_state.get("eigh_backend", "")).strip().lower()
                    if vc_label != "":
                        gblup_variance_component_label = vc_label
                    elif vc_path in {"fast", "marker_fast"} or eigh_backend.startswith("marker_fast"):
                        gblup_variance_component_label = "REML/FaST"
                    elif vc_path == "grm" or eigh_backend != "":
                        gblup_variance_component_label = "REML/GRM"
                    else:
                        m_eff = 0
                        try:
                            if _looks_like_packed_payload(trait_packed_ctx):
                                m_eff = int(
                                    np.asarray(
                                        typing.cast(dict[str, typing.Any], trait_packed_ctx)["packed"]
                                    ).shape[0]
                                )
                            elif train_snp is not None:
                                m_eff = int(np.asarray(train_snp).shape[0])
                        except Exception:
                            m_eff = 0
                        n_eff = int(np.asarray(train_pheno).reshape(-1).shape[0])
                        gblup_variance_component_label = (
                            "REML/FaST"
                            if (m_eff > 0 and n_eff > m_eff)
                            else "REML/GRM"
                        )
            gs_output.emit_method_detail_lines(
                logger,
                cv_mode=cv_mode_row,
                is_gblup=bool(_is_gblup_method(m_key)),
                detail_rows=typing.cast(
                    list[tuple[str, str]] | None,
                    res_obj.get("method_detail_rows"),
                ),
                gblup_kernel_label=(
                    _gblup_mode_label(_gblup_method_kernel_mode(m_key))
                    if _is_gblup_method(m_key)
                    else None
                ),
                gblup_variance_component=gblup_variance_component_label,
                gblup_backend_label=(
                    "packed active" if _looks_like_packed_payload(trait_packed_ctx) else "dense"
                ),
                gblup_pve=float(res_obj.get("pve_final", np.nan)),
            )

            model_out_export: str | None = None
            effect_out_export: str | None = None
            effect_meta: dict[str, typing.Any] | None = None
            model_export_meta: dict[str, typing.Any] | None = None
            can_model_export = bool(_is_jxmodel_export_supported(m_key))
            can_effect_export = bool(_is_effect_export_supported(m_key))
            if (args.model is None) and (not top_enabled) and can_effect_export:
                state_raw = res_obj.get("model_state", None)
                if isinstance(state_raw, dict) and len(state_raw) > 0:
                    packed_ctx_for_export = (
                        typing.cast(dict[str, typing.Any], trait_packed_ctx)
                        if _looks_like_packed_payload(trait_packed_ctx)
                        else None
                    )
                    if can_model_export:
                        state_export, model_export_meta = _prepare_model_state_for_export_raw_012(
                            model_state=dict(state_raw),
                            packed_ctx=packed_ctx_for_export,
                        )
                    else:
                        state_export = dict(state_raw)
                        model_export_meta = {}
                    if can_model_export:
                        payload = {
                            "format": _JXMODEL_FORMAT,
                            "method": str(m_key),
                            "method_display": str(m_disp),
                            "trait": str(trait_name),
                            "created_at_unix": float(time.time()),
                            "pve_final": float(res_obj.get("pve_final", np.nan)),
                            "model_state": dict(state_export),
                            "export_scale_meta": dict(model_export_meta or {}),
                        }
                        model_out = os.path.join(gs_model_dir, f"{trait_name}.{m_key}.jxmodel")
                        _save_jxmodel(model_out, payload)
                        saved_result_paths.append(str(model_out))
                        model_out_export = str(model_out)
                    try:
                        fallback_marker_count = None
                        if _looks_like_packed_payload(trait_packed_ctx):
                            fallback_marker_count = int(
                                np.asarray(
                                    typing.cast(dict[str, typing.Any], trait_packed_ctx)["packed"]
                                ).shape[0]
                            )
                        elif train_snp is not None:
                            fallback_marker_count = int(np.asarray(train_snp).shape[0])
                        effect_table, effect_meta = _build_method_effect_table(
                            model_state=dict(state_export),
                            packed_ctx=packed_ctx_for_export,
                            genotype_prefix_hint=(str(gfile) if _is_plink_prefix(_strip_plink_suffix(str(gfile))) else None),
                            fallback_marker_count=fallback_marker_count,
                        )
                        trait_effect_tables[str(m_key)] = pd.DataFrame(effect_table)
                        trait_effect_meta_by_method[str(m_key)] = dict(effect_meta or {})
                        effect_out_export = str(out_effect_merged)
                    except Exception as ex:
                        logger.warning(
                            f"Effect table build failed for trait={trait_name}, method={m_key}: {ex}"
                        )
                        effect_meta = {
                            "effect_file": str(out_effect_merged),
                            "rows": 0,
                            "effect_kind": "unknown",
                            "effect_column": "unknown",
                            "effect_source": "export_failed",
                            "beta_available": False,
                            "beta_source": "export_failed",
                            "beta_non_nan_rows": 0,
                            "beta_scale": "unknown",
                            "metadata_source": "unknown",
                            "notes": [f"export_failed: {ex}"],
                        }
                        trait_effect_meta_by_method[str(m_key)] = dict(effect_meta)
                        effect_out_export = str(out_effect_merged)
                    exported_model_artifacts.append(
                        {
                            "trait": str(trait_name),
                            "method": str(m_key),
                            "model_file": str(model_out_export or ""),
                            "effect_file": str(effect_out_export or ""),
                            "model_export_meta": dict(model_export_meta or {}),
                            "effect_meta": dict(effect_meta or {}),
                        }
                    )
                else:
                    logger.warning(
                        f"Skip effect/model export for method={m_key}: fitted model state is unavailable."
                    )

            pred_arr = np.asarray(res_obj["test_pred"], dtype=float).reshape(-1, 1)
            n_pred = int(pred_arr.shape[0])
            if n_pred != expected_test_n:
                if (n_pred == 0) and (expected_test_n > 0):
                    logger.warning(
                        f"{m_key} returned empty test predictions while expected n_test={expected_test_n}; "
                        "filling with NaN."
                    )
                    pred_arr = np.full((expected_test_n, 1), np.nan, dtype=float)
                else:
                    raise RuntimeError(
                        f"Prediction length mismatch for method={m_key}: "
                        f"got n_pred={n_pred}, expected n_test={expected_test_n}."
                    )
            pred_by_method[m_key] = pred_arr
            ordered_keys = [str(mm) for mm in trait_methods if str(mm) in pred_by_method]
            outpred_partial = pd.DataFrame(
                np.concatenate([pred_by_method[k] for k in ordered_keys], axis=1),
                columns=[method_display_map.get(k, k) for k in ordered_keys],
                index=samples[testmask],
            )
            outpred_partial.to_csv(out_tsv, sep="\t", float_format="%.4f")

            detail_rows_safe: list[dict[str, str]] = []
            for row in list(
                typing.cast(
                    list[tuple[str, str]] | None,
                    res_obj.get("method_detail_rows"),
                )
                or []
            ):
                if len(row) >= 2:
                    detail_rows_safe.append(
                        {"name": str(row[0]), "value": str(row[1])}
                    )
            fold_rows_safe: list[dict[str, typing.Any]] = []
            for row in list(res_obj.get("fold_rows", []) or []):
                if len(row) < 7:
                    continue
                fold_rows_safe.append(
                    {
                        "method": str(row[0]),
                        "fold": int(row[1]),
                        "pearson": float(row[2]),
                        "spearman": float(row[3]),
                        "r2": float(row[4]),
                        "pve": float(row[5]),
                        "elapsed_sec": float(row[6]),
                    }
                )
            rrblup_pcg_traces_safe: list[dict[str, typing.Any]] = []
            for trace_row in list(res_obj.get("rrblup_pcg_trace_rows", []) or []):
                trace_items_safe: list[dict[str, typing.Any]] = []
                for rec in list(trace_row.get("trace", []) or []):
                    trace_items_safe.append(
                        {
                            "iter": int(max(0, int(rec.get("iter", 0) or 0))),
                            "total": int(max(1, int(rec.get("total", 1) or 1))),
                            "rel_res": float(rec.get("rel_res", np.nan)),
                        }
                    )
                rrblup_pcg_traces_safe.append(
                    {
                        "phase": str(trace_row.get("phase", "")),
                        "phase_label": str(trace_row.get("phase_label", "")),
                        "fold": (
                            None
                            if trace_row.get("fold", None) is None
                            else int(trace_row.get("fold"))
                        ),
                        "solver": str(trace_row.get("solver", "")),
                        "converged": bool(trace_row.get("converged", False)),
                        "iters": int(max(0, int(trace_row.get("iters", 0) or 0))),
                        "max_iter": int(max(1, int(trace_row.get("max_iter", 1) or 1))),
                        "trace": trace_items_safe,
                    }
                )
            if str(m_key) == "rrBLUP" and len(rrblup_pcg_traces_safe) > 0:
                _log_file_only(f"[rrBLUP-PCG][trait={trait_name}][method={m_key}] residual_trace_begin")
                for trace_row in rrblup_pcg_traces_safe:
                    phase_label = str(trace_row.get("phase_label", "")).strip() or str(
                        trace_row.get("phase", "")
                    ).strip()
                    for rec in list(trace_row.get("trace", []) or []):
                        _log_file_only(
                            "[rrBLUP-PCG] "
                            f"trait={trait_name} method={m_key} phase={phase_label} "
                            f"iter={int(rec.get('iter', 0))}/{int(rec.get('total', 1))} "
                            f"rel_res={float(rec.get('rel_res', np.nan)):.12g}"
                        )
                _log_file_only(f"[rrBLUP-PCG][trait={trait_name}][method={m_key}] residual_trace_end")
            trait_method_summary[str(trait_name)][str(m_key)] = {
                "method_display": str(m_disp),
                "cv_mode": str(res_obj.get("cv_mode", "train_cv")),
                "elapsed_total_sec": float(res_obj.get("elapsed_total_sec", np.nan)),
                "elapsed_cv_sec": float(res_obj.get("elapsed_cv_sec", np.nan)),
                "elapsed_fit_sec": float(res_obj.get("elapsed_fit_sec", np.nan)),
                "elapsed_predict_sec": float(res_obj.get("elapsed_predict_sec", np.nan)),
                "pve_final": float(res_obj.get("pve_final", np.nan)),
                "gblup_final_state": dict(
                    typing.cast(dict[str, typing.Any] | None, res_obj.get("gblup_final_state")) or {}
                ),
                "rrblup_pve_final": dict(
                    typing.cast(dict[str, typing.Any] | None, res_obj.get("rrblup_pve_final")) or {}
                ),
                "rrblup_pcg_traces": rrblup_pcg_traces_safe,
                "method_details": detail_rows_safe,
                "fold_rows": fold_rows_safe,
                "model_file": (str(model_out_export) if model_out_export is not None else None),
                "effect_file": (str(effect_out_export) if effect_out_export is not None else None),
                "effect_meta": (dict(effect_meta) if isinstance(effect_meta, dict) else None),
            }

        method_train_sample_idx = np.ascontiguousarray(train_sample_idx, dtype=np.int64)
        method_test_sample_idx = np.ascontiguousarray(test_sample_idx, dtype=np.int64)
        if (
            (not trait_use_packed_lmm)
            and _looks_like_packed_payload(trait_packed_ctx)
            and (trait_packed_index_map is not None)
        ):
            method_train_sample_idx = np.ascontiguousarray(
                np.asarray(trait_packed_index_map[train_sample_idx], dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )
            method_test_sample_idx = np.ascontiguousarray(
                np.asarray(trait_packed_index_map[test_sample_idx], dtype=np.int64).reshape(-1),
                dtype=np.int64,
            )

        loaded_train_snp_ml = train_snp_ml
        loaded_test_snp_ml = test_snp_ml
        if model_mode and any(str(mm) in _ML_METHOD_MAP for mm in trait_methods):
            if loaded_train_snp_ml is None or loaded_test_snp_ml is None:
                if _looks_like_packed_payload(trait_packed_ctx):
                    packed_payload_local = typing.cast(dict[str, typing.Any], trait_packed_ctx)
                    loaded_train_snp_ml = _decode_packed_subset_to_dense_raw_f32(
                        packed_payload_local,
                        np.ascontiguousarray(
                            np.asarray(method_train_sample_idx, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        ),
                    )
                    loaded_test_snp_ml = _decode_packed_subset_to_dense_raw_f32(
                        packed_payload_local,
                        np.ascontiguousarray(
                            np.asarray(method_test_sample_idx, dtype=np.int64).reshape(-1),
                            dtype=np.int64,
                        ),
                    )
                elif train_snp is not None and test_snp is not None:
                    loaded_train_snp_ml = np.ascontiguousarray(
                        np.asarray(train_snp, dtype=np.float32),
                        dtype=np.float32,
                    )
                    loaded_test_snp_ml = np.ascontiguousarray(
                        np.asarray(test_snp, dtype=np.float32),
                        dtype=np.float32,
                    )

        t_stage = time.time()
        if model_mode:
            method_results = []
            for m in trait_methods:
                model_file = ""
                model_state: dict[str, typing.Any] | None = None
                if bundle_mode:
                    if trait_bundle_entry is None:
                        raise RuntimeError("Internal error: missing TOP bundle trait entry.")
                    m_bundle = str(trait_bundle_entry.get("selected_method", "")).strip()
                    if m_bundle != str(m):
                        raise ValueError(
                            f"TOP bundle trait method mismatch for {trait_name}: "
                            f"expected {m}, got {m_bundle}."
                        )
                    raw_state = trait_bundle_entry.get("model_state", None)
                    if not isinstance(raw_state, dict) or len(raw_state) == 0:
                        raise ValueError(
                            f"TOP bundle trait model_state is missing for trait={trait_name}, method={m}."
                        )
                    model_state = dict(raw_state)
                    model_file = str(loaded_bundle_file or args.model)
                else:
                    model_file = _resolve_jxmodel_file(
                        model_arg=str(args.model),
                        trait_name=str(trait_name),
                        method=str(m),
                        prefix_hint=str(args.prefix),
                    )
                    artifact = _load_jxmodel(model_file)
                    artifact_method = str(artifact.get("method", "")).strip()
                    if artifact_method != "" and artifact_method != str(m):
                        raise ValueError(
                            f"Loaded model mismatch: expected method={m}, got {artifact_method} in {model_file}."
                        )
                    raw_state = artifact.get("model_state", None)
                    if not isinstance(raw_state, dict) or len(raw_state) == 0:
                        raise ValueError(f"Loaded model artifact missing model_state: {model_file}")
                    model_state = dict(raw_state)
                res = _run_loaded_model_task(
                    method=str(m),
                    model_state=dict(model_state),
                    train_pheno=train_pheno,
                    train_snp=train_snp,
                    test_snp=test_snp,
                    train_snp_ml=loaded_train_snp_ml,
                    test_snp_ml=loaded_test_snp_ml,
                    packed_ctx=trait_packed_ctx if _looks_like_packed_payload(trait_packed_ctx) else None,
                    train_sample_indices=method_train_sample_idx,
                    test_sample_indices=method_test_sample_idx,
                    cv_splits=None,
                    limit_predtrain=args.limit_predtrain,
                )
                res["model_file"] = str(model_file)
                method_results.append(res)
                _emit_method_header(str(m))
                _emit_method_block_and_artifacts(str(m), res)
        else:
            method_results = _run_methods_parallel(
                methods=trait_methods,
                train_pheno=train_pheno,
                train_snp=train_snp,
                test_snp=test_snp,
                train_snp_add=train_snp_add,
                test_snp_add=test_snp_add,
                train_snp_ml=train_snp_ml,
                test_snp_ml=test_snp_ml,
                pca_dec=args.pcd,
                cv_splits=cv_splits,
                n_jobs=int(max(1, args.thread)),
                strict_cv=bool(args.strict_cv),
                force_fast=bool(args.force_fast),
                packed_ctx=trait_packed_ctx,
                train_sample_indices=method_train_sample_idx,
                test_sample_indices=method_test_sample_idx,
                train_sample_ids=np.asarray(samples[trainmask], dtype=str).reshape(-1),
                limit_predtrain=args.limit_predtrain,
                elapsed_offset_by_method=method_elapsed_offsets,
                rrblup_solver=rrblup_solver,
                rrblup_adamw_cfg=rrblup_adamw_cfg,
                bayes_auto_r2_cfg=bayes_auto_r2_cfg,
                emit_cv_progress_bar=True,
                emit_method_summary=False,
                on_method_start=_emit_method_header,
                on_method_complete=_emit_method_block_and_artifacts,
            )
        if _GS_DEBUG_STAGE:
            logger.info(
                f"[GS-DEBUG][{trait_name}] stage=run_methods_parallel "
                f"elapsed={time.time() - t_stage:.3f}s"
            )
        method_result_map = {str(x["method"]): x for x in method_results}
        trait_method_keys = {str(m) for m in trait_methods}
        merged_effect_written = False
        merged_effect_method_keys: list[str] = []
        if (args.model is None) and (not top_enabled) and (len(trait_effect_tables) > 0):
            try:
                merge_method_order: list[str] = []
                for m in trait_methods:
                    m_key = str(m)
                    tb = trait_effect_tables.get(m_key, None)
                    if tb is None:
                        continue
                    has_finite_effect = False
                    eff_col = _resolve_effect_col_from_table(tb)
                    if eff_col is not None:
                        try:
                            beta_vec = np.asarray(tb[eff_col], dtype=np.float64).reshape(-1)
                            has_finite_effect = bool(np.any(np.isfinite(beta_vec)))
                        except Exception:
                            has_finite_effect = False
                    if not has_finite_effect:
                        meta_skip = dict(trait_effect_meta_by_method.get(m_key, {}) or {})
                        notes_skip = list(meta_skip.get("notes", []) or [])
                        notes_skip.append("excluded_from_merged_effect:effect_unavailable")
                        meta_skip["notes"] = notes_skip
                        meta_skip["merged_effect_included"] = False
                        trait_effect_meta_by_method[m_key] = meta_skip
                        _log_file_only(
                            "Merged effect export: skip method "
                            f"{m_key} for trait={trait_name} (effect unavailable)."
                        )
                        continue
                    merge_method_order.append(m_key)
                    meta_keep = dict(trait_effect_meta_by_method.get(m_key, {}) or {})
                    meta_keep["merged_effect_included"] = True
                    trait_effect_meta_by_method[m_key] = meta_keep

                merged_effect_method_keys = list(merge_method_order)
                if len(merged_effect_method_keys) > 0:
                    merged_effect_tbl = _build_trait_merged_effect_table(
                        method_tables=dict(trait_effect_tables),
                        method_order=list(merge_method_order),
                        method_display_map=dict(method_display_map),
                    )
                    os.makedirs(os.path.dirname(out_effect_merged), exist_ok=True)
                    merged_effect_tbl.to_csv(
                        out_effect_merged,
                        sep="\t",
                        index=False,
                        float_format="%.6g",
                    )
                    saved_result_paths.append(str(out_effect_merged))
                    merged_effect_written = True
            except Exception as ex:
                logger.warning(
                    f"Merged effect export failed for trait={trait_name}: {ex}"
                )
        if merged_effect_written:
            merged_effect_method_set = set(merged_effect_method_keys)
            for m in trait_methods:
                m_key = str(m)
                if m_key in trait_method_summary.get(str(trait_name), {}):
                    if m_key in merged_effect_method_set:
                        trait_method_summary[str(trait_name)][m_key]["effect_file"] = str(
                            out_effect_merged
                        )
                    else:
                        trait_method_summary[str(trait_name)][m_key]["effect_file"] = None
                    if m_key in trait_effect_meta_by_method:
                        trait_method_summary[str(trait_name)][m_key]["effect_meta"] = dict(
                            trait_effect_meta_by_method[m_key]
                        )
            for art in exported_model_artifacts:
                trait_art = str(art.get("trait", "")).strip()
                method_art = str(art.get("method", "")).strip()
                if (trait_art != str(trait_name)) or (method_art not in trait_method_keys):
                    continue
                if method_art in merged_effect_method_set:
                    art["effect_file"] = str(out_effect_merged)
                    art["effect_column"] = str(
                        method_display_map.get(method_art, method_art)
                    )
                else:
                    art["effect_file"] = ""
                    art["effect_column"] = ""
                if method_art in trait_effect_meta_by_method:
                    art["effect_meta"] = dict(trait_effect_meta_by_method[method_art])
        else:
            for m in trait_methods:
                m_key = str(m)
                if m_key in trait_method_summary.get(str(trait_name), {}):
                    cur_eff = trait_method_summary[str(trait_name)][m_key].get("effect_file", None)
                    if str(cur_eff or "").strip() == str(out_effect_merged):
                        trait_method_summary[str(trait_name)][m_key]["effect_file"] = None
            for art in exported_model_artifacts:
                trait_art = str(art.get("trait", "")).strip()
                method_art = str(art.get("method", "")).strip()
                if (trait_art != str(trait_name)) or (method_art not in trait_method_keys):
                    continue
                if str(art.get("effect_file", "")).strip() == str(out_effect_merged):
                    art["effect_file"] = ""
        top_selected_method = ""
        top_metric_scores: dict[str, dict[str, float]] = {}
        if top_enabled:
            if model_mode and bundle_mode and (trait_bundle_entry is not None):
                top_selected_method = str(trait_bundle_entry.get("selected_method", "")).strip()
                if top_selected_method == "":
                    raise RuntimeError(
                        f"TOP bundle trait entry missing selected_method for trait={trait_name}."
                    )
                if top_selected_method not in method_result_map:
                    raise RuntimeError(
                        f"TOP bundle selected method={top_selected_method} not present in loaded results "
                        f"for trait={trait_name}."
                    )
                score_pack = trait_bundle_entry.get("metrics", None)
                if isinstance(score_pack, dict):
                    top_metric_scores[top_selected_method] = {
                        "pearson": float(score_pack.get("pearson", np.nan)),
                        "spearman": float(score_pack.get("spearman", np.nan)),
                        "r2": float(score_pack.get("r2", np.nan)),
                        "rmse": float(score_pack.get("rmse", np.nan)),
                        "nrmse": float(score_pack.get("nrmse", np.nan)),
                    }
                else:
                    top_metric_scores[top_selected_method] = {}
                logger.info(
                    "TOP selected model for trait %s (from bundle): %s",
                    str(trait_name),
                    method_display_map.get(str(top_selected_method), str(top_selected_method)),
                )
            else:
                top_selected_method, top_metric_scores = _select_top_method_for_trait(
                    methods=trait_methods,
                    method_result_map=method_result_map,
                    train_truth=np.asarray(train_pheno, dtype=np.float64).reshape(-1),
                    metric=str(args.model_select_metric),
                )
            if not model_mode:
                sel_state_try = method_result_map[top_selected_method].get("model_state", None)
                if not (isinstance(sel_state_try, dict) and len(sel_state_try) > 0):
                    exportable_methods = [
                        str(mm)
                        for mm in trait_methods
                        if isinstance(method_result_map.get(str(mm), {}).get("model_state", None), dict)
                        and len(typing.cast(dict[str, typing.Any], method_result_map.get(str(mm), {}).get("model_state", {}))) > 0
                    ]
                    if len(exportable_methods) > 0:
                        picked = "rrBLUP" if "rrBLUP" in exportable_methods else exportable_methods[0]
                        if picked != "rrBLUP":
                            metric_key = str(args.model_select_metric).strip().lower()
                            prefer_small = metric_key in {"rmse", "nrmse"}
                            picked_score = float(
                                dict(top_metric_scores.get(picked, {})).get(metric_key, np.nan)
                            )
                            for cand in exportable_methods[1:]:
                                s = float(dict(top_metric_scores.get(cand, {})).get(metric_key, np.nan))
                                if not np.isfinite(s):
                                    continue
                                if (not np.isfinite(picked_score)) or (
                                    (s < picked_score) if prefer_small else (s > picked_score)
                                ):
                                    picked = str(cand)
                                    picked_score = float(s)
                        if picked != top_selected_method:
                            top_selected_method = str(picked)
            top_metric_key = str(args.model_select_metric).strip().lower()
            top_metric_val = float(
                dict(top_metric_scores.get(str(top_selected_method), {})).get(top_metric_key, np.nan)
            )
            logger.info(
                "TOP selected model for trait %s: %s (%s=%.4f)",
                str(trait_name),
                method_display_map.get(str(top_selected_method), str(top_selected_method)),
                top_metric_key,
                top_metric_val,
            )
            sel_res = method_result_map[top_selected_method]
            sel_oof = np.asarray(sel_res.get("oof_pred"), dtype=np.float64).reshape(-1)
            sel_truth = np.asarray(sel_res.get("oof_truth", train_pheno), dtype=np.float64).reshape(-1)
            if (not model_mode) and (int(sel_oof.size) != int(sel_truth.size)):
                raise RuntimeError(
                    f"TOP selected method {top_selected_method} returned mismatched OOF vectors "
                    f"for trait={trait_name}: pred={sel_oof.size}, truth={sel_truth.size}"
                )
            sel_test = np.asarray(sel_res.get("test_pred"), dtype=np.float64).reshape(-1)
            full_pred = np.full((int(samples.shape[0]),), np.nan, dtype=np.float64)
            if int(sel_oof.size) == int(np.sum(trainmask)):
                full_pred[trainmask] = sel_oof
            if int(sel_test.size) == int(np.sum(testmask)):
                full_pred[testmask] = sel_test
            elif int(sel_test.size) == 0 and int(np.sum(testmask)) == 0:
                pass
            else:
                raise RuntimeError(
                    f"TOP selected method {top_selected_method} returned mismatched test predictions "
                    f"for trait={trait_name}: pred={sel_test.size}, expected={int(np.sum(testmask))}"
                )
            top_trait_selected_method[str(trait_name)] = str(top_selected_method)
            top_trait_metric_scores[str(trait_name)] = dict(top_metric_scores)
            top_train_ids_by_trait[str(trait_name)] = np.asarray(samples[trainmask], dtype=str).reshape(-1)
            top_train_true_by_trait[str(trait_name)] = np.asarray(sel_truth, dtype=np.float64).reshape(-1)
            top_train_oof_by_trait[str(trait_name)] = np.asarray(sel_oof, dtype=np.float64).reshape(-1)
            top_test_ids_by_trait[str(trait_name)] = np.asarray(samples[testmask], dtype=str).reshape(-1)
            top_test_pred_by_trait[str(trait_name)] = np.asarray(full_pred, dtype=np.float64).reshape(-1)
            state_sel = sel_res.get("model_state", None)
            if isinstance(state_sel, dict) and len(state_sel) > 0:
                top_trait_model_state[str(trait_name)] = dict(state_sel)

        all_rows: list[tuple[str, int, float, float, float, float, float]] = []
        if cv_splits is not None:
            method_order = {str(m): i for i, m in enumerate(trait_methods)}
            for method in trait_methods:
                res = method_result_map.get(method)
                if res is None:
                    continue
                for row in res.get("fold_rows", []):
                    all_rows.append(row)
            all_rows.sort(
                key=lambda r: (
                    int(r[1]),
                    int(method_order.get(str(r[0]), 10**6)),
                    str(r[0]),
                )
            )
        if cv_splits is not None:
            for method in trait_methods:
                res = method_result_map.get(method)
                if res is None:
                    continue
                best_test = res.get("best_test")
                best_train = res.get("best_train")
                if best_test is None or best_train is None:
                    continue
                fig = plt.figure(figsize=(5, 4), dpi=300)
                gsplot.scatterh(
                    best_test,
                    best_train,
                    color_set=color_set[0],
                    fig=fig,
                    rasterized=False,
                )
                for ax in fig.axes:
                    if str(ax.get_ylabel()).strip() == "Predicted Value":
                        ax.set_ylabel(
                            f"Predicted Value ({method_display_map.get(str(method), str(method))})"
                        )
                        break
                svg_method = _sanitize_artifact_token(
                    method_display_map.get(str(method), str(method))
                )
                out_svg = os.path.join(gs_model_dir, f"{trait_name}.{svg_method}.svg")
                # Avoid tight_layout warnings on dense decorations.
                fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.98)
                with mpl.rc_context({"svg.fonttype": "none"}):
                    fig.savefig(
                        out_svg,
                        format="svg",
                        transparent=False,
                        facecolor="white",
                        bbox_inches="tight",
                    )
                plt.close(fig)
                saved_result_paths.append(str(out_svg))

        missing_methods = [m for m in trait_methods if m not in method_result_map]
        if len(missing_methods) > 0:
            raise RuntimeError(
                "Missing GS results for methods: " + ", ".join(missing_methods)
            )

        pred_cols: list[np.ndarray] = []
        pred_colnames: list[str] = []
        for m in trait_methods:
            m_key = str(m)
            m_disp = method_display_map.get(m_key, m_key)
            pred_arr = pred_by_method.get(m_key)
            if pred_arr is None:
                res = method_result_map.get(m_key)
                if res is None:
                    continue
                pred_arr = np.asarray(res["test_pred"], dtype=float).reshape(-1, 1)
                n_pred = int(pred_arr.shape[0])
                if n_pred != expected_test_n:
                    if (n_pred == 0) and (expected_test_n > 0):
                        logger.warning(
                            f"{m_key} returned empty test predictions while expected n_test={expected_test_n}; "
                            "filling with NaN."
                        )
                        pred_arr = np.full((expected_test_n, 1), np.nan, dtype=float)
                    else:
                        raise RuntimeError(
                            f"Prediction length mismatch for method={m_key}: "
                            f"got n_pred={n_pred}, expected n_test={expected_test_n}."
                        )
            pred_cols.append(np.asarray(pred_arr, dtype=float).reshape(-1, 1))
            pred_colnames.append(str(m_disp))

        if len(pred_cols) == 0:
            raise RuntimeError("No GS test predictions were collected from methods.")

        outpred = pd.DataFrame(
            np.concatenate(pred_cols, axis=1),
            columns=pred_colnames,
            index=samples[testmask],
        )
        outpred.to_csv(out_tsv, sep="\t", float_format="%.4f")
        saved_result_paths.append(str(out_tsv))

        if cv_splits is not None:
            gs_output.emit_cv_fold_table(
                logger,
                all_rows=all_rows,
                method_result_map=typing.cast(dict[str, dict[str, typing.Any]], method_result_map),
                method_display_map=method_display_map,
            )

        for m in trait_methods:
            res = method_result_map.get(m)
            if res is None:
                continue
            fold_rows = list(res.get("fold_rows", []))
            if len(fold_rows) > 0:
                pear_mean = float(np.nanmean([float(x[2]) for x in fold_rows]))
                spear_mean = float(np.nanmean([float(x[3]) for x in fold_rows]))
                r2_mean = float(np.nanmean([float(x[4]) for x in fold_rows]))
                elapsed_mean = float(np.nanmean([float(x[6]) for x in fold_rows]))
            else:
                pear_mean = float("nan")
                spear_mean = float("nan")
                r2_mean = float("nan")
                elapsed_mean = float("nan")
            pve_raw = res.get("pve_final")
            try:
                pve_val = float(pve_raw)
                if not np.isfinite(pve_val):
                    pve_val = float("nan")
            except Exception:
                pve_val = float("nan")
            gs_summary_rows.append(
                {
                    "trait": str(trait_name),
                    "model": str(method_display_map.get(str(m), str(m))),
                    "cv_mode": str(res.get("cv_mode", "train_cv")),
                    "n_train": int(np.sum(trainmask)),
                    "n_test": int(expected_test_n),
                    "pearsonr_cv_mean": pear_mean,
                    "spearmanr_cv_mean": spear_mean,
                    "r2_cv_mean": r2_mean,
                    "pve_final": pve_val,
                    "time_cv_mean_sec": elapsed_mean,
                }
            )
        if cv_splits is not None:
            logger.info(f"-"*60)
            selected_cv_model_key = gs_output.select_cv_best_model(
                all_rows=all_rows,
                method_order={str(m): i for i, m in enumerate(trait_methods)},
            )
            if selected_cv_model_key is not None and str(selected_cv_model_key).strip() != "":
                logger.info(
                    "Selected model: %s",
                    method_display_map.get(str(selected_cv_model_key), str(selected_cv_model_key)),
                )
        # log_success(logger, f"Saved predictions to {format_path_for_display(out_tsv)}")
        log_success(logger, f"Trait {trait_name} finished in {(time.time() - t_trait):.2f} secs")

    if top_enabled:
        available_traits = [
            str(t)
            for t in trait_order
            if (
                str(t) in top_trait_selected_method
                and str(t) in top_test_pred_by_trait
            )
        ]
        if len(available_traits) == 0:
            raise RuntimeError("TOP requested but no trait has valid model/prediction payload.")

        top_traits = [str(t) for t in available_traits]
        top_model_payload: dict[str, typing.Any] | None = None
        top_train_ids: list[str] = []
        y_true_df: pd.DataFrame | None = None
        y_pred_df: pd.DataFrame | None = None

        if model_mode:
            if top_model_payload_loaded is None:
                raise RuntimeError(
                    "TOP --model mode requires top_model_state in bundle, but none was found."
                )
            top_model_payload = dict(top_model_payload_loaded)
            model_traits_raw = top_model_payload.get("trait_names", [])
            model_traits = [str(x) for x in list(model_traits_raw or [])]
            if len(model_traits) > 0:
                top_traits = [t for t in model_traits if t in set(available_traits)]
                if len(top_traits) == 0:
                    raise RuntimeError(
                        "TOP bundle trait_names do not overlap with current phenotype traits."
                    )
        else:
            trait_train_counts: dict[str, int] = {}
            sample_all = np.asarray(samples, dtype=str).reshape(-1)
            n_all = int(sample_all.shape[0])
            sid_to_pos = {str(s): i for i, s in enumerate(sample_all.tolist())}
            k_top = int(len(top_traits))
            y_true_full = np.full((n_all, k_top), np.nan, dtype=np.float64)
            y_pred_full = np.full((n_all, k_top), np.nan, dtype=np.float64)

            for t_idx, trait in enumerate(top_traits):
                sid = np.asarray(
                    top_train_ids_by_trait.get(trait, np.array([], dtype=str)),
                    dtype=str,
                ).reshape(-1)
                y_true = np.asarray(
                    top_train_true_by_trait.get(trait, np.array([], dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
                y_oof = np.asarray(
                    top_train_oof_by_trait.get(trait, np.array([], dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
                if int(sid.size) != int(y_true.size) or int(y_true.size) != int(y_oof.size):
                    raise RuntimeError(
                        f"TOP trait payload mismatch for {trait}: ids={sid.size}, true={y_true.size}, oof={y_oof.size}"
                    )
                trait_train_counts[str(trait)] = int(sid.size)
                pred_full_col = np.asarray(
                    top_test_pred_by_trait.get(trait, np.full((n_all,), np.nan, dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
                if int(pred_full_col.size) != n_all:
                    raise RuntimeError(
                        f"TOP trait prediction length mismatch for {trait}: "
                        f"pred={pred_full_col.size}, expected={n_all}"
                    )
                y_pred_full[:, t_idx] = pred_full_col
                for ii in range(int(sid.size)):
                    pos = sid_to_pos.get(str(sid[ii]), None)
                    if pos is None:
                        continue
                    y_true_full[int(pos), t_idx] = float(y_true[ii])

            y_true_df = pd.DataFrame(y_true_full, index=sample_all, columns=top_traits)
            y_pred_df = pd.DataFrame(y_pred_full, index=sample_all, columns=top_traits)

            valid_pred_rows = np.isfinite(y_pred_full).all(axis=1)
            if int(np.sum(valid_pred_rows)) < n_all:
                dropped = int(n_all - int(np.sum(valid_pred_rows)))
                logger.warning(
                    f"TOP fitting dropped {dropped} rows due to non-finite predicted trait matrix."
                )
                y_true_df = y_true_df.loc[valid_pred_rows]
                y_pred_df = y_pred_df.loc[valid_pred_rows]
                y_true_full = np.asarray(y_true_df, dtype=np.float64)
                y_pred_full = np.asarray(y_pred_df, dtype=np.float64)
            else:
                y_true_full = np.asarray(y_true_df, dtype=np.float64)
                y_pred_full = np.asarray(y_pred_df, dtype=np.float64)

            observed_any = np.isfinite(y_true_full).any(axis=1)
            top_train_ids = [str(x) for x in y_true_df.index.to_list()]
            low_obs_threshold = 20
            low_obs_traits = [
                f"{tr}={cnt}"
                for tr, cnt in trait_train_counts.items()
                if int(cnt) < int(low_obs_threshold)
            ]
            if len(low_obs_traits) > 0:
                logger.warning(
                    "TOP low-observation traits detected (<%d): %s",
                    int(low_obs_threshold),
                    ", ".join(low_obs_traits),
                )
            if int(np.sum(observed_any)) < 2:
                logger.warning(
                    "TOP fitting is skipped: less than 2 samples have any observed phenotype "
                    "across selected traits."
                )
                logger.warning(
                    "Per-trait observed counts: "
                    + ", ".join(f"{k}={v}" for k, v in trait_train_counts.items())
                )
            else:
                top_selected_models = [str(top_trait_selected_method[t]) for t in top_traits]
                with CliStatus("Fitting TOP model from OOF predictions...", enabled=use_spinner) as task:
                    try:
                        top_model_payload = _jxrs.top_fit_model(
                            top_train_ids,
                            top_traits,
                            y_true_full,
                            y_pred_full,
                            selected_models=top_selected_models,
                            mode=str(args.top_mode),
                            exact_threshold=int(args.top_exact_threshold),
                            max_iter=int(args.top_max_iter),
                            tol=float(args.top_tol),
                            l2=float(args.top_l2),
                            batch_size=int(args.top_batch_size),
                            epochs=int(args.top_epochs),
                            learning_rate=float(args.top_lr),
                            seed=int(args.top_seed),
                            calibration_mode=str(args.top_calibration),
                        )
                    except Exception:
                        task.fail("Fitting TOP model ...Failed")
                        raise
                    task.complete(
                        f"Fitting TOP model (n={int(y_true_full.shape[0])}, k={int(y_true_full.shape[1])})"
                    )

        if top_model_payload is not None:
            weights = np.asarray(top_model_payload.get("weights", []), dtype=np.float64).reshape(-1)
            top_metrics_rows: list[dict[str, typing.Any]] = []
            trait_models_payload: dict[str, dict[str, typing.Any]] = {}
            for i, trait in enumerate(top_traits):
                method_sel = str(top_trait_selected_method.get(trait, "")).strip()
                score_pack = dict(top_trait_metric_scores.get(trait, {}).get(method_sel, {}))
                obs_n = int(
                    np.asarray(
                        top_train_true_by_trait.get(str(trait), np.array([], dtype=np.float64)),
                        dtype=np.float64,
                    ).reshape(-1).shape[0]
                )
                total_n = int(np.asarray(samples, dtype=str).reshape(-1).shape[0])
                miss_n = int(max(0, total_n - obs_n))
                warn_flag = (
                    "LOW_OBSERVED_N"
                    if (obs_n > 0 and obs_n < 20)
                    else ""
                )
                top_metrics_rows.append(
                    {
                        "trait": str(trait),
                        "observed_n": int(obs_n),
                        "missing_n": int(miss_n),
                        "selected_gs_model": method_sel,
                        "weight": (float(weights[i]) if i < int(weights.size) else float("nan")),
                        "pearson": float(score_pack.get("pearson", np.nan)),
                        "spearman": float(score_pack.get("spearman", np.nan)),
                        "r2": float(score_pack.get("r2", np.nan)),
                        "rmse": float(score_pack.get("rmse", np.nan)),
                        "nrmse": float(score_pack.get("nrmse", np.nan)),
                        "warning": str(warn_flag),
                    }
                )
                trait_models_payload[str(trait)] = {
                    "selected_method": str(method_sel),
                    "observed_n": int(obs_n),
                    "missing_n": int(miss_n),
                    "warning": str(warn_flag),
                    "metrics": dict(score_pack),
                    "model_state": dict(top_trait_model_state.get(str(trait), {})),
                }

            top_weights_out = os.path.join(gs_model_dir, f"{args.prefix}.gs.TOP.weights.tsv")
            pd.DataFrame(top_metrics_rows).to_csv(
                top_weights_out,
                sep="\t",
                index=False,
                float_format="%.6g",
            )
            saved_result_paths.append(str(top_weights_out))
            log_success(logger, f"Saved TOP weights to {format_path_for_display(top_weights_out)}")

            if not model_mode:
                top_model_out = os.path.join(gs_model_dir, f"{args.prefix}.gs.TOP.jxmodel")
                _save_jxmodel(
                    top_model_out,
                    {
                        "format": _JXMODEL_FORMAT,
                        "method": _JXMODEL_TOP_BUNDLE_METHOD,
                        "method_display": "GS_TOP_BUNDLE",
                        "trait": "MULTI",
                        "created_at_unix": float(time.time()),
                        "model_select_metric": str(args.model_select_metric).strip().lower(),
                        "top_model_state": dict(top_model_payload),
                        "trait_models": trait_models_payload,
                        "top_mask_meta": {
                            "missing_aware": True,
                            "distance_normalization": "observed_weight",
                            "min_trait_observed_warning": 20,
                            "n_samples": int(np.asarray(samples, dtype=str).reshape(-1).shape[0]),
                            "n_traits": int(len(top_traits)),
                            "traits": [str(x) for x in top_traits],
                        },
                    },
                )
                saved_result_paths.append(str(top_model_out))
                log_success(logger, f"Saved TOP bundle model to {format_path_for_display(top_model_out)}")

            pred_full_df = pd.DataFrame(
                {
                    str(t): np.asarray(top_test_pred_by_trait[str(t)], dtype=np.float64).reshape(-1)
                    for t in top_traits
                },
                index=np.asarray(samples, dtype=str).reshape(-1),
            )
            cand_valid_mask = np.isfinite(np.asarray(pred_full_df, dtype=np.float64)).all(axis=1)
            candidate_df = pred_full_df.loc[cand_valid_mask]
            if int(candidate_df.shape[0]) == 0:
                logger.warning("TOP ranking is skipped: no candidate rows with finite multi-trait predictions.")
            else:
                std_pack = dict(top_model_payload.get("standardizer", {}))
                mean_vec = np.asarray(std_pack.get("mean", []), dtype=np.float64).reshape(-1)
                std_vec = np.asarray(std_pack.get("std", []), dtype=np.float64).reshape(-1)
                target_values, target_source = _resolve_top_target_values(
                    select_arg=None,
                    # select_arg=(None if args.select is None else str(args.select)),
                    trait_names=top_traits,
                    trait_mean=mean_vec,
                    trait_std=std_vec,
                    logger=logger,
                )
                if target_values is not None:
                    with CliStatus("Ranking candidates by TOP target values...", enabled=use_spinner) as task:
                        try:
                            rank_payload = _jxrs.top_rank_to_target_values(
                                top_model_payload,
                                top_traits,
                                [str(x) for x in candidate_df.index.to_list()],
                                np.asarray(candidate_df, dtype=np.float64),
                                np.asarray(target_values, dtype=np.float64).reshape(-1),
                            )
                        except Exception:
                            task.fail("Ranking candidates by TOP ...Failed")
                            raise
                        task.complete(f"Ranking candidates by TOP (n={int(candidate_df.shape[0])})")

                    rank_rows: list[dict[str, typing.Any]] = []
                    for rec in list(rank_payload.get("records", [])):
                        raw = np.asarray(rec.get("pred_traits_raw", []), dtype=np.float64).reshape(-1)
                        cal = np.asarray(rec.get("pred_traits_calibrated", []), dtype=np.float64).reshape(-1)
                        row: dict[str, typing.Any] = {
                            "rank": int(rec.get("rank", 0)),
                            "sample_id": str(rec.get("sample_id", "")),
                            "top_distance": float(rec.get("distance", np.nan)),
                            "top_similarity": float(rec.get("similarity", np.nan)),
                            "target_source": str(target_source),
                        }
                        for i, trait in enumerate(top_traits):
                            row[f"target_{trait}"] = float(target_values[i]) if i < int(target_values.size) else float("nan")
                            row[f"{trait}_raw"] = float(raw[i]) if i < int(raw.size) else float("nan")
                            row[f"{trait}_cal"] = float(cal[i]) if i < int(cal.size) else float("nan")
                        rank_rows.append(row)
                    rank_df = pd.DataFrame(rank_rows)
                    if "rank" in rank_df.columns:
                        rank_df = rank_df.sort_values("rank", ascending=True)
                    top_rank_out = f"{outprefix}.top.rank.tsv"
                    rank_df.to_csv(top_rank_out, sep="\t", index=False, float_format="%.6g")
                    saved_result_paths.append(str(top_rank_out))
                    log_success(logger, f"Saved TOP rank to {format_path_for_display(top_rank_out)}")

    run_finished_unix = float(time.time())
    summary_out = os.path.join(gs_model_dir, "summary.json")
    ordered_saved_paths = _order_gs_saved_result_paths([str(x) for x in saved_result_paths])
    result_files_for_summary = _order_gs_saved_result_paths(
        ordered_saved_paths + [str(summary_out)]
    )
    method_display_requested = [str(_method_display_name(str(x))) for x in methods]
    summary_payload: dict[str, typing.Any] = {
        "format": "janusx.gs.summary.v1",
        "created_at_unix": run_finished_unix,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(run_finished_unix)),
        "started_at_unix": float(run_started_unix),
        "started_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime(run_started_unix)),
        "elapsed_sec": float(max(run_finished_unix - t_start, 0.0)),
        "prefix": str(args.prefix),
        "output_dir": str(args.out),
        "outprefix": str(outprefix),
        "model_dir": str(gs_model_dir),
        "genotype_input": str(gfile),
        "phenotype_input": str(args.pheno),
        "log_file": str(log_path),
        "traits": [str(x) for x in trait_order],
        "methods_requested": [str(x) for x in methods],
        "methods_display": method_display_requested,
        "summary_rows": [dict(x) for x in gs_summary_rows],
        "trait_method_details": dict(trait_method_summary),
        "top_enabled": bool(top_enabled),
        "top_selected_by_trait": dict(top_trait_selected_method),
        "model_artifacts": list(exported_model_artifacts),
        "result_files": result_files_for_summary,
        "resource": {
            "memory_peak_rss": None,
            "memory_note": "Not collected yet in GS workflow.",
        },
        "usage": {
            "model_directory": str(gs_model_dir),
            "single_trait_model_file_pattern": "{trait}.{method}.jxmodel",
            "effect_file_pattern": "{prefix}.{trait}.gs.effect",
            "summary_file": "summary.json",
            "loaded_model_hint": (
                "Use --model <model_dir> with the target trait and method. "
                "Both new ({trait}.{method}.jxmodel) and legacy (*.gs.{method}.jxmodel) names are supported."
            ),
        },
    }
    try:
        os.makedirs(gs_model_dir, exist_ok=True)
        with open(summary_out, "w", encoding="utf-8") as sfh:
            json.dump(_json_safe(summary_payload), sfh, indent=2, ensure_ascii=False)
        saved_result_paths = list(result_files_for_summary)
        # log_success(logger, f"Saved GS summary to {format_path_for_display(summary_out)}")
    except Exception as ex:
        logger.warning(f"Failed to write GS summary.json: {ex}")
    saved_result_paths = _order_gs_saved_result_paths([str(x) for x in saved_result_paths])
    if len(saved_result_paths) > 0:
        logger.info("")
        saved_body = _format_gs_saved_result_report(
            saved_result_paths,
            trait_order=[str(x) for x in trait_order],
        )
        if saved_body.strip() == "":
            saved_body = "\n".join(
                [f"  {format_path_for_display(str(p))}" for p in saved_result_paths]
            )
        log_success(logger, f"Results saved:\n{saved_body}")

    # ----------------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished, total time: {round(time.time() - t_start, 2)} secs\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    log_success(logger, endinfo)
    if bool(return_result):
        return {
            "status": "done",
            "error": "",
            "genofile": str(gfile),
            "outprefix": str(outprefix),
            "model_dir": str(gs_model_dir),
            "summary_json": str(summary_out),
            "log_file": str(log_path),
            "summary_rows": [dict(x) for x in gs_summary_rows],
            "result_files": [str(x) for x in saved_result_paths],
            "elapsed_sec": float(max(time.time() - t_start, 0.0)),
            "traits": [str(x) for x in trait_order],
            "methods": [str(_method_display_name(str(x))) for x in methods],
        }


def main(
    argv: typing.Optional[list[str]] = None,
    log: bool = True,
    return_result: bool = False,
) -> typing.Optional[dict[str, typing.Any]]:
    args = parse_args(argv)
    from janusx.gs.runner import run_gs_args
    return run_gs_args(args, log=log, return_result=return_result)


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
