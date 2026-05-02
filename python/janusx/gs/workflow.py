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
import socket
import argparse
import sys
import re
import math
import gc
import threading
import subprocess
import shutil
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


def _warn_rust_gblup_backend_fallback_once(backend: str) -> None:
    b = str(backend).strip().lower() or "unknown"
    with _RUST_GBLUP_BACKEND_WARN_LOCK:
        if b in _RUST_GBLUP_BACKEND_WARNED:
            return
        _RUST_GBLUP_BACKEND_WARNED.add(b)
    logging.getLogger(__name__).warning(
        "Warning: Packed Rust GBLUP detected rust BLAS backend='%s' (OpenBLAS preferred). "
        "Falling back to compatibility GS path; this may run slower.",
        b,
    )


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
    Then cap by common math-thread env vars when present.
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
    #   n <= 10k: exact REML/FaST path (stable, no subsample lambda)
    #   n > 10k : PCG path (subsample REML lambda, no AdamW by default)
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
            "n_sub": 0,
            "repeats": 0,
            "ok_repeats": 0,
        }

    sub_min = int(max(16, int(cfg_use.get("lambda_subsample_min_n", _RRBLUP_LAMBDA_SUBSAMPLE_MIN_N))))
    sub_max = int(max(sub_min, int(cfg_use.get("lambda_subsample_max_n", _RRBLUP_LAMBDA_SUBSAMPLE_MAX_N))))
    sub_target = int(max(sub_min, int(cfg_use.get("lambda_subsample_n", _RRBLUP_LAMBDA_SUBSAMPLE_MAX_N))))
    n_sub = int(min(max(1, n_train), min(sub_max, max(sub_min, sub_target))))
    rep_cfg = int(cfg_use.get("lambda_subsample_repeats", _RRBLUP_LAMBDA_SUBSAMPLE_REPEATS))
    repeats = int(min(20, max(5, rep_cfg)))
    seed = int(cfg_use.get("lambda_subsample_seed", 42))
    rng = np.random.default_rng(seed)
    log10_vals: list[float] = []
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

    def _emit(event: str, **payload: typing.Any) -> None:
        if progress_hook is None:
            return
        try:
            progress_hook(str(event), dict(payload))
        except Exception:
            return

    _emit(
        "pcg_lambda_subsample_start",
        total=int(repeats),
        n_sub=int(n_sub),
        n_train=int(n_train),
    )
    for rep in range(repeats):
        if n_sub >= n_train:
            local_idx = np.arange(n_train, dtype=np.int64)
        else:
            local_idx = np.asarray(
                rng.choice(n_train, size=n_sub, replace=False),
                dtype=np.int64,
            )
            local_idx.sort()
        sub_abs = np.ascontiguousarray(train_abs[local_idx], dtype=np.int64)
        y_sub = np.ascontiguousarray(y_vec[local_idx], dtype=np.float64)
        try:
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
        except Exception:
            lam_sub_k = float("nan")
        lam_sub_eq = float(lam_sub_k * m_scale) if (np.isfinite(lam_sub_k) and lam_sub_k > 0.0) else float("nan")
        if np.isfinite(lam_sub_eq) and lam_sub_eq > 0.0:
            log10_vals.append(float(np.log10(lam_sub_eq)))
        _emit(
            "pcg_lambda_subsample_iter",
            iter=int(rep + 1),
            total=int(repeats),
            lambda_equation=float(lam_sub_eq),
            lambda_k=float(lam_sub_k),
            m_effective=int(m_effective),
        )
    if len(log10_vals) == 0:
        _emit(
            "pcg_lambda_subsample_end",
            repeats=int(repeats),
            ok_repeats=0,
            lambda_equation=float("nan"),
        )
        return {
            "lambda_equation": float("nan"),
            "n_sub": int(n_sub),
            "repeats": int(repeats),
            "ok_repeats": 0,
        }
    lam_eq = float(10.0 ** float(np.median(np.asarray(log10_vals, dtype=np.float64))))
    _emit(
        "pcg_lambda_subsample_end",
        repeats=int(repeats),
        ok_repeats=int(len(log10_vals)),
        lambda_equation=float(lam_eq),
        m_effective=int(m_effective),
    )
    return {
        "lambda_equation": float(lam_eq),
        "n_sub": int(n_sub),
        "repeats": int(repeats),
        "ok_repeats": int(len(log10_vals)),
        "m_effective": int(m_effective),
    }


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
    lam0 = float(cfg_use.get("lambda_value", 10000.0))
    lr0 = float(cfg_use.get("lr", 1e-2))
    if (not np.isfinite(lam0)) or lam0 < 0.0:
        lam0 = 10000.0
    if (not np.isfinite(lr0)) or lr0 <= 0.0:
        lr0 = 1e-2
    grid_size = int(max(1, min(4, int(cfg_use.get("grid_size", 4)))))
    lam_anchor = float(max(lam0, 10000.0))
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
    sidx = np.ascontiguousarray(np.asarray(sample_indices, dtype=np.int64).reshape(-1))
    blk = _jxrs.bed_packed_decode_rows_f32(
        packed,
        int(n_samples),
        ridx,
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
    m = int(packed.shape[0])
    if mean_raw is not None and inv_raw is not None:
        mean = np.ascontiguousarray(np.asarray(mean_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
        inv = np.ascontiguousarray(np.asarray(inv_raw, dtype=np.float32).reshape(-1), dtype=np.float32)
        if int(mean.shape[0]) == m and int(inv.shape[0]) == m:
            return mean, inv

    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    if int(maf.shape[0]) != m:
        raise ValueError("Packed context mismatch: maf length != packed rows.")
    n_samples = int(packed_ctx["n_samples"])
    if n_samples <= 0:
        raise ValueError("Packed context invalid: n_samples must be > 0.")

    mean_f64: np.ndarray
    var_f64: np.ndarray
    use_rust_stats = bool(_jxrs is not None and hasattr(_jxrs, "bed_packed_decode_stats_f64"))
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

    m = int(np.asarray(packed_ctx["packed"]).shape[0])
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

    lambda_value = float(cfg_use.get("lambda_value", 10000.0))
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
    rrblup_solver: typing.Literal["exact", "adamw", "pcg", "auto"] = "auto",
    rrblup_adamw_cfg: dict[str, typing.Any] | None = None,
    rrblup_runtime_state: dict[str, typing.Any] | None = None,
    rrblup_progress_hook: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
    gblup_runtime_state: dict[str, typing.Any] | None = None,
    bayes_auto_r2: float | None = None,
    bayes_runtime_state: dict[str, typing.Any] | None = None,
    bayes_auto_cfg: dict[str, typing.Any] | None = None,
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

    # Dominance-kernel and additive+dominance GBLUP variants (pyBLUP/JAX backend)
    if _is_gblup_method(str(method)) and (_gblup_method_kernel_mode(str(method)) in {"d", "ad"}):
        mode = _gblup_method_kernel_mode(str(method))
        if (not _HAS_ADBLUP_PY) or KernelBLUP is None or Gmatrix is None:
            raise ImportError(
                "GBLUP(d/ad) (JAX backend) is unavailable. "
                f"Original import error: {_ADBLUP_PY_IMPORT_ERROR}"
            )
        gadd_train = np.asarray(Xtrain, dtype=np.float32)
        gadd_test = np.asarray(Xtest, dtype=np.float32)
        if gadd_train.ndim != 2 or gadd_test.ndim != 2:
            raise ValueError(
                f"GBLUP(d/ad) expects 2D genotype matrices. got train={gadd_train.shape}, test={gadd_test.shape}"
            )
        if gadd_train.shape[0] != gadd_test.shape[0]:
            raise ValueError(
                f"GBLUP(d/ad) marker mismatch: train={gadd_train.shape[0]}, test={gadd_test.shape[0]}"
            )

        ghet_train = (gadd_train == 1.0).astype(np.float32, copy=False)
        Ghet_train = Gmatrix(ghet_train)
        if mode == "d":
            model = KernelBLUP(Y.reshape(-1, 1), G=[Ghet_train], progress=False)
        else:
            Gadd_train = Gmatrix(gadd_train)
            model = KernelBLUP(Y.reshape(-1, 1), G=[Gadd_train, Ghet_train], progress=False)

        if need_train_pred:
            if mode == "d":
                yhat_train = model.predict(G=[Ghet_train])
            else:
                yhat_train = model.predict(G=[Gadd_train, Ghet_train])
        else:
            yhat_train = np.zeros((0, 1), dtype=float)
        n_train = int(gadd_train.shape[1])
        if int(gadd_test.shape[1]) > 0:
            gadd_all = np.concatenate([gadd_train, gadd_test], axis=1)
            ghet_all = (gadd_all == 1.0).astype(np.float32, copy=False)
            Ghet_all = Gmatrix(ghet_all)
            Ghet_cross = Ghet_all[n_train:, :n_train]
            if mode == "d":
                yhat_test = model.predict(G=[Ghet_cross])
            else:
                Gadd_all = Gmatrix(gadd_all)
                Gadd_cross = Gadd_all[n_train:, :n_train]
                yhat_test = model.predict(G=[Gadd_cross, Ghet_cross])
        else:
            yhat_test = np.zeros((0, 1), dtype=float)

        pve = float("nan")
        try:
            var = np.asarray(getattr(model, "var", []), dtype=float).reshape(-1)
            if var.size >= 2 and np.all(np.isfinite(var)):
                den = float(np.sum(var))
                if den > 0:
                    pve = float(np.sum(var[:-1]) / den)
        except Exception:
            pve = float("nan")
        return np.asarray(yhat_train, dtype=float), np.asarray(yhat_test, dtype=float), pve

    # Linear mixed models
    if method in ("GBLUP", "rrBLUP"):
        n_train_local = int(Y.reshape(-1).shape[0])
        n_snp_local = int(
            Xtrain["packed"].shape[0]
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
        kinship = 1 if method == "GBLUP" else None

        def _emit_rrblup_progress(event: str, **payload: typing.Any) -> None:
            if rrblup_progress_hook is None:
                return
            try:
                rrblup_progress_hook(str(event), dict(payload))
            except Exception:
                return

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
                Xtrain["packed"].shape[0]
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
            if can_use_rust_gblup and (rust_backend != "openblas"):
                _warn_rust_gblup_backend_fallback_once(rust_backend)
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
                with runtime_thread_stage(
                    blas_threads=int(max(1, int(n_jobs))),
                    rayon_threads=1,
                ):
                    rust_blas_in_stage = get_rust_blas_threads()
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
                    ) = (
                        _jxrs.gblup_reml_packed_bed(  # type: ignore[union-attr]
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
                    )
                if gblup_runtime_state is not None:
                    gblup_runtime_state["rust_backend"] = str(rust_backend)
                    gblup_runtime_state["eigh_backend"] = str(_eigh_backend)
                    gblup_runtime_state["eigh_sec"] = float(_eigh_elapsed)
                    gblup_runtime_state["lambda_reml"] = float(_lambda_opt)
                    gblup_runtime_state["ml"] = float(_ml)
                    gblup_runtime_state["reml"] = float(_reml)
                    gblup_runtime_state["m_effective"] = int(_m_eff)
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
            requested_solver = str(rrblup_solver).strip().lower()
            if not is_packed_input:
                if requested_solver == "pcg":
                    raise ValueError(
                        "rrBLUP solver=pcg requires packed PLINK input (bfile) in this build."
                    )
                resolved_rr_solver = "exact"
            elif (_jxrs is None) or (not hasattr(_jxrs, "rrblup_pcg_bed")):
                if requested_solver == "pcg":
                    raise RuntimeError(
                        "Rust rrBLUP-PCG kernel is unavailable. Rebuild/install JanusX extension."
                    )
                resolved_rr_solver = "exact"
            else:
                packed_train = typing.cast(dict[str, typing.Any], Xtrain)
                source_prefix_raw = packed_train.get("source_prefix", None)
                if source_prefix_raw is None or str(source_prefix_raw).strip() == "":
                    raise ValueError(
                        "Packed rrBLUP-PCG requires source PLINK prefix in packed context."
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
                lambda_raw = float(rr_cfg.get("lambda_value", 10000.0))
                lambda_scale = str(rr_cfg.get("lambda_scale", "equation")).strip().lower()
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
                auto_pcg_min_n = int(
                    max(2, int(rr_cfg.get("auto_pcg_min_n", _RRBLUP_AUTO_PCG_MIN_N)))
                )
                auto_pcg_ref_n = int(max(0, int(rr_cfg.get("auto_pcg_ref_n", n_train_local))))
                lambda_auto_enabled = _cfg_truthy(rr_cfg.get("lambda_auto", "on"), default=True)
                if (
                    str(requested_solver) == "auto"
                    and int(auto_pcg_ref_n) > int(auto_pcg_min_n)
                    and bool(lambda_auto_enabled)
                ):
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
                    if np.isfinite(lam_auto) and lam_auto > 0.0:
                        lambda_equation = float(lam_auto)
                        lambda_raw = _rrblup_lambda_equation_to_raw(
                            lambda_equation=float(lambda_equation),
                            lambda_scale=lambda_scale,
                            n_train=int(n_train_local),
                        )
                        lambda_source = "subsample_reml"
                if (not np.isfinite(lambda_equation)) or (lambda_equation <= 0.0):
                    lambda_equation = float(1e-8)
                pcg_tol = float(rr_cfg.get("pcg_tol", 1e-4))
                pcg_max_iter = int(max(1, int(rr_cfg.get("pcg_max_iter", 100))))
                pcg_block_rows = int(
                    max(1, int(rr_cfg.get("pcg_block_rows", rr_cfg.get("snp_block_size", 4096))))
                )
                pcg_std_eps = float(max(np.finfo(np.float32).eps, float(rr_cfg.get("pcg_std_eps", 1e-12))))

                if not (np.isfinite(pcg_tol) and pcg_tol > 0.0):
                    raise ValueError(f"rrBLUP PCG tol must be finite and > 0, got {pcg_tol!r}.")
                if not (np.isfinite(pcg_std_eps) and pcg_std_eps > 0.0):
                    raise ValueError(
                        f"rrBLUP PCG std_eps must be finite and > 0, got {pcg_std_eps!r}."
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
                _emit_rrblup_progress(
                    "pcg_start",
                    total=int(pcg_max_iter),
                    tol=float(pcg_tol),
                    lambda_equation=float(lambda_equation),
                )
                def _on_pcg_progress(done: int, total_from_backend: int) -> None:
                    try:
                        _emit_rrblup_progress(
                            "pcg_iter",
                            iter=int(max(0, int(done))),
                            total=int(max(1, int(total_from_backend))),
                        )
                    except Exception:
                        return
                pcg_progress_cb = _on_pcg_progress if rrblup_progress_hook is not None else None
                pcg_progress_every = 1 if pcg_progress_cb is not None else 0
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
                        int(max(1, int(n_jobs))),
                        progress_callback=pcg_progress_cb,
                        progress_every=int(pcg_progress_every),
                    )
                except TypeError:
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
                        int(max(1, int(n_jobs))),
                    )
                rr_out_t = tuple(rr_out)
                if len(rr_out_t) >= 9:
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
                    iters=int(pcg_iters),
                    converged=bool(pcg_converged),
                    rel_res=float(pcg_rel_res),
                    total=int(pcg_max_iter),
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
                pve_mode = str(rr_cfg.get("pve_mode", "lambda")).strip().lower()
                if pve_mode not in {"lambda", "trainvar"}:
                    pve_mode = "lambda"
                pve = float(pve_lambda if pve_mode == "lambda" else float(pve_trainvar))

                if rrblup_runtime_state is not None:
                    rrblup_runtime_state["solver"] = "pcg"
                    rrblup_runtime_state["pcg_converged"] = bool(pcg_converged)
                    rrblup_runtime_state["pcg_iters"] = int(pcg_iters)
                    rrblup_runtime_state["pcg_rel_res"] = float(pcg_rel_res)
                    rrblup_runtime_state["m_effective"] = int(m_eff)
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
                    rrblup_runtime_state["pve_mode_used"] = str(pve_mode)
                    rrblup_runtime_state["pve_used"] = float(pve)
                    rrblup_runtime_state["pve_lambda"] = float(pve_lambda)
                    rrblup_runtime_state["pve_lambda_formula"] = float(pve_lambda_formula)
                    rrblup_runtime_state["pve_lambda_trace"] = float(pve_lambda_trace)
                    rrblup_runtime_state["pve_trainvar"] = float(pve_trainvar)
                    rrblup_runtime_state["k_trace_mean"] = float(k_trace_mean)

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
                    base_lambda = float(max(float(rr_cfg_base.get("lambda_value", 10000.0)), 10000.0))
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
            if rrblup_runtime_state is not None:
                rrblup_runtime_state["solver"] = "adamw"
                rrblup_runtime_state["auto_grid_enabled"] = bool(use_auto_grid)
                rrblup_runtime_state["selected_lambda"] = float(selected_cfg.get("lambda_value", np.nan))
                rrblup_runtime_state["selected_lr"] = float(selected_cfg.get("lr", np.nan))
                rrblup_runtime_state["selected_epochs"] = int(selected_cfg.get("epochs", fit.get("best_epoch", 1)))
                rrblup_runtime_state["best_epoch"] = int(max(1, int(fit.get("best_epoch", 1))))
                rrblup_runtime_state["epochs_ran"] = int(max(1, int(fit.get("epochs_ran", fit.get("best_epoch", 1)))))
                rrblup_runtime_state["stopped_early"] = bool(fit.get("stopped_early", False))
                rrblup_runtime_state["best_val_loss"] = float(fit.get("best_val_loss", np.nan))
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
        if method == "rrBLUP" and rrblup_runtime_state is not None:
            rrblup_runtime_state["solver"] = str(resolved_rr_solver)
            rrblup_runtime_state["pve_used"] = float(model.pve)
            rrblup_runtime_state["pve_exact"] = float(model.pve)
            # Exact rrBLUP follows REML/variance-component PVE from pyBLUP.
            # Keep trainvar slot aligned to the reported PVE for unified logging.
            rrblup_runtime_state["pve_trainvar"] = float(model.pve)
            rrblup_runtime_state["pve_lambda"] = float("nan")
            rrblup_runtime_state["pve_mode_used"] = "exact_reml"
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
                    "seed": None,
                }
                if str(method) in {"BayesB", "BayesCpi"}:
                    packed_kwargs["prob_in"] = 0.5
                    packed_kwargs["counts"] = 5.0
                if str(method) == "BayesA":
                    packed_kwargs["min_abs_beta"] = 1e-9

                packed_fit_fn = getattr(_jxrs, str(packed_func_name))
                packed_fit_ret = packed_fit_fn(**packed_kwargs)
                if str(method) == "BayesCpi":
                    beta_raw, alpha_raw, _varb_mean, _vare, h2_mean, _var_h2 = packed_fit_ret
                else:
                    beta_raw, alpha_raw, _varb, _vare, h2_mean, _var_h2 = packed_fit_ret

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
        return (
            np.asarray(train_pred, dtype=float).reshape(-1, 1),
            np.asarray(model.predict(Xtest), dtype=float).reshape(-1, 1),
            pve,
        )

    if method in _ML_METHOD_MAP:
        model = MLGS(
            y=Y.reshape(-1),
            M=Xtrain,
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
                yhat_train = model.predict(Xtrain)
            elif int(train_pred_idx.size) == 0:
                yhat_train = np.zeros((0, 1), dtype=float)
            else:
                yhat_train = model.predict(np.asarray(Xtrain)[:, train_pred_idx])
        else:
            yhat_train = np.zeros((0, 1), dtype=float)
        if int(Xtest.shape[1]) > 0:
            yhat_test = model.predict(Xtest)
        else:
            yhat_test = np.zeros((0, 1), dtype=float)
        pve = float(
            (model.best_metrics_ or {}).get("r2", np.nan)
        )
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
    progress_queue: typing.Any = None,
    progress_hook: typing.Any = None,
    search_progress_hook: typing.Any = None,
    rrblup_progress_hook: typing.Callable[[str, dict[str, typing.Any]], None] | None = None,
    limit_predtrain: int | None = None,
    rrblup_solver: str = "auto",
    rrblup_adamw_cfg: dict[str, typing.Any] | None = None,
    bayes_auto_r2_cache: dict[str, float] | None = None,
    bayes_auto_r2_cfg: dict[str, typing.Any] | None = None,
) -> dict[str, typing.Any]:
    fold_rows: list[tuple[str, int, float, float, float, float, float]] = []
    rrblup_pve_rows: list[dict[str, typing.Any]] = []
    rrblup_pve_final: dict[str, typing.Any] | None = None
    rrblup_final_state: dict[str, typing.Any] | None = None
    rrblup_final_cfg: dict[str, typing.Any] | None = None
    gblup_final_state: dict[str, typing.Any] | None = None
    bayes_r2_rows: list[float] = []
    bayes_r2_final = float("nan")
    bayes_r2_source_final = ""
    bayes_r2_n_used_final = 0
    bayes_r2_n_total_final = 0
    best_test = None
    best_train = None
    best_r2 = -np.inf
    ml_tuning_cache: dict[str, typing.Any] | None = None
    rrblup_cfg_base = dict(rrblup_adamw_cfg or {})
    if method == "rrBLUP":
        rrblup_cfg_base["auto_pcg_ref_n"] = int(np.asarray(train_pheno).reshape(-1).shape[0])
    packed_lmm_methods = {"GBLUP", "rrBLUP", "BayesA", "BayesB", "BayesCpi"}
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
        pred = np.ascontiguousarray(
            np.asarray(precomputed_hash_ztz_test_pred, dtype=np.float64).reshape(-1, 1),
            dtype=np.float64,
        )
        pve_final = (
            float(precomputed_hash_ztz_pve)
            if precomputed_hash_ztz_pve is not None
            else float("nan")
        )
        return {
            "method": method,
            "method_display": _method_display_name(method),
            "fold_rows": [],
            "best_test": None,
            "best_train": None,
            "test_pred": pred,
            "pve_final": float(pve_final),
            "ml_tuning": None,
            "rrblup_final_state": None,
            "rrblup_final_cfg": None,
            "rrblup_cv_selected": None,
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
                    if rrblup_cv_reuse_enabled and rrblup_cv_selected is not None:
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
                if method == "GBLUP":
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
                    if np.isfinite(sel_lam) and sel_lam >= 0.0:
                        rrblup_cv_selected = {"lambda_value": float(sel_lam)}
                        if np.isfinite(sel_lr) and sel_lr > 0.0:
                            rrblup_cv_selected["lr"] = float(sel_lr)
                        if _GS_DEBUG_STAGE:
                            msg_lr = (
                                f" lr={float(sel_lr):.6g}"
                                if (np.isfinite(sel_lr) and sel_lr > 0.0)
                                else ""
                            )
                            print(
                                "[GS-DEBUG] rrBLUP CV reuse armed "
                                f"lambda={sel_lam:.6g}{msg_lr}",
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
                    rrblup_pve_rows.append(
                        {
                            "fold": int(fold_id),
                            "solver": str(rr_state_call.get("solver", str(rrblup_solver).strip().lower())),
                            "pve_mode": str(rr_cfg_mode),
                            "pve_used": pve_used_f,
                            "pve_trainvar": pve_trainvar_f,
                            "pve_lambda": pve_lambda_f,
                            "pve_exact": _float_or_nan(rr_state_call.get("pve_exact", np.nan)),
                            "iter_like": iter_like,
                        }
                    )
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

    if use_hash_cv_compact:
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
            "test_pred": np.asarray(test_pred, dtype=float).reshape(-1, 1),
            "pve_final": float(fit_final["pve"]),
            "ml_tuning": ml_tuning_cache,
            "rrblup_final_state": None,
            "rrblup_final_cfg": None,
            "rrblup_cv_selected": None,
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

    if skip_final_fit:
        test_pred = np.zeros((0, 1), dtype=float)
        if len(fold_rows) > 0:
            pve_final = float(np.nanmean([float(r[5]) for r in fold_rows]))
        else:
            pve_final = float("nan")
        if method == "rrBLUP":
            rrblup_final_cfg = dict(rrblup_cfg_base)
            if rrblup_cv_reuse_enabled and rrblup_cv_selected is not None:
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
            rrblup_pve_final = {
                "phase": "final(skipped)",
                "solver": str(rrblup_solver).strip().lower(),
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
    elif use_custom_gblup_final:
        fit = _fit_gblup_reml_from_grm(
            np.asarray(train_pheno, dtype=np.float64).reshape(-1),
            np.asarray(gblup_cv_grm_full, dtype=np.float64),
        )
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
    else:
        rr_cfg_final: dict[str, typing.Any] | None = rrblup_adamw_cfg
        rr_state_final: dict[str, typing.Any] | None = None
        rr_progress_final = None
        gblup_state_final: dict[str, typing.Any] | None = None
        bayes_r2_final_call: float | None = None
        bayes_state_final: dict[str, typing.Any] | None = None
        bayes_r2_final_key: str | None = None
        if method == "rrBLUP":
            rr_cfg_final = dict(rrblup_cfg_base)
            if rrblup_cv_reuse_enabled and rrblup_cv_selected is not None:
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
        if method == "GBLUP":
            gblup_state_final = {}
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
        )
        if method == "GBLUP" and gblup_state_final is not None:
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
            rrblup_pve_final = {
                "phase": "final",
                "solver": str(rr_state_final.get("solver", str(rrblup_solver).strip().lower())),
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
    return {
        "method": method,
        "method_display": _method_display_name(method),
        "fold_rows": fold_rows,
        "rrblup_pve_rows": rrblup_pve_rows,
        "rrblup_pve_final": rrblup_pve_final,
        "best_test": best_test,
        "best_train": best_train,
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
    limit_predtrain: int | None = None,
    elapsed_offset_by_method: dict[str, float] | None = None,
    rrblup_solver: str = "auto",
    rrblup_adamw_cfg: dict[str, typing.Any] | None = None,
    bayes_auto_r2_cfg: dict[str, typing.Any] | None = None,
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
            rows.append(("variance component", "REML/FaST"))
            if mode in {"d", "ad"}:
                rows.append(("backend", "pyBLUP/JAX"))
            else:
                rows.append(
                    (
                        "backend",
                        ("packed active" if _looks_like_packed_payload(packed_ctx) else "dense"),
                    )
                )
                gblup_state = dict(
                    typing.cast(
                        dict[str, typing.Any] | None,
                        result.get("gblup_final_state"),
                    )
                    or {}
                )
                eigh_backend = str(gblup_state.get("eigh_backend", "")).strip()
                if eigh_backend != "":
                    rows.append(("eigh backend", eigh_backend))
                try:
                    eigh_sec = float(gblup_state.get("eigh_sec", np.nan))
                except Exception:
                    eigh_sec = float("nan")
                if np.isfinite(eigh_sec):
                    rows.append(("eigh time", f"{eigh_sec:.3f}s"))
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
            solver_req = str(rrblup_solver).strip().lower()
            solver_used = str(rr_state.get("solver", solver_req)).strip().lower()
            if solver_req != solver_used:
                rows.append(("solver", f"{solver_req} -> {solver_used.upper()}"))
            else:
                rows.append(("solver", solver_used.upper()))

            if solver_used == "pcg":
                lam_raw = float(rr_state.get("selected_lambda", rr_cfg.get("lambda_value", np.nan)))
                if np.isfinite(lam_raw):
                    lam_scale = str(rr_cfg.get("lambda_scale", "equation")).strip().lower()
                    rows.append(("lambda", f"{lam_raw:.6g} ({lam_scale})"))
                rows.append(
                    (
                        "PCG",
                        (
                            f"tol={float(rr_cfg.get('pcg_tol', np.nan)):.1e}, "
                            f"max_iter={int(max(1, int(rr_cfg.get('pcg_max_iter', 0))))}"
                        ),
                    )
                )
            elif solver_used == "adamw":
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

            pve_mode = str(rr_state.get("pve_mode_used", rr_cfg.get("pve_mode", "lambda"))).strip().lower()
            if pve_mode != "":
                rows.append(("PVE mode", pve_mode))
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

    def _emit_method_details_plain(
        method: str,
        result: dict[str, typing.Any],
        *,
        line_printer: typing.Callable[[str], None],
    ) -> None:
        rows = _method_detail_rows(str(method), result)
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

    show_method_progress = cv_splits is not None
    # Force serial execution for all GS methods to avoid cross-method contention
    # and keep runtime behavior consistent.
    method_parallel_jobs = 1
    bayes_auto_r2_cache_shared: dict[str, float] = {}
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
            with progress:
                for m in methods:
                    m_disp = _method_display(str(m))
                    cv_total = int(max(1, fold_total_map.get(m, 1)))
                    has_search_stage = bool((m in _ML_METHOD_MAP) and (not strict_cv))
                    has_rrblup_iter_progress = bool(
                        rrblup_progress_enabled_by_method.get(str(m), False)
                    )
                    search_task_id: int | None = None
                    search_done = 0
                    search_total = 0
                    search_total_fixed = False
                    cv_task_id: int | None = None
                    cv_done = 0
                    adam_task_id: int | None = None
                    adam_done = 0
                    adam_total = 0
                    adam_stage = "fit"
                    adam_grid_trial_total = 0
                    adam_grid_trial_epochs = 0
                    adam_grid_completed_epochs = 0
                    adam_grid_active_trial = 0
                    lambda_task_id: int | None = None
                    lambda_done = 0
                    lambda_total = 0

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
                        nonlocal cv_task_id
                        if not show_method_progress:
                            return
                        if cv_task_id is None:
                            cv_task_id = progress.add_task(
                                description="",
                                total=cv_total,
                                label=f"{m_disp} cv",
                                counter=f"{cv_done}/{cv_total}",
                            )

                    def _adam_stage_key(payload: dict[str, typing.Any]) -> str:
                        stage = str(payload.get("stage", "fit")).strip().lower()
                        if stage == "pcg":
                            return "pcg"
                        return "grid" if stage == "grid" else "fit"

                    def _adam_label(stage: str, done: int, total: int) -> str:
                        if str(stage) == "grid":
                            return f"{m_disp} adam search"
                        if str(stage) == "pcg":
                            return f"{m_disp} pcg"
                        return f"{m_disp} adam"

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
                    ) -> None:
                        nonlocal adam_task_id, adam_done, adam_total, adam_stage
                        stage, target_done, target_total = _adam_progress_values(str(event), payload)
                        adam_stage = str(stage)
                        adam_done = int(target_done)
                        adam_total = int(max(1, int(target_total)))
                        label = _adam_label(adam_stage, adam_done, adam_total)
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
                        if adam_task_id is None:
                            return
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

                    def _ensure_lambda_task(
                        *,
                        done: int,
                        total: int,
                    ) -> None:
                        nonlocal lambda_task_id, lambda_done, lambda_total
                        lambda_done = int(max(0, int(done)))
                        lambda_total = int(max(1, int(total)))
                        if lambda_task_id is None:
                            lambda_task_id = progress.add_task(
                                description="",
                                total=lambda_total,
                                label=f"{m_disp} lambda search",
                                counter=f"{lambda_done}/{lambda_total}",
                            )
                        progress.update(
                            lambda_task_id,
                            total=lambda_total,
                            completed=min(lambda_done, lambda_total),
                            label=f"{m_disp} lambda search",
                            counter=f"{lambda_done}/{lambda_total}",
                        )

                    def _close_lambda_task(*, complete: bool = False) -> None:
                        nonlocal lambda_task_id, lambda_done, lambda_total
                        if lambda_task_id is not None:
                            if complete:
                                progress.update(
                                    lambda_task_id,
                                    total=max(1, int(lambda_total)),
                                    completed=max(0, int(lambda_total)),
                                    label=f"{m_disp} lambda search",
                                    counter=f"{int(max(0, lambda_total))}/{int(max(1, lambda_total))}",
                                )
                            try:
                                progress.remove_task(lambda_task_id)
                            except Exception:
                                pass
                        lambda_task_id = None
                        lambda_done = 0
                        lambda_total = 0

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
                        if not has_rrblup_iter_progress:
                            return
                        ev = str(event)
                        if ev == "pcg_lambda_subsample_start":
                            total = int(max(1, int(payload.get("total", payload.get("repeats", 1)))))
                            _ensure_lambda_task(done=0, total=total)
                            return
                        if ev == "pcg_lambda_subsample_iter":
                            total = int(max(1, int(payload.get("total", max(1, int(lambda_total or 1))))))
                            done = int(max(0, int(payload.get("iter", lambda_done))))
                            _ensure_lambda_task(done=min(done, total), total=total)
                            return
                        if ev == "pcg_lambda_subsample_end":
                            total = int(max(1, int(payload.get("total", payload.get("repeats", max(1, int(lambda_total or 1)))))))
                            done = int(max(0, int(payload.get("iter", payload.get("ok_repeats", total)))))
                            _ensure_lambda_task(done=min(done, total), total=total)
                            _close_lambda_task(complete=True)
                            return
                        if ev == "pcg_start":
                            _close_lambda_task()
                            pp = dict(payload)
                            pp["stage"] = "pcg"
                            _ensure_adam_task(pp, event="start")
                            return
                        if ev == "pcg_iter":
                            pp = dict(payload)
                            pp["stage"] = "pcg"
                            if adam_task_id is None:
                                _ensure_adam_task(pp, event="epoch")
                            if adam_task_id is None:
                                return
                            stage, target, total = _adam_progress_values("epoch", pp)
                            adam_stage = str(stage)
                            adam_done = int(target)
                            adam_total = int(max(1, int(total)))
                            progress.update(
                                adam_task_id,
                                total=adam_total,
                                completed=adam_done,
                                label=_adam_label(adam_stage, adam_done, adam_total),
                                counter=_adam_counter(adam_done, adam_total),
                            )
                            return
                        if ev == "pcg_end":
                            pp = dict(payload)
                            pp["stage"] = "pcg"
                            if "iter" not in pp and ("iters" in pp):
                                pp["iter"] = pp.get("iters")
                            if adam_task_id is None:
                                _ensure_adam_task(pp, event="end")
                            if adam_task_id is None:
                                return
                            stage, target, total = _adam_progress_values("end", pp)
                            adam_stage = str(stage)
                            adam_done = int(target)
                            adam_total = int(max(1, int(total)))
                            progress.update(
                                adam_task_id,
                                total=adam_total,
                                completed=adam_done,
                                label=_adam_label(adam_stage, adam_done, adam_total),
                                counter=_adam_counter(adam_done, adam_total),
                            )
                            # Reset per-round PCG sub-progress elapsed timer:
                            # close current task at round end so next round creates
                            # a fresh task with elapsed starting from 0.
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
                                label=_adam_label(adam_stage, adam_done, adam_total),
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
                                label=_adam_label(adam_stage, adam_done, adam_total),
                                counter=_adam_counter(adam_done, adam_total),
                            )

                    def _cv_hook(method_name: str, inc: int) -> None:
                        nonlocal cv_done
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
                                label=f"{m_disp} cv",
                                counter=f"{cv_done}/{cv_total}",
                            )

                    if show_method_progress and (not has_search_stage):
                        _ensure_cv_task()
                    t0 = time.monotonic()
                    try:
                        res = _run_method_task(
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
                            packed_ctx=packed_ctx,
                            train_sample_indices=train_sample_indices,
                            test_sample_indices=test_sample_indices,
                            progress_queue=None,
                            progress_hook=_cv_hook,
                            search_progress_hook=_search_hook,
                            rrblup_progress_hook=(_adam_hook if has_rrblup_iter_progress else None),
                            limit_predtrain=limit_predtrain,
                            rrblup_solver=rrblup_solver,
                            rrblup_adamw_cfg=rrblup_adamw_cfg,
                            bayes_auto_r2_cache=bayes_auto_r2_cache_shared,
                            bayes_auto_r2_cfg=bayes_auto_r2_cfg,
                        )
                    except Exception:
                        _close_search_task()
                        _close_adam_task()
                        _close_lambda_task()
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
                    if show_method_progress:
                        _ensure_cv_task()
                        left_cv = max(0, cv_total - int(cv_done))
                        if left_cv > 0 and cv_task_id is not None:
                            cv_done += left_cv
                            progress.update(
                                cv_task_id,
                                advance=left_cv,
                                label=f"{m_disp} cv",
                                counter=f"{cv_done}/{cv_total}",
                            )
                        if cv_task_id is not None:
                            try:
                                progress.remove_task(cv_task_id)
                            except Exception:
                                pass
                    _close_adam_task()
                    _close_lambda_task()
                    out.append(res)
                    elapsed = format_elapsed(time.monotonic() - t0)
                    progress.console.print(
                        f"[green]{success_symbol()} {m_disp} ...Finished [{elapsed}][/green]"
                    )
                    _emit_method_details_plain(
                        m,
                        res,
                        line_printer=lambda s: progress.console.print(s, highlight=False),
                    )
            return out
        for m in methods:
            t0 = time.monotonic()
            method_offset = _method_offset(str(m))
            m_disp = _method_display(str(m))
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
            lambda_done = 0
            lambda_total = 0

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
                    desc=f"{m_disp} cv",
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

            def _adam_desc(stage: str, done: int, total: int) -> str:
                if str(stage) == "grid":
                    return f"{m_disp} adam search"
                if str(stage) == "pcg":
                    return f"{m_disp} pcg"
                return f"{m_disp} adam"

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
            ) -> None:
                nonlocal adam_bar, adam_done, adam_total, adam_stage
                if not enable_tqdm_progress:
                    return
                stage, target_done, target_total = _adam_progress_values(str(event), payload)
                adam_stage = str(stage)
                adam_done = int(target_done)
                adam_total = int(max(1, int(target_total)))
                desc = _adam_desc(adam_stage, adam_done, adam_total)
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
                    if adam_done > 0:
                        adam_bar.n = int(min(adam_done, adam_total))
                else:
                    adam_bar.total = int(max(1, adam_total))
                    adam_bar.n = int(min(max(0, adam_done), max(1, adam_total)))
                    adam_bar.set_description_str(desc, refresh=False)
                adam_bar.set_postfix_str("", refresh=False)
                adam_bar.refresh()

            def _close_adam_bar() -> None:
                nonlocal adam_bar, adam_done, adam_total, adam_stage
                nonlocal adam_grid_trial_total, adam_grid_trial_epochs
                nonlocal adam_grid_completed_epochs, adam_grid_active_trial
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

            def _ensure_lambda_bar(
                payload: dict[str, typing.Any],
                *,
                event: str = "start",
            ) -> None:
                nonlocal lambda_bar, lambda_done, lambda_total
                if not enable_tqdm_progress:
                    return
                total = int(max(1, int(payload.get("total", payload.get("repeats", max(1, int(lambda_total or 1)))))))
                if str(event) == "start":
                    done = 0
                elif str(event) == "end":
                    done = int(max(0, int(payload.get("iter", payload.get("ok_repeats", total)))))
                else:
                    done = int(max(0, int(payload.get("iter", lambda_done))))
                done = int(min(done, total))
                lambda_done = int(done)
                lambda_total = int(total)
                if lambda_bar is None:
                    assert tqdm is not None
                    lambda_bar = tqdm(
                        total=lambda_total,
                        desc=f"{m_disp} lambda search",
                        unit="cfg",
                        leave=False,
                        dynamic_ncols=True,
                        position=(2 if show_method_progress else 1),
                        bar_format="{desc}: |{bar}| {n_fmt}/{total_fmt} "
                        "[{elapsed}, {rate_fmt}{postfix}]",
                    )
                lambda_bar.total = int(max(1, lambda_total))
                lambda_bar.n = int(min(max(0, lambda_done), max(1, lambda_total)))
                lambda_bar.set_description_str(
                    f"{m_disp} lambda search",
                    refresh=False,
                )
                lambda_bar.set_postfix_str("", refresh=False)
                lambda_bar.refresh()

            def _close_lambda_bar() -> None:
                nonlocal lambda_bar, lambda_done, lambda_total
                if lambda_bar is not None:
                    lambda_bar.close()
                    lambda_bar = None
                lambda_done = 0
                lambda_total = 0

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
                if not has_rrblup_iter_progress:
                    return
                ev = str(event)
                if ev == "pcg_lambda_subsample_start":
                    _ensure_lambda_bar(payload, event="start")
                    return
                if ev == "pcg_lambda_subsample_iter":
                    _ensure_lambda_bar(payload, event="iter")
                    return
                if ev == "pcg_lambda_subsample_end":
                    _ensure_lambda_bar(payload, event="end")
                    _close_lambda_bar()
                    return
                if ev == "pcg_start":
                    _close_lambda_bar()
                    pp = dict(payload)
                    pp["stage"] = "pcg"
                    _ensure_adam_bar(pp, event="start")
                    return
                if ev == "pcg_iter":
                    pp = dict(payload)
                    pp["stage"] = "pcg"
                    if adam_bar is None:
                        _ensure_adam_bar(pp, event="epoch")
                    if adam_bar is None:
                        return
                    stage, target, total = _adam_progress_values("epoch", pp)
                    adam_stage = str(stage)
                    adam_done = int(target)
                    adam_total = int(max(1, int(total)))
                    adam_bar.total = adam_total
                    adam_bar.n = int(min(max(0, adam_done), adam_total))
                    adam_bar.set_description_str(
                        _adam_desc(adam_stage, adam_done, adam_total),
                        refresh=False,
                    )
                    adam_bar.set_postfix_str("", refresh=False)
                    adam_bar.refresh()
                    return
                if ev == "pcg_end":
                    pp = dict(payload)
                    pp["stage"] = "pcg"
                    if "iter" not in pp and ("iters" in pp):
                        pp["iter"] = pp.get("iters")
                    if adam_bar is None:
                        _ensure_adam_bar(pp, event="end")
                    if adam_bar is None:
                        return
                    stage, target, total = _adam_progress_values("end", pp)
                    adam_stage = str(stage)
                    adam_done = int(target)
                    adam_total = int(max(1, int(total)))
                    adam_bar.total = adam_total
                    adam_bar.n = int(min(max(0, adam_done), adam_total))
                    adam_bar.set_description_str(
                        _adam_desc(adam_stage, adam_done, adam_total),
                        refresh=False,
                    )
                    adam_bar.set_postfix_str("", refresh=False)
                    adam_bar.refresh()
                    # Reset per-round PCG sub-progress elapsed timer by closing
                    # this bar at the end of each PCG round.
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
                        _adam_desc(adam_stage, adam_done, adam_total),
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
                        _adam_desc(adam_stage, adam_done, adam_total),
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
                        f"{m_disp} cv",
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
                    res = _run_method_task(
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
                        packed_ctx=packed_ctx,
                        train_sample_indices=train_sample_indices,
                        test_sample_indices=test_sample_indices,
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
                    )
                except Exception:
                    _close_search_bar()
                    _close_adam_bar()
                    _close_lambda_bar()
                    _close_cv_bar()
                    elapsed = format_elapsed((time.monotonic() - t0) + method_offset)
                    if method_status is not None:
                        method_status.fail(f"{m_disp} ...Failed")
                    else:
                        print_failure(f"{m_disp} ...Failed [{elapsed}]", force_color=True)
                    raise

                _close_search_bar()
                _close_lambda_bar()
                if enable_tqdm_progress and show_method_progress:
                    _ensure_cv_bar()
                    if cv_bar is not None:
                        left_cv = max(0, cv_total - int(cv_done))
                        if left_cv > 0:
                            cv_done += left_cv
                            cv_bar.update(left_cv)
                        cv_bar.set_description_str(
                            f"{m_disp} cv",
                            refresh=True,
                        )
                _close_adam_bar()
                _close_lambda_bar()
                _close_cv_bar()

                out.append(res)
                elapsed = format_elapsed((time.monotonic() - t0) + method_offset)
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
                    results_by_method[method] = fut.result()
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
                elapsed = format_elapsed(
                    (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                    + _method_offset(str(method))
                )
                method_disp = _method_display(str(method))
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
                            results_by_method[method] = res
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
                        results_by_method[method] = res
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
                results_by_method[method] = fut.result()

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
    maf = np.ascontiguousarray(np.asarray(packed_ctx["maf"], dtype=np.float32).reshape(-1))
    n_samples = int(packed_ctx["n_samples"])
    if packed.ndim != 2:
        raise ValueError("Invalid packed payload: packed must be 2D.")
    if int(packed.shape[0]) != int(maf.shape[0]):
        raise ValueError("Packed payload shape mismatch between packed rows and maf.")

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

    ridx = np.ascontiguousarray(np.arange(int(packed.shape[0]), dtype=np.int64))
    decoded = _jxrs.bed_packed_decode_rows_f32(
        packed,
        int(n_samples),
        ridx,
        row_flip,
        maf,
        None,
    )
    dense = np.ascontiguousarray(np.asarray(decoded, dtype=np.float32), dtype=np.float32)
    if dense.ndim != 2 or dense.shape != (int(packed.shape[0]), int(n_samples)):
        raise ValueError(
            "Packed decode shape mismatch: "
            f"expected ({int(packed.shape[0])},{int(n_samples)}), got {dense.shape}"
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
        default="auto",
        help=argparse.SUPPRESS,
    )
    optional_group.add_argument(
        "--rrblup-lambda",
        type=float,
        default=10000.0,
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
        "auto_pcg_min_n": int(args.rrblup_auto_pcg_min_n),
        "lambda_subsample_n": int(args.rrblup_lambda_subsample_n),
        "lambda_subsample_min_n": int(_RRBLUP_LAMBDA_SUBSAMPLE_MIN_N),
        "lambda_subsample_max_n": int(_RRBLUP_LAMBDA_SUBSAMPLE_MAX_N),
        "lambda_subsample_repeats": int(args.rrblup_lambda_subsample_repeats),
        "lambda_subsample_seed": int(args.rrblup_lambda_subsample_seed),
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
        "pcg_block_rows": int(args.rrblup_snp_block_size),
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
    configure_genotype_cache_from_out(args.out)
    log_path = f"{outprefix}.gs.log"
    logger = setup_logging(log_path)
    if thread_capped:
        logger.warning(
            f"Requested threads={requested_threads} exceeds detected available={detected_threads}; "
            f"using {int(args.thread)}."
        )

    manual_packed_preprocess_requested = bool(
        (args.hash_dim is not None) or (getattr(args, "ldprune_spec", None) is not None)
    )
    rr_solver_mode = str(rrblup_solver).strip().lower()
    pcg_requested = bool(
        bool(args.rrBLUP) and (rr_solver_mode == "pcg")
    )
    gfile_is_plink = _is_plink_prefix(str(gfile))
    gblup_add_requested = bool("a" in gblup_modes)
    gblup_nonadd_requested = bool(any(m in {"d", "ad"} for m in gblup_modes))
    bayes_requested = bool(bool(args.BayesA) or bool(args.BayesB) or bool(args.BayesCpi))
    packed_model_requested = bool(gblup_add_requested or bool(args.rrBLUP) or bayes_requested)
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
        and (bool(args.vcf) or bool(args.hmp) or bool(args.file))
    ):
        # For Bayes on VCF/HMP inputs, prefer packed path by default to avoid
        # dense genotype peak memory. Keep legacy size-threshold probing for
        # other packed-eligible model combinations.
        if bayes_requested and (bool(args.vcf) or bool(args.hmp)):
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
        and (bool(args.vcf) or bool(args.hmp) or bool(args.file))
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
        and (bool(args.vcf) or bool(args.hmp) or bool(args.file))
    )
    hard_packed_preprocess_requested = bool(
        manual_packed_preprocess_requested or pcg_packed_preprocess_requested
    )
    packed_preprocess_requested = bool(
        hard_packed_preprocess_requested or auto_packed_lmm_requested
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
                        # For -file inputs, allow TXT/TSV/CSV sources to be packed
                        # into PLINK BED for packed preprocessing / auto packed LMM flow.
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
                            "when source resolves to VCF/HMP/TXT. "
                            f"Current input could not be converted to BED: {cached_prefix}. "
                            "For -file inputs backed by .npy/.bin, please convert to -bfile first."
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
    if len(methods) == 0:
        logger.error(
            "No model selected. Use "
            "--GBLUP [a|d|ad]/--rrBLUP/--BayesA/--BayesB/--BayesCpi/--RF/--ET/--GBDT/--XGB/--SVM/--ENET."
        )
        raise SystemExit(1)
    ml_methods = {"RF", "ET", "GBDT", "XGB", "SVM", "ENET"}
    gblup_d_like_methods = [
        m for m in methods
        if _is_gblup_method(str(m)) and (_gblup_method_kernel_mode(str(m)) in {"d", "ad"})
    ]
    if (len(gblup_d_like_methods) > 0) and (not _HAS_ADBLUP_PY):
        methods = [m for m in methods if m not in set(gblup_d_like_methods)]
        logger.warning(
            "Skip GBLUP(d/ad) because JAX backend dependency is unavailable. "
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
            "off"
            if args.cv is None
            else f"{int(args.cv)}-fold, strict={bool(args.strict_cv)}"
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
                        ("Train prediction", ("All" if args.limit_predtrain is None else int(args.limit_predtrain))),
                    ],
                ),
                ("Enabled models", model_rows),
            ],
            footer_rows=[("Output prefix", outprefix)],
            line_max_chars=62,
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
    packed_eligible_methods = {
        _GBLUP_METHOD_ADD,
        "rrBLUP",
        "BayesA",
        "BayesB",
        "BayesCpi",
    }
    use_packed_lmm = bool(
        gfile_is_plink
        and len(methods) > 0
        and all(m in packed_eligible_methods for m in methods)
    )
    ldprune_spec = typing.cast(
        _GsLdPruneSpec | None,
        getattr(args, "ldprune_spec", None),
    )
    preprocess_qc_requested = bool(
        (args.hash_dim is not None) or (ldprune_spec is not None)
    )
    packed_lmm_ctx: dict[str, typing.Any] | None = None
    packed_qc_baseline_ctx: dict[str, typing.Any] | None = None
    if use_packed_lmm:
        with CliStatus(f"Loading genotype from {gsrc}...", enabled=use_spinner) as task:
            try:
                sample_ids, packed_lmm_ctx = _load_plink_packed_for_lmm(
                    str(gfile),
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            n = int(sample_ids.shape[0])
            m = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
            task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m}, packed)")
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
                )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            n = int(sample_ids.shape[0])
            m = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
            task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m}, packed)")
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
    elif gfile_is_plink and (ldprune_spec is not None):
        # For non-GBLUP/rrBLUP model sets, allow global LD prune by
        # pruning on packed BED first, then decoding once to dense matrix.
        with CliStatus(f"Loading genotype from {gsrc}...", enabled=use_spinner) as task:
            try:
                sample_ids, packed_lmm_ctx = _load_plink_packed_for_lmm(
                    str(gfile),
                    maf=float(args.maf),
                    missing_rate=float(args.geno),
                )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            n = int(sample_ids.shape[0])
            m = int(packed_lmm_ctx["packed"].shape[0]) if packed_lmm_ctx is not None else 0
            task.complete(f"Loading genotype from {gsrc} (n={n}, nSNP={m}, packed)")
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
                sample_ids, geno = _load_genotype_with_rust_gfreader(
                    gfile,
                    maf=args.maf,
                    missing_rate=args.geno,
                )
            except Exception:
                task.fail(f"Loading genotype from {gsrc} ...Failed")
                raise
            m, n = geno.shape
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
        if need_std_geno:
            geno = (geno - geno.mean(axis=1, keepdims=True)) / (
                geno.std(axis=1, keepdims=True) + 1e-6
            )  # standardization for marker-effect models

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
                    args.cv is not None
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
                    args.cv is None
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
        if args.cv is not None:
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
                int(trait_packed_ctx["packed"].shape[0])
                if (trait_use_packed_lmm and trait_packed_ctx is not None)
                else int(train_snp.shape[0] if train_snp is not None else 0)
            )
        if gblup_backend_label is None and (_GBLUP_METHOD_ADD in methods) and (train_snp is not None):
            gblup_backend_n = int(train_pheno.shape[0])
            gblup_backend_m = int(train_snp.shape[0])
            gblup_backend_label = "ZTZ" if gblup_backend_n > gblup_backend_m else "kinship"

        sep_w = max(40, min(80, _terminal_columns()))
        logger.info("*" * sep_w)
        logger.info(
            f"* Genomic Selection for trait: {trait_name}\n"
            f"Train size: {np.sum(trainmask)}, Test size: {np.sum(testmask)}, EffSNPs: {eff_snp}"
        )
        if (
            bool(debug_mode)
            and gblup_backend_label is not None
            and gblup_backend_n is not None
            and gblup_backend_m is not None
        ):
            logger.info(
                f"GBLUP backend(auto): {gblup_backend_label} "
                f"(n={gblup_backend_n}, m={gblup_backend_m})"
            )
        t_stage = time.time()
        method_results = _run_methods_parallel(
            methods=methods,
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
            train_sample_indices=train_sample_idx,
            test_sample_indices=test_sample_idx,
            limit_predtrain=args.limit_predtrain,
            elapsed_offset_by_method=method_elapsed_offsets,
            rrblup_solver=rrblup_solver,
            rrblup_adamw_cfg=rrblup_adamw_cfg,
            bayes_auto_r2_cfg=bayes_auto_r2_cfg,
        )
        if _GS_DEBUG_STAGE:
            logger.info(
                f"[GS-DEBUG][{trait_name}] stage=run_methods_parallel "
                f"elapsed={time.time() - t_stage:.3f}s"
            )
        method_result_map = {str(x["method"]): x for x in method_results}
        method_display_map = {str(m): _method_display_name(str(m)) for m in methods}

        if args.cv is not None:
            logger.info("-" * 60)
            row_fmt = (
                "{fold:>4} "
                "{method:<10} "
                "{pear:>8} "
                "{spear:>9} "
                "{r2:>6} "
                "{h2:>6} "
                "{time:>10}"
            )
            logger.info(
                row_fmt.format(
                    fold="Fold",
                    method="Method",
                    pear="Pearsonr",
                    spear="Spearmanr",
                    r2="R2",
                    h2="h2/PVE",
                    time="time(secs)",
                )
            )
            method_order = {str(m): i for i, m in enumerate(methods)}
            all_rows: list[tuple[str, int, float, float, float, float, float]] = []
            for method in methods:
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
            for row in all_rows:
                m_name, fold_id, pear, spear, r2, pve, t_fold = row
                logger.info(
                    row_fmt.format(
                        fold=int(fold_id),
                        method=method_display_map.get(str(m_name), str(m_name)),
                        pear=f"{float(pear):.3f}",
                        spear=f"{float(spear):.3f}",
                        r2=f"{float(r2):.3f}",
                        h2=f"{float(pve):.3f}",
                        time=f"{float(t_fold):.3f}",
                    )
                )
        if args.cv is not None:
            for method in methods:
                res = method_result_map.get(method)
                if res is None:
                    continue
                best_test = res.get("best_test")
                best_train = res.get("best_train")
                if best_test is None or best_train is None:
                    continue
                fig = plt.figure(figsize=(5, 4), dpi=300)
                gsplot.scatterh(best_test, best_train, color_set=color_set[0], fig=fig)
                for ax in fig.axes:
                    if str(ax.get_ylabel()).strip() == "Predicted Value":
                        ax.set_ylabel(
                            f"Predicted Value ({method_display_map.get(str(method), str(method))})"
                        )
                        break
                out_svg = f"{outprefix}.{trait_name}.gs.{method}.svg"
                # Avoid tight_layout warnings on dense decorations.
                fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.98)
                fig.savefig(
                    out_svg,
                    transparent=False,
                    facecolor="white",
                    bbox_inches="tight",
                )
                plt.close(fig)

        missing_methods = [m for m in methods if m not in method_result_map]
        if len(missing_methods) > 0:
            raise RuntimeError(
                "Missing GS results for methods: " + ", ".join(missing_methods)
            )
        expected_test_n = int(np.sum(testmask))
        outpred_list: list[np.ndarray] = []
        for m in methods:
            pred_arr = np.asarray(method_result_map[m]["test_pred"], dtype=float).reshape(-1, 1)
            n_pred = int(pred_arr.shape[0])
            if n_pred == expected_test_n:
                outpred_list.append(pred_arr)
                continue
            if (n_pred == 0) and (expected_test_n > 0):
                logger.warning(
                    f"{m} returned empty test predictions while expected n_test={expected_test_n}; "
                    "filling with NaN."
                )
                outpred_list.append(np.full((expected_test_n, 1), np.nan, dtype=float))
                continue
            raise RuntimeError(
                f"Prediction length mismatch for method={m}: "
                f"got n_pred={n_pred}, expected n_test={expected_test_n}."
            )
        for m in methods:
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
                    "n_train": int(np.sum(trainmask)),
                    "n_test": int(expected_test_n),
                    "pearsonr_cv_mean": pear_mean,
                    "spearmanr_cv_mean": spear_mean,
                    "r2_cv_mean": r2_mean,
                    "pve_final": pve_val,
                    "time_cv_mean_sec": elapsed_mean,
                }
            )
        logger.info(f"-"*60) if args.cv is not None else None
        # Stack predictions from all models: shape (n_test, n_methods)
        outpred = pd.DataFrame(
            np.concatenate(outpred_list, axis=1),
            columns=[method_display_map.get(str(m), str(m)) for m in methods],
            index=samples[testmask],
        )
        out_tsv = f"{outprefix}.{trait_name}.gs.tsv"
        outpred.to_csv(out_tsv, sep="\t", float_format="%.4f")
        saved_result_paths.append(str(out_tsv))
        log_success(logger, f"Saved predictions to {format_path_for_display(out_tsv)}")
        log_success(logger, f"Trait {trait_name} finished in {(time.time() - t_trait):.2f} secs")

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
