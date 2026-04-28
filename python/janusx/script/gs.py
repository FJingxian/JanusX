# -*- coding: utf-8 -*-
"""
JanusX: Genomic Selection Command-Line Interface

Supported models
----------------
  - GBLUP  : Genomic Best Linear Unbiased Prediction (GBLUP, kinship = 1)
  - adBLUP : Additive + dominance kernel BLUP (pyBLUP/JAX backend)
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
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.helptext import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_file_input_exists,
    format_path_for_display,
    ensure_plink_prefix_exists,
)
from ._common.progress import ProgressAdapter, build_rich_progress, rich_progress_available
from ._common.status import (
    CliStatus,
    failure_symbol,
    log_success,
    print_success,
    print_failure,
    format_elapsed,
    stdout_is_tty,
    success_symbol,
)
from ._common.genocache import configure_genotype_cache_from_out
from ._common.colspec import parse_zero_based_index_specs
from ._common.threads import maybe_warn_non_openblas, require_openblas_by_default

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
    if p not in {"auto", "align", "blas", "balanced", "split", "rust"}:
        p = "auto"
    if p == "align":
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
    # Default to rust-priority runtime split for better stability on large packed-BED workloads.
    policy_raw = str(os.getenv("JX_THREAD_POLICY", "rust")).strip().lower() or "rust"
    explicit_blas = _parse_nonnegative_int(os.getenv("JX_MLM_BLAS_THREADS", "").strip())
    explicit_rust = _parse_nonnegative_int(os.getenv("JX_MLM_RUST_THREADS", "").strip())
    gblup_only = bool(len(methods) > 0 and all(m in {"GBLUP", "rrBLUP"} for m in methods))

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
    if (not gblup_only) and (not bool(plan["explicit"])):
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


def GSapi(
    Y: np.ndarray,
    Xtrain: typing.Any,
    Xtest: typing.Any,
    method: typing.Literal["GBLUP", "adBLUP", "rrBLUP", "BayesA", "BayesB", "BayesCpi", "RF", "ET", "GBDT", "XGB", "SVM", "ENET"],
    PCAdec: bool = False,
    n_jobs: int = 1,
    force_fast: bool = False,
    ml_fixed_params: dict[str, typing.Any] | None = None,
    need_train_pred: bool = True,
    packed_train_indices: np.ndarray | None = None,
    packed_test_indices: np.ndarray | None = None,
    train_pred_indices: np.ndarray | None = None,
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
    method : {'GBLUP', 'adBLUP', 'rrBLUP', 'BayesA', 'BayesB', 'BayesCpi', 'RF', 'ET', 'GBDT', 'XGB', 'SVM', 'ENET'}
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

    # Multi-kernel additive+dominance BLUP (pyBLUP/JAX backend)
    if method == "adBLUP":
        if (not _HAS_ADBLUP_PY) or KernelBLUP is None or Gmatrix is None:
            raise ImportError(
                "adBLUP (JAX backend) is unavailable. "
                f"Original import error: {_ADBLUP_PY_IMPORT_ERROR}"
            )
        gadd_train = np.asarray(Xtrain, dtype=np.float32)
        gadd_test = np.asarray(Xtest, dtype=np.float32)
        if gadd_train.ndim != 2 or gadd_test.ndim != 2:
            raise ValueError(
                f"adBLUP expects 2D genotype matrices. got train={gadd_train.shape}, test={gadd_test.shape}"
            )
        if gadd_train.shape[0] != gadd_test.shape[0]:
            raise ValueError(
                f"adBLUP marker mismatch: train={gadd_train.shape[0]}, test={gadd_test.shape[0]}"
            )

        ghet_train = (gadd_train == 1.0).astype(np.float32, copy=False)
        Gadd_train = Gmatrix(gadd_train)
        Ghet_train = Gmatrix(ghet_train)
        model = KernelBLUP(Y.reshape(-1, 1), G=[Gadd_train, Ghet_train], progress=False)

        if need_train_pred:
            yhat_train = model.predict(G=[Gadd_train, Ghet_train])
        else:
            yhat_train = np.zeros((0, 1), dtype=float)
        n_train = int(gadd_train.shape[1])
        if int(gadd_test.shape[1]) > 0:
            gadd_all = np.concatenate([gadd_train, gadd_test], axis=1)
            ghet_all = (gadd_all == 1.0).astype(np.float32, copy=False)
            Gadd_all = Gmatrix(gadd_all)
            Ghet_all = Gmatrix(ghet_all)
            Gadd_cross = Gadd_all[n_train:, :n_train]
            Ghet_cross = Ghet_all[n_train:, :n_train]
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
        kinship = 1 if method == "GBLUP" else None
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
                f"force_fast={int(bool(force_fast))}",
                flush=True,
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
        if _GS_DEBUG_STAGE:
            print(
                f"[GS-DEBUG] GSapi predict_done method={method} "
                f"elapsed={time.time() - t_pred:.3f}s",
                flush=True,
            )
        return pred_train, pred_test, model.pve

    if method in ("BayesA", "BayesB", "BayesCpi"):
        model = BAYES(Y.reshape(-1, 1), Xtrain, method=method)
        pve = model.pve
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
    limit_predtrain: int | None = None,
) -> dict[str, typing.Any]:
    fold_rows: list[tuple[str, int, float, float, float, float, float]] = []
    best_test = None
    best_train = None
    best_r2 = -np.inf
    ml_tuning_cache: dict[str, typing.Any] | None = None
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
            "fold_rows": [],
            "best_test": None,
            "best_train": None,
            "test_pred": pred,
            "pve_final": float(pve_final),
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
        if precomputed_train_grm is not None:
            gblup_cv_grm_full = np.asarray(precomputed_train_grm, dtype=np.float64)
            if precomputed_test_train_grm is not None:
                gblup_final_test_train = np.asarray(
                    precomputed_test_train_grm,
                    dtype=np.float64,
                )
            gblup_force_kinship = True
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
            or (packed_payload is not None)
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
                if (packed_payload is not None) and (method in ("GBLUP", "rrBLUP")):
                    if train_sample_indices is None:
                        raise ValueError("Packed LMM requires train sample indices.")
                    fold_train = packed_payload
                    fold_test = packed_payload
                    fold_train_idx_arg = np.ascontiguousarray(train_sample_indices[fold_train_idx], dtype=np.int64)
                    fold_test_idx_arg = np.ascontiguousarray(train_sample_indices[fold_test_idx], dtype=np.int64)
                    fold_pca = False
                elif method == "adBLUP":
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
            "fold_rows": fold_rows,
            "best_test": best_test,
            "best_train": best_train,
            "test_pred": np.asarray(test_pred, dtype=float).reshape(-1, 1),
            "pve_final": float(fit_final["pve"]),
        }

    final_train_idx_arg = None
    final_test_idx_arg = None
    if (packed_payload is not None) and (method in ("GBLUP", "rrBLUP")):
        if train_sample_indices is None or test_sample_indices is None:
            raise ValueError("Packed LMM requires train/test sample indices.")
        final_train = packed_payload
        final_test = packed_payload
        final_pca = False
        final_train_idx_arg = np.ascontiguousarray(train_sample_indices, dtype=np.int64)
        final_test_idx_arg = np.ascontiguousarray(test_sample_indices, dtype=np.int64)
    elif method == "adBLUP":
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
    skip_final_fit = False
    use_custom_gblup_final = bool(
        method == "GBLUP"
        and gblup_cv_grm_full is not None
        and (gblup_force_kinship or precomputed_train_grm is not None)
    )
    if cv_splits is not None:
        if use_custom_gblup_final:
            skip_final_fit = bool(
                gblup_final_test_train is None
                or int(gblup_final_test_train.shape[0]) == 0
            )
        elif (packed_payload is not None) and (method in ("GBLUP", "rrBLUP")):
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
        )
        if ml_tuning_cache is not None:
            pve_final = float(ml_tuning_cache.get("pve", np.nan))
    return {
        "method": method,
        "fold_rows": fold_rows,
        "best_test": best_test,
        "best_train": best_train,
        "test_pred": test_pred,
        "pve_final": float(pve_final),
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
) -> list[dict[str, typing.Any]]:
    if len(methods) == 0:
        return []

    model_n_jobs = max(1, int(n_jobs))
    offset_map = dict(elapsed_offset_by_method or {})

    def _method_offset(name: str) -> float:
        try:
            return max(0.0, float(offset_map.get(str(name), 0.0)))
        except Exception:
            return 0.0

    show_method_progress = cv_splits is not None
    # Force serial execution for all GS methods to avoid cross-method contention
    # and keep runtime behavior consistent.
    method_parallel_jobs = 1
    if method_parallel_jobs <= 1:
        out: list[dict[str, typing.Any]] = []
        fold_total_map = {
            m: (int(len(cv_splits)) if cv_splits is not None else 1)
            for m in methods
        }
        animate_method_progress = bool(show_method_progress)
        if animate_method_progress and rich_progress_available():
            progress = build_rich_progress(
                description_template="{task.fields[label]}",
                show_elapsed=True,
                show_remaining=False,
                finished_text=" ",
                transient=True,
            )
            assert progress is not None
            with progress:
                for m in methods:
                    cv_total = int(max(1, fold_total_map.get(m, 1)))
                    has_search_stage = bool((m in _ML_METHOD_MAP) and (not strict_cv))
                    search_task_id: int | None = None
                    search_done = 0
                    search_total = 0
                    search_total_fixed = False
                    cv_task_id: int | None = None
                    cv_done = 0

                    def _close_search_task() -> None:
                        nonlocal search_task_id, search_done, search_total
                        if search_task_id is None:
                            return
                        left = max(0, int(search_total) - int(search_done))
                        if left > 0:
                            progress.update(
                                search_task_id,
                                advance=left,
                                label=f"{m} search {int(search_total)}/{int(search_total)}",
                            )
                        try:
                            progress.remove_task(search_task_id)
                        except Exception:
                            pass
                        search_task_id = None

                    def _ensure_cv_task() -> None:
                        nonlocal cv_task_id
                        if cv_task_id is None:
                            cv_task_id = progress.add_task(
                                description="",
                                total=cv_total,
                                label=f"{m} cv {cv_done}/{cv_total}",
                            )

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
                                    label=f"{m} search {search_done}/{search_total}",
                                )
                            else:
                                progress.update(
                                    search_task_id,
                                    total=int(max(1, search_total)),
                                    label=f"{m} search {search_done}/{search_total}",
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
                                    label=f"{m} search {search_done}/{search_total}",
                                )
                            else:
                                progress.update(
                                    search_task_id,
                                    total=int(max(1, search_total)),
                                    label=f"{m} search {search_done}/{search_total}",
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
                                    label=f"{m} search {search_done}/{search_total}",
                                )
                            left = max(0, int(search_total) - int(search_done))
                            adv = min(left, inc)
                            if adv > 0:
                                search_done += adv
                                progress.update(
                                    search_task_id,
                                    total=int(max(1, search_total)),
                                    advance=int(adv),
                                    label=f"{m} search {search_done}/{search_total}",
                                )

                    def _cv_hook(method_name: str, inc: int) -> None:
                        nonlocal cv_done
                        if str(method_name) != str(m):
                            return
                        _close_search_task()
                        _ensure_cv_task()
                        left = max(0, cv_total - int(cv_done))
                        adv = min(left, int(max(0, int(inc))))
                        if adv > 0 and cv_task_id is not None:
                            cv_done += adv
                            progress.update(
                                cv_task_id,
                                advance=adv,
                                label=f"{m} cv {cv_done}/{cv_total}",
                            )

                    if not has_search_stage:
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
                            limit_predtrain=limit_predtrain,
                        )
                    except Exception:
                        _close_search_task()
                        if cv_task_id is not None:
                            try:
                                progress.remove_task(cv_task_id)
                            except Exception:
                                pass
                        elapsed = format_elapsed(time.monotonic() - t0)
                        progress.console.print(
                            f"[red]{failure_symbol()} {m} ...Failed [{elapsed}][/red]"
                        )
                        raise
                    _close_search_task()
                    _ensure_cv_task()
                    left_cv = max(0, cv_total - int(cv_done))
                    if left_cv > 0 and cv_task_id is not None:
                        cv_done += left_cv
                        progress.update(
                            cv_task_id,
                            advance=left_cv,
                            label=f"{m} cv {cv_done}/{cv_total}",
                        )
                    if cv_task_id is not None:
                        try:
                            progress.remove_task(cv_task_id)
                        except Exception:
                            pass
                    out.append(res)
                    elapsed = format_elapsed(time.monotonic() - t0)
                    progress.console.print(
                        f"[green]{success_symbol()} {m} ...Finished [{elapsed}][/green]"
                    )
            return out
        for m in methods:
            t0 = time.monotonic()
            method_offset = _method_offset(str(m))
            cv_total = int(max(1, fold_total_map.get(m, 1)))
            has_search_stage = bool((m in _ML_METHOD_MAP) and (not strict_cv))
            enable_tqdm_progress = bool(show_method_progress and _HAS_TQDM and stdout_is_tty())
            enable_method_spinner = bool(
                (not show_method_progress)
                and stdout_is_tty()
                and (method_offset <= 0.0)
            )
            search_bar = None
            cv_bar = None
            search_done = 0
            search_total = 0
            search_total_fixed = False
            cv_done = 0

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
                if (not enable_tqdm_progress) or cv_bar is not None:
                    return
                assert tqdm is not None
                cv_bar = tqdm(
                    total=cv_total,
                    desc=f"{m} cv",
                    unit="fold",
                    leave=False,
                    dynamic_ncols=True,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| "
                    "[{elapsed}, {rate_fmt}{postfix}]",
                )

            def _ensure_search_bar() -> None:
                nonlocal search_bar, search_total
                if (not enable_tqdm_progress) or search_bar is not None:
                    return
                assert tqdm is not None
                search_bar = tqdm(
                    total=int(max(1, search_total)),
                    desc=f"{m} search",
                    unit="cfg",
                    leave=False,
                    dynamic_ncols=True,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| "
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
                        if stage != "":
                            search_bar.set_postfix_str(stage, refresh=False)
                        search_bar.update(int(adv))

            def _cv_hook(method_name: str, inc: int) -> None:
                nonlocal cv_done
                if str(method_name) != str(m):
                    return
                _close_search_bar()
                _ensure_cv_bar()
                if cv_bar is None:
                    return
                left = max(0, cv_total - int(cv_done))
                adv = min(left, int(max(0, int(inc))))
                if adv > 0:
                    cv_done += adv
                    cv_bar.update(adv)

            if enable_tqdm_progress and (not has_search_stage):
                _ensure_cv_bar()

            method_status = (
                CliStatus(f"Loading {m}...", enabled=True, use_process=True)
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
                        limit_predtrain=limit_predtrain,
                    )
                except Exception:
                    _close_search_bar()
                    _close_cv_bar()
                    elapsed = format_elapsed((time.monotonic() - t0) + method_offset)
                    if method_status is not None:
                        method_status.fail(f"{m} ...Failed")
                    else:
                        print_failure(f"{m} ...Failed [{elapsed}]", force_color=True)
                    raise

                _close_search_bar()
                if enable_tqdm_progress:
                    _ensure_cv_bar()
                    if cv_bar is not None:
                        left_cv = max(0, cv_total - int(cv_done))
                        if left_cv > 0:
                            cv_done += left_cv
                            cv_bar.update(left_cv)
                _close_cv_bar()

                out.append(res)
                elapsed = format_elapsed((time.monotonic() - t0) + method_offset)
                if method_status is not None:
                    method_status.complete(f"{m} ...Finished")
                else:
                    print_success(f"{m} ...Finished [{elapsed}]", force_color=True)
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
                    print_failure(
                        f"{method} ...Failed [{elapsed}]",
                        force_color=True,
                    )
                    raise
                elapsed = format_elapsed(
                    (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                    + _method_offset(str(method))
                )
                print_success(
                    f"{method} ...Finished [{elapsed}]",
                    force_color=True,
                )
    elif bool(show_method_progress and rich_progress_available()):
        progress = build_rich_progress(
            description_template="{task.fields[method]}",
            field_templates=["{task.fields[suffix]}"],
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
                task_map[method] = progress.add_task(
                    description="",
                    total=task_total_map[method],
                    method=method,
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
                            _rich_print_success(method, elapsed)
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
                    _rich_print_failure(method, elapsed)
                raise
            finally:
                try:
                    manager.shutdown()
                except Exception:
                    pass
    elif bool(show_method_progress and _HAS_TQDM and stdout_is_tty()):
        pbar = tqdm(
            total=len(methods),
            desc="Methods",
            unit="method",
            leave=False,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| "
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
                        pbar.update(1)
                        pbar.set_postfix(method=method)
                        elapsed = format_elapsed(
                            (time.monotonic() - method_start_ts.get(method, time.monotonic()))
                            + _method_offset(str(method))
                        )
                        print_success(
                            f"{method} ...Finished [{elapsed}]",
                            force_color=True,
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
    sample_ids, _ = inspect_genotype_file(str(genotype_prefix))
    packed_raw, miss_raw, maf_raw, _std_raw, n_samples = load_bed_2bit_packed(str(genotype_prefix))
    packed_n = int(n_samples)
    sample_ids_arr = np.asarray(sample_ids, dtype=str)
    if packed_n != int(sample_ids_arr.shape[0]):
        raise ValueError(
            f"Packed sample size mismatch: packed n={packed_n}, expected {sample_ids_arr.shape[0]}"
        )
    miss_arr = np.ascontiguousarray(np.asarray(miss_raw, dtype=np.float32).reshape(-1))
    maf_arr = np.ascontiguousarray(np.asarray(maf_raw, dtype=np.float32).reshape(-1))
    keep = np.ones(maf_arr.shape[0], dtype=np.bool_)
    maf_thr = float(maf)
    if maf_thr > 0.0:
        keep &= (maf_arr >= maf_thr) & (maf_arr <= (1.0 - maf_thr))
    miss_thr = float(missing_rate)
    if miss_thr < 1.0:
        keep &= miss_arr <= miss_thr
    if not np.any(keep):
        raise ValueError(
            "No SNPs left after packed BED filtering. Please relax --maf/--geno thresholds."
        )
    site_keep = np.ascontiguousarray(np.asarray(keep, dtype=np.bool_).reshape(-1))
    packed = np.ascontiguousarray(np.asarray(packed_raw, dtype=np.uint8))
    if not np.all(keep):
        packed = np.ascontiguousarray(packed[keep], dtype=np.uint8)
        miss_arr = np.ascontiguousarray(miss_arr[keep], dtype=np.float32)
        maf_arr = np.ascontiguousarray(maf_arr[keep], dtype=np.float32)
    packed_ctx = {
        "packed": packed,
        "missing_rate": miss_arr,
        "maf": maf_arr,
        "n_samples": int(packed_n),
        "site_keep": site_keep,
        "source_prefix": str(genotype_prefix),
    }
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


def _apply_packed_ld_prune_for_gs(
    *,
    packed_ctx: dict[str, typing.Any],
    genotype_prefix: str,
    spec: _GsLdPruneSpec,
    n_jobs: int,
) -> dict[str, typing.Any]:
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

def main(log: bool = True) -> None:
    t_start = time.time()
    use_spinner = stdout_is_tty()
    parser = CliArgumentParser(
        prog="jx gs",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx gs -vcf geno.vcf.gz -p pheno.tsv -GBLUP -cv 5",
            "jx gs -hmp geno.hmp.gz -p pheno.tsv -GBLUP -cv 5",
            "jx gs -file geno_prefix -p pheno.tsv -GBLUP -cv 5",
            "jx gs -vcf geno.vcf.gz -p pheno.tsv -adBLUP -cv 5",
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
        action="store_true",
        default=False,
        help="Use GBLUP model for training and prediction "
             "(default: %(default)s).",
    )
    model_group.add_argument(
        "-adBLUP", "--adBLUP",
        action="store_true",
        default=False,
        help="Use additive+dominance kernel BLUP via pyBLUP/JAX backend "
             "(default: %(default)s).",
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

    args, extras = parser.parse_known_args()
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
    debug_mode = bool(getattr(args, "debug", False))
    try:
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))
    if (args.limit_predtrain is not None) and (int(args.limit_predtrain) < 0):
        parser.error("--limit-predtrain/--limit-train must be >= 0.")
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

    packed_preprocess_requested = bool(
        (args.hash_dim is not None) or (getattr(args, "ldprune_spec", None) is not None)
    )
    gfile_is_plink = _is_plink_prefix(str(gfile))
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
                        # into PLINK BED when hash/LD-prune preprocessing is requested.
                        prefer_plink_for_txt=True,
                    )
                except Exception:
                    task.fail("Preparing PLINK cache for packed GS ...Failed")
                    raise
                if not _is_plink_prefix(str(cached_prefix)):
                    task.fail("Preparing PLINK cache for packed GS ...Failed")
                    logger.error(
                        "--hash/--ldprune require PLINK BED input. "
                        "Auto-cache to PLINK is supported for -vcf/-hmp and for -file "
                        "when source resolves to VCF/HMP/TXT. "
                        f"Current input could not be converted to BED: {cached_prefix}. "
                        "For -file inputs backed by .npy/.bin, please convert to -bfile first."
                    )
                    raise SystemExit(1)
                gfile = str(cached_prefix).replace("\\", "/")
                gfile_is_plink = True
                task.complete(
                    "Preparing PLINK cache for packed GS ...Finished "
                    f"({format_path_for_display(gfile)})"
                )
        else:
            logger.error(
                "--hash/--ldprune require PLINK BED input. "
                "Auto-cache to PLINK is supported for -vcf/-hmp/-file when convertible."
            )
            raise SystemExit(1)

    # Configuration summary
    if log:
        ncol_cfg = (
            "All"
            if args.ncol is None
            else ",".join(str(int(i)) for i in args.ncol)
        )
        cfg_rows: list[tuple[str, object]] = [
            ("Genotype file", gfile),
            ("Phenotype file", args.pheno),
            ("Analysis Pcol", ncol_cfg),
        ]
        model_count = 0
        if args.GBLUP:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "GBLUP"))
        if args.adBLUP:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "adBLUP"))
        if args.rrBLUP:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "rrBLUP"))
        if args.BayesA:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "BayesA"))
        if args.BayesB:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "BayesB"))
        if args.BayesCpi:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "BayesCpi"))
        if args.RF:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "RF"))
        if args.ET:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "ET"))
        if args.GBDT:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "GBDT"))
        if args.XGB:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "XGB"))
        if args.SVM:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "SVM"))
        if args.ENET:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "ENET"))
        cfg_rows.extend(
            [
                ("Use PCA", args.pcd),
                ("MAF threshold", args.maf),
                ("Missing rate", args.geno),
                (
                    "LD prune",
                    (
                        args.ldprune_spec.label()
                        if getattr(args, "ldprune_spec", None) is not None
                        else "off"
                    ),
                ),
                ("Hash dim", ("off" if args.hash_dim is None else int(args.hash_dim))),
                (
                    "Hash standardize",
                    ("n/a" if args.hash_dim is None else (not bool(args.hash_raw))),
                ),
                ("Hash seed", ("" if args.hash_dim is None else int(args.hash_seed))),
                ("Threads", f"{args.thread} ({detected_threads} available)"),
                ("Limit train-pred", ("All" if args.limit_predtrain is None else int(args.limit_predtrain))),
                ("Strict CV", args.strict_cv),
                ("Force fast Gram", args.force_fast),
            ]
        )
        emit_cli_configuration(
            logger,
            app_title="JanusX - GS",
            config_title="GS CONFIG",
            host=socket.gethostname(),
            sections=[("General", cfg_rows)],
            footer_rows=[("Output prefix", outprefix)],
            line_max_chars=60,
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

    # ------------------------------------------------------------------
    # Collect methods to run
    # ------------------------------------------------------------------
    methods: list[str] = []
    if args.GBLUP:
        methods.append("GBLUP")
    if args.adBLUP:
        methods.append("adBLUP")
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
            "--GBLUP/--adBLUP/--rrBLUP/--BayesA/--BayesB/--BayesCpi/--RF/--ET/--GBDT/--XGB/--SVM/--ENET."
        )
        raise SystemExit(1)
    ml_methods = {"RF", "ET", "GBDT", "XGB", "SVM", "ENET"}

    if ("adBLUP" in methods) and (not _HAS_ADBLUP_PY):
        methods = [m for m in methods if m != "adBLUP"]
        logger.warning(
            "Skip model adBLUP because backend dependency is unavailable. "
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

    # Runtime-thread default tuning for packed GBLUP/rrBLUP:
    # When users do not explicitly set a policy, prefer aligned caps so
    # `-t` applies to both BLAS and Rust runtimes consistently.
    policy_env_raw = str(os.getenv("JX_THREAD_POLICY", "")).strip()
    gblup_only_models = bool(len(methods) > 0 and all(m in {"GBLUP", "rrBLUP"} for m in methods))
    if (args.hash_dim is not None) and ("adBLUP" in methods):
        logger.error("--hash is currently not supported together with adBLUP.")
        raise SystemExit(1)
    hash_gblup_only_mode = bool(
        (args.hash_dim is not None)
        and (len(methods) == 1)
        and (methods[0] == "GBLUP")
    )
    if (policy_env_raw == "") and bool(gfile_is_plink) and gblup_only_models:
        os.environ["JX_THREAD_POLICY"] = "align"
        if bool(debug_mode):
            logger.info(
                "Thread runtime hint: packed GBLUP/rrBLUP detected; "
                "using align policy by default (set JX_THREAD_POLICY to override)."
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
        maybe_warn_non_openblas(
            logger=logger,
            strict=require_openblas_by_default(),
        )

    if bool(debug_mode):
        logger.info("Effective models: " + ", ".join(methods))
    if ("adBLUP" in methods) and args.pcd:
        logger.info("Note: --pcd is ignored for adBLUP.")

    # ------------------------------------------------------------------
    # Load genotype
    # ------------------------------------------------------------------
    gsrc = os.path.basename(str(gfile).rstrip("/\\")) or str(gfile)
    use_packed_lmm = bool(
        gfile_is_plink
        and len(methods) > 0
        and all(m in {"GBLUP", "rrBLUP"} for m in methods)
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
                need_raw_geno = bool(("adBLUP" in methods) or any(m in ml_methods for m in methods))
                geno_raw = (
                    np.asarray(geno, dtype=np.float32, copy=need_std_geno)
                    if need_raw_geno
                    else None
                )
                geno_add_raw = geno_raw if ("adBLUP" in methods) else None
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
        need_raw_geno = bool(("adBLUP" in methods) or any(m in ml_methods for m in methods))
        geno_raw = (
            np.asarray(geno, dtype=np.float32, copy=need_std_geno)
            if need_raw_geno
            else None
        )
        geno_add_raw = geno_raw if ("adBLUP" in methods) else None
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
        need_raw_geno = bool(("adBLUP" in methods) or any(m in ml_methods for m in methods))
        geno_raw = (
            np.asarray(geno, dtype=np.float32, copy=need_std_geno)
            if need_raw_geno
            else None
        )
        geno_add_raw = geno_raw if ("adBLUP" in methods) else None
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
        need_raw_geno = bool(("adBLUP" in methods) or any(m in ml_methods for m in methods))
        geno_raw = (
            np.asarray(geno, dtype=np.float32, copy=need_std_geno)
            if need_raw_geno
            else None
        )
        geno_add_raw = geno_raw if ("adBLUP" in methods) else None
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
            if "adBLUP" in methods:
                if geno_add_raw is None:
                    raise RuntimeError("Internal error: additive genotype matrix for adBLUP is missing.")
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
        if gblup_backend_label is None and ("GBLUP" in methods) and (train_snp is not None):
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
        )
        if _GS_DEBUG_STAGE:
            logger.info(
                f"[GS-DEBUG][{trait_name}] stage=run_methods_parallel "
                f"elapsed={time.time() - t_stage:.3f}s"
            )
        method_result_map = {str(x["method"]): x for x in method_results}

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
                        method=str(m_name),
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
                        ax.set_ylabel(f"Predicted Value ({method})")
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
        outpred_list = [method_result_map[m]["test_pred"] for m in methods]
        logger.info(f"-"*60) if args.cv is not None else None
        # Stack predictions from all models: shape (n_test, n_methods)
        outpred = pd.DataFrame(
            np.concatenate(outpred_list, axis=1),
            columns=methods,
            index=samples[testmask],
        )
        out_tsv = f"{outprefix}.{trait_name}.gs.tsv"
        outpred.to_csv(out_tsv, sep="\t", float_format="%.4f")
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


if __name__ == "__main__":
    from janusx.script._common.interrupt import install_interrupt_handlers
    install_interrupt_handlers()
    main()
