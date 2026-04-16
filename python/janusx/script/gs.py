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
from scipy.stats import pearsonr, spearmanr
try:
    from sklearn.model_selection import KFold
except Exception:
    KFold = None  # type: ignore[assignment]
from janusx.bioplotkit.sci_set import color_set
from janusx.bioplotkit import gsplot
from janusx._optional_deps import format_missing_dependency_message
from janusx.gfreader import inspect_genotype_file, load_genotype_chunks, load_bed_2bit_packed
from janusx.pyBLUP.kfold import kfold
from janusx.pyBLUP.mlm import BLUP as MLMBLUP
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
from ._common.progress import build_rich_progress, rich_progress_available
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
            if (packed_ctx is not None) and (method in ("GBLUP", "rrBLUP")):
                if train_sample_indices is None:
                    raise ValueError("Packed LMM requires train sample indices.")
                fold_train = packed_ctx
                fold_test = packed_ctx
                fold_train_idx_arg = np.ascontiguousarray(train_sample_indices[train_idx], dtype=np.int64)
                fold_test_idx_arg = np.ascontiguousarray(train_sample_indices[test_idx], dtype=np.int64)
                fold_pca = False
            elif method == "adBLUP":
                if train_snp_add is None:
                    raise ValueError(f"{method} requires additive raw genotype matrix.")
                fold_train = train_snp_add[:, train_idx]
                fold_test = train_snp_add[:, test_idx]
                fold_pca = False
            elif method in _ML_METHOD_MAP:
                if train_snp_ml is None:
                    raise ValueError(f"{method} requires raw genotype matrix.")
                fold_train = train_snp_ml[:, train_idx]
                fold_test = train_snp_ml[:, test_idx]
                fold_pca = pca_dec
            else:
                fold_train = train_snp[:, train_idx]
                fold_test = train_snp[:, test_idx]
                fold_pca = pca_dec
            yhat_train, yhat_test, pve = GSapi(
                train_pheno[train_idx],
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
            ttest = np.concatenate([train_pheno[test_idx], yhat_test], axis=1)
            ttrain = None
            if need_fold_train_pred:
                y_train_fold = train_pheno[train_idx]
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

    final_train_idx_arg = None
    final_test_idx_arg = None
    if (packed_ctx is not None) and (method in ("GBLUP", "rrBLUP")):
        if train_sample_indices is None or test_sample_indices is None:
            raise ValueError("Packed LMM requires train/test sample indices.")
        final_train = packed_ctx
        final_test = packed_ctx
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
) -> list[dict[str, typing.Any]]:
    if len(methods) == 0:
        return []

    model_n_jobs = max(1, int(n_jobs))
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
            cv_total = int(max(1, fold_total_map.get(m, 1)))
            has_search_stage = bool((m in _ML_METHOD_MAP) and (not strict_cv))
            enable_tqdm_progress = bool(show_method_progress and _HAS_TQDM and stdout_is_tty())
            enable_method_spinner = bool((not show_method_progress) and stdout_is_tty())
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
                    elapsed = format_elapsed(time.monotonic() - t0)
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
                elapsed = format_elapsed(time.monotonic() - t0)
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
                        time.monotonic() - method_start_ts.get(method, time.monotonic())
                    )
                    print_failure(
                        f"{method} ...Failed [{elapsed}]",
                        force_color=True,
                    )
                    raise
                elapsed = format_elapsed(
                    time.monotonic() - method_start_ts.get(method, time.monotonic())
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
                            elapsed = format_elapsed(time.monotonic() - method_start_ts.get(method, time.monotonic()))
                            _rich_print_success(method, elapsed)
            except Exception:
                for method, tid in task_map.items():
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    elapsed = format_elapsed(time.monotonic() - method_start_ts.get(method, time.monotonic()))
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
                        elapsed = format_elapsed(time.monotonic() - method_start_ts.get(method, time.monotonic()))
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
    }
    return sample_ids_arr, packed_ctx


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
        "-pcd", "--pcd",
        action="store_true",
        default=False,
        help="Enable PCA-based dimensionality reduction on genotypes "
             "(default: %(default)s).",
    )
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
        help=(
            "Limit train-set predictions per outer CV fold for diagnostics. "
            "Use 0 to disable fold train predictions; default predicts all training samples."
        ),
    )
    optional_group.add_argument(
        "-strict-cv", "--strict-cv",
        action="store_true",
        default=False,
        help="For ML methods, retune hyperparameters inside every outer CV fold. "
             "By default, ML methods tune once per trait and reuse best params across outer folds.",
    )
    optional_group.add_argument(
        "-force-fast", "--force-fast",
        action="store_true",
        default=False,
        help="Force pyBLUP LMM Gram strategy to fast mode and never use lowmem auto-switch.",
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
    try:
        args.ncol = parse_zero_based_index_specs(args.ncol, label="-n/--n")
    except ValueError as e:
        parser.error(str(e))
    if (args.limit_predtrain is not None) and (int(args.limit_predtrain) < 0):
        parser.error("--limit-predtrain/--limit-train must be >= 0.")

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
    if args.bfile:
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

    logger.info("Effective models: " + ", ".join(methods))
    if ("adBLUP" in methods) and args.pcd:
        logger.info("Note: --pcd is ignored for adBLUP.")

    # ------------------------------------------------------------------
    # Load genotype
    # ------------------------------------------------------------------
    gsrc = os.path.basename(str(gfile).rstrip("/\\")) or str(gfile)
    use_packed_lmm = bool(
        args.bfile
        and len(methods) > 0
        and all(m in {"GBLUP", "rrBLUP"} for m in methods)
    )
    packed_lmm_ctx: dict[str, typing.Any] | None = None
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
        samples = sample_ids
        geno = None
        geno_raw = None
        geno_add_raw = None
        geno_ml_raw = None
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

    # ------------------------------------------------------------------
    # Genomic Selection for each phenotype
    # ------------------------------------------------------------------
    single_trait_mode = (pheno.shape[1] == 1)
    for trait_name in pheno.columns:
        logger.info("*" * 60)
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
        if use_packed_lmm:
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
        eff_snp = (
            int(packed_lmm_ctx["packed"].shape[0])
            if (use_packed_lmm and packed_lmm_ctx is not None)
            else int(train_snp.shape[0] if train_snp is not None else 0)
        )
        logger.info(
            f"* Genomic Selection for trait: {trait_name}\n"
            f"Train size: {np.sum(trainmask)}, Test size: {np.sum(testmask)}, EffSNPs: {eff_snp}"
        )

        if train_pheno.size == 0:
            logger.warning(f"No non-missing phenotypes for trait {trait_name}; skipped.")
            continue

        # 5-fold cross-validation on training population
        cv_splits = None
        if args.cv is not None:
            cv_splits = build_cv_splits(
                n_samples=int(train_sample_idx.shape[0]),
                n_splits=int(args.cv),
                seed=42,
            )

        if not use_packed_lmm:
            t_stage = time.time()
            assert geno is not None
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
        if single_trait_mode and (not use_packed_lmm):
            # In single-trait runs, release full matrices once train/test views are materialized.
            # This avoids carrying an extra full-genotype buffer through model fitting.
            geno = np.empty((0, 0), dtype=np.float32)
            geno_raw = None
            geno_add_raw = None
            geno_ml_raw = None
            gc.collect()
            if _GS_DEBUG_STAGE:
                logger.info(f"[GS-DEBUG][{trait_name}] stage=release_full_geno done=1")
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
            packed_ctx=packed_lmm_ctx,
            train_sample_indices=train_sample_idx,
            test_sample_indices=test_sample_idx,
            limit_predtrain=args.limit_predtrain,
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
