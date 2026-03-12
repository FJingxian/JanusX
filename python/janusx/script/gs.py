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
  - Tab-delimited text file
  - First column: sample IDs
  - Remaining columns: phenotype traits
  - Duplicated IDs will be averaged.

Cross-validation
----------------
  - 5-fold cross-validation is performed within the training population for each model.
  - For each method, the fold with the highest R^2 on the validation set is reported
    and (optionally) visualized.

Genomic selection workflow
--------------------------
  1. Load genotypes and phenotypes.
  2. Filter SNPs by MAF/missing rate thresholds (default 0.02/0.05) and
     mean-impute missing genotypes during Rust `gfreader` loading.
  3. For each phenotype column:
       - Split individuals into training (non-missing phenotype) and test sets.
       - Run 5-fold CV on the training set for each selected model.
       - Report Pearson, Spearman, and R² per fold.
       - Use the best fold for diagnostic plotting (if enabled).
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
from janusx.bioplotkit.sci_set import color_set
from janusx.bioplotkit import gsplot
from janusx.gfreader import inspect_genotype_file, load_genotype_chunks
from janusx.pyBLUP.kfold import kfold
from janusx.pyBLUP.mlm import BLUP as MLMBLUP
from janusx.pyBLUP.bayes import BAYES
from janusx.pyBLUP.ml import MLGS, _HAS_XGBOOST, _XGBOOST_IMPORT_ERROR
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
    ensure_plink_prefix_exists,
)
from ._common.status import (
    CliStatus,
    get_rich_spinner_name,
    print_success,
    print_failure,
    format_elapsed,
)

try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
    )
    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    _HAS_RICH_PROGRESS = False

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


def _tune_ml_method_once(
    method: str,
    Y: np.ndarray,
    Xtrain: np.ndarray,
    PCAdec: bool,
    n_jobs: int,
) -> dict[str, typing.Any]:
    empty_test = np.zeros((Xtrain.shape[0], 0), dtype=Xtrain.dtype)
    Xtrain_tuned, _ = _apply_optional_pca(Xtrain, empty_test, enabled=PCAdec)
    model = MLGS(
        y=Y.reshape(-1),
        M=Xtrain_tuned,
        method=typing.cast(typing.Any, _ML_METHOD_MAP[method]),
        seed=42,
        cv=_mlgs_inner_cv(int(Y.shape[0])),
        n_jobs=max(1, int(n_jobs)),
        fit_on_init=True,
        verbose=False,
    )
    return {
        "params": model.get_final_params(),
        "pve": float((model.best_metrics_ or {}).get("r2", np.nan)),
        "best_metrics": dict(model.best_metrics_ or {}),
    }


def GSapi(
    Y: np.ndarray,
    Xtrain: np.ndarray,
    Xtest: np.ndarray,
    method: typing.Literal["GBLUP", "adBLUP", "rrBLUP", "BayesA", "BayesB", "BayesCpi", "RF", "ET", "GBDT", "XGB", "SVM", "ENET"],
    PCAdec: bool = False,
    n_jobs: int = 1,
    ml_fixed_params: dict[str, typing.Any] | None = None,
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

    Returns
    -------
    yhat_train : np.ndarray
        Predicted phenotypes for training individuals, shape (n_train, 1).
    yhat_test : np.ndarray
        Predicted phenotypes for test individuals, shape (n_test, 1).
    pve : float
        Variance-component PVE/h2 for mixed models and Bayes models,
        or predictive PVE (inner-CV R²) for ML models.
    """
    Xtrain, Xtest = _apply_optional_pca(Xtrain, Xtest, enabled=PCAdec)

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

        yhat_train = model.predict(G=[Gadd_train, Ghet_train])
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
        model = MLMBLUP(Y.reshape(-1, 1), Xtrain, kinship=kinship)
        return model.predict(Xtrain), model.predict(Xtest), model.pve

    if method in ("BayesA", "BayesB", "BayesCpi"):
        model = BAYES(Y.reshape(-1, 1), Xtrain, method=method)
        pve = model.pve
        return model.predict(Xtrain), model.predict(Xtest), pve

    if method in _ML_METHOD_MAP:
        model = MLGS(
            y=Y.reshape(-1),
            M=Xtrain,
            method=typing.cast(typing.Any, _ML_METHOD_MAP[method]),
            seed=42,
            cv=_mlgs_inner_cv(int(Y.shape[0])),
            n_jobs=max(1, int(n_jobs)),
            fit_on_init=False if ml_fixed_params is not None else True,
            verbose=False,
        )
        if ml_fixed_params is not None:
            model.fit_with_params(ml_fixed_params)
        yhat_train = model.fit_predict()
        if int(Xtest.shape[1]) > 0:
            yhat_test = model.predict(Xtest)
        else:
            yhat_test = np.zeros((0, 1), dtype=float)
        pve = float(
            (model.best_metrics_ or {}).get("r2", np.nan)
        )
        return yhat_train, yhat_test, pve

    raise ValueError(f"Unsupported GS method: {method}")


def _run_method_task(
    method: str,
    train_pheno: np.ndarray,
    train_snp: np.ndarray,
    test_snp: np.ndarray,
    train_snp_add: np.ndarray | None,
    test_snp_add: np.ndarray | None,
    train_snp_ml: np.ndarray | None,
    test_snp_ml: np.ndarray | None,
    pca_dec: bool,
    cv_splits: typing.Optional[list[tuple[np.ndarray, np.ndarray]]],
    n_jobs: int,
    strict_cv: bool,
    progress_queue: typing.Any = None,
    progress_hook: typing.Any = None,
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
        )

    if cv_splits is not None:
        for fold_id, (test_idx, train_idx) in enumerate(cv_splits, start=1):
            t_fold = time.time()
            if method == "adBLUP":
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
                ml_fixed_params=(None if ml_tuning_cache is None else ml_tuning_cache.get("params")),
            )
            if ml_tuning_cache is not None:
                pve = float(ml_tuning_cache.get("pve", np.nan))
            ttest = np.concatenate([train_pheno[test_idx], yhat_test], axis=1)
            ttrain = np.concatenate([train_pheno[train_idx], yhat_train], axis=1)

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

    if method == "adBLUP":
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
        ml_fixed_params=(None if ml_tuning_cache is None else ml_tuning_cache.get("params")),
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
    train_snp: np.ndarray,
    test_snp: np.ndarray,
    train_snp_add: np.ndarray | None,
    test_snp_add: np.ndarray | None,
    train_snp_ml: np.ndarray | None,
    test_snp_ml: np.ndarray | None,
    pca_dec: bool,
    cv_splits: typing.Optional[list[tuple[np.ndarray, np.ndarray]]],
    n_jobs: int,
    strict_cv: bool,
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
        task_total_map = {
            m: (fold_total_map[m] + 1) if show_method_progress else fold_total_map[m]
            for m in methods
        }
        if show_method_progress and _HAS_RICH_PROGRESS and sys.stdout.isatty():
            progress = Progress(
                SpinnerColumn(
                    spinner_name=get_rich_spinner_name(),
                    style="cyan",
                    finished_text=" ",
                ),
                TextColumn("{task.fields[method]}{task.fields[suffix]}"),
                BarColumn(),
                TextColumn("{task.percentage:>6.1f}%"),
                TimeElapsedColumn(),
                transient=True,
            )
            with progress:
                task_map: dict[str, int] = {}
                done_map: dict[str, int] = {}
                for idx, m in enumerate(methods):
                    total_n = int(max(1, task_total_map.get(m, 1)))
                    task_map[m] = progress.add_task(
                        description="",
                        total=total_n,
                        method=m,
                        suffix="" if idx == 0 else " waiting",
                        start=False,
                    )
                    done_map[m] = 0

                for idx, m in enumerate(methods):
                    total_n = int(max(1, task_total_map.get(m, 1)))
                    tid = int(task_map[m])
                    progress.update(tid, suffix="")
                    try:
                        progress.start_task(tid)
                    except Exception:
                        pass
                    t0 = time.monotonic()
                    try:
                        def _hook(method_name: str, inc: int) -> None:
                            if str(method_name) != str(m):
                                return
                            left = max(0, total_n - int(done_map[m]))
                            adv = min(left, int(max(0, int(inc))))
                            if adv > 0:
                                progress.update(tid, advance=adv)
                                done_map[m] = int(done_map[m]) + adv

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
                            progress_queue=None,
                            progress_hook=_hook,
                        )
                    except Exception:
                        try:
                            progress.remove_task(tid)
                        except Exception:
                            pass
                        elapsed = format_elapsed(time.monotonic() - t0)
                        progress.console.print(
                            f"[red]✘ {m} ...Failed [{elapsed}][/red]"
                        )
                        raise
                    left_final = max(0, total_n - int(done_map[m]))
                    if left_final > 0:
                        progress.update(tid, advance=left_final)
                    try:
                        progress.remove_task(tid)
                    except Exception:
                        pass
                    out.append(res)
                    elapsed = format_elapsed(time.monotonic() - t0)
                    progress.console.print(
                        f"[green]✔︎ {m} ...Finished [{elapsed}][/green]"
                    )
            return out
        for m in methods:
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
                )
            except Exception:
                elapsed = format_elapsed(time.monotonic() - t0)
                print_failure(f"{m} ...Failed [{elapsed}]", force_color=True)
                raise
            out.append(res)
            elapsed = format_elapsed(time.monotonic() - t0)
            print_success(f"{m} ...Finished [{elapsed}]", force_color=True)
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
                    None,
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
    elif _HAS_RICH_PROGRESS and sys.stdout.isatty():
        progress = Progress(
            SpinnerColumn(
                spinner_name=get_rich_spinner_name(),
                style="cyan",
                finished_text=" ",
            ),
            TextColumn("{task.fields[method]}"),
            BarColumn(),
            TextColumn("{task.percentage:>6.1f}%{task.fields[suffix]}"),
            TimeElapsedColumn(),
            transient=True,
        )
        with progress:
            def _rich_print_success(method_name: str, elapsed_text: str) -> None:
                progress.console.print(
                    f"[green]✔︎ {method_name} ...Finished [{elapsed_text}][/green]"
                )

            def _rich_print_failure(method_name: str, elapsed_text: str) -> None:
                progress.console.print(
                    f"[red]✘ {method_name} ...Failed [{elapsed_text}][/red]"
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
                            prog_q,
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
    elif _HAS_TQDM:
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
                        prog_q,
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
                    None,
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


# ======================================================================
# CLI
# ======================================================================

def main(log: bool = True) -> None:
    t_start = time.time()
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    parser = CliArgumentParser(
        prog="jx gs",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog([
            "jx gs -vcf geno.vcf.gz -p pheno.tsv -GBLUP -cv 5",
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
        help="Phenotype file (tab-delimited, sample IDs in the first column).",
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
        "-n", "--ncol",
        action="extend",
        nargs="+",
        type=int,
        default=None,
        help="Zero-based phenotype column indices to analyze. "
             "Supports repeated inputs, e.g. -n 0 -n 3 or -n 0 3. "
             "If not set, all phenotype columns will be processed "
             "(default: %(default)s).",
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
        default=max(1, int(os.cpu_count() or 1)),
        help="Threads for ML methods (RF/ET/GBDT/XGB/SVM/ENET). "
             "(default: all available cores).",
    )
    optional_group.add_argument(
        "-strict-cv", "--strict-cv",
        action="store_true",
        default=False,
        help="For ML methods, retune hyperparameters inside every outer CV fold. "
             "By default, ML methods tune once per trait and reuse best params across outer folds.",
    )
    optional_group.add_argument(
        "-plot", "--plot",
        action="store_true",
        default=False,
        help="Enable visualization of 5-fold CV and model performance "
             "(default: %(default)s).",
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
    has_genotype = bool(args.vcf or args.bfile or args.file)
    has_pheno = bool(args.pheno)
    if (not has_pheno) and (not has_genotype):
        parser.error(
            "the following arguments are required: -p/--pheno & "
            "(-vcf VCF | -file FILE | -bfile BFILE)"
        )
    if not has_pheno:
        parser.error("the following arguments are required: -p/--pheno")
    if not has_genotype:
        parser.error("the following arguments are required: (-vcf VCF | -file FILE | -bfile BFILE)")
    if len(extras) > 0:
        parser.error("unrecognized arguments: " + " ".join(extras))

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
        raise ValueError("No genotype input detected. Use -vcf, -file or -bfile.")

    gfile = gfile.replace("\\", "/")  # Normalize Windows-style paths
    args.out = os.path.normpath(args.out if args.out is not None else ".")
    outprefix = os.path.join(args.out, args.prefix).replace("\\", "/")

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    log_path = f"{outprefix}.gs.log".replace("//", "/")
    logger = setup_logging(log_path)

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
                ("Threads", args.thread),
                ("Strict CV", args.strict_cv),
            ]
        )
        if args.plot:
            cfg_rows.append(("Plot mode", args.plot))
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
    logger.info(f"Loading phenotype from {args.pheno}...")
    with CliStatus("Loading phenotype...", enabled=use_spinner) as task:
        try:
            pheno = pd.read_csv(args.pheno, sep="\t")
            # First column is sample ID; average duplicated IDs
            pheno = pheno.groupby(pheno.columns[0]).mean()
            pheno.index = pheno.index.astype(str)
        except Exception:
            task.fail("Loading phenotype ...Failed")
            raise
        task.complete("Loading phenotype ...Finished")

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
    if args.adBLUP and (not _HAS_ADBLUP_PY):
        logger.error(
            "adBLUP (JAX backend) is unavailable. "
            f"Original import error: {_ADBLUP_PY_IMPORT_ERROR}"
        )
        raise SystemExit(1)
    if args.XGB and (not _HAS_XGBOOST):
        logger.error(
            "XGB is unavailable because xgboost could not be imported. "
            f"Original import error: {_XGBOOST_IMPORT_ERROR}"
        )
        raise SystemExit(1)
    if args.adBLUP and args.pcd:
        logger.info("Note: --pcd is ignored for adBLUP.")

    # ------------------------------------------------------------------
    # Load genotype
    # ------------------------------------------------------------------
    gsrc = os.path.basename(str(gfile).rstrip("/")) or str(gfile)
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
    need_std_geno = bool(args.GBLUP or args.rrBLUP or args.BayesA or args.BayesB or args.BayesCpi)
    need_raw_geno = bool(args.adBLUP or args.RF or args.ET or args.GBDT or args.XGB or args.SVM or args.ENET)
    geno_raw = (
        np.asarray(geno, dtype=np.float32, copy=need_std_geno)
        if need_raw_geno
        else None
    )
    geno_add_raw = geno_raw if args.adBLUP else None
    geno_ml_raw = geno_raw if (args.RF or args.ET or args.GBDT or args.XGB or args.SVM or args.ENET) else None
    if need_std_geno:
        geno = (geno - geno.mean(axis=1, keepdims=True)) / (
            geno.std(axis=1, keepdims=True) + 1e-6
        )  # standardization for marker-effect models

    # ------------------------------------------------------------------
    # Genomic Selection for each phenotype
    # ------------------------------------------------------------------
    for trait_name in pheno.columns:
        logger.info("*" * 60)
        t_trait = time.time()

        p = pheno[trait_name]
        namark = p.isna()
        trainmask = np.isin(samples, p.index[~namark])
        testmask = ~trainmask

        train_snp = geno[:, trainmask]
        train_pheno = p.loc[samples[trainmask]].values.reshape(-1, 1)
        logger.info(f"* Genomic Selection for trait: {trait_name}\nTrain size: {np.sum(trainmask)}, Test size: {np.sum(testmask)}, EffSNPs: {train_snp.shape[0]}")

        if train_pheno.size == 0:
            logger.info(f"No non-missing phenotypes for trait {trait_name}; skipped.")
            continue

        # 5-fold cross-validation on training population
        cv_splits = None
        if args.cv is not None:
            cv_splits = list(kfold(train_snp.shape[1], k=int(args.cv), seed=None))

        test_snp = geno[:, testmask]
        train_snp_add = None
        test_snp_add = None
        train_snp_ml = None
        test_snp_ml = None
        if args.adBLUP:
            if geno_add_raw is None:
                raise RuntimeError("Internal error: additive genotype matrix for adBLUP is missing.")
            train_snp_add = geno_add_raw[:, trainmask]
            test_snp_add = geno_add_raw[:, testmask]
        if args.RF or args.ET or args.GBDT or args.XGB or args.SVM or args.ENET:
            if geno_ml_raw is None:
                raise RuntimeError("Internal error: raw genotype matrix for MLGS is missing.")
            train_snp_ml = geno_ml_raw[:, trainmask]
            test_snp_ml = geno_ml_raw[:, testmask]
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
                    r2="R²",
                    h2="h²/PVE",
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

        if args.plot and args.cv is not None:
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
                out_svg = f"{outprefix}.{trait_name}.gs.{method}.svg"
                fig.tight_layout()
                fig.savefig(out_svg, transparent=False, facecolor="white")
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
        logger.info(f"Saved predictions to {out_tsv}".replace("//", "/"))
        logger.info(f"Trait {trait_name} finished in {(time.time() - t_trait):.2f} secs")

    # ----------------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished, total time: {round(time.time() - t_start, 2)} secs\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
