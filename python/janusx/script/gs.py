# -*- coding: utf-8 -*-
"""
JanusX: Genomic Selection Command-Line Interface

Supported models
----------------
  - GBLUP  : Genomic Best Linear Unbiased Prediction (GBLUP, kinship = 1)
  - rrBLUP : Ridge regression BLUP (rrBLUP, kinship = None)
  - BayesA : Bayesian marker effect model (via pyBLUP.bayes)
  - BayesB : Bayesian variable selection model (via pyBLUP.bayes)
  - BayesCpi : Bayesian variable selection model with shared variance (via pyBLUP.bayes)

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
  2. Filter SNPs by MAF/missing rate thresholds (default 0.02/0.05) and impute
     by mode (via QK).
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
from joblib import cpu_count as joblib_cpu_count
from janusx.bioplotkit.sci_set import color_set
from janusx.bioplotkit import gsplot
from janusx.gfreader import breader, vcfreader
from janusx.pyBLUP import BLUP, kfold
from janusx.pyBLUP.bayes import BAYES
from ._common.log import setup_logging
from ._common.config_render import emit_cli_configuration
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
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
        TimeRemainingColumn,
    )
    _HAS_RICH_PROGRESS = True
except Exception:
    Progress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    TimeRemainingColumn = None  # type: ignore[assignment]
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

def GSapi(
    Y: np.ndarray,
    Xtrain: np.ndarray,
    Xtest: np.ndarray,
    method: typing.Literal["GBLUP", "rrBLUP", "BayesA", "BayesB", "BayesCpi"],
    PCAdec: bool = False,
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
    method : {'GBLUP', 'rrBLUP', 'BayesA', 'BayesB', 'BayesCpi'}
        Prediction model.
    PCAdec : bool, optional
        If True, perform PCA-based dimensionality reduction before modeling.
        PCA is computed on the concatenated matrix [Xtrain, Xtest].

    Returns
    -------
    yhat_train : np.ndarray
        Predicted phenotypes for training individuals, shape (n_train, 1).
    yhat_test : np.ndarray
        Predicted phenotypes for test individuals, shape (n_test, 1).
    pve : float
        Proportion of variance explained (GBLUP/rrBLUP) or posterior mean h2 (Bayes).
    """
    # Optional PCA-based dimensionality reduction
    if PCAdec:
        Xtt = np.concatenate([Xtrain, Xtest], axis=1)  # (m, n_train + n_test)
        Xtt = (Xtt - np.mean(Xtt, axis=1, keepdims=True)) / (
            np.std(Xtt, axis=1, keepdims=True) + 1e-8
        )
        # Simple PCA via eigendecomposition of X^T X
        val, vec = np.linalg.eigh(Xtt.T @ Xtt / Xtt.shape[0])
        idx = np.argsort(val)[::-1]
        val, vec = val[idx], vec[:, idx]
        # Retain components explaining up to 90% variance
        dim = np.sum(np.cumsum(val) / np.sum(val) <= 0.9)
        vec = val[:dim] * vec[:, :dim]
        Xtrain, Xtest = vec[: Xtrain.shape[1], :].T, vec[Xtrain.shape[1] :, :].T

    # Linear mixed models
    if method in ("GBLUP", "rrBLUP"):
        kinship = 1 if method == "GBLUP" else None
        model = BLUP(Y.reshape(-1, 1), Xtrain, kinship=kinship)
        return model.predict(Xtrain), model.predict(Xtest), model.pve

    if method in ("BayesA", "BayesB", "BayesCpi"):
        model = BAYES(Y.reshape(-1, 1), Xtrain, method=method)
        pve = model.pve
        return model.predict(Xtrain), model.predict(Xtest), pve

    raise ValueError(f"Unsupported GS method: {method}")


def _run_method_task(
    method: str,
    train_pheno: np.ndarray,
    train_snp: np.ndarray,
    test_snp: np.ndarray,
    pca_dec: bool,
    cv_splits: typing.Optional[list[tuple[np.ndarray, np.ndarray]]],
    progress_queue: typing.Any = None,
) -> dict[str, typing.Any]:
    fold_rows: list[tuple[str, int, float, float, float, float, float]] = []
    best_test = None
    best_train = None
    best_r2 = -np.inf

    if cv_splits is not None:
        for fold_id, (test_idx, train_idx) in enumerate(cv_splits, start=1):
            t_fold = time.time()
            yhat_train, yhat_test, pve = GSapi(
                train_pheno[train_idx],
                train_snp[:, train_idx],
                train_snp[:, test_idx],
                method=method,
                PCAdec=pca_dec,
            )
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

            if r2 > best_r2:
                best_r2 = r2
                best_test = ttest
                best_train = ttrain

    _, test_pred, pve_final = GSapi(
        train_pheno,
        train_snp,
        test_snp,
        method=method,
        PCAdec=pca_dec,
    )
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
    pca_dec: bool,
    cv_splits: typing.Optional[list[tuple[np.ndarray, np.ndarray]]],
) -> list[dict[str, typing.Any]]:
    if len(methods) == 0:
        return []

    show_method_progress = cv_splits is not None
    n_jobs = int(min(len(methods), max(1, int(joblib_cpu_count()))))
    if n_jobs <= 1:
        out: list[dict[str, typing.Any]] = []
        for m in methods:
            t0 = time.monotonic()
            try:
                res = _run_method_task(m, train_pheno, train_snp, test_snp, pca_dec, cv_splits)
            except Exception:
                elapsed = format_elapsed(time.monotonic() - t0)
                print_failure(f"Method: {m} ...Failed [{elapsed}]", force_color=True)
                raise
            out.append(res)
            elapsed = format_elapsed(time.monotonic() - t0)
            print_success(f"Method: {m} ...Finished [{elapsed}]", force_color=True)
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
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_map = {
                ex.submit(
                    _run_method_task,
                    m,
                    train_pheno,
                    train_snp,
                    test_snp,
                    pca_dec,
                    cv_splits,
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
                        f"Method: {method} ...Failed [{elapsed}]",
                        force_color=True,
                    )
                    raise
                elapsed = format_elapsed(
                    time.monotonic() - method_start_ts.get(method, time.monotonic())
                )
                print_success(
                    f"Method: {method} ...Finished [{elapsed}]",
                    force_color=True,
                )
    elif _HAS_RICH_PROGRESS and sys.stdout.isatty():
        progress = Progress(
            SpinnerColumn(
                spinner_name=get_rich_spinner_name(),
                style="cyan",
                finished_text=" ",
            ),
            TextColumn("Method: {task.fields[method]}"),
            BarColumn(),
            TextColumn("{task.percentage:>6.1f}%{task.fields[suffix]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        with progress:
            def _rich_print_success(method_name: str, elapsed_text: str) -> None:
                progress.console.print(
                    f"[green]✔︎ Method: {method_name} ...Finished [{elapsed_text}][/green]"
                )

            def _rich_print_failure(method_name: str, elapsed_text: str) -> None:
                progress.console.print(
                    f"[red]✘ Method: {method_name} ...Failed [{elapsed_text}][/red]"
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
                with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
                    future_map = {
                        ex.submit(
                            _run_method_task,
                            m,
                            train_pheno,
                            train_snp,
                            test_snp,
                            pca_dec,
                            cv_splits,
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
                       "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        method_start_ts = {m: time.monotonic() for m in methods}
        manager = mp.Manager()
        prog_q = manager.Queue() if cv_splits is not None else None
        try:
            with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
                future_map = {
                    ex.submit(
                        _run_method_task,
                        m,
                        train_pheno,
                        train_snp,
                        test_snp,
                        pca_dec,
                        cv_splits,
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
                            f"Method: {method} ...Finished [{elapsed}]",
                            force_color=True,
                        )
        finally:
            pbar.close()
            try:
                manager.shutdown()
            except Exception:
                pass
    else:
        with cf.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_map = {
                ex.submit(
                    _run_method_task,
                    m,
                    train_pheno,
                    train_snp,
                    test_snp,
                    pca_dec,
                    cv_splits,
                    None,
                ): m
                for m in methods
            }
            for fut in cf.as_completed(list(future_map.keys())):
                method = future_map[fut]
                results_by_method[method] = fut.result()

    return [results_by_method[m] for m in methods if m in results_by_method]


# ======================================================================
# CLI
# ======================================================================

def main(log: bool = True) -> None:
    t_start = time.time()
    use_spinner = bool(getattr(sys.stdout, "isatty", lambda: False)())
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ------------------------------------------------------------------
    # Required arguments
    # ------------------------------------------------------------------
    required_group = parser.add_argument_group("Required Arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
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
    required_group.add_argument(
        "-p", "--pheno",
        type=str,
        required=True,
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
        type=int,
        default=None,
        help="Zero-based phenotype column index to analyze. "
             "If not set, all phenotype columns will be processed "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-cv", "--cv",
        type=int,
        default=None,
        help="K fold of cross-validazation for models. "
             "(default: %(default)s).",
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

    args = parser.parse_args()

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
    elif args.bfile:
        gfile = args.bfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    else:
        raise ValueError("No genotype input detected. Use -vcf or -bfile.")

    gfile = gfile.replace("\\", "/")  # Normalize Windows-style paths
    args.out = args.out if args.out is not None else "."

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    log_path = f"{args.out}/{args.prefix}.gs.log".replace("\\", "/").replace("//", "/")
    logger = setup_logging(log_path)

    # Configuration summary
    if log:
        cfg_rows: list[tuple[str, object]] = [
            ("Genotype file", gfile),
            ("Phenotype file", args.pheno),
            ("Analysis Pcol", args.ncol if args.ncol is not None else "All"),
        ]
        model_count = 0
        if args.GBLUP:
            model_count += 1
            cfg_rows.append((f"Used model{model_count}", "GBLUP"))
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
        cfg_rows.extend(
            [
                ("Use PCA", args.pcd),
                ("MAF threshold", args.maf),
                ("Missing rate", args.geno),
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
            footer_rows=[("Output prefix", f"{args.out}/{args.prefix}")],
            line_max_chars=60,
        )

    checks: list[bool] = []
    if args.bfile:
        checks.append(ensure_plink_prefix_exists(logger, gfile, "Genotype PLINK prefix"))
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
        if not (0 <= args.ncol < pheno.shape[1]):
            logger.error("IndexError: phenotype column index out of range.")
            raise SystemExit(1)
        pheno = pheno.iloc[:, [args.ncol]]

    # ------------------------------------------------------------------
    # Collect methods to run
    # ------------------------------------------------------------------
    methods: list[str] = []
    if args.GBLUP:
        methods.append("GBLUP")
    if args.rrBLUP:
        methods.append("rrBLUP")
    if args.BayesA:
        methods.append("BayesA")
    if args.BayesB:
        methods.append("BayesB")
    if args.BayesCpi:
        methods.append("BayesCpi")
    if len(methods) == 0:
        logger.error(
            "No model selected. Use --GBLUP/--rrBLUP/--BayesA/--BayesB/--BayesCpi."
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Load genotype
    # ------------------------------------------------------------------
    if args.vcf:
        geno_df = vcfreader(
            gfile,
            maf=args.maf,
            miss=args.geno,
            impute=True,
            dtype='float32',
            progress_desc="Loading genotype",
            show_progress=True,
        )
    elif args.bfile:
        geno_df = breader(
            gfile,
            maf=args.maf,
            miss=args.geno,
            impute=True,
            dtype='float32',
            progress_desc="Loading genotype",
            show_progress=True,
        )
    else:
        raise ValueError("Genotype input not recognized.")
    logger.info(
        f"* Filter SNPs with MAF < {args.maf} or missing rate > {args.geno}; "
        "impute with mean."
    )
    logger.info("  Tip: Use genotype matrices imputed by BEAGLE/IMPUTE2 whenever possible.")
    logger.info(f"Completed, cost: {round(time.time() - t_loading, 3)} secs")
    m, n = geno_df.shape
    n = n - 2  # First 2 columns usually CHR and POS
    logger.info(f"Loaded SNP: {m}, individuals: {n}")

    samples = geno_df.columns[2:].astype(str)
    geno = geno_df.iloc[:, 2:].values
    geno = (geno - geno.mean(axis=1,keepdims=True)) / (geno.std(axis=1,keepdims=True)+1e-6) # standardization of genotype

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
        method_results = _run_methods_parallel(
            methods=methods,
            train_pheno=train_pheno,
            train_snp=train_snp,
            test_snp=test_snp,
            pca_dec=args.pcd,
            cv_splits=cv_splits,
        )
        method_result_map = {str(x["method"]): x for x in method_results}

        if args.cv is not None:
            logger.info("-" * 60)
            logger.info("Method Fold Pearsonr Spearmanr R² h² time(secs)")
            for method in methods:
                res = method_result_map.get(method)
                if res is None:
                    continue
                for row in res.get("fold_rows", []):
                    m_name, fold_id, pear, spear, r2, pve, t_fold = row
                    logger.info(
                        f"{m_name} {fold_id} {pear:.3f} {spear:.3f} {r2:.3f} {pve:.3f} {t_fold:.3f}"
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
                out_svg = f"{args.out}/{args.prefix}.{trait_name}.gs.{method}.svg"
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
        out_tsv = f"{args.out}/{args.prefix}.{trait_name}.gs.tsv"
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
