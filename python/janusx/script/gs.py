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
from janusx.gfreader import breader, vcfreader
from janusx.pyBLUP import BLUP, kfold
from janusx.pyBLUP.bayes import BAYES
from ._common.log import setup_logging
from ._common.pathcheck import (
    ensure_all_true,
    ensure_file_exists,
    ensure_plink_prefix_exists,
)


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


# ======================================================================
# CLI
# ======================================================================

def main(log: bool = True) -> None:
    t_start = time.time()
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

    logger.info("Genomic Selection Module")
    logger.info(f"Host: {socket.gethostname()}\n")

    # Configuration summary
    if log:
        logger.info("*" * 60)
        logger.info("GENOMIC SELECTION CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file:   {gfile}")
        logger.info(f"Phenotype file:  {args.pheno}")
        if args.ncol is not None:
            logger.info(f"Analysis Pcol:   {args.ncol}")
        else:
            logger.info("Analysis Pcol:   All")

        model_count = 0
        if args.GBLUP:
            model_count += 1
            logger.info(f"Used model{model_count}:     GBLUP")
        if args.rrBLUP:
            model_count += 1
            logger.info(f"Used model{model_count}:     rrBLUP")
        if args.BayesA:
            model_count += 1
            logger.info(f"Used model{model_count}:     BayesA")
        if args.BayesB:
            model_count += 1
            logger.info(f"Used model{model_count}:     BayesB")
        if args.BayesCpi:
            model_count += 1
            logger.info(f"Used model{model_count}:     BayesCpi")
        logger.info(f"Use PCA:         {args.pcd}")
        logger.info(f"MAF threshold:   {args.maf}")
        logger.info(f"Missing rate:    {args.geno}")
        if args.plot:
            logger.info(f"Plot mode:       {args.plot}")
        logger.info(f"Output prefix:   {args.out}/{args.prefix}")
        logger.info("*" * 60 + "\n")

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
    pheno = pd.read_csv(args.pheno, sep="\t")
    # First column is sample ID; average duplicated IDs
    pheno = pheno.groupby(pheno.columns[0]).mean()
    pheno.index = pheno.index.astype(str)

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
        logger.info(f"Loading genotype from {gfile}...")
        geno_df = vcfreader(gfile,maf=args.maf,miss=args.geno,impute=True,dtype='float32')
    elif args.bfile:
        logger.info(f"Loading genotype from {gfile}.bed...")
        geno_df = breader(gfile,maf=args.maf,miss=args.geno,impute=True,dtype='float32')
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
        if args.cv is not None:
            kfoldset = kfold(train_snp.shape[1], k=int(args.cv), seed=None)
            spt = ' '
            logger.info(f"-"*60)
            logger.info(f"Method{spt}Fold{spt}Pearsonr{spt}Spearmanr{spt}R²{spt}h²{spt}time(secs)")
        outpred_list = []
        for method in methods:

            fold_test_pairs = []
            fold_train_pairs = []
            r2_test = []
            fold_id = 0

            if args.cv is not None:
                for test_idx, train_idx in kfoldset:
                    fold_id += 1
                    t_fold = time.time()

                    yhat_train, yhat_test, pve = GSapi(
                        train_pheno[train_idx],
                        train_snp[:, train_idx],
                        train_snp[:, test_idx],
                        method=method,
                        PCAdec=args.pcd,
                    )

                    ttest = np.concatenate([train_pheno[test_idx], yhat_test], axis=1)
                    ttrain = np.concatenate([train_pheno[train_idx], yhat_train], axis=1)

                    fold_test_pairs.append(ttest)
                    fold_train_pairs.append(ttrain)

                    ss_res = np.sum((ttest[:, 0] - ttest[:, 1]) ** 2)
                    ss_tot = np.sum((ttest[:, 0] - ttest[:, 0].mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                    r2_test.append(r2)

                    pear = pearsonr(ttest[:, 0], ttest[:, 1]).statistic
                    spear = spearmanr(ttest[:, 0], ttest[:, 1]).statistic

                    logger.info(
                        f"{method}{spt}"
                        f"{fold_id}{spt}"
                        f"{pear:.3f}{spt}"
                        f"{spear:.3f}{spt}"
                        f"{r2:.3f}{spt}"
                        f"{pve:.3f}{spt}"
                        f"{(time.time() - t_fold):.3f}{spt}"
                    )

                # Use the fold with highest R² for plotting
                best_idx = int(np.argmax(r2_test))
                best_test = fold_test_pairs[best_idx]
                best_train = fold_train_pairs[best_idx]

                if args.plot:
                    fig = plt.figure(figsize=(5, 4), dpi=300)
                    gsplot.scatterh(best_test, best_train, color_set=color_set[0], fig=fig)
                    out_svg = f"{args.out}/{args.prefix}.{trait_name}.gs.{method}.svg"
                    fig.tight_layout()
                    fig.savefig(out_svg, transparent=False, facecolor="white")
                    plt.close(fig)

            # ------------------------------------------------------------------
            # Final prediction on test population
            # ------------------------------------------------------------------
            test_snp = geno[:, testmask]
            _, test_pred, pve = GSapi(
                train_pheno,
                train_snp,
                test_snp,
                method=method,
                PCAdec=args.pcd,
            )
            outpred_list.append(test_pred)
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
