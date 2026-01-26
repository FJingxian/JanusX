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
  - LMM/LM always run in streaming mode via rust2py.gfreader.load_genotype_chunks.
  - FarmCPU always runs on the full in-memory genotype matrix.

Caching
-------
  - GRM (kinship) and PCA (Q matrix) are cached in the genotype directory
    for streaming LMM/LM runs:
      * GRM: {geno_prefix}.k.{method}.npy
      * Q   : {geno_prefix}.q.{pcdim}.txt

Covariates
----------
  - The --cov option is shared by LMM, LM, and FarmCPU.
  - For LMM/LM, the covariate file must match the genotype sample order
    (inspect_genotype_file IDs).
  - For FarmCPU, the covariate file must match the genotype sample order
    (famid from the genotype matrix).

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import os
import time
import socket
import argparse
import logging
from typing import Union
import uuid

from janusx.pyBLUP.QK2 import GRM

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import cpu_count
from tqdm import tqdm
import psutil
from janusx.bioplotkit import GWASPLOT
from janusx.pyBLUP import QK
from janusx.gfreader import breader, vcfreader
from janusx.gfreader import (
    load_genotype_chunks,
    inspect_genotype_file,
    auto_mmap_window_mb,
)
from janusx.pyBLUP import LMM, LM, FastLMM, farmcpu
from ._common.log import setup_logging


# ======================================================================
# Basic utilities
# ======================================================================

def _section(logger:logging.Logger, title: str) -> None:
    """Emit a formatted log section header with a leading blank line."""
    logger.info("")
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)


def fastplot(
    gwasresult: pd.DataFrame,
    phenosub: np.ndarray,
    xlabel: str = "",
    outpdf: str = "fastplot.pdf",
) -> None:
    """
    Generate diagnostic plots for GWAS results: phenotype histogram, Manhattan, and QQ.
    """
    mpl.rcParams["font.size"] = 12
    results = gwasresult.astype({"pos": "int64"})
    fig = plt.figure(figsize=(16, 4), dpi=300)
    layout = [["A", "B", "B", "C"]]
    axes:dict[str,plt.Axes] = fig.subplot_mosaic(mosaic=layout)

    gwasplot = GWASPLOT(results)

    # A: phenotype distribution
    axes["A"].hist(phenosub, bins=15)
    axes["A"].set_xlabel(xlabel)
    axes["A"].set_ylabel("Count")

    # B: Manhattan plot
    gwasplot.manhattan(-np.log10(1 / results.shape[0]), ax=axes["B"],rasterized=True)

    # C: QQ plot
    gwasplot.qq(ax=axes["C"])

    plt.tight_layout()
    plt.savefig(outpdf, transparent=False, facecolor="white")


def determine_genotype_source(args) -> tuple[str, str]:
    """
    Resolve genotype input and output prefix from CLI arguments.
    """
    if args.vcf:
        gfile = args.vcf
        prefix = os.path.basename(gfile).replace(".gz", "").replace(".vcf", "")
    elif args.bfile:
        gfile = args.bfile
        prefix = os.path.basename(gfile)
    else:
        raise ValueError("No genotype input specified. Use -vcf or -bfile.")

    if args.prefix is not None:
        prefix = args.prefix

    gfile = gfile.replace("\\", "/")
    return gfile, prefix


def genotype_cache_prefix(genofile: str) -> str:
    """
    Construct a cache prefix within the genotype directory.
    """
    base = os.path.basename(genofile)
    if base.endswith(".vcf.gz"):
        base = base[: -len(".vcf.gz")]
    elif base.endswith(".vcf"):
        base = base[: -len(".vcf")]
    cache_dir = os.path.dirname(genofile) or "."
    return os.path.join(cache_dir, base).replace("\\", "/")


def load_phenotype(phenofile: str, ncol: Union[list[int] , None], logger) -> pd.DataFrame:
    """
    Load and preprocess phenotype table.

    Assumptions
    -----------
      - First column contains sample IDs.
      - Duplicated IDs are averaged.
    """
    logger.info(f"Loading phenotype from {phenofile}...")
    pheno = pd.read_csv(phenofile, sep="\t")
    pheno = pheno.groupby(pheno.columns[0]).mean()
    pheno.index = pheno.index.astype(str)

    assert pheno.shape[1] > 0, (
        "No phenotype data found. Please check the phenotype file format.\n"
        f"{pheno.head()}"
    )

    if ncol is not None:
        assert np.min(ncol) < pheno.shape[1], "Phenotype column index out of range."
        ncol = [i for i in ncol if i in range(pheno.shape[1])]
        logger.info("Phenotypes to be analyzed: " + "\t".join(pheno.columns[ncol]))
        pheno = pheno.iloc[:, ncol]

    return pheno


# ======================================================================
# Low-memory LMM/LM: streaming GRM + PCA with caching
# ======================================================================

def build_grm_streaming(
    genofile: str,
    n_samples: int,
    n_snps: int,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    method: int,
    mmap_window_mb: Union[int , None],
    logger,
) -> tuple[np.ndarray, int]:
    """
    Build GRM in a streaming fashion using rust2py.gfreader.load_genotype_chunks.
    """
    logger.info(f"Building GRM (streaming), method={method}")
    grm = np.zeros((n_samples, n_samples), dtype="float32")
    pbar = tqdm(total=n_snps, desc="GRM (streaming)", ascii=False)
    process = psutil.Process()

    varsum = 0.0
    eff_m = 0
    for genosub, _sites in load_genotype_chunks(
        genofile,
        chunk_size,
        maf_threshold,
        max_missing_rate,
        mmap_window_mb=mmap_window_mb,
    ):
        # genosub: (m_chunk, n_samples)
        genosub:np.ndarray
        maf = genosub.mean(axis=1, dtype="float32", keepdims=True) / 2
        genosub = genosub - 2 * maf

        if method == 1:
            grm += genosub.T @ genosub
            varsum += np.sum(2 * maf * (1 - maf))
        elif method == 2:
            w = 1.0 / (2 * maf * (1 - maf))              # (m_chunk,1)
            grm += (genosub.T * w.ravel()) @ genosub     # (n_samples, n_samples)
        else:
            raise ValueError(f"Unsupported GRM method: {method}")

        eff_m += genosub.shape[0]
        pbar.update(genosub.shape[0])

        if eff_m % (10 * chunk_size) == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    # force bar to 100% even if SNPs were filtered in Rust
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    if method == 1:
        grm = (grm + grm.T) / varsum / 2
    else:  # method == 2
        grm = (grm + grm.T) / eff_m / 2

    logger.info("GRM construction finished.")
    return grm, eff_m


def load_or_build_grm_with_cache(
    genofile: str,
    cache_prefix: str,
    mgrm: str,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    mmap_limit: bool,
    logger:logging.Logger,
) -> tuple[np.ndarray, int]:
    """
    Load or build a GRM with caching for streaming LMM/LM runs.
    """
    ids, n_snps = inspect_genotype_file(genofile)
    n_samples = len(ids)
    method_is_builtin = mgrm in ["1", "2"]

    if method_is_builtin:
        km_path = f"{cache_prefix}.k.{mgrm}"
        if os.path.exists(f'{km_path}.npy'):
            logger.info(f"Loading cached GRM from {km_path}.npy...")
            grm = np.load(f'{km_path}.npy',mmap_mode='r')
            grm = grm.reshape(n_samples, n_samples)
            eff_m = n_snps  # approximate; exact effective M not critical here
        else:
            method_int = int(mgrm)
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
                logger=logger,
            )
            np.save(f'{km_path}.npy', grm)
            grm = np.load(f'{km_path}.npy',mmap_mode='r')
            logger.info(f"Cached GRM written to {km_path}.npy")
    else:
        assert os.path.isfile(mgrm), f"GRM file not found: {mgrm}"
        logger.info(f"Loading GRM from {mgrm}...")
        grm = np.genfromtxt(mgrm, dtype="float32")
        assert grm.size == n_samples * n_samples, (
            f"GRM size mismatch: expected {n_samples*n_samples}, got {grm.size}"
        )
        grm = grm.reshape(n_samples, n_samples)
        eff_m = n_snps

    logger.info(f"GRM shape: {grm.shape}")
    return grm, eff_m


def build_pcs_from_grm(grm: np.ndarray, dim: int, logger: logging.Logger) -> np.ndarray:
    """
    Compute leading principal components from GRM.
    """
    logger.info(f"Computing top {dim} PCs from GRM...")
    _, eigvec = np.linalg.eigh(grm)
    pcs = eigvec[:, -dim:]
    logger.info("PC computation finished.")
    return pcs


def load_or_build_q_with_cache(
    grm: np.ndarray,
    cache_prefix: str,
    pcdim: str,
    logger,
) -> np.ndarray:
    """
    Load or build Q matrix (PCs) with caching for streaming LMM/LM.
    """
    n = grm.shape[0]

    if pcdim in np.arange(1, n).astype(str):
        dim = int(pcdim)
        q_path = f"{cache_prefix}.q.{pcdim}.txt"
        if os.path.exists(q_path):
            logger.info(f"Loading cached Q matrix from {q_path}...")
            qmatrix = np.genfromtxt(q_path, dtype="float32")
        else:
            qmatrix = build_pcs_from_grm(grm, dim, logger)
            np.savetxt(q_path, qmatrix, fmt="%.6f")
            logger.info(f"Cached Q matrix written to {q_path}")
    elif pcdim == "0":
        logger.info("PC dimension set to 0; using empty Q matrix.")
        qmatrix = np.zeros((n, 0), dtype="float32")
    elif os.path.isfile(pcdim):
        logger.info(f"Loading Q matrix from {pcdim}...")
        qmatrix = np.genfromtxt(pcdim, dtype="float32")
        assert qmatrix.shape[0] == n, (
            f"Q matrix row count mismatch: expected {n}, got {qmatrix.shape[0]}"
        )
    else:
        raise ValueError(f"Unknown Q/PC option: {pcdim}")

    logger.info(f"Q matrix shape: {qmatrix.shape}")
    return qmatrix


def _load_covariate_for_streaming(
    cov_path: Union[str , None],
    n_samples: int,
    logger,
) -> Union[np.ndarray , None]:
    """
    Load covariate matrix for streaming LMM/LM.

    Assumptions
    -----------
      - The covariate file is aligned with the genotype sample order
        given by inspect_genotype_file (one row per sample).
    """
    if cov_path is None:
        return None

    logger.info(f"Loading covariate matrix for streaming models from {cov_path}...")
    cov_all = np.genfromtxt(cov_path, dtype="float32")
    if cov_all.ndim == 1:
        cov_all = cov_all.reshape(-1, 1)
    assert cov_all.shape[0] == n_samples, (
        f"Covariate rows ({cov_all.shape[0]}) do not match sample count "
        f"({n_samples}) from genotype metadata."
    )
    logger.info(f"Covariate matrix (streaming) shape: {cov_all.shape}")
    return cov_all


def prepare_streaming_context(
    genofile: str,
    phenofile: str,
    pheno_cols: Union[list[int] , None],
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    mgrm: str,
    pcdim: str,
    cov_path: Union[str , None],
    mmap_limit: bool,
    logger,
):
    """
    Prepare all shared resources for streaming LMM/LM once:
      - phenotype
      - genotype metadata (ids, n_snps)
      - GRM + Q (cached)
      - covariates (optional)
    """
    pheno = load_phenotype(phenofile, pheno_cols, logger)

    ids, n_snps = inspect_genotype_file(genofile)
    ids = np.array(ids).astype(str)
    n_samples = len(ids)
    logger.info(f"Genotype meta: {n_samples} samples, {n_snps} SNPs.")

    cache_prefix = genotype_cache_prefix(genofile)
    logger.info(f"Cache prefix (genotype folder): {cache_prefix}")

    grm, eff_m = load_or_build_grm_with_cache(
        genofile=genofile,
        cache_prefix=cache_prefix,
        mgrm=mgrm,
        maf_threshold=maf_threshold,
        max_missing_rate=max_missing_rate,
        chunk_size=chunk_size,
        mmap_limit=mmap_limit,
        logger=logger,
    )

    qmatrix = load_or_build_q_with_cache(
        grm=grm,
        cache_prefix=cache_prefix,
        pcdim=pcdim,
        logger=logger,
    )

    cov_all = _load_covariate_for_streaming(cov_path, n_samples, logger)

    return pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m


def run_chunked_gwas_lmm_lm(
    model_name: str,
    genofile: str,
    pheno: pd.DataFrame,
    ids: np.ndarray,
    n_snps: int,
    outprefix: str,
    maf_threshold: float,
    max_missing_rate: float,
    chunk_size: int,
    mmap_limit: bool,
    grm: np.ndarray,
    qmatrix: np.ndarray,
    cov_all: Union[np.ndarray , None],
    eff_m: int,
    plot: bool,
    threads: int,
    logger:logging.Logger,
) -> None:
    """
    Run LMM or LM GWAS using a streaming, low-memory pipeline.

    Important: This function assumes pheno/ids/grm/q/cov have already been prepared
    once (no repeated "Loading phenotype" / "Loading GRM/Q" logs).
    """
    model_map = {"lmm": LMM, "lm": LM, "fastlmm": FastLMM}
    model_key = model_name.lower()
    ModelCls = model_map[model_key]
    model_label = {"lmm": "LMM", "lm": "LM", "fastlmm": "fastLMM"}[model_key]
    # Keep output file suffixes consistent and lowercase.
    model_tag = model_label.lower()

    process = psutil.Process()
    n_cores = psutil.cpu_count(logical=True) or cpu_count()

    for pname in pheno.columns:
        logger.info(f"Streaming {model_label} GWAS for trait: {pname}")

        cpu_t0 = process.cpu_times()
        rss0 = process.memory_info().rss
        t0 = time.time()
        peak_rss = rss0

        pheno_sub = pheno[pname].dropna()
        sameidx = np.isin(ids, pheno_sub.index)
        if np.sum(sameidx) == 0:
            logger.info(
                f"No overlapping samples between genotype and phenotype {pname}. Skipped."
            )
            continue

        y_vec = pheno_sub.loc[ids[sameidx]].values
        # Build covariate matrix X_cov for this trait
        X_cov = qmatrix[sameidx]
        if cov_all is not None:
            X_cov = np.concatenate([X_cov, cov_all[sameidx]], axis=1)

        if model_key in ("lmm", "fastlmm"):
            Ksub = grm[np.ix_(sameidx, sameidx)]
            mod = ModelCls(y=y_vec, X=X_cov, kinship=Ksub)
            logger.info(
                f"Samples: {np.sum(sameidx)}, Total SNPs: {eff_m}, PVE(null): {mod.pve:.3f}"
            )
        else:
            mod = ModelCls(y=y_vec, X=X_cov)
            logger.info(f"Samples: {np.sum(sameidx)}, Total SNPs: {eff_m}")

        done_snps = 0
        has_results = False
        out_tsv = f"{outprefix}.{pname}.{model_tag}.tsv"
        tmp_tsv = f"{out_tsv}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        wrote_header = False
        mmap_window_mb = (
            auto_mmap_window_mb(genofile, len(ids), n_snps, chunk_size)
            if mmap_limit else None
        )

        process.cpu_percent(interval=None)
        pbar = tqdm(total=n_snps, desc=f"{model_label}-{pname}", ascii=False)

        sample_sub = None if genofile.endswith('.vcf') or genofile.endswith('.vcf.gz') else ids[sameidx]
        for genosub, sites in load_genotype_chunks(
            genofile,
            chunk_size,
            maf_threshold,
            max_missing_rate,
            sample_ids=sample_sub,
            mmap_window_mb=mmap_window_mb,
        ):
            genosub:np.ndarray
            genosub = genosub[:, sameidx]  if sample_sub is None else genosub # (m_chunk, n_use)
            m_chunk = genosub.shape[0]
            if m_chunk == 0:
                continue

            maf_chunk = np.mean(genosub, axis=1) / 2
            results = mod.gwas(genosub-2*maf_chunk.reshape(-1,1), threads=threads) # Centralization of input genotype
            info_chunk = [
                (s.chrom, s.pos, s.ref_allele, s.alt_allele) for s in sites
            ]
            if not info_chunk:
                continue

            chroms, poss, allele0, allele1 = zip(*info_chunk)
            chunk_df = pd.DataFrame(
                {
                    "chrom": chroms,
                    "pos": poss,
                    "allele0": allele0,
                    "allele1": allele1,
                    "maf": maf_chunk,
                    "beta": results[:, 0],
                    "se": results[:, 1],
                    "pwald": results[:, 2],
                }
            )
            if results.shape[1] > 3:
                chunk_df["plrt"] = results[:, 3]
                chunk_df["plrt"] = chunk_df["plrt"].map(lambda x: f"{x:.4e}")
            chunk_df["pos"] = chunk_df["pos"].astype(int)

            chunk_df["pwald"] = chunk_df["pwald"].map(lambda x: f"{x:.4e}")
            chunk_df.to_csv(
                tmp_tsv,
                sep="\t",
                float_format="%.4f",
                index=False,
                header=not wrote_header,
                mode="w" if not wrote_header else "a",
            )
            wrote_header = True
            has_results = True

            done_snps += m_chunk
            pbar.update(m_chunk)

            mem_info = process.memory_info()
            peak_rss = max(peak_rss, mem_info.rss)
            if done_snps % (10 * chunk_size) == 0:
                mem_gb = mem_info.rss / 1024**3
                pbar.set_postfix(memory=f"{mem_gb:.2f} GB")

        pbar.n = pbar.total
        pbar.refresh()
        pbar.close()

        cpu_t1 = process.cpu_times()
        rss1 = process.memory_info().rss
        t1 = time.time()

        wall = t1 - t0
        user_cpu = cpu_t1.user - cpu_t0.user
        sys_cpu = cpu_t1.system - cpu_t0.system
        total_cpu = user_cpu + sys_cpu

        avg_cpu_pct = 100.0 * total_cpu / wall / (n_cores or 1) if wall > 0 else 0.0
        avg_rss_gb = (rss0 + rss1) / 2 / 1024**3
        peak_rss_gb = peak_rss / 1024**3

        logger.info(
            f"Effective SNP: {done_snps} | "
            f"Resource usage for {model_label} / {pname}: \n"
            f"wall={wall:.2f} s, "
            f"avg CPU={avg_cpu_pct:.1f}% of {n_cores} cores, "
            f"avg RSS={avg_rss_gb:.2f} GB, "
            f"peak RSS ~ {peak_rss_gb:.2f} GB\n"
        )

        if not has_results:
            logger.info(f"No SNPs passed filters for trait {pname}.")
            if os.path.exists(tmp_tsv):
                os.remove(tmp_tsv)
            continue

        if plot:
            plot_df = pd.read_csv(
                tmp_tsv,
                sep="\t",
                usecols=["chrom", "pos", "pwald"],
                dtype={"chrom": str, "pos": "int64"},
            )
            plot_df["pwald"] = pd.to_numeric(plot_df["pwald"], errors="coerce")
            fastplot(
                plot_df,
                y_vec,
                xlabel=pname,
                outpdf=f"{outprefix}.{pname}.{model_tag}.svg",
            )

        os.replace(tmp_tsv, out_tsv)
        logger.info(f"Saved {model_label} results to {out_tsv}".replace("//", "/"))
        logger.info("")  # ensure blank line between traits


# ======================================================================
# High-memory FarmCPU: full genotype + QK
# ======================================================================

def prepare_qk_and_filter(
    geno: np.ndarray,
    ref_alt: pd.DataFrame,
    maf_threshold: float,
    max_missing_rate: float,
    logger,
):
    """
    Filter SNPs and impute missing values using QK, then update ref_alt.
    """
    logger.info(
        "* Filtering SNPs (MAF < "
        f"{maf_threshold} or missing rate > {max_missing_rate}; mode imputation)..."
    )
    logger.info("  Tip: if available, use pre-imputed genotypes from BEAGLE/IMPUTE2.")
    qkmodel = QK(geno, maff=maf_threshold, missf=max_missing_rate)
    geno_filt = qkmodel.M

    ref_alt_filt = ref_alt.loc[qkmodel.SNPretain].copy()
    # Swap REF/ALT for extremely rare alleles
    ref_alt_filt.iloc[qkmodel.maftmark, [0, 1]] = ref_alt_filt.iloc[
        qkmodel.maftmark, [1, 0]
    ]
    ref_alt_filt["maf"] = qkmodel.maf
    logger.info("Filtering and imputation finished.")
    return geno_filt, ref_alt_filt, qkmodel


def build_qmatrix_farmcpu(
    gfile_prefix: str,
    geno: np.ndarray,
    qdim: str,
    cov_path: Union[str , None],
    logger,
) -> np.ndarray:
    """
    Build or load Q matrix for FarmCPU (PCs + optional covariates).
    """
    if qdim in np.arange(0, 30).astype(str):
        q_path = f"{gfile_prefix}.q.{qdim}.txt"
        if os.path.exists(q_path):
            logger.info(f"* Loading Q matrix from {q_path}...")
            qmatrix = np.genfromtxt(q_path,dtype="float32")
        elif qdim == "0":
            qmatrix = np.array([]).reshape(geno.shape[1], 0)
        else:
            logger.info(f"* PCA dimension for FarmCPU Q matrix: {qdim}")
            qmatrix, _eigval = np.linalg.eigh(GRM(geno))
            qmatrix = qmatrix[:, -int(qdim):]
            np.savetxt(q_path, qmatrix, fmt="%.6f")
            logger.info(f"Cached Q matrix written to {q_path}")
    else:
        logger.info(f"* Loading Q matrix from {qdim}...")
        qmatrix = np.genfromtxt(qdim, dtype="float32")

    if cov_path:
        cov_arr = np.genfromtxt(cov_path, dtype=float)
        if cov_arr.ndim == 1:
            cov_arr = cov_arr.reshape(-1, 1)
        assert cov_arr.shape[0] == geno.shape[1], (
            f"Covariate rows ({cov_arr.shape[0]}) do not match sample count "
            f"({geno.shape[1]}) in genotype matrix."
        )
        logger.info(f"Appending covariate matrix for FarmCPU: shape={cov_arr.shape}")
        qmatrix = np.concatenate([qmatrix, cov_arr], axis=1)

    logger.info(f"Q matrix (FarmCPU) shape: {qmatrix.shape}")
    return qmatrix


def run_farmcpu_fullmem(
    args,
    gfile: str,
    prefix: str,
    logger: logging.Logger,
    pheno_preloaded: Union[pd.DataFrame , None] = None,
) -> None:
    """
    Run FarmCPU in high-memory mode (full genotype + QK + PCA).

    If pheno_preloaded is provided, it will reuse that phenotype table to avoid
    repeated "Loading phenotype ..." logs and repeated I/O.
    """
    t_loading = time.time()
    phenofile = args.pheno
    outfolder = args.out
    qdim = args.qcov
    cov = args.cov

    logger.info("* FarmCPU pipeline: loading genotype and phenotype")
    pheno = pheno_preloaded if pheno_preloaded is not None else load_phenotype(
        phenofile, args.ncol, logger
    )
    if gfile.endswith('vcf') or gfile.endswith('vcf.gz'):
        geno = vcfreader(gfile, args.chunksize,maf=args.maf,miss=args.geno,impute=True)
    else:
        geno = breader(gfile, args.chunksize,maf=args.maf,miss=args.geno,impute=True)
    ref_alt = geno.iloc[:,:2]
    famid = geno.columns[2:]
    geno = geno.iloc[:,2:].values    
    logger.info(
        f"Genotype and phenotype loaded in {(time.time() - t_loading):.2f} seconds"
    )
    assert geno.size > 0, "After filtering, number of SNPs is zero for FarmCPU."

    gfile_prefix = gfile.replace(".vcf", "").replace(".gz", "")
    qmatrix = build_qmatrix_farmcpu(
        gfile_prefix=gfile_prefix,
        geno=geno,
        qdim=qdim,
        cov_path=cov,
        logger=logger,
    )

    for phename in pheno.columns:
        logger.info(f"* FarmCPU GWAS for trait: {phename}")
        t_trait = time.time()

        p = pheno[phename].dropna()
        famidretain = np.isin(famid, p.index)
        if np.sum(famidretain) == 0:
            logger.info(f"Trait {phename}: no overlapping samples, skipped.")
            continue

        snp_sub = geno[:, famidretain]
        p_sub = p.loc[famid[famidretain]].values.reshape(-1, 1)
        q_sub = qmatrix[famidretain]

        logger.info(f"Samples: {np.sum(famidretain)}, SNPs: {snp_sub.shape[0]}")
        maf = snp_sub.mean(axis=1)/2
        res = farmcpu(
            y=p_sub,
            M=snp_sub,
            X=q_sub,
            chrlist=ref_alt.reset_index().iloc[:, 0].values,
            poslist=ref_alt.reset_index().iloc[:, 1].values,
            iter=20,
            threads=args.thread,
        )
        res_df = pd.DataFrame(res, columns=["beta", "se", "pwald"], index=ref_alt.index)
        res_df['maf'] = maf
        res_df = pd.concat([ref_alt, res_df], axis=1)
        res_df = res_df.reset_index()
        res_df.columns = ['chrom','pos','allele0','allele1','beta','se','pwald','maf']
        res_df = res_df[['chrom','pos','allele0','allele1','maf','beta','se','pwald']]

        if args.plot:
            fastplot(
                res_df,
                p_sub,
                xlabel=phename,
                outpdf=f"{outfolder}/{prefix}.{phename}.farmcpu.svg",
            )

        res_df = res_df.astype({"pwald": "object","pos":int})
        res_df.loc[:, "pwald"] = res_df["pwald"].map(lambda x: f"{x:.4e}")
        out_tsv = f"{outfolder}/{prefix}.{phename}.farmcpu.tsv"
        res_df.to_csv(out_tsv, sep="\t", float_format="%.4f", index=None)
        logger.info(f"FarmCPU results saved to {out_tsv}".replace("//", "/"))
        logger.info(f"Trait {phename} finished in {time.time() - t_trait:.2f} s")
        logger.info("")


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    required_group = parser.add_argument_group("Required arguments")

    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help="Input genotype in PLINK binary format "
             "(prefix for .bed, .bim, .fam).",
    )

    required_group.add_argument(
        "-p", "--pheno", type=str, required=True,
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

    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-n", "--ncol", action="extend", nargs="*",
        default=None, type=int,
        help="Zero-based phenotype column indices to analyze. "
             'E.g., "-n 0 -n 3" to analyze the 1st and 4th traits '
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-k", "--grm", type=str, default="1",
        help="GRM option: 1 (centering), 2 (standardization), "
             "or a path to a precomputed GRM file (default: %(default)s).",
    )
    optional_group.add_argument(
        "-q", "--qcov", type=str, default="0",
        help="Number of principal components for Q matrix or path to Q file "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-c", "--cov", type=str, default=None,
        help="Path to additional covariate file. "
             "For LMM/LM, the file must be aligned with the genotype sample "
             "order from inspect_genotype_file (one row per sample). "
             "For FarmCPU, it must follow the genotype sample order "
             "(famid) (default: %(default)s).",
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
        "-plot", "--plot", action="store_true", default=False,
        help="Generate diagnostic plots (histogram, Manhattan, QQ; default: %(default)s).",
    )
    optional_group.add_argument(
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Number of SNPs per chunk for streaming LMM/LM "
             "(affects GRM and GWAS; default: %(default)s).",
    )
    optional_group.add_argument(
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=-1,
        help="Number of CPU threads (-1 uses all available cores; default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for results (default: %(default)s).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix for output files (default: %(default)s).",
    )

    return parser.parse_args()


def main(log: bool = True):
    t_start = time.time()
    args = parse_args()

    if args.thread <= 0:
        args.thread = cpu_count()

    gfile, prefix = determine_genotype_source(args)

    os.makedirs(args.out, 0o755, exist_ok=True)
    outprefix = f"{args.out}/{prefix}".replace("\\", "/").replace("//", "/")
    log_path = f"{outprefix}.gwas.log"
    logger = setup_logging(log_path)

    logger.info(
        "JanusX - High Performance GWAS CLI "
        "(LMM/LM: streaming low-memory; FarmCPU: full-memory)"
    )
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("GWAS CONFIGURATION")
        logger.info("*" * 60)
        logger.info(f"Genotype file:    {gfile}")
        logger.info(f"Phenotype file:   {args.pheno}")
        logger.info(f"Phenotype cols:   {args.ncol if args.ncol is not None else 'All'}")
        logger.info(f"Mmap limit:       {args.mmap_limit}")
        logger.info(
            f"Models:           "
            f"{'LMM ' if args.lmm else ''}"
            f"{'fastlmm ' if args.fastlmm else ''}"
            f"{'LM ' if args.lm else ''}"
            f"{'FarmCPU' if args.farmcpu else ''}"
        )
        logger.info(f"GRM option:       {args.grm}")
        logger.info(f"Q option:         {args.qcov}")
        if args.cov:
            logger.info(f"Covariate file:   {args.cov}")
        logger.info(f"Maf threshold:    {args.maf}")
        logger.info(f"Miss threshold:   {args.geno}")
        logger.info(f"Chunk size:       {args.chunksize}")
        logger.info(f"Threads:          {args.thread} ({cpu_count()} available)")
        logger.info(f"Output prefix:    {outprefix}")
        logger.info("*" * 60 + "\n")

    try:
        assert os.path.isfile(args.pheno), f"Cannot find phenotype file {args.pheno}"
        grm_is_valid = args.grm in ["1", "2"] or os.path.isfile(args.grm)
        q_is_valid = args.qcov in np.arange(0, 30).astype(str) or os.path.isfile(args.qcov)
        assert grm_is_valid, f"{args.grm} is neither GRM method nor an existing GRM file."
        assert q_is_valid, f"{args.qcov} is neither PC dimension nor Q matrix file."
        assert args.cov is None or os.path.isfile(args.cov), f"Covariate file {args.cov} does not exist."
        assert (args.lm or args.lmm or args.fastlmm or args.farmcpu), (
            "No model selected. Use -lm, -lmm, -fastlmm, and/or -farmcpu."
        )

        # --- prepare streaming context once if needed ---
        pheno = None
        ids = None
        n_snps = None
        grm = None
        qmatrix = None
        cov_all = None
        eff_m = None

        if args.lmm or args.lm or args.fastlmm:
            _section(logger, "Prepare streaming context (phenotype/genotype meta/GRM/Q/cov)")
            pheno, ids, n_snps, grm, qmatrix, cov_all, eff_m = prepare_streaming_context(
                genofile=gfile,
                phenofile=args.pheno,
                pheno_cols=args.ncol,
                maf_threshold=args.maf,
                max_missing_rate=args.geno,
                chunk_size=args.chunksize,
                mgrm=args.grm,
                pcdim=args.qcov,
                cov_path=args.cov,
                mmap_limit=args.mmap_limit,
                logger=logger,
            )

        # --- run streaming LMM ---
        if args.lmm:
            _section(logger, "Run streaming LMM")
            run_chunked_gwas_lmm_lm(
                model_name="lmm",
                genofile=gfile,
                pheno=pheno,
                ids=ids,
                n_snps=n_snps,
                outprefix=outprefix,
                maf_threshold=args.maf,
                max_missing_rate=args.geno,
                chunk_size=args.chunksize,
                mmap_limit=args.mmap_limit,
                grm=grm,
                qmatrix=qmatrix,
                cov_all=cov_all,
                eff_m=eff_m,
                plot=args.plot,
                threads=args.thread,
                logger=logger,
            )

        # --- run streaming LM ---
        if args.fastlmm:
            _section(logger, "Run streaming fastLMM (fixed lambda)")
            run_chunked_gwas_lmm_lm(
                model_name="fastlmm",
                genofile=gfile,
                pheno=pheno,
                ids=ids,
                n_snps=n_snps,
                outprefix=outprefix,
                maf_threshold=args.maf,
                max_missing_rate=args.geno,
                chunk_size=args.chunksize,
                mmap_limit=args.mmap_limit,
                grm=grm,
                qmatrix=qmatrix,
                cov_all=cov_all,
                eff_m=eff_m,
                plot=args.plot,
                threads=args.thread,
                logger=logger,
            )

        if args.lm:
            _section(logger, "Run streaming LM")
            run_chunked_gwas_lmm_lm(
                model_name="lm",
                genofile=gfile,
                pheno=pheno,
                ids=ids,
                n_snps=n_snps,
                outprefix=outprefix,
                maf_threshold=args.maf,
                max_missing_rate=args.geno,
                chunk_size=args.chunksize,
                mmap_limit=args.mmap_limit,
                grm=grm,
                qmatrix=qmatrix,
                cov_all=cov_all,
                eff_m=eff_m,
                plot=args.plot,
                threads=args.thread,
                logger=logger,
            )

        # --- run FarmCPU (full memory) ---
        if args.farmcpu:
            _section(logger, "Run FarmCPU (full memory)")
            run_farmcpu_fullmem(
                args=args,
                gfile=gfile,
                prefix=prefix,
                logger=logger,
                pheno_preloaded=pheno,  # 若 streaming 已加载 pheno，则复用，避免重复 log
            )

    except Exception as e:
        logger.exception(f"Error in JanusX GWAS pipeline: {e}")

    lt = time.localtime()
    endinfo = (
        f"\nFinished. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
