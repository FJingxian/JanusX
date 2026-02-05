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


def _read_id_file(path: str, logger, label: str) -> Union[np.ndarray, None]:
    if not os.path.isfile(path):
        logger.warning(f"{label} ID file not found: {path}")
        return None
    try:
        df = pd.read_csv(
            path, sep=None, engine="python", header=None, usecols=[0],
            dtype=str, keep_default_na=False
        )
    except Exception:
        df = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=[0],
            dtype=str, keep_default_na=False
        )
    if not df.empty and df.iloc[0, 0] == "":
        # sep=None can mis-parse single-column ID files; fallback to whitespace
        df = pd.read_csv(
            path, sep=r"\s+", header=None, usecols=[0],
            dtype=str, keep_default_na=False
        )
    if df.empty:
        logger.warning(f"{label} ID file is empty: {path}")
        return None
    ids = df.iloc[:, 0].astype(str).str.strip().to_numpy()
    if ids.size == 0:
        logger.warning(f"{label} ID file has no usable IDs: {path}")
        return None
    return ids


def _read_matrix_with_ids(path: str, logger, label: str) -> tuple[Union[np.ndarray, None], np.ndarray]:
    try:
        df = pd.read_csv(
            path, sep=None, engine="python", header=None,
            dtype={0: str}, keep_default_na=False
        )
    except Exception:
        df = pd.read_csv(
            path, sep=r"\s+", header=None,
            dtype={0: str}, keep_default_na=False
        )
    if df.shape[1] < 2:
        raise ValueError(f"{label} file must have IDs in column 1 and data in columns 2+.")
    ids = df.iloc[:, 0].astype(str).str.strip().to_numpy()
    data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
    return ids, data


def load_phenotype(
    phenofile: str,
    ncol: Union[list[int] , None],
    logger,
    id_col: int = 0,
) -> pd.DataFrame:
    """
    Load and preprocess phenotype table.

    Assumptions
    -----------
      - By default, the first column contains sample IDs.
      - If needed, set id_col=1 to use the second column as IDs (PLINK FID/IID).
    - Duplicated IDs are averaged.
    """
    logger.info(f"Loading phenotype from {phenofile}...")
    try:
        df = pd.read_csv(phenofile, sep=None, engine="python", header=None)
    except Exception:
        df = pd.read_csv(phenofile, sep=r"\s+", header=None)

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
    pheno = pheno.groupby(pheno.index).mean()

    assert pheno.shape[1] > 0, (
        "No phenotype data found. Please check the phenotype file format.\n"
        f"{pheno.head()}"
    )

    if ncol is not None:
        assert np.min(ncol) < pheno.shape[1], "Phenotype column index out of range."
        ncol = [i for i in ncol if i in range(pheno.shape[1])]
        logger.info(
            "Phenotypes to be analyzed: " + "\t".join(map(str, pheno.columns[ncol]))
        )
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
) -> tuple[np.ndarray, int, Union[np.ndarray, None]]:
    """
    Load or build a GRM with caching for streaming LMM/LM runs.
    """
    ids, n_snps = inspect_genotype_file(genofile)
    n_samples = len(ids)
    method_is_builtin = mgrm in ["1", "2"]

    grm_ids = None
    if method_is_builtin:
        km_path = f"{cache_prefix}.k.{mgrm}"
        id_path = f"{km_path}.npy.id"
        if os.path.exists(f'{km_path}.npy'):
            logger.info(f"Loading cached GRM from {km_path}.npy...")
            grm = np.load(f'{km_path}.npy',mmap_mode='r')
            grm = grm.reshape(n_samples, n_samples)
            grm_ids = _read_id_file(id_path, logger, "GRM")
            if grm_ids is not None and len(grm_ids) != n_samples:
                raise ValueError(
                    f"GRM ID count ({len(grm_ids)}) does not match GRM shape ({n_samples})."
                )
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
            pd.Series(ids).to_csv(id_path, sep="\t", index=False, header=False)
            grm_ids = ids
            grm = np.load(f'{km_path}.npy',mmap_mode='r')
            logger.info(f"Cached GRM written to {km_path}.npy")
    else:
        assert os.path.isfile(mgrm), f"GRM file not found: {mgrm}"
        logger.info(f"Loading GRM from {mgrm}...")
        if mgrm.endswith('.npy'):
            grm = np.load(mgrm,mmap_mode='r')
        else:
            grm = np.genfromtxt(mgrm, dtype="float32")
        grm_ids = _read_id_file(f"{mgrm}.id", logger, "GRM")
        if grm_ids is None:
            assert grm.size == n_samples * n_samples, (
                f"GRM size mismatch: expected {n_samples*n_samples}, got {grm.size}"
            )
            grm = grm.reshape(n_samples, n_samples)
        else:
            assert grm.size == len(grm_ids) * len(grm_ids), (
                f"GRM size mismatch: expected {len(grm_ids)*len(grm_ids)}, got {grm.size}"
            )
            grm = grm.reshape(len(grm_ids), len(grm_ids))
        eff_m = n_snps

    logger.info(f"GRM shape: {grm.shape}")
    return grm, eff_m, grm_ids


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
    ids: np.ndarray,
    logger,
) -> tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Load or build Q matrix (PCs) with caching for streaming LMM/LM.
    When loading from file, the first column is treated as sample IDs and
    the remaining columns are PCs.
    """
    n = grm.shape[0]

    q_ids = None
    if pcdim in np.arange(1, n).astype(str):
        dim = int(pcdim)
        q_path = f"{cache_prefix}.q.{pcdim}.txt"
        if os.path.exists(q_path):
            logger.info(f"Loading cached Q matrix from {q_path}...")
            try:
                q_ids, qmatrix = _read_matrix_with_ids(q_path, logger, "Q")
            except Exception:
                qmatrix = np.genfromtxt(q_path, dtype="float32")
                q_ids = None
                logger.warning("Q cache has no IDs; assuming genotype order.")
        else:
            qmatrix = build_pcs_from_grm(grm, dim, logger)
            df = pd.DataFrame(np.column_stack([ids.astype(str), qmatrix]))
            df.to_csv(q_path, sep="\t", header=False, index=False)
            q_ids = ids
            logger.info(f"Cached Q matrix written to {q_path}")
    elif pcdim == "0":
        logger.info("PC dimension set to 0; using empty Q matrix.")
        qmatrix = np.zeros((n, 0), dtype="float32")
        q_ids = ids
    elif os.path.isfile(pcdim):
        logger.info(f"Loading Q matrix from {pcdim}...")
        q_ids, qmatrix = _read_matrix_with_ids(pcdim, logger, "Q")
    else:
        raise ValueError(f"Unknown Q/PC option: {pcdim}")

    logger.info(f"Q matrix shape: {qmatrix.shape}")
    return qmatrix, q_ids


def _load_covariate_for_streaming(
    cov_path: Union[str , None],
    logger,
) -> tuple[Union[np.ndarray , None], Union[np.ndarray, None]]:
    """
    Load covariate matrix for streaming LMM/LM.

    Assumptions
    -----------
      - The first column contains sample IDs.
      - Remaining columns are covariates.
    """
    if cov_path is None:
        logger.info("No covariate file provided; skipping covariates.")
        return None, None

    if not os.path.isfile(cov_path):
        logger.warning(f"Covariate file not found: {cov_path}; skipping covariates.")
        return None, None

    logger.info(f"Loading covariate matrix for streaming models from {cov_path}...")
    cov_ids, cov_all = _read_matrix_with_ids(cov_path, logger, "Covariate")
    if cov_all.ndim == 1:
        cov_all = cov_all.reshape(-1, 1)
    logger.info(f"Covariate matrix (streaming) shape: {cov_all.shape}")
    return cov_all, cov_ids


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
    pheno = load_phenotype(phenofile, pheno_cols, logger, id_col=0)

    ids, n_snps = inspect_genotype_file(genofile)
    ids = np.array(ids).astype(str)
    n_samples = len(ids)
    logger.info(f"Genotype meta: {n_samples} samples, {n_snps} SNPs.")

    cache_prefix = genotype_cache_prefix(genofile)
    logger.info(f"Cache prefix (genotype folder): {cache_prefix}")
    # GRM stream...
    grm, eff_m, grm_ids = load_or_build_grm_with_cache(
        genofile=genofile,
        cache_prefix=cache_prefix,
        mgrm=mgrm,
        maf_threshold=maf_threshold,
        max_missing_rate=max_missing_rate,
        chunk_size=chunk_size,
        mmap_limit=mmap_limit,
        logger=logger,
    )
    # PCA stream...
    qmatrix, q_ids = load_or_build_q_with_cache(
        grm=grm,
        cache_prefix=cache_prefix,
        pcdim=pcdim,
        ids=ids,
        logger=logger,
    )

    cov_all, cov_ids = _load_covariate_for_streaming(cov_path, logger)

    # -----------------------------------------
    # Align all data sources to shared IDs
    # -----------------------------------------
    geno_ids = ids.astype(str)
    pheno_ids = pheno.index.astype(str).to_numpy()

    # If optional ID files are missing, inherit genotype order
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
            pheno_alt = load_phenotype(phenofile, pheno_cols, logger, id_col=1)
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

    logger.info(
        f"Sample intersection: geno={len(geno_ids)}, pheno={len(pheno_ids)}, "
        f"grm={'NA' if grm_ids is None else len(grm_ids)}, "
        f"q={'NA' if q_ids is None else len(q_ids)}, "
        f"cov={'NA' if cov_ids is None else len(cov_ids)} -> {len(common_ids)}"
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

    grm_idx = [grm_index[sid] for sid in ids]
    grm = grm[np.ix_(grm_idx, grm_idx)]

    q_idx = [q_index[sid] for sid in ids]
    qmatrix = qmatrix[q_idx]

    if cov_all is not None:
        cov_idx = [cov_index[sid] for sid in ids]
        cov_all = cov_all[cov_idx]

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
    sample_ids: Union[np.ndarray, None] = None,
) -> np.ndarray:
    """
    Build or load Q matrix for FarmCPU (PCs + optional covariates).
    """
    def _maybe_load_with_ids(path: str, expect_rows: int):
        df = pd.read_csv(
            path, sep=None, engine="python", header=None,
            dtype=str, keep_default_na=False
        )
        if df.shape[0] != expect_rows:
            raise ValueError(
                f"Q matrix rows ({df.shape[0]}) do not match sample count ({expect_rows})."
            )
        if sample_ids is not None:
            col0 = df.iloc[:, 0].astype(str).str.strip()
            overlap = len(set(col0) & set(sample_ids))
            if overlap >= int(0.9 * len(sample_ids)) and df.shape[1] > 1:
                data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
                index = {sid: i for i, sid in enumerate(col0)}
                missing = [sid for sid in sample_ids if sid not in index]
                if missing:
                    raise ValueError(f"Q matrix missing {len(missing)} sample IDs (e.g. {missing[:5]}).")
                return data[[index[sid] for sid in sample_ids]]
        # Fallback: treat all columns as numeric Q matrix
        q = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
        return q

    if qdim in np.arange(0, 30).astype(str):
        q_path = f"{gfile_prefix}.q.{qdim}.txt"
        if os.path.exists(q_path):
            logger.info(f"* Loading Q matrix from {q_path}...")
            qmatrix = _maybe_load_with_ids(q_path, geno.shape[1])
        elif qdim == "0":
            qmatrix = np.array([]).reshape(geno.shape[1], 0)
        else:
            logger.info(f"* PCA dimension for FarmCPU Q matrix: {qdim}")
            _eigval, eigvec = np.linalg.eigh(GRM(geno))
            qmatrix = eigvec[:, -int(qdim):]
            if sample_ids is not None:
                df = pd.DataFrame(np.column_stack([sample_ids.astype(str), qmatrix]))
                df.to_csv(q_path, sep="\t", header=False, index=False)
            else:
                np.savetxt(q_path, qmatrix, fmt="%.6f")
            logger.info(f"Cached Q matrix written to {q_path}")
    else:
        logger.info(f"* Loading Q matrix from {qdim}...")
        qmatrix = _maybe_load_with_ids(qdim, geno.shape[1])

    if cov_path:
        # cov file may contain IDs in the first column
        df = pd.read_csv(
            cov_path, sep=None, engine="python", header=None,
            dtype=str, keep_default_na=False
        )
        if sample_ids is not None:
            col0 = df.iloc[:, 0].astype(str).str.strip()
            overlap = len(set(col0) & set(sample_ids))
            if overlap >= int(0.9 * len(sample_ids)) and df.shape[1] > 1:
                cov_arr = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
                index = {sid: i for i, sid in enumerate(col0)}
                missing = [sid for sid in sample_ids if sid not in index]
                if missing:
                    raise ValueError(f"Covariate file missing {len(missing)} sample IDs (e.g. {missing[:5]}).")
                cov_arr = cov_arr[[index[sid] for sid in sample_ids]]
            else:
                cov_arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")
        else:
            cov_arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32")

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
        sample_ids=famid.astype(str),
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
