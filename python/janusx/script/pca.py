# -*- coding: utf-8 -*-
"""
JanusX: Principal Component Analysis (PCA) Command-Line Interface

Design overview
---------------
Input modes:
  - VCF    : genotype in VCF/VCF.GZ format
  - BFILE  : genotype in PLINK binary format (.bed/.bim/.fam prefix)
  - GRM    : precomputed genetic relationship matrix (GRM) plus ID file
  - QCOV : precomputed PCA results (eigenvec/eigenval) for
             visualization only

PCA computation strategy
------------------------
  - For VCF/BFILE:
      * Stream genotypes via rust2py.gfreader.load_genotype_chunks.
      * Build the GRM in a low-memory, chunk-based fashion.
      * Perform eigendecomposition of the GRM using numpy.linalg.eigh.

  - For GRM:
      * Read the precomputed GRM from .grm.txt or .grm.npy.
      * Perform eigendecomposition using numpy.linalg.eigh.

  - For QCOV:
      * Load PC coordinates and eigenvalues directly.
      * Produce only visualization outputs.

Output:
  - {prefix}.eigenvec      : sample IDs + PC coordinates (first column is ID)
  - {prefix}.eigenval      : eigenvalues (variance along each PC)
  - Optional plots:
      * {prefix}.eigenvec.2D.pdf
      * {prefix}.eigenvec.3D.gif
"""

import os
from typing import Union
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import time
import socket
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil

mpl.use("Agg")
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
from janusx.bioplotkit.sci_set import color_set
from janusx.bioplotkit import PCSHOW
from janusx.gfreader import (
    load_genotype_chunks,
    inspect_genotype_file,
    auto_mmap_window_mb,
)
from ._common.log import setup_logging


# ======================================================================
# Helpers: GRM-based PCA (aligned with GWAS module)
# ======================================================================

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

def load_group_table(group_path: str) -> tuple[pd.DataFrame, Union[str , None], Union[str , None]]:
    group_df = pd.read_csv(group_path, sep="\t", header=None, index_col=0)
    if group_df.shape[1] == 0:
        raise ValueError(f"Group file has no columns: {group_path}")
    if group_df.shape[1] == 1:
        group_df.columns = ["group"]
        return group_df, "group", None
    group_df = group_df.iloc[:, :2]
    group_df.columns = ["group", "label"]
    return group_df, "group", "label"

def build_grm_streaming_for_pca(
    genofile: str,
    maf_threshold: float = 0.02,
    max_missing_rate: float = 0.05,
    chunk_size: int = 100_000,
    mmap_limit: bool = False,
    logger=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the GRM in streaming mode via rust2py.gfreader.load_genotype_chunks
    and return (GRM, sample_ids).

    This follows the same strategy as the GWAS module:
      - genotypes are read in chunks
      - SNPs are filtered by MAF and missing rate inside the Rust reader
      - genotypes are centered by allele frequency
      - GRM is accumulated and symmetrized at the end
    """
    if logger is not None:
        logger.info("* Building GRM (streaming) for PCA")

    # Inspect genotype meta information
    sample_ids, n_snps = inspect_genotype_file(genofile)
    sample_ids = np.array(sample_ids, dtype=str)
    n_samples = len(sample_ids)

    grm = np.zeros((n_samples, n_samples), dtype="float32")
    pbar = tqdm(total=n_snps, desc="GRM (streaming)", ascii=True)
    process = psutil.Process()

    varsum = 0.0
    eff_m = 0
    mmap_window_mb = (
        auto_mmap_window_mb(genofile, n_samples, n_snps, chunk_size)
        if mmap_limit else None
    )
    for genosub, _sites in load_genotype_chunks(
        genofile,
        chunk_size,
        maf_threshold,
        max_missing_rate,
        mmap_window_mb=mmap_window_mb,
    ):
        # genosub: (m_chunk, n_samples)
        # MAF per SNP
        maf = genosub.mean(axis=1, dtype="float32", keepdims=True) / 2
        # Center genotypes: G' = G - 2p
        genosub = genosub - 2 * maf

        # Method 1: standard centered GRM (same as GWAS module default)
        grm += genosub.T @ genosub
        varsum += float(np.sum(2 * maf * (1 - maf)))

        eff_m += genosub.shape[0]
        pbar.update(genosub.shape[0])

        # Periodic memory logging
        if eff_m % (10 * chunk_size) == 0:
            mem = process.memory_info().rss / 1024**3
            pbar.set_postfix(memory=f"{mem:.2f} GB")

    # Force progress bar to 100% even if some SNPs were filtered in Rust
    pbar.n = pbar.total
    pbar.refresh()
    pbar.close()

    if eff_m == 0 or varsum == 0:
        raise RuntimeError("No SNPs remained after filtering; GRM for PCA is empty.")

    # Symmetrize and scale
    grm = (grm + grm.T) / (2 * varsum)

    if logger is not None:
        logger.info(f"GRM construction finished. Effective SNPs: {eff_m}")
        logger.info(f"GRM shape: {grm.shape}")

    return grm, sample_ids


def eigendecompose_grm(grm: np.ndarray, logger=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform eigen decomposition of a symmetric GRM:
      GRM = V Λ V^T

    Returns:
      eigenvec : columns are eigenvectors ordered by descending eigenvalue
      eigenval : corresponding eigenvalues (1D array)
    """
    if logger is not None:
        logger.info("* Performing eigen decomposition of GRM...")

    eigval, eigvec = np.linalg.eigh(grm)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    if logger is not None:
        logger.info("Eigen decomposition finished.")

    return eigvec, eigval


# ======================================================================
# Main CLI
# ======================================================================

def main(log: bool = True):
    t_start = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ------------------------- Required arguments -------------------------
    required_group = parser.add_argument_group("Required Arguments")
    geno_group = required_group.add_mutually_exclusive_group(required=True)
    geno_group.add_argument(
        "-vcf", "--vcf", type=str,
        help="Input genotype file in VCF format (.vcf or .vcf.gz).",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help=(
            "Input genotype in PLINK binary format "
            "(prefix for .bed, .bim, .fam)."
        ),
    )
    geno_group.add_argument(
        "-k", "--grm", type=str,
        help=(
            "GRM prefix for PCA (expects {prefix}.grm.id and "
            "{prefix}.grm.txt or {prefix}.grm.npy)."
        ),
    )
    geno_group.add_argument(
        "-q", "--qcov", type=str,
        help=(
            "Prefix of existing PCA result files for visualization only "
            "({prefix}.eigenval, {prefix}.eigenvec)."
        ),
    )

    # ------------------------- Optional arguments -------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for PCA results (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix of output files (default: inferred from input file name).",
    )
    optional_group.add_argument(
        "-dim", "--dim", type=int, default=3,
        help="Number of leading principal components to output (default: %(default)s).",
    )
    optional_group.add_argument(
        "-mmap-limit", "--mmap-limit", action="store_true", default=False,
        help="Enable windowed mmap for BED inputs (auto: 2x chunk size).",
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
        "-chunksize", "--chunksize", type=int, default=100_000,
        help="Number of SNPs per chunk for streaming GRM "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-plot", "--plot", action="store_true", default=False,
        help="Generate 2D scatter plots for PC1 vs PC2 and PC1 vs PC3 "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-plot3D", "--plot3D", action="store_true", default=False,
        help="Generate a 3D rotating GIF for PC1-PC3 (default: %(default)s).",
    )
    optional_group.add_argument(
        "-group", "--group", type=str, default=None,
        help=(
            "Group file with two columns: sample ID and group label (no header). "
            "Optional third column will be used as a text annotation tag "
            "(default: %(default)s)."
        ),
    )
    optional_group.add_argument(
        "-color", "--color", type=int, default=-1,
        help="Color palette index for PCA plots, 0-6; -1 uses auto palette "
             "(default: %(default)s).",
    )

    args = parser.parse_args()

    # ------------------------- Resolve input file & output prefix -------------------------
    if args.vcf:
        gfile = args.vcf
        args.prefix = os.path.basename(gfile).replace(".gz", "").replace(".vcf", "") \
            if args.prefix is None else args.prefix
    elif args.bfile:
        gfile = args.bfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    elif args.grm:
        gfile = args.grm
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    elif args.qcov:
        gfile = args.qcov
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    else:
        raise ValueError("No valid input found; one of --vcf/--bfile/--grm/--qcov must be provided.")

    gfile = gfile.replace("\\", "/")
    args.out = args.out if args.out is not None else "."

    # Keep index for logging but convert to actual color palette
    palette_idx = args.color
    if args.color == -1:
        args.color = None
    else:
        assert 0 <= args.color <= 6, "Color set index out of range; please use 0-6."
        args.color = color_set[palette_idx]

    # ------------------------- Logging -------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    log_path = f"{args.out}/{args.prefix}.pca.log".replace("\\", "/").replace("//", "/")
    logger = setup_logging(log_path)

    logger.info("JanusX - Principal Component Analysis Module")
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("PCA CONFIGURATION")
        logger.info("*" * 60)
        if args.vcf or args.bfile:
            logger.info(f"Genotype file:    {gfile}")
            logger.info(f"Output PCs:       top {args.dim}")
            logger.info(f"MAF threshold:    {args.maf}")
            logger.info(f"Missing rate:     {args.geno}")
            logger.info(f"Chunk size:       {args.chunksize}")
            logger.info(f"Mmap limit:       {args.mmap_limit}")
        elif args.grm:
            logger.info(f"GRM prefix:       {gfile}")
            logger.info(f"Output PCs:       top {args.dim}")
        elif args.qcov:
            logger.info(f"PCA prefix:       {gfile} (visualization only)")
        if args.plot or args.plot3D:
            logger.info(f"2D visualization: {args.plot}")
            logger.info(f"3D visualization (GIF): {args.plot3D}")
        if args.group:
            logger.info(f"Group file:      {args.group}")
            logger.info(
                f"Color palette:   {'auto' if palette_idx == -1 else f'index {palette_idx}'}"
            )
        logger.info(f"Output prefix:    {args.out}/{args.prefix}")
        logger.info("*" * 60 + "\n")

    # ------------------------- PCA core logic -------------------------
    t_loading = time.time()

    eigenvec = None
    eigenval = None
    samples = None

    # --- Case 1: VCF / BFILE -> streaming GRM -> PCA (aligned with GWAS) ---
    if args.vcf or args.bfile:
        logger.info("* PCA from genotype (VCF/BFILE) using streaming GRM.")
        logger.info(f"  MAF filter = {args.maf}, missing rate filter = {args.geno}.")

        # Build GRM in streaming mode
        grm, samples = build_grm_streaming_for_pca(
            genofile=gfile,
            maf_threshold=args.maf,
            max_missing_rate=args.geno,
            chunk_size=int(args.chunksize),
            mmap_limit=args.mmap_limit,
            logger=logger,
        )

        # Eigen decomposition
        eigenvec, eigenval = eigendecompose_grm(grm, logger=logger)

        logger.info(
            f"Completed PCA from genotype in {round(time.time() - t_loading, 3)} seconds"
        )

        # Save core PCA results
        df_vec = pd.DataFrame(
            np.column_stack([samples.astype(str), eigenvec[:, : args.dim]])
        )
        df_vec.to_csv(
            f"{args.out}/{args.prefix}.eigenvec",
            sep="\t",
            header=False,
            index=False,
            float_format="%.6f",
        )
        np.savetxt(
            f"{args.out}/{args.prefix}.eigenval",
            eigenval,
            fmt="%.2f",
        )
        logger.info(
            f'Saved eigen results in "{args.out}" with files '
            f'"{args.prefix}.eigenvec", '
            f'"{args.prefix}.eigenval"'
        )

    # --- Case 2: GRM prefix -> load GRM -> PCA ---
    elif args.grm:
        logger.info("* PCA from precomputed GRM.")
        # Support both prefix-based GRM and direct GRM file paths
        grm_file = None
        if os.path.isfile(gfile):
            grm_file = gfile
        else:
            if os.path.exists(f"{gfile}.grm.txt"):
                grm_file = f"{gfile}.grm.txt"
            elif os.path.exists(f"{gfile}.grm.npy"):
                grm_file = f"{gfile}.grm.npy"

        if grm_file is None:
            raise ValueError("GRM matrix (.grm.txt/.grm.npy or direct path) not found.")

        logger.info(f"Loading GRM from {grm_file} ...")
        if grm_file.endswith(".npy"):
            grm = np.load(grm_file)
        else:
            grm = np.genfromtxt(grm_file)

        id_path = f"{grm_file}.id"
        samples = _read_id_file(id_path, logger, "GRM")
        if samples is None:
            raise ValueError(f"GRM ID file not found: {id_path}")
        logger.info(f"Loaded GRM sample IDs from {id_path}")

        if grm.shape[0] != len(samples):
            raise ValueError(
                f"GRM size mismatch: {grm.shape[0]} != {len(samples)} (ID count)."
            )

        # Eigen decomposition
        eigenvec, eigenval = eigendecompose_grm(grm, logger=logger)
        logger.info(
            f"Completed PCA from GRM in {round(time.time() - t_loading, 3)} seconds"
        )

        # Save core PCA results
        df_vec = pd.DataFrame(
            np.column_stack([samples.astype(str), eigenvec[:, : args.dim]])
        )
        df_vec.to_csv(
            f"{args.out}/{args.prefix}.eigenvec",
            sep="\t",
            header=False,
            index=False,
            float_format="%.6f",
        )
        np.savetxt(
            f"{args.out}/{args.prefix}.eigenval",
            eigenval,
            fmt="%.2f",
        )
        logger.info(
            f'Saved eigen results in "{args.out}" with files '
            f'"{args.prefix}.eigenvec", '
            f'"{args.prefix}.eigenval"'
        )

    # --- Case 3: qcov prefix -> load PC results only for plotting ---
    elif args.qcov:
        logger.info("* Loading existing PC results for visualization only.")
        samples, eigenvec = _read_matrix_with_ids(f"{gfile}.eigenvec", logger, "Eigenvec")
        eigenval = np.genfromtxt(f"{gfile}.eigenval")
        logger.info(
            f"Loaded PC results from {gfile}.eigenvec(.id/.eigenval) in "
            f"{round(time.time() - t_loading, 3)} seconds"
        )

    # Safety check
    if eigenvec is None or eigenval is None or samples is None:
        raise RuntimeError("PCA results are not available; check input arguments.")

    # ------------------------- Visualization -------------------------
    if args.plot or args.plot3D:
        exp = 100 * eigenval / np.sum(eigenval)
        df_pc = pd.DataFrame(
            eigenvec[:, :3],
            index=samples,
            columns=[
                f"PC{i + 1}({round(float(exp[i]), 2)}%)" for i in range(3)
            ],
        )

        group = None
        textanno = None
        if args.group:
            group_df, group, textanno = load_group_table(args.group)
            df_pc = df_pc.join(group_df, how="left")

    if args.plot:
        logger.info("* Generating 2D PCA scatter plots...")

        pcshow = PCSHOW(df_pc)
        fig = plt.figure(figsize=(10, 4), dpi=300)
        ax1 = fig.add_subplot(121)
        ax1.set_xlabel(df_pc.columns[0])
        ax1.set_ylabel(df_pc.columns[1])
        ax2 = fig.add_subplot(122)
        ax2.set_xlabel(df_pc.columns[0])
        ax2.set_ylabel(df_pc.columns[2])

        pcshow.pcplot(
            df_pc.columns[0],
            df_pc.columns[1],
            group=group,
            ax=ax1,
            color_set=args.color,
            anno_tag=textanno,
        )
        pcshow.pcplot(
            df_pc.columns[0],
            df_pc.columns[2],
            group=group,
            ax=ax2,
            color_set=args.color,
            anno_tag=textanno,
        )

        plt.tight_layout()
        out_pdf = f"{args.out}/{args.prefix}.eigenvec.2D.pdf"
        plt.savefig(out_pdf, transparent=True)
        plt.close()
        logger.info(f"2D PCA figure saved to {out_pdf.replace('//', '/')}")

    if args.plot3D:
        logger.info("* Generating 3D PCA rotating GIF...")
        pcshow = PCSHOW(df_pc)
        out_gif = f"{args.out}/{args.prefix}.eigenvec.3D.gif"
        pcshow.pcplot3D_gif(
            df_pc.columns[0],
            df_pc.columns[1],
            df_pc.columns[2],
            group=group,
            anno_tag=textanno,
            color_set=args.color,
            out_gif=out_gif,
        )
        logger.info(f"3D PCA GIF saved to {out_gif.replace('//', '/')}")

    # ------------------------- Final logging -------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished PCA. Total wall time: {round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
