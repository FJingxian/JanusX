# -*- coding: utf-8 -*-
"""
JanusX - Principal Component Analysis (PCA) Command-Line Interface

Design summary
--------------
Input modes:
  - VCF    : genotype in VCF / VCF.GZ format
  - BFILE  : genotype in PLINK binary format (.bed/.bim/.fam prefix)
  - GRM    : precomputed genetic relationship matrix (GRM) + ID file
  - PCFILE : precomputed PCA results (eigenvec / eigenval / eigenvec.id) for visualization only

PCA computation strategy
------------------------
  - For VCF/BFILE:
      * Use rust2py.gfreader.load_genotype_chunks to stream genotype data
      * Build GRM in a low-memory, chunk-based fashion (same logic as GWAS module)
      * Perform eigen decomposition of GRM via numpy.linalg.eigh

  - For GRM:
      * Read precomputed GRM from .grm.txt or .grm.npz
      * Perform eigen decomposition of GRM via numpy.linalg.eigh

  - For PCFILE:
      * Load PC coordinates and eigenvalues directly
      * Only produce 2D/3D visualization

Output:
  - {prefix}.eigenvec      : PC coordinates (samples, dim)
  - {prefix}.eigenvec.id   : sample IDs
  - {prefix}.eigenval      : eigenvalues (variance along each PC)
  - Optional plots:
      * {prefix}.eigenvec.2D.pdf
      * {prefix}.eigenvec.3D.html
"""

import os
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

from JanusX.bioplotkit.sci_set import color_set
from JanusX.bioplotkit import PCSHOW
from JanusX.gfreader import load_genotype_chunks, inspect_genotype_file
from ._common.log import setup_logging


# ======================================================================
# Helpers: GRM-based PCA (aligned with GWAS module)
# ======================================================================

def load_group_table(group_path: str) -> tuple[pd.DataFrame, str | None, str | None]:
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
    maf_threshold: float = 0.01,
    max_missing_rate: float = 0.05,
    chunk_size: int = 100_000,
    logger=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a GRM in streaming mode using rust2py.gfreader.load_genotype_chunks
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

    for genosub, _sites in load_genotype_chunks(
        genofile,
        chunk_size,
        maf_threshold,
        max_missing_rate,
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
        help="Input genotype file in VCF format (.vcf or .vcf.gz)",
    )
    geno_group.add_argument(
        "-bfile", "--bfile", type=str,
        help=(
            "Input genotype in PLINK binary format "
            "(prefix for .bed, .bim, .fam)"
        ),
    )
    geno_group.add_argument(
        "-grm", "--grm", type=str,
        help=(
            "GRM prefix for PCA (expects {prefix}.grm.id and "
            "{prefix}.grm.txt or {prefix}.grm.npz)"
        ),
    )
    geno_group.add_argument(
        "-pcfile", "--pcfile", type=str,
        help=(
            "Prefix of existing PCA result files for visualization only "
            "({prefix}.eigenval, {prefix}.eigenvec, {prefix}.eigenvec.id)"
        ),
    )

    # ------------------------- Optional arguments -------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-o", "--out", type=str, default=None,
        help="Output directory for PCA results (default: same as genotype/GRM/PC files)",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix of output files (default: inferred from input file name)",
    )
    optional_group.add_argument(
        "-dim", "--dim", type=int, default=3,
        help="Number of leading principal components to output (default: %(default)s)",
    )
    optional_group.add_argument(
        "-plot", "--plot", action="store_true", default=False,
        help="Generate 2D scatter plots for PC1 vs PC2 and PC1 vs PC3 (default: %(default)s)",
    )
    optional_group.add_argument(
        "-plot3D", "--plot3D", action="store_true", default=False,
        help="Generate 3D interactive scatter plot for PC1–PC3 (default: %(default)s)",
    )
    optional_group.add_argument(
        "-gif3D", "--gif3D", action="store_true", default=False,
        help="Generate 3D rotating GIF for PC1–PC3 (default: %(default)s)",
    )
    optional_group.add_argument(
        "-group", "--group", type=str, default=None,
        help=(
            "Group file with two columns: sample ID and group label (no header). "
            "Optional third column will be used as text annotation tag (default: %(default)s)"
        ),
    )
    optional_group.add_argument(
        "-color", "--color", type=int, default=-1,
        help="Color palette index for PCA plots, 0–6; -1 uses auto palette (default: %(default)s)",
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
    elif args.pcfile:
        gfile = args.pcfile
        args.prefix = os.path.basename(gfile) if args.prefix is None else args.prefix
    else:
        raise ValueError("No valid input found; one of --vcf/--bfile/--grm/--pcfile must be provided.")

    gfile = gfile.replace("\\", "/")
    args.out = os.path.dirname(gfile) if args.out is None else args.out

    # Keep index for logging but convert to actual color palette
    palette_idx = args.color
    if args.color == -1:
        args.color = None
    else:
        assert 0 <= args.color <= 6, "Color set index out of range; please use 0–6."
        args.color = color_set[palette_idx]

    # ------------------------- Logging -------------------------
    os.makedirs(args.out, 0o755, exist_ok=True)
    log_path = f"{args.out}/{args.prefix}.pca.log".replace("\\", "/").replace("//", "/")
    logger = setup_logging(log_path)

    logger.info("JanusX – Principal Component Analysis Module")
    logger.info(f"Host: {socket.gethostname()}\n")

    if log:
        logger.info("*" * 60)
        logger.info("PCA CONFIGURATION")
        logger.info("*" * 60)
        if args.vcf or args.bfile:
            logger.info(f"Genotype file:    {gfile}")
            logger.info(f"Output PCs:       top {args.dim}")
        elif args.grm:
            logger.info(f"GRM prefix:       {gfile}")
            logger.info(f"Output PCs:       top {args.dim}")
        elif args.pcfile:
            logger.info(f"PCA prefix:       {gfile} (visualization only)")
        if args.plot or args.plot3D:
            logger.info(f"2D visualization: {args.plot}")
            logger.info(f"3D visualization: {args.plot3D}")
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
        logger.info("  MAF filter = 0.01, missing rate filter = 0.05.")

        # Build GRM in streaming mode
        grm, samples = build_grm_streaming_for_pca(
            genofile=gfile,
            maf_threshold=0.01,
            max_missing_rate=0.05,
            chunk_size=100_000,
            logger=logger,
        )

        # Eigen decomposition
        eigenvec, eigenval = eigendecompose_grm(grm, logger=logger)

        logger.info(
            f"Completed PCA from genotype in {round(time.time() - t_loading, 3)} seconds"
        )

        # Save core PCA results
        np.savetxt(f"{args.out}/{args.prefix}.eigenvec.id", samples, fmt="%s")
        np.savetxt(
            f"{args.out}/{args.prefix}.eigenvec",
            eigenvec[:, : args.dim],
            fmt="%.6f",
        )
        np.savetxt(
            f"{args.out}/{args.prefix}.eigenval",
            eigenval,
            fmt="%.2f",
        )
        logger.info(
            f'Saved eigen results in "{args.out}" with files '
            f'"{args.prefix}.eigenvec", "{args.prefix}.eigenvec.id", '
            f'"{args.prefix}.eigenval"'
        )

    # --- Case 2: GRM prefix -> load GRM -> PCA ---
    elif args.grm:
        logger.info("* PCA from precomputed GRM.")
        assert os.path.exists(f"{gfile}.grm.id"), "GRM ID file (.grm.id) not found."
        assert os.path.exists(f"{gfile}.grm.txt") or os.path.exists(f"{gfile}.grm.npz"), \
            "GRM matrix (.grm.txt or .grm.npz) not found."

        if os.path.exists(f"{gfile}.grm.txt"):
            logger.info(f"Loading GRM from {gfile}.grm.txt and {gfile}.grm.id ...")
            grm = np.genfromtxt(f"{gfile}.grm.txt")
        else:
            logger.info(f"Loading GRM from {gfile}.grm.npz and {gfile}.grm.id ...")
            grm = np.load(f"{gfile}.grm.npz")["arr_0"]

        samples = np.genfromtxt(f"{gfile}.grm.id", dtype=str)

        # Eigen decomposition
        eigenvec, eigenval = eigendecompose_grm(grm, logger=logger)
        logger.info(
            f"Completed PCA from GRM in {round(time.time() - t_loading, 3)} seconds"
        )

        # Save core PCA results
        np.savetxt(f"{args.out}/{args.prefix}.eigenvec.id", samples, fmt="%s")
        np.savetxt(
            f"{args.out}/{args.prefix}.eigenvec",
            eigenvec[:, : args.dim],
            fmt="%.6f",
        )
        np.savetxt(
            f"{args.out}/{args.prefix}.eigenval",
            eigenval,
            fmt="%.2f",
        )
        logger.info(
            f'Saved eigen results in "{args.out}" with files '
            f'"{args.prefix}.eigenvec", "{args.prefix}.eigenvec.id", '
            f'"{args.prefix}.eigenval"'
        )

    # --- Case 3: PCFILE prefix -> load PC results only for plotting ---
    elif args.pcfile:
        logger.info("* Loading existing PC results for visualization only.")
        eigenvec = np.genfromtxt(f"{gfile}.eigenvec")
        samples = np.genfromtxt(f"{gfile}.eigenvec.id", dtype=str)
        eigenval = np.genfromtxt(f"{gfile}.eigenval")
        logger.info(
            f"Loaded PC results from {gfile}.eigenvec(.id/.eigenval) in "
            f"{round(time.time() - t_loading, 3)} seconds"
        )

    # Safety check
    if eigenvec is None or eigenval is None or samples is None:
        raise RuntimeError("PCA results are not available; check input arguments.")

    # ------------------------- Visualization -------------------------
    if args.plot or args.plot3D or args.gif3D:
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
        logger.info("* Generating 3D PCA scatter plot...")
        pcshow = PCSHOW(df_pc)
        fig = pcshow.pcplot3D(
            df_pc.columns[0],
            df_pc.columns[1],
            df_pc.columns[2],
            group=group,
            anno_tag=textanno,
            color_set=args.color,
        )

        out_html = f"{args.out}/{args.prefix}.eigenvec.3D.html"
        fig.write_html(out_html, config={"displayModeBar": False})
        logger.info(f"3D PCA figure saved to {out_html.replace('//', '/')}")

    if args.gif3D:
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
