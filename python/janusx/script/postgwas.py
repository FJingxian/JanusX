# -*- coding: utf-8 -*-
"""
JanusX: Post-GWAS Visualization and Annotation

Examples
--------
  # Basic usage with default column names (#CHROM, POS, p)
  -f result.assoc.txt

  # Specify alternative column names
  -f result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald"

  # Specify output path and format
  -f result.assoc.txt -chr "chr" -pos "pos" -pvalue "P_wald" \
    --out test --format pdf
  # Results will be saved as:
  #   test/result.assoc.manh.pdf
  #   test/result.assoc.qq.pdf

Citation
--------
  https://github.com/FJingxian/JanusX/
"""

import logging
import os
from ._common.log import setup_logging
from ._common.pathcheck import ensure_all_true, ensure_file_exists

# Ensure matplotlib uses a non-interactive backend.
for key in ["MPLBACKEND"]:
    if key in os.environ:
        del os.environ[key]

import matplotlib as mpl
mpl.use("Agg")
from janusx.bioplotkit import GWASPLOT
from janusx.bioplotkit.sci_set import color_set

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import numpy as np
import argparse
import time
import socket
from ._common.readanno import readanno
from joblib import Parallel, delayed
import warnings


def _auto_colors(n: int) -> list[str]:
    if n <= 10:
        cmap = plt.get_cmap("tab10")
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("turbo")
    return [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]


def GWASplot(file: str, args, logger:logging.Logger) -> None:
    """
    Plot Manhattan/QQ figures and optionally annotate significant hits
    for a single GWAS result file.
    """
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["axes.unicode_minus"] = False

    # Silence pandas chained-assignment warnings in this script
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*ChainedAssignmentError.*",
    )

    args.prefix = (
        os.path.basename(file)
        .replace(".tsv", "")
        .replace(".txt", "")
    )

    chr_col, pos_col, p_col = args.chr, args.pos, args.pvalue

    df = pd.read_csv(file, sep="\t", usecols=[chr_col, pos_col, p_col])

    # Bonferroni-style default threshold if not provided
    threshold = args.threshold if args.threshold is not None else 0.05 / df.shape[0]

    # ------------------------------------------------------------------
    # 1. Visualization: Manhattan & QQ
    # ------------------------------------------------------------------
    if args.noplot:
        t_plot = time.time()
        logger.info("* Visualizing GWAS results...")

        plotmodel = GWASPLOT(df, chr_col, pos_col, p_col, 0.1)

        # ----------------- Manhattan plot -----------------
        fig = plt.figure(figsize=(8, 4), dpi=300)
        ax = fig.add_subplot(111)

        # Rasterization for non-PDF formats to keep file size reasonable
        rasterized = False if args.format == "pdf" else True

        plot_colors = args.color
        if plot_colors is None:
            plot_colors = _auto_colors(len(plotmodel.chr_ids))

        if args.highlight:
            # Highlight specific SNPs (bed-like file: chr, start, end, gene, desc)
            df_hl = pd.read_csv(args.highlight, sep="\t", header=None)
            # If gene name is missing, fall back to chr_pos
            gene_mask = df_hl[3].isna()
            df_hl.loc[gene_mask, 3] = (
                df_hl.loc[gene_mask, 0].astype(str)
                + "_"
                + df_hl.loc[gene_mask, 1].astype(str)
            )
            df_hl = df_hl.set_index([0, 1])

            # Intersect highlight positions with SNPs in the plot model
            df_hl_idx = df_hl.index[df_hl.index.isin(plotmodel.df.index)]
            if len(df_hl_idx) == 0:
                logger.warning("Nothing to highlight. Check the BED file.")
                plotmodel.manhattan(
                    -np.log10(threshold),
                    ax=ax,
                    color_set=plot_colors,
                    rasterized=rasterized,
                )
            else:
                # Highlight points
                ax.scatter(
                    plotmodel.df.loc[df_hl_idx, "x"],
                    -np.log10(plotmodel.df.loc[df_hl_idx, "y"]),
                    marker="D",
                    color="red",
                    zorder=10,
                    s=32,
                    edgecolors="black",
                )
                ax.hlines(
                    y=-np.log10(threshold),
                    xmin=-1e10,
                    xmax=1e10,
                    linestyle="dashed",
                    color="grey",
                )

                # Add gene labels
                for idx in df_hl_idx:
                    text = df_hl.loc[idx, 3]
                    ax.text(
                        plotmodel.df.loc[idx, "x"],
                        -np.log10(plotmodel.df.loc[idx, "y"]),
                        s=text,
                        ha="center",
                        zorder=11,
                    )

                # Plot background Manhattan points, excluding highlighted ones
                plotmodel.manhattan(
                    None,
                    ax=ax,
                    color_set=plot_colors,
                    ignore=df_hl_idx,
                    rasterized=rasterized,
                )
        else:
            # Standard Manhattan with threshold line
            plotmodel.manhattan(
                -np.log10(threshold),
                ax=ax,
                color_set=plot_colors,
                rasterized=rasterized,
            )

        plt.tight_layout()
        manh_path = f"{args.out}/{args.prefix}.manh.{args.format}"
        plt.savefig(manh_path, transparent=False, facecolor="white")
        plt.close(fig)

        # ----------------- QQ plot -----------------
        fig = plt.figure(figsize=(5, 4), dpi=300)
        ax2 = fig.add_subplot(111)
        plotmodel.qq(ax=ax2, color_set=plot_colors)
        plt.tight_layout()
        qq_path = f"{args.out}/{args.prefix}.qq.{args.format}"
        plt.savefig(qq_path, transparent=False, facecolor="white")
        plt.close(fig)

        logger.info(
            f"Manhattan and QQ plots saved to:\n"
            f"  {manh_path}\n"
            f"  {qq_path}"
        )
        logger.info(f"Visualization completed in {round(time.time() - t_plot, 2)} seconds.\n")

    # ------------------------------------------------------------------
    # 2. Annotation of significant loci
    # ------------------------------------------------------------------
    if args.anno:
        logger.info("* Annotating significant SNPs...")
        if os.path.exists(args.anno):
            t_anno = time.time()

            # Keep SNPs passing threshold
            df_filter = df.loc[
                df[p_col] <= threshold,
                [chr_col, pos_col, p_col],
            ].set_index([chr_col, pos_col])

            # Read annotation (GFF/bed) into unified annotation table
            # After readanno:
            #   anno[0] = chr
            #   anno[1] = start
            #   anno[2] = end
            #   anno[3] = gene ID
            #   anno[4], anno[5] = description fields
            anno = readanno(args.anno, args.descItem)
            anno_chr = anno[0].astype(str)

            # Exact overlap annotation
            desc_exact = [
                anno.loc[
                    (anno_chr == str(idx[0]))
                    & (anno[1] <= idx[1])
                    & (anno[2] >= idx[1])
                ]
                for idx in df_filter.index
            ]
            df_filter["desc"] = [
                (
                    f"{x.iloc[0, 3]};{x.iloc[0, 4]};{x.iloc[0, 5]}"
                    if not x.empty
                    else "NA;NA;NA"
                )
                for x in desc_exact
            ]

            # Optional broadened window around SNP (± annobroaden kb)
            if args.annobroaden is not None:
                kb = args.annobroaden * 1_000
                desc_broad = [
                    anno.loc[
                        (anno_chr == str(idx[0]))
                        & (anno[1] <= idx[1] + kb)
                        & (anno[2] >= idx[1] - kb)
                    ]
                    for idx in df_filter.index
                ]
                df_filter["broaden"] = [
                    (
                        f"{'|'.join(x.iloc[:, 3])};"
                        f"{'|'.join(x.iloc[:, 4])};"
                        f"{'|'.join(x.iloc[:, 5])}"
                        if not x.empty
                        else "NA;NA;NA"
                    )
                    for x in desc_broad
                ]
            else:
                if "broaden" in df_filter.columns:
                    df_filter = df_filter.drop(columns=["broaden"])

            logger.info(df_filter)

            anno_path = f"{args.out}/{args.prefix}.{threshold}.anno.tsv"
            df_filter.to_csv(anno_path, sep="\t")
            logger.info(f"Annotation table saved to {anno_path}")
            logger.info(f"Annotation completed in {round(time.time() - t_anno, 2)} seconds.\n")
        else:
            logger.info(f"Annotation file not found: {args.anno}\n")


def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ------------------------------------------------------------------
    # Required arguments
    # ------------------------------------------------------------------
    required_group = parser.add_argument_group("Required Arguments")
    required_group.add_argument(
        "-f", "--file", nargs="+", type=str, required=True,
        help="One or more GWAS result files (tab-delimited).",
    )

    # ------------------------------------------------------------------
    # Optional arguments
    # ------------------------------------------------------------------
    optional_group = parser.add_argument_group("Optional Arguments")
    optional_group.add_argument(
        "-chr", "--chr", type=str, default="chrom",
        help="Column name for chromosome (default: %(default)s).",
    )
    optional_group.add_argument(
        "-pos", "--pos", type=str, default="pos",
        help="Column name for base position (default: %(default)s).",
    )
    optional_group.add_argument(
        "-pvalue", "--pvalue", type=str, default="pwald",
        help="Column name for p-value (default: %(default)s).",
    )
    optional_group.add_argument(
        "-threshold", "--threshold", type=float, default=None,
        help="P-value threshold; if not set, use 0.05 / nSNP (default: %(default)s).",
    )
    optional_group.add_argument(
        "-noplot", "--noplot", action="store_false", default=True,
        help="Disable plotting Manhattan/QQ figures (default: %(default)s).",
    )
    optional_group.add_argument(
        "-color", "--color", type=int, default=0,
        help="Color style index for Manhattan and QQ (0-6); -1 uses auto palette "
             "(default: %(default)s).",
    )
    optional_group.add_argument(
        "-hl", "--highlight", type=str, default=None,
        help=(
            "BED-like file of SNPs to highlight, e.g.:\n"
            "  chr\\tpos\\tpos\\tgene\\tfunction"
        ),
    )
    optional_group.add_argument(
        "-format", "--format", type=str, default="png",
        help="Output figure format: pdf, png, svg, tif (default: %(default)s).",
    )
    optional_group.add_argument(
        "-a", "--anno", type=str, default=None,
        help="Annotation file (.gff or .bed) for SNP annotation (default: %(default)s).",
    )
    optional_group.add_argument(
        "-ab", "--annobroaden", type=float, default=None,
        help="Broaden the annotation window around SNPs (Kb) (default: %(default)s).",
    )
    optional_group.add_argument(
        "-descItem", "--descItem", type=str, default="description",
        help="Attribute key used as description in the GFF file (default: %(default)s).",
    )
    optional_group.add_argument(
        "-o", "--out", type=str, default=".",
        help="Output directory for plots and annotation (default: current directory).",
    )
    optional_group.add_argument(
        "-prefix", "--prefix", type=str, default=None,
        help="Prefix of the log file (default: JanusX).",
    )
    optional_group.add_argument(
        "-t", "--thread", type=int, default=-1,
        help="Number of CPU threads (-1 uses all available cores; default: %(default)s).",
    )

    args = parser.parse_args()

    args.out = args.out if args.out is not None else "."
    args.prefix = "JanusX" if args.prefix is None else args.prefix

    # Create output directory if needed
    if args.out != "":
        os.makedirs(args.out, mode=0o755, exist_ok=True)
    else:
        args.out = "."

    log_path = f"{args.out}/{args.prefix}.postGWAS.log".replace("//", "/")
    logger = setup_logging(log_path)

    # ------------------------------------------------------------------
    # Basic checks and configuration
    # ------------------------------------------------------------------
    if not (-1 <= int(args.color) <= 6):
        logger.error("Color set index out of range; please use 0-6 or -1.")
        raise SystemExit(1)

    args.format = str(args.format).lower()
    if args.format not in ["pdf", "png", "svg", "tif"]:
        logger.error(
            f"Unsupported figure format: {args.format} "
            "(choose from: pdf, png, svg, tif)"
        )
        raise SystemExit(1)

    if args.color == -1:
        args.color = None
    else:
        args.color = color_set[args.color]

    # ------------------------------------------------------------------
    # Configuration summary
    # ------------------------------------------------------------------
    logger.info("JanusX - Post-GWAS visualization and annotation")
    logger.info(f"Host: {socket.gethostname()}\n")

    logger.info("*" * 60)
    logger.info("POST-GWAS CONFIGURATION")
    logger.info("*" * 60)
    logger.info(f"Input files:   {args.file}")
    logger.info(f"Chr column:    {args.chr}")
    logger.info(f"Pos column:    {args.pos}")
    logger.info(f"P-value column:{args.pvalue}")
    logger.info(
        f"Threshold:     {args.threshold if args.threshold is not None else '0.05 / nSNP'}"
    )
    if args.noplot:
        logger.info("Visualization:")
        logger.info(
            f"  Color set:   {'auto' if args.color is None else args.color}"
        )
        logger.info(f"  Highlight:   {args.highlight}")
        logger.info(f"  Format:      {args.format}")
    if args.anno:
        logger.info("Annotation:")
        logger.info(f"  Anno file:   {args.anno}")
        logger.info(f"  Window (kb): {args.annobroaden}")
    logger.info(f"Output prefix: {args.out}/{args.prefix}")
    logger.info(
        f"Threads:       {args.thread} "
        f"({'All cores' if args.thread == -1 else 'User-specified'})"
    )
    logger.info("*" * 60 + "\n")

    checks: list[bool] = [ensure_file_exists(logger, f, "GWAS result file") for f in args.file]
    if args.highlight:
        checks.append(ensure_file_exists(logger, args.highlight, "Highlight file"))
    if args.anno:
        checks.append(ensure_file_exists(logger, args.anno, "Annotation file"))
    if not ensure_all_true(checks):
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Parallel processing of all input files
    # ------------------------------------------------------------------
    Parallel(n_jobs=args.thread, backend="loky")(
        delayed(GWASplot)(file, args, logger) for file in args.file
    )

    # ------------------------------------------------------------------
    # Final logging
    # ------------------------------------------------------------------
    lt = time.localtime()
    endinfo = (
        f"\nFinished post-GWAS analysis. Total wall time: "
        f"{round(time.time() - t_start, 2)} seconds\n"
        f"{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} "
        f"{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}"
    )
    logger.info(endinfo)


if __name__ == "__main__":
    main()
